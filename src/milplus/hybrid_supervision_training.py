"""Full DLA hybrid paper-reference phases; released challenge RF stays separate.

The paper omits optimizer, classifier-reset and epoch-size details. These are
explicit reference choices, rather than claims to recover an unpublished trainer.
"""
import random
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .hybrid_supervision_source import build_model, source_transforms
from .hybrid_supervision_ops import refine_labels, top_patch_indices, image_objective
from .hybrid_supervision_checkpoint import cpu_state, save, load, exists, checkpoint_path
from .io import digest_file, digest_object, write_json


def rgb_tensor(pixels):
    if pixels.dtype != np.uint8 or pixels.shape != (512, 512, 3):
        raise ValueError('Hybrid paper training needs original uint8 RGB512 patches')
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float().div(255)



def gradient_state(model):
    return {k: None if p.grad is None else p.grad.detach().cpu().clone() for k,p in model.named_parameters()}


def restore_gradients(model, state):
    if set(state)!=set(dict(model.named_parameters())):
        raise ValueError('Hybrid checkpoint gradient parameter identities changed')
    for k,p in model.named_parameters():
        value=state[k]
        if value is not None and (value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all()):
            raise ValueError('Hybrid actual gradient shape/dtype/finiteness changed')
        p.grad=None if value is None else value.to(p.device).clone()

def fine_references(bundle, assets):
    return [(row['slide_id'], int(index), i) for row in bundle.records('train')
            for i, index in enumerate(assets.meta['slides'][row['slide_id']]['fine_indices'])]


def pixel_and_mask(assets, rows, reference, kind, pseudo=None, augment=True):
    sid, index = reference[:2]
    row = rows[sid]
    if row['subset'] != 'train':
        raise ValueError('Hybrid pseudo/fine fitting cannot access held-out patients')
    with assets.open(row) as f:
        pixels = f[assets.schema['pixel_key']][index]
        if kind == 'fine':
            mask = f[assets.schema['fine_masks_key']][reference[2]].astype(np.float32)/255
        elif kind == 'negative':
            mask = np.zeros((512, 512), dtype=np.float32)
        elif kind == 'positive':
            if pseudo is None:
                raise ValueError('Positive hybrid fitting requires the actual previous segmentation probabilities')
            with h5py.File(pseudo, 'r') as source:
                value = torch.from_numpy(source[sid]['probability'][index])
            mask = refine_labels(value, 1).numpy()
        else:
            raise ValueError('Unknown hybrid supervision source')
    y = torch.from_numpy(mask.copy())
    if augment:
        # Joint transformations operate on original uint8 pixels before ToTensor.
        k = random.randrange(4)
        pixels, y = np.rot90(pixels, k, (0, 1)), torch.rot90(y, k, (0, 1))
        if random.randrange(2):
            pixels, y = pixels[:, ::-1, :], y.flip(1)
        from PIL import Image
        image = Image.fromarray(np.ascontiguousarray(pixels))
        image = source_transforms().RandomJitter(.4, .4, .4)(image)[0]
        pixels = np.asarray(image)
    x = rgb_tensor(pixels)
    return x, y


def score_contract(bundle, assets, model):
    return digest_object({'inputs': assets.contract, 'slides': [r['slide_id'] for r in bundle.records('train')],
        'model': {k: {'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
                  for k,v in model.state_dict().items()}})


@torch.no_grad()
def score_training_patches(bundle, assets, model, output, device):
    path = Path(output)
    rows = bundle.records('train')
    contract = score_contract(bundle, assets, model)
    # The caller uses a unique phase-bound path and records its file hash in the
    # pipeline checkpoint; stale predictions cannot silently survive a rerun.
    if path.exists():
        raise FileExistsError('Do not replace an existing hybrid pseudo-label phase')
    model.eval()
    pools = {'positive': [], 'negative': []}
    with h5py.File(path, 'x') as target:
        target.attrs['contract'] = contract
        for row in rows:
            count = assets.count(row)
            group = target.create_group(row['slide_id'])
            values = group.create_dataset('probability', shape=(count, 512, 512), dtype=np.float32,
                                          chunks=(1, 512, 512), compression='lzf')
            maxima = []
            with assets.open(row) as original:
                rgb = original[assets.schema['pixel_key']]
                # Exact original inference batch16 and its actual tail.
                for start in range(0, count, 16):
                    x = torch.stack([rgb_tensor(a) for a in rgb[start:start+16]]).to(device)
                    probability = model(x).sigmoid()[:, 0].cpu()
                    if probability.shape != (len(x), 512, 512) or not torch.isfinite(probability).all():
                        raise FloatingPointError('Full hybrid DLA34Up produced invalid pixel probabilities')
                    values[start:start+len(x)] = probability.numpy()
                    maxima.extend(probability.flatten(1).amax(1).tolist())
            group.create_dataset('maximum', data=np.asarray(maxima, dtype=np.float32))
            for i, maximum in enumerate(maxima):
                if maximum >= .4:
                    pools['positive' if row['target'] else 'negative'].append([row['slide_id'], i])
    return {'path': str(path), 'sha256': digest_file(path), 'contract': contract, 'pools': pools}


def segmentation_phase(bundle, assets, model, output, epochs, device, progress, pseudo=None):
    """Full30epoch paper phase; all10-patch batches preserve2:1:7 when mixed."""
    from .engine import rng_state, restore_rng
    out = Path(output);out.mkdir(parents=True, exist_ok=True)
    state = load(out/'last.pt') if exists(out/'last.pt') else None
    fine = fine_references(bundle, assets)
    if not fine:
        raise ValueError('Hybrid initialization requires observed TRAIN fine masks')
    rows = {r['slide_id']: r for r in bundle.records('train')}
    if pseudo is not None:
        if digest_file(pseudo['path']) != pseudo['sha256'] or any(not pseudo['pools'][k] for k in ['positive', 'negative']):
            raise ValueError('Hybrid mixed phase needs actual retained positive and hard-negative TRAIN pools')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps_per_epoch = (len(fine)+9)//10
    contract = digest_object({'inputs': assets.contract, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch,
                              'pseudo': None if pseudo is None else {k:v for k,v in pseudo.items() if k!='path'}, 'reference_batch': 10, 'ratio': [2, 1, 7], 'lr': .001})
    history, done = [], 0
    if state is not None:
        if state['contract'] != contract:
            raise ValueError('Hybrid segmentation phase source/input/pseudo/budget changed')
        model.load_state_dict(state['model'], strict=True);optimizer.load_state_dict(state['optimizer'])
        restore_gradients(model,state['gradients'])
        history, done = state['history'], state['updates'];restore_rng(state['rng'])
        if state['finished']:
            return checkpoint_path(out/'last.pt')
    for step in range(done, epochs*steps_per_epoch):
        model.train()
        if pseudo is None:
            selections = [('fine', random.choice(fine)) for _ in range(10)]
        else:
            selections = [('fine', random.choice(fine)) for _ in range(2)]
            selections += [('positive', random.choice(pseudo['pools']['positive']))]
            selections += [('negative', random.choice(pseudo['pools']['negative'])) for _ in range(7)]
            random.shuffle(selections)
        pairs = [pixel_and_mask(assets, rows, ref, kind, pseudo=None if pseudo is None else pseudo['path'])
                 for kind, ref in selections]
        x = torch.stack([a for a, _ in pairs]).to(device)
        y = torch.stack([b for _, b in pairs]).unsqueeze(1).to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Full hybrid pixel soft-label CE became nonfinite')
        optimizer.zero_grad();loss.backward();optimizer.step()
        history.append({'step': step+1, 'epoch': step//steps_per_epoch+1, 'loss': float(loss.detach()),
                        'ordered_supervision': selections})
        finished = step+1 == epochs*steps_per_epoch
        save({'contract': contract, 'model': cpu_state(model), 'gradients': gradient_state(model), 'optimizer': optimizer.state_dict(),
              'rng': rng_state(), 'history': history, 'updates': step+1, 'finished': finished}, out/'last.pt')
        progress.record(out.name, step+1, history, finished)
    return checkpoint_path(out/'last.pt')


def classification_phase(bundle, assets, model, output, epochs, device, progress, scored):
    """Two equally weighted image bags/update; each is16 actual selected512RGB."""
    from .engine import rng_state, restore_rng
    out = Path(output);out.mkdir(parents=True, exist_ok=True)
    if digest_file(scored['path']) != scored['sha256']:
        raise ValueError('Hybrid actual selection probabilities changed')
    train = bundle.records('train');rows = {r['slide_id']: r for r in train}
    groups = {c: [r['slide_id'] for r in train if r['target'] == c] for c in [0, 1]}
    if any(not groups[c] for c in groups):
        raise ValueError('Hybrid image learning needs both TRAIN image classes')
    top = {}
    with h5py.File(scored['path'], 'r') as f:
        for row in train:
            maximum = torch.from_numpy(f[row['slide_id']]['maximum'][:])
            if len(maximum) < 16:
                raise ValueError('Hybrid Top16 cannot invent missing image patches')
            top[row['slide_id']] = torch.argsort(maximum, descending=True, stable=True)[:16].tolist()
    steps_per_epoch = max(len(groups[0]), len(groups[1]))
    contract = digest_object({'inputs': assets.contract, 'epochs': epochs, 'top16': top,
                              'score_sha256': scored['sha256'], 'balanced_image_batch': 2,
                              'steps_per_epoch': steps_per_epoch, 'lr': .001})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, done = [], 0
    state = load(out/'last.pt') if exists(out/'last.pt') else None
    if state is not None:
        if state['contract'] != contract:
            raise ValueError('Hybrid image classification selection/input/budget changed')
        model.load_state_dict(state['model'], strict=True);optimizer.load_state_dict(state['optimizer'])
        restore_gradients(model,state['gradients'])
        history, done = state['history'], state['updates'];restore_rng(state['rng'])
        if state['finished']:
            return checkpoint_path(out/'last.pt')
    for step in range(done, epochs*steps_per_epoch):
        model.train();optimizer.zero_grad();order = [0, 1];random.shuffle(order)
        selected = [random.choice(groups[c]) for c in order]
        terms = []
        for sid in selected:
            row = rows[sid]
            with assets.open(row) as f:
                data = f[assets.schema['pixel_key']]
                x = torch.stack([rgb_tensor(data[i]) for i in top[sid]]).to(device)
            logits = model(x)
            loss, probability = image_objective(logits, torch.tensor([row['target']], dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Hybrid mean-probability image BCE became nonfinite')
            (loss/2).backward();terms.append(float(loss.detach()))
        optimizer.step()
        history.append({'step': step+1, 'epoch': step//steps_per_epoch+1,
                        'loss': float(np.mean(terms)), 'balanced_image_order': selected})
        finished = step+1 == epochs*steps_per_epoch
        save({'contract': contract, 'model': cpu_state(model), 'gradients': gradient_state(model), 'optimizer': optimizer.state_dict(),
              'rng': rng_state(), 'history': history, 'updates': step+1, 'finished': finished,
              'last_epoch_training_loss': float(np.mean([r['loss'] for r in history[-steps_per_epoch:]]))}, out/'last.pt')
        progress.record(out.name, step+1, history, finished)
    return checkpoint_path(out/'last.pt')

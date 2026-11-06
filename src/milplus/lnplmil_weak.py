"""Actual LPA-trained ResNet18 and separate frozen ImageNet feature branch.

These are explicit reference choices, not an invented original author trainer:
ImageNet initialization, SGD/Nesterov .02/.9, cosine50epochs, batch256,
horizontal flips, ImageNet normalization, minimum TRAIN-CE checkpoint. The
underlying full TorchVision BasicBlock/ResNet equations are pinned byte-for-byte.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file, digest_object, read_json, write_json

ASSETS = {
    'resnet18-5c106cde.pth': '5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8',
    'torchvision_v061_resnet.py': '8c27d6ea12e675320481292d67479179718df406e3442365af369be5d3852019',
}
REFERENCE_RECIPE = {'epochs': 50, 'batch_size': 256, 'lr': .02, 'momentum': .9,
                    'weight_decay': 0., 'nesterov': True, 'lr_schedule': 'epoch_cosine_to_zero',
                    'augmentation': 'TRAIN_horizontal_flip_p0.5_no_resize',
                    'normalization': 'ImageNet_RGB_mean0.485_0.456_0.406_std0.229_0.224_0.225',
                    'selection': 'minimum_observed_TRAIN_patch_CE',
                    'status': 'explicit_reference_unpublished_LNPL_weak_recipe'}


def assets(root):
    root = Path(root)
    for name, expected in ASSETS.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest_file(path) != expected:
            raise ValueError('LNPL reference needs verified official ResNet18 source and actual ImageNet weights')
    return root


def original_definitions(root):
    root = assets(root)
    tree = ast.parse((root / 'torchvision_v061_resnet.py').read_text())
    wanted = {'conv3x3', 'conv1x1', 'BasicBlock', 'Bottleneck', 'ResNet'}
    nodes = [v for v in tree.body if isinstance(v, (ast.FunctionDef, ast.ClassDef)) and v.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('Pinned official ResNet18 operators changed')
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / 'torchvision_v061_resnet.py'), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in wanted})


def build(root, branch, classes=2):
    if branch not in {'weak', 'frozen'} or type(classes) is not int or classes not in {2, 9}:
        raise ValueError('LNPL explicitly uses binary/nine-class weak ResNet18 and a separate frozen512D ResNet18')
    source = original_definitions(root)
    net = source.ResNet(source.BasicBlock, [2, 2, 2, 2])
    net.load_state_dict(torch.load(Path(root) / 'resnet18-5c106cde.pth', map_location='cpu', weights_only=True), strict=True)
    if branch == 'weak':
        net.fc = nn.Linear(512, classes)
    else:
        net.fc = nn.Identity()
        net.requires_grad_(False)
        net.eval()
    return net


def rgb_tensor(image, augment=False):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise ValueError('Use actual uint8 RGB224, without synthetic image views of frozen features')
    value = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.)
    if augment and torch.rand(()).item() < .5:
        value = value.flip(-1)
    mean = value.new_tensor([.485, .456, .406])[:, None, None]
    std = value.new_tensor([.229, .224, .225])[:, None, None]
    # Explicit CHW layout: channels-last and contiguous CPU convolution kernels
    # can round differently despite exactly equal input values. Keep the same
    # canonical layout for the adapter and literal source numerical oracle.
    return ((value - mean) / std).contiguous()


def optimizer(net):
    return torch.optim.SGD(net.parameters(), lr=.02, momentum=.9, nesterov=True, weight_decay=0.)


def fit(raw, output, device='cpu', seed=2026, epochs=50, batch_size=256, resume=False, stop_after_updates=None):
    """Full actual RGB/LPA weak stage, with exact batch-boundary recovery.

    Explicit non-native budgets are software checks and remain recorded. There
    is no label inheritance, pseudo-label supervision, test selection or model
    width override. Frozen ImageNet features are not derived from this model.
    """
    from .engine import seed_everything, rng_state, restore_rng
    from .hybrid_supervision_checkpoint import save, load, exists, cpu_state, _lock
    from .provenance import software_identity
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or type(seed) is not int or stop_after_updates is not None
            and (type(stop_after_updates) is not int or stop_after_updates < 1)):
        raise ValueError('Declare positive weak-stage budgets and an actual update-boundary interruption')
    dev = torch.device(device)
    if dev.type not in {'cpu', 'cuda'} or dev.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested LNPL weak-stage device unavailable; no CPU fallback')
    out = Path(output)
    request = {'backend': 'lnplmil_actual_LPA_weak_ResNet18_reference', 'raw_contract': raw.contract,
               'manifest': str(raw.path), 'classes': len(raw.schema['weak_classes']),
               'observed_TRAIN_label_order_sha256': digest_object(raw.labelled), 'seed': seed,
               'device': str(device), 'epochs': epochs, 'batch_size': batch_size,
               'recipe': REFERENCE_RECIPE, 'assets': ASSETS, 'software': software_identity(),
               'torch_threads': torch.get_num_threads(), 'not_original_unpublished_recipe': True}
    if resume:
        if read_json(out / 'request.json') != request:
            raise ValueError('LNPL weak resume changed raw/LPA identity, source, numerical runtime or recipe')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'request.json', request)
    lock = _lock(out, resume)
    try:
        seed_everything(seed)
        net = build(raw.asset_root, 'weak', request['classes']).to(dev)
        opt = optimizer(net)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        epoch, offset, updates, total_loss, count, order = 0, 0, 0, 0., 0, None
        best, history = None, []
        contract = digest_object(request)
        if resume:
            state = load(out / 'last.pt')
            if state['contract'] != contract:
                raise ValueError('LNPL weak checkpoint no longer matches measured complete inputs')
            net.load_state_dict(state['model'], strict=True)
            opt.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            epoch, offset, updates = state['epoch'], state['offset'], state['updates']
            total_loss, count, order = state['total_loss'], state['count'], state['order']
            best, history = state['best'], state['history']
            restore_rng(state['rng'])
        invocation_updates = 0
        while epoch < epochs:
            net.train()
            if order is None:
                order = torch.randperm(len(raw.labelled)).tolist()
            for start in range(offset, len(order), batch_size):
                entries = [raw.labelled[i] for i in order[start:start + batch_size]]
                images, labels = raw.labelled_batch(entries)
                x = torch.stack([rgb_tensor(a, augment=True) for a in images]).to(dev)
                y = torch.tensor(labels, dtype=torch.long, device=dev)
                opt.zero_grad(set_to_none=True)
                logits = net(x)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Actual LNPL LPA weak objective became nonfinite')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise FloatingPointError('Actual full weak ResNet18 gradients became nonfinite')
                opt.step()
                updates += 1
                invocation_updates += 1
                total_loss += float(loss.detach()) * len(entries)
                count += len(entries)
                offset = min(start + batch_size, len(order))
                finished_epoch = offset == len(order)
                if finished_epoch:
                    value = total_loss / count
                    history.append({'epoch': epoch + 1, 'observed_TRAIN_CE': value,
                                    'observed_patch_count': count, 'updates': updates,
                                    'lr': opt.param_groups[0]['lr']})
                    if best is None or value < best:
                        best = value
                        save({'model': cpu_state(net), 'contract': contract, 'epoch': epoch + 1,
                              'training_loss': value}, out / 'selected.pt')
                    scheduler.step()
                    epoch += 1
                    offset, total_loss, count, order = 0, 0., 0, None
                save({'model': cpu_state(net), 'optimizer': opt.state_dict(),
                      'scheduler': scheduler.state_dict(), 'gradients': {name: None if p.grad is None
                      else p.grad.detach().cpu().clone() for name, p in net.named_parameters()},
                      'epoch': epoch, 'offset': offset, 'updates': updates, 'order': order,
                      'total_loss': total_loss, 'count': count, 'best': best, 'history': history,
                      'rng': rng_state(), 'contract': contract}, out / 'last.pt')
                write_json(out / 'history.json', history)
                done = epoch == epochs
                interrupted = not done and stop_after_updates is not None and invocation_updates >= stop_after_updates
                write_json(out / 'status.json', {'state': 'completed' if done else 'interrupted' if interrupted else 'running',
                    'completed_epochs': epoch, 'next_patch_offset': offset, 'updates': updates,
                    'native50epoch256batch': epochs == 50 and batch_size == 256,
                    'formal_admission': False})
                if interrupted:
                    return out
                if finished_epoch:
                    break
        return out
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


@torch.no_grad()
def score_raw(raw, weak_run, output, device='cpu', batch_size=64):
    """Export real weak probabilities and separate untouched ImageNet features."""
    from .hybrid_supervision_checkpoint import load, checkpoint_path
    from .provenance import software_identity
    import h5py
    run, output = Path(weak_run), Path(output)
    request = read_json(run / 'request.json')
    if (read_json(run / 'status.json')['state'] != 'completed'
            or request['raw_contract'] != raw.contract or request['software'] != software_identity()
            or request['assets'] != ASSETS or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Finish the bound actual LPA weak stage before scoring any raw patches')
    state = load(run / 'selected.pt')
    if state['contract'] != digest_object(request):
        raise ValueError('LNPL selected weak checkpoint contract changed')
    dev = torch.device(device)
    if dev.type not in {'cpu', 'cuda'} or dev.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested LNPL scoring device unavailable')
    weak = build(raw.asset_root, 'weak', len(raw.schema['weak_classes'])).to(dev)
    weak.load_state_dict(state['model'], strict=True)
    weak.eval()
    frozen = build(raw.asset_root, 'frozen').to(dev)
    selected_sha = digest_file(checkpoint_path(run / 'selected.pt'))
    output.mkdir(parents=True, exist_ok=False)
    index = {}
    for number, row in enumerate(raw.rows):
        path = output / ('slide_' + str(number) + '.h5')
        with raw.open(row) as source, h5py.File(path, 'x') as target:
            n = row['shape'][0]
            scored = target.create_dataset('scored_features', (n, 515), dtype='float32')
            target.create_dataset('patch_ids', data=source[raw.schema['patch_ids_key']][:])
            target.create_dataset('coords', data=source[raw.schema['coords_key']][:])
            for start in range(0, n, batch_size):
                images = source[raw.schema['pixels_key']][start:start + batch_size]
                x = torch.stack([rgb_tensor(a) for a in images]).to(dev)
                probabilities = weak(x).softmax(dim=1)[:, raw.schema['positive_weak_class_ids']].sum(dim=1)
                features = frozen(x)
                if features.shape != (len(x), 512) or not torch.isfinite(features).all():
                    raise ValueError('The separate original ImageNet ResNet18 must emit real512D features')
                coords = source[raw.schema['coords_key']][start:start + batch_size]
                value = torch.cat((probabilities[:, None].cpu(), torch.from_numpy(coords).float(), features.cpu()), dim=1)
                scored[start:start + len(x)] = value.numpy()
            target.attrs['raw_manifest_sha256'] = raw.contract
            target.attrs['weak_checkpoint_sha256'] = selected_sha
            target.attrs['frozen_ImageNet_weights_sha256'] = ASSETS['resnet18-5c106cde.pth']
            target.attrs['original_RGB_sha256'] = row['sha256']
        index[row['slide_id']] = {'path': str(path.resolve()), 'sha256': digest_file(path),
            **{k: row[k] for k in ['global_patient_id', 'patient_id', 'subset', 'target']},
            'patch_ids_sha256': row['patch_ids_sha256'], 'coords_sha256': row['coords_sha256'],
            'patch_count': row['shape'][0]}
    write_json(output / 'manifest.json', {'kind': 'lnplmil_actual_weak_probabilities_and_frozen_ImageNet_features',
        'raw_manifest': str(raw.path), 'raw_manifest_sha256': raw.contract,
        'weak_checkpoint': str(checkpoint_path(run / 'selected.pt')), 'weak_checkpoint_sha256': selected_sha,
        'weak_request': str((run / 'request.json').resolve()), 'weak_request_sha256': digest_file(run / 'request.json'),
        'scorer_sha256': digest_file(Path(__file__)), 'software': request['software'],
        'separate_frozen_weights': ASSETS['resnet18-5c106cde.pth'], 'slides': index,
        'weak_reference_recipe': request['recipe'], 'not_original_unpublished_recipe': True})
    return output / 'manifest.json'

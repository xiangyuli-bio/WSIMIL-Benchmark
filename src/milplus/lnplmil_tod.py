"""Complete native TOD training over actually scored/selected LNPL patches.

The released CE-overwrite/one-shot-SP and paper residual/auxiliary/disjoint-SP
conditions remain distinct. Neither accepts scores or features in place of the
mandatory bound, fully trained actual-RGB weak-classifier stage.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .io import digest_file, digest_object, read_json, write_json
from .lnplmil_source import (model, paper_model, select_top200, filter_superpatches,
                            paper_disjoint_superpatches, objective, validate_scored_features, COMMIT)
from .lnplmil_optim import create_optimizer

CONDITIONS = {'released_source_with_reference_weak', 'paper_reference_with_reference_weak'}


def canonical_candidate_order(value):
    """Keep every actual retained row; define score-tie input order by level0 XY.

    Disjoint HNSW can return the identical retained multiset in different
    neighbor order. TOD is order-aware, so tie ordering is an experimental
    condition. No score, feature, coordinate or instance multiplicity changes.
    The following original torch.topk operation itself remains unchanged.
    """
    validate_scored_features(value)
    if value.device.type != 'cpu':
        raise ValueError('Canonical LNPL candidate ordering occurs on actual CPU SP output')
    coordinates = value[:, 1:3].numpy()
    indices = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    return value[torch.from_numpy(indices.copy())]


def verified_scoring(path):
    from .lnplmil_weak import ASSETS, REFERENCE_RECIPE
    from . import lnplmil_weak
    from .hybrid_supervision_checkpoint import load
    from .provenance import software_identity
    meta = read_json(path)
    if (meta.get('kind') != 'lnplmil_actual_weak_probabilities_and_frozen_ImageNet_features'
            or meta.get('scorer_sha256') != digest_file(Path(lnplmil_weak.__file__))
            or meta.get('software') != software_identity()
            or digest_file(meta['weak_request']) != meta['weak_request_sha256']
            or digest_file(meta['weak_checkpoint']) != meta['weak_checkpoint_sha256']
            or meta['separate_frozen_weights'] != ASSETS['resnet18-5c106cde.pth']):
        raise ValueError('LNPL requires bound actual weak/scorer/source/weight evidence before TOD')
    request = read_json(meta['weak_request'])
    state = load(meta['weak_checkpoint'])
    status = read_json(Path(meta['weak_request']).parent / 'status.json')
    if (request['backend'] != 'lnplmil_actual_LPA_weak_ResNet18_reference'
            or request['raw_contract'] != meta['raw_manifest_sha256']
            or request['recipe'] != REFERENCE_RECIPE or state['contract'] != digest_object(request)
            or status['state'] != 'completed' or request['software'] != meta['software']):
        raise ValueError('Complete the genuine bound LPA training, not a feature-only surrogate')
    return meta


def prepare_selection(scored_manifest, output, condition, radius=50, ratio=.4, canonical_hnsw=False):
    if condition not in CONDITIONS:
        raise ValueError('Choose an explicit full paper or released LNPL condition')
    if type(radius) is not int or radius != 50 or ratio != .4:
        raise ValueError('This native condition retains source SP50/ROI0.4; distinct settings need separate protocols')
    if type(canonical_hnsw) is not bool:
        raise ValueError('Declare the original default-thread or canonical single-build-thread HNSW condition')
    from .lnplmil_raw import RawInputs
    from .hybrid_supervision_checkpoint import load
    scored_manifest, output = Path(scored_manifest).resolve(), Path(output).resolve()
    meta = verified_scoring(scored_manifest)
    raw = RawInputs(meta['raw_manifest'])
    if raw.contract != meta['raw_manifest_sha256'] or set(meta['slides']) != set(raw.meta['slides']):
        raise ValueError('LNPL scored patches no longer bind the measured raw patient/patch manifest')
    if digest_file(meta['weak_checkpoint']) != meta['weak_checkpoint_sha256']:
        raise ValueError('LNPL selected actual weak checkpoint bytes changed')
    output.mkdir(parents=True, exist_ok=False)
    slides = {}
    for number, (name, row) in enumerate(meta['slides'].items()):
        original = raw.meta['slides'][name]
        if (digest_file(row['path']) != row['sha256']
                or any(row[k] != original[k] for k in ['global_patient_id', 'patient_id', 'subset', 'target'])):
            raise ValueError('LNPL actual scoring bytes/patient labels/partitions changed')
        raw.verify(original)
        with h5py.File(row['path'], 'r') as file:
            for key, expected in [('raw_manifest_sha256', raw.contract),
                                  ('weak_checkpoint_sha256', meta['weak_checkpoint_sha256']),
                                  ('original_RGB_sha256', original['sha256']),
                                  ('frozen_ImageNet_weights_sha256', meta['separate_frozen_weights'])]:
                if file.attrs.get(key) != expected:
                    raise ValueError('LNPL scored file has a different raw/weak/frozen-weight contract')
            value = torch.from_numpy(file['scored_features'][:])
            validate_scored_features(value)
            coords = file['coords'][:]
            ids = file['patch_ids'].asstr()[:].tolist()
            if (value.shape != (row['patch_count'], 515)
                    or digest_object(ids) != original['patch_ids_sha256']
                    or digest_object(coords.tolist()) != original['coords_sha256']
                    or not np.array_equal(value[:, 1:3].numpy(), coords)):
                raise ValueError('LNPL scored rows no longer preserve actual patch coordinates and identities')
        if condition == 'released_source_with_reference_weak':
            filtered = filter_superpatches(value, radius=radius, ratio=ratio, canonical=canonical_hnsw)
            details = {'algorithm': 'literal_source_one_shot_representatives_with_minimum_input_fallback'}
        else:
            filtered, details = paper_disjoint_superpatches(value, radius=radius, ratio=ratio, canonical=canonical_hnsw)
        if canonical_hnsw:
            filtered = canonical_candidate_order(filtered)
        selected, _ = select_top200(filtered)
        coordinate_index = {tuple(v): i for i, v in enumerate(coords.tolist())}
        indices = []
        for item in selected:
            key = tuple(item[1:3].long().tolist())
            if key not in coordinate_index:
                raise ValueError('SP-LNPL fabricated a spatial patch')
            index = coordinate_index[key]
            if not torch.equal(value[index], item):
                raise ValueError('SP-LNPL changed the actual weak probability or frozen feature row')
            indices.append(index)
        path = output / ('selected_' + str(number) + '.h5')
        with h5py.File(path, 'x') as file:
            file.create_dataset('scored_features', data=selected.numpy())
            file.create_dataset('original_indices', data=np.asarray(indices, dtype=np.int64))
            file.create_dataset('patch_ids', data=np.asarray([ids[i] for i in indices], dtype=h5py.string_dtype('utf-8')))
        slides[name] = {**{k: row[k] for k in ['global_patient_id', 'patient_id', 'subset', 'target']},
            'path': str(path), 'sha256': digest_file(path), 'original_scored_file': row['path'],
            'original_scored_sha256': row['sha256'], 'original_patch_count': len(value),
            'selected_actual_indices': indices, 'selection': details}
    manifest = {'kind': 'lnplmil_bound_complete_native_Top200', 'condition': condition,
                'source_commit': COMMIT, 'scored_manifest': str(scored_manifest),
                'scored_manifest_sha256': digest_file(scored_manifest),
                'raw_manifest': str(raw.path), 'raw_contract': raw.contract, 'slides': slides,
                'positive_weak_class_ids': raw.schema['positive_weak_class_ids'],
                'HNSW_execution': {'canonical_single_build_thread': canonical_hnsw,
                    'indexThreadQty': 1 if canonical_hnsw else 'original_default_parallel',
                    'candidate_order': 'level0_XY_before_original_TopK' if canonical_hnsw else 'original_source_return_order',
                    'M':16,'post':0,'efConstruction':400,'query_ef':90},
                'native': {'top_k': 200, 'feature_dimension': 512, 'radius': radius, 'ROI_ratio': ratio}}
    write_json(output / 'manifest.json', manifest)
    return output / 'manifest.json'


class SelectedInputs:
    def __init__(self, manifest):
        from .lnplmil_raw import RawInputs
        self.path = Path(manifest).resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('kind') != 'lnplmil_bound_complete_native_Top200'
                or self.meta.get('condition') not in CONDITIONS or self.meta.get('source_commit') != COMMIT
                or digest_file(self.meta['scored_manifest']) != self.meta['scored_manifest_sha256']):
            raise ValueError('Use the full bound LNPL weak/scored/selected pipeline')
        self.raw = RawInputs(self.meta['raw_manifest'])
        self.scoring = verified_scoring(self.meta['scored_manifest'])
        if self.raw.contract != self.meta['raw_contract'] or set(self.meta['slides']) != set(self.raw.meta['slides']):
            raise ValueError('LNPL selected raw patient/patch/annotation contract changed')
        self.rows = [{'slide_id': k, **v} for k, v in self.meta['slides'].items()]
        self.contract = digest_file(self.path)
        self.condition = self.meta['condition']
        self._verified = {}
        for row in self.rows:
            original = self.raw.meta['slides'][row['slide_id']]
            if any(row[k] != original[k] for k in ['global_patient_id', 'patient_id', 'subset', 'target']):
                raise ValueError('LNPL selected slide labels/partitions changed')

    def records(self, subset):
        return [r for r in self.rows if r['subset'] == subset]

    def load(self, row):
        self.raw.verify(self.raw.meta['slides'][row['slide_id']])
        path = Path(row['path'])
        info = path.stat()
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            if digest_file(path) != row['sha256']:
                raise ValueError('LNPL actual selected patch bytes changed')
            self._verified[str(path)] = signature
        with h5py.File(path, 'r') as file:
            value = torch.from_numpy(file['scored_features'][:])
        if len(value) != 200:
            raise ValueError('TOD retains the native Top200, never a smaller test head')
        return validate_scored_features(value)


@torch.no_grad()
def validation(net, selected, device, amp):
    net.eval()
    total = 0.
    rows = selected.records('val')
    for row in rows:
        x = selected.load(row).to(device)
        y = torch.tensor([row['target']], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = net(data=x[None])['logits']
            # Released validation_loss is the bag CE, including the explicit
            # paper-auxiliary condition: no validation ratio labels for choice.
            total += float(F.cross_entropy(logits, y))
    if not rows or not np.isfinite(total):
        raise ValueError('LNPL validation needs actual finite patient-disjoint VAL predictions')
    return total / len(rows)


def apply_accumulated_step(net, optimizer, scaler, amp):
    """Preserve native AMP dynamic-loss-scaling skip semantics and exact counts.

    A scaled-gradient overflow is handled by GradScaler, as in the original
    Lightning precision16 path. CPU/non-AMP nonfinite gradients still fail.
    Lookahead's actual step counter distinguishes a real update from a skip.
    """
    scaler.unscale_(optimizer)
    if not amp and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
        raise FloatingPointError('Full TOD non-AMP accumulated gradients became nonfinite')
    before = optimizer.param_groups[0]['lookahead_step']
    scaler.step(optimizer)
    scaler.update()
    after = optimizer.param_groups[0]['lookahead_step']
    if after not in {before, before + 1} or after == before and not amp:
        raise ValueError('Original Lookahead update counter no longer matches actual AMP stepping')
    optimizer.zero_grad(set_to_none=True)
    return after > before


def fit(selected_manifest, output, device='cpu', seed=2026, epochs=200, patience=10,
        resume=False, stop_after_batches=None):
    from .engine import seed_everything, rng_state, restore_rng
    from .hybrid_supervision_checkpoint import save, load, _lock, cpu_state
    from .provenance import software_identity
    selected = SelectedInputs(selected_manifest)
    if (type(seed) is not int or type(epochs) is not int or epochs < 1
            or type(patience) is not int or patience < 1
            or stop_after_batches is not None and (type(stop_after_batches) is not int or stop_after_batches < 1)):
        raise ValueError('Use explicit native200max/patience10 or declared software budgets')
    dev = torch.device(device)
    if dev.type not in {'cpu', 'cuda'} or dev.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested actual TOD device unavailable; no device fallback')
    amp = dev.type == 'cuda'
    out = Path(output)
    request = {'backend': 'lnplmil_complete_original_TOD', 'condition': selected.condition,
        'selected_manifest': str(selected.path), 'selected_contract': selected.contract,
        'source_commit': COMMIT, 'software': software_identity(), 'seed': seed, 'device': str(device),
        'epochs': epochs, 'patience': patience, 'gradient_accumulation': 2,
        'precision': 'CUDA_float16_GradScaler' if amp else 'explicit_CPU_float32_reference',
        'torch_threads': torch.get_num_threads(), 'optimizer': 'literal_RAdam_Lookahead_2e-4_wd1e-5_k6_alpha0.5',
        'full_model': '4layers_dim256_heads8_ff1024_input512_patches200', 'test_used_in_selection': False}
    if resume:
        if read_json(out / 'request.json') != request:
            raise ValueError('TOD resume changed actual data/source/loss/precision/recipe')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'request.json', request)
    lock = _lock(out, resume)
    try:
        seed_everything(seed)
        net = (model() if selected.condition == 'released_source_with_reference_weak' else paper_model()).to(dev)
        opt = create_optimizer(net)
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        rows = selected.records('train')
        epoch, offset, batches, updates, total_loss, order = 0, 0, 0, 0, 0., None
        best, bad_epochs, history, finished = None, 0, [], False
        amp_skipped_updates = 0
        opt.zero_grad(set_to_none=True)
        contract = digest_object(request)
        if resume:
            state = load(out / 'last.pt')
            if state['contract'] != contract:
                raise ValueError('Full TOD checkpoint contract changed')
            net.load_state_dict(state['model'], strict=True)
            opt.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            for name, parameter in net.named_parameters():
                value = state['gradients'][name]
                parameter.grad = None if value is None else value.to(dev).clone()
            epoch, offset, batches, updates = (state[k] for k in ['epoch', 'offset', 'batches', 'updates'])
            total_loss, order, best, bad_epochs, history, finished = (state[k] for k in ['total_loss', 'order', 'best', 'bad_epochs', 'history', 'finished'])
            amp_skipped_updates = state['amp_skipped_updates']
            restore_rng(state['rng'])
        invocation_batches = 0
        while not finished:
            net.train()
            if order is None:
                order = torch.randperm(len(rows)).tolist()
            for index in range(offset, len(order)):
                row = rows[order[index]]
                x = selected.load(row).to(dev)
                y = torch.tensor([row['target']], dtype=torch.long, device=dev)
                variant = 'released_overwritten_ce' if selected.condition == 'released_source_with_reference_weak' else 'paper_auxiliary'
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    values = objective(net, x, y, variant=variant)
                    scaled_loss = values['loss'] / 2
                scaler.scale(scaled_loss).backward()
                batches += 1
                invocation_batches += 1
                offset = index + 1
                total_loss += float(values['loss'].detach())
                if offset % 2 == 0 or offset == len(order):
                    stepped = apply_accumulated_step(net, opt, scaler, amp)
                    updates += int(stepped)
                    amp_skipped_updates += int(not stepped)
                at_boundary = offset == len(order)
                if at_boundary:
                    val = validation(net, selected, dev, amp)
                    improved = best is None or val < best
                    best = val if improved else best
                    bad_epochs = 0 if improved else bad_epochs + 1
                    if improved:
                        save({'model': cpu_state(net), 'contract': contract, 'epoch': epoch + 1,
                              'VAL_bag_CE': val}, out / 'selected.pt')
                    history.append({'epoch': epoch + 1, 'TRAIN_objective': total_loss / len(rows),
                                    'VAL_bag_CE': val, 'updates': updates, 'bad_epochs': bad_epochs})
                    epoch += 1
                    total_loss, offset, order = 0., 0, None
                    finished = epoch >= epochs or bad_epochs >= patience
                state = {'model': cpu_state(net), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'gradients': {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                  for name, p in net.named_parameters()},
                    'epoch': epoch, 'offset': offset, 'batches': batches, 'updates': updates,
                    'total_loss': total_loss, 'order': order, 'best': best, 'bad_epochs': bad_epochs,
                    'history': history, 'finished': finished, 'rng': rng_state(), 'contract': contract}
                state['amp_skipped_updates'] = amp_skipped_updates
                save(state, out / 'last.pt')
                write_json(out / 'history.json', history)
                interrupted = not finished and stop_after_batches is not None and invocation_batches >= stop_after_batches
                write_json(out / 'status.json', {'state': 'completed' if finished else 'interrupted' if interrupted else 'running',
                    'completed_epochs': epoch, 'batches': batches, 'updates': updates,
                    'amp_skipped_updates': amp_skipped_updates,
                    'native200max_patience10': epochs == 200 and patience == 10,
                    'condition': selected.condition, 'formal_admission': False})
                if interrupted:
                    return out
                if at_boundary:
                    break
        return out
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


@torch.no_grad()
def predict(run, subset='test', device='cpu'):
    from .hybrid_supervision_checkpoint import load
    from .provenance import software_identity
    run = Path(run)
    request = read_json(run / 'request.json')
    if (subset not in {'val', 'test'} or read_json(run / 'status.json')['state'] != 'completed'
            or request['software'] != software_identity()):
        raise ValueError('Complete actual LNPL stages and retain the exact source before selected held-out inference')
    selected = SelectedInputs(request['selected_manifest'])
    if selected.contract != request['selected_contract']:
        raise ValueError('LNPL selected inputs changed before held-out inference')
    dev = torch.device(device)
    if dev.type not in {'cpu', 'cuda'} or dev.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested LNPL selected inference device unavailable')
    net = (model() if selected.condition == 'released_source_with_reference_weak' else paper_model()).to(dev)
    state = load(run / 'selected.pt')
    if state['contract'] != digest_object(request):
        raise ValueError('LNPL selected model contract changed')
    net.load_state_dict(state['model'], strict=True)
    net.eval()
    records = []
    for row in selected.records(subset):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            values = net(data=selected.load(row).to(dev)[None])['Y_prob'][0].float().cpu()
        records.append({**{k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'target']},
                        'subset': subset, 'p_0': float(values[0]), 'p_1': float(values[1]),
                        'prediction': int(values.argmax())})
    frame = pd.DataFrame(records)
    frame.to_csv(run / (subset + '_predictions.csv'), index=False)
    if subset == 'test':
        status = read_json(run / 'status.json')
        status.update(test_evaluated=True, test_prediction_sha256=digest_file(run / 'test_predictions.csv'))
        write_json(run / 'status.json', status)
    return frame

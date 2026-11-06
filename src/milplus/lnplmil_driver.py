"""Unified public LNPL run: actual LPA weak training -> scoring -> SP -> TOD."""
from pathlib import Path
import uuid

import torch

from .data import Bundle
from .engine import seed_everything
from .hybrid_supervision_checkpoint import _lock
from .io import digest_file, read_json, write_json
from .lnplmil_spec import PROFILES
from .provenance import software_identity

BY_ID = {p['id']: p for p in PROFILES}


def inputs(data, fm=None, split=None):
    bundle = Bundle(data, fm, split)
    if bundle.lnplmil is None:
        raise ValueError('Prepare actual RGB224/TRAIN patch annotations before the complete LNPL method')
    return bundle, bundle.lnplmil, bundle.check(deep=True)


def bound_pointer(out, name):
    pointer = read_json(out / (name + '.json'))
    if digest_file(pointer['manifest']) != pointer['sha256']:
        raise ValueError('LNPL ' + name + ' cache manifest bytes changed')
    return Path(pointer['manifest'])


def train(data, output, method, fm=None, split=None, seed=2026, device='cpu', config=None,
          resume=False, stop_after=None):
    from .lnplmil_weak import fit as weak_fit, score_raw
    from .lnplmil_tod import fit as tod_fit, prepare_selection, SelectedInputs, verified_scoring
    if method not in BY_ID:
        raise ValueError('Choose an explicit full LNPL source or paper-reference condition')
    cfg = {'weak_epochs': 50, 'weak_batch_size': 256, 'epochs': 200, 'patience': 10, 'threads': 2}
    if set(config or {}) - set(cfg):
        raise ValueError('LNPL keeps full ResNet18/TOD/SP/Top200; only declared phase budgets and threads vary')
    cfg.update(config or {})
    if (any(type(v) is not int or v < 1 for v in cfg.values()) or type(seed) is not int
            or stop_after is not None and (type(stop_after) is not int or stop_after < 1)):
        raise ValueError('LNPL needs explicit positive phase budgets and actual progress interruption units')
    bundle, raw, inventory = inputs(data, fm, split)
    out = Path(output).expanduser().resolve()
    request = {'backend': 'lnplmil_full_raw_LPA_pipeline', 'data': str(bundle.root), 'method': method,
        'fm': bundle.fm, 'split': bundle.split, 'config': cfg, 'seed': seed, 'device': str(device),
        'software': software_identity(), 'bundle_sha256': bundle.fingerprint,
        'raw_contract': raw.contract, 'feature_inventory_sha256': inventory['feature_inventory_sha256'],
        'synthetic': bool(bundle.config.get('synthetic', False)), 'condition': BY_ID[method]['condition'],
        'weak_training_recipe': 'explicit_reference_not_original_unpublished_configuration', 'formal_admission': False}
    if resume:
        if read_json(out / 'request.json') != request:
            raise ValueError('Full LNPL resume changed patient/raw/LPA/source/precision/recipe identity')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'request.json', request)
    lock = _lock(out, resume)
    try:
        torch.set_num_threads(cfg['threads'])
        seed_everything(seed)
        remaining = stop_after
        weak = out / 'weak'
        from .hybrid_supervision_checkpoint import load, exists
        previous = load(weak / 'last.pt')['updates'] if exists(weak / 'last.pt') else 0
        weak_fit(raw, weak, device, seed, cfg['weak_epochs'], cfg['weak_batch_size'],
                 resume=(weak / 'request.json').exists(), stop_after_updates=remaining)
        weak_status = read_json(weak / 'status.json')
        if remaining is not None:
            remaining -= weak_status['updates'] - previous
        if weak_status['state'] != 'completed' or remaining is not None and remaining <= 0:
            write_json(out / 'status.json', {'state': 'interrupted', 'phase': 'weak',
                'weak_updates': weak_status['updates'], 'weak_finished': weak_status['state'] == 'completed',
                'formal_admission': False})
            return out
        if (out / 'scoring.json').exists():
            scored = bound_pointer(out, 'scoring')
            verified_scoring(scored)
        else:
            scored = score_raw(raw, weak, out / ('scored_' + uuid.uuid4().hex), device)
            write_json(out / 'scoring.json', {'manifest': str(scored), 'sha256': digest_file(scored)})
        if (out / 'selection.json').exists():
            selected = bound_pointer(out, 'selection')
            selected_inputs = SelectedInputs(selected)
            if (selected_inputs.condition != request['condition']
                    or selected_inputs.meta['HNSW_execution']['canonical_single_build_thread'] is not True
                    or selected_inputs.meta['HNSW_execution']['candidate_order'] != 'level0_XY_before_original_TopK'):
                raise ValueError('LNPL source/paper selection condition changed')
        else:
            selected = prepare_selection(scored, out / ('selected_' + uuid.uuid4().hex), request['condition'], canonical_hnsw=True)
            write_json(out / 'selection.json', {'manifest': str(selected), 'sha256': digest_file(selected)})
        tod = out / 'tod'
        tod_fit(selected, tod, device, seed, cfg['epochs'], cfg['patience'],
                resume=(tod / 'request.json').exists(), stop_after_batches=remaining)
        status = read_json(tod / 'status.json')
        if bundle.check(deep=True)['feature_inventory_sha256'] != inventory['feature_inventory_sha256']:
            raise ValueError('LNPL actual RGB or observed annotations changed during complete fitting')
        prior = read_json(out / 'status.json') if (out / 'status.json').exists() else {}
        write_json(out / 'status.json', {**prior, 'state': status['state'], 'phase': 'tod',
            'weak_updates': weak_status['updates'], 'weak_finished': True,
            'TOD_batches': status['batches'], 'TOD_updates': status['updates'],
            'TOD_completed_epochs': status['completed_epochs'], 'formal_admission': False})
        write_json(out / 'method_protocol.json', {**BY_ID[method], 'weak_recipe': read_json(weak / 'request.json')['recipe'],
            'actual_budgets': cfg, 'synthetic': request['synthetic'], 'native_budgets':
            cfg['weak_epochs'] == 50 and cfg['weak_batch_size'] == 256 and cfg['epochs'] == 200 and cfg['patience'] == 10,
            'stop_after_units': 'actual weak optimizer updates followed by actual TOD slide minibatches',
            'HNSW_execution': 'explicit_canonical_indexThreadQty1_M16_post0_efConstruction400_query90_and_level0XY_candidate_tie_order',
            'cache_partial_generations': 'preserved; new generations committed only after complete verification',
            'COAD_survival_implemented': False})
        return out
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def evaluate(run, subset='test', device='cpu'):
    from .lnplmil_tod import predict
    from .engine import score
    out = Path(run).expanduser().resolve()
    request = read_json(out / 'request.json')
    if (subset not in {'val', 'test'} or read_json(out / 'status.json')['state'] != 'completed'
            or request['software'] != software_identity()):
        raise ValueError('Complete the full LNPL protocol with its unchanged source before held-out evaluation')
    bundle, raw, inventory = inputs(request['data'], request['fm'], request['split'])
    if (bundle.fingerprint != request['bundle_sha256'] or raw.contract != request['raw_contract']
            or inventory['feature_inventory_sha256'] != request['feature_inventory_sha256']):
        raise ValueError('LNPL raw data/observed labels/patient partitions changed before selected inference')
    torch.set_num_threads(request['config']['threads'])
    frame = predict(out / 'tod', subset, device)
    frame['source_slides'] = frame.slide_id
    metrics = score(bundle, frame)
    frame.to_csv(out / (subset + '_predictions.csv'), index=False)
    write_json(out / (subset + '_metrics.json'), metrics)
    if subset == 'test':
        status = read_json(out / 'status.json')
        status.update(test_evaluated=True, test_prediction_sha256=digest_file(out / 'test_predictions.csv'))
        write_json(out / 'status.json', status)
    return metrics

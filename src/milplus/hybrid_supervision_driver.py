"""Paper-defined hybrid segmentation/EM/Top16 classifier reference pipeline."""
from pathlib import Path
import importlib.metadata
import uuid

import numpy as np
import pandas as pd
import torch

from .data import Bundle
from .hybrid_supervision_source import build_model, COMMIT
from .hybrid_supervision_ops import top_patch_indices
from .hybrid_supervision_training import segmentation_phase, classification_phase, score_training_patches, score_contract, rgb_tensor
from .hybrid_supervision_checkpoint import _lock
from .hybrid_supervision_checkpoint import Progress, Interrupted, save, load, exists, checkpoint_path
from .io import digest_file, digest_object, read_json, write_json
from .provenance import software_identity

PROFILE = 'hybrid_supervision_paper_full_reference'


def inputs(data, fm=None, split=None):
    bundle = Bundle(data, fm, split)
    if getattr(bundle, 'hybrid_supervision', None) is None:
        raise ValueError('Prepare actual512RGB, TRAIN pixel masks, original DLA34 and patient-disjoint hybrid inputs')
    return bundle, bundle.hybrid_supervision, bundle.check(deep=True)


def train(data, output, method=PROFILE, fm=None, split=None, seed=2026, device='cpu', config=None,
          resume=False, stop_after=None):
    from .engine import seed_everything, rng_state, restore_rng
    if method != PROFILE or type(seed) is not int:
        raise ValueError('Choose the complete hybrid paper-reference protocol and integer seed')
    cfg = {'segmentation_epochs': 30, 'classification_epochs': 15, 'em_rounds': 2, 'threads': 2}
    supplied = config or {}
    if set(supplied)-set(cfg):
        raise ValueError('Hybrid reference keeps full DLA34/512RGB/Top16/R2:1:7; only explicit phase budgets/threads vary')
    cfg.update(supplied)
    if any(type(v) is not int or v < 1 for v in cfg.values()):
        raise ValueError('Declare positive integer hybrid phase budgets, rounds and threads')
    bundle, assets, inventory = inputs(data, fm, split)
    out = Path(output).expanduser().resolve()
    request = {'backend': 'hybrid_supervision_full_paper_reference', 'method': PROFILE, 'data': str(bundle.root),
        'fm': bundle.fm, 'split': bundle.split, 'seed': seed, 'device': str(device), 'config': cfg,
        'software': software_identity(), 'image_runtime': {'Pillow': importlib.metadata.version('Pillow')}, 'bundle_sha256': bundle.fingerprint, 'raw_contract': assets.contract,
        'feature_inventory_sha256': inventory['feature_inventory_sha256'],
        'synthetic': bool(bundle.config.get('synthetic', False)), 'formal_admission': False,
        'reference_choices': {'segmentation_batch': 10, 'mixed_batch_counts': [2, 1, 7],
            'segmentation_epoch': 'ceil(number_of_observed_TRAIN_fine_patches/10)updates',
            'optimizer': 'Adam1e-3 defaultbetas/wd0, adapted from released segmentation optimizer',
            'classification': 'Two balanced image bags/update, mean of16sigmoid patch probabilities',
            'classification_reset_each_round': True, 'classification_epoch': 'max(TRAINpositive,TRAINnegative)balanced_pair_updates',
            'augmentations': 'Joint512RGB/mask90degree rotations/flips and sourceRGBjitter0.4, no physical FOV resize',
            'round_selection': 'Strict minimum final-epoch TRAIN classification loss, as paper round-selection principle',
            'unpublished_training_details': 'Explicit reference reconstruction; no claim of original private trainer or paper score'}}
    if resume:
        if read_json(out/'request.json') != request:
            raise ValueError('Hybrid resume changed data, fine masks, software, initialization, recipe or device')
    else:
        out.mkdir(parents=True, exist_ok=False);write_json(out/'request.json', request)
    lock = _lock(out, resume)
    try:
        seed_everything(seed);torch.set_num_threads(cfg['threads']);dev = torch.device(device)
        if dev.type not in {'cpu', 'cuda'} or dev.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Requested hybrid CPU/CUDA runtime unavailable; no device fallback')
        progress = Progress(out, stop_after)
        boundary_path = out/'round_boundary.pt'
        boundary = load(boundary_path) if exists(boundary_path) else None
        contract = digest_object(request)
        if boundary is not None and boundary['contract'] != contract:
            raise ValueError('Hybrid round boundary no longer matches complete input/source contract')
        start = boundary['next_round'] if boundary is not None else 0
        best_loss = boundary['best_loss'] if boundary is not None else None
        history = boundary['history'] if boundary is not None else []
        for number in range(start, cfg['em_rounds']+1):
            segmentation = build_model(assets.weights, 'segmentation').to(dev)
            if number:
                previous = load(out/('round_'+str(number-1))/'segmentation/last.pt')
                segmentation.load_state_dict(previous['model'], strict=True)
            if boundary is not None:
                restore_rng(boundary['rng'])
            round_dir = out/('round_'+str(number));round_dir.mkdir(exist_ok=True)
            pseudo = None
            if number:
                pseudo = read_json(out/('round_'+str(number-1))/'scored.json')
                previous_score = load(out/('round_'+str(number-1))/'score_rng.pt')
                if pseudo != previous_score['scored'] or digest_file(pseudo['path']) != pseudo['sha256']:
                    raise ValueError('Hybrid previous-round actual pseudo probabilities changed')
            seg = segmentation_phase(bundle, assets, segmentation, round_dir/'segmentation',
                cfg['segmentation_epochs'], dev, progress, pseudo)
            del segmentation
            segmentation = build_model(assets.weights, 'segmentation').to(dev)
            state = load(seg);segmentation.load_state_dict(state['model'], strict=True);restore_rng(state['rng'])
            score_meta = round_dir/'scored.json'
            if score_meta.exists():
                scored = read_json(score_meta)
                score_rng = load(round_dir/'score_rng.pt')
                if (scored != score_rng['scored'] or digest_file(scored['path']) != scored['sha256']
                        or scored['contract'] != score_contract(bundle, assets, segmentation)):
                    raise ValueError('Hybrid prepared TRAIN pseudo/Top16 cache changed')
                restore_rng(score_rng['rng'])
            else:
                scored = score_training_patches(bundle, assets, segmentation, round_dir/('TRAIN_probability.'+uuid.uuid4().hex+'.h5'), dev)
                save({'rng': rng_state(), 'scored': scored}, round_dir/'score_rng.pt');write_json(score_meta, scored)
            del segmentation
            classifier = build_model(assets.weights, 'classification').to(dev)
            cls = classification_phase(bundle, assets, classifier, round_dir/'classification',
                cfg['classification_epochs'], dev, progress, scored)
            state = load(cls);loss = state['last_epoch_training_loss']
            del classifier
            if best_loss is None or loss < best_loss:
                best_loss = loss
                write_json(out/'selected.json', {'round': number, 'segmentation': str(seg), 'segmentation_sha256': digest_file(seg),
                    'classification': str(cls), 'classification_sha256': digest_file(cls), 'training_loss': loss})
            history.append({'round': number, 'TRAIN_classification_loss': loss,
                'segmentation_updates': load(seg)['updates'], 'classification_updates': state['updates'],
                'retained_positive_pseudo': len(scored['pools']['positive']),
                'retained_hard_negative_pseudo': len(scored['pools']['negative'])})
            write_json(out/'history.json', history)
            boundary = {'contract': contract, 'next_round': number+1, 'rng': rng_state(),
                        'best_loss': best_loss, 'history': history}
            save(boundary, boundary_path)
        if bundle.check(deep=True)['feature_inventory_sha256'] != inventory['feature_inventory_sha256']:
            raise ValueError('Hybrid actual RGB/fine annotation bytes changed during fitting')
        write_json(out/'method_protocol.json', {'source_commit': COMMIT, 'profile': PROFILE,
            'paper': 'https://arxiv.org/abs/2107.00934', 'config': cfg,
            'core': 'FullImageNetDLA34Up pixelBCE,TRAINonlysoftpseudoV4/T0.4,R2:1:7,previous-roundsegmentationwarmstart,DLA34Top16meanprobabilityBCE,balancedimages,TRAINlossroundselection',
            'input_patch_pixels': 512, 'stride': 384, 'mpp': .5,
            'source_challenge_RF_is_a_different_condition': True,
            'reference_choices': request['reference_choices'], 'formal_admission': False,
            'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'})
        write_json(out/'status.json', {'state': 'completed', 'completed_rounds': len(history),
                                     'test_evaluated': False, 'formal_admission': False})
        return out
    except Interrupted:
        write_json(out/'status.json', {'state': 'interrupted', 'formal_admission': False});return out
    except BaseException as error:
        write_json(out/'status.json', {'state': 'failed', 'error': repr(error)});raise
    finally:
        lock.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(run, subset='test', device='cpu'):
    from .engine import seed_everything, score
    out = Path(run).resolve();request = read_json(out/'request.json')
    if subset not in {'val', 'test'} or read_json(out/'status.json')['state'] != 'completed':
        raise ValueError('Complete all hybrid segmentation/EM/classification phases before held-out evaluation')
    if request['software'] != software_identity() or request['image_runtime']!={'Pillow':importlib.metadata.version('Pillow')}:
        raise ValueError('Hybrid actual software identity changed')
    bundle, assets, inventory = inputs(request['data'], request['fm'], request['split'])
    if bundle.fingerprint != request['bundle_sha256'] or inventory['feature_inventory_sha256'] != request['feature_inventory_sha256']:
        raise ValueError('Hybrid patient/data/fine annotation contract changed')
    selected = read_json(out/'selected.json')
    for key in ['segmentation', 'classification']:
        if digest_file(selected[key]) != selected[key+'_sha256']:
            raise ValueError('Hybrid selected round checkpoint bytes changed')
    seed_everything(request['seed']);torch.set_num_threads(request['config']['threads']);dev = torch.device(device)
    if dev.type not in {'cpu', 'cuda'} or dev.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested actual hybrid evaluation device is unavailable')
    segmentation = build_model(assets.weights, 'segmentation').to(dev)
    classifier = build_model(assets.weights, 'classification').to(dev)
    segmentation.load_state_dict(load(selected['segmentation'])['model'], strict=True);segmentation.eval()
    classifier.load_state_dict(load(selected['classification'])['model'], strict=True);classifier.eval()
    records = []
    for row in bundle.records(subset):
        probabilities = []
        with assets.open(row) as f:
            rgb = f[assets.schema['pixel_key']]
            for start in range(0, len(rgb), 16):
                x = torch.stack([rgb_tensor(a) for a in rgb[start:start+16]]).to(dev)
                probabilities.append(segmentation(x).sigmoid()[:, 0].cpu())
            indices = top_patch_indices(torch.cat(probabilities), 16).tolist()
            x = torch.stack([rgb_tensor(rgb[i]) for i in indices]).to(dev)
        probability = float(classifier(x).sigmoid().mean())
        if not np.isfinite(probability):
            raise FloatingPointError('Hybrid selected-round image probability became nonfinite')
        records.append({**{k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'target']},
                        'source_slides': row['slide_id'], 'subset': subset, 'p_0': 1-probability,
                        'p_1': probability, 'prediction': int(probability >= .5)})
    frame = pd.DataFrame(records);metrics = score(bundle, frame)
    frame.to_csv(out/(subset+'_predictions.csv'), index=False);write_json(out/(subset+'_metrics.json'), metrics)
    if subset == 'test':
        status = read_json(out/'status.json')
        status.update(test_evaluated=True, test_prediction_sha256=digest_file(out/'test_predictions.csv'))
        write_json(out/'status.json', status)
    return metrics

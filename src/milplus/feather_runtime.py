"""Complete FEATHER downstream candidate with original local pretrained assets.

No randomly initialized model is used by this runtime. Architecture-only
checks belong to separate component tests and never enter this public path.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import random

import numpy as np
import torch

from .data import Bundle
from .feather_source import build_pretrained
from .feather_training import fit, new_progress, setup, knn_probabilities
from .io import atomic_torch_save, digest_file, read_json, write_json
from .provenance import software_identity


def runtime_identity():
    versions = {}
    required = {'transformers': '4.53.1', 'tokenizers': '0.21.4',
                'huggingface-hub': '0.30.2', 'safetensors': '0.5.3', 'loguru': '0.7.3'}
    for name in required:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    software = software_identity()
    if (versions != required or torch.__version__.split('+')[0] != '2.5.1'
            or int(np.__version__.split('.')[0]) != 1):
        raise ValueError('Use the isolated FEATHER transformers4.53.1 native dependencies and explicit Torch2.5.1/NumPy1 port')
    return {'software': software, 'extra_versions': versions,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'numpy': np.__version__,
            'math_environment': {key: os.environ.get(key) for key in [
                'MKL_CBWR', 'MKL_ENABLE_INSTRUCTIONS', 'MKL_DYNAMIC', 'OMP_DYNAMIC',
                'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG',
                'ATEN_CPU_CAPABILITY', 'TORCH_LINEAR_FLATTEN_3D']},
            'torch_build_configuration': torch.__config__.show(),
            'backend_observed_before_initialization': effective_backend_identity(),
            'training_policy': {'deterministic_algorithms': True,
                'cudnn_deterministic': True, 'cudnn_benchmark': False,
                'other_backend_flags_preserved_and_checkpoint_bound': True}}


def effective_backend_identity():
    """Actual math branch and flags, not an inference from package versions.

CPU and CUDA use separate builds/conditions. A fresh process must match these
effective values before checkpoint restoration; no tolerances are changed.
    """
    return {'intraop_threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(),
        'cpu_capability': torch.backends.cpu.get_cpu_capability(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cuda_allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'cuda_allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


def rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in value['cuda']])


def initialize(request):
    seed = request['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(request['config']['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if request['device'].startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError('Actual FEATHER CUDA finetuning device is unavailable')
        torch.cuda.set_device(torch.device(request['device']))


def actual_inputs(request):
    if runtime_identity() != request['runtime']:
        raise ValueError('Actual FEATHER runtime changed')
    bundle = Bundle(request['data'], request['fm'], request['split'])
    inputs = getattr(bundle, 'feather', None)
    if inputs is None or inputs.contract != request['input_contract']:
        raise ValueError('Complete native FEATHER pretrained and patient inputs changed')
    inputs.verify(deep=True)
    return bundle, inputs


def save(path, model, optimizer, scheduler, scaler, generator, state, request, inputs):
    inputs.verify()
    payload = {'schema_version': 2, 'request': request, 'input_contract': inputs.contract,
        'effective_backend': effective_backend_identity(),
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'optimizer_step_called': bool(getattr(optimizer, '_opt_called', False)),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'sampler_rng': generator.get_state(), 'progress': state, 'rng': rng(),
        'gradients': {name: None if p.grad is None else p.grad.detach().clone()
                      for name, p in model.named_parameters()},
        'original_pretrained_evidence': model._milplus_feather_evidence}
    atomic_torch_save(payload, path)
    write_json(str(path) + '.sha256.json', {'sha256': digest_file(path), 'bytes': path.stat().st_size})


def restore(path, model, optimizer, scheduler, scaler, generator, request, inputs):
    proof = read_json(str(path) + '.sha256.json')
    if digest_file(path) != proof['sha256'] or path.stat().st_size != proof['bytes']:
        raise ValueError('Actual complete FEATHER checkpoint bytes changed')
    value = torch.load(path, map_location='cpu', weights_only=False)
    if (value.get('schema_version') != 2 or value.get('effective_backend') != effective_backend_identity()
            or value['request'] != request
            or value['input_contract'] != inputs.contract
            or value['original_pretrained_evidence'] != model._milplus_feather_evidence):
        raise ValueError('FEATHER recovery changed native assets, actual inputs, full recipe or runtime')
    model.load_state_dict(value['model'], strict=True)
    optimizer.load_state_dict(value['optimizer'])
    optimizer._opt_called = value['optimizer_step_called']
    scheduler.load_state_dict(value['scheduler'])
    scaler.load_state_dict(value['scaler'])
    generator.set_state(value['sampler_rng'].cpu())
    parameters = dict(model.named_parameters())
    if set(parameters) != set(value['gradients']):
        raise ValueError('FEATHER recovery gradient parameter set changed')
    for name, parameter in parameters.items():
        grad = value['gradients'][name]
        if grad is not None and (grad.shape != parameter.shape or grad.dtype != parameter.dtype):
            raise ValueError('FEATHER recovery gradient shape or dtype changed')
        parameter.grad = None if grad is None else grad.to(parameter.device).clone()
    restore_rng(value['rng'])
    return value['progress']


@torch.no_grad()
def embeddings(model, bundle, subset, device):
    model.eval()
    rows = bundle.records(subset)
    result = []
    for row in rows:
        x, unused = bundle.load(row)
        embedding, unused = model.forward_features(x.to(device).unsqueeze(0), return_attention=False)
        if embedding.shape != (1, 512) or not torch.isfinite(embedding).all():
            raise ValueError('Actual original FEATHER preclassifier embedding is not finite512D')
        result.append(embedding.float().cpu().numpy()[0])
    return np.asarray(result, dtype=np.float32).reshape(-1, 512), rows


def train(request, output, *, resume=False, stop_after_batch=None):
    out = Path(output)
    bundle, inputs = actual_inputs(request)
    initialize(request)
    model = build_pretrained(inputs.asset, len(bundle.classes)).to(request['device'])
    write_json(out / 'actual_pretrained_model_evidence.json', model._milplus_feather_evidence)
    recipe = request['config']['recipe']
    if recipe['mode'] == 'frozen_knn20':
        if resume or stop_after_batch is not None:
            raise ValueError('Frozen FEATHER KNN is a deterministic feature evaluation, not an interrupted optimizer run')
        if len(bundle.records('train')) < 20:
            raise ValueError('FEATHER frozen KNN keeps k20 and needs at least20 TRAIN slides')
        original = {name: x.detach().cpu().clone() for name, x in model.state_dict().items()}
        model.requires_grad_(False)
        x, rows = embeddings(model, bundle, 'train', request['device'])
        path = out / 'frozen_TRAIN_bank.npz'
        np.savez(path, features=x, targets=np.asarray([int(r['target']) for r in rows], dtype=np.int64))
        if any(not torch.equal(x.cpu(), original[name]) for name, x in model.state_dict().items()):
            raise ValueError('Frozen FEATHER embedding extraction mutated a pretrained parameter')
        write_json(out / 'frozen_TRAIN_bank.json', {'slides': [r['slide_id'] for r in rows],
            'patients': [r['global_patient_id'] for r in rows], 'sha256': digest_file(path),
            'input_contract': inputs.contract, 'pretrained_contract': inputs.asset.contract,
            'all_model_tensors_unchanged': True, 'heldout_samples_in_bank': False})
        inputs.verify(deep=True)
        write_json(out / 'native_status.json', {'state': 'completed', 'mode': recipe['mode'],
            'train_batches': 0, 'optimizer_updates': 0, 'test_evaluated': False, 'formal_admission': False})
        return
    optimizer, scheduler, scaler, generator = setup(model, device=request['device'], seed=request['seed'], recipe=recipe)
    state = new_progress()
    if resume:
        state = restore(out / 'last.pt', model, optimizer, scheduler, scaler, generator, request, inputs)
    train_rows = bundle.records('train')
    val_rows = bundle.records('val') if recipe['mode'] == 'finetune_val20' else []

    def load(row):
        return bundle.load(row)[0]

    def checkpoint(progress, improved):
        save(out / 'last.pt', model, optimizer, scheduler, scaler, generator, progress, request, inputs)
        if improved:
            save(out / 'best.pt', model, optimizer, scheduler, scaler, generator, progress, request, inputs)
        write_json(out / 'history.json', progress['history'])
        write_json(out / 'progress.json', progress)

    result = fit(model, optimizer, scheduler, scaler, generator, train_rows, val_rows,
        load, state, device=request['device'], classes=len(bundle.classes), recipe=recipe,
        on_checkpoint=checkpoint, stop_after_batch=stop_after_batch)
    inputs.verify(deep=True)
    write_json(out / 'native_status.json', {'state': result, 'epochs': state['epochs'],
        'train_batches': state['train_batches'], 'optimizer_updates': state['optimizer_updates'],
        'best_epoch': state['best_epoch'], 'best_validation_loss': state['best_validation_loss'],
        'mode': recipe['mode'], 'heldout_selection': recipe['mode'] == 'finetune_val20',
        'test_evaluated': False, 'formal_admission': False})


@torch.no_grad()
def evaluate(request, output, subset):
    out = Path(output)
    if subset not in {'val', 'test'} or read_json(out / 'native_status.json')['state'] != 'completed':
        raise ValueError('FEATHER evaluation requires a completed original-pretrained downstream run')
    bundle, inputs = actual_inputs(request)
    initialize(request)
    model = build_pretrained(inputs.asset, len(bundle.classes)).to(request['device'])
    recipe = request['config']['recipe']
    neighbors = None
    if recipe['mode'] == 'frozen_knn20':
        model.requires_grad_(False)
        proof = read_json(out / 'frozen_TRAIN_bank.json')
        if (proof['input_contract'] != inputs.contract or proof['pretrained_contract'] != inputs.asset.contract
                or proof['slides'] != [r['slide_id'] for r in bundle.records('train')]
                or proof['patients'] != [r['global_patient_id'] for r in bundle.records('train')]
                or digest_file(out / 'frozen_TRAIN_bank.npz') != proof['sha256']):
            raise ValueError('FEATHER frozen reference bank membership or actual pretrained initialization changed')
        with np.load(out / 'frozen_TRAIN_bank.npz', allow_pickle=False) as bank:
            x, rows = embeddings(model, bundle, subset, request['device'])
            probabilities, indices = knn_probabilities(bank['features'], bank['targets'], x, len(bundle.classes))
            neighbors = [[proof['slides'][i] for i in group] for group in indices]
    else:
        optimizer, scheduler, scaler, generator = setup(model, device=request['device'], seed=request['seed'], recipe=recipe)
        path = out / ('best.pt' if recipe['mode'] == 'finetune_val20' else 'last.pt')
        restore(path, model, optimizer, scheduler, scaler, generator, request, inputs)
        rows = bundle.records(subset)
        model.eval()
        output_probabilities = []
        for row in rows:
            x, unused = bundle.load(row)
            result, unused = model(x.to(request['device']).unsqueeze(0))
            p = result['logits'].float().softmax(-1).cpu().numpy()[0]
            if not np.isfinite(p).all():
                raise FloatingPointError('FEATHER actual prediction probabilities are nonfinite')
            output_probabilities.append(p)
        probabilities = np.asarray(output_probabilities)
    predictions = []
    for i, (row, p) in enumerate(zip(rows, probabilities)):
        value = {'slide_id': row['slide_id'], 'patient_id': row['patient_id'],
            'global_patient_id': row['global_patient_id'], 'target': int(row['target']),
            **{'p_' + str(j): float(v) for j, v in enumerate(p)}}
        if neighbors is not None:
            value['TRAIN_neighbors'] = neighbors[i]
        predictions.append(value)
    inputs.verify(deep=True)
    write_json(out / (subset + '_predictions_native.json'), predictions)
    write_json(out / (subset + '_selection.json'), {'mode': recipe['mode'],
        'input_contract': inputs.contract, 'pretrained_contract': inputs.asset.contract,
        'heldout_test_used_for_selection': False, 'formal_admission': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-batch', type=int)
    parser.add_argument('--evaluate', choices=['val', 'test'])
    args = parser.parse_args()
    request = read_json(args.request)
    if args.evaluate:
        evaluate(request, args.output, args.evaluate)
    else:
        train(request, args.output, resume=args.resume, stop_after_batch=args.stop_after_batch)


if __name__ == '__main__':
    main()

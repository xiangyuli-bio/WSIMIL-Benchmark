"""Complete native PAT/dual-head DSPA2025 reference in an isolated process.

All checkpoints bind every parameter, optimizer state, gradient, sampler and
random state, the actual math backend, concepts and patients. Original expert
asset identity and DSPA-U specification remain unverified; no admission claim.
"""
import argparse
from collections import OrderedDict
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch

from .data import Bundle
from .dspamil_configuration import validate as validate_config
from .dspamil_equations import EquationChoices
from .dspamil_metrics import metrics
from .dspamil_reference import DSPAMICCAIReference, HeadReference
from .dspamil_training import fit, new_progress, setup
from .io import digest_file, read_json, write_json
from .provenance import software_identity


def effective_backend():
    return {'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'cpu_capability': torch.backends.cpu.get_cpu_capability(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cuda_fp16_reduced_precision': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'cuda_bf16_reduced_precision': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


def runtime_identity():
    if torch.__version__.split('+')[0] != '2.5.1':
        raise ValueError('DSPA requires the explicit reviewed Torch2.5.1 numerical runtime')
    return {'software': software_identity(), 'torch': torch.__version__, 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'build': torch.__config__.show(),
        'observed_before_initialize': effective_backend(),
        'math_environment': {key: os.environ.get(key) for key in [
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_CBWR', 'MKL_ENABLE_INSTRUCTIONS',
            'MKL_DYNAMIC', 'OMP_DYNAMIC', 'CUBLAS_WORKSPACE_CONFIG', 'ATEN_CPU_CAPABILITY']},
        'explicit_policy': 'deterministic_FP32_highest_matmul_no_TF32_no_AMP'}


def initialize(request):
    seed = request['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(request['config']['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if request['device'].startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError('The actual requested DSPA CUDA device is unavailable')
        torch.cuda.set_device(torch.device(request['device']))
        torch.cuda.manual_seed_all(seed)


def rng(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if device.startswith('cuda:') else None}


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in value['cuda']])


def inputs_for(request):
    validate_config(request['config'])
    if runtime_identity() != request['runtime'] or software_identity() != request['software']:
        raise ValueError('DSPA actual source, packages or numerical runtime changed')
    bundle = Bundle(request['data'], request['fm'], request['split'])
    inputs = bundle.dspamil
    if inputs is None or inputs.contract != request['input_contract']:
        raise ValueError('DSPA actual concepts, feature population or patient binding changed')
    if (bundle.config.get('evaluation_unit', 'slide')
            != request['config']['recipe']['reference_choices']['evaluation_unit']):
        raise ValueError('DSPA validation and final evaluation units changed')
    inputs.verify(deep=True)
    return bundle, inputs


def build_model(request, inputs):
    config = validate_config(request['config'])
    model = DSPAMICCAIReference(inputs.pattern_concepts,
        equations=EquationChoices(**config['equations']), heads=HeadReference(**config['heads']),
        concept_contract=inputs.contract, mask_convention=config['mask_convention']).to(request['device'])
    if sum(p.numel() for p in model.parameters()) != 162690072:
        raise ValueError('DSPA reference must retain its complete162690072parameter PAT/dual-head graph')
    model._milplus_dspa_evidence = {'PAT_parameters': 151120896,
        'full_parameters': 162690072, 'original_source_commit': model.get_extra_state()['source_commit'],
        'concept_contract': inputs.contract, 'mask_convention': config['mask_convention'],
        'equations': config['equations'], 'head_reference': config['heads'],
        'original_expert_concepts_independently_verified': False,
        'unpublished_architecture_and_training_choices_are_explicit_references': True,
        'checkpoint_serialization': 'torch.save_file_object_constant_ZIP_prefix_complete_payload',
        'DSPA_U_implemented': False, 'formal_admission': False}
    return model


def checkpoint_view(value):
    """Preserve values/types/tensor storage, canonicalize carrier string aliases.

After restoration, old history slide strings and newly loaded bundle strings
can have different object identities. Those identities are not training state;
their pickle memo layout must not prevent exact-byte artifact deduplication.
Tensor/NumPy objects and OrderedDict metadata remain intact and unconverted.
    """
    if type(value) is str:
        return sys.intern(value)
    if type(value) in {dict, OrderedDict}:
        result = type(value)((checkpoint_view(key), checkpoint_view(item)) for key, item in value.items())
        if hasattr(value, '__dict__'):
            result.__dict__.update({key: checkpoint_view(item) for key, item in value.__dict__.items()})
        return result
    if type(value) is list:
        return [checkpoint_view(item) for item in value]
    if type(value) is tuple:
        return tuple(checkpoint_view(item) for item in value)
    return value


def verify_model_concepts(model, inputs):
    if not torch.equal(model.dual_heads.pattern_concepts.detach().cpu(), inputs.pattern_concepts):
        raise ValueError('DSPA loaded/trained concept buffer differs from its actual patient-bound assets')


def save(path, model, optimizer, generator, progress, request, inputs):
    path = Path(path)
    inputs.verify()
    verify_model_concepts(model, inputs)
    payload = {'schema_version': 1, 'kind': 'DSPA2025_complete_PAT_reference_checkpoint',
        'request': request, 'input_contract': inputs.contract,
        'effective_backend': effective_backend(), 'model_evidence': model._milplus_dspa_evidence,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'sampler_rng': generator.get_state(), 'progress': progress, 'rng': rng(request['device']),
        'gradients': {name: None if p.grad is None else p.grad.detach().clone()
                      for name, p in model.named_parameters()}}
    # A file object keeps the serialized ZIP root independent of output names
    # and PIDs. The entire payload is retained; no numerical equality is relaxed.
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        torch.save(checkpoint_view(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    write_json(str(path) + '.sha256.json', {'bytes': path.stat().st_size, 'sha256': digest_file(path)})


def restore(path, model, optimizer, generator, request, inputs):
    path = Path(path)
    proof = read_json(str(path) + '.sha256.json')
    if path.stat().st_size != proof['bytes'] or digest_file(path) != proof['sha256']:
        raise ValueError('DSPA complete checkpoint bytes changed')
    value = torch.load(path, map_location='cpu', weights_only=False)
    if (value.get('schema_version') != 1 or value.get('kind') != 'DSPA2025_complete_PAT_reference_checkpoint'
            or value.get('request') != request or value.get('input_contract') != inputs.contract
            or value.get('effective_backend') != effective_backend()
            or value.get('model_evidence') != model._milplus_dspa_evidence):
        raise ValueError('DSPA recovery changed complete recipe, model, patient/concept inputs or actual math backend')
    parameters = dict(model.named_parameters())
    if set(parameters) != set(value['gradients']):
        raise ValueError('DSPA checkpoint gradient parameter set changed')
    for name, parameter in parameters.items():
        grad = value['gradients'][name]
        if grad is not None and (grad.shape != parameter.shape or grad.dtype != parameter.dtype):
            raise ValueError('DSPA checkpoint gradient shape or dtype changed')
    model.load_state_dict(value['model'], strict=True)
    verify_model_concepts(model, inputs)
    optimizer.load_state_dict(value['optimizer'])
    generator.set_state(value['sampler_rng'].cpu())
    for name, parameter in parameters.items():
        grad = value['gradients'][name]
        parameter.grad = None if grad is None else grad.to(parameter.device).clone()
    restore_rng(value['rng'])
    return value['progress']


def train(request, output, *, resume=False, stop_after_batch=None):
    out = Path(output)
    bundle, inputs = inputs_for(request)
    initialize(request)
    model = build_model(request, inputs)
    write_json(out / 'actual_model_evidence.json', model._milplus_dspa_evidence)
    recipe = request['config']['recipe']
    optimizer, generator = setup(model, seed=request['seed'], recipe=recipe)
    progress = (restore(out / 'last.pt', model, optimizer, generator, request, inputs)
                if resume else new_progress())

    def checkpoint(state, improved):
        save(out / 'last.pt', model, optimizer, generator, state, request, inputs)
        if improved:
            save(out / 'best.pt', model, optimizer, generator, state, request, inputs)
        write_json(out / 'progress.json', state)
        write_json(out / 'history.json', state['history'])

    def audit(state, row, losses):
        if state['train_batches'] in request['config']['audit_checkpoint_updates']:
            save(out / f'audit_update{state["train_batches"]}.pt', model, optimizer,
                 generator, state, request, inputs)

    result = fit(model, optimizer, generator, bundle.records('train'), bundle.records('val'),
        inputs, progress, device=request['device'], recipe=recipe,
        equations=EquationChoices(**request['config']['equations']),
        on_checkpoint=checkpoint, stop_after_batch=stop_after_batch, on_batch=audit)
    inputs.verify(deep=True)
    write_json(out / 'native_status.json', {'state': result, 'epochs': progress['epochs'],
        'train_batches': progress['train_batches'], 'optimizer_updates': progress['optimizer_updates'],
        'best_epoch': progress['best_epoch'], 'best_validation_qwk': progress['best_validation_qwk'],
        'full_PAT_and_all_dual_mechanisms': True, 'actual_model_parameters': 162690072,
        'test_evaluated': False, 'original_expert_assets_verified': False,
        'paper_reproduction': False, 'formal_admission': False})


@torch.no_grad()
def evaluate(request, output, subset):
    out = Path(output)
    if subset not in {'val', 'test'} or read_json(out / 'native_status.json')['state'] != 'completed':
        raise ValueError('DSPA evaluation requires completed full native training and locked held-out population')
    bundle, inputs = inputs_for(request)
    initialize(request)
    model = build_model(request, inputs)
    optimizer, generator = setup(model, seed=request['seed'], recipe=request['config']['recipe'])
    progress = restore(out / 'best.pt', model, optimizer, generator, request, inputs)
    model.eval()
    rule = request['config']['recipe']['reference_choices']['ordinal_decision']
    predictions = []
    for row in bundle.records(subset):
        x = inputs.load_features(row).to(request['device']).unsqueeze(0)
        # No true-grade concept or pattern target is queried or passed.
        values = model(x)
        p = values['student_logits'].float().softmax(-1).cpu().numpy()[0]
        predictions.append({'slide_id': row['slide_id'], 'patient_id': row['patient_id'],
            'global_patient_id': row['global_patient_id'], 'subset': subset,
            'target': int(row['target']), 'dspamil_decision': rule,
            **{f'p_{i}': float(value) for i, value in enumerate(p)}})
    result = metrics(pd.DataFrame(predictions), rule=rule,
        unit=request['config']['recipe']['reference_choices']['evaluation_unit'])
    inputs.verify(deep=True)
    write_json(out / (subset + '_predictions_native.json'), predictions)
    write_json(out / (subset + '_metrics_native.json'), result)
    write_json(out / (subset + '_selection.json'), {'best_epoch': progress['best_epoch'],
        'validation_qwk': progress['best_validation_qwk'], 'student_only': True,
        'true_grade_concepts_consumed': False, 'test_used_for_selection': False,
        'input_contract': inputs.contract, 'decision': rule,
        'original_expert_assets_verified': False, 'formal_admission': False})


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

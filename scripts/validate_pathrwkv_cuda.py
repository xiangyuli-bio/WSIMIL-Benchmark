"""Actual native CUDA operator/gradient/state and complete unified GPU CLI validation."""
import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml

from milplus.data import Bundle
from milplus.engine import evaluate, seed_everything
from milplus.io import digest_file, read_json, write_json
from milplus.pathrwkv import PathRWKVMIL
from milplus.pathrwkv_backend import COMMIT, Backend
from milplus.pathrwkv_demo import create_pathrwkv_demo
from milplus.provenance import software_identity


def difference(actual, expected, atol=3e-4, rtol=3e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {'maximum_absolute': float((actual-expected).abs().max()),
            'relative_l2': float((actual-expected).norm()/expected.norm().clamp_min(1e-12)),
            'atol': atol, 'rtol': rtol}


def kernels():
    result = []
    source, reference = Backend('native'), Backend('torch')
    for shape in [(1, 1, 64), (2, 7, 128), (1, 131, 192)]:
        seed_everything(29)
        b, t, c = shape
        args = [(torch.randn(*shape, device='cuda')*.2).requires_grad_() for _ in range(4)]
        args[3] = (args[3].detach()-2).requires_grad_()
        args.append((torch.randn(c//64, 64, device='cuda')*.2).requires_grad_())
        other = [x.detach().clone().requires_grad_() for x in args]
        y, expected = source(*args), reference(*other)
        item = {'shape': list(shape), 'forward': difference(y, expected)}
        weights = torch.randn_like(y)
        gradients = torch.autograd.grad((y*weights).sum(), args)
        ref_gradients = torch.autograd.grad((expected*weights).sum(), other, allow_unused=True)
        # With a zero initial state and T=1, decay changes only the final
        # unreturned state, never the output. Tensor autograd therefore has
        # no decay edge; the native backward correctly exposes a zero tensor.
        # Permit precisely this analytic structural zero, not arbitrary missing
        # gradients in longer sequences or in any other input.
        unused = [i for i, gradient in enumerate(ref_gradients) if gradient is None]
        assert unused == ([3] if t == 1 else []), unused
        ref_gradients = tuple(torch.zeros_like(other[i]) if gradient is None else gradient
                              for i, gradient in enumerate(ref_gradients))
        item['reference_structural_zero_gradients'] = ['log_decay'] if t == 1 else []
        item['gradients'] = {name: difference(x, z) for name, x, z in
                             zip(['receptance', 'key', 'value', 'log_decay', 'bonus'], gradients, ref_gradients)}
        state = torch.randn(b, c//64, 64, 64, device='cuda')*.1
        ref_state = state.clone()
        chunks, ref_chunks = [], []
        with torch.no_grad():
            for start in range(0, t, 5):
                inputs = [x[:, start:start+5] for x in args[:4]]+[args[4]]
                x, state = source(*inputs, state)
                z, ref_state = reference(*inputs, ref_state)
                chunks.append(x);ref_chunks.append(z)
        item['stateful_forward'] = difference(torch.cat(chunks, 1), torch.cat(ref_chunks, 1))
        item['final_matrix_state'] = difference(state, ref_state)
        result.append(item)
    return result, source.receipt


def network(root):
    bundle = Bundle(create_pathrwkv_demo(root/'operator_model_data', dimension=1024))
    results = []
    for precision in ['32', 'bf16-mixed']:
        seed_everything(41)
        native = PathRWKVMIL(bundle, hidden=768, max_tiles=9, inference_chunk=5,
                            precision=precision, backend='native').cuda()
        tensor = PathRWKVMIL(bundle, hidden=768, max_tiles=9, inference_chunk=5,
                            precision=precision, backend='torch').cuda()
        tensor.network.load_state_dict(copy.deepcopy(native.network.state_dict()))
        x = torch.randn(4, 9, bundle.dim, device='cuda', requires_grad=True)
        other = x.detach().clone().requires_grad_()
        coords = torch.arange(9, device='cuda').view(1, 9, 1).expand(4, -1, 2).float()*224
        targets, masks = native.tasks.tensors(bundle.records('train')[:4], 'cuda')
        before = torch.cuda.get_rng_state()
        with native.autocast('cuda'):
            outputs = native.network(x, coords)
            loss = native.objective(outputs, targets, masks)
        torch.cuda.set_rng_state(before)
        with tensor.autocast('cuda'):
            expected = tensor.network(other, coords)
            ref_loss = tensor.objective(expected, targets, masks)
        # BF16 may round accumulated CUDA/torch-reference reductions differently.
        tolerance = .035 if precision == 'bf16-mixed' else 3e-4
        item = {'precision': precision,
                'heads': {name: difference(outputs[name], expected[name], tolerance, tolerance) for name in outputs},
                'joint_loss': difference(loss, ref_loss, tolerance, tolerance)}
        loss.backward();ref_loss.backward()
        item['input_gradient'] = difference(x.grad, other.grad, tolerance, tolerance)
        item['parameter_gradients'] = {}
        for (name, p), (ref_name, q) in zip(native.network.named_parameters(), tensor.network.named_parameters()):
            assert name == ref_name and (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                item['parameter_gradients'][name] = difference(p.grad, q.grad, tolerance, tolerance)
        native.eval()
        with torch.no_grad(), native.autocast('cuda'):
            streamed = native.stream(((x[0, i:i+5].detach(), coords[0, i:i+5]) for i in range(0, 9, 5)), 'cuda')
            projected = native.network.input_projector(x[:1].detach())
            parallel = native.network.head(native.network.backbone.forward_parallel(projected, coords[:1]))
        item['repaired_stream_full_sequence'] = {name: difference(streamed[name], parallel[name], tolerance, tolerance) for name in streamed}
        results.append(item)
        del native, tensor, outputs, expected, loss, ref_loss
        torch.cuda.empty_cache()
    return results


def cli_runs(root):
    results = []
    for dimension, precision in [(1024, '32'), (1536, 'bf16-mixed')]:
        data = create_pathrwkv_demo(root/f'cli_data_{dimension}', dimension=dimension)
        bundle = Bundle(data)
        for profile in ['pathrwkv_source', 'pathrwkv_repaired']:
            # Full original hidden768/depth2/head64; short two-epoch software check.
            config = {'epochs': 2, 'threads': 2, 'model': {
                'max_tiles': 13, 'inference_chunk': 11, 'precision': precision, 'backend': 'native'}}
            config_path = root/f'{profile}_{dimension}.yaml'
            config_path.write_text(yaml.safe_dump(config))
            out = root/f'{profile}_{dimension}'
            command = [sys.executable, '-c', 'from milplus.cli import main; main()', 'run',
                '--data', str(data), '--method', profile, '--output', str(out),
                '--device', 'cuda', '--config', str(config_path)]
            with (root/f'{profile}_{dimension}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert not (out/'test_predictions.csv').exists()
            state = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            extra = state['model']['_extra_state']
            updates = 2*(len(bundle.records('train'))//4)
            assert extra['optimizer_steps'] == updates and extra['completed_epochs'] == 2
            assert state['model']['network.input_projector.0.weight'].shape == (768, dimension)
            assert state['model']['network.backbone.MTL_module.weight'].shape == (3*768, 768)
            assert all(k in state['model'] for k in ['network.head.task_heads.disease.weight',
                'network.head.task_heads.age.weight', 'network.head.task_heads.OS.weight'])
            history = read_json(out/'history.json')
            values = [h['validation']['pathrwkv_val_loss'] for h in history]
            selected = values.index(min(values))
            assert torch.load(out/'best.pt', map_location='cpu', weights_only=False)['epoch'] == selected
            metrics = evaluate(out, device='cuda')
            checksum = digest_file(out/'test_predictions.csv')
            assert metrics == evaluate(out, device='cuda') and checksum == digest_file(out/'test_predictions.csv')
            assert len(pd.read_csv(out/'test_predictions.csv')) == len(bundle.records('test'))
            results.append({'profile': profile, 'feature_dimension': dimension, 'hidden': 768,
                'heads': 3, 'depth': 2, 'precision': precision, 'epochs': 2,
                'optimizer_updates': updates, 'selected_epoch': selected,
                'test_prediction_sha256': checksum, 'metrics': metrics})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve();root.mkdir(parents=True, exist_ok=False)
    report = {'state': 'running', 'source_commit': COMMIT, 'device': 'cuda', 'synthetic': True}
    start = time.monotonic()
    try:
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA device required; no CPU fallback')
        torch.set_num_threads(2)
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), software=software_identity())
        write_json(root/'validation.json', report)
        report['kernels'], report['native_source_receipt'] = kernels()
        report['phase'] = 'native_kernels_passed';write_json(root/'validation.json', report)
        report['full_model'] = network(root)
        report['phase'] = 'native_model_gradients_passed';write_json(root/'validation.json', report)
        report['cli_runs'] = cli_runs(root)
        assert report['software'] == software_identity()
        report.update(state='passed', phase='completed', peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            scope='Actual pinned CUDA kernels/all input gradients/matrix carry, native768 heterogeneous model gradients and four full unified CLI train/reload/predict runs. Synthetic only; two epochs and short training sequences; no natural WSI or published metric reproduction.')
    except BaseException as error:
        report.update(state='failed', error_type=type(error).__name__, error=str(error))
        (root/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-start
        write_json(root/'validation.json', report)
    print(json.dumps({k: report[k] for k in ['state', 'gpu', 'torch', 'torch_cuda', 'wall_seconds']}))


if __name__ == '__main__':
    main()

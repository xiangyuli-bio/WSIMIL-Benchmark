"""Explicit fused/reference MoMIL CUDA admission; missing GPU is never a passing result."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from milplus.io import write_json
from milplus.momil import author_network
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[1, 7, 65, 257])
    parser.add_argument('--input-dim', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--state-size', type=int, default=64)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output; previous evidence is not overwritten')
    if min(args.lengths+[args.input_dim, args.hidden, args.state_size]) < 1 or args.hidden % 32:
        raise ValueError('Positive dimensions and hidden divisible by32 required')
    evidence = {'software': software_identity(), 'scope': 'full fixed MoMIL fused/reference outputs and parameter gradients, synthetic CUDA only',
                'settings': vars(args), 'checks': []}
    if not torch.cuda.is_available():
        write_json(output, {**evidence, 'state': 'not_run', 'reason': 'CUDA GPU is unavailable'})
        raise SystemExit(2)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        fused, identity = author_network(args.hidden, args.state_size, 'cuda')
        reference, _ = author_network(args.hidden, args.state_size, 'torch')
        evidence.update(backend=identity, gpu=torch.cuda.get_device_name(0))
        for length in args.lengths:
            torch.manual_seed(2026)
            a = fused(args.input_dim, 3, 0., 'relu').cuda()
            b = reference(args.input_dim, 3, 0., 'relu').cuda()
            b.load_state_dict(a.state_dict()); x = torch.randn(length, args.input_dim, device='cuda')
            y = torch.tensor([1], device='cuda'); outputs = []
            for model in (a, b):
                prediction = model(x)[0]; outputs.append(prediction)
                F.cross_entropy(prediction, y).backward()
            torch.testing.assert_close(*outputs, rtol=3e-4, atol=2e-5)
            maximum = 0.
            for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
                if name != other or p.grad is None or q.grad is None:
                    raise AssertionError('Missing/reordered gradient: '+name)
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=5e-5,
                                           msg=lambda message, name=name: name+': '+message)
                maximum = max(maximum, float((p.grad-q.grad).abs().max()))
            evidence['checks'].append({'length': length, 'state': 'passed', 'gradient_max_abs': maximum,
                                       'logit_max_abs': float((outputs[0]-outputs[1]).abs().max())})
            write_json(output, {**evidence, 'state': 'running'})
        write_json(output, {**evidence, 'state': 'passed'})
    except Exception as error:
        write_json(output, {**evidence, 'state': 'failed', 'error': repr(error)})
        raise
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

"""Admit a MamMIL fused CUDA environment by comparing full networks with explicit SSD."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from milplus.io import write_json
from milplus.mammil import author_network
from milplus.mammil_graph import build_graph
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[1, 7, 65, 257])
    parser.add_argument('--input-dim', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--state-size', type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output; previous CUDA evidence is not overwritten')
    if min(args.lengths+[args.input_dim, args.hidden, args.state_size]) < 1 or args.hidden % 32:
        raise ValueError('Positive dimensions required; hidden must be divisible by32')
    evidence = {'software': software_identity(), 'scope': 'Full source/repaired MamMIL fused CUDA versus explicit SSD outputs/gradients; synthetic inputs only',
                'settings': vars(args), 'checks': []}
    if not torch.cuda.is_available():
        write_json(output, {**evidence, 'state': 'not_run', 'reason': 'CUDA GPU is unavailable'})
        raise SystemExit(2)
    try:
        evidence['gpu'] = torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for profile in ['mammil_source', 'mammil']:
            fused, identity = author_network(profile, 'cuda')
            reference, _ = author_network(profile, 'torch')
            evidence['backend'] = identity
            for length in args.lengths:
                torch.manual_seed(2026)
                settings = {'input_dim': args.input_dim, 'embed_dim': args.hidden, 'layer_n': 2,
                            'd_state': args.state_size, 'num_classes': 2, 'dropout': 0.}
                left, right = fused(**settings).cuda(), reference(**settings).cuda()
                right.load_state_dict(left.state_dict())
                features = np.random.default_rng(2026).normal(size=(length, args.input_dim)).astype('float32')
                coords = np.column_stack([np.arange(length)*256., np.zeros(length)])
                arrays = build_graph(features, coords)
                graph = Data(x=torch.from_numpy(features), **{k: torch.from_numpy(v) for k, v in arrays.items()}).cuda()
                label = torch.tensor([1], device='cuda')
                results = []
                for network in (left, right):
                    result = network(data=graph, label=label)
                    result['loss'].backward(); results.append(result)
                torch.testing.assert_close(results[0]['logits'], results[1]['logits'], rtol=3e-4, atol=2e-5)
                torch.testing.assert_close(results[0]['loss'], results[1]['loss'], rtol=3e-4, atol=2e-5)
                max_gradient, inactive = 0., []
                for (name, a), (other, b) in zip(left.named_parameters(), right.named_parameters()):
                    if name != other:
                        raise AssertionError('Parameter ordering differs')
                    if a.grad is None or b.grad is None:
                        if a.grad is not None or b.grad is not None or profile != 'mammil_source' or not name.startswith('layers.1.'):
                            raise AssertionError('Unexpected inactive gradient: '+name)
                        inactive.append(name); continue
                    torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=5e-5,
                                               msg=lambda message, name=name: name+': '+message)
                    max_gradient = max(max_gradient, float((a.grad-b.grad).abs().max()))
                evidence['checks'].append({'profile': profile, 'length': length, 'state': 'passed',
                    'logit_max_abs': float((results[0]['logits']-results[1]['logits']).abs().max()),
                    'gradient_max_abs': max_gradient, 'inactive_parameters': inactive})
                write_json(output, {**evidence, 'state': 'running'})
        write_json(output, {**evidence, 'state': 'passed'})
    except Exception as error:
        write_json(output, {**evidence, 'state': 'failed', 'error': repr(error)})
        raise
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

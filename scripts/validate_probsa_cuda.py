"""Actually execute full ProbSA routes on CUDA when hardware is available."""
import argparse
from pathlib import Path

import torch

from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import write_json
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    if not torch.cuda.is_available():
        write_json(output/'validation.json', {'state': 'not_run', 'reason': 'CUDA hardware/runtime unavailable', 'software': identity})
        print('not_run: CUDA hardware/runtime unavailable')
        return
    checks = []
    for method, dimension in [('probsa_abmil_probmean', 1536), ('probsa_transformer_probmean', 1024)]:
        data = create_demo(output/(method+str(dimension)+'_data'), patients=30, classes=2, dimension=dimension)
        run = train(data, method, output/(method+str(dimension)), device='cuda', config={'epochs': 3, 'threads': 1, 'model': {'cycles': 2}})
        first = evaluate(run, device='cuda'); second = evaluate(run, device='cuda')
        if first != second:
            raise ValueError('CUDA ProbSA repeated metrics differ')
        checks.append({'method': method, 'dimension': dimension, 'state': 'passed', 'metrics': first})
    if software_identity() != identity:
        raise ValueError('ProbSA source changed during CUDA verification')
    write_json(output/'validation.json', {'state': 'passed', 'software': identity, 'device': torch.cuda.get_device_name(0),
        'checks': checks, 'scope': 'Synthetic CUDA complete method recipes only; no WSI/paper-score reproduction'})


if __name__ == '__main__': main()

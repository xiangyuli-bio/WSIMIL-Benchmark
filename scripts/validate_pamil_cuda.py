"""Record actual CUDA admission of full PAMIL staged routes, never inferred GPU success."""
import argparse
from pathlib import Path

import torch

from milplus.engine import evaluate, train
from milplus.io import write_json
from milplus.pamil_demo import create_pamil_demo
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
    for method, dimension in [('pamil_prototype_source', 1024), ('pamil_prototype_repaired', 1536), ('pamil_prototype_multilabel', 768)]:
        data = create_pamil_demo(output/(method+'_data'), multilabel=method.endswith('multilabel'), dimension=dimension)
        run = train(data, method, output/method, device='cuda', config={'epochs': 3, 'threads': 1,
            'model': {'prototypes': 6, 'warmup_epochs': 1, 'push_epochs': [2], 'last_epochs': 1}})
        first = evaluate(run, device='cuda'); second = evaluate(run, device='cuda')
        if first != second:
            raise ValueError('CUDA PAMIL repeated metrics differ')
        checks.append({'method': method, 'dimension': dimension, 'state': 'passed', 'metrics': first})
    if software_identity() != identity:
        raise ValueError('PAMIL runtime source changed during CUDA verification')
    write_json(output/'validation.json', {'state': 'passed', 'software': identity, 'device': torch.cuda.get_device_name(0),
        'checks': checks, 'scope': 'Synthetic CUDA full-stage source-recipe checks; no real WSI/paper scores'})


if __name__ == '__main__': main()

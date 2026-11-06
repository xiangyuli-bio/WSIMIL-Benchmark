"""Actually execute full StableMIL routes on CUDA when hardware is available."""
import argparse
from pathlib import Path

import torch

from milplus.engine import evaluate, train
from milplus.io import write_json
from milplus.provenance import software_identity
from milplus.stable_demo import create_stable_demo


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
    for method, dimension in [('stablemil_classification_source',1536),('stablemil_classification_repaired',1024),('stablemil_survival_source',1536),('stablemil_survival_repaired',1024)]:
        data = create_stable_demo(output/(method+str(dimension)+'_data'), patients=30, classes=3, dimension=dimension, survival='survival' in method)
        run = train(data, method, output/(method+str(dimension)), device='cuda', config={'epochs': 3, 'threads': 1})
        first = evaluate(run, device='cuda'); second = evaluate(run, device='cuda')
        if first != second:
            raise ValueError('CUDA StableMIL repeated metrics differ')
        checks.append({'method': method, 'dimension': dimension, 'state': 'passed', 'metrics': first})
    if software_identity() != identity:
        raise ValueError('StableMIL source changed during CUDA verification')
    write_json(output/'validation.json', {'state': 'passed', 'software': identity, 'device': torch.cuda.get_device_name(0),
        'checks': checks, 'scope': 'Synthetic CUDA complete method recipes only; no WSI/paper-score reproduction'})


if __name__ == '__main__': main()

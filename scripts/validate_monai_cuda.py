"""Execute native raw-image MONAI routes on CUDA, or record the actual missing hardware."""
import argparse
from pathlib import Path

import torch

from milplus.engine import evaluate, train
from milplus.io import digest_file, write_json
from milplus.monai_mil import MODES
from milplus.provenance import software_identity
from milplus.raw_demo import create_raw_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-weights')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    if not torch.cuda.is_available():
        write_json(output/'validation.json', {'state': 'not_run', 'reason': 'CUDA hardware/runtime unavailable', 'software': identity})
        print('not_run: CUDA hardware/runtime unavailable')
        return
    data = create_raw_demo(output/'data', patients=30, classes=6, tile_size=256, initial_weights=args.initial_weights)
    checks = []
    for profile in MODES:
        run = train(data, profile, output/profile, device='cuda',
                    config={'epochs': 2, 'threads': 1, 'model': {'amp': args.amp}})
        metrics = evaluate(run, device='cuda')
        sha = digest_file(run/'test_predictions.csv')
        if metrics != evaluate(run, device='cuda') or sha != digest_file(run/'test_predictions.csv'):
            raise ValueError('MONAI CUDA repeated predictions differ')
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        assert state['model']['network.myfc.weight'].shape[0] == 5
        assert state['model']['_extra_state']['optimizer_steps'] > 0
        checks.append({'method': profile, 'state': 'passed', 'metrics': metrics, 'test_predictions_sha256': sha,
                       'optimizer_steps': state['model']['_extra_state']['optimizer_steps']})
        del state
    if software_identity() != identity:
        raise ValueError('MONAI software changed during validation')
    write_json(output/'validation.json', {'state': 'passed', 'software': identity, 'device': torch.cuda.get_device_name(0),
        'checks': checks, 'amp': args.amp, 'scope': 'Full native ResNet50/four-layer Transformer or pyramid, six ordinal grades, native256px/44tile/batch4 RGB recipes; synthetic software only, not real WSI or paper-score reproduction'})


if __name__ == '__main__':
    main()

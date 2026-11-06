"""Run full native ResNet34/two-phase Campanella on real CUDA, or record not_run."""
import argparse
from pathlib import Path

import torch

from milplus.engine import evaluate, train
from milplus.io import digest_file, write_json
from milplus.provenance import software_identity
from milplus.raw_region_demo import create_raw_region_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-weights')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    if not torch.cuda.is_available():
        write_json(output/'validation.json', {'state': 'not_run', 'reason': 'CUDA hardware/runtime unavailable', 'software': identity})
        print('not_run: CUDA hardware/runtime unavailable')
        return
    if not args.initial_weights:
        raise ValueError('Native CUDA validation requires the actual local ImageNetV1 ResNet34 initialization')
    data = create_raw_region_demo(output/'data', initial_weights=args.initial_weights)
    run = train(data, 'campanella_rnn', output/'run', device='cuda', seed=6,
                config={'threads': 1, 'model': {'cnn_epochs': 10, 'rnn_epochs': 2}})
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    native = state['model']
    assert native['cnn.conv1.weight'].shape == (64, 3, 7, 7)
    assert native['rnn.fc1.weight'].shape == (128, 512)
    assert native['_extra_state']['cnn_updates'] == 10 and native['_extra_state']['rnn_updates'] == 2
    assert native['_extra_state']['rnn_initialized'] and native['_extra_state']['phase'] == 'rnn'
    metrics = evaluate(run, device='cuda')
    sha = digest_file(run/'test_predictions.csv')
    if metrics != evaluate(run, device='cuda') or sha != digest_file(run/'test_predictions.csv'):
        raise ValueError('Native Campanella CUDA repeated predictions differ')
    if software_identity() != identity:
        raise ValueError('Campanella software changed during CUDA validation')
    write_json(output/'validation.json', {'state': 'passed', 'software': identity, 'device': torch.cuda.get_device_name(0),
        'metrics': metrics, 'test_predictions_sha256': sha, 'cnn_updates': 10, 'rnn_updates': 2,
        'scope': 'Native full ResNet34/ImageNetV1, real OpenSlide reads of synthetic tiled TIFF, source224px/10step128hidden, full Adam/SGD phases, native batch settings with short-dataset tail batches, explicit10CNN/2RNN-epoch software budget. Not natural WSI or paper-number reproduction.'})


if __name__ == '__main__':
    main()

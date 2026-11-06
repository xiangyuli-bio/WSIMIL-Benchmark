"""Run the actual CLI and default C2Aug paper-reference mechanisms on native FM widths."""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from milplus.c2aug import PROFILE
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dimension', type=int, choices=[1024, 1536], default=1024)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Actual validation epoch count must be positive')
    root = Path(args.output).resolve();root.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    if args.device == 'cuda' and not torch.cuda.is_available():
        write_json(root/'validation.json', {'state': 'not_run', 'reason': 'CUDA unavailable'})
        return
    data = create_demo(root/'data', patients=48, dimension=args.dimension)
    cfg = {'epochs': args.epochs, 'threads': 1}
    config = root/'validation_config.yaml';config.write_text(yaml.safe_dump(cfg))
    run = root/'run'
    command = [sys.executable, '-c', 'from milplus.cli import main; main()', 'run',
        '--data', str(data), '--method', 'C2Aug', '--output', str(run),
        '--config', str(config), '--device', args.device]
    with (root/'cli_train.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    bundle = Bundle(data)
    request = read_json(run/'request.json')
    assert request['method'] == PROFILE and request['synthetic']
    actual = request['config']
    assert actual['lr'] == 1e-4 and actual['weight_decay'] == 0 and actual['grad_clip'] is None
    assert actual['epochs'] == args.epochs and actual['patience'] == args.epochs
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    history = read_json(run/'history.json')
    updates = len(bundle.records('train'))*args.epochs
    extra = state['model']['_extra_state']
    assert len(history) == args.epochs and state['epoch'] == args.epochs-1
    assert extra['optimizer_steps'] == updates and extra['completed_epochs'] == args.epochs
    assert state['model']['student.prototypes'].shape == (8, 512)
    assert state['model']['student.augment.reduce.weight'].shape == (256, args.dimension)
    assert state['model']['student.mil.cls_token'].shape == (1, 1, 512)
    assert state['model']['queue'].shape == (256, 128)
    assert extra['queue_count'] == min(updates, 256) and extra['queue_pointer'] == updates % 256
    assert all(v > 0 for v in extra['augmentation_counts'].values())
    train_ids = {r['slide_id'] for r in bundle.records('train')}
    assert {x for x in extra['queue_ids'] if x is not None} <= train_ids
    assert all(h['train']['bag'] > 0 and h['train']['group'] > 0 for h in history)
    values = [h['validation']['c2aug_val_macro_accuracy'] for h in history]
    selected = values.index(max(values))
    assert torch.load(run/'best.pt', map_location='cpu', weights_only=False)['epoch'] == selected
    assert not (run/'test_predictions.csv').exists()
    metrics = evaluate(run, device=args.device)
    checksum = digest_file(run/'test_predictions.csv')
    assert metrics == evaluate(run, device=args.device) and checksum == digest_file(run/'test_predictions.csv')
    assert len(pd.read_csv(run/'test_predictions.csv')) == len(bundle.records('test'))
    assert software_identity() == identity
    report = {'state': 'passed', 'software': identity, 'profile': PROFILE,
        'feature_dimension': args.dimension, 'native_hidden': 512, 'views': 8,
        'reduced_dimension': 256, 'prototypes': 8, 'queue_size': 256,
        'paper_epoch_budget': 200, 'actual_epochs': args.epochs, 'optimizer_updates': updates,
        'completed_full_paper_epoch_budget': args.epochs == 200,
        'first_strict_best_val_epoch': selected, 'augmentation_counts': extra['augmentation_counts'],
        'test_prediction_sha256': checksum, 'metrics': metrics, 'device': args.device,
        'scope': 'Actual unified CLI, complete original512 TransMIL and all paper-reference mechanisms; real checkpoint reload and deterministic prediction on synthetic features. No natural WSI or paper-score reproduction claim.'}
    write_json(root/'validation.json', report)
    print(f'C2Aug native{args.dimension}/512 CLI passed: {args.epochs} actual epochs, {updates} complete Adam/EMA/FIFO updates')


if __name__ == '__main__':
    main()

"""Native-width validation of the full Wibawa supplementary CE/Adam recipe."""
import argparse
from pathlib import Path

import torch

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, seed_everything, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.msattention import PROFILES, MultiAttentionMIL
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 2:
        raise ValueError('Native validation needs at least two actual training epochs')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    report = {'state': 'running', 'software': identity, 'profiles': [], 'device': args.device,
        'scope': 'Synthetic native512/1280 inputs, unchanged original128hidden at all3depths, full shuffled-bag CE/Adam updates, source/repaired dropout, validation selector, best/last checkpoint and repeated CSV. Default10epochs matches supplementary budget. No private tumour-filter/WSI/paper-number claim.'}
    if args.device == 'cuda' and not torch.cuda.is_available():
        report.update(state='not_run', reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json', report); print(report['reason']); return
    for dimension in [512, 1280]:
        data = create_demo(output/f'data_{dimension}', patients=30, classes=3, dimension=dimension)
        bundle = Bundle(data)
        for profile in sorted(PROFILES):
            run = train(data, profile, output/f'{profile}_{dimension}', seed=11, device=args.device,
                        config={'threads': 1, 'epochs': args.epochs})
            state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
            steps = len(units(bundle, 'train'))*args.epochs
            assert state['model']['_extra_state']['optimizer_steps'] == steps
            assert read_json(run/'status.json')['completed_epochs'] == args.epochs
            optimizer = state['optimizers'][0]
            assert all(int(value['step']) == steps for value in optimizer['state'].values())
            seed_everything(11)
            initial = MultiAttentionMIL(bundle, profile)
            for key in ['fc1.0.weight', 'fc2.0.weight', 'fc3.0.weight', 'fc4.0.weight',
                        'attention1.0.weight', 'attention2.0.weight', 'attention3.0.weight']:
                assert not torch.equal(state['model']['network.'+key], initial.network.state_dict()[key]), key
            assert state['model']['network.fc1.0.weight'].shape == (128, dimension)
            for stage in [2, 3]:
                assert state['model'][f'network.fc{stage}.0.weight'].shape == (128, 128)
            history = read_json(run/'history.json')
            expected_epoch = min(range(len(history)), key=lambda e: history[e]['validation']['log_loss'])
            best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            assert best['epoch'] == expected_epoch
            metrics = evaluate(run, device=args.device); sha = digest_file(run/'test_predictions.csv')
            assert metrics == evaluate(run, device=args.device) and sha == digest_file(run/'test_predictions.csv')
            report['profiles'].append({'id': profile, 'input_dimension': dimension, 'hidden': 128,
                'epochs': args.epochs, 'optimizer_updates': steps, 'state': 'passed',
                'selected_epoch': expected_epoch, 'metrics': metrics, 'test_predictions_sha256': sha})
            write_json(output/'validation.json', report)
    if software_identity() != identity:
        raise ValueError('Software changed during native MultiAttentionMIL validation')
    report['state'] = 'passed'; write_json(output/'validation.json', report)
    print('All four native-width/source-repaired training and reload combinations passed')


if __name__ == '__main__':
    main()

"""Train all native-width TTIS routes with original optimizer and TEST-only selection."""
import argparse
from pathlib import Path

import torch

from milplus.data import Bundle
from milplus.engine import evaluate, seed_everything, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity
from milplus.ttis import PROFILES, TTISMIL
from milplus.ttis_demo import create_ttis_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 2:
        raise ValueError('Native validation requires at least two epochs across Lookahead synchronization')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    report = {'state': 'running', 'software': identity, 'profiles': [], 'device': args.device,
        'scope': 'Synthetic native 1024 input/512 hidden/384 attention; all original heads and full optimizer/scheduler/checkpoint/inference route. Explicit sklearn_cpu two-stage clustering, synthetic semantic vectors, FP32. Patience extended to requested epochs solely to verify the complete training budget. No natural MUSK/WSI, native cuML/BF16 or paper-number claim.'}
    if args.device == 'cuda' and not torch.cuda.is_available():
        report.update(state='not_run', reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json', report); print(report['reason']); return
    data = create_ttis_demo(output/'data', dimension=1024, classes=3)
    bundle = Bundle(data)
    for profile in sorted(PROFILES):
        run = train(data, profile, output/profile, seed=11, device=args.device,
                    config={'threads': 1, 'epochs': args.epochs, 'patience': args.epochs})
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        steps = len(units(bundle, 'train'))*args.epochs
        extra = state['model']['_extra_state']
        assert extra['optimizer_steps'] == steps and extra['completed_epochs'] == args.epochs
        assert extra['scheduler']['last_epoch'] == args.epochs
        assert read_json(run/'status.json')['completed_epochs'] == args.epochs
        optimizer = state['optimizers'][0]
        assert all(group['lookahead_step'] == steps for group in optimizer['param_groups'])
        assert any('lookahead_slow_buff' in value for value in optimizer['state'].values())
        active = [value for value in optimizer['state'].values() if 'step' in value]
        assert active and all(int(value['step']) == steps for value in active)
        seed_everything(11)
        initial = TTISMIL(bundle, profile)
        before = initial.network.state_dict()
        changed = [name for name, value in initial.network.named_parameters()
                   if not torch.equal(state['model']['network.'+name], before[name])]
        assert len(changed) >= 4
        if 'clam' in profile:
            instance_updated = any(name.startswith('instance_classifiers.') for name in changed)
            assert instance_updated == (profile == 'ttis_clam_full_loss')
        history = read_json(run/'history.json')
        selected = min(range(len(history)), key=lambda e: history[e]['validation']['ttis_val_loss'])
        assert torch.load(run/'best.pt', map_location='cpu', weights_only=False)['epoch'] == selected
        metrics = evaluate(run, device=args.device); sha = digest_file(run/'test_predictions.csv')
        assert metrics == evaluate(run, device=args.device) and sha == digest_file(run/'test_predictions.csv')
        manifest = read_json(run/'test_instances/manifest.json')
        assert len(manifest['records']) == len(bundle.records('test'))
        assert all(len(row['selected_per_view']) == 2+len(bundle.ttis.semantic['prompts']) for row in manifest['records'])
        report['profiles'].append({'id': profile, 'state': 'passed', 'input_dimension': 1024,
            'hidden': 512, 'attention_dim': 384, 'epochs': args.epochs, 'optimizer_updates': steps,
            'updated_parameter_tensors': len(changed), 'selected_epoch': selected,
            'metrics': metrics, 'test_predictions_sha256': sha})
        write_json(output/'validation.json', report)
        print(profile, 'passed', steps, 'updates', flush=True)
    if software_identity() != identity:
        raise ValueError('Software changed during native TTIS validation')
    report['state'] = 'passed'; write_json(output/'validation.json', report)
    print('All seven native TTIS routes passed')


if __name__ == '__main__':
    main()

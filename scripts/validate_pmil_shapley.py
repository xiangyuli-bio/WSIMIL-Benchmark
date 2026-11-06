"""Validate all six native PMIL source/repaired routes with actual EM round restarts."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, seed_everything, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.pmil_shapley import PROFILES, PMILShapley, tensor_tree_sha
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    output = Path(args.output).resolve();output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    report = {'state': 'running', 'software': identity, 'profiles': [], 'device': args.device,
        'scope': 'Synthetic1024-dimensional features, original512hidden and128/256attention, full native objectives/Adam/round reset/Shapley/pseudo-bag loop. Explicit2rounds*2epochs,max2pseudo budget. No natural WSI or paper-number claim.'}
    if args.device == 'cuda' and not torch.cuda.is_available():
        report.update(state='not_run', reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json', report)
        print(report['reason']);return
    data = create_demo(output/'data', patients=30, classes=3, dimension=1024)
    bundle = Bundle(data)
    budget = {'rounds': 2, 'round_epochs': 2, 'max_pseudo': 2, 'pseudo_step': 1, 'search_rate': 2, 'sample_rate': 2}
    for profile in PROFILES:
        run = train(data, profile, output/profile, seed=11, device=args.device, config={'threads': 1, 'model': budget})
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        network, extra = state['model'], state['model']['_extra_state']
        assert extra['training_done'] and extra['round_id'] == 1 and extra['reset_count'] == 2
        assert extra['optimizer_steps'] == len(units(bundle, 'train'))*6
        assert len(extra['completed_rounds']) == 2 and extra['events'][-1]['estimator'] == 'shap'
        assert extra['events'][-1]['partition_sha256'] == tensor_tree_sha(extra['partitions'])
        seed_everything(11)
        initial = PMILShapley(bundle, profile, **budget).to(args.device)
        initial.configure_training(read_json(run/'request.json')['config']);initial.prepare(bundle, 11, args.device)
        projection = 'network.feature.0.weight' if initial.head == 'abmil' else 'network.attention_net.0.weight'
        assert network[projection].shape == (512, 1024)
        assert not torch.equal(network[projection], initial.state_dict()[projection].cpu())
        optimizer = state['optimizers'][0]
        names = list(dict(initial.named_parameters()))
        ids = optimizer['param_groups'][0]['params']
        assert len(names) == len(ids)
        counts = Counter(r['target'] for r in bundle.records('train'))
        for name, key in zip(names, ids):
            steps = int(optimizer['state'][key]['step'])
            expected = counts[int(name.split('.')[2])]*4 if name.startswith('network.instance_classifiers.') else len(units(bundle, 'train'))*4
            assert steps == expected
        del initial
        best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        selected = torch.load(run/'rounds/round_1.pt', map_location='cpu', weights_only=False)
        assert best['model']['_extra_state']['round_id'] == 1
        assert tensor_tree_sha({k.removeprefix('network.'): v for k, v in best['model'].items() if k.startswith('network.')}) == tensor_tree_sha(selected['network'])
        metrics = evaluate(run, device=args.device);sha = digest_file(run/'test_predictions.csv')
        assert metrics == evaluate(run, device=args.device) and sha == digest_file(run/'test_predictions.csv')
        report['profiles'].append({'id': profile, 'state': 'passed', 'updates': extra['optimizer_steps'],
            'metrics': metrics, 'test_predictions_sha256': sha, 'partition_sha256': extra['events'][-1]['partition_sha256']})
        write_json(output/'validation.json', report)
    if software_identity() != identity:
        raise ValueError('Software changed during native PMIL validation')
    report['state'] = 'passed';write_json(output/'validation.json', report)
    print('All six native PMIL full two-round training/reload routes passed')


if __name__ == '__main__':
    main()

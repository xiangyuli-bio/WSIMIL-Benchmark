"""Native-width full-budget validation of all original/repaired 2DMamba task routes."""
import argparse
from pathlib import Path

import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.mamba2d_demo import create_mamba2d_demo
from milplus.mamba2d_source import VARIANTS
from milplus.provenance import software_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--backend', choices=['torch', 'cuda'], default='torch')
    p.add_argument('--native-cuda-binary')
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()
    if args.epochs < 2 or args.backend == 'cuda' and args.device != 'cuda':
        raise ValueError('Native verification requires >=2 epochs and CUDA device for CUDA backend')
    output = Path(args.output).resolve();output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    report = {'state': 'running', 'software': identity, 'profiles': [], 'device': args.device, 'backend': args.backend,
        'scope': 'Synthetic real-coordinate sparse physical 6x4 grids, native1024 input/128 hidden/state16, default20epochs. Full AdamW/CE/cosine or source NLL(alpha0)/32 per-patient updates, task-specific validation, native pad attention and repeated prediction. Explicit source versus derivative versus primary-equation scan routes; not natural WSI or paper-number reproduction.'}
    if args.device == 'cuda' and not torch.cuda.is_available():
        report.update(state='not_run', reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json', report);print(report['reason']);return
    selected = ['mamba2dmil_source'] if args.backend == 'cuda' else list(VARIANTS)
    for survival in [False, True]:
        task = 'survival' if survival else 'classification'
        data = create_mamba2d_demo(output/('data_'+task), dimension=1024, survival=survival)
        b = Bundle(data)
        for profile in selected:
            options = {'backend': args.backend}
            if args.native_cuda_binary:options['cuda_binary'] = args.native_cuda_binary
            run = train(data, profile, output/(profile+'_'+task), device=args.device, seed=11,
                config={'epochs': args.epochs, 'threads': 1, 'model': options})
            last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
            steps = len(units(b, 'train'))*args.epochs
            extra = last['model']['_extra_state']
            assert extra['optimizer_steps'] == steps and extra['completed_epochs'] == args.epochs
            assert read_json(run/'status.json')['completed_epochs'] == args.epochs
            assert last['model']['network._fc1.0.weight'].shape == (128, 1024)
            assert all(int(value['step']) == steps for value in last['optimizers'][0]['state'].values())
            if survival:
                assert extra['scheduler'] is None and last['model']['prepared']
                assert last['model']['bin_edges'].shape == (3,)
            else:
                assert extra['scheduler']['last_epoch'] == args.epochs
                assert extra['scheduler']['T_max'] == args.epochs
            history = read_json(run/'history.json')
            metric = 'c_index' if survival else 'mamba2d_bag_auc'
            values = [epoch['validation'][metric] for epoch in history]
            expected = values.index(max(values)) if survival else max(i for i, v in enumerate(values) if v == max(values))
            assert torch.load(run/'best.pt', map_location='cpu', weights_only=False)['epoch'] == expected
            result = evaluate(run, device=args.device);sha = digest_file(run/'test_predictions.csv')
            assert result == evaluate(run, device=args.device) and sha == digest_file(run/'test_predictions.csv')
            manifest = read_json(run/'test_instances/manifest.json')
            assert len(manifest['records']) == len(b.records('test'))
            assert all(row['grid_cells'] == 24 and 0 < row['pad_attention_mass'] < 1 for row in manifest['records'])
            report['profiles'].append({'id': profile, 'task': task, 'state': 'passed', 'input_dimension': 1024,
                'hidden': 128, 'state_size': 16, 'epochs': args.epochs, 'optimizer_updates': steps,
                'selected_epoch': expected, 'metrics': result, 'test_predictions_sha256': sha})
            write_json(output/'validation.json', report);print(profile, task, steps, 'updates passed', flush=True)
    if software_identity() != identity:
        raise ValueError('Software changed during native 2DMamba validation')
    report['state'] = 'passed';write_json(output/'validation.json', report)
    print('All requested native 2DMamba task routes passed')


if __name__ == '__main__':
    main()

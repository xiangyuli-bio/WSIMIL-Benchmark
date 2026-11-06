"""Validate every MAMMOTH head at native dimensions, with an explicit CPU/CUDA budget."""
import argparse
from pathlib import Path

import torch

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, seed_everything, train, units
from milplus.io import digest_file, write_json
from milplus.mammoth import HEADS, MammothMIL
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Native software-validation epochs must be positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = software_identity()
    result = {'state': 'running', 'software': identity, 'device': args.device, 'profiles': [],
        'scope': 'Synthetic frozen1024-dimensional FM bags, native30experts/16heads/10slots, all eight complete classification recipes. Explicit software epoch budget; not natural WSI or paper-score reproduction.'}
    if args.device == 'cuda' and not torch.cuda.is_available():
        result.update(state='not_run', reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json', result)
        print(result['reason'])
        return
    data = create_demo(output/'data', patients=30, classes=3, dimension=1024)
    for name in HEADS:
        run = train(data, name, output/name, device=args.device, seed=17,
            config={'epochs': args.epochs, 'threads': 1, 'model': {'precision': 'amp' if args.device == 'cuda' else 'fp32'}})
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        model = state['model']
        assert model['network.patch_embed.slot_embeds'].shape == (30, 16, 10, (8 if 'ilra' in name else 16))
        assert model['_extra_state']['optimizer_steps'] > 0
        assert model['_extra_state']['scheduler']['last_epoch'] == args.epochs
        assert state['optimizers'][0]['param_groups'][0]['lr'] == 0
        seed_everything(17)
        initial = MammothMIL(Bundle(data), name)
        if args.device == 'cpu':
            expected_steps = args.epochs*len(units(Bundle(data), 'train'))
            assert model['_extra_state']['optimizer_steps'] == expected_steps
            assert model['_extra_state']['skipped_updates'] == 0
            optimizer = state['optimizers'][0]
            names = list(dict(initial.named_parameters()))
            ids = optimizer['param_groups'][0]['params']
            assert len(ids) == len(names)
            conditional = {}
            for parameter_id, parameter_name in zip(ids, names):
                if parameter_id not in optimizer['state']:
                    continue  # The author's DSMIL auxiliary fcc is not used by its forward.
                steps = int(optimizer['state'][parameter_id]['step'])
                if parameter_name.startswith('network.instance_classifiers.'):
                    assert 0 < steps <= expected_steps
                    conditional[parameter_name] = steps
                else:
                    assert steps == expected_steps
            if name == 'mammoth_clam':
                assert len(conditional) == 6
                assert sum(v for k, v in conditional.items() if k.endswith('.weight')) == expected_steps
                for k, v in conditional.items():
                    if k.endswith('.weight'):
                        assert conditional[k.removesuffix('.weight')+'.bias'] == v
        assert not torch.equal(initial.network.patch_embed.wq.weight, model['network.patch_embed.wq.weight'])
        del initial
        first = evaluate(run, device=args.device)
        sha = digest_file(run/'test_predictions.csv')
        if first != evaluate(run, device=args.device) or sha != digest_file(run/'test_predictions.csv'):
            raise ValueError('Native MAMMOTH repeated predictions differ')
        result['profiles'].append({'id': name, 'state': 'passed', 'metrics': first,
            'test_predictions_sha256': sha, 'optimizer_steps': model['_extra_state']['optimizer_steps']})
        write_json(output/'validation.json', result)
    if software_identity() != identity:
        raise ValueError('Software changed while native MAMMOTH validation was running')
    result['state'] = 'passed'
    write_json(output/'validation.json', result)
    print('Passed all eight native MAMMOTH training/reload/repeated-prediction routes')


if __name__ == '__main__':
    main()

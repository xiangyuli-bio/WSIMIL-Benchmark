"""Full 30/15/2 default recipe, actual subprocess recovery, exact all-state checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import h5py
import numpy as np
import torch
import milplus
from milplus.hybrid_supervision_checkpoint import load
from milplus.io import digest_file
from milplus.provenance import software_identity


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--import-root', required=True)
    p.add_argument('--source-sha256', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(a.import_root).resolve()
    assert Path(milplus.__file__).resolve().is_relative_to(root)
    assert software_identity()['source_sha256'] == a.source_sha256
    assert torch.cuda.is_available()
    state = {'state': 'running_full_default_recipe', 'pid': os.getpid(),
             'software': software_identity(), 'full_native_budget': {'segmentation_epochs': 30,
             'classification_epochs': 15, 'em_rounds': 2}, 'patch_pixels': 512,
             'widths_reduced': False, 'synthetic': True, 'paper_score_reproduction': False,
             'reference_reconstruction': True, 'formal_admission': False,
             'actual_cli_invocations': []}

    def record():
        (out/'validation.json').write_text(json.dumps(state, indent=2)+'\n')

    def cli(label, *args):
        env = os.environ.copy()
        # Source tests and working-directory packages must not shadow the wheel.
        env['PYTHONPATH'] = str(root)+':'+env.get('MILPLUS_NATIVE_RUNTIME_OVERLAYS', '')
        code = ('import pathlib,runpy,milplus;from milplus.provenance import software_identity;'
                'assert pathlib.Path(milplus.__file__).resolve().is_relative_to(pathlib.Path('+repr(str(root))+'));'
                'assert software_identity()["source_sha256"]=='+repr(a.source_sha256)+';'
                'runpy.run_module("milplus",run_name="__main__")')
        with (out/(label+'.log')).open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-c', code, *map(str, args)],
                                    cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['actual_cli_invocations'].append({'label': label, 'pid': proc.pid})
            record()
            rc = proc.wait()
        state['actual_cli_invocations'][-1]['returncode'] = rc
        record()
        assert rc == 0, label+': '+(out/(label+'.log')).read_text()[-5000:]

    record()
    try:
        data = out/'data'
        cli('demo', 'demo-hybrid-supervision', '--output', data, '--weights', a.weights)
        cli('check', 'check', '--data', data, '--method', 'Hybrid Supervision Learning')
        baseline, resumed = out/'baseline', out/'resumed'
        def run_args(destination):
            # No short-budget config is supplied: the actual profile defaults run.
            return ['run', '--data', data, '--method', 'Hybrid Supervision Learning',
                    '--output', destination, '--seed', 102, '--device', 'cuda:0']
        cli('continuous', *run_args(baseline))
        assert json.loads((baseline/'status.json').read_text())['completed_rounds'] == 3
        state['continuous_full_default_recipe'] = 'passed'
        record()
        expected = [('round_0', 'segmentation', 30), ('round_0', 'classification', 15),
                    ('round_1', 'segmentation', 30)]
        for i, (round_name, phase, updates) in enumerate(expected):
            args = run_args(resumed)+(['--resume'] if i else [])+['--stop-after', updates]
            cli('interrupted_'+str(i), *args)
            assert json.loads((resumed/'status.json').read_text())['state'] == 'interrupted'
            value = load(resumed/round_name/phase/'last.pt')
            assert value['updates'] == updates and value['finished']
            state['actual_phase_interruptions'] = i+1
            record()
        cli('complete_resume', *run_args(resumed), '--resume')
        assert json.loads((resumed/'status.json').read_text())['completed_rounds'] == 3
        for number in range(3):
            for phase, updates in [('segmentation', 30), ('classification', 15)]:
                left = load(baseline/('round_'+str(number))/phase/'last.pt')
                right = load(resumed/('round_'+str(number))/phase/'last.pt')
                for key in ['model', 'gradients', 'optimizer', 'rng', 'history', 'updates', 'finished']:
                    equal(left[key], right[key])
                assert left['updates'] == updates and left['finished']
            ma, mb = [json.loads((v/('round_'+str(number))/'scored.json').read_text()) for v in [baseline, resumed]]
            equal({k:v for k,v in ma.items() if k!='path'}, {k:v for k,v in mb.items() if k!='path'})
            with h5py.File(ma['path'], 'r') as x, h5py.File(mb['path'], 'r') as y:
                assert set(x.keys()) == set(y.keys())
                for sid in x:
                    for key in ['probability', 'maximum']:
                        equal(x[sid][key][:], y[sid][key][:])
                # Two actual TRAIN patients, sixteen actual512RGB patches each.
                assert len(x) == 2 and all(x[sid]['probability'].shape == (16,512,512) for sid in x)
            if number:
                assert ma['pools']['positive'] and ma['pools']['negative']
        left, right = [json.loads((v/'selected.json').read_text()) for v in [baseline, resumed]]
        equal({k:left[k] for k in ['round','training_loss']}, {k:right[k] for k in ['round','training_loss']})
        history = json.loads((resumed/'history.json').read_text())
        assert len(history) == 3 and all(v['segmentation_updates']==30 and v['classification_updates']==15 for v in history)
        assert left['training_loss'] == min(v['TRAIN_classification_loss'] for v in history)
        protocol = json.loads((resumed/'method_protocol.json').read_text())
        assert protocol['config'] == {'segmentation_epochs':30,'classification_epochs':15,'em_rounds':2,'threads':2}
        assert protocol['reference_choices']['mixed_batch_counts'] == [2,1,7]
        assert not (resumed/'test_predictions.csv').exists()
        cli('selected_reload_1', 'evaluate', '--run', resumed, '--device', 'cuda:0')
        first = digest_file(resumed/'test_predictions.csv')
        cli('selected_reload_2', 'evaluate', '--run', resumed, '--device', 'cuda:0')
        assert first == digest_file(resumed/'test_predictions.csv')
        status = json.loads((resumed/'status.json').read_text())
        assert status['test_evaluated'] and status['test_prediction_sha256'] == first
        state.update(state='passed_full_default_30_15_2_native',
            all6_phase_models_gradients_Adam_RNG_history_exact=True,
            actual_three_distinct_phase_interruptions=True,
            all_TRAIN_pseudo_probabilities_exact=True,
            actual_top16_and_mixed_batch_2_1_7=True,
            full_round_loss_selection_exact=True,
            fresh_selected_CLI_reloads=2, selected_prediction_sha256=first,
            full_history=history, source_package_sha256=a.source_sha256)
        record()
    except BaseException:
        state.update(state='failed', traceback=traceback.format_exc())
        record()
        raise

if __name__ == '__main__':
    main()

"""Actual public CLI: original512RGB/fine labels, full models and phase recovery."""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from milplus.hybrid_supervision_checkpoint import load
from milplus.io import digest_file
from test_hybrid_supervision import WEIGHTS
from test_pmil_prototypical_components import equal


def cli(*args, okay=True):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'milplus', *map(str, args)], cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root/'src')+':'+os.environ.get('PYTHONPATH', ''),
             'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}, capture_output=True, text=True)
    if okay:
        assert result.returncode == 0, result.stdout[-1500:]+result.stderr[-5000:]
    else:
        assert result.returncode != 0
    return result


def test_complete_public_CLI_two_rounds_three_actual_phase_interruptions_all_state_and_two_fresh_reloads(tmp_path):
    data = tmp_path/'data'
    cli('demo-hybrid-supervision', '--output', data, '--weights', WEIGHTS)
    cli('check', '--data', data, '--method', 'Hybrid Supervision Learning')
    cfg = tmp_path/'cfg.json'
    cfg.write_text(json.dumps({'segmentation_epochs': 1, 'classification_epochs': 1, 'em_rounds': 1, 'threads': 2}))
    baseline, resumed = tmp_path/'baseline', tmp_path/'resumed'
    device = os.environ.get('MILPLUS_HYBRID_CLI_DEVICE', 'cpu')
    cli('run', '--data', data, '--method', 'Hybrid Supervision Learning', '--output', baseline,
        '--config', cfg, '--seed', 102, '--device', device)
    for number in range(4):
        args = ['run', '--data', data, '--method', 'Hybrid Supervision Learning', '--output', resumed,
                '--config', cfg, '--seed', 102, '--device', device]
        if number:
            args += ['--resume']
        if number < 3:
            args += ['--stop-after', 1]
        cli(*args)
        assert json.loads((resumed/'status.json').read_text())['state'] == ('interrupted' if number < 3 else 'completed')
    for number in range(2):
        for phase in ['segmentation', 'classification']:
            left = load(baseline/('round_'+str(number))/phase/'last.pt')
            right = load(resumed/('round_'+str(number))/phase/'last.pt')
            for key in ['model', 'gradients', 'optimizer', 'rng', 'history', 'updates', 'finished']:
                equal(left[key], right[key])
            assert left['updates'] == 1
    assert not (resumed/'test_predictions.csv').exists()
    cli('evaluate', '--run', resumed, '--device', device)
    prediction = digest_file(resumed/'test_predictions.csv')
    cli('evaluate', '--run', resumed, '--device', device)
    assert prediction == digest_file(resumed/'test_predictions.csv')
    status = json.loads((resumed/'status.json').read_text())
    assert status['test_evaluated'] is True and status['test_prediction_sha256'] == prediction
    a, b = [json.loads((p/'selected.json').read_text()) for p in [baseline, resumed]]
    assert a['round'] == b['round'] and a['training_loss'] == b['training_loss']
    protocol = json.loads((resumed/'method_protocol.json').read_text())
    assert protocol['input_patch_pixels'] == 512 and protocol['source_challenge_RF_is_a_different_condition']
    assert protocol['reference_choices']['mixed_batch_counts'] == [2, 1, 7]
    assert protocol['formal_admission'] is False


def test_frozen_FM_features_cannot_substitute_raw_hybrid_supervision(tmp_path):
    data = tmp_path/'plain'
    cli('demo', '--output', data)
    output = tmp_path/'invalid'
    result = cli('run', '--data', data, '--method', 'Hybrid Supervision Learning', '--output', output, okay=False)
    assert not output.exists() and 'TRAINfine' in result.stderr

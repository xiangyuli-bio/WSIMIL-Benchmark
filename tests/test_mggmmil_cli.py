"""Public path+method, full native model, selected reload and exact interruption."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from milplus.mggmmil_demo import create_demo
from milplus.io import digest_file
from test_pmil_prototypical_components import equal


def cli(*args, okay=True):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'milplus', *map(str, args)], cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root/'src')+':'+os.environ.get('PYTHONPATH', ''),
             'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
        capture_output=True, text=True)
    if okay:
        assert result.returncode == 0, result.stdout[-2000:]+result.stderr[-5000:]
    else:
        assert result.returncode != 0
    return result


def test_public_CLI_full_native_model_epoch50_selection_two_real_resume_points_and_fresh_reload(tmp_path):
    data = create_demo(tmp_path/'data')
    cfg = tmp_path/'cfg.json'
    cfg.write_text(json.dumps({'epochs': 50, 'threads': 2}))
    baseline, resumed = tmp_path/'baseline', tmp_path/'resumed'
    cli('run', '--data', data, '--method', 'MGGMMIL', '--output', baseline, '--config', cfg, '--seed', 101)
    cli('run', '--data', data, '--method', 'MGGMMIL', '--output', resumed, '--config', cfg,
        '--seed', 101, '--stop-after', 1)
    assert json.loads((resumed/'status.json').read_text())['state'] == 'interrupted'
    assert not (resumed/'best.pt').exists()
    cli('run', '--data', data, '--method', 'MGGMMIL', '--output', resumed, '--config', cfg,
        '--seed', 101, '--resume', '--stop-after', 49)
    assert json.loads((resumed/'status.json').read_text())['state'] == 'interrupted'
    assert not (resumed/'best.pt').exists()
    cli('run', '--data', data, '--method', 'MGGMMIL', '--output', resumed, '--config', cfg,
        '--seed', 101, '--resume')
    for name in ['last.pt', 'best.pt']:
        left = torch.load(baseline/name, map_location='cpu', weights_only=False)
        right = torch.load(resumed/name, map_location='cpu', weights_only=False)
        equal(left, right)
        assert left['epoch'] == 49
        extra = left['model']['_extra_state']
        assert extra['completed_epochs'] == 50 and extra['optimizer_updates'] == 100
        assert extra['selection']['selected_epoch'] == 50
    assert not (resumed/'test_predictions.csv').exists()
    cli('evaluate', '--run', resumed)
    sha = digest_file(resumed/'test_predictions.csv')
    cli('evaluate', '--run', resumed)
    assert digest_file(resumed/'test_predictions.csv') == sha
    request = json.loads((resumed/'request.json').read_text())
    assert request['config']['model'] == {} and request['config']['lr'] == 3e-5
    protocol = json.loads((resumed/'method_protocol.json').read_text())
    assert protocol['completed_epochs'] == 50 and protocol['optimizer_updates'] == 100
    assert protocol['formal_admission'] is False


def test_public_CLI_missing_original_graphs_fails_before_creating_run(tmp_path):
    data = create_demo(tmp_path/'data')
    (data/'graphs.json').unlink()
    out = tmp_path/'run'
    cli('run', '--data', data, '--method', 'MGGMMIL', '--output', out, okay=False)
    assert not out.exists()

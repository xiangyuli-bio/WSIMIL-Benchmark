"""Real unified full-model training/reload/recovery and immutable input checks."""
import copy
from pathlib import Path

import h5py
import pytest
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.io import digest_file
from milplus.libramil_demo import create_libramil_demo
from milplus.registry import inspect_method

WEIGHTS = '/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth'


@pytest.fixture(scope='module')
def native_data(tmp_path_factory):
    torch.set_num_threads(2)
    return create_libramil_demo(tmp_path_factory.mktemp('libramil')/'data', WEIGHTS, patients=30, patches=257)


@pytest.mark.parametrize('method', ['libramil_source_p4', 'libramil_readme_p6'])
def test_full_unified_training_real_native_inputs_and_two_checkpoint_reloads(native_data, tmp_path, method):
    path = train(native_data, method, tmp_path/'run', seed=2026, device='cpu', config={'epochs':2, 'threads':1})
    state = torch.load(path/'last.pt', weights_only=False)
    extra = state['model']['_extra_state']
    assert extra['completed_epochs'] == 2
    assert extra['optimizer_updates'] == 2*len(Bundle(native_data).records('train'))
    assert state['optimizers'][0]['param_groups'][0]['weight_decay'] == .01
    assert extra['scheduler']['last_epoch'] == extra['optimizer_updates']
    assert not (path/'test_predictions.csv').exists()
    first = evaluate(path, device='cpu'); fingerprint = digest_file(path/'test_predictions.csv')
    assert evaluate(path, device='cpu') == first
    assert digest_file(path/'test_predictions.csv') == fingerprint


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def test_unified_interruption_resume_restores_all_source_weights_moments_scheduler_and_rng(native_data, tmp_path):
    cfg = {'epochs':2, 'threads':1}
    full = train(native_data, 'libramil_source_p4', tmp_path/'full', seed=2026, config=cfg)
    part = train(native_data, 'libramil_source_p4', tmp_path/'part', seed=2026, config=cfg, stop_after=1)
    train(native_data, 'libramil_source_p4', part, seed=2026, config=cfg, resume=True)
    a = torch.load(full/'last.pt', weights_only=False); b = torch.load(part/'last.pt', weights_only=False)
    for key in ['model', 'optimizers', 'history', 'best', 'bad']:
        equal(a[key], b[key])
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])


@pytest.mark.parametrize('mutation', ['features', 'labels', 'path', 'metadata', 'text'])
def test_bound_full_inputs_reject_mutation(native_data, mutation):
    bundle = Bundle(native_data); assets = bundle.libramil
    row = copy.deepcopy(bundle.records('train')[0])
    if mutation == 'labels':
        row['target'] = (row['target']+1)%3
        with pytest.raises(ValueError): assets.load(row)
    elif mutation == 'path':
        row['path'] = bundle.records('train')[1]['path']
        with pytest.raises(ValueError): assets.load(row)
    else:
        path = Path(assets.meta['text_path'] if mutation == 'text' else assets.meta['metadata_path'])
        if mutation == 'features': path = Path(assets.meta['records'][row['slide_id']]['path'])
        before = path.read_bytes()
        try:
            path.write_bytes(before+b'changed')
            with pytest.raises(ValueError): assets.load(row)
        finally:
            path.write_bytes(before)


def test_native_source_alias_is_registered():
    assert inspect_method('Libra-MIL')['id'] == 'libramil_source_p4'


def test_full_unified_loop_matches_author_train_VAL_TEST_rng_and_selected_state(native_data, tmp_path):
    from milplus.engine import seed_everything
    from milplus.libramil import LibraMIL
    from milplus.libramil_training import original_training, source_optimizer, SourceSelection
    from torch.utils.data import DataLoader
    actual = train(native_data, 'libramil_source_p4', tmp_path/'actual', seed=2026,
                   config={'epochs':2, 'threads':1})
    expected = torch.load(actual/'last.pt', weights_only=False)
    bundle = Bundle(native_data)
    seed_everything(2026); torch.set_num_threads(1)
    reference = LibraMIL(bundle)
    optimizer, scheduler = source_optimizer(reference.network, len(bundle.records('train')), epochs=2)
    source = original_training(); selection = SourceSelection()
    for epoch in range(2):
        source.train(reference.network, reference.loader('train')[1], torch.device('cpu'), epoch,
                     optimizer, scheduler, len(bundle.classes))
        metrics = source.evaluate(reference.network, reference.loader('val')[1], torch.device('cpu'), epoch, len(bundle.classes))
        # Run the actual author's unused TEST evaluation as the independent
        # synthetic RNG oracle. Unified training itself does not access TEST.
        source.evaluate(reference.network, reference.loader('test')[1], torch.device('cpu'), epoch, len(bundle.classes))
        selection.observe(epoch, float(metrics[0]), float(metrics[1]))
    for key, value in reference.network.state_dict().items():
        equal(value, expected['model']['network.'+key])
    equal(optimizer.state_dict(), expected['optimizers'][0])
    equal(scheduler.state_dict(), expected['model']['_extra_state']['scheduler'])
    equal(selection.state_dict(), expected['model']['_extra_state']['selection'])
    assert torch.equal(torch.get_rng_state(), expected['rng']['torch'])


def test_actual_cli_path_and_original_method_name(native_data, tmp_path):
    import os
    import subprocess
    import sys
    import json
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps({'epochs':1, 'threads':1}))
    result = subprocess.run([sys.executable, '-m', 'milplus', 'run', '--data', str(native_data),
                             '--method', 'Libra-MIL', '--output', str(tmp_path/'cli_run'),
                             '--config', str(cfg)], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    state = torch.load(tmp_path/'cli_run/last.pt', weights_only=False)
    assert state['model']['_extra_state']['completed_epochs'] == 1

"""Genuine epoch interruption, full state recovery, and independent reload."""
import copy

import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.sparseconvmil import SparseConvMIL
from milplus.sparseconvmil_demo import create_sparseconvmil_demo


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('profile,classes', [('sparseconvmil_raw_source_c32', 2), ('sparseconvmil_raw_source_c128', 3)])
def test_genuine_source_resume_final_weights_and_deterministic_reload(tmp_path, profile, classes):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=5*classes, classes=classes)
    cfg = {'epochs': 3, 'threads': 2}
    continuous = train(data, profile, tmp_path/'continuous', seed=37, config=cfg)
    resumed = train(data, profile, tmp_path/'resumed', seed=37, config=cfg, stop_after=1)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(data, profile, resumed, seed=37, config=cfg, resume=True)
    a = torch.load(continuous/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'history', 'epoch', 'best', 'bad']:
        equal(a[key], b[key])
    assert a['epoch'] == 2 and a['best'] == 3 and a['bad'] == 0
    best = torch.load(continuous/'best.pt', map_location='cpu', weights_only=False)
    equal(a['model'], best['model']); assert best['epoch'] == 2
    protocol = read_json(continuous/'method_protocol.json')
    assert protocol['backend']['backend'] == 'original_compiled_sparseconvnet'
    assert protocol['completed_epochs'] == 3 and protocol['tiles_per_slide'] == 5
    assert protocol['optimizer_updates'] == 3*((3*classes+1)//2)
    assert evaluate(continuous) == evaluate(continuous) == evaluate(resumed)
    bundle = Bundle(data); model = SparseConvMIL(bundle, profile)
    model.load_state_dict(a['model'])
    extra = copy.deepcopy(a['model']['_extra_state'])
    for corruption in ['updates', 'order', 'indices', 'contract', 'epochs']:
        state = copy.deepcopy(extra)
        if corruption == 'updates':
            state['optimizer_updates'] += 1
        elif corruption == 'order':
            state['train_orders'][0].reverse()
        elif corruption == 'indices':
            state['samples_history'][0][0]['indices'][0] = 999999
        elif corruption == 'contract':
            state['contract'] = 'changed_original_input'
        else:
            state['completed_epochs'] += 1
        with pytest.raises(ValueError):
            model.set_extra_state(state)


def test_generic_feature_cap_and_invented_validation_selection_are_refused(tmp_path):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10)
    cfg = {'epochs': 1, 'threads': 2}
    for number, invalid in enumerate([{'max_instances': 4}, {'grad_clip': 5.}, {'selection_metric': 'macro_auc'}]):
        with pytest.raises(ValueError, match='no generic feature cap'):
            train(data, 'SparseConvMIL', tmp_path/f'invalid_{number}', config={**cfg, **invalid})

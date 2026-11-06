"""Genuine resume, original scheduler/selection state, and full text identity."""
import copy

import numpy as np
import pytest
import torch

from milplus.conmil import Bags, ConMIL
from milplus.conmil_demo import create_conmil_demo
from milplus.data import Bundle
from milplus.engine import train, evaluate, rng_state, restore_rng
from milplus.io import read_json


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


@pytest.mark.parametrize('profile,classes', [('conmil_pos_source', 2), ('conmil_posneg_source', 2),
    ('conmil_posneg_multiclass_repaired', 3)])
def test_genuine_interruption_and_full_cosine_adam_resume(tmp_path, profile, classes):
    data = create_conmil_demo(tmp_path/'data', patients=5*classes, classes=classes)
    cfg = {'epochs': 3, 'threads': 2}
    continuous = train(data, profile, tmp_path/'continuous', seed=83, config=cfg)
    resumed = train(data, profile, tmp_path/'resumed', seed=83, config=cfg, stop_after=1)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(data, profile, resumed, seed=83, config=cfg, resume=True)
    a = torch.load(continuous/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'epoch', 'history', 'best', 'bad']:
        equal(a[key], b[key])
    assert a['epoch'] == 2 and a['model']['_extra_state']['scheduler']['last_epoch'] == 3
    assert a['optimizers'][0]['param_groups'][0]['lr'] == 0
    accuracy = [h['validation']['conmil_val_accuracy'] for h in a['history']]
    best = torch.load(continuous/'best.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == accuracy.index(max(accuracy))
    assert evaluate(continuous) == evaluate(continuous) == evaluate(resumed)
    bundle = Bundle(data); model = ConMIL(bundle, profile)
    model.load_state_dict(a['model'])
    for corrupt in ['updates', 'epoch', 'order', 'samples', 'scheduler', 'source', 'unexpected']:
        state = copy.deepcopy(a['model']['_extra_state'])
        if corrupt == 'updates':
            state['optimizer_updates'] += 1
        elif corrupt == 'epoch':
            state['completed_epochs'] += 1
        elif corrupt == 'order':
            state['train_orders'][0].reverse()
        elif corrupt == 'samples':
            state['samples_history'][0][0]['indices'][0] = 99999
        elif corrupt == 'scheduler':
            state['scheduler']['last_epoch'] -= 1
        elif corrupt == 'source':
            state['contract'] = 'different_native_text_or_source'
        else:
            state['unused_future_key'] = True
        with pytest.raises(ValueError):
            model.set_extra_state(state)
    state = copy.deepcopy(a['model']); state['positive_queries'][0, 0] += .1
    with pytest.raises(ValueError, match='fixed original'):
        model.load_state_dict(state)


def test_original_train_random_cap_and_complete_holdout_are_distinct(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10, patches=4097)
    bundle = Bundle(data); model = ConMIL(bundle)
    trace = []; dataset = Bags(model, bundle.records('train'), training=True, trace=trace)
    torch.manual_seed(11); saved = rng_state()
    value, _, _ = dataset[0]
    after = rng_state(); restore_rng(saved)
    indices = torch.randperm(4097)[:4096]
    expected = bundle.conmil.load(bundle.records('train')[0])[indices]
    assert torch.equal(value, expected) and trace[0]['indices'] == indices.tolist()
    equal(after, rng_state())
    val, _, _ = Bags(model, bundle.records('val'))[0]
    assert len(val) == 4097
    model.eval()
    frame = model.predict_subset(bundle, 'val', 'cpu', {})
    assert frame.conmil_patches.tolist() == [4097, 4097]


def test_source_zero_initial_best_and_repaired_first_zero_are_explicit(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    source = ConMIL(Bundle(data), 'conmil_posneg_source')
    repaired_data = create_conmil_demo(tmp_path/'repair', patients=15, classes=3)
    repair = ConMIL(Bundle(repaired_data), 'conmil_posneg_multiclass_repaired')
    assert not source.selection_improved(0., -float('inf'))
    assert repair.selection_improved(0., -float('inf'))
    assert not source.selection_improved(.5, .5) and not repair.selection_improved(.5, .5)


def test_generic_clipping_feature_cap_and_auc_selection_are_refused(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    for index, cfg in enumerate([{'grad_clip': 5.}, {'max_instances': 10}, {'selection_metric': 'macro_auc'}]):
        with pytest.raises(ValueError, match='TRAIN-only cap'):
            train(data, 'ConMIL', tmp_path/f'bad_{index}', config={'epochs': 1, **cfg})

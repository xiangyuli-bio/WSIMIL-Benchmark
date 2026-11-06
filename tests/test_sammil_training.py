"""Genuine interruption/recovery and independent prediction of full recipes."""
import copy

import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.engine import train, evaluate, seed_everything
from milplus.io import read_json
from milplus.sammil import SAMMIL
from milplus.sammil_demo import create_sammil_demo


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left: equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): equal(a, b)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    else:
        assert left == right


@pytest.mark.parametrize('profile,classes', [('sammil_attn_source_control', 2),
    ('sammil_attn_objective_repaired', 3)])
def test_native_full_recipe_exact_resume_all_adam_scheduler_teacher_gradients_and_rng(tmp_path, profile, classes):
    root = create_sammil_demo(tmp_path/'data', patients=10*classes, dimension=1024, classes=classes, patches=32)
    cfg = {'epochs': 3, 'threads': 2}
    full = train(root, profile, tmp_path/'full', config=cfg, seed=76)
    resumed = train(root, profile, tmp_path/'resumed', config=cfg, seed=76, stop_after=1)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(root, profile, resumed, config=cfg, seed=76, resume=True)
    a = torch.load(full/'last.pt', weights_only=False); b = torch.load(resumed/'last.pt', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'history', 'best', 'bad', 'epoch']:
        equal(a[key], b[key])
    assert read_json(full/'method_protocol.json')['optimizer_updates'] > 0
    assert len(a['optimizers'][0]['state']) == len([k for k in a['model'] if k.startswith('student.')])
    first = evaluate(resumed); probability = (resumed/'test_predictions.csv').read_bytes()
    assert evaluate(resumed) == first
    assert (resumed/'test_predictions.csv').read_bytes() == probability
    assert read_json(resumed/'request.json')['synthetic'] is True
    # Exact teacher gradient state is required for the source control although
    # those source gradients never update its teacher. Repair redirects them.
    assert bool(a['model']['_extra_state']['teacher_gradients']) == profile.endswith('control')
    bundle = Bundle(root); seed_everything(76)
    model = SAMMIL(bundle, profile)
    config = read_json(full/'request.json')['config']; model.configure_training(config)
    optimizer = model.make_optimizer(config['lr'], config['weight_decay'])
    model.load_state_dict(a['model']); optimizer.load_state_dict(a['optimizers'][0]); model.validate_optimizer_state()
    for corruption in ['updates', 'order', 'cosine', 'history', 'recipe', 'teacher']:
        bad = copy.deepcopy(a['model'])
        if corruption == 'updates': bad['_extra_state']['optimizer_updates'] -= 1
        if corruption == 'order': bad['_extra_state']['train_orders'][0] = []
        if corruption == 'cosine': bad['_extra_state']['scheduler']['last_epoch'] += 1
        if corruption == 'history': bad['_extra_state']['lr_history'][0] *= .5
        if corruption == 'recipe': bad['_extra_state']['recipe']['epochs'] += 1
        if corruption == 'teacher':
            key = next(key for key in bad if key.startswith('teacher.') and isinstance(bad[key], torch.Tensor))
            bad[key].add_(1.)
        with pytest.raises(ValueError): model.load_state_dict(bad)


def test_source_tie_stopping_and_first_optimum_are_separate(tmp_path):
    root = create_sammil_demo(tmp_path/'data', patients=18, dimension=1024, classes=2, patches=16)
    model = SAMMIL(Bundle(root), 'sammil_attn_source_control')
    cfg = dict(epochs=200, lr=.0002, weight_decay=1e-5, patience=20,
        selection_metric='sammil_val_auc', max_instances=None, grad_clip=None)
    model.configure_training(cfg)
    for epoch in range(71):
        model.epoch_index = epoch
        model.after_validation({'sammil_val_auc': .8 if epoch == 0 else .7})
    assert model.training_done is False
    model.epoch_index = 71; model.after_validation({'sammil_val_auc': .7})
    assert model.training_done is True
    model.epoch_index = 72; model.after_validation({'sammil_val_auc': .8})
    assert model.training_done is False and model.early_bad == 0
    assert model.early_best_auc == .8


@pytest.mark.parametrize('option,value', [('max_instances', 16), ('grad_clip', 1.),
    ('selection_metric', 'test_auc')])
def test_incomplete_or_test_guided_recipe_refused(tmp_path, option, value):
    root = create_sammil_demo(tmp_path/'data', patients=18, dimension=1024, classes=2, patches=16)
    with pytest.raises(ValueError):
        train(root, 'SAM-MIL', tmp_path/'run', config={'epochs': 1, option: value})

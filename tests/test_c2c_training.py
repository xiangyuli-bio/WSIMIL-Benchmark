"""Real source-stage interruption/recovery, selected reload and audit-state guards."""
import copy

import numpy as np
import pytest
import torch

from milplus.c2c import C2C
from milplus.c2c_demo import create_c2c_demo
from milplus.data import Bundle
from milplus.engine import train, evaluate, seed_everything
from milplus.io import read_json


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b): equal(left, right)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize('profile,classes', [('c2c_cluster_source', 2),
    ('c2c_cluster_default_bn_source', 2), ('c2c_cluster_multiclass_repaired', 3)])
def test_genuine_resume_all_network_adam_clustering_sampling_bn_and_rng(tmp_path, profile, classes):
    root = create_c2c_demo(tmp_path/'data', patients=5*classes, classes=classes, patches=64, patch_pixels=64)
    cfg = {'epochs': 3, 'threads': 2}
    full = train(root, profile, tmp_path/'full', seed=12, config=cfg)
    resumed = train(root, profile, tmp_path/'resumed', seed=12, config=cfg, stop_after=1)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(root, profile, resumed, seed=12, config=cfg, resume=True)
    a = torch.load(full/'last.pt', weights_only=False); b = torch.load(resumed/'last.pt', weights_only=False)
    for key in ['model', 'optimizers', 'history', 'best', 'bad', 'epoch', 'rng']:
        equal(a[key], b[key])
    assert read_json(full/'request.json')['synthetic'] is True
    extra = a['model']['_extra_state']; n = len(Bundle(root).records('train'))
    assert extra['optimizer_updates'] == 3*n and len(extra['cluster_history']) == 3
    assert all(row['train']['kl_active_groups'] >= 1 for row in a['history'])
    accuracies = [row['validation']['c2c_val_accuracy'] for row in a['history']]
    best = torch.load(full/'best.pt', weights_only=False)
    assert best['epoch'] == max(i for i, value in enumerate(accuracies) if value == max(accuracies))
    del best
    first = evaluate(resumed); predictions = (resumed/'test_predictions.csv').read_bytes()
    assert evaluate(resumed) == first and (resumed/'test_predictions.csv').read_bytes() == predictions
    bundle = Bundle(root); seed_everything(12); model = C2C(bundle, profile)
    config = read_json(full/'request.json')['config']; model.configure_training(config)
    optimizer = model.make_optimizer(config['lr'], config['weight_decay'])
    model.load_state_dict(a['model']); optimizer.load_state_dict(a['optimizers'][0]); model.validate_optimizer_state()
    for corruption in ['updates', 'order', 'clusters_missing', 'centers', 'labels', 'samples', 'weights']:
        bad = copy.deepcopy(a['model']); state = bad['_extra_state']; slide = next(iter(state['clusters']))
        if corruption == 'updates': state['optimizer_updates'] -= 1
        if corruption == 'order': state['train_orders'][0] = []
        if corruption == 'clusters_missing': state['clusters'].pop(slide)
        if corruption == 'centers': state['clusters'][slide]['centers'][0, 0] += .1
        if corruption == 'labels': state['clusters'][slide]['labels'][0] = 100
        if corruption == 'samples': state['samples_history'][-1][slide] = []
        if corruption == 'weights': state['recipe']['lr'] *= 2
        with pytest.raises(ValueError): model.load_state_dict(bad)


@pytest.mark.parametrize('option,value', [('max_instances', 16), ('grad_clip', .1),
    ('selection_metric', 'macro_auc')])
def test_reduced_or_changed_recipe_is_not_accepted_as_complete_source(tmp_path, option, value):
    root = create_c2c_demo(tmp_path/'data', patients=10, patches=8, patch_pixels=64)
    with pytest.raises(ValueError):
        train(root, 'C2C', tmp_path/'run', config={'epochs': 1, option: value})

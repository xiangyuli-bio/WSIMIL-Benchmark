"""Genuine end-to-end raw-image continuation and independent selected reload."""
import copy

import numpy as np
import pytest
import torch

from milplus.center_embeddings import CenterEmbeddingMIL
from milplus.center_embeddings_demo import create_center_embeddings_demo
from milplus.data import Bundle
from milplus.engine import evaluate, train
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
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('profile', ['center_embeddings_bag_classifier_repaired', 'center_embeddings_soft_assignment_repaired'])
def test_genuine_raw32x256_resume_retains_bn_dual_adam_and_exhaustive_selection(tmp_path, profile):
    root = create_center_embeddings_demo(tmp_path/'data')
    config = {'epochs': 2, 'threads': 1, 'model': {'train_batch_size': 1}}
    a = train(root, profile, tmp_path/'continuous', seed=81, config=config)
    b = train(root, profile, tmp_path/'resume', seed=81, config=config, stop_after=1)
    assert read_json(b/'status.json')['state'] == 'interrupted'
    train(root, profile, b, seed=81, config=config, resume=True)
    sa = torch.load(a/'last.pt', map_location='cpu', weights_only=False)
    sb = torch.load(b/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'epoch', 'history', 'best', 'bad']:
        equal(sa[key], sb[key])
    assert sa['epoch'] == 1 and len(sa['optimizers']) == 2
    extra = sa['model']['_extra_state']
    assert extra['optimizer_updates'] == 4
    assert all(set(trace['scanned'].values()) == {33} for trace in extra['train_selections'])
    best = torch.load(a/'best.pt', map_location='cpu', weights_only=False)
    accuracy = [h['validation']['center_embeddings_val_balanced_accuracy'] for h in sa['history']]
    assert best['epoch'] == max(i for i, value in enumerate(accuracy) if value == max(accuracy))
    assert evaluate(a) == evaluate(a) == evaluate(b)
    model = CenterEmbeddingMIL(Bundle(root), profile, train_batch_size=1)
    model.load_state_dict(sa['model'])
    for corrupt in ['updates', 'selection', 'order', 'scanned', 'source', 'unknown']:
        state = copy.deepcopy(extra)
        if corrupt == 'updates':
            state['optimizer_updates'] += 1
        elif corrupt == 'selection':
            values = next(iter(state['train_selections'][0]['selection'].values())); values[0] = 999999
        elif corrupt == 'order':
            state['train_selections'][0]['bag_order'][0] = 'heldout_slide'
        elif corrupt == 'scanned':
            key = next(iter(state['train_selections'][0]['scanned'])); state['train_selections'][0]['scanned'][key] -= 1
        elif corrupt == 'source':
            state['contract'] = 'changed_source_or_inputs'
        else:
            state['unrecognized'] = True
        with pytest.raises(ValueError):
            model.set_extra_state(state)


def test_generic_feature_cap_clipping_and_auc_selection_are_refused(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data')
    for index, cfg in enumerate([{'max_instances': 32}, {'grad_clip': 1.}, {'selection_metric': 'macro_auc'}]):
        with pytest.raises(ValueError):
            train(root, 'Center Embeddings', tmp_path/f'bad_{index}', config={'epochs': 1, **cfg})

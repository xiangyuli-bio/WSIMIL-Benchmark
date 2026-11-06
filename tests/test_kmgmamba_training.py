import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, train
from milplus.kmgmamba import KMGMamba
from milplus.kmgmamba_demo import create_kmgmamba_demo
from milplus.upstream import load_definitions, source_root


@pytest.fixture(autouse=True)
def runtime():
    if not source_root('kmgmamba').is_dir():
        pytest.skip('Fetch the pinned KMG-Mamba source')
    torch.set_num_threads(1)


@pytest.fixture
def bundle(tmp_path):
    return Bundle(create_kmgmamba_demo(tmp_path/'data', patients=20, dimension=1024, classes=2))


def configuration(epochs=3):
    return {'epochs': epochs, 'patience': 20, 'lr': .0002, 'weight_decay': 1e-5,
            'max_instances': None, 'grad_clip': None, 'selection_metric': 'kmgmamba_val_auc'}


def identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            identical(left, right)
    else:
        assert a == b


def test_actual_author_trainer_and_loader_adam_weights_and_rng_match(bundle):
    model = KMGMamba(bundle, num_workers=0); model.configure_training(configuration())
    class AuthorReturn(nn.Module):
        def __init__(self, network):
            super().__init__(); self.network = network
        def forward(self, *inputs):
            return self.network(*inputs)[:3]
    original = AuthorReturn(copy.deepcopy(model.network))
    actual_opt = model.make_optimizer(.0002, 1e-5)
    original_opt = torch.optim.Adam(original.parameters(), lr=.0002, weight_decay=1e-5)
    helpers = load_definitions('kmgmamba', 'utils/utils.py',
        ['collate_tranformer', 'get_split_loader', 'calculate_error'],
        {'torch': torch, 'np': np, 'DataLoader': DataLoader, 'RandomSampler': RandomSampler,
         'SequentialSampler': SequentialSampler, 'device': torch.device('cpu')})
    author = load_definitions('kmgmamba', 'utils/core_utils.py', ['Accuracy_Logger', 'train_loop'],
        {'torch': torch, 'np': np, 'nn': nn, 'calculate_error': helpers.calculate_error})
    rows = bundle.records('train')
    class AuthorDataset(Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            row = rows[index]; low, _, high, _ = bundle.kmgmamba.load(row)
            return low, high, row['target']
    loader = helpers.get_split_loader(AuthorDataset(), training=True, mode='transformer')
    for epoch in range(2):
        initial = rng_state(); model.start_epoch(epoch)
        losses = model.epoch(bundle, actual_opt, 'cpu'); actual_rng = rng_state()
        restore_rng(initial)
        author.train_loop(SimpleNamespace(), epoch, original, loader, original_opt, 2)
        identical(model.network.state_dict(), original.network.state_dict())
        identical(actual_opt.state_dict(), original_opt.state_dict())
        assert torch.equal(actual_rng['torch'], rng_state()['torch'])
        assert len(losses) == len(rows)
        assert all(abs(row['loss']-.5*(row['bag_ce']+row['coarse_prototype_svm']+row['fine_prototype_svm'])) < 5e-7 for row in losses)
    assert model.optimizer_updates == 2*len(rows)


@pytest.mark.parametrize('stop_after', [1, 2])
def test_full_resume_model_all_adam_selection_history_rng_and_double_reload(bundle, tmp_path, stop_after):
    cfg = {'epochs': 3, 'threads': 1, 'model': {'num_workers': 0}}
    full = train(bundle.root, 'KMG-Mamba', tmp_path/'full', seed=2026, config=cfg)
    partial = train(bundle.root, 'KMG-Mamba', tmp_path/'partial', seed=2026, config=cfg, stop_after=stop_after)
    assert not (partial/'test_predictions.csv').exists()
    train(bundle.root, 'KMG-Mamba', partial, seed=2026, config=cfg, resume=True)
    a, b = [torch.load(path/'last.pt', weights_only=False) for path in [full, partial]]
    identical(a['model'], b['model']); identical(a['optimizers'], b['optimizers'])
    assert a['history'] == b['history']
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    assert a['model']['_extra_state']['optimizer_updates'] == 3*len(bundle.records('train'))
    assert not (full/'test_predictions.csv').exists()
    evaluate(full); first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full); pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))
    for mutation in ['optimizer_updates', 'train_orders', 'best_auc', 'selection_history']:
        state = copy.deepcopy(a['model']['_extra_state'])
        if mutation == 'optimizer_updates':
            state[mutation] -= 1
        elif mutation == 'train_orders':
            state[mutation][0] = state[mutation][0][:-1]
        elif mutation == 'best_auc':
            state[mutation] = -.1
        else:
            state[mutation][-1]['bad'] += 1
        model = KMGMamba(bundle, num_workers=0); model.configure_training(configuration())
        with pytest.raises(ValueError):
            model.set_extra_state(state)


def test_source_validation_auc_latest_ties_and_stop_boundary_match(bundle, tmp_path):
    model = KMGMamba(bundle, num_workers=0); model.configure_training(configuration(100))
    author = load_definitions('kmgmamba', 'utils/core_utils.py', ['EarlyStopping'], {'np': np, 'torch': torch})
    original = author.EarlyStopping(patience=20, stop_epoch=50)
    for epoch, auc in enumerate([.6, .6, .7, .7]+[.5]*60):
        previous = original.best_score if original.best_score is not None else -float('inf')
        model.epoch_index = epoch; model.after_validation({'kmgmamba_val_auc': auc})
        original(epoch, auc, nn.Linear(1, 1), ckpt_name=str(tmp_path/'source.pt'))
        assert model.best_auc == original.best_score and model.early_bad == original.counter
        assert model.training_done == original.early_stop
        assert model.selection_improved(auc, previous) == (auc >= previous)
        if model.training_done:
            assert epoch == 51
            break
    else:
        pytest.fail('Original stop boundary not reached')


@pytest.mark.parametrize('override', [{'max_instances': 8}, {'grad_clip': 1.}, {'selection_metric': 'macro_auc'}])
def test_source_recipe_refuses_unregistered_sampling_clipping_or_selection(bundle, override):
    model = KMGMamba(bundle, num_workers=0)
    with pytest.raises(ValueError):
        model.configure_training({**configuration(), **override})


def test_explicit_multiclass_profile_trains_and_reloads_all_three_heads(tmp_path):
    data = create_kmgmamba_demo(tmp_path/'multi', patients=30, classes=3, dimension=1536)
    run = train(data, 'kmgmamba_multiclass_repaired', tmp_path/'run', seed=736,
                config={'epochs': 2, 'threads': 1, 'model': {'num_workers': 0}})
    evaluate(run)
    frame = pd.read_csv(run/'test_predictions.csv')
    assert {'p_0', 'p_1', 'p_2'} <= set(frame) and set(frame.target) == {0, 1, 2}
    checkpoint = torch.load(run/'last.pt', weights_only=False)
    assert checkpoint['model']['network.instance_classifier.weight'].shape == (3, 96)
    assert checkpoint['model']['network.pool_fn.attention.attention.3.weight'].shape == (3, 128)

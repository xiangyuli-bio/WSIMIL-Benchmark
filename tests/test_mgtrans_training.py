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
from milplus.mgtrans import MGTrans, bandwidth
from milplus.mgtrans_demo import create_mgtrans_demo
from milplus.upstream import load_definitions, source_root


@pytest.fixture(autouse=True)
def runtime():
    if not source_root('mgtrans').is_dir():
        pytest.skip('Fetch the pinned MG-Trans source')
    torch.set_num_threads(1)


@pytest.fixture
def bundle(tmp_path):
    # 36 actual training slides exercise the original32-example MI stage.
    return Bundle(create_mgtrans_demo(tmp_path/'data', patients=36, classes=3, roots=16))


def configuration(epochs=2):
    return {'epochs': epochs, 'patience': 20, 'lr': .0001, 'weight_decay': 1e-5,
            'max_instances': None, 'grad_clip': None, 'selection_metric': 'mgtrans_val_error'}


def assert_identical(left, right):
    """Exact tensor and metadata comparison; torch's helper cannot compare strings."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


def test_full_native_training_loop_adam_mi_and_rng_agree(bundle):
    model = MGTrans(bundle, num_workers=0)
    model.configure_training(configuration())
    native = copy.deepcopy(model.network)
    actual_opt = model.make_optimizer(.0001, 1e-5)
    native_opt = torch.optim.Adam(native.parameters(), lr=.0001, weight_decay=1e-5)
    native_helpers = load_definitions('mgtrans', 'utils/utils.py',
        ['collate_tranformer', 'get_split_loader', 'calculate_error'],
        {'torch': torch, 'np': np, 'DataLoader': DataLoader, 'RandomSampler': RandomSampler,
         'SequentialSampler': SequentialSampler, 'device': torch.device('cpu')})
    observed_mi = []
    def mi(*args, **kwargs):
        value = model.source_util.calculate_MI(*args, **kwargs)
        observed_mi.append(float(value.detach()))
        return value
    author = load_definitions('mgtrans', 'utils/core_utils.py', ['Accuracy_Logger', 'train_loop'],
        {'torch': torch, 'np': np, 'nn': nn, 'pdist': __import__('scipy').spatial.distance.pdist,
         'squareform': __import__('scipy').spatial.distance.squareform,
         'calculate_MI': mi, 'calculate_error': native_helpers.calculate_error})
    rows = bundle.records('train')
    class NativeDataset(Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            x, c, y, d, mask = bundle.mgtrans.load(row)
            return x, c, y, d, row['target'], mask
    loader = native_helpers.get_split_loader(NativeDataset(), training=True, mode='transformer')
    initial_rng = rng_state()
    model.start_epoch(0)
    losses = model.epoch(bundle, actual_opt, 'cpu')
    actual_rng = rng_state()
    restore_rng(initial_rng)
    author.train_loop(SimpleNamespace(model_type='MG_Trans'), 0, native, loader, native_opt, 3)
    assert len(losses) == 36 and model.mi_updates == [5] and len(observed_mi) == 10
    assert all(np.isfinite(observed_mi)) and any(value != 0 for value in observed_mi)
    torch.testing.assert_close(model.network.state_dict(), native.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(actual_opt.state_dict(), native_opt.state_dict(), atol=0, rtol=0)
    assert torch.equal(actual_rng['torch'], rng_state()['torch'])
    np.testing.assert_array_equal(actual_rng['numpy'][1], rng_state()['numpy'][1])


def test_exact_resume_preserves_full_mi_graph_adam_and_selection(bundle, tmp_path):
    cfg = {'epochs': 2, 'threads': 1, 'model': {'num_workers': 0}}
    full = train(bundle.root, 'MG-Trans', tmp_path/'full', seed=2026, config=cfg)
    partial = train(bundle.root, 'MG-Trans', tmp_path/'partial', seed=2026, config=cfg, stop_after=1)
    assert not (partial/'test_predictions.csv').exists()
    train(bundle.root, 'MG-Trans', partial, seed=2026, config=cfg, resume=True)
    a, b = [torch.load(path/'last.pt', weights_only=False) for path in [full, partial]]
    assert_identical(a['model'], b['model'])
    assert_identical(a['optimizers'], b['optimizers'])
    assert a['history'] == b['history']
    state = a['model']['_extra_state']
    assert state['optimizer_updates'] == 72 and state['mi_updates'] == [5, 5]
    assert not (full/'test_predictions.csv').exists()
    evaluate(full); before = pd.read_csv(full/'test_predictions.csv')
    evaluate(full); pd.testing.assert_frame_equal(before, pd.read_csv(full/'test_predictions.csv'))
    # Checkpoints must reject removing MI updates even if weights still load.
    model = MGTrans(bundle, num_workers=0); model.configure_training(configuration())
    broken = copy.deepcopy(state); broken['mi_updates'][0] = 0
    with pytest.raises(ValueError, match='sliding MI stage'):
        model.set_extra_state(broken)


def test_native_latest_tie_and_zero_based_stop_epoch_are_preserved(bundle, tmp_path):
    model = MGTrans(bundle, num_workers=0); model.configure_training(configuration(200))
    author = load_definitions('mgtrans', 'utils/core_utils.py', ['EarlyStopping'],
        {'np': np, 'torch': torch}, replacements=[('np.Inf', 'np.inf')])
    reference = author.EarlyStopping(patience=20, stop_epoch=50)
    errors = [.2, .2, .1, .1]+[.3]*60
    for epoch, error in enumerate(errors):
        previous = reference.best_score if reference.best_score is not None else -float('inf')
        model.epoch_index = epoch
        model.after_validation({'mgtrans_val_error': error})
        reference(epoch, error, nn.Linear(1, 1), ckpt_name=str(tmp_path/'native.pt'))
        assert model.early_bad == reference.counter and model.training_done == reference.early_stop
        assert model.selection_improved(-error, previous) == (-error >= previous)
        if model.training_done:
            assert epoch == 51
            break
    else:
        pytest.fail('Source early-stop boundary was never reached')


@pytest.mark.parametrize('override', [{'max_instances': 8}, {'grad_clip': 1.}, {'selection_metric': 'macro_auc'}])
def test_source_recipe_cannot_silently_drop_required_mechanisms(bundle, override):
    model = MGTrans(bundle, num_workers=0)
    with pytest.raises(ValueError):
        model.configure_training({**configuration(), **override})


def test_source_degenerate_bandwidth_fails_instead_of_silent_surrogate():
    with pytest.raises(ValueError, match='degenerate'):
        bandwidth(torch.zeros(32, 192))

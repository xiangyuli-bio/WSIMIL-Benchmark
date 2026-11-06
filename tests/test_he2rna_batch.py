"""Actual batch convolution, stochastic updates and signed ragged input checks."""
import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import units
from milplus.he2rna import load_he2rna
from milplus.models import build_bundle_model
from milplus.regression import regression_epoch
from milplus.upstream import source_root


@pytest.fixture
def setup(tmp_path, monkeypatch):
    if not source_root('he2rna').is_dir():
        pytest.skip('Fetch pinned HE2RNA source')
    monkeypatch.setenv('MILPLUS_HE2RNA_CACHE', str(tmp_path/'cache'))
    torch.set_num_threads(1)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, regression=True))
    model = build_bundle_model('he2rna', bundle, {'layers': [16, 12], 'ks': [1, 2, 4],
                               'dropout': .25, 'input_mode': 'tiles', 'batch_size': 3})
    model.prepare(bundle, 2, 'cpu')
    return model, bundle


@pytest.mark.parametrize('training', [False, True])
def test_native_positive_batch_preserves_output_gradients_and_dropout_stream(setup, training):
    model, bundle = setup
    author = load_he2rna().HE2RNA(bundle.dim, 3, layers=[16, 12], ks=[1, 2, 4], dropout=.25)
    author.load_state_dict(model.network.state_dict())
    model.train(training); author.train(training)
    bags = [torch.rand(9, bundle.dim, requires_grad=True) for _ in range(3)]
    reference_bags = [x.detach().clone().requires_grad_() for x in bags]
    state = torch.get_rng_state()
    np.random.seed(191)
    actual = model.forward_batch(bags)['raw']
    actual_rng, actual_np = torch.get_rng_state(), np.random.get_state()
    torch.set_rng_state(state); np.random.seed(191)
    expected = author(torch.stack([x.T for x in reference_bags]))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(actual_rng, torch.get_rng_state())
    np.testing.assert_array_equal(actual_np[1], np.random.get_state()[1])
    assert actual_np[2:] == np.random.get_state()[2:]
    target = torch.randn(3, 3)
    F.mse_loss(actual, target).backward(); F.mse_loss(expected, target).backward()
    for left, right in zip(bags, reference_bags):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    for left, right in zip(model.network.parameters(), author.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_ragged_signed_and_zero_real_rows_remain_valid_and_padding_cannot_win(setup):
    model, bundle = setup
    model = build_bundle_model('he2rna', bundle, {'layers': [], 'ks': [1, 2, 4],
                               'dropout': 0., 'input_mode': 'tiles'})
    with torch.no_grad():
        model.network.conv0.weight.fill_(1.)
        model.network.conv0.bias.zero_()
    bags = [-torch.ones(5, bundle.dim), torch.zeros(9, bundle.dim), -torch.ones(7, bundle.dim)*2]
    bags = [x.requires_grad_() for x in bags]
    actual = model.forward_batch(bags, k=4)
    expected = torch.tensor([[-bundle.dim]*3, [0.]*3, [-2*bundle.dim]*3])
    torch.testing.assert_close(actual['raw'], expected, atol=0, rtol=0)
    actual['raw'].sum().backward()
    assert all(x.grad.isfinite().all() and x.grad.abs().sum() > 0 for x in bags)
    model.eval()
    torch.testing.assert_close(model.forward_batch(bags)['raw'], expected, atol=0, rtol=0)
    assert (actual['prediction'] == 0).all()


def test_ragged_batch_optimizer_and_rng_match_native_batched_conv(setup):
    model, bundle = setup
    reference = copy.deepcopy(model)
    groups = units(bundle, 'train')[:5]
    original_load = bundle.load
    lengths = {rows[0]['slide_id']: 5+i for i, rows in enumerate(groups)}
    def load(row, **kwargs):
        n = lengths.get(row['slide_id'], 8)
        return tuple(x[:n] if x is not None else None for x in original_load(row, **kwargs))
    bundle.load = load
    actual_opt, reference_opt = [torch.optim.Adam(m.parameters(), lr=.0003) for m in [model, reference]]
    state = torch.get_rng_state()
    np.random.seed(121)
    actual_losses = regression_epoch(model, bundle, groups, actual_opt, 'cpu',
                                    {'max_instances': None, 'grad_clip': 1e5})
    actual_rng, actual_np = torch.get_rng_state(), np.random.get_state()
    torch.set_rng_state(state); np.random.seed(121)
    expected_losses = []
    for start in [0, 3]:
        window = groups[start:start+3]
        bags = [bundle.load(rows[0])[0] for rows in window]
        padded = torch.stack([F.pad(x.T, (0, max(len(b) for b in bags)-len(x))) for x in bags])
        k = reference.training_k()
        scores = reference.network.conv(padded)
        raw = torch.stack([scores[i, :, :len(x)].topk(k, dim=1).values.sum(1)/k
                           for i, x in enumerate(bags)])
        target = torch.stack([bundle.regression_targets.tensor(rows[0]['global_patient_id']) for rows in window])
        loss = F.mse_loss(raw, target)
        reference_opt.zero_grad(set_to_none=True); loss.backward(); reference_opt.step()
        expected_losses.append(float(loss.detach()))
    np.testing.assert_allclose([r['loss'] for r in actual_losses], expected_losses, atol=0, rtol=0)
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(actual_opt.state_dict(), reference_opt.state_dict(), atol=0, rtol=0)
    assert torch.equal(actual_rng, torch.get_rng_state())
    np.testing.assert_array_equal(actual_np[1], np.random.get_state()[1])
    assert actual_np[2:] == np.random.get_state()[2:]


@pytest.mark.parametrize('defect', ['empty', 'short', 'dtype', 'nonfinite'])
def test_invalid_batch_is_rejected_before_convolution(setup, defect):
    model, bundle = setup
    bags = [torch.randn(7, bundle.dim), torch.randn(9, bundle.dim)]
    if defect == 'empty':
        bags = []
    elif defect == 'short':
        bags[1] = bags[1][:3]
    elif defect == 'dtype':
        bags[1] = bags[1].double()
    else:
        bags[1][0, 0] = float('nan')
    with pytest.raises(ValueError):
        model.forward_batch(bags)

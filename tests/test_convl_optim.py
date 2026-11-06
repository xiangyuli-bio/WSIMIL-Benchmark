"""Independent full original update-path and exact Lookahead/RAdam recovery."""
import copy
from collections import OrderedDict

import pytest
import torch
from torch import nn

from milplus.convl_optim import create_source_optimizer, original_optimizer_classes


def tiny_parameters():
    # Dedicated optimizer check; full native ConVL is checked separately.
    return nn.Sequential(OrderedDict([('projection', nn.Linear(4, 3)), ('norm', nn.LayerNorm(3))]))


def step(model, optimizer, index):
    optimizer.zero_grad()
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)/13+index/100
    model(x).square().sum().backward()
    optimizer.step()


def test_actual_weight_decay_partition_and_original_lookahead_first_step():
    torch.manual_seed(2024)
    model = tiny_parameters(); opt = create_source_optimizer(model)
    assert opt.k == 5 and opt.alpha == .5 and len(opt.param_groups) == 2
    assert [group['weight_decay'] for group in opt.param_groups] == [0., 1e-5]
    assert {id(p) for p in opt.param_groups[1]['params']} == {id(model.projection.weight)}
    assert opt.param_groups is opt.optimizer.param_groups
    assert all(group['counter'] == 0 for group in opt.param_groups)
    step(model, opt, 0)
    assert all(group['counter'] == 1 for group in opt.param_groups)
    assert len(opt.state) == len(list(model.parameters()))


def test_every_weight_moment_and_slow_weight_matches_actual_author37updates():
    torch.manual_seed(25)
    actual = tiny_parameters(); reference = copy.deepcopy(actual)
    opt = create_source_optimizer(actual)
    RAdam, Lookahead = original_optimizer_classes()
    # Actual author classes and exact original decay partition. Only no state
    # serialization is attempted on the legacy reference lacking modern hooks.
    no_decay = [p for n, p in reference.named_parameters() if p.ndim == 1 or n.endswith('.bias')]
    decay = [p for n, p in reference.named_parameters() if p.ndim != 1 and not n.endswith('.bias')]
    old = Lookahead(RAdam([{'params': no_decay, 'weight_decay': 0.},
                          {'params': decay, 'weight_decay': 1e-5}], lr=.0002, weight_decay=0.))
    for i in range(37):
        step(actual, opt, i); step(reference, old, i)
        for a, b in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in ['step', 'exp_avg', 'exp_avg_sq']:
                torch.testing.assert_close(opt.optimizer.state[a][name], old.optimizer.state[b][name], rtol=0, atol=0)
            torch.testing.assert_close(opt.state[a]['slow_param'], old.state[b]['slow_param'], rtol=0, atol=0)
        assert opt.optimizer.buffer == old.optimizer.buffer
        assert [g['counter'] for g in opt.param_groups] == [g['counter'] for g in old.param_groups]


@pytest.mark.parametrize('cut', [1, 5, 7, 11])
def test_resume_retains_both_moments_slow_weights_all_counters_and_buffer(cut):
    torch.manual_seed(81)
    actual = tiny_parameters(); opt = create_source_optimizer(actual)
    for i in range(cut):
        step(actual, opt, i)
    resumed = copy.deepcopy(actual); restored = create_source_optimizer(resumed)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert restored.param_groups is restored.optimizer.param_groups
    for i in range(cut, 23):
        step(actual, opt, i); step(resumed, restored, i)
        for a, b in zip(actual.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(opt.state[a]['slow_param'], restored.state[b]['slow_param'], rtol=0, atol=0)
            for name in ['step', 'exp_avg', 'exp_avg_sq']:
                torch.testing.assert_close(opt.optimizer.state[a][name], restored.optimizer.state[b][name], rtol=0, atol=0)
        assert opt.optimizer.buffer == restored.optimizer.buffer

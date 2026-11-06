"""Actual Torch GradScaler overflow and default2000-step growth, plus corruption."""
import copy

import pytest
import torch

from milplus.convl_precision import validate_scaler_history


@pytest.fixture(scope='module')
def actual_scaler_history():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.)
    # This is only a real scaler state-machine test on CPU, not an alternative
    # precision route or a reduced native ConVL model. Source FP16 still needs CUDA.
    scaler = torch.amp.GradScaler('cpu')
    traces = []
    for index in range(2002):
        optimizer.zero_grad(); scaler.scale(parameter.sum()).backward()
        if index in [0, 2001]:
            parameter.grad.fill_(float('inf'))
        before = scaler.get_scale(); scaler.step(optimizer); scaler.update(); after = scaler.get_scale()
        traces.append({'scale_before': before, 'scale_after': after, 'skipped': after < before})
    assert traces[0]['scale_after'] == 32768. and traces[2000]['scale_after'] == 65536.
    return [traces], scaler.state_dict()


def test_actual_default_scaler_skip_growth_and_backoff_are_preserved(actual_scaler_history):
    history, saved = actual_scaler_history
    validate_scaler_history(True, history, saved)
    assert saved['scale'] == 32768. and saved['_growth_tracker'] == 0


@pytest.mark.parametrize('field,value', [('scale', 1.), ('_growth_tracker', 1), ('growth_factor', 4.),
    ('growth_interval', 1000), ('backoff_factor', .25)])
def test_changed_scaler_cannot_silently_change_resumed_training(actual_scaler_history, field, value):
    history, saved = actual_scaler_history
    corrupt = dict(saved); corrupt[field] = value
    with pytest.raises(ValueError, match='ConVL'):
        validate_scaler_history(True, history, corrupt)


def test_lost_window_and_rewritten_overflow_transition_are_rejected(actual_scaler_history):
    history, saved = actual_scaler_history
    bad = copy.deepcopy(history); bad[0].pop(0)
    with pytest.raises(ValueError, match='prior scale or window'):
        validate_scaler_history(True, bad, saved)
    bad = copy.deepcopy(history); bad[0][-1]['skipped'] = False
    with pytest.raises(ValueError, match='overflow/growth transition'):
        validate_scaler_history(True, bad, saved)


def test_fp32_cannot_accept_amp_scale_or_overflow_state():
    validate_scaler_history(False, [[{'scale_before': 1., 'scale_after': 1., 'skipped': False}]], {})
    with pytest.raises(ValueError, match='FP32'):
        validate_scaler_history(False, [[{'scale_before': 2., 'scale_after': 1., 'skipped': True}]], {})

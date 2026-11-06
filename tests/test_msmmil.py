"""MSMMIL source equations, complete trainer, Lookahead state, and fresh-object continuation."""
import ast
import copy
import json
import math
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Optional

import pandas as pd
import pytest
import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, load_unit, restore_rng, rng_state, train, units
from milplus.execution import matrix
from milplus.mamba_mil import backend_functions
from milplus.msmmil import MSMMIL, msmmil_epoch
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def reference(hidden=16, same_kernel=False):
    root, sha = verified_source('msmmil')
    scope = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'Optional': Optional, 'Tensor': Tensor,
             'rearrange': rearrange, 'repeat': repeat, 'Optimizer': torch.optim.Optimizer,
             'defaultdict': defaultdict, 'pl': SimpleNamespace(LightningModule=nn.Module)}
    def read(path, names):
        text = pinned_text(root, sha, path)
        if path == 'models/MKMIL.py' and hidden != 256:
            text = text.replace('512', str(2*hidden)).replace('256', str(hidden))
        nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)  # noqa: S102 -- independent pinned author ASTs
    read('mamba/mamba_ssm/ops/selective_scan_interface.py', ['selective_scan_ref', 'mamba_inner_ref'])
    scope['selective_scan_fn'] = scope['selective_scan_ref']
    def convolution(x, w, bias, activation):
        assert activation == 'silu'
        return F.silu(F.conv1d(F.pad(x, (w.shape[-1]-1, 0)), w[:, None], bias, groups=x.shape[1]))
    scope['causal_conv1d_fn'] = convolution
    def no_out(xz, cw, cb, xw, dw, A, *args, **kwargs):
        return scope['mamba_inner_ref'](xz, cw, cb, xw, dw,
            torch.eye(xz.shape[1]//2, device=xz.device, dtype=xz.dtype), None, A, *args, **kwargs).transpose(1, 2)
    scope['mamba_inner_fn_no_out_proj'] = no_out
    scope['mamba_inner_fn'] = scope['mamba_inner_ref']
    if same_kernel:
        scope['mamba_inner_fn_no_out_proj'], scope['mamba_inner_fn'], _ = backend_functions('torch', 16)
    read('mamba/mamba_ssm/modules/atrousfourway_mamba.py',
         ['TransposeTokenReEmbedding', 'BiAttn', 'AtrousFourWayMamba'])
    scope['AFWMamba'] = scope['AtrousFourWayMamba']
    read('models/MKMIL.py', ['initialize_weights', 'MKMIL'])
    read('models/model_interface.py', ['ModelInterface'])
    read('MyOptimizer/optim_factory.py', ['add_weight_decay'])
    read('MyOptimizer/radam.py', ['RAdam'])
    read('MyOptimizer/lookahead.py', ['Lookahead'])
    return scope


def make(path, **options):
    dimension = options.pop('dimension', 16)
    bundle = Bundle(create_demo(path, patients=20, dimension=dimension))
    model = MSMMIL(bundle, **{'hidden': 16, 'scan_chunk': 16, **options})
    return model, bundle


def assert_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('rate', [2, 4, 10])
@pytest.mark.parametrize('length', [1, 7, 20, 27])
def test_complete_source_output_and_all_gradients(tmp_path, rate, length):
    model, _ = make(tmp_path/'data', rate=rate, dropout=0.)
    raw = reference()['MKMIL'](2, 0., 'relu', n_features=16, layer=2, rate=rate, type='AFWMamba')
    raw.load_state_dict(model.network.state_dict())
    x = torch.randn(length, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = model(x)['logits'], raw(data=y[None])['logits']
    torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)
    F.cross_entropy(a, torch.tensor([1])).backward()
    F.cross_entropy(b, torch.tensor([1])).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=4e-6)
    for (name, p), (other, q) in zip(model.network.named_parameters(), raw.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=6e-6)


@pytest.mark.parametrize('length', [1, 27])
def test_native_1024_features_256_hidden_default_two_layers(tmp_path, length):
    model, _ = make(tmp_path/'data', dimension=1024, hidden=256, dropout=0.)
    expected = reference(256)['MKMIL'](3, 0., 'relu', n_features=1024, layer=2, rate=10, type='AFWMamba')
    # Include multiclass behavior while preserving the author's native widths.
    model.network.classifier = nn.Linear(512, 3)
    expected.load_state_dict(model.network.state_dict())
    x = torch.randn(length, 1024)
    a, b = model(x)['logits'], expected(data=x[None])['logits']
    torch.testing.assert_close(a, b, rtol=3e-5, atol=4e-6)
    a.square().mean().backward(); b.square().mean().backward()
    for name, p in model.network.named_parameters():
        q = dict(expected.named_parameters())[name]
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=7e-4, atol=1e-5)


@pytest.mark.parametrize('length', [1, 19, 20, 21, 60])
def test_checkerboard_zero_padding_flip_and_exact_inverse(length):
    helper = reference()['TransposeTokenReEmbedding']
    x = torch.arange(2*length, dtype=torch.float32).reshape(1, 2, length)
    height = math.ceil(length/10)
    height += height % 2
    grid = F.pad(x, (0, height*10-length)).transpose(1, 2).reshape(1, height, 10, 2)
    expected = [grid[:, 0::2, 0::2].flatten(1, 2).transpose(1, 2),
                grid[:, 1::2, 0::2].transpose(1, 2).flatten(1, 2).transpose(1, 2),
                grid[:, 0::2, 1::2].flatten(1, 2).transpose(1, 2).flip(-1),
                grid[:, 1::2, 1::2].transpose(1, 2).flatten(1, 2).transpose(1, 2).flip(-1)]
    parts = helper.patches_reemmbedding(x, 10)
    for a, b in zip(parts, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored = helper.restore(helper.restore_x0_x2(parts[0], parts[2], 10),
                              helper.restore_x1_x3(parts[1], parts[3], 10), 10, length)
    torch.testing.assert_close(x, restored, rtol=0, atol=0)


def test_all_three_scans_active_and_same_attention_called_three_times(tmp_path):
    model, _ = make(tmp_path/'data')
    counts = [0, 0]
    hooks = []
    for index, layer in enumerate(model.network.layers):
        def callback(module, args, output, index=index):
            counts[index] += 1
        hooks.append(layer[1].att.register_forward_hook(callback))
    F.cross_entropy(model(torch.randn(27, 16))['logits'], torch.tensor([1])).backward()
    assert counts == [3, 3]
    for layer in model.network.layers:
        for suffix in ['', '_b', '_c']:
            for component in ['conv1d', 'x_proj', 'dt_proj']:
                assert getattr(layer[1], component+suffix).weight.grad.abs().sum() > 0
            assert getattr(layer[1], 'A'+suffix+'_log' if suffix else 'A_log').grad.abs().sum() > 0
            assert getattr(layer[1], 'D'+suffix).grad.abs().sum() > 0
    for hook in hooks:
        hook.remove()


def test_author_radam_lookahead_thirteen_updates_and_weight_decay_groups(tmp_path):
    model, _ = make(tmp_path/'data')
    raw = reference()
    expected = copy.deepcopy(model.network)
    opt = model.make_optimizer(.002, .01)
    fast = raw['RAdam'](raw['add_weight_decay'](expected, .01), lr=.002, weight_decay=0.)
    lookahead = raw['Lookahead'](fast)
    no_decay = {id(p) for p in opt.param_groups[0]['params']}
    for name, p in model.network.named_parameters():
        assert (id(p) in no_decay) == (p.ndim == 1 or name.endswith('.bias'))
    for step in range(13):
        opt.zero_grad(); fast.zero_grad()
        for p, q in zip(model.network.parameters(), expected.parameters()):
            g = torch.randn_like(p)*.01
            p.grad, q.grad = g.clone(), g.clone()
        opt.step(); model.after_step(); lookahead.step()
        assert_state(model.network.state_dict(), expected.state_dict())
        if step < 5:
            assert not model.slow
    assert model.steps == 13
    for (name, _), q in zip(model.network.named_parameters(), expected.parameters()):
        assert_state(model.slow[name], lookahead.state[q]['slow_buffer'])


@pytest.mark.parametrize('bags', [4, 5])
def test_full_source_training_step_three_epochs_and_lightning_accumulation(tmp_path, bags):
    model, bundle = make(tmp_path/'data')
    raw = reference(same_kernel=True)
    network = raw['MKMIL'](2, .5, 'relu', n_features=16, layer=2, rate=10, type='AFWMamba')
    network.load_state_dict(model.network.state_dict())
    opt = model.make_optimizer(2e-4, 1e-5)
    fast = raw['RAdam'](raw['add_weight_decay'](network, 1e-5), lr=2e-4, weight_decay=0.)
    lookahead = raw['Lookahead'](fast)
    interface = SimpleNamespace(model=network, loss=nn.CrossEntropyLoss(), data=[{'count': 0, 'correct': 0} for _ in range(2)])
    groups = units(bundle, 'train')[:bags]
    for epoch in range(3):
        state = rng_state()
        fast.zero_grad(set_to_none=True)
        for index, rows in enumerate(groups):
            x, _ = load_unit(bundle, rows, 'cpu')
            batch = (x[None], torch.tensor([rows[0]['target']]), None, None)
            loss = raw['ModelInterface'].training_step(interface, batch, index)['loss']
            (loss/2).backward()
            if (index+1) % 2 == 0 or index+1 == len(groups):
                lookahead.step(); fast.zero_grad(set_to_none=True)
        restore_rng(state)
        msmmil_epoch(model, bundle, groups, opt, 'cpu', {})
        assert_state(model.network.state_dict(), network.state_dict())
        for p, q in zip(model.network.parameters(), network.parameters()):
            assert_state(opt.state[p], fast.state[q])
    assert model.steps == 3*math.ceil(bags/2)


@pytest.mark.parametrize('stop_after', [1, 2, 3])
def test_exact_fresh_objects_resume_across_slow_weight_updates(tmp_path, stop_after):
    data = create_demo(tmp_path/'data', patients=20)
    config = {'epochs': 4, 'threads': 1, 'model': {'hidden': 8, 'scan_chunk': 16}}
    full = train(data, 'MSMMIL', tmp_path/'full', config=config)
    resume = train(data, 'MSMMIL', tmp_path/'resume', config=config, stop_after=stop_after)
    train(data, 'MSMMIL', resume, config=config, resume=True)
    for kind in ['last', 'best']:
        a, b = [torch.load(p/(kind+'.pt'), weights_only=False) for p in (full, resume)]
        assert_state(a['model'], b['model']); assert_state(a['optimizers'], b['optimizers'])
        assert a['history'] == b['history']
    assert evaluate(full) == evaluate(resume)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(resume/'test_predictions.csv'))
    protocol = json.loads((full/'method_protocol.json').read_text())
    assert protocol['optimizer_steps'] > 12
    request = json.loads((full/'request.json').read_text())
    assert request['config']['selection_metric'] == 'log_loss'
    assert request['config']['grad_clip'] is None
    assert request['config']['patience'] == 40


def test_backend_and_slow_state_identity(tmp_path):
    model, _ = make(tmp_path/'data')
    state = model.get_extra_state()
    state['backend'] = {**state['backend'], 'torch': 'changed'}
    with pytest.raises(ValueError, match='backend identity'):
        model.set_extra_state(state)
    with pytest.raises(ValueError, match='max_instances'):
        model.configure_training({'max_instances': 2})
    assert matrix(tmp_path/'data', ['MSMMIL'], ['toy'], [2026], tmp_path/'matrix.json')[0]['state'] == 'ready'


@pytest.mark.parametrize('options', [{'rate': 3}, {'rate': 0}, {'rate': True}, {'hidden': 1},
                                   {'layers': 0}, {'scan_chunk': 0}, {'accumulation': 0},
                                   {'dropout': float('nan')}, {'dropout': 1.}, {'backend': 'auto'}])
def test_invalid_model_options_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        make(tmp_path/'data', **options)


def test_reject_wrong_task_and_patient_training(tmp_path):
    bundle = Bundle(create_demo(tmp_path/'surv', patients=20, survival=True))
    with pytest.raises(ValueError, match='classification only'):
        MSMMIL(bundle)
    _, bundle = make(tmp_path/'class')
    bundle.config['training_unit'] = 'patient'
    with pytest.raises(ValueError, match='individual slide'):
        MSMMIL(bundle)


@pytest.mark.parametrize('shape', [(0, 16), (3, 15), (1, 3, 16)])
def test_invalid_feature_shapes_rejected(tmp_path, shape):
    model, _ = make(tmp_path/'data')
    with pytest.raises(ValueError, match='finite nonempty'):
        model(torch.empty(shape))


def test_nonfinite_features_rejected(tmp_path):
    model, _ = make(tmp_path/'data')
    with pytest.raises(ValueError, match='finite nonempty'):
        model(torch.full((3, 16), float('nan')))


def test_real_cli_check_run_evaluate_family_name(tmp_path):
    data = create_demo(tmp_path/'data', patients=20)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'epochs': 1, 'threads': 1, 'model': {'hidden': 8, 'scan_chunk': 16}}))
    common = [sys.executable, '-m', 'milplus']
    for args in [['check', '--data', str(data), '--method', 'MSMMIL', '--deep'],
                 ['run', '--data', str(data), '--method', 'MSMMIL', '--config', str(config), '--output', str(tmp_path/'run')],
                 ['evaluate', '--run', str(tmp_path/'run')]]:
        result = subprocess.run(common+args, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['n'] > 0
    assert json.loads((tmp_path/'run/request.json').read_text())['method'] == 'msmmil'

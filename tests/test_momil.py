"""MoMIL's pinned SSD references, three literal orders, author trainer and named extension."""
import ast
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.momil import MoMIL, backend_namespace, momil_epoch, sampled_groups
from milplus.ssd_scan import split_conv_scan, ssd_scan
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def reference(hidden=32, state_size=8, same_kernel=False, repair=True, repair_tail=False):
    root, sha = verified_source('momil')
    ns = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math,
          'rearrange': rearrange, 'repeat': repeat, 'PyTorchModelHubMixin': type('Hub', (), {})}
    def read(path, names, patches=()):
        code = subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)
        for a, b in patches:
            assert a in code; code = code.replace(a, b)
        nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'author/'+path, 'exec'), ns)  # noqa: S102 -- independent fixed Git ASTs
    base = 'mamba_latest/mamba_ssm/'
    read(base+'ops/selective_scan_interface.py', ['selective_scan_ref'])
    ns['selective_scan_fn'] = ns['selective_scan_ref']
    read(base+'ops/triton/ssd_combined.py', ['ssd_selective_scan'],
         [('from mamba_ssm.ops.selective_scan_interface import selective_scan_fn', '')])
    read(base+'ops/triton/layernorm_gated.py', ['rms_norm_ref'])
    ns['rmsnorm_fn'] = ns['rms_norm_ref']
    read(base+'ops/triton/layernorm_gated.py', ['RMSNorm'])
    ns['RMSNormGated'] = ns['RMSNorm']
    def conv(x, weight, bias, activation):
        assert activation == 'silu'
        return F.silu(F.conv1d(F.pad(x, (weight.shape[-1]-1, 0)), weight[:, None], bias, groups=x.shape[1]))
    ns['causal_conv1d_fn'] = conv
    read(base+'ops/triton/ssd_combined.py', ['mamba_split_conv1d_scan_ref'])
    ns['mamba_split_conv1d_scan_combined'] = lambda *a, seq_idx=None, **kw: ns['mamba_split_conv1d_scan_ref'](*a, **kw)
    if same_kernel:
        mamba, _ = backend_namespace('torch', 16)
    else:
        read(base+'modules/mamba2.py', ['Mamba2']); mamba = ns['Mamba2']
    ns['Mamba2'] = lambda d_model, d_state, d_conv, expand: mamba(d_model=d_model, d_state=state_size, d_conv=d_conv, expand=expand)
    patches = [('512', str(hidden))]
    if repair:
        patches += [('super(SpeMIL, self).__init__()', 'super(MoMIL, self).__init__()')]
    read('models/MoMIL.py', ['initialize_weights', 'TransposeTokenReEmbedding', 'MoMIL'], patches)
    read('utils/core_utils.py', ['Accuracy_Logger', 'train_loop'])
    read('utils/utils.py', ['calculate_error', 'get_optim']); ns['optim'] = torch.optim
    read('utils/survival_utils.py', ['nll_loss', 'NLLSurvLoss', 'l1_reg_all'])
    tail_patches = [('loss = loss / gc + loss_reg', 'loss = loss / min(gc, len(loader) - (batch_idx // gc) * gc) + loss_reg'),
                    ('if (batch_idx + 1) % gc == 0:', 'if (batch_idx + 1) % gc == 0 or batch_idx + 1 == len(loader):')] if repair_tail else ()
    read('utils/survival_core_utils.py', ['train_loop_survival'], tail_patches)
    # Reporting only: training updates do not depend on this metric callback.
    ns['concordance_index_censored'] = lambda *a, **k: (0.5,)
    return ns


def fixture(path, survival=False, classes=2, **options):
    data = create_demo(path, patients=max(20, 10*classes), classes=classes, survival=survival)
    bundle = Bundle(data)
    profile = 'momil_survival_extension' if survival else 'momil'
    model = MoMIL(bundle, profile, **{'hidden': 32, 'state_size': 8, 'scan_chunk': 16, **options})
    model.prepare(bundle, 2026, 'cpu')
    return model, bundle


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
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


def test_unrepaired_author_constructor_fails():
    raw = reference(repair=False)
    with pytest.raises(NameError, match='SpeMIL'):
        raw['MoMIL'](16, 2, 0., 'relu')


@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('length,chunk', [(1, 1), (17, 5), (65, 16)])
def test_version_224_independent_ssd_and_cross_chunk_gradients(groups, length, chunk):
    torch.manual_seed(18)
    inputs = [torch.randn(2, length, 4, 3), torch.randn(2, length, 4), -torch.rand(4),
              torch.randn(2, length, groups, 5), torch.randn(2, length, groups, 5),
              torch.randn(4), torch.randn(2, length, 4, 3), torch.randn(4)]
    a, b = [[v.clone().requires_grad_() for v in inputs] for _ in range(2)]
    x = ssd_scan(*a, dt_limit=(.01, .2), checkpoint_chunk=chunk)
    y = reference()['ssd_selective_scan'](*b, dt_softplus=True, dt_limit=(.01, .2))
    torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-6)
    x.square().sum().backward(); y.square().sum().backward()
    for p, q in zip(a, b):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize('norm_before_gate', [False, True])
def test_version_224_full_projected_convolution_scan_gradients(norm_before_gate):
    torch.manual_seed(20)
    heads, hd, state, length = 4, 3, 5, 17
    dim = heads*hd
    inputs = [torch.randn(1, length, 2*dim+2*state+heads), torch.randn(dim+2*state, 4),
              torch.randn(dim+2*state), torch.randn(heads), -torch.rand(heads), torch.randn(heads)]
    a, b = [[v.clone().requires_grad_() for v in inputs] for _ in range(2)]
    wa, wb = [torch.randn(dim).requires_grad_() for _ in range(2)]; wb.data.copy_(wa.data)
    pa, pb = [torch.randn(7, dim).requires_grad_() for _ in range(2)]; pb.data.copy_(pa.data)
    x = split_conv_scan(*a, 256, headdim=hd, rmsnorm_weight=wa, outproj_weight=pa,
                        norm_before_gate=norm_before_gate, checkpoint_chunk=5)
    y = reference()['mamba_split_conv1d_scan_ref'](*b, 256, headdim=hd, rmsnorm_weight=wb,
                        outproj_weight=pb, norm_before_gate=norm_before_gate)
    torch.testing.assert_close(x, y, atol=5e-6, rtol=1e-5)
    x.square().sum().backward(); y.square().sum().backward()
    for p, q in zip(a+[wa, pa], b+[wb, pb]):
        torch.testing.assert_close(p.grad, q.grad, atol=6e-4, rtol=5e-4)


@pytest.mark.parametrize('length', [1, 7, 17, 65])
@pytest.mark.parametrize('survival', [False, True])
def test_complete_graph_outputs_and_all_parameter_gradients(tmp_path, length, survival):
    model, bundle = fixture(tmp_path/'data', survival, classes=3, dropout=0.)
    raw = reference(); net = raw['MoMIL'](16, 4 if survival else 3, 0., 'relu', survival=survival)
    net.load_state_dict(model.network.state_dict())
    x = torch.randn(length, 16)
    actual = model(x); original = net(x)
    if survival:
        expected = {'hazards': original[0], 'survival': original[1]}
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, atol=5e-6, rtol=2e-5)
        a = model.objective(actual, bundle.records('train')[0]['time'], 1)
        b = raw['nll_loss'](original[0], original[1], model.time_label([bundle.records('train')[0]['time']]), torch.zeros(1), alpha=0.)
    else:
        torch.testing.assert_close(actual['logits'], original[0], atol=6e-6, rtol=3e-5)
        a = F.cross_entropy(actual['logits'], torch.tensor([1])); b = F.cross_entropy(original[0], torch.tensor([1]))
    torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)
    a.backward(); b.backward()
    for (name, p), (other, q) in zip(model.network.named_parameters(), net.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=7e-6, rtol=5e-4, msg=lambda message, name=name: name+': '+message)


@pytest.mark.parametrize('survival,dimension', [(False, 1024), (True, 1536)])
def test_native_512_width_64_state_gradients(tmp_path, survival, dimension):
    data = create_demo(tmp_path/'data', patients=20, dimension=dimension, survival=survival)
    bundle = Bundle(data); model = MoMIL(bundle, 'momil_survival_extension' if survival else 'momil', dropout=0.)
    model.prepare(bundle, 2026, 'cpu')
    raw = reference(512, 64); net = raw['MoMIL'](dimension, 4 if survival else 2, 0., 'relu', survival=survival)
    net.load_state_dict(model.network.state_dict()); x = torch.randn(5, dimension)
    out = model(x); expected = net(x)
    if survival:
        a, b = out['hazards'], expected[0]
    else:
        a, b = out['logits'], expected[0]
    torch.testing.assert_close(a, b, atol=5e-6, rtol=3e-5)
    a.square().sum().backward(); b.square().sum().backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), net.named_parameters()):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=7e-4, msg=lambda message, name=name: name+': '+message)


@pytest.mark.parametrize('length', [1, 2, 3, 5, 16, 17, 65])
def test_literal_channel_reverse_square_transpose_and_padded_tokens(tmp_path, length):
    model, _ = fixture(tmp_path/'data', dropout=0.)
    x = torch.randn(length, 16); seen = []
    handles = [layer[0][0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
               for layer in [model.network.layers, model.network.layers_1, model.network.layers_2]]
    outputs = []
    handles.append(model.network.norm.register_forward_pre_hook(lambda module, args: outputs.append(args[0].shape)))
    try:
        model(x)
    finally:
        for handle in handles: handle.remove()
    h = model.network._fc1(x[None]); width = math.ceil(math.sqrt(length)); padding = width**2-length
    h = torch.cat([h, h[:, :padding]], 1)
    expected = [h, h.flip(-1), rearrange(h, 'b (k w) d -> b (w k) d', w=width)]
    for a, b in zip(seen, expected):
        assert torch.equal(a, b)
    assert outputs == [torch.Size([1, 3*width**2, 32])]


def test_initialization_rng_matches_author_recursive_initialization(tmp_path):
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    torch.manual_seed(44); model = MoMIL(bundle, hidden=32, state_size=8)
    raw = reference(same_kernel=True)
    torch.manual_seed(44); net = raw['MoMIL'](16, 2, .25, 'relu')
    equal(model.network.state_dict(), net.state_dict())


def test_original_classification_trainer_six_adam_updates(tmp_path):
    model, bundle = fixture(tmp_path/'data', classes=7)
    raw = reference(same_kernel=True); net = raw['MoMIL'](16, 7, .25, 'relu')
    net.load_state_dict(model.network.state_dict())
    optimizer = model.make_optimizer(1e-4, 1e-5)
    other = raw['get_optim'](net, SimpleNamespace(opt='adam', lr=1e-4, reg=1e-5))
    groups = units(bundle, 'train')[:3]
    loader = [(bundle.load(rows[0])[0], torch.tensor([rows[0]['target']])) for rows in groups]
    for epoch in range(2):
        state = rng_state(); raw['train_loop'](epoch, net, loader, other, 7, loss_fn=nn.CrossEntropyLoss())
        restore_rng(state); momil_epoch(model, bundle, groups, optimizer, 'cpu', {})
        equal(model.network.state_dict(), net.state_dict()); equal(optimizer.state_dict(), other.state_dict())
    assert model.optimizer_steps == 6


@pytest.mark.parametrize('l1_weight', [0., 1e-5])
def test_generic_author_survival_trainer_full_windows(tmp_path, l1_weight):
    model, bundle = fixture(tmp_path/'data', True, accumulation_steps=3, weighted_sampling=False, l1_weight=l1_weight)
    raw = reference(same_kernel=True); net = raw['MoMIL'](16, 4, .25, 'relu', survival=True)
    net.load_state_dict(model.network.state_dict())
    optimizer = model.make_optimizer(1e-4, 1e-5); other = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    groups = units(bundle, 'train')[:6]
    loader = []
    for rows in groups:
        x = torch.cat([bundle.load(r)[0] for r in sorted(rows, key=lambda row: row['slide_id'])])
        loader.append((x, torch.empty(0), model.time_label([rows[0]['time']]), rows[0]['time'], torch.tensor([1-rows[0]['event']])))
    for epoch in range(2):
        state = rng_state()
        raw['train_loop_survival'](epoch, net, loader, other, 4, loss_fn=raw['NLLSurvLoss'](alpha=0.),
                                  reg_fn=raw['l1_reg_all'] if l1_weight else None, lambda_reg=l1_weight, gc=3)
        restore_rng(state); momil_epoch(model, bundle, groups, optimizer, 'cpu', {})
        equal(model.network.state_dict(), net.state_dict()); equal(optimizer.state_dict(), other.state_dict())
    assert model.optimizer_steps == 4


def test_short_survival_window_flushes_with_actual_denominator(tmp_path):
    model, bundle = fixture(tmp_path/'data', True, accumulation_steps=4, weighted_sampling=False, dropout=0.)
    raw = reference(same_kernel=True, repair_tail=True)
    net = raw['MoMIL'](16, 4, 0., 'relu', survival=True); net.load_state_dict(model.network.state_dict())
    optimizer = model.make_optimizer(1e-4, 1e-5); other = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    groups = units(bundle, 'train')[:6]; loader = []
    for rows in groups:
        x = torch.cat([bundle.load(r)[0] for r in sorted(rows, key=lambda row: row['slide_id'])])
        loader.append((x, torch.empty(0), model.time_label([rows[0]['time']]), rows[0]['time'], torch.tensor([1-rows[0]['event']])))
    raw['train_loop_survival'](0, net, loader, other, 4, loss_fn=raw['NLLSurvLoss'](alpha=0.), gc=4)
    momil_epoch(model, bundle, groups, optimizer, 'cpu', {})
    equal(model.network.state_dict(), net.state_dict()); equal(optimizer.state_dict(), other.state_dict())
    assert model.optimizer_steps == 2


@pytest.mark.parametrize('survival', [False, True])
def test_class_joint_bin_sampling_uses_source_inverse_frequency(tmp_path, survival):
    model, bundle = fixture(tmp_path/'data', survival, weighted_sampling=True)
    groups = units(bundle, 'train')
    labels = [(int(model.time_label([rows[0]['time']]).item()), 1-rows[0]['event']) if survival else rows[0]['target'] for rows in groups]
    weights = torch.tensor([len(labels)/labels.count(v) for v in labels], dtype=torch.double)
    state = rng_state(); expected = torch.multinomial(weights, len(groups), True).tolist(); restore_rng(state)
    assert [id(g) for g in sampled_groups(model, groups)] == [id(groups[i]) for i in expected]


@pytest.mark.parametrize('survival', [False, True])
@pytest.mark.parametrize('early', [False, True])
def test_all_policies_exact_resume_and_reload(tmp_path, survival, early):
    model, bundle = fixture(tmp_path/'data', survival)
    cfg = {'epochs': 2, 'threads': 1, 'model': {'hidden': 32, 'state_size': 8, 'scan_chunk': 16, 'early_stopping': early,
           'accumulation_steps': 3 if survival else 1}}
    full = train(bundle.root, model.profile, tmp_path/'full', config=cfg)
    resumed = train(bundle.root, model.profile, tmp_path/'resumed', config=cfg, stop_after=1)
    train(bundle.root, model.profile, resumed, config=cfg, resume=True)
    for key in ['last', 'best']:
        a, b = [torch.load(path/(key+'.pt'), weights_only=False) for path in [full, resumed]]
        equal(a['model'], b['model']); equal(a['optimizers'], b['optimizers']); assert a['history'] == b['history']
    assert evaluate(full) == evaluate(resumed)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(resumed/'test_predictions.csv'))
    protocol = json.loads((full/('survival_protocol.json' if survival else 'method_protocol.json')).read_text())
    assert protocol['optimizer_steps'] > 0 and protocol['epoch'] == 1
    if survival:
        assert not (full/'method_protocol.json').exists()
    if not early:
        assert torch.load(full/'best.pt', weights_only=False)['epoch'] == 1


def test_last_checkpoint_policy_survives_worsening_validation(tmp_path, monkeypatch):
    from milplus import engine
    model, bundle = fixture(tmp_path/'data')
    calls = []
    def score(bundle, predictions):
        calls.append(1); return {'log_loss': float(len(calls))}
    monkeypatch.setattr(engine, 'score', score)
    cfg = {'epochs': 3, 'model': {'hidden': 32, 'state_size': 8, 'early_stopping': False}, 'threads': 1}
    run = train(bundle.root, model.profile, tmp_path/'run', config=cfg)
    assert torch.load(run/'best.pt', weights_only=False)['epoch'] == 2


def test_train_only_survival_bins_and_patient_slide_order(tmp_path):
    model, bundle = fixture(tmp_path/'data', True)
    observed = bundle.rows[(bundle.rows.subset == 'train') & (bundle.rows.event == 1)].drop_duplicates('global_patient_id').time.to_numpy()
    np.testing.assert_array_equal(model.bin_edges.numpy(), np.quantile(observed, [.25, .5, .75]))
    rows = next(g for g in units(bundle, 'train') if len(g) > 1)
    seen = []; handle = model.network.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    try:
        model.forward_unit(bundle, list(reversed(rows)), 'cpu', {})
    finally:
        handle.remove()
    assert torch.equal(seen[0], torch.cat([bundle.load(r)[0] for r in sorted(rows, key=lambda row: row['slide_id'])]))


@pytest.mark.parametrize('options', [{'hidden': 16}, {'layers': 0}, {'state_size': True}, {'backend': 'auto'},
                                   {'dropout': float('nan')}, {'activation': 'tanh'}, {'accumulation_steps': 2}, {'weighted_sampling': 1}])
def test_invalid_options_fail_explicitly(tmp_path, options):
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    with pytest.raises((ValueError, RuntimeError)):
        MoMIL(bundle, **options)


def test_backend_identity_and_global_truncation_are_not_silently_changed(tmp_path):
    model, _ = fixture(tmp_path/'data')
    with pytest.raises(ValueError, match='max_instances'):
        model.configure_training({'max_instances': 3})
    state = model.get_extra_state(); state['backend'] = {'name': 'different'}
    with pytest.raises(ValueError, match='backend identity'):
        model.set_extra_state(state)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='requires a GPU'):
            backend_namespace('cuda')


def test_family_cli(tmp_path):
    root = create_demo(tmp_path/'data', patients=20)
    config = tmp_path/'config.yaml'; config.write_text(yaml.safe_dump({'epochs': 1, 'threads': 1, 'model': {'hidden': 32, 'state_size': 8}}))
    for args in [['check', '--data', str(root), '--method', 'MoMIL', '--deep'],
                 ['run', '--data', str(root), '--method', 'MoMIL', '--config', str(config), '--output', str(tmp_path/'run')],
                 ['evaluate', '--run', str(tmp_path/'run')]]:
        result = subprocess.run([sys.executable, '-m', 'milplus', *args], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['n'] > 0

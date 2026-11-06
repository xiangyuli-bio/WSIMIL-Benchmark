"""MamMIL: independent SSD equations, topology assets, author graphs/trainers and continuation."""
import ast
import json
import math
import random
import subprocess
import sys
import warnings
from functools import partial
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import DeepGCNLayer, GENConv

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.mammil import PROFILES, MamMIL, NoHubMixin, backend_namespace, mammil_epoch
from milplus.mammil_graph import asset_checksum, build_graph, graph_asset
from milplus.ssd_scan import gated_rms, split_conv_scan, ssd_scan
from milplus.upstream import pinned_text, verified_source

ROUTES = [(p, survival) for p in sorted(PROFILES) for survival in [False, True]
          if p != 'mammil_patient_union' or survival]


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def reference(profile='mammil_source', same_kernel=False):
    ns = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'np': np, 'rearrange': rearrange,
          'repeat': repeat, 'partial': partial, 'warnings': warnings, 'PyTorchModelHubMixin': NoHubMixin,
          'LayerNorm': nn.LayerNorm, 'ReLU': nn.ReLU, 'GENConv': GENConv, 'DeepGCNLayer': DeepGCNLayer,
          'Optimizer': torch.optim.Optimizer,
          '_calculate_fan_in_and_fan_out': nn.init._calculate_fan_in_and_fan_out,
          'pl': SimpleNamespace(LightningModule=nn.Module)}
    def read(path, names, patches=(), source='mammil'):
        source_root, commit = verified_source(source)
        code = pinned_text(source_root, commit, path)
        for old, new in patches:
            assert old in code
            code = code.replace(old, new)
        nodes = [node for node in ast.parse(code).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)  # noqa: S102 -- independently read pinned source
    read('mamba/mamba_ssm/ops/selective_scan_interface.py', ['selective_scan_ref'])
    ns['selective_scan_fn'] = ns['selective_scan_ref']
    read('mamba/mamba_ssm/ops/triton/ssd_combined.py', ['ssd_selective_scan'],
         [('from mamba_ssm.ops.selective_scan_interface import selective_scan_fn', '')])
    read('mamba/mamba_ssm/ops/triton/layernorm_gated.py', ['rms_norm_ref'])
    ns['rmsnorm_fn'] = ns['rms_norm_ref']
    read('mamba/mamba_ssm/ops/triton/layernorm_gated.py', ['RMSNorm'])
    ns['RMSNormGated'] = ns['RMSNorm']
    def conv(x, weight, bias, activation):
        assert activation == 'silu'
        return F.silu(F.conv1d(F.pad(x, (weight.shape[-1]-1, 0)), weight[:, None], bias, groups=x.shape[1]))
    ns['causal_conv1d_fn'] = conv
    read('mamba/mamba_ssm/ops/triton/ssd_combined.py', ['mamba_split_conv1d_scan_ref'])
    ns['mamba_split_conv1d_scan_combined'] = lambda *a, seq_idx=None, **kw: ns['mamba_split_conv1d_scan_ref'](*a, **kw)
    if same_kernel:
        direct, _identity = backend_namespace('torch', 16)
        ns['DirectMamba'] = direct
    else:
        read('mamba/mamba_ssm/modules/direct_mamba.py', ['DirectMamba'])
    read('timm/layers/weight_init.py', ['_trunc_normal_', 'trunc_normal_', 'trunc_normal_tf_', 'variance_scaling_', 'lecun_normal_'], source='timm_mfemil')
    read('utils/survival_utils.py', ['nll_loss', 'NLLSurvLoss'])
    patches = [('from utils.survival_utils import NLLSurvLoss', '')]
    if profile != 'mammil_source':
        patches.append(('for idx, (layer, ppeg) in enumerate(zip(self.layers, self.ppeg)):',
                        'for idx, layer in enumerate(self.layers):\n            ppeg = self.ppeg[idx] if idx < len(self.ppeg) else None'))
    read('models/MamMIL2.py', ['_init_weights', 'segm_init_weights', 'MamMIL2'], patches)
    read('models/model_interface.py', ['ModelInterface'])
    ns['ClassificationInterface'] = ns['ModelInterface']
    read('models/model_interface_survival.py', ['ModelInterface'])
    ns['SurvivalInterface'] = ns['ModelInterface']
    read('MyOptimizer/radam.py', ['RAdam'])
    read('MyOptimizer/optim_factory.py', ['add_weight_decay'])
    return ns


def fixture(path, profile='mammil', survival=False, **options):
    dim = options.pop('dimension', 16)
    root = create_demo(path, patients=20, dimension=dim, survival=survival)
    if survival and profile != 'mammil_patient_union':
        for name in ['slides', 'features']:
            frame = pd.read_csv(root/(name+'.csv'))
            frame[frame.slide_id.str.endswith('_0')].to_csv(root/(name+'.csv'), index=False)
    bundle = Bundle(root)
    model = MamMIL(bundle, profile=profile, **{'hidden': 32, 'state_size': 8, 'scan_chunk': 16, **options})
    model.prepare(bundle, 2026, 'cpu')
    return model, bundle


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('length,chunk', [(1, 1), (13, 5), (65, 16)])
def test_independent_grouped_ssd_values_and_all_gradients(groups, length, chunk):
    torch.manual_seed(7)
    args = [torch.randn(2, length, 4, 3), torch.randn(2, length, 4), -torch.rand(4),
            torch.randn(2, length, groups, 5), torch.randn(2, length, groups, 5),
            torch.randn(4), torch.randn(2, length, 4, 3), torch.randn(4)]
    a, b = [[v.clone().requires_grad_() for v in args] for _ in range(2)]
    actual = ssd_scan(*a, checkpoint_chunk=chunk, dt_limit=(.01, .2))
    expected = reference()['ssd_selective_scan'](*b, dt_softplus=True, dt_limit=(.01, .2))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    weight = torch.randn_like(actual); (actual*weight).sum().backward(); (expected*weight).sum().backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, atol=6e-6, rtol=8e-5)


@pytest.mark.parametrize('before_gate', [False, True])
@pytest.mark.parametrize('group_size', [None, 4])
def test_gated_rms_source_equations_and_gradients(before_gate, group_size):
    args = [torch.randn(2, 7, 16), torch.randn(16), torch.randn(16), torch.randn(2, 7, 16)]
    a, b = [[v.clone().requires_grad_() for v in args] for _ in range(2)]
    x = gated_rms(*a, group_size=group_size, norm_before_gate=before_gate)
    y = reference()['rms_norm_ref'](*b, eps=1e-5, group_size=group_size, norm_before_gate=before_gate)
    torch.testing.assert_close(x, y, atol=8e-7, rtol=3e-6)
    x.square().sum().backward(); y.square().sum().backward()
    for p, q in zip(a, b):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize('norm_before_gate', [False, True])
def test_full_split_convolution_ssd_and_normalization_gradients(norm_before_gate):
    torch.manual_seed(18)
    heads, hd, state, groups, length = 4, 3, 5, 2, 13
    dim = heads*hd
    args = [torch.randn(1, length, 2*dim+2*groups*state+heads), torch.randn(dim+2*groups*state, 4),
            torch.randn(dim+2*groups*state), torch.randn(heads), -torch.rand(heads), torch.randn(heads)]
    a, b = [[v.clone().requires_grad_() for v in args] for _ in range(2)]
    wa, wb = [torch.randn(dim).requires_grad_() for _ in range(2)]; wb.data.copy_(wa.data)
    x = split_conv_scan(*a, chunk_size=256, headdim=hd, ngroups=groups, rmsnorm_weight=wa,
                         norm_before_gate=norm_before_gate, checkpoint_chunk=5)
    y = reference()['mamba_split_conv1d_scan_ref'](*b, chunk_size=256, headdim=hd, ngroups=groups,
                                                  rmsnorm_weight=wb, norm_before_gate=norm_before_gate)
    torch.testing.assert_close(x, y, atol=5e-6, rtol=1e-5)
    x.square().sum().backward(); y.square().sum().backward()
    for p, q in zip(a+[wa], b+[wb]):
        torch.testing.assert_close(p.grad, q.grad, atol=4e-5, rtol=3e-4)


@pytest.mark.parametrize('profile', ['mammil', 'mammil_source'])
@pytest.mark.parametrize('length', [1, 7, 20])
def test_full_network_source_output_and_active_gradients(tmp_path, profile, length):
    model, bundle = fixture(tmp_path/'data', profile)
    raw = reference(profile)['MamMIL2'](input_dim=16, embed_dim=32, layer_n=2, d_state=8, num_classes=2)
    raw.load_state_dict(model.network.state_dict())
    x, coords = torch.randn(length, 16), torch.stack([torch.arange(length), torch.zeros(length)], 1)
    arrays = build_graph(x.numpy(), coords.numpy())
    graph = Data(x=x, **{name: torch.from_numpy(value) for name, value in arrays.items()})
    label = torch.tensor([1]); state = rng_state(); a = model(graph, label)
    restore_rng(state); b = raw(data=graph, label=label)
    torch.testing.assert_close(a['logits'], b['logits'], atol=4e-6, rtol=3e-5)
    a['task_loss'].backward(); b['loss'].backward()
    for (name, p), (other, q) in zip(model.network.named_parameters(), raw.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None), name
        if p.grad is None:
            assert profile == 'mammil_source' and name.startswith('layers.1.'), name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=8e-6, rtol=8e-4)
    assert len(model.graph_manifest) == len(bundle.rows)


def test_scan_activation_and_source_loop_repair_are_explicit(tmp_path):
    counts = {}
    for profile in ['mammil', 'mammil_source']:
        model, bundle = fixture(tmp_path/profile, profile)
        called = [0, 0]
        for index, layer in enumerate(model.network.layers):
            def hook(module, args, output, index=index, called=called):
                called[index] += 1
            layer[1].register_forward_hook(hook)
        model.forward_unit(bundle, units(bundle, 'train')[0], 'cpu', {}, train=True)['task_loss'].backward()
        counts[profile] = called
    assert counts == {'mammil': [1, 1], 'mammil_source': [1, 0]}


@pytest.mark.parametrize('profile', ['mammil', 'mammil_source'])
def test_native_1024_features_512_hidden_128_state_all_active_gradients(tmp_path, profile):
    model, _ = fixture(tmp_path/'data', profile, dimension=1024, hidden=512, state_size=128)
    raw = reference(profile)['MamMIL2'](input_dim=1024, embed_dim=512, layer_n=2, d_state=128, num_classes=2)
    raw.load_state_dict(model.network.state_dict())
    x, coords = torch.randn(5, 1024), torch.stack([torch.arange(5), torch.zeros(5)], 1)
    arrays = build_graph(x.numpy(), coords.numpy())
    graph = Data(x=x, **{name: torch.from_numpy(value) for name, value in arrays.items()})
    state = rng_state(); actual = model(graph, torch.tensor([1]))
    restore_rng(state); expected = raw(data=graph, label=torch.tensor([1]))
    torch.testing.assert_close(actual['logits'], expected['logits'], atol=5e-6, rtol=3e-5)
    actual['task_loss'].backward(); expected['loss'].backward()
    for name, p in model.network.named_parameters():
        q = dict(raw.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-3)


def test_cosine_weighted_forest_and_disconnected_traversals(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 16)).astype('float32'); x[0] = 0.
    coords = np.c_[np.r_[np.arange(6), 1000+np.arange(6)], np.zeros(12)].astype('float32')
    arrays = build_graph(x, coords, neighbors=2, graph_seed=2024)
    graph = nx.Graph(); graph.add_nodes_from(range(12))
    weights = []
    for a, b in arrays['edge_index'].T:
        norms = np.linalg.norm(x[a])*np.linalg.norm(x[b])
        value = np.float32(1-np.dot(x[a], x[b])/norms) if norms else np.float32(1)
        weights.append(value)
        graph.add_edge(int(a), int(b), edge_weight=float(value))
    np.testing.assert_array_equal(arrays['edge_weight'], weights)
    tree = nx.minimum_spanning_tree(graph, weight='edge_weight')
    assert nx.number_connected_components(tree) == 2
    np.testing.assert_array_equal(arrays['dfs_preorder_index'], list(nx.dfs_preorder_nodes(tree)))
    np.testing.assert_array_equal(arrays['dfs_postorder_index'], list(nx.dfs_postorder_nodes(tree)))
    state = random.getstate(); random.seed(2024); bfs = []
    for nodes in nx.connected_components(tree):
        sub = tree.subgraph(nodes); root = random.choice(list(sub.nodes()))
        bfs.extend([root]+[b for _, b in nx.bfs_edges(sub, root)])
    random.setstate(state)
    np.testing.assert_array_equal(arrays['bfs_levelorder_index'], bfs)


@pytest.mark.parametrize('count', [1, 7, 25])
def test_graph_cache_freezes_assets_without_consuming_training_rng(tmp_path, count):
    x = np.random.default_rng(3).normal(size=(count, 16)).astype('float32')
    xy = np.c_[np.arange(count), np.zeros(count)].astype('float32')
    state = random.getstate()
    arrays, identity = graph_asset(x, xy, tmp_path, allow_build=True)
    assert random.getstate() == state
    other, expected = graph_asset(x, xy, tmp_path, expected=identity)
    assert identity == expected
    equal(arrays, other)
    for name in ['dfs_preorder_index', 'dfs_postorder_index', 'bfs_levelorder_index']:
        equal(arrays[name].sort().values, torch.arange(count))


def test_corrupt_or_missing_cache_and_changed_coordinates_are_rejected(tmp_path):
    rng = np.random.default_rng(5); x = rng.normal(size=(12, 16)).astype('float32'); coords = rng.normal(size=(12, 2)).astype('float32')
    arrays, identity = graph_asset(x, coords, tmp_path, allow_build=True)
    with pytest.raises(ValueError, match='builder identity changed'):
        graph_asset(x, coords+1, tmp_path, expected=identity)
    path = tmp_path/(identity['key']+'.npz')
    with np.load(path) as saved:
        data = {k: saved[k] for k in saved.files}
    data['edge_weight'] = data['edge_weight']+.1
    # Even a modified self-checksum cannot override the trusted checkpoint manifest.
    data['checksum'] = asset_checksum({k: data[k] for k in arrays})
    np.savez(path, **data)
    with pytest.raises(ValueError, match='differs from checkpoint'):
        graph_asset(x, coords, tmp_path, expected=identity)
    path.unlink()
    with pytest.raises(FileNotFoundError, match='asset missing'):
        graph_asset(x, coords, tmp_path, expected=identity)


@pytest.mark.parametrize('profile,survival', ROUTES)
def test_original_training_step_and_thirteen_radam_updates(tmp_path, profile, survival):
    model, bundle = fixture(tmp_path/'data', profile, survival)
    raw = reference(profile, same_kernel=True)
    net = raw['MamMIL2'](input_dim=16, embed_dim=32, layer_n=2, d_state=8,
                            num_classes=4 if survival else 2, survival=survival)
    net.load_state_dict(model.network.state_dict())
    opt = model.make_optimizer(1e-4, .05)
    other = raw['RAdam'](raw['add_weight_decay'](net, .05), lr=1e-4, weight_decay=0.)
    no_op = SimpleNamespace(update=lambda *a: None)
    instance = SimpleNamespace(model=net, hparams=SimpleNamespace(model=SimpleNamespace(name='MamMIL2')),
        log=lambda *a, **k: None, train_AUC=no_op, train_metrics=no_op, data=[{'count': 0, 'correct': 0} for _ in range(2)],
        train_risk_scores=[], train_censorship=[], train_event_time=[])
    groups = units(bundle, 'train')
    for index in range(13):
        rows = groups[index % len(groups)]
        graph = model.graph(bundle, rows, 'cpu')
        batch = {'data': graph, 'label': model.time_label([rows[0]['time']]) if survival else torch.tensor([rows[0]['target']]),
                 'c': torch.tensor([1-rows[0]['event'] if survival else 0], dtype=torch.float32),
                 'event_time': torch.tensor([rows[0]['time'] if survival else 0.])}
        state = rng_state(); other.zero_grad(set_to_none=True)
        interface = raw['SurvivalInterface' if survival else 'ClassificationInterface']
        loss = interface.training_step(instance, batch, index)['loss']; loss.backward(); other.step()
        restore_rng(state); mammil_epoch(model, bundle, [rows], opt, 'cpu', {})
        equal(model.network.state_dict(), net.state_dict())
        for p, q in zip(model.network.parameters(), net.parameters()):
            equal(opt.state[p], other.state[q])
    assert model.optimizer_steps == 13


@pytest.mark.parametrize('profile,survival', ROUTES)
def test_all_task_routes_exact_resume_and_reload(tmp_path, profile, survival):
    _model, bundle = fixture(tmp_path/'data', profile, survival)
    config = {'epochs': 2, 'threads': 1, 'model': {'hidden': 32, 'state_size': 8, 'scan_chunk': 16}}
    full = train(bundle.root, profile, tmp_path/'full', config=config)
    resumed = train(bundle.root, profile, tmp_path/'resumed', config=config, stop_after=1)
    train(bundle.root, profile, resumed, config=config, resume=True)
    for kind in ['last', 'best']:
        a, b = [torch.load(path/(kind+'.pt'), weights_only=False) for path in (full, resumed)]
        equal(a['model'], b['model']); equal(a['optimizers'], b['optimizers']); assert a['history'] == b['history']
    assert evaluate(full) == evaluate(resumed)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(resumed/'test_predictions.csv'))
    protocol = json.loads((full/('survival_protocol.json' if survival else 'method_protocol.json')).read_text())
    assert protocol['active_scans'] == (1 if profile == 'mammil_source' else 2)


def test_patient_union_offsets_edges_and_traversals_without_cross_slide_links(tmp_path):
    model, bundle = fixture(tmp_path/'data', 'mammil_patient_union', True)
    rows = next(g for g in units(bundle, 'train') if len(g) == 2)
    graph = model.graph(bundle, rows, 'cpu')
    first = len(bundle.load(min(rows, key=lambda r: r['slide_id']))[0])
    assert torch.equal(graph.edge_index[0] < first, graph.edge_index[1] < first)
    for name in ['dfs_preorder_index', 'dfs_postorder_index', 'bfs_levelorder_index']:
        equal(getattr(graph, name).sort().values, torch.arange(len(graph.x)))
    with pytest.raises(ValueError, match='one WSI per patient'):
        MamMIL(bundle)


def test_real_cli_family_name(tmp_path):
    root = create_demo(tmp_path/'data', patients=20)
    config = tmp_path/'config.json'; config.write_text(json.dumps({'epochs': 1, 'threads': 1, 'model': {'hidden': 32, 'state_size': 8, 'scan_chunk': 16}}))
    for args in [['prepare-mammil-graphs', '--data', str(root), '--output', str(tmp_path/'graphs.json')],
                 ['check', '--data', str(root), '--method', 'MamMIL', '--deep'],
                 ['run', '--data', str(root), '--method', 'MamMIL', '--config', str(config), '--output', str(tmp_path/'run')],
                 ['evaluate', '--run', str(tmp_path/'run')]]:
        result = subprocess.run([sys.executable, '-m', 'milplus', *args], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['n'] > 0
    manifest = json.loads((tmp_path/'graphs.json').read_text())
    checkpoint = torch.load(tmp_path/'run'/'last.pt', weights_only=False)
    assert checkpoint['model']['_extra_state']['graphs'] == manifest['graphs']


@pytest.mark.parametrize('options', [{'hidden': 16}, {'layers': 1}, {'state_size': 0}, {'scan_chunk': 0},
                                   {'neighbors': 0}, {'dropout': float('nan')}, {'backend': 'auto'}])
def test_invalid_model_options(tmp_path, options):
    with pytest.raises(ValueError):
        fixture(tmp_path/'data', **options)


def test_truncation_and_backend_identity_change_fail(tmp_path):
    model, _ = fixture(tmp_path/'data')
    with pytest.raises(ValueError, match='max_instances'):
        model.configure_training({'max_instances': 2})
    extra = model.get_extra_state(); extra['backend'] = {'name': 'changed'}
    with pytest.raises(ValueError, match='numerical backend'):
        model.set_extra_state(extra)

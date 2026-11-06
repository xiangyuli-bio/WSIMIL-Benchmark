"""Separate same-name methods; compare author mechanisms and full recipes."""
import copy
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.nn import functional as F

from milplus.channel_camil import ChannelCAMIL, author_radam, channel_epoch, load_channel_camil
from milplus.context_camil import ContextCAMIL, KerasAdam, cached_neighbors, camil_neighbors
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.io import digest_file, write_json
from milplus.registry import check_compatibility, require_runnable, resource
from milplus.upstream import load_definitions, source_root, verified_source


def source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch pinned '+name+' source')
    torch.set_num_threads(1)


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('gated', [False, True])
def test_channel_native_four_blocks_author_output_loss_and_gradients(dim, gated):
    source('camil_channel')
    model = ChannelCAMIL(dim, 3, gated=gated)
    reference = load_channel_camil().CAMIL(3, input_dim=dim, temperature=1.2, dropout=.15, n_layers=4, gate=gated)
    reference.load_state_dict(model.network.state_dict())
    x, y = torch.randn(19, dim), torch.tensor([2])
    torch.manual_seed(23); actual = model(x)['logits']
    torch.manual_seed(23); expected = reference(data=x[None])['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss, ref_loss = F.cross_entropy(actual, y), F.cross_entropy(expected, y)
    loss.backward(); ref_loss.backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), reference.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    active = ['layers.0.mcab.cam.linear.0.weight', 'layers.0.mcab.local_feature3.0.weight',
              'layers.0.mcab.local_feature5.0.weight', 'layers.3.trans_layer.attn.to_qkv.weight']
    for name in active:
        assert dict(model.network.named_parameters())[name].grad.abs().sum() > 0


def test_channel_mcab_zero_padding_three_sigmoid_gates_and_order():
    source('camil_channel')
    block = load_channel_camil().MCAB(dim=16, temperature=1.2)
    x = torch.randn(1, 11, 16)
    a = F.pad(x, (0, 0, 0, 5)).transpose(1, 2).reshape(1, 16, 4, 4)
    b, c = block.local_feature3(a), block.local_feature5(a)
    h = a+b+c
    gate = torch.sigmoid((block.cam.linear(h.amax((2, 3)))+block.cam.linear(h.mean((2, 3))))*1.2)
    ga, gb, gc = [v[:, :, None, None] for v in gate.chunk(3, -1)]
    expected = (a*ga+b*gb+c*gc).flatten(2).transpose(1, 2)[:, :11]
    torch.testing.assert_close(block(x), expected, atol=0, rtol=0)
    assert not torch.allclose(ga+gb+gc, torch.ones_like(ga))  # not softmax competition
    order = torch.randperm(11)
    assert not torch.allclose(block(x[:, order]), block(x)[:, order])


def test_channel_author_radam_and_lookahead_thirteen_updates():
    source('camil_channel')
    model = ChannelCAMIL(16, 3, hidden=32, dropout=0.)
    ref = copy.deepcopy(model.network)
    opt = model.make_optimizer(.002, .01)
    groups = load_definitions('camil_channel', 'MyOptimizer/optim_factory.py', ['add_weight_decay'], {})
    ref_fast = author_radam()(groups.add_weight_decay(ref, .01), lr=.002, weight_decay=0.)
    raw = load_definitions('camil_channel', 'MyOptimizer/lookahead.py', ['Lookahead'],
                           {'torch': torch, 'Optimizer': torch.optim.Optimizer, 'defaultdict': defaultdict})
    ref_opt = raw.Lookahead(ref_fast)
    for step in range(13):
        opt.zero_grad(); ref_fast.zero_grad()
        # Controlled external gradients isolate the optimizer equation and all
        # parameters, including the exact sixth/twelfth-step slow updates.
        for p, q in zip(model.network.parameters(), ref.parameters()):
            gradient = torch.randn_like(p)*.01
            p.grad = gradient.clone(); q.grad = gradient.clone()
        opt.step(); model.after_step(); ref_opt.step()
        for p, q in zip(model.network.parameters(), ref.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        if step < 5:
            assert not model.slow
    assert model.steps == 13
    assert all(v['step'] == 13 for v in opt.state.values())
    for (name, _), p in zip(model.network.named_parameters(), ref.parameters()):
        torch.testing.assert_close(model.slow[name], ref_opt.state[p]['slow_buffer'], atol=0, rtol=0)


def test_channel_gradient_accumulation_and_final_short_window(tmp_path):
    source('camil_channel')
    root = create_demo(tmp_path/'data', patients=20)
    bundle = Bundle(root)
    model = ChannelCAMIL(16, 2, hidden=32, dropout=0., accumulation=2)
    reference = copy.deepcopy(model)
    opt, ref_opt = [m.make_optimizer(.001, .01) for m in (model, reference)]
    groups = units(bundle, 'train')[:5]
    config = {'max_instances': None, 'grad_clip': 100.}
    torch.manual_seed(182); channel_epoch(model, bundle, groups, opt, 'cpu', config)
    torch.manual_seed(182); ref_opt.zero_grad()
    for i, rows in enumerate(groups):
        x, _ = bundle.load(rows[0])
        (F.cross_entropy(reference(x)['logits'], torch.tensor([rows[0]['target']]))/2).backward()
        if i in [1, 3, 4]:
            nn.utils.clip_grad_norm_(reference.parameters(), 100.)
            ref_opt.step(); reference.after_step(); ref_opt.zero_grad()
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), atol=0, rtol=0)
    assert model.steps == 3


@pytest.mark.parametrize('n', [2, 7, 19, 33])
def test_context_graph_author_physical_distance_and_cosine_normalization(n):
    source('camil_context')
    rng = np.random.default_rng(98)
    coords = rng.normal(size=(n, 2))*2000
    x = rng.normal(size=(n, 16))
    x[0] = 0.  # author NaN cosine is converted to zero by the loader
    ids = np.argsort(cdist(coords, coords), axis=1)[:, :16]
    values = np.array([[cdist(x[i:i+1], x[j:j+1], 'cosine')[0, 0] for j in row] for i, row in enumerate(ids)])
    values = np.nan_to_num(values)[:, 1:]
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    similarity = np.exp(-np.divide(values, norm, out=np.zeros_like(values), where=norm > 0))
    weights = np.concatenate([similarity.max(1, keepdims=True), similarity], 1)[:, :8]
    actual_ids, actual_weights = camil_neighbors(x, coords, chunk=3)
    np.testing.assert_array_equal(actual_ids, ids[:, :8])
    np.testing.assert_allclose(actual_weights, weights, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(camil_neighbors(x, coords, chunk=19)[0], actual_ids)


def test_context_dense_vs_sparse_neighbor_attention_and_singleton():
    source('camil_context')
    model = ContextCAMIL(16, 3, hidden=64, dim_head=8, landmarks=16)
    x = torch.randn(19, 16); coords = torch.randn(19, 2)
    ids, weights = camil_neighbors(x.numpy(), coords.numpy())
    ids, weights = torch.from_numpy(ids), torch.from_numpy(weights)
    output = model.forward_edges(x, ids, weights)
    dense = model.project(x); global_x = model.nystrom(dense[None])[0]+dense
    pair_scores = model.query(global_x) @ model.key(global_x).T / 16
    adjacency = torch.zeros(19, 19).scatter(1, ids, weights)
    raw = (adjacency*pair_scores).sum(1)
    torch.testing.assert_close(output['neighbor_scores'], raw, atol=3e-7, rtol=1e-5)
    local = raw.softmax(0)[:, None]*model.value(dense)
    gate = torch.sigmoid(-local).square()
    combined = 2*local*gate+2*global_x*(1-gate)
    attn = model.pool_w(model.pool_v(combined).tanh()*model.pool_u(combined).sigmoid()).softmax(0)
    expected = model.classifier((attn*combined).sum(0, keepdim=True))
    torch.testing.assert_close(output['logits'], expected, atol=3e-7, rtol=1e-5)
    F.cross_entropy(output['logits'], torch.tensor([2])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(model(x[:1], coords[:1])['logits']).all()
    with pytest.raises(ValueError, match='physical coordinates'):
        model(x)
    with pytest.raises(ValueError, match='distinct'):
        model(x, coords[:1].expand(19, 2))


def test_context_graph_cache_invalidation_and_corruption(tmp_path, monkeypatch):
    import milplus.context_camil as module
    rng = np.random.default_rng(58)
    x, coords = rng.normal(size=(19, 16)), rng.normal(size=(19, 2))
    calls = []
    original = module.camil_neighbors
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, 'camil_neighbors', counted)
    a = cached_neighbors(x, coords, tmp_path)
    original_path = next(tmp_path.glob('*.npz'))
    b = cached_neighbors(x, coords, tmp_path)
    assert len(calls) == 1
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
    cached_neighbors(x+.1, coords, tmp_path)
    cached_neighbors(x, coords+.2, tmp_path)
    assert len(calls) == 3
    with np.load(original_path) as data:
        values = dict(data)
    values['weights'] = values['weights']+.2
    np.savez(original_path, **values)
    # Locate the original cache explicitly: changed-feature/coordinate keys
    # also retain valid independent results.
    failures = 0
    for xx, xy in [(x, coords), (x+.1, coords), (x, coords+.2)]:
        try:
            cached_neighbors(xx, xy, tmp_path)
        except ValueError as error:
            assert 'cache is invalid' in str(error)
            failures += 1
    assert failures == 1


def test_context_regular_tile_grid_matches_author_preprocessor_with_distance_ties():
    source('camil_context')
    module = load_definitions('camil_context', 'feature_extractor/compute_feats.py', ['generate_values_resnet'],
                             {'np': np, 'distance': distance, 'pairwise_distances': pairwise_distances})
    coords = np.array([(x*256, y*256) for x in range(5) for y in range(5)])
    x = np.random.default_rng(67).normal(size=(25, 16)).astype(np.float32)
    _, distances, expected_ids = module.generate_values_resnet(x, coords)
    val = distances[:, 1:]
    similarity = np.exp(-val/np.linalg.norm(val, axis=1, keepdims=True))
    expected_weights = np.concatenate([similarity.max(1, keepdims=True), similarity], 1)[:, :8]
    ids, weights = camil_neighbors(x, coords)
    np.testing.assert_array_equal(ids, expected_ids[:, :8])
    np.testing.assert_allclose(weights, expected_weights, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('classes,n', [(2, 19), (3, 256), (3, 257)])
def test_context_full_tensorflow_author_forward_loss_gradients_optimizer_and_scheduler(tmp_path, classes, n):
    source('camil_context')
    python = Path(os.environ.get('MILPLUS_TF_PYTHON', '/root/milplus_validation_envs/camil_tf/bin/python'))
    if not python.is_file():
        pytest.skip('Set MILPLUS_TF_PYTHON to isolated camil-tf-requirements.txt environment')
    torch.manual_seed(112)
    model = ContextCAMIL(512, classes)
    x = (torch.randn(n, 512)*.3).requires_grad_()
    coords = torch.randn(n, 2)
    ids, weights = camil_neighbors(x.detach().numpy(), coords.numpy())
    np.savez(tmp_path/'input.npz', x=x.detach().numpy(), indices=ids, edge_weights=weights,
             classes=classes, target=classes-1, **{name: p.detach().numpy() for name, p in model.named_parameters()})
    author, commit = verified_source('camil_context')
    script = Path(__file__).resolve().parents[1]/'scripts/camil_tf_reference.py'
    process = subprocess.run([str(python), str(script), '--source', str(author), '--commit', commit,
                    '--input', str(tmp_path/'input.npz'), '--output', str(tmp_path/'reference.npz')],
                    check=False, env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '2', 'TF_ENABLE_ONEDNN_OPTS': '0',
                         'OMP_NUM_THREADS': '1'}, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr
    ref = np.load(tmp_path/'reference.npz')
    out = model.forward_edges(x, torch.from_numpy(ids), torch.from_numpy(weights))
    loss = F.cross_entropy(out['logits'], torch.tensor([classes-1])); loss.backward()
    probability = out['logits'].softmax(-1).detach().numpy()
    np.testing.assert_allclose(probability[:, 1:] if classes == 2 else probability, ref['probability'], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(out['neighbor_scores'].detach(), ref['raw'], atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(out['attention'].T.detach(), ref['attention'], atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(loss.detach(), ref['loss'], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(x.grad, ref['input_gradient'], atol=2e-6, rtol=3e-4)
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
        np.testing.assert_allclose(p.grad, ref['grad.'+name], atol=2e-6, rtol=4e-4, err_msg=name)
    assert float(ref['declared_but_unused_l2']) > 0
    assert bool(ref['raw_1d_softmax_rejected']) and bool(ref['raw_singleton_squeeze_rejected'])
    variable = nn.Parameter(torch.tensor([1., -2., 3.]))
    optimizer = KerasAdam([variable], lr=.01)
    for i, values in enumerate([[1e-5, .1, -.2], [.02, -.1, 1e-6], [.4, .5, -.6]]):
        variable.grad = torch.tensor(values); optimizer.step()
        np.testing.assert_allclose(variable.detach(), ref['adam_updates'][i], atol=3e-7, rtol=1e-6)
    opt = model.make_optimizer(.01, 0.)
    for i in range(24):
        model.after_validation({'log_loss': 1.})
        assert opt.param_groups[0]['lr'] == pytest.approx(ref['plateau_lrs'][i])
    write_json(tmp_path/'camil_comparison.json', {
        'state': 'passed', 'classes': classes, 'instances': n, 'commit': commit,
        'tensorflow_version': str(ref['tensorflow_version']), 'tensorflow_addons_version': str(ref['tensorflow_addons_version']),
        'input_sha256': digest_file(tmp_path/'input.npz'), 'reference_sha256': digest_file(tmp_path/'reference.npz'),
        'loss_max_absolute_error': float(abs(float(loss.detach())-float(ref['loss']))),
        'parameter_gradient_max_absolute_error': {name: float(np.max(np.abs(p.grad.numpy()-ref['grad.'+name])))
                                                  for name, p in model.named_parameters()},
        'input_gradient_max_absolute_error': float(np.max(np.abs(x.grad.numpy()-ref['input_gradient']))),
        'optimizer_steps_compared': 3, 'scheduler_epochs_compared': 24,
        'raw_1d_softmax_rejected': True, 'raw_singleton_squeeze_rejected': True,
        'scope': 'synthetic CPU TensorFlow equation comparison with documented compatibility patches'})


@pytest.mark.parametrize('method', ['camil_channel', 'camil_context'])
def test_camil_exact_resume_software_and_input_checks(tmp_path, method):
    source(method)
    root = create_demo(tmp_path/'data', patients=20)
    opts = {'hidden': 32} if method == 'camil_channel' else {'hidden': 64, 'dim_head': 8, 'landmarks': 16}
    cfg = {'epochs': 3, 'threads': 1, 'model': opts}
    full = train(root, method, tmp_path/'full', config=cfg)
    partial = train(root, method, tmp_path/'partial', config=cfg, stop_after=1)
    train(root, method, partial, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    assert a['history'] == b['history']
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    torch.testing.assert_close(a['optimizers'], b['optimizers'], atol=0, rtol=0)
    assert evaluate(full) == evaluate(partial)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(partial/'test_predictions.csv'))
    bundle = Bundle(root); bundle.config['training_unit'] = 'patient'
    with pytest.raises(ValueError, match='individual slide'):
        check_compatibility(require_runnable(method), bundle)
    feat = root/pd.read_csv(root/'features.csv').iloc[0].path
    with h5py.File(feat, 'r+') as f:
        key = 'coords' if method == 'camil_context' else 'features'
        f[key][:] = f[key][:]+.2
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(full)


def test_camil_names_and_source_identity_are_distinct():
    assert require_runnable('camil_channel')['family'] == 'camil_channel'
    assert require_runnable('camil_context')['family'] == 'camil_context'
    a, b = [resource('sources.lock.json')[key] for key in ['camil_channel', 'camil_context']]
    assert a['url'] != b['url'] and a['commit'] != b['commit']
    with pytest.raises(ValueError, match='Unknown/ambiguous'):
        require_runnable('CAMIL')

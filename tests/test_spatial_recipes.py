"""Author parity, spatial provenance, masked batches, ranking and recovery."""
import copy
from collections import defaultdict

import numpy as np
import pytest
import torch
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Delaunay, KDTree
from sklearn.neighbors import KDTree as SKTree
from torch.nn.attention import SDPBackend

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, require_runnable
from milplus.spatial_attention import (
    SM_METHODS,
    ProbabilisticSpatialMIL,
    SmoothMIL,
    load_psamil,
    padded_bags,
    smooth_adjacency,
    spatial_epoch,
)
from milplus.upstream import load_definitions, source_root


def require_source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch the pinned '+name+' author source')
    if name == 'slidegraph':
        pytest.importorskip('torch_geometric')
    torch.set_num_threads(1)


def bags():
    torch.manual_seed(44)
    points = torch.tensor([[0, 0], [256, 0], [512, 0], [0, 256], [256, 256], [512, 256]])
    return [(torch.randn(n, 16), points[:n]) for n in (4, 6)]


def author_smooth():
    sm = load_definitions('smmil', 'code/models/modules/Sm.py', ['ApproxSm', 'ExactSm'], {'torch': torch})
    scope = {'torch': torch, 'ApproxSm': sm.ApproxSm, 'ExactSm': sm.ExactSm}
    pool = load_definitions('smmil', 'code/models/modules/MILAttentionPool.py', ['MILAttentionPool'], scope)
    scope['SDP_BACKEND'] = [SDPBackend.MATH]
    tr = load_definitions('smmil', 'code/models/modules/MILTransformer.py',
        ['MultiheadSelfAttention', 'Identity', 'TransformerEncoderLayer', 'TransformerEncoder'], scope)
    return load_definitions('smmil', 'code/models/DeepMILAttModel.py', ['DeepMILAttModel'],
        {'torch': torch, 'MILAttentionPool': pool.MILAttentionPool, 'TransformerEncoder': tr.TransformerEncoder}).DeepMILAttModel


@pytest.mark.parametrize('profile', sorted(SM_METHODS))
def test_sm_author_probabilities_loss_and_gradients(profile):
    require_source('smmil')
    model = SmoothMIL(16, 2, 256, profile=profile, hidden=32, attention_dim=8)
    base = author_smooth()(input_dim=16, emb_dim=32,
        transformer_encoder_kwargs=model.network.transformer_encoder_kwargs,
        pool_kwargs=model.network.pool_kwargs)
    base.load_state_dict(model.network.state_dict())
    x, c = bags()[1]
    adjacency = smooth_adjacency(x, c, 256).unsqueeze(0)
    expected, _ = base(x[None], adjacency, torch.ones(1, len(x), dtype=torch.bool), return_att=True)
    actual = model(x, c, torch.tensor([1]))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(expected, torch.ones(1))
    torch.testing.assert_close(actual['binary_logit'], expected)
    torch.testing.assert_close(actual['total_loss'], loss)
    loss.backward(); actual['total_loss'].backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), base.named_parameters()):
        if p.grad is None or q.grad is None:
            assert p.grad is q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)


def test_sm_adjacency_equations_and_isolated_nodes():
    x, c = bags()[1]
    distances = torch.cdist(c.float(), c.float())
    weights = torch.exp(-torch.cdist(x, x)/x.shape[1])*(distances <= np.sqrt(2)*256)
    weights.fill_diagonal_(0)
    inv = weights.sum(1).rsqrt()
    torch.testing.assert_close(smooth_adjacency(x, c, 256).to_dense(), inv[:, None]*weights*inv[None])
    assert smooth_adjacency(x[:1], c[:1], 256).to_dense().item() == 1
    isolated = smooth_adjacency(x[:2], torch.tensor([[0, 0], [9999, 9999]]), 256)
    assert isolated._nnz() == 0


@pytest.mark.parametrize('profile', sorted(SM_METHODS))
def test_sm_masked_batch_matches_unpadded_predictions(profile):
    require_source('smmil')
    model = SmoothMIL(16, 2, 256, profile=profile, hidden=32, attention_dim=8).eval()
    samples = bags()
    batched = model.batch(samples)['logits']
    separate = torch.cat([model(x, c)['logits'] for x, c in samples])
    torch.testing.assert_close(batched, separate, atol=2e-6, rtol=2e-5)


def test_sm_exact_solver_and_transformer_eval_dropout():
    require_source('smmil')
    model = SmoothMIL(16, 2, 256, mode='exact', hidden=32)
    x, c = bags()[1]
    out = model(x, c, torch.tensor([0]))
    out['total_loss'].backward()
    assert torch.isfinite(out['total_loss'])
    model.max_dense_bytes = 1
    with pytest.raises(ValueError, match='max_dense_bytes'):
        model(x, c)
    tr = SmoothMIL(16, 2, 256, profile='smmil_transformer', hidden=32, transformer_dropout=.4).eval()
    torch.testing.assert_close(tr(x, c)['logits'], tr(x, c)['logits'], atol=0, rtol=0)


def test_psa_author_weighted_objective_decay_gradients_and_padding():
    require_source('psamil')
    model = ProbabilisticSpatialMIL(16, 3, 256, hidden=8, attention_chunk=2)
    model.class_weights.copy_(torch.tensor([.5, .25, .125]))
    author = copy.deepcopy(model.network)
    samples = bags()
    x, coords, mask = padded_bags(samples)
    labels = torch.tensor([0, 2])
    expected = author(x, coords/256, mask)
    output = model.batch(samples)
    torch.testing.assert_close(output['logits'], expected)
    weights = model.class_weights[labels]
    loss = (torch.nn.functional.cross_entropy(expected, labels, reduction='none')*weights).sum()/weights.sum()
    loss = loss-author.alpha*author.diversity_loss()
    actual = model.objective(output, labels)
    torch.testing.assert_close(loss, actual)
    loss.backward(); actual.backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), author.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad)
        if name.endswith('lambda_p'):
            assert p.grad.abs().sum() > 0
    separate = torch.cat([model(a, b)['logits'] for a, b in samples])
    torch.testing.assert_close(output['logits'], separate, atol=2e-6, rtol=2e-5)
    optimizer = model.make_optimizer(.001, .0001)
    assert [group['lr'] for group in optimizer.param_groups] == [.001, .1]
    ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert ids == {id(p) for p in model.parameters()}
    model.max_distance_bytes = 1
    with pytest.raises(ValueError, match='distance matrix'):
        model.batch(samples)


@pytest.mark.parametrize('method', ['smmil_early', 'psamil', 'slidegraph'])
def test_spatial_metadata_and_patient_frames_required(tmp_path, method):
    root = create_demo(tmp_path/'data', patients=20)
    config = yaml.safe_load((root/'task.yaml').read_text())
    spatial = config['encoders']['toy'].pop('spatial')
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='spatial'):
        check_compatibility(require_runnable(method), Bundle(root))
    config['encoders']['toy']['spatial'] = spatial
    config['training_unit'] = 'patient'
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='individual slide'):
        check_compatibility(require_runnable(method), Bundle(root))


def test_graph_clustering_matches_author_procedure_and_delaunay():
    require_source('slidegraph')
    from milplus.slidegraph import cluster_graph
    torch.manual_seed(88)
    x = torch.randn(25, 16)
    points = np.asarray([(a*1000, b*1000) for a in range(5) for b in range(5)])
    # Literal independent reference for the procedural source's condensed distance.
    raw = x.numpy()
    selected = raw[:, np.ptp(raw, axis=0) > 1e-4]
    indices, distances = SKTree(points).query_radius(points, r=2000, return_distance=True, sort_results=True)
    parts = []
    for i in range(len(points)-1):
        d = np.ones(len(points))
        d[indices[i]] = 1-np.exp(-.001*np.linalg.norm(selected[i]-selected[indices[i]], axis=1))*np.exp(-.003*distances[i])
        parts.append(d[i+1:])
    ids = fcluster(linkage(np.concatenate(parts), method='average'), .8, criterion='distance')
    centers = np.asarray([np.round(points[ids == i].mean(0)) for i in np.unique(ids)])
    features = np.asarray([raw[ids == i].mean(0) for i in np.unique(ids)])
    connect = load_definitions('slidegraph', 'features_to_graph.py', ['connectClusters'],
        {'np': np, 'Delaunay': Delaunay, 'KDTree': KDTree, 'defaultdict': defaultdict},
        replacements=(('dtype = np.int)', 'dtype = int)'),)).connectClusters
    graph = cluster_graph(x, torch.tensor(points))
    np.testing.assert_allclose(graph.coords.numpy(), centers)
    np.testing.assert_allclose(graph.x.numpy(), features)
    expected_edges = {tuple(p) for p in np.argwhere(connect(centers, 4000) > 0)}
    assert {tuple(p) for p in graph.edge_index.numpy().T} == expected_edges
    with pytest.raises(ValueError, match='quadratic'):
        cluster_graph(x, torch.tensor(points), max_pair_bytes=1)


def test_graph_ranking_loss_and_batches_preserve_all_samples():
    require_source('slidegraph')
    from milplus.slidegraph import connect_centroids, ranking_batches, ranking_loss
    scores = torch.tensor([.1, .8, .3, -.1], requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0])
    actual = ranking_loss(scores, labels)
    expected = torch.stack([torch.relu(1-(labels[i]-labels[j])*(scores[i]-scores[j]))
        for i in range(4) for j in range(i+1, 4) if labels[i] != labels[j]]).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, scores)[0], torch.autograd.grad(expected, scores)[0])
    groups = [[{'target': int(i > 12), 'index': i}] for i in range(17)]
    batches = ranking_batches(groups, 4, 2026)
    assert sorted(row[0]['index'] for batch in batches for row in batch) == list(range(17))
    assert all({r[0]['target'] for r in batch} == {0, 1} for batch in batches)
    assert connect_centroids(np.array([[0, 0]]), 10).shape == (2, 0)
    assert connect_centroids(np.array([[0, 0], [2, 0], [4, 0]]), 3).shape == (2, 4)


def test_graph_train_only_normalization_platt_and_author_network(tmp_path):
    require_source('slidegraph')
    from torch_geometric.data import Batch

    from milplus.slidegraph import author_graph_class, author_platt_class
    root = create_demo(tmp_path/'data', patients=20)
    bundle = Bundle(root)
    # A sentinel ensures prepare/calibration cannot silently touch validation/test.
    original = bundle.load
    def guarded(row, **kwargs):
        assert row['subset'] == 'train'
        return original(row, **kwargs)
    bundle.load = guarded
    model = build_bundle_model('slidegraph', bundle)
    model.prepare(bundle, 2026, 'cpu')
    groups = units(bundle, 'train')
    x = torch.cat([model.graph(bundle, rows).x for rows in groups])
    torch.testing.assert_close(model.feature_mean, x.mean(0))
    torch.testing.assert_close(model.feature_scale, x.std(0))
    model.eval()
    author = author_graph_class()(16, 1, layers=[16, 16, 8], dropout=.5,
                                  pooling='mean', conv='EdgeConv', gembed=False, aggr='max').eval()
    author.load_state_dict(model.network.state_dict())
    batch = Batch.from_data_list([model.graph(bundle, rows) for rows in groups[:3]])
    batch.x = (batch.x-model.feature_mean)/model.feature_scale
    expected = author(batch)[0].flatten()
    output = model.batch(bundle, groups[:3], 'cpu')
    torch.testing.assert_close(output['score'], expected)
    from milplus.slidegraph import ranking_loss
    paired_labels = torch.tensor([rows[0]['target'] for rows in groups[:3]])
    ranking_loss(output['score'], paired_labels).backward()
    ranking_loss(expected, paired_labels).backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), author.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad)
    labels = [rows[0]['target'] for rows in groups]
    scores = [float(model.batch(bundle, [rows], 'cpu')['score'].item()) for rows in groups]
    fitted = author_platt_class()().fit(labels, scores)
    model.finish_epoch(bundle, 'cpu', {})
    np.testing.assert_allclose(model.calibration.numpy(), [fitted.A, fitted.B])
    output = model.batch(bundle, groups[:3], 'cpu')
    np.testing.assert_allclose(output['probabilities'][:, 1].detach(), fitted.transform(np.asarray(scores[:3])), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('method', sorted(SM_METHODS) + ['psamil', 'slidegraph'])
def test_spatial_full_train_resume_and_reload(tmp_path, method):
    require_source('smmil' if method in SM_METHODS else method)
    root = create_demo(tmp_path/'data', patients=20)
    options = {} if method == 'slidegraph' else {'hidden': 16, 'batch_size': 3}
    config = {'epochs': 2, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=config)
    interrupted = train(root, method, tmp_path/'resumed', config=config, stop_after=1)
    train(root, method, interrupted, config=config, resume=True)
    a = torch.load(full/'last.pt', weights_only=False)
    b = torch.load(interrupted/'last.pt', weights_only=False)
    assert a['history'] == b['history']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    assert evaluate(full) == evaluate(interrupted)


@pytest.mark.parametrize('method', ['smmil_early', 'psamil'])
def test_spatial_class_weights_and_schedules(tmp_path, method):
    require_source('smmil' if method.startswith('smmil') else method)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    options = {'hidden': 16, 'batch_size': 3}
    if method.startswith('smmil'):
        options['balance_loss'] = True
    model = build_bundle_model(method, bundle, options)
    model.prepare(bundle, 2026, 'cpu')
    counts = np.bincount(bundle.rows.query("subset == 'train'").target)
    if method == 'psamil':
        np.testing.assert_allclose(model.class_weights.numpy(), 1/counts)
        optimizer = model.make_optimizer(.002, .0001)
    else:
        assert model.pos_weight.item() == counts[0]/counts[1]
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    cfg = {'epochs': 10, 'lr': .002, 'weight_decay': .0001, 'max_instances': None, 'grad_clip': 5.}
    groups = units(bundle, 'train')
    model.start_epoch(1)
    logs = spatial_epoch(model, bundle, groups, optimizer, 'cpu', cfg)
    if method == 'psamil':
        num_batches = int(np.ceil(len(groups)/3))
        reference = load_psamil()[1].piecewise_linear([
            {'from': .00002, 'to': .002, 'steps': .1}, {'from': .002, 'to': .002, 'steps': .9},
            {'from': .002, 'to': .00002, 'steps': -1}], num_batches*10)
        np.testing.assert_allclose([l['lr'] for l in logs], reference[num_batches:num_batches*2])
        assert optimizer.param_groups[1]['lr'] == 100*optimizer.param_groups[0]['lr']
    else:
        assert logs[0]['lr'] == pytest.approx(.002*.28)

"""Real scale contracts, pinned-author equations, dedicated recipes and exact recovery."""
import copy
import random

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.io import read_json
from milplus.models import build_bundle_model
from milplus.multiscale import (
    child_indices,
    csmil_epoch,
    hagmil_epoch,
    load_hagmil,
    stable_masked_softmax,
)
from milplus.registry import check_compatibility, require_runnable
from milplus.stochastic import inference_rng
from milplus.upstream import load_definitions, source_root


def source(method):
    name = 'hagmil' if method == 'hagmil_ce' else method
    if not source_root(name).is_dir():
        pytest.skip(f'Fetch author source with milplus setup --method {method}')
    if name == 'hagmil':
        pytest.importorskip('admin_torch')
        pytest.importorskip('nystrom_attention')
    torch.set_num_threads(1)


def fixture(tmp_path, method, dimension=16):
    source(method)
    kind = 'aligned_centers' if method == 'csmil' else 'quadtree'
    root = create_demo(tmp_path/'data', patients=20, dimension=dimension, multiscale=kind)
    options = {'clusters': 3} if method == 'csmil' else {
        'hidden_dims': [32, 48, 32, 32], 'k_sample': 2, 'train_parents': 4, 'eval_parents': 4}
    bundle = Bundle(root)
    model = build_bundle_model(method, bundle, options)
    if hasattr(model, 'prepare'):
        model.prepare(bundle, 2026, 'cpu')
    return root, bundle, model, options


@pytest.mark.parametrize('kind', ['aligned_centers', 'quadtree'])
def test_multiscale_inventory_hashes_every_scale_and_parent(tmp_path, kind):
    root = create_demo(tmp_path/'data', patients=20, multiscale=kind)
    bundle = Bundle(root)
    before = bundle.check(deep=True)['feature_inventory_sha256']
    row = bundle.records('test')[0]
    path = root/bundle.multiscale.rows(row)[-1]['path']
    with h5py.File(path, 'r+') as f:
        f['features'][0, 0] += 1
    assert bundle.check(deep=True)['feature_inventory_sha256'] != before
    if kind == 'quadtree':
        parent_file = root/bundle.multiscale.rows(row)[-1]['parent_path']
        parents = np.load(parent_file)
        parents[0] = -1
        np.save(parent_file, parents)
        with pytest.raises(ValueError, match='Parent indices'):
            bundle.check(deep=True)


@pytest.mark.parametrize('kind', ['aligned_centers', 'quadtree'])
def test_scale_center_mismatch_is_rejected(tmp_path, kind):
    root = create_demo(tmp_path/'data', patients=20, multiscale=kind)
    bundle = Bundle(root)
    row = bundle.records('train')[0]
    path = root/bundle.multiscale.rows(row)[1]['path']
    with h5py.File(path, 'r+') as f:
        f['coords'][0, 0] += 2
    with pytest.raises(ValueError, match='centers'):
        bundle.multiscale.load(row)


def test_quadtree_duplicate_quadrant_and_false_fov_rejected(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, multiscale='quadtree')
    bundle = Bundle(root)
    path = root/bundle.multiscale.rows(bundle.records('train')[0])[1]['path']
    with h5py.File(path, 'r+') as f:
        f['coords'][0] = f['coords'][1]
    with pytest.raises(ValueError, match='Duplicate quadtree'):
        bundle.check(deep=True)
    config = yaml.safe_load((root/'task.yaml').read_text())
    config['encoders']['toy']['multiscale']['levels'][1]['field_of_view_um'] = 200
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='2:1'):
        Bundle(root)


def test_scale_sets_and_reused_single_scale_rejected(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, multiscale='aligned_centers')
    middle = pd.read_csv(root/'middle.csv')
    middle.iloc[:-1].to_csv(root/'middle.csv', index=False)
    with pytest.raises(ValueError, match='Scale slide sets'):
        Bundle(root)
    coarse = pd.read_csv(root/'coarse.csv')
    coarse.to_csv(root/'middle.csv', index=False)
    with pytest.raises(ValueError, match='distinct feature files'):
        Bundle(root)


@pytest.mark.parametrize('method', ['csmil', 'hagmil_ce'])
def test_contract_refuses_plain_features_and_broken_sampling(tmp_path, method):
    _, bundle, model, _ = fixture(tmp_path, method)
    row = bundle.records('train')[0]
    check_compatibility(require_runnable(method), bundle)
    with pytest.raises(ValueError, match='max_instances'):
        model.forward_unit(bundle, [row], 'cpu', {'max_instances': 8})
    with pytest.raises(ValueError, match='individual slide'):
        model.forward_unit(bundle, [row, row], 'cpu', {})
    plain = Bundle(create_demo(tmp_path/'plain', patients=20))
    with pytest.raises(ValueError, match='multiscale kind'):
        check_compatibility(require_runnable(method), plain)


def test_stable_mask_handles_extremes_and_empty_clusters():
    x = torch.tensor([[1., 10000., -10000.]], requires_grad=True)
    p = stable_masked_softmax(x, torch.tensor([[True, False, False]]))
    torch.testing.assert_close(p, torch.tensor([[1., 0., 0.]]))
    p.sum().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match='unmasked'):
        stable_masked_softmax(x, torch.zeros_like(x))


@pytest.mark.parametrize('dimension', [16, 24])
def test_csmil_matches_author_probability_loss_and_gradients(tmp_path, dimension):
    _, _, model, _ = fixture(tmp_path, 'csmil', dimension)
    author = load_definitions('csmil', 'Train_Test_Code/DeepAttnMISL_model_no21.py',
        ['DeepAttnMIL_Surv'], {'torch': torch, 'nn': torch.nn}, replacements=(
            ('nn.Conv2d(2048, 64, 1)', f'nn.Conv2d({dimension}, 64, 1)'),
            ('hh = x[i].type(torch.FloatTensor).to("cuda")', 'hh = x[i].float()')))
    reference = author.DeepAttnMIL_Surv(3).eval()
    reference.load_state_dict(model.network.state_dict())
    model.eval()
    features = [torch.randn(9, dimension, requires_grad=True) for _ in range(3)]
    labels = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 2])  # deliberately empty middle cluster
    with inference_rng('cpu', 4):
        bags, mask, _ = model.sample(features, labels)
    p = reference(bags, mask)
    logit = model.network(bags, mask)
    torch.testing.assert_close(logit.sigmoid(), p, atol=2e-6, rtol=2e-5)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
    reference_loss = -p.log().mean()
    torch.testing.assert_close(loss, reference_loss)
    first = torch.autograd.grad(loss, features, retain_graph=True)
    second = torch.autograd.grad(reference_loss, features)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=2e-4)
        assert a.abs().sum() > 0


def test_csmil_sampling_masks_replacement_and_inference_rng(tmp_path):
    _, bundle, model, _ = fixture(tmp_path, 'csmil')
    model.eval()
    row = bundle.records('test')[0]
    before = copy.deepcopy(model.assignments)
    torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    left = model.forward_unit(bundle, [row], 'cpu', {})
    right = model.forward_unit(bundle, [row], 'cpu', {})
    torch.testing.assert_close(left['logits'], right['logits'], rtol=0, atol=0)
    assert left['selected_indices'] == right['selected_indices']
    torch.testing.assert_close(torch_state, torch.get_rng_state(), atol=0, rtol=0)
    np.testing.assert_equal(np_state, np.random.get_state())
    assert py_state == random.getstate() and before.keys() == model.assignments.keys()
    for k in before:
        torch.testing.assert_close(before[k], model.assignments[k])
    model.bag_mean, model.bag_std = 100, 0
    features = [torch.randn(6, 16) for _ in range(3)]
    with inference_rng('cpu', 2):
        _, mask, selected = model.sample(features, torch.tensor([0, 0, 0, 2, 2, 2]))
    assert selected[1] == [] and mask.tolist() == [[True, True, True, False, True, True, True]]
    assert any(len(set(s)) < len(s) for s in selected if s)


def test_hag_model_matches_unmodified_author_logits_and_loss(tmp_path):
    _, _, model, _ = fixture(tmp_path, 'hagmil_ce')
    helper = load_hagmil()
    namespace = {'torch': torch, 'nn': torch.nn, 'np': np, 'F': torch.nn.functional,
                 **{k: getattr(helper, k) for k in ('Attn_Net', 'Attn_Net_Gated', 'TransLayer')}}
    original = load_definitions('hagmil', 'models/model.py',
                               ['create_attention_net', 'IAMBlock', 'IATModel'], namespace)
    reference = original.IATModel(feat_in=16, hidden_dims=[32, 48, 32, 32], k_sample=2).eval()
    reference.load_state_dict(model.branches[0].state_dict())
    model.eval()
    x = torch.randn(9, 16, requires_grad=True)
    label = torch.tensor([1])
    result = model.branch(0, x, label)
    ref_logits, _, _, ref_attn, ref_extra = reference(x, label, True)
    torch.testing.assert_close(result['logits'], ref_logits, atol=0, rtol=0)
    torch.testing.assert_close(result['attention'], ref_attn, atol=0, rtol=0)
    torch.testing.assert_close(result['instance_loss'], ref_extra['instance_loss'], atol=0, rtol=0)
    ref_loss = .5*torch.nn.functional.cross_entropy(ref_logits, label, label_smoothing=.1) + .5*ref_extra['instance_loss']
    actual = model.objective(result, label, 0)
    torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0], torch.autograd.grad(ref_loss, x)[0])
    assert model.branch(0, torch.randn(1, 16))['logits'].shape == (1, 2)


def test_hag_selection_preserves_rank_order_with_explicit_parents():
    parent = torch.tensor([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2])
    selected = child_indices(parent, torch.tensor([2, 0]))
    assert selected.tolist() == [2, 5, 8, 11, 1, 4, 7, 10]
    assert parent[selected].tolist() == [2]*4 + [0]*4


def test_hag_three_optimizer_steps_match_author_schedule(tmp_path):
    _, bundle, model, _ = fixture(tmp_path, 'hagmil_ce')
    reference = copy.deepcopy(model)
    actual_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ref_opt = torch.optim.Adam(reference.parameters(), lr=1e-4)
    rows = units(bundle, 'train')[0]
    with inference_rng('cpu', 91):
        hagmil_epoch(model, bundle, [rows], actual_opt, 'cpu', {'grad_clip': 1e20})
    features, _, _ = bundle.multiscale.load(rows[0])
    label = torch.tensor([rows[0]['target']])
    current = torch.arange(len(features[0]))
    with inference_rng('cpu', 91):
        for level in range(3):
            ref_opt.zero_grad(set_to_none=True)
            logits, _, _, attention, extra = reference.branches[level](features[level][current], label, True)
            loss = .5*torch.nn.functional.cross_entropy(logits, label, label_smoothing=.1 if level < 2 else 0.)
            loss = loss + .5*extra['instance_loss']
            loss.backward()
            ref_opt.step()
            if level < 2:
                chosen = current[attention.detach().flatten().argsort(descending=True)[:4]]
                current = torch.tensor([p*4+c for p in chosen.tolist() for c in range(4)])
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)


def test_hag_different_fm_width_and_multiclass_gradients(tmp_path):
    source('hagmil_ce')
    root = create_demo(tmp_path/'data', patients=30, classes=3, dimension=24, multiscale='quadtree')
    bundle = Bundle(root)
    model = build_bundle_model('hagmil_ce', bundle, {'hidden_dims': [32]*4, 'k_sample': 2, 'subtyping': True})
    features, _, _ = bundle.multiscale.load(bundle.records('train')[0])
    label = torch.tensor([2])
    result = model.branch(0, features[0], label)
    assert result['logits'].shape == (1, 3)
    model.objective(result, label, 0).backward()
    for classifier in model.branches[0].instance_classifiers:
        assert classifier.weight.grad.abs().sum() > 0


def test_csmil_provided_clustering_checkpointed_without_test_mutation(tmp_path):
    root, bundle, _, _ = fixture(tmp_path, 'csmil')
    index = pd.read_csv(root/'coarse.csv')
    paths = []
    for row in bundle.rows.to_dict('records'):
        path = f"coarse/{row['slide_id']}_clusters.npy"
        np.save(root/path, np.array([0]*8+[2]*8, dtype=np.int64))
        paths.append(path)
    index['cluster_labels_path'] = paths
    index.to_csv(root/'coarse.csv', index=False)
    bundle = Bundle(root)
    model = build_bundle_model('csmil', bundle, {'clusters': 3, 'clustering': 'provided'})
    model.prepare(bundle, 0, 'cpu')
    assert set(model.assignments) == {r['slide_id'] for r in bundle.records('train')}
    clone = build_bundle_model('csmil', bundle, {'clusters': 3, 'clustering': 'provided'})
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    row = bundle.records('test')[0]
    torch.testing.assert_close(model.forward_unit(bundle, [row], 'cpu', {})['logits'],
                               clone.forward_unit(bundle, [row], 'cpu', {})['logits'], rtol=0, atol=0)
    assert row['slide_id'] not in model.assignments


@pytest.mark.parametrize('method', ['csmil', 'hagmil_ce'])
def test_dedicated_recipe_updates_required_parameters(tmp_path, method):
    _, bundle, model, _ = fixture(tmp_path, method)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    count = []
    optimizer.register_step_post_hook(lambda *args: count.append(1))
    groups = units(bundle, 'train')[:5]
    losses = {'csmil': csmil_epoch, 'hagmil_ce': hagmil_epoch}[method](
        model, bundle, groups, optimizer, 'cpu', {'max_instances': None, 'grad_clip': 5.})
    assert len(count) == (2 if method == 'csmil' else 15)
    assert np.isfinite([v['loss'] for v in losses]).all()
    names = ['network.res_attention.2.weight', 'network.attention.2.weight'] if method == 'csmil' else [
        f'branches.{level}.{part}' for level in range(3)
        for part in ['fc5.weight', 'instance_classifiers.0.weight', 'layers.0.layer.attn.to_qkv.weight']]
    for name in names:
        assert not torch.equal(before[name], dict(model.named_parameters())[name])


@pytest.mark.parametrize('method', ['csmil', 'hagmil_ce'])
def test_multiscale_train_reload_and_exact_resume(tmp_path, method):
    root, bundle, _, options = fixture(tmp_path, method)
    config = {'epochs': 2, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=config)
    split = train(root, method, tmp_path/'split', config=config, stop_after=1)
    train(root, method, split, config=config, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in (full, split)]
    assert a['history'] == b['history']
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    assert not (full/'test_predictions.csv').exists()
    evaluate(full)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))
    assert read_json(full/'method_protocol.json')['kind'] == bundle.multiscale.kind
    row = bundle.records('test')[0]
    path = root/bundle.multiscale.rows(row)[2]['path']
    with h5py.File(path, 'r+') as f:
        f['features'][0, 0] += .25
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(full)

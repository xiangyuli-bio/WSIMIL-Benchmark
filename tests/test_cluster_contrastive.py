"""CBMIL recipe, source-equivalence, cluster provenance and checkpoint checks."""
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.nn import functional as F

from milplus.cluster_contrastive import contrastive_batches, load_cbmil
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, rng_state, train
from milplus.models import build_model
from milplus.recipes import make_optimizers
from milplus.registry import check_compatibility, require_runnable
from milplus.stochastic import inference_rng
from milplus.upstream import source_root

CFG = {'selected_instances': 16, 'clusters': 3, 'batch_size': 4, 'contrastive_dim': 8}


@pytest.fixture(autouse=True)
def author_source():
    if not source_root('cbmil').is_dir():
        pytest.skip('Prepare locked CBMIL source with milplus setup --method cbmil')
    torch.set_num_threads(1)


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    else:
        assert a == b


def add_cluster_labels(root):
    bundle = Bundle(root)
    frame = pd.read_csv(root/'features.csv')
    paths = {}
    for row in bundle.rows.to_dict('records'):
        x, _ = bundle.load(row)
        name = f"features/{row['slide_id']}_clusters.npy"
        np.save(root/name, np.arange(len(x)) % 3)
        paths[row['slide_id']] = name
    frame['cluster_labels_path'] = frame.slide_id.map(paths)
    frame.to_csv(root/'features.csv', index=False)


@pytest.mark.parametrize('clustering', ['kmeans', 'provided'])
def test_cbmil_train_reload_and_exact_resume(tmp_path, clustering):
    root = create_demo(tmp_path/'data', patients=20)
    if clustering == 'provided':
        add_cluster_labels(root)
    cfg = {'epochs': 3, 'threads': 1, 'model': {**CFG, 'clustering': clustering, 'view_instances': 12}}
    full = train(root, 'cbmil', tmp_path/'full', config=cfg)
    part = train(root, 'cbmil', tmp_path/'part', config=cfg, stop_after=1)
    train(root, 'cbmil', part, config=cfg, resume=True)
    a, b = [torch.load(run/'last.pt', weights_only=False) for run in [full, part]]
    for key in ['model', 'optimizers', 'history', 'rng']:
        same(a[key], b[key])
    assert set(a['model']['_extra_state']['assignments']) == set(Bundle(root).rows.query("subset == 'train'").slide_id)
    assert a['history'][-1]['train']['instance_loss'] > 0 and a['history'][-1]['train']['contrastive_loss'] > 0
    assert not (full/'test_predictions.csv').exists()
    evaluate(full)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))
    if clustering == 'provided':
        row = Bundle(root).records('test')[0]
        path = root/row['cluster_labels_path']
        np.save(path, (np.load(path)+1) % 3)
        with pytest.raises(ValueError, match='inputs differ'):
            evaluate(full)


@pytest.mark.parametrize('dimension', [24, 512])
def test_cbmil_author_loss_and_all_branches_update(dimension):
    torch.manual_seed(22)
    model = build_model('cbmil', dimension, 3, CFG)
    bags = [torch.randn(30, dimension) for _ in range(4)]
    views = list(zip(*[model.views(bag, torch.arange(30) % 3) for bag in bags]))
    labels = torch.tensor([0, 1, 2, 0])
    with inference_rng('cpu', 4):
        actual = model.batch_loss(views, labels)
    with inference_rng('cpu', 4):
        expected_loss, expected_logits = model.network.train_forward(views, labels)
    torch.testing.assert_close(actual['loss'], expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual['logits'], expected_logits, rtol=0, atol=0)
    same(actual['loss'], .9*(.7*actual['bag_loss']+.3*actual['instance_loss'])+.1*actual['contrastive_loss'])
    optimizer = make_optimizers(model, 'cbmil', .0001, .0001)[0]
    before = {name: value.clone() for name, value in model.named_parameters()}
    actual['loss'].backward()
    optimizer.step()
    branches = ['network.selecting.ranking.q', 'network.pp_mil.patch_mil.instance_classifiers',
                'network.pp_mil.cluster_attention', 'network.contrastive_fc']
    if dimension != 512:
        branches.append('input_projection')
    for branch in branches:
        parameters = [(name, value) for name, value in model.named_parameters() if name.startswith(branch)]
        assert parameters
        assert any(value.grad is not None and value.grad.abs().sum() > 0 for _, value in parameters)
        assert any(not torch.equal(before[name], value) for name, value in parameters)
    if dimension == 512:
        assert isinstance(model.input_projection, torch.nn.Identity)


@pytest.mark.parametrize('batch', [2, 4, 5])
def test_nt_xent_matches_manual_value_and_gradient(batch):
    torch.manual_seed(8)
    a, b = torch.randn(batch, 7, requires_grad=True), torch.randn(batch, 7, requires_grad=True)
    loss = load_cbmil().NT_Xent(batch_size=4, temperature=.7)(a, b)
    z = F.normalize(torch.cat((a, b)), dim=1)
    similarity = z @ z.T / .7
    similarity = similarity.masked_fill(torch.eye(2*batch, dtype=torch.bool), -torch.inf)
    positives = (torch.arange(2*batch)+batch) % (2*batch)
    reference = F.cross_entropy(similarity, positives)
    torch.testing.assert_close(loss, reference)
    actual_grad = torch.autograd.grad(loss, (a, b), retain_graph=True)
    expected_grad = torch.autograd.grad(reference, (a, b))
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


def test_cbmil_views_mixup_and_tail_batch_contracts():
    assert contrastive_batches(list(range(9)), 4) == [list(range(4)), list(range(4, 9))]
    assert contrastive_batches(list(range(4)), 8) == [list(range(4))]
    with pytest.raises(ValueError, match='two bags'):
        contrastive_batches([0], 4)
    with pytest.raises(ValueError, match='batch_size'):
        build_model('cbmil', 16, 2, {**CFG, 'batch_size': 1})
    model = build_model('cbmil', 512, 2, {**CFG, 'view_instances': 12})
    x = torch.randn(30, 512)
    full, subset = model.views(x, torch.arange(30) % 3)
    assert [len(group) for group in full] == [10, 10, 10]
    assert [len(group) for group in subset] == [4, 4, 4]
    for full_group, selected in zip(full, subset):
        assert all(any(torch.equal(row, source) for source in full_group) for row in selected)
    inputs = torch.randn(4, 9, 8)
    mixed, lam, permutation = load_cbmil().mixup(inputs, .8)
    assert ((lam >= .8) & (lam <= 1)).all()
    torch.testing.assert_close(mixed, lam[..., None]*inputs+(1-lam[..., None])*inputs[permutation])


def test_cbmil_prediction_rng_and_no_training_state_mutation():
    model = build_model('cbmil', 16, 2, CFG).eval()
    x = torch.randn(30, 16)
    before, state = rng_state(), {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
    with torch.no_grad():
        a, b = model(x), model(x)
    same(a, b)
    same(before, rng_state())
    same(state, model.state_dict())


def test_cluster_index_validation_and_patient_bag_rejection(tmp_path):
    root = create_demo(tmp_path/'data', patients=20)
    add_cluster_labels(root)
    bundle = Bundle(root)
    bundle.check(deep=True)
    row = bundle.records('train')[0]
    path = root/row['cluster_labels_path']
    labels = np.load(path)
    for malformed in [labels[:-1], labels.astype(float), -labels]:
        np.save(path, malformed)
        with pytest.raises(ValueError, match='aligned nonnegative integer'):
            bundle.check(deep=True)
    np.save(path, labels)
    config = yaml.safe_load((root/'task.yaml').read_text())
    config['training_unit'] = 'patient'
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='individual slide bags'):
        check_compatibility(require_runnable('cbmil'), Bundle(root))
    with pytest.raises(ValueError, match='exactly'):
        build_model('cbmil', 16, 2, CFG).assign(torch.randn(20, 16), torch.zeros(20, dtype=torch.long))

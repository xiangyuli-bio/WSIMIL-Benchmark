"""Author parity, training-only initialization and resumable context/RNN recipes."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.context_clustering import MiCoClassification, MiCoSurvival, cluster_centers, patient_bags
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.discrete_survival import discrete_nll
from milplus.efficient_sequence import EfficientSequenceMIL, load_efficient
from milplus.engine import evaluate, train
from milplus.upstream import load_definitions, source_root


def source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch the pinned '+name+' source')
    torch.set_num_threads(1)


def original_mico():
    return load_definitions('mico', 'models/MiCo.py',
        ['GatedAttention', 'Mlp', 'ClusterReducer', 'MiCo'],
        {'torch': torch, 'nn': nn, 'F': F}).MiCo


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('hard,similarity', [(False, 'l2'), (True, 'dot')])
def test_mico_author_classification_routes_centers_loss_gradients(dim, hard, similarity):
    source('mico')
    model = MiCoClassification(dim, 3, hidden=16, clusters=8, enhancers=2,
                              hard=hard, similarity=similarity)
    model.core.initialized.fill_(True)
    reference = original_mico()(dim_in=dim, embedding_dim=16, num_clusters=8,
        num_classes=3, survival=False, num_enhancers=2, hard=hard, similarity_method=similarity)
    reference.load_state_dict(model.core.network.state_dict())
    x, label = torch.randn(13, dim), torch.tensor([2])
    torch.manual_seed(37); actual = model(x)['logits']
    torch.manual_seed(37); expected = reference(data=x[None])['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    F.cross_entropy(actual, label).backward(); F.cross_entropy(expected, label).backward()
    for (name, p), (_, q) in zip(model.core.network.named_parameters(), reference.named_parameters()):
        if name == 'similarity_bias':
            assert p.grad is None and q.grad is None  # unused in original forward
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert model.core.network.cluster_centers.grad.abs().sum() > 0
    before = model.core.network.cluster_centers.detach().clone()
    opt = model.make_optimizer(.001, .01); opt.step()
    assert not torch.equal(before, model.core.network.cluster_centers)
    assert {g['weight_decay'] for g in opt.param_groups} == {0, .01}


@pytest.mark.parametrize('event', [0, 1])
def test_mico_author_separate_slides_survival_and_censor_objective(event):
    source('mico')
    model = MiCoSurvival(16, hidden=16, clusters=8, enhancers=2).eval()
    model.core.initialized.fill_(True)
    reference = original_mico()(dim_in=16, embedding_dim=16, num_clusters=8,
        num_classes=4, survival=True, num_enhancers=2).eval()
    reference.load_state_dict(model.core.network.state_dict())
    bags = [torch.randn(1, 16), torch.randn(9, 16)]
    actual = model(bags); expected = reference(data=[x[None] for x in bags])
    torch.testing.assert_close(actual['hazards'], expected['hazards'], atol=0, rtol=0)
    torch.testing.assert_close(actual['survival'], expected['S'])
    torch.testing.assert_close(actual['logits'], -expected['S'].sum(1, keepdim=True))
    hazard, survival = expected['hazards'][0, 2], expected['S'][0]
    original_loss = -(survival[1].log()+hazard.log()) if event else -.85*survival[2].log()
    loss = discrete_nll(actual['hazard_logits'], torch.tensor([2]), torch.tensor([event]), .15)
    torch.testing.assert_close(loss, original_loss)
    loss.backward(); original_loss.backward()
    for p, q in zip(model.core.network.parameters(), reference.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
    assert not torch.allclose(actual['hazards'], model([torch.cat(bags)])['hazards'])


def test_faiss_author_centers_seed_train_only_initialization_and_limits(tmp_path):
    source('mico')
    faiss = pytest.importorskip('faiss')
    values = np.random.default_rng(43).normal(size=(200, 16)).astype('float32')
    raw = load_definitions('mico', 'run_clustering.py', ['FaissKmeans'], {'faiss': faiss, 'np': np})
    threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        expected, _ = raw.FaissKmeans(4, niter=5, gpu=False).train(values)
    finally:
        faiss.omp_set_num_threads(threads)
    np.testing.assert_array_equal(cluster_centers(values, 4, iterations=5), expected)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    model = MiCoClassification(16, 2, hidden=16, clusters=4, enhancers=2,
                              cluster_slides=3, cluster_iterations=5)
    original_load, seen = bundle.load, []
    def recording_load(row):
        seen.append(row['slide_id'])
        assert row['subset'] == 'train'
        return original_load(row)
    bundle.load = recording_load
    model.prepare(bundle, 42, 'cpu')
    wanted = sorted(bundle.records('train'), key=lambda row: row['slide_id'])[:3]
    assert seen == model.core.initialization_slides == [r['slide_id'] for r in wanted]
    values = torch.cat([original_load(r)[0] for r in wanted]).numpy()
    np.testing.assert_array_equal(model.core.network.cluster_centers.detach().numpy(), cluster_centers(values, 4, 5))
    model.core.max_clustering_bytes = 1
    with pytest.raises(ValueError, match='max_clustering_bytes'):
        model.prepare(bundle, 42, 'cpu')
    with pytest.raises(ValueError, match='distinct'):
        cluster_centers(np.ones((20, 16)), 4)
    model.core.network.similarity_scale.data.zero_()
    with pytest.raises(FloatingPointError, match='scale'):
        model(torch.randn(3, 16))


def test_mico_patient_budget_preserves_slide_boundaries(tmp_path):
    source('mico')
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, survival=True))
    rows = bundle.records('train')[:2]
    full = patient_bags(bundle, rows, 'cpu')
    small = patient_bags(bundle, rows, 'cpu', 5)
    indices = torch.linspace(0, sum(map(len, full))-1, 5).long()
    assert len(full) == len(small) == 2 and sum(map(len, small)) == 5
    torch.testing.assert_close(torch.cat(small), torch.cat(full)[indices])


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
@pytest.mark.parametrize('dim,classes', [(16, 2), (32, 3)])
def test_efficient_author_output_bce_l2_and_gradient_parity(kind, dim, classes):
    source('efficientmil')
    model = EfficientSequenceMIL(dim, classes, 'efficientmil_'+kind, hidden=16, selected_patches=8)
    common, module, _ = load_efficient(kind)
    reference = common.MILNet(common.FCLayer(dim, classes), module.BClassifier(
        input_size=dim, output_class=classes, dropout=.1, bidirectional=True,
        big_lambda=8, selection_strategy='aps', use_intelligent_selector=True,
        **{kind+'_hidden_size': 16, kind+'_num_layers': 2}))
    reference.load_state_dict(model.network.state_dict())
    x, label = torch.randn(17, dim), torch.tensor([classes-1])
    torch.manual_seed(77); actual = model(x, label=label)
    torch.manual_seed(77); instances, logits, _, embedding = reference(x)
    target = F.one_hot(label, classes).float()
    loss = .5*(F.binary_cross_entropy_with_logits(logits, target)+
               F.binary_cross_entropy_with_logits(instances.max(0, keepdim=True).values, target))
    loss += 1e-4*sum(p.norm(2)**2 for p in reference.parameters() if p.requires_grad)
    torch.testing.assert_close(actual['logits'], logits, atol=0, rtol=0)
    torch.testing.assert_close(actual['embedding'], embedding, atol=0, rtol=0)
    torch.testing.assert_close(actual['total_loss'], loss)
    actual['total_loss'].backward(); loss.backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), reference.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)
    assert sum(p.grad.abs().sum() for p in model.network.i_classifier.parameters()) > 0


def test_efficient_selection_limits_and_reproducible_random_inference():
    source('efficientmil')
    model = EfficientSequenceMIL(16, 2, hidden=16, selected_patches=8).eval()
    with pytest.raises(ValueError, match='two patches'):
        model(torch.randn(1, 16))
    model.max_similarity_bytes = 10
    with pytest.raises(ValueError, match='similarity matrix'):
        model(torch.randn(4, 16))
    random = EfficientSequenceMIL(16, 2, hidden=16, selected_patches=3, selection='random-k').eval()
    x = torch.randn(17, 16); rng = torch.get_rng_state().clone()
    actual = random(x)['logits']
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(actual, random(x)['logits'], atol=0, rtol=0)
    with pytest.raises(ValueError, match='custom backend'):
        load_efficient('mamba')


def test_efficient_aps_diversity_entropy_and_selected_order():
    source('efficientmil')
    common, _, _ = load_efficient('gru')
    selector = common.IntelligentPatchSelector('hybrid')
    x, scores = torch.randn(1, 7, 16), torch.randn(1, 7, 3)
    norm = F.normalize(x, dim=-1)
    diversity = 1-((norm@norm.transpose(1, 2)).sum(-1)-1)/6
    probs = scores.softmax(-1)
    entropy = -(probs*(probs+1e-8).log()).sum(-1)
    combined = scores.max(-1).values+.3*diversity+.3*entropy
    indices, attention = selector(x, scores, 4)
    assert torch.equal(indices, combined.topk(4, dim=1).indices)
    torch.testing.assert_close(attention, combined.softmax(1), atol=0, rtol=0)


def test_efficient_author_adam_cosine_schedule():
    source('efficientmil')
    model = EfficientSequenceMIL(16, 2, hidden=16)
    model.configure_training({'epochs': 5, 'lr': .0002})
    optimizer = model.make_optimizer(.0002, .00001)
    assert optimizer.param_groups[0]['betas'] == (.5, .9)
    assert optimizer.param_groups[0]['weight_decay'] == .00001
    for epoch in range(6):
        expected = 5e-6+(.0002-5e-6)*(1+np.cos(np.pi*epoch/5))/2
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)
        optimizer.step(); model.after_validation({})


@pytest.mark.parametrize('method', ['mico', 'mico_survival', 'efficientmil_gru', 'efficientmil_lstm'])
def test_context_recurrent_exact_resume_reloaded_predictions(tmp_path, method):
    source('mico' if method.startswith('mico') else 'efficientmil')
    root = create_demo(tmp_path/'data', patients=20, survival=method.endswith('survival'))
    options = {'hidden': 16}
    if method.startswith('mico'):
        pytest.importorskip('faiss')
        options.update(clusters=4, enhancers=2, cluster_slides=3, cluster_iterations=5)
    else:
        options.update(selected_patches=8)
    cfg = {'epochs': 3, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=cfg)
    partial = train(root, method, tmp_path/'partial', config=cfg, stop_after=1)
    train(root, method, partial, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    assert a['history'] == b['history']
    for key in list(a['model']):
        if key.endswith('_extra_state'):
            assert a['model'].pop(key) == b['model'].pop(key)
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    assert evaluate(full) == evaluate(partial)


def test_agdr_author_threshold_scorer_does_not_receive_gradient():
    source('agdrmil')
    get_act = load_definitions('agdrmil', 'utils/variate.py', ['get_act'], {'torch': torch}).get_act
    raw = load_definitions('agdrmil', 'model.py', ['initialize_weights', 'att_split',
        'CosineSimilarityWeighted', 'RowMask', 'mil_aggregation', 'AGDR_MIL'],
        {'torch': torch, 'nn': nn, 'F': F, 'get_act': get_act})
    torch.manual_seed(43)
    model = raw.AGDR_MIL(in_dim=16)
    first, second = model(torch.randn(1, 64, 16))
    loss = .4*F.cross_entropy(first, torch.tensor([1]))+F.cross_entropy(second, torch.tensor([1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.att.parameters())
    assert all(p.grad is not None for p in model.reduce_dim.parameters())
    assert all(p.grad is not None for p in model.classifier_2.parameters())

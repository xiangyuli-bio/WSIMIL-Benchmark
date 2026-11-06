"""Actual released components; full pretrained TITAN training is a separate gate."""
import copy
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from milplus.dyko_source import (COMMIT, FaissBackend, author_classes,
    classification_loss, original_shot_selection, source_adam, source_assets)
from milplus.upstream import verified_source


def test_original_source_and_complete_native768_component_shapes():
    torch.set_num_threads(1)
    root, commit = verified_source('dyko')
    assert commit == COMMIT and (root/'models/DyKo.py').is_file()
    classes = author_classes('cpu_faiss_reference')
    adapter = classes.Adapter(768, 4)
    assert adapter.fc[0].weight.shape == (192, 768)
    assert adapter.fc[2].weight.shape == (768, 192)
    assert sum(p.numel() for p in adapter.parameters()) == 294912
    cluster = classes.ClusterHead(768, 10)
    text, image = cluster(torch.randn(1, 257, 768), torch.randn(1, 257, 768))
    assert text.shape == image.shape == (1, 257, 10)
    torch.testing.assert_close(text.sum(1), torch.ones(1, 10))
    # Preserve the author's dim=1 normalization; class probabilities do NOT sum
    # to one across the final axis in this actual [1,N,K] forward route.
    assert not torch.allclose(text.sum(-1), torch.ones(1, 257))


@pytest.mark.parametrize('cohort,classes', [('CAMELYON16', 2), ('NSCLC', 2), ('RCC', 3), ('UBC', 5)])
def test_all_full_released_native768_banks_and_ordered_prompts(cohort, classes):
    assets = source_assets(cohort)
    assert assets['bank'].shape == (1000, 768)
    assert len(assets['prompt_text']) == classes == assets['classes']
    assert all(isinstance(x, str) and len(x) > 20 for x in assets['prompt_text'])


def oracle_retrieval(features, bank, clusters=10, top=10):
    # Independent direct execution of every released equation and actual FAISS
    # CPU clustering. Native GPU equivalence is explicitly not inferred here.
    km = faiss.Kmeans(d=768, k=min(clusters, len(features)), niter=20,
                      min_points_per_centroid=5, verbose=False, gpu=False, seed=42)
    km.train(features.detach().cpu().numpy().astype('float32'))
    centroids = F.normalize(torch.from_numpy(km.centroids), dim=1)
    bank_norm = F.normalize(bank, dim=1)
    scores = (centroids @ bank_norm.T).softmax(0)
    assignments = scores.argmax(0)
    selected = torch.zeros(len(bank), dtype=torch.bool)
    for index in range(clusters):
        candidates = torch.where(assignments == index)[0]
        if len(candidates):
            rank = scores[:, candidates].max(0).values.argsort(descending=True)
            selected[candidates[rank[:top]]] = True
    chosen = bank_norm[selected]
    weights = (F.normalize(features, dim=1) @ chosen.T / .01).softmax(1)
    return F.normalize(weights @ chosen, dim=1), selected


@pytest.mark.parametrize('patches', [7, 257])
def test_full_native768_retrieval_values_and_feature_concept_gradients_match_source(patches):
    torch.set_num_threads(1); faiss.omp_set_num_threads(1); torch.manual_seed(2024)
    classes = author_classes('cpu_faiss_reference')
    features = torch.randn(patches, 768, requires_grad=True)
    bank = source_assets('CAMELYON16')['bank'].clone().requires_grad_()
    actual = classes.DyKo.get_concepts_refined(None, features, bank, n_clu=10, topK=10)
    reference, selected = oracle_retrieval(features, bank)
    assert torch.equal(actual, reference) and 0 < selected.sum() <= 100
    probe = torch.randn_like(actual)
    grad_a = torch.autograd.grad((actual*probe).sum(), (features, bank), retain_graph=True)
    grad_b = torch.autograd.grad((reference*probe).sum(), (features, bank))
    for a, b in zip(grad_a, grad_b):
        assert torch.equal(a, b) and torch.isfinite(a).all() and a.abs().sum() > 0
    assert grad_a[1][~selected].abs().sum() == 0
    call = classes.backend.calls[0]
    assert call['actual_backend'] == 'cpu_faiss_reference'
    assert call['requested']['niter'] == 20 and call['requested']['seed'] == 42


@pytest.mark.parametrize('fold', range(4))
@pytest.mark.parametrize('shots', [4, 8, 16])
def test_selection_matches_every_released_author_shot_cache(fold, shots):
    root, _ = verified_source('dyko')
    split = pd.read_csv(root/f'splits/4foldcls/camelyon16/fold{fold}.csv')
    frame = pd.DataFrame({'train_slide_id': split.train_slide_id.dropna(),
                          'train_label': split.train_label.dropna()}).reset_index(drop=True)
    original = pd.read_csv(root/f'splits/4foldcls/camelyon16/shot_samples/fold{fold}_shot{shots}_seed1024.csv')
    actual = original_shot_selection(frame, shots, 1024)
    pd.testing.assert_frame_equal(actual, original, check_dtype=False)
    assert len(actual) == shots*2
    assert set(actual.train_slide_id) <= set(frame.train_slide_id)
    assert not set(actual.train_slide_id) & set(split.val_slide_id.dropna())
    assert not set(actual.train_slide_id) & set(split.test_slide_id.dropna())


def test_actual_classification_training_step_preserves_unscaled_structural_KL_and_gradients():
    torch.set_num_threads(1); torch.manual_seed(15)
    classes = author_classes('cpu_faiss_reference')
    cluster = classes.ClusterHead(768, 10)
    features = torch.randn(1, 257, 768, requires_grad=True)
    concepts = torch.randn(1, 257, 768, requires_grad=True)
    text_prob, image_prob = cluster(concepts, features)
    logits = torch.randn(1, 3, requires_grad=True)
    label = torch.tensor([2])
    results = {'logits': logits, 'Y_prob': logits.softmax(1), 'Y_hat': logits.argmax(1, keepdim=True),
               'text_cluster_prob': text_prob, 'image_cluster_prob': image_prob}
    actual = classification_loss(results, label)
    ce = F.cross_entropy(logits, label)
    expected = ce + F.kl_div(image_prob.log(), text_prob, reduction='batchmean')
    assert torch.equal(actual, expected) and actual > ce
    a = torch.autograd.grad(actual, (features, concepts, logits), retain_graph=True)
    b = torch.autograd.grad(expected, (features, concepts, logits))
    for x, y in zip(a, b):
        assert torch.equal(x, y) and torch.isfinite(x).all() and x.abs().sum() > 0


@pytest.mark.parametrize('weight_decay', [0., .00001])
def test_original_adam_decay_groups_and_actual_moment_updates(weight_decay):
    torch.set_num_threads(1); torch.manual_seed(72)
    classes = author_classes('cpu_faiss_reference')
    actual = classes.ClusterHead(768, 10)
    reference = copy.deepcopy(actual)
    opt = source_adam(actual, weight_decay=weight_decay)
    if weight_decay:
        no_decay = [p for n, p in reference.named_parameters() if len(p.shape) == 1 or n.endswith('.bias')]
        decay = [p for n, p in reference.named_parameters() if len(p.shape) != 1 and not n.endswith('.bias')]
        groups = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]
    else:
        groups = reference.parameters()
    refopt = torch.optim.Adam(groups, lr=.001, weight_decay=0.)
    x, y = torch.randn(1, 257, 768), torch.randn(1, 257, 768)
    for _ in range(2):
        for model, optimizer in [(actual, opt), (reference, refopt)]:
            optimizer.zero_grad(); a, b = model(x, y)
            F.kl_div(b.log(), a, reduction='batchmean').backward(); optimizer.step()
    for a, b in zip(actual.parameters(), reference.parameters()):
        assert torch.equal(a, b)
        sa, sb = opt.state[a], refopt.state[b]
        assert all(torch.equal(sa[key], sb[key]) for key in sa)


def test_missing_native_titan_fails_instead_of_constructing_a_fake_text_encoder():
    classes = author_classes('cpu_faiss_reference')
    with pytest.raises(FileNotFoundError, match='authorized local TITAN'):
        classes.DyKo()


def test_cuda_faiss_route_never_silently_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        FaissBackend('cuda_faiss')


@pytest.mark.parametrize('shots,seed', [(0, 42), (True, 42), (4, -1), (4, 2**32), (4, True)])
def test_invalid_shot_specification_fails(shots, seed):
    frame = pd.DataFrame({'train_slide_id': ['s1', 's2'], 'train_label': [0, 1]})
    with pytest.raises(ValueError):
        original_shot_selection(frame, shots, seed)


def test_shot_selection_uses_all_available_when_a_class_has_fewer_than_requested():
    frame = pd.DataFrame({'train_slide_id': ['s1', 's2', 's3'], 'train_label': [0, 1, 1]})
    result = original_shot_selection(frame, 4, 42)
    assert len(result) == 3 and set(result.train_slide_id) == set(frame.train_slide_id)

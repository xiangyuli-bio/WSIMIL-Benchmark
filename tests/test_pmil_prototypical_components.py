"""Full native PMIL source kernels; not a complete-pipeline admission claim."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import pairwise_distances
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from milplus.pmil_prototypical_source import (
    COMMIT, IMAGENET_SHA256, bag_definitions, bag_model, bag_objective, bag_step,
    encoder, encoder_definitions, patch_tensor, select_indices,
)
from milplus.pmil_prototypical_clustering import discover, similarity, ap
from milplus.upstream import load_definitions, verified_source


WEIGHTS = Path(os.environ.get('MILPLUS_PMIL_WEIGHTS', Path.home()/'.cache/milplus/pmil/resnet34-333f7ec4.pth'))


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.fixture
def selected_encoder(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(24)
    model = encoder(WEIGHTS)
    path = tmp_path/'selected_native_encoder.pth'
    torch.save({'state_dict': model.state_dict()}, path)
    return path


def test_actual_full_original_resnet34_ImageNet_state_logits_features_gradients():
    torch.set_num_threads(2)
    source = encoder_definitions()
    torch.manual_seed(24)
    original = source.ResNet(source.BasicBlock, [3, 4, 6, 3])
    original.load_state_dict(torch.load(WEIGHTS, weights_only=True), strict=True)
    original.fc = nn.Linear(512, 2)
    torch.manual_seed(24)
    actual = encoder(WEIGHTS)
    equal(original.state_dict(), actual.state_dict())
    assert sum(p.numel() for p in actual.parameters()) > 21_000_000
    assert len(actual.layer1) == 3 and len(actual.layer2) == 4
    assert len(actual.layer3) == 6 and len(actual.layer4) == 3
    pixels = np.random.default_rng(93).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    x = torch.stack([patch_tensor(v) for v in pixels])
    a, b = original(x), actual(x)
    assert a[0].shape == (2, 2) and a[1].shape == (2, 512)
    equal(a, b)
    F.cross_entropy(a[0], torch.tensor([0, 1])).backward()
    F.cross_entropy(b[0], torch.tensor([0, 1])).backward()
    for p, q in zip(original.parameters(), actual.parameters()):
        equal(p.grad, q.grad)
    equal(original.state_dict(), actual.state_dict())


def test_literal_source_constructors_dual_losses_every_gradient_and_Adam(selected_encoder):
    source = bag_definitions(WEIGHTS)
    torch.manual_seed(39)
    original = source.PBMIL(source.IClassifier(str(selected_encoder)), source.BClassifier(7))
    torch.manual_seed(39)
    prototypes = torch.arange(7*512, dtype=torch.float32).reshape(7, 512)/1000
    actual = bag_model(WEIGHTS, selected_encoder, prototypes)
    equal(original.state_dict(), actual.state_dict())
    aopt = torch.optim.Adam(original.parameters(), lr=1e-4, weight_decay=1e-4)
    bopt = torch.optim.Adam(actual.parameters(), lr=1e-4, weight_decay=1e-4)
    trainer = load_definitions('pmil_prototypical', 'pmil.py', ['train'],
        {'torch': torch, 'np': np, 'F': F, 'roc_curve': roc_curve, 'auc': auc,
         'args': SimpleNamespace(nepochs=30)},
        replacements=[('input = input.cuda()', "input = input.to('cpu')"),
                      ('target = target.cuda()', "target = target.to('cpu')")])
    generator = torch.Generator().manual_seed(47)
    bags = [(torch.randn(17+i, 512, generator=generator), i % 2, 'slide_'+str(i)) for i in range(4)]
    for epoch in range(2):
        loader = DataLoader(bags, batch_size=1, shuffle=False, num_workers=0)
        trainer.train(epoch, prototypes, loader, original, nn.CrossEntropyLoss(), aopt)
        for x, y, _ in DataLoader(bags, batch_size=1, shuffle=False, num_workers=0):
            bag_step(actual, x[0], prototypes, y, bopt)
        equal(original.state_dict(), actual.state_dict())
        equal(aopt.state_dict(), bopt.state_dict())
        for p, q in zip(original.parameters(), actual.parameters()):
            equal(p.grad, q.grad)
    assert not actual.training
    assert all(p.grad is None for p in actual.i_classifier.features.parameters())
    assert actual.i_classifier.fc.weight.grad is not None
    assert all(p.grad is not None for p in actual.b_classifier.parameters())
    out = bag_objective(actual, bags[0][0], prototypes, torch.tensor([0]))
    equal(out['loss'], out['bag_ce']+out['max_ce'])


def test_complete_two_level_affinity_propagation_matches_original_functions(tmp_path):
    generator = np.random.default_rng(24)
    records, features = [], {}
    for i in range(4):
        name = 'train_'+str(i)
        x = generator.normal(size=(17+i, 512)).astype(np.float32)
        records.append({'slide_id': name, 'subset': 'train'})
        features[name] = x
        np.save(tmp_path/(name+'.npy'), x)
    args = SimpleNamespace(feat_format='.npy', feat_dir=str(tmp_path), preference=0, damping=.5,
        lamb=.25, local_cluster=str(tmp_path/'local_original.ckpt'))
    source = load_definitions('pmil_prototypical', 'cluster.py',
        ['local_clustering', 'global_clustering', 'euclidean_similarity'],
        {'torch': torch, 'np': np, 'pd': __import__('pandas'), 'os': os, 'time': __import__('time'),
         'AffinityPropagation': AffinityPropagation, 'pairwise_distances': pairwise_distances, 'args': args})
    local_indices, local_features, slide_indices, patch_indices = source.local_clustering(
        [r['slide_id']+'.npy' for r in records], [])
    centers, expected, model = source.global_clustering(local_features)
    actual = discover(records, lambda row: features[row['slide_id']])
    equal(actual['features'], expected)
    equal(actual['global_centers'], centers.tolist())
    assert actual['global_labels'] == model.labels_.tolist()
    assert [r['centers'] for r in actual['local']] == [r.tolist() for r in local_indices]
    expected_origins = [{'slide_id': records[slide_indices[i]]['slide_id'], 'patch_index': int(patch_indices[i])} for i in centers]
    assert actual['origins'] == expected_origins
    assert actual['all_patch_candidates'] == 17+18+19+20
    for value in features.values():
        equal(similarity(value), source.euclidean_similarity(value))


@pytest.mark.parametrize('k', [1, 5, 1000])
def test_original_topk_positive_probability_tie_selection(k):
    scores = np.array([.1, .9, .7, .9, .3, .9, .8], dtype=np.float32)
    expected = np.lexsort((scores, np.zeros(len(scores), dtype=np.int64)))[-k:].tolist()
    assert select_indices(scores, k) == expected


def test_original_preprocessing_pixels_unchanged_and_nonRGB_rejected():
    from PIL import Image
    from torchvision import transforms
    a = np.random.default_rng(53).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    expected = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3, [.1]*3)])(Image.fromarray(a))
    equal(patch_tensor(a), expected)
    for x in [a[:224], a.astype(np.float32), a[:, :, 0]]:
        with pytest.raises(ValueError, match='actual uint8 RGB256'):
            patch_tensor(x)


def test_validation_and_TEST_cannot_enter_prototype_discovery():
    for subset in ['val', 'test', None]:
        with pytest.raises(ValueError, match='TRAIN'):
            discover([{'slide_id': 'held_out', 'subset': subset}], lambda _: pytest.fail('held-out features were accessed'))
    with pytest.raises(ValueError, match='Duplicate'):
        discover([{'slide_id':'a', 'subset':'train'}]*2, lambda _: pytest.fail('invalid rows accessed'))


def test_undefined_AP_distances_cannot_silently_become_kmeans_or_random_prototypes():
    with pytest.raises(ValueError, match='undefined'):
        ap(np.ones((17, 512), dtype=np.float32))
    for x in [np.ones((1, 512), dtype=np.float32), np.ones((17, 128), dtype=np.float32),
              np.ones((17, 512), dtype=np.float64)]:
        with pytest.raises(ValueError, match='actual'):
            ap(x)


def test_invalid_initialization_and_wrong_method_identity_fail(tmp_path):
    path = tmp_path/'wrong.pth'
    path.write_bytes(b'no random initialization fallback')
    with pytest.raises(ValueError, match='actual original'):
        encoder(path)
    assert verified_source('pmil_prototypical')[1] == COMMIT
    from milplus.io import digest_file
    assert digest_file(WEIGHTS) == IMAGENET_SHA256


@pytest.mark.parametrize('n', [0, 4])
def test_fixed_five_patch_selection_requires_real_instances(n):
    with pytest.raises(ValueError, match='at least five'):
        bag_objective(nn.Identity(), torch.zeros(n, 512), torch.zeros(3, 512), torch.tensor([0]))

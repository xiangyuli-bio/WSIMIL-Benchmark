"""Full original DLA operators and all distinct hybrid supervision mechanisms."""
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.engine import seed_everything, rng_state, restore_rng
from milplus.hybrid_supervision_source import modules, build_model, source_stage2, validate_weights, COMMIT
from milplus.hybrid_supervision_ops import refine_labels, top_patch_indices, image_objective, released_pool_draw, paper_pool_draw
from milplus.hybrid_supervision_demo import create_demo
from milplus.hybrid_supervision_data import prepare, HybridInputs
from milplus.upstream import load_definitions, verified_source
from test_pmil_prototypical_components import equal

WEIGHTS = Path(os.environ.get('MILPLUS_HYBRID_WEIGHTS', Path.home()/'.cache/milplus/hybrid/dla34-ba72cf86.pth'))


def test_actual_full_pretrained_DLA34Up512_pixel_BCE_Adam_every_gradient_and_source_train(tmp_path):
    torch.set_num_threads(2);seed_everything(102)
    actual = build_model(WEIGHTS, 'segmentation')
    seed_everything(102);source = modules()
    original_zoo = source.dla_bn.model_zoo
    source.dla_bn.model_zoo = SimpleNamespace(load_url=lambda *a, **kw: torch.load(WEIGHTS, weights_only=True))
    try:
        original = source.dla_up_bn.dla34up_bn(1, 'imagenet', down_ratio=2)
    finally:
        source.dla_bn.model_zoo = original_zoo
    equal(actual.state_dict(), original.state_dict())
    assert sum(p.numel() for p in actual.parameters()) == 18_094_265
    assert actual.base.channels == [16, 32, 64, 128, 256, 512]
    assert len(actual.base.level3.tree1.tree1.conv1.weight.shape) == 4
    x = torch.rand(2, 3, 512, 512)
    y = torch.rand(2, 1, 512, 512)
    actual.train();original.train()
    left, right = actual(x), original(x)
    assert left.shape == (2, 1, 512, 512)
    equal(left, right)
    # Reset running statistics before the literal original training body.
    original.load_state_dict(actual.state_dict())
    aopt = torch.optim.Adam(actual.parameters(), lr=1e-3)
    bopt = torch.optim.Adam(original.parameters(), lr=1e-3)
    output = actual(x);loss = F.binary_cross_entropy_with_logits(output, y)
    aopt.zero_grad();loss.backward();aopt.step()
    from PIL import Image
    log = tmp_path/'train.log';log.write_text('')
    # Original destructive diagnostic-viz commands are routed to an owned
    # no-op sink; they do not participate in model/loss/optimizer/RNG equations.
    class SinkImage:
        def __init__(self, image):self.image = image
        def save(self, path):pass
    original_train = load_definitions('hybrid_supervision', 'Stage1_M_step_retrain.py',
        ['AverageMeter', 'channel_fill', 'train'],
        {'torch': torch, 'np': np, 'time': __import__('time'), 'datetime': __import__('datetime'),
         'os': SimpleNamespace(system=lambda command: None),
         'Image': SimpleNamespace(fromarray=lambda array: SinkImage(array))})
    args = SimpleNamespace(device=torch.device('cpu'), log_file=str(log), out_dir=str(tmp_path))
    original_train.train(args, [(x, y[:, 0], ['first', 'second'])], [], original, nn.BCEWithLogitsLoss(), bopt, 0, 1)
    equal(actual.state_dict(), original.state_dict());equal(aopt.state_dict(), bopt.state_dict())
    for a, b in zip(actual.parameters(), original.parameters()):equal(a.grad, b.grad)
    assert verified_source('hybrid_supervision')[1] == COMMIT


def test_paper_top16_full_DLA34_mean_probability_BCE_is_not_mean_logit_pooling():
    torch.set_num_threads(2);seed_everything(102)
    model = build_model(WEIGHTS, 'classification')
    assert sum(p.numel() for p in model.parameters()) == 15_271_345
    assert model.fc.in_channels == 512 and model.fc.out_channels == 1
    logits = torch.linspace(-5, 2, 16).reshape(16, 1).requires_grad_()
    loss, probability = image_objective(logits, torch.tensor([1]))
    expected = logits.sigmoid().mean(0)
    torch.testing.assert_close(probability, expected, rtol=0, atol=0)
    torch.testing.assert_close(loss, -expected.log().mean(), rtol=1e-6, atol=1e-7)
    assert not torch.allclose(probability, logits.mean(0).sigmoid())
    loss.backward();assert torch.isfinite(logits.grad).all() and (logits.grad != 0).all()
    scores = torch.linspace(0, 1, 19).reshape(19, 1, 1).expand(19, 512, 512)
    assert top_patch_indices(scores).tolist() == list(range(18, 2, -1))
    assert top_patch_indices(torch.ones(19, 512, 512)).tolist() == list(range(16))
    with pytest.raises(ValueError, match='actual scored'):
        top_patch_indices(scores[:15])


def test_paper_pixel_pseudolabel_V4_clip_threshold_and_hard_negative_are_literal_equations():
    probability = torch.tensor([[0, .001, .002, .0025, .1, .249, .25, .8, 1]], dtype=torch.float32)
    actual = refine_labels(probability, 1)
    expected = (probability*4).clamp(max=1)
    expected[expected < .01] = 0
    equal(actual, expected);equal(refine_labels(probability, 0), torch.zeros_like(probability))
    assert not torch.equal(actual, probability)
    source = refine_labels(probability, 1, quantized_source=True)
    q = (probability.numpy()*255).astype(np.uint8).astype(np.float32)*4
    q[q > 255] = 255;q[q < 255*.01] = 0
    expected = torch.from_numpy(q.astype(np.uint8).astype(np.float32)/255)
    equal(source, expected)
    assert not torch.equal(source, actual)


def test_released_sampling_edge40_and50_are_fine_and_paper_ratio_is_separate(monkeypatch):
    available = {'fine': ['fine'], 'positive': ['positive'], 'negative': ['negative']}
    counts = {k: 0 for k in available}
    for draw in range(101):
        monkeypatch.setattr(random, 'randint', lambda *a, draw=draw: draw)
        kind, item = released_pool_draw(available);assert kind == item;counts[kind] += 1
    assert counts == {'fine': 52, 'positive': 9, 'negative': 40}
    counts = {k: 0 for k in available}
    for draw in range(10):
        monkeypatch.setattr(random, 'randrange', lambda *a, draw=draw: draw)
        kind, _ = paper_pool_draw(available['fine'], available['positive'], available['negative'])
        counts[kind] += 1
    assert counts == {'fine': 2, 'positive': 1, 'negative': 7}


def test_released_RF_feature_and_patient_rules_are_original_and_not_the_paper_CNN():
    source = source_stage2()
    heatmap = np.zeros((64, 64), dtype=np.uint8);heatmap[10:31, 20:42] = 255
    features = source.feature_extract(heatmap)
    assert features.shape == (4,) and features[0] == 255 and features[3] == 21*22
    assert source.node2patient([0, 0, 0, 0, 0]) == 0
    assert source.node2patient([1, 0, 0, 0, 0]) == 1
    assert source.node2patient([2, 0, 0, 0, 0]) == 2
    assert source.node2patient([3, 2, 0, 0, 0]) == 3
    assert source.node2patient([3, 2, 2, 2, 0]) == 4


def test_full512RGB_fine_masks_patient_binding_and_missing_fine_supervision_rejection(tmp_path):
    data = create_demo(tmp_path/'data', WEIGHTS)
    bundle = Bundle(data);assets = bundle.hybrid_supervision
    assert all(assets.count(row) == 16 for row in bundle.rows.to_dict('records'))
    assert assets.meta['slides']['synthetic_1']['fine_positive_mask_count'] == 1
    assert assets.meta['slides']['synthetic_2']['fine_indices'] == []
    with pytest.raises(ValueError, match='full segmentation'):
        bundle.load(bundle.records('train')[0])
    with h5py.File(data/'pixels/synthetic_1.h5', 'r+') as f:f['fine_masks'][0] = 0
    with pytest.raises(ValueError, match='bytes changed'):
        assets.verify(bundle.records('train')[1])


def test_hybrid_all329_profiles154_families_and328_prior_profile_bytes_are_preserved():
    from milplus.registry import profiles, methods, inspect_method
    root = Path(__file__).resolve().parents[1]
    prior = json.loads((root/'tests/fixtures/hybrid_prior328_profiles.json').read_text())
    historical_ids = {p['id'] for p in prior} | {'hybrid_supervision_paper_full_reference'}
    current = [p for p in profiles() if p['id'] in historical_ids]
    assert len(prior) == 328 and len(current) == 329
    assert [p for p in current if p['family'] != 'hybrid_supervision'] == prior
    assert len({p['family'] for p in current}) == 154 and len(methods()) == 164
    assert inspect_method('Hybrid Supervision Learning')['id'] == 'hybrid_supervision_paper_full_reference'

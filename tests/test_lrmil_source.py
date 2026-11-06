"""Independent complete-author and physical-region proofs, not a tiny encoder."""
import copy
import gc
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.lrmil_source import (SOURCE_LAYERS, author_components, native_backbone,
    normalize_rgb, patch_objective, student_objective, teacher_metadata)
from milplus.upstream import load_definitions

WEIGHTS = Path('/root/.cache/milplus/assets/focus/conch.pth')


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(75)
    yield
    gc.collect()


def test_full_native_encoder_author_output_and_every_parameter_gradient_match():
    actual = native_backbone(WEIGHTS)
    class NCHWPatchEmbedding(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
        def forward(self, x):
            return self.original(x).permute(0, 3, 1, 2)
    visual = nn.Module()
    visual.trunk = copy.deepcopy(actual.visual)
    visual.trunk.patch_embed = NCHWPatchEmbedding(visual.trunk.patch_embed)
    conch = nn.Module()
    conch.visual = visual
    models, _, _ = author_components()
    original = models.StudentBackbone(conch, SOURCE_LAYERS)
    assert sum(p.numel() for p in actual.parameters()) == 86_250_240
    assert len(actual.visual.blocks) == 12
    image = torch.randint(0, 256, (1, 256, 256, 3), dtype=torch.uint8)
    inputs = normalize_rgb(image)
    left, right = actual(inputs), original(inputs)
    for key in ['final']:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    for key in ['layer_cls', 'layer_tokens']:
        assert set(left[key]) == set(right[key]) == set(SOURCE_LAYERS)
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(left[key][layer], right[key][layer], atol=0, rtol=0)
    assert left['final'].shape == (1, 768)
    assert all(value.shape == (1, 16, 16, 768) for value in left['layer_tokens'].values())
    target = torch.randn_like(left['final'])
    torch.nn.functional.mse_loss(left['final'], target).backward()
    torch.nn.functional.mse_loss(right['final'], target).backward()
    expected = {name.replace('.patch_embed.original.', '.patch_embed.'): p
                for name, p in original.named_parameters()}
    assert expected.keys() == dict(actual.named_parameters()).keys()
    for name, p in actual.named_parameters():
        q = expected[name]
        assert p.grad is not None and q.grad is not None, name
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_actual_dynamic_patch_layout_exposes_original_backbone_failure():
    actual = native_backbone(WEIGHTS)
    models, _, _ = author_components()
    original = models.StudentBackbone(actual.conch_model, SOURCE_LAYERS)
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        assert actual.visual.patch_embed(x).shape == (1, 16, 16, 768)
        with pytest.raises(RuntimeError):
            original(x)
        output = actual(x)
    assert output['final'].shape == (1, 768)


def test_named_position_interpolation_uses_true_2d_pretrained_grid():
    from timm.layers import resample_abs_pos_embed
    actual = native_backbone(WEIGHTS, position_policy='spatial_interpolation')
    trunk = actual.visual
    pixels = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = actual(pixels)
        x = trunk.patch_embed(pixels).reshape(1, 256, 768)
        x = torch.cat([trunk.cls_token.expand(1, -1, -1), x], dim=1)
        spatial = resample_abs_pos_embed(trunk.pos_embed, (16, 16),
            num_prefix_tokens=1, interpolation='bicubic', antialias=True)
        assert not torch.equal(spatial, trunk.pos_embed[:, :257])
        x = trunk.pos_drop(x+spatial)
        expected = {}
        for layer, block in enumerate(trunk.blocks):
            x = block(x)
            if layer in SOURCE_LAYERS:
                expected[layer] = x[:, 1:].reshape(1, 16, 16, 768)
        final = trunk.norm(x)[:, 0]
    torch.testing.assert_close(output['final'], final, atol=0, rtol=0)
    for layer in SOURCE_LAYERS:
        torch.testing.assert_close(output['layer_tokens'][layer], expected[layer], atol=0, rtol=0)


def test_source_teacher_tuple_defect_and_explicit_correct_metadata(tmp_path):
    models, _, _ = author_components()
    model = models.MILClassifier(768, 3)
    features = torch.randn(27, 768)
    helper = load_definitions('lrmil', 'utils/trainer.py', ['save_teacher_meta'],
                              {'os': os, 'torch': torch})
    batch = {'E20': features[None], 'coords_20x': torch.zeros(1, 27, 2),
             'label': torch.tensor([2]), 'wsi_name': ['slide'], 'pid': ['patient']}
    helper.save_teacher_meta(model, [batch], str(tmp_path), 1, 'cpu')
    broken = torch.load(tmp_path/'teacher_meta/fold1/slide.pt', weights_only=False)
    correct = teacher_metadata(model, features)
    assert broken['attention'].shape == (768,) and broken['bag_repr'].shape == (27,)
    assert correct['attention'].shape == (27,) and correct['bag_repr'].shape == (768,)
    logits, bag, attention = model(features)
    torch.testing.assert_close(correct['attention'], attention, atol=0, rtol=0)
    torch.testing.assert_close(correct['bag_repr'], bag, atol=0, rtol=0)
    torch.testing.assert_close(correct['logits'], logits, atol=0, rtol=0)


@pytest.mark.parametrize('classes,gated', [(2, False), (3, False), (3, True)])
def test_all_four_student_losses_and_gradients_match_author_and_manual_terms(classes, gated):
    models, losses, helpers = author_components()
    actual = models.StudentMIL(768, classes, gated)
    original = copy.deepcopy(actual)
    low = torch.tensor([[i*1024, 0] for i in range(9)])
    high = torch.tensor([[i*1024+256*x, 256*y] for i in range(9) for y in range(4) for x in range(4)])
    feats = torch.randn(9, 768)
    metadata = {'attention': torch.linspace(.1, 2., len(high)), 'logits': torch.randn(classes)}
    target = torch.tensor(classes-1)
    terms, logits, guided = student_objective(actual, feats, target, metadata, low, high)
    expected_logits, _, attention = original(feats)
    expected_guided = helpers.build_teacher_guided_attention(low, high, metadata['attention'])
    manual_guided = metadata['attention'].reshape(9, 16).mean(1)
    torch.testing.assert_close(guided, manual_guided, atol=0, rtol=0)
    torch.testing.assert_close(guided, expected_guided, atol=0, rtol=0)
    ce = losses.ce_loss(expected_logits, target)
    bag = losses.bag_kd_loss(expected_logits, metadata['logits'])
    soft = losses.attention_kl_loss(attention, expected_guided)
    hard = losses.hard_topk_instance_loss(feats, expected_guided, original.inst_classifier, 4)
    manual_bag = (F.softmax(metadata['logits'], -1)*(F.log_softmax(metadata['logits'], -1)-F.log_softmax(expected_logits, -1))).sum()
    manual_soft = (F.softmax(guided, 0)*(F.log_softmax(guided, 0)-F.log_softmax(attention, 0))).sum()/len(guided)
    torch.testing.assert_close(bag, manual_bag, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(soft, manual_soft, atol=1e-7, rtol=1e-6)
    expected_loss = ce+.5*bag+.5*soft+.3*hard
    for key, value in [('loss', expected_loss), ('ce', ce), ('bag_kd', bag),
                       ('soft_attention_kd', soft), ('hard_instance_ce', hard)]:
        torch.testing.assert_close(terms[key], value, atol=0, rtol=0)
    terms['loss'].backward()
    expected_loss.backward()
    for name, p in actual.named_parameters():
        q = dict(original.named_parameters())[name]
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert actual.inst_classifier.weight.grad.abs().sum() > 0


def test_complete_patch_region_loss_and_token_gradients_match_independent_geometry():
    tokens = {layer: torch.randn(2, 16, 16, 768, requires_grad=True) for layer in SOURCE_LAYERS}
    final = torch.randn(2, 768, requires_grad=True)
    coordinates = torch.tensor([[0, 0], [256, 0], [0, 256], [768, 768]])
    items = [{'coords_5x': torch.zeros(2, dtype=torch.long), 'coords_20x': coordinates,
              'teacher_layers': {layer: torch.randn(4, 768) for layer in SOURCE_LAYERS},
              'teacher_final': torch.randn(4, 768), 'teacher_attn': torch.tensor([.1, .2, .3, .4])}
             for _ in range(2)]
    output = {'final': final, 'layer_tokens': tokens}
    terms = patch_objective(output, items, lambda_layer=.7, lambda_attn=.4)
    each_layer, each_attention = [], []
    for index, item in enumerate(items):
        cells = []
        for layer in SOURCE_LAYERS:
            regions = []
            for patch, (x, y) in enumerate(coordinates.tolist()):
                student = tokens[layer][index, y//64:y//64+4, x//64:x//64+4].mean((0, 1))
                # Independently derive the physical matching, while using the
                # same PyTorch MSE kernel to demand bitwise gradient equality.
                regions.append(F.mse_loss(student, item['teacher_layers'][layer][patch]))
            cells.append(torch.stack(regions).mean())
        each_layer.append(torch.stack(cells).mean())
        distances = (item['teacher_final']-final[index]).square().mean(1)
        each_attention.append((distances*item['teacher_attn']/item['teacher_attn'].sum()).sum())
    manual_layer = torch.stack(each_layer).mean()
    manual_attention = torch.stack(each_attention).mean()
    manual_loss = 0.
    for layer_loss, attention_loss in zip(each_layer, each_attention):
        manual_loss = manual_loss + .7*layer_loss + .4*attention_loss
    manual_loss = manual_loss/len(items)
    torch.testing.assert_close(terms['layer_mse'], manual_layer, atol=0, rtol=0)
    torch.testing.assert_close(terms['attention_weighted_mse'], manual_attention, atol=0, rtol=0)
    torch.testing.assert_close(terms['loss'], manual_loss, atol=0, rtol=0)
    values = [final, *tokens.values()]
    left = torch.autograd.grad(terms['loss'], values, retain_graph=True)
    right = torch.autograd.grad(manual_loss, values)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert a.abs().sum() > 0


def test_rgb_normalization_uses_conch_constants_without_resize_or_hidden_augmentation():
    image = torch.full((2, 256, 256, 3), 127, dtype=torch.uint8)
    value = normalize_rgb(image)
    assert value.shape == (2, 3, 256, 256)
    for channel, (mean, std) in enumerate(zip([.48145466, .4578275, .40821073],
                                             [.26862954, .26130258, .27577711])):
        assert torch.allclose(value[:, channel], torch.full_like(value[:, channel], (127/255-mean)/std), atol=1e-7, rtol=0)
    with pytest.raises(ValueError):
        normalize_rgb(image.float())


def test_no_random_or_reduced_encoder_fallback_for_missing_weights(tmp_path):
    path = tmp_path/'wrong.pth'
    torch.save({'fake': torch.ones(1)}, path)
    with pytest.raises(ValueError, match='actual registered pretrained'):
        native_backbone(path)

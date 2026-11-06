"""Pinned complete LRMIL components and the actual pretrained CONCH ViT.

Integration remains in progress. These components alone do not constitute a
complete three-phase runnable method or an admission receipt.
"""
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .focus_network import CONCH_COMMIT, WEIGHTS_BYTES, WEIGHTS_SHA, checked_source
from .io import digest_file
from .upstream import load_definitions, verified_source

COMMIT = '66ea1c5a447ed2a830bb1af6739ab9261e61000d'
TIMM_VERSION = '0.9.12'
SOURCE_LAYERS = (0, 1, 10)


def author_components():
    _, commit = verified_source('lrmil')
    if commit != COMMIT:
        raise ValueError('LRMIL source differs from the reviewed fixed commit')
    models = load_definitions('lrmil', 'models/models.py',
        ['AttentionMIL', 'MILClassifier', 'StudentMIL', 'StudentBackbone'],
        {'math': math, 'Iterable': Iterable, 'torch': torch, 'nn': nn, 'F': F})
    losses = load_definitions('lrmil', 'utils/losses.py',
        ['ce_loss', 'bag_kd_loss', 'attention_kl_loss', 'hard_topk_instance_loss',
         'patch_kd_loss', 'attention_weighted_feature_loss'], {'torch': torch, 'F': F})
    helpers = load_definitions('lrmil', 'utils/trainer.py',
        ['build_teacher_guided_attention'], {'torch': torch})
    return models, losses, helpers


def native_backbone(weights, position_policy='source_slice', layers=SOURCE_LAYERS):
    """Use all150 native visual tensors, no smaller or random encoder fallback.

    The author StudentBackbone assumes NCHW patch embedding, while the actual
    dynamic CONCH timm trunk returns NHWC. Only that layout is explicitly ported.
    The released leading-position slice and a named spatial interpolation are
    separate conditions; neither silently replaces the other.
    """
    author_components()
    checked_source('conch_focus', CONCH_COMMIT)
    weights = Path(weights)
    if (not weights.is_file() or weights.stat().st_size != WEIGHTS_BYTES
            or digest_file(weights) != WEIGHTS_SHA):
        raise ValueError('LRMIL needs the actual registered pretrained CONCH checkpoint')
    if (position_policy not in {'source_slice', 'spatial_interpolation'}
            or not isinstance(layers, (list, tuple)) or not layers
            or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 0 <= i < 12 for i in layers)):
        raise ValueError('LRMIL requires explicit native ViT layer and position policies')
    import timm
    from timm.layers import resample_abs_pos_embed
    from timm.models.vision_transformer import VisionTransformer
    if timm.__version__ != TIMM_VERSION:
        raise ValueError('LRMIL native CONCH port requires pinned timm0.9.12')
    trunk = VisionTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        img_size=448, patch_size=16, num_classes=0, dynamic_img_size=True)
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    selected = {key.removeprefix('visual.trunk.'): value
                for key, value in state.items() if key.startswith('visual.trunk.')}
    if len(selected) != 150 or sum(value.numel() for value in selected.values()) != 86_250_240:
        raise ValueError('LRMIL complete native pretrained visual state is missing')
    trunk.load_state_dict(selected, strict=True)
    if str(trunk.patch_embed.output_fmt).split('.')[-1] != 'NHWC':
        raise ValueError('LRMIL reviewed timm patch embedding layout changed')
    conch = nn.Module()
    conch.visual = nn.Module()
    conch.visual.trunk = trunk

    def position_tokens(visual, tokens):
        if position_policy == 'source_slice':
            return tokens+visual.pos_embed[:, :tokens.shape[1], :]
        grid = math.isqrt(tokens.shape[1]-1)
        if grid*grid != tokens.shape[1]-1:
            raise ValueError('LRMIL native image must form a square patch grid')
        positional = resample_abs_pos_embed(visual.pos_embed, (grid, grid),
            num_prefix_tokens=1, interpolation='bicubic', antialias=True)
        return tokens+positional

    source = load_definitions('lrmil', 'models/models.py', ['StudentBackbone'],
        {'math': math, 'Iterable': Iterable, 'torch': torch, 'nn': nn,
         'position_tokens': position_tokens},
        replacements=[
            ('x = x.flatten(2).transpose(1, 2)',
             'x = x.reshape(batch_size, -1, x.shape[-1])'),
            ('x = x + trunk.pos_embed[:, : x.shape[1], :]',
             'x = position_tokens(trunk, x)')])
    model = source.StudentBackbone(conch, layers_to_return=layers)
    model._milplus_lrmil_identity = {'source': COMMIT, 'CONCH_source': CONCH_COMMIT,
        'weights_sha256': WEIGHTS_SHA, 'timm_version': TIMM_VERSION,
        'timm_vision_source_sha256': digest_file(Path(timm.__file__).parent/'models/vision_transformer.py'),
        'visual_width': 768, 'visual_layers': 12, 'visual_heads': 12,
        'visual_parameter_count': 86_250_240, 'image_size': 256, 'patch_size': 16,
        'selected_layers': list(layers), 'position_policy': position_policy,
        'layout_port': 'actual dynamic CONCH NHWC patches -> ordered spatial tokens',
        'unused_towers': 'Omit unused text/caption towers; all actual visual trunk tensors retained; no cold-start full-factory RNG equality claim'}
    return model


def normalize_rgb(images):
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1:] != (256, 256, 3):
        raise ValueError('LRMIL needs actual NHWC uint8 RGB256 patches')
    x = images.permute(0, 3, 1, 2).float()/255.
    mean = x.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = x.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    return (x-mean)/std


def teacher_metadata(teacher, high_features):
    """Explicitly repair the released save_teacher_meta tuple misassignment."""
    logits, bag, attention_logits = teacher(high_features)
    if (attention_logits.shape != (len(high_features),) or bag.shape != (768,)
            or logits.ndim != 1 or not all(torch.isfinite(x).all()
                for x in [logits, bag, attention_logits])):
        raise ValueError('LRMIL actual teacher bag/logit/attention output differs')
    return {'logits': logits, 'bag_repr': bag, 'attention': attention_logits}


def patch_objective(output, items, lambda_layer=1., lambda_attn=1., layers=SOURCE_LAYERS):
    """Actual released stage1 objective, including its attention-weighted term."""
    if (not items or output['final'].shape != (len(items), 768)
            or set(output['layer_tokens']) != set(layers)):
        raise ValueError('LRMIL patch KD requires all selected native layers')
    _, losses, _ = author_components()
    components, total = [], 0.0
    for index, item in enumerate(items):
        layer_tokens = {layer: tokens[index:index+1]
                        for layer, tokens in output['layer_tokens'].items()}
        layer = losses.patch_kd_loss(layer_tokens, item['teacher_layers'],
            item['coords_5x'], item['coords_20x'], layers)
        attention = losses.attention_weighted_feature_loss(output['final'][index],
            item['teacher_final'], item['teacher_attn'])
        components.append({'layer_mse': layer, 'attention_weighted_mse': attention,
                           'loss': lambda_layer*layer+lambda_attn*attention})
        # Match the actual author's sequential batch-item accumulation order.
        total = total + float(lambda_layer)*layer + float(lambda_attn)*attention
    result = {key: torch.stack([row[key] for row in components]).mean()
              for key in components[0]}
    result['loss'] = total/len(items)
    if not all(torch.isfinite(value) for value in result.values()):
        raise FloatingPointError('LRMIL patch-level source objective is nonfinite')
    return result


def student_objective(student, features, label, metadata, coords_low, coords_high,
                      temperature_bag=1., temperature_attn=1., lambda_ce=1.,
                      lambda_bag=.5, lambda_soft=.5, lambda_hard=.3,
                      topk=4, teacher_attn_agg='mean'):
    """All four released student terms, including the source gate/normalization."""
    _, losses, helpers = author_components()
    logits, _, attention = student(features)
    ce = losses.ce_loss(logits, label)
    bag = losses.bag_kd_loss(logits, metadata['logits'], temperature_bag)
    guided = helpers.build_teacher_guided_attention(coords_low, coords_high,
        metadata['attention'], agg=teacher_attn_agg)
    soft = logits.new_zeros(())
    hard = logits.new_zeros(())
    if guided.sum().abs() > 1e-6:
        soft = losses.attention_kl_loss(attention, guided, temperature_attn)
        hard = losses.hard_topk_instance_loss(features, guided, student.inst_classifier, topk)
    loss = lambda_ce*ce+lambda_bag*bag+lambda_soft*soft+lambda_hard*hard
    terms = {'loss': loss, 'ce': ce, 'bag_kd': bag, 'soft_attention_kd': soft,
             'hard_instance_ce': hard}
    if not all(torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError('LRMIL student source objective is nonfinite')
    return terms, logits, guided

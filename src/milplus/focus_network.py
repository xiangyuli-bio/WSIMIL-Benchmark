"""Actual pinned FOCUS and full pretrained CONCH text tower, privately loaded."""
import math
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .io import digest_file
from .upstream import load_definitions, load_single, pinned_text, verified_source

COMMIT = '66c4015d5ba09657f4c8183bc06947faecd5b01f'
CONCH_COMMIT = '141cc09c7d4ff33d8eda562bd75169b457f71a62'
WEIGHTS_SHA = '40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c'
WEIGHTS_BYTES = 802235437
TOKENIZER = 'conch/open_clip_custom/tokenizers/conch_byte_level_bpe_uncased.json'


def checked_source(source, expected):
    root, sha = verified_source(source)
    if sha != expected:
        raise ValueError('FOCUS source/dependency differs from reviewed commit: '+source)
    return root, sha


def text_builder():
    checked_source('conch_focus', CONCH_COMMIT)
    scope = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'random': random,
        'OrderedDict': OrderedDict, 'Callable': Callable, 'Optional': Optional,
        'Sequence': Sequence, 'Tuple': tuple, 'checkpoint': checkpoint}
    transformer = load_definitions('conch_focus', 'conch/open_clip_custom/transformer.py',
        ['LayerNormFp32', 'LayerNorm', 'QuickGELU', 'LayerScale', 'ResidualAttentionBlock',
         'Transformer', 'TextTransformer'], scope)
    return load_definitions('conch_focus', 'conch/open_clip_custom/coca_model.py',
        ['CLIPTextCfg', '_build_text_tower'],
        {'dataclass': dataclass, 'Optional': Optional, 'nn': nn, 'TextTransformer': transformer.TextTransformer})


def native_text_model(weights):
    # The unused visual tower is never retained by FOCUS on frozen features.
    # Load all actual native text tensors; keep the trainable transformer intact.
    root, sha = checked_source('conch_focus', CONCH_COMMIT)
    if weights.stat().st_size != WEIGHTS_BYTES or digest_file(weights) != WEIGHTS_SHA:
        raise ValueError('FOCUS requires the registered actual native pretrained CONCH checkpoint')
    import json
    cfg = json.loads(pinned_text(root, sha, 'conch/open_clip_custom/model_configs/conch_ViT-B-16.json'))
    expected = {'context_length': 128, 'vocab_size': 32007, 'width': 768,
                'heads': 12, 'layers': 12, 'embed_cls': True, 'output_tokens': True}
    if cfg['text_cfg'] != expected or cfg['embed_dim'] != 512:
        raise ValueError('FOCUS requires the actual full native CONCH text architecture')
    text = text_builder()._build_text_tower(cfg['embed_dim'], cfg['text_cfg'])
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    selected = {k.removeprefix('text.'): v for k, v in state.items() if k.startswith('text.')}
    if len(selected) != 150:
        raise ValueError('FOCUS actual pretrained text state is incomplete')
    text.load_state_dict(selected, strict=True)
    model = nn.Module()
    model.text = text
    return model


def author_classes(weights):
    checked_source('focus', COMMIT)
    root, sha = checked_source('conch_focus', CONCH_COMMIT)
    # Original tokenizer reads its local JSON; verify that it is the tracked
    # pinned blob before executing the real module (no remote tokenizer fetch).
    if (root/TOKENIZER).read_text() != pinned_text(root, sha, TOKENIZER):
        raise ValueError('FOCUS actual local CONCH tokenizer bytes changed')
    tokenizer = load_single('conch_focus', 'conch/open_clip_custom/custom_tokenizer.py')

    def factory(config, path):
        if config != 'conch_ViT-B-16' or path != 'ckpts/conch.pth':
            raise ValueError('FOCUS original pretrained factory arguments changed')
        return native_text_model(weights), None

    return load_definitions('focus', 'models/model_FOCUS.py', ['TextEncoder', 'PromptLearner', 'FOCUS'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'create_model_from_pretrained': factory,
         'get_tokenizer': tokenizer.get_tokenizer, 'tokenize': tokenizer.tokenize})


class FOCUSReference(nn.Module):
    def __init__(self, weights, prompts, dimension=512, window_size=8, sim_threshold=.8, max_context_length=8192):
        super().__init__()
        if (type(dimension) is not int or dimension < 1 or type(window_size) is not int or window_size < 1
                or type(max_context_length) is not int or max_context_length < 1
                or isinstance(sim_threshold, bool) or not isinstance(sim_threshold, (float, int))
                or not np.isfinite(sim_threshold) or not -1 <= sim_threshold <= 1
                or not isinstance(prompts, list) or len(prompts) < 4 or len(prompts) % 2
                or any(not isinstance(s, str) or not s.strip() for s in prompts)):
            raise ValueError('FOCUS requires valid full feature dimensions, token selection and ordered 2C prompts')
        self.dimension, self.classes = dimension, len(prompts)//2
        cfg = SimpleNamespace(input_size=dimension, window_size=window_size,
            sim_threshold=sim_threshold, max_context_length=max_context_length, text_prompt=prompts)
        self.network = author_classes(weights).FOCUS(cfg, num_classes=self.classes)

    def forward(self, features, label=None):
        if (features.ndim != 2 or not len(features) or features.shape[1] != self.dimension
                or not features.is_floating_point() or not torch.isfinite(features).all()):
            raise ValueError('FOCUS needs the complete finite ordered [N,D] feature bag')
        if label is None:
            target = torch.zeros(1, dtype=torch.long, device=features.device)
        else:
            target = torch.as_tensor(label, device=features.device).reshape(-1)
            if target.shape != (1,) or target.dtype != torch.int64 or not 0 <= int(target[0]) < self.classes:
                raise ValueError('FOCUS requires one bag class ID')
        probabilities, prediction, loss = self.network(None, features, target)
        if not torch.isfinite(probabilities).all() or not torch.isfinite(loss):
            raise ValueError('FOCUS nonfinite native complete objective')
        return {'probabilities': probabilities, 'prediction': prediction, 'loss': loss if label is not None else None}

    def protocol(self):
        return {'source': COMMIT, 'conch_source': CONCH_COMMIT, 'pretrained_weights_sha256': WEIGHTS_SHA,
            'feature_dimension': self.dimension, 'native_visual_dimension': 512,
            'text_width': 768, 'text_layers': 12, 'text_heads': 12, 'text_projection': 512,
            'text_context_length': 128, 'trainable_prompt_tokens': 16,
            'trainable_text_parameters': sum(p.numel() for p in self.network.text_encoder.parameters()),
            'visual_projection': 'full Dto512/LN/ReLU/dropout0.25; complete cached input, no FM update',
            'text_path': 'Actual unmasked CONCH Transformer/BOS-position0 pooling/projection from FOCUS source; not standard CONCH causal/CLS encode_text',
            'prompts': 'Ordered low then high C descriptions; replace original positions1..16 by shared16 context, only high C queries used; fixed original token prefix/suffix buffers',
            'selection': 'local sequence windows mean+biasedstd, text dot-product relevance mean over C, top min(8192,N) at native setting, restore row order',
            'compression': 'sequence chunks8, retain first and adjacent cosine<threshold; no physical coordinate kernel',
            'source_preserved': 'Original actual FOCUS definitions, trainable full pretrained text encoder, full512/8-head visual-text attention and CE; second scale unused by original forward',
            'allocation_port': 'Omit only unused CONCH vision tower allocation. Its unused RNG initialization differs from original factory; no seedwise cold-start equality claim',
            'dependency_pin': 'Explicit reviewed CONCH commit; original FOCUS README did not pin historical CONCH version',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

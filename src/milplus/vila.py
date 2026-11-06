"""Pinned ViLa-MIL: learned prompts through CLIP and dual-scale prototype decoders."""
import math
import re
import types
import warnings
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def clip_text_classes():
    return load_definitions('openai_clip', 'clip/model.py',
        ['LayerNorm', 'QuickGELU', 'ResidualAttentionBlock', 'Transformer'],
        {'torch': torch, 'nn': nn, 'OrderedDict': OrderedDict})


class CLIPText(nn.Module):
    def __init__(self, width, layers, heads, length, vocabulary, dimension):
        super().__init__()
        definitions = clip_text_classes()
        mask = torch.full((length, length), float('-inf')).triu(1)
        self.transformer = definitions.Transformer(width, layers, heads, mask)
        self.token_embedding = nn.Embedding(vocabulary, width)
        self.positional_embedding = nn.Parameter(torch.zeros(length, width))
        self.ln_final = definitions.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.zeros(width, dimension))
        self.dtype = torch.float32
        self.visual = types.SimpleNamespace(input_resolution=224)


def load_clip_text(text, dimension):
    state = torch.load(text.checkpoint_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise TypeError('CLIP checkpoint must be a tensor state dict, not a serialized model')
    prefixes = ('transformer.', 'token_embedding.', 'positional_embedding', 'ln_final.', 'text_projection')
    state = {k: v.float() for k, v in state.items() if k.startswith(prefixes) and isinstance(v, torch.Tensor)}
    width = state['ln_final.weight'].numel()
    length, pos_width = state['positional_embedding'].shape
    vocabulary, token_width = state['token_embedding.weight'].shape
    indices = {int(re.match(r'transformer.resblocks.(\d+)\.', k).group(1)) for k in state if k.startswith('transformer.resblocks.')}
    layers = len(indices)
    heads = text.metadata.get('text_heads', 8)
    if indices != set(range(layers)) or layers < 1 or not isinstance(heads, int) or heads < 1 or width % heads:
        raise ValueError('CLIP text layer/head configuration is invalid')
    if pos_width != width or token_width != width or state['text_projection'].shape != (width, dimension):
        raise ValueError('Image dimension does not match CLIP text projection')
    if not text.metadata.get('synthetic') and (width, layers, heads, length, vocabulary, dimension) != (512, 12, 8, 77, 49408, 1024):
        raise ValueError('Native ViLa needs RN50 text dimensions; other backbones require a named adaptation')
    if text.tokens.shape[1] != length or (text.tokens < 0).any() or (text.tokens >= vocabulary).any():
        raise ValueError('CLIP token ids do not match checkpoint context/vocabulary')
    eot, sot = text.metadata['eot_token_id'], text.metadata['sot_token_id']
    if eot != vocabulary-1 or sot != vocabulary-2:
        raise ValueError('CLIP EOT/SOT ids must match tokenizer vocabulary')
    if not (text.tokens[:, 0] == sot).all() or not ((text.tokens == eot).sum(1) == 1).all():
        raise ValueError('Each prompt requires one SOT prefix and exactly one EOT')
    ends = text.tokens.argmax(-1)
    if (ends < 2).any() or (ends+16 >= length).any():
        raise ValueError('Prompt is empty or exceeds context after inserting 16 learned tokens')
    for row, end in zip(text.tokens, ends):
        if (row[end+1:] != 0).any() or (row[1:end] == sot).any():
            raise ValueError('Malformed CLIP prompt padding/special tokens')
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite CLIP checkpoint weights')
    model = CLIPText(width, layers, heads, length, vocabulary, dimension)
    model.load_state_dict(state, strict=True)
    return model


def load_vila(clip_model, tokens, prompts, corrected=False, structured=True):
    # The source MHA accepts 2D keys with 3D queries; keep that exact computation.
    ns = {k: getattr(F, k) for k in ('linear', 'softmax', 'dropout', 'pad')}
    ns.update(torch=torch, nn=nn, Tensor=torch.Tensor, Optional=Optional, Module=nn.Module,
        Parameter=nn.Parameter, _LinearWithBias=nn.Linear, xavier_uniform_=nn.init.xavier_uniform_,
        xavier_normal_=nn.init.xavier_normal_, constant_=nn.init.constant_, warnings=warnings,
        has_torch_function=torch.overrides.has_torch_function, handle_torch_function=torch.overrides.handle_torch_function)
    attention = load_definitions('vilamil', 'models/model_utils.py', ['MultiheadAttention', 'multi_head_attention_forward'], ns)
    token_map = {}
    for prompt, row in zip(prompts, tokens):
        key = prompt.replace('_', ' ')
        if key in token_map and not torch.equal(token_map[key], row):
            raise ValueError('Identical prompts have inconsistent token ids')
        token_map[key] = row
    clip = types.SimpleNamespace(load=lambda *a, **kw: (clip_model, None),
                                 tokenize=lambda p: token_map[p][None])
    patches = [('.T.cuda()', '.T'), ('compents.squeeze()', 'compents.squeeze(1)'),
               ('compents_high.squeeze()', 'compents_high.squeeze(1)'),
               ('text_context_features.squeeze()', 'text_context_features.squeeze(1)'),
               ('text_context_features_high.squeeze()', 'text_context_features_high.squeeze(1)'),
               ('self.tokenized_prompts = tokenized_prompts', 'self.register_buffer("tokenized_prompts", tokenized_prompts)')]
    if corrected:
        patches += [('tokenized_prompts.argmax(dim=-1)', 'tokenized_prompts.argmax(dim=-1) + 16')]
    if structured:
        patches += [('loss = self.loss_ce(logits, label)', 'loss = self.loss_ce(logits, label) if label is not None else None'),
                    ('return Y_prob, Y_hat, loss', 'return {"logits": logits, "probabilities": Y_prob, "ce_loss": loss}')]
    return load_definitions('vilamil', 'models/model_ViLa_MIL.py',
        ['TextEncoder', 'PromptLearner', '_no_grad_trunc_normal_', 'trunc_normal_', 'ViLa_MIL_Model'],
        {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'warnings': warnings, 'clip': clip,
         '_tokenizer': types.SimpleNamespace(encode=lambda x: [0]), 'MultiheadAttention': attention.MultiheadAttention}, patches)


class ViLaMIL(nn.Module):
    def __init__(self, dim, classes, text, *, profile='vilamil', hidden=192, prototypes=16):
        super().__init__()
        if profile not in {'vilamil', 'vilamil_source'}:
            raise ValueError('Unknown ViLa recipe')
        if text is None or text.kind != 'learned_dual_scale_prompts':
            raise ValueError('ViLa needs descriptors, token ids and CLIP text weights')
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (hidden, prototypes)):
            raise ValueError('ViLa hidden/prototypes must be positive integers')
        clip = load_clip_text(text, dim)
        corrected = profile == 'vilamil'
        definitions = load_vila(clip, text.tokens, text.prompts, corrected=corrected)
        config = types.SimpleNamespace(input_size=dim, hidden_size=hidden, prototype_number=prototypes, text_prompt=text.prompts)
        self.network = definitions.ViLa_MIL_Model(config, num_classes=classes)
        self.dim, self.profile, self.text_metadata = dim, profile, text.metadata
        # Paper says frozen encoder; public source trains it. Expose both named profiles.
        if corrected:
            self.network.text_encoder.requires_grad_(False)

    def forward(self, low, high, label=None):
        for x in (low, high):
            if x.ndim != 2 or len(x) < 1 or x.shape[1] != self.dim or not torch.isfinite(x).all():
                raise ValueError('ViLa requires two finite nonempty [N,D] bags')
        return self.network(low, None, high, None, label)

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        if len(rows) != 1 or bundle.dual_scale is None:
            raise ValueError('ViLa requires one slide with two independently sampled scales')
        low = load_unit(bundle, rows, device, config['max_instances'], train)[0]
        high = bundle.dual_scale.load(rows[0]['slide_id']).to(device)
        limit = config['max_instances']
        if limit and len(high) > limit:
            idx = torch.randperm(len(high), device=device)[:limit] if train else torch.linspace(0, len(high)-1, limit, device=device).long()
            high = high[idx]
        target = torch.tensor([rows[0]['target']], device=device) if train else None
        return self(low, high, target)

    def protocol(self):
        corrected = self.profile == 'vilamil'
        return {'source': 'Jiangbo-Shi/ViLa-MIL; original prototype and context-guided decoders',
                'profile': self.profile, 'learned_context_tokens': 16,
                'text_encoder_trainable': not corrected, 'end_token_position': 'shifted by 16' if corrected else 'unshifted source behavior',
                'comparison_scope': 'paper-aligned frozen text with EOT fix' if corrected else 'literal public source training behavior',
                'optimizer': 'Adam; CE; source scheduler was constructed but never stepped',
                'preprocessing': 'independent low/high scale bags; native RN50 image/text width 1024; no fake second scale',
                'selection': 'PULS validation cross entropy; patient-disjoint task splits', 'text': self.text_metadata}


def vila_epoch(model, bundle, groups, optimizer, device, config):
    losses = []
    for rows in groups:
        output = model.forward_unit(bundle, rows, device, config, train=True)
        loss = output['ce_loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite ViLa loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append({'loss': float(loss.detach())})
    return losses

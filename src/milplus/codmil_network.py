"""Complete native CoD kernel with explicit C-way, full-D/T and physical-map ports.

Author definitions remain private Git objects. This implements the complete
released two-scale/masked-normal-text loss and explicit dimension/class ports,
not a generic attention baseline or a formal method admission.
"""
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, pinned_text, verified_source

COMMIT = '0fd748dc91ffd889bf13dbffd4db633b158809d6'


def pinned(relative):
    root, sha = verified_source('codmil')
    if sha != COMMIT:
        raise ValueError('Wrong CoD-MIL author source')
    return pinned_text(root, sha, relative)



def native_classes():
    state = {'torch': torch, 'nn': nn, 'F': F, 'Module': nn.Module,
             'Optional': Optional, 'Parameter': nn.Parameter, '_LinearWithBias': nn.Linear,
             'xavier_uniform_': nn.init.xavier_uniform_, 'xavier_normal_': nn.init.xavier_normal_,
             'constant_': nn.init.constant_}
    helpers = load_definitions('codmil', 'models/model_utils.py', ['MultiheadAttention'], state)
    code = pinned('models/model_CoT.py')
    replacements = [
        ('self.L = 1024', 'self.L = config.input_size'),
        ('self.fc_text = nn.Sequential(nn.Linear(self.L, self.L)',
         'self.fc_text = nn.Sequential(nn.Linear(config.text_dimension, self.L)'),
        ('MultiheadAttention(embed_dim=1024, num_heads=1)',
         'MultiheadAttention(embed_dim=config.input_size, num_heads=1)'),
        ('attention_only=False, labels=None):', 'attention_only=False, labels=None, mapping=None):'),
        ('text_feature[:2]', 'text_feature[:self.num_classes]'),
        ('text_feature[4:-1]', 'text_feature[2*self.num_classes:-1]'),
        ('text_feature[2:4]', 'text_feature[self.num_classes:2*self.num_classes]'),
        ('mask = torch.ones_like(A, dtype=torch.bool).squeeze()',
         'mask = torch.ones_like(A, dtype=torch.bool).reshape(-1)'),
    ]
    for before, after in replacements:
        assert before in code, before
        code = code.replace(before, after)
    lines = code.splitlines()
    match = [i for i, line in enumerate(lines) if line.strip().startswith('map = torch.load(')]
    assert len(match) == 1
    lines[match[0]] = '        map = mapping  # bound physical row map, no hardcoded file I/O'
    return load_definitions('codmil', 'models/model_CoT.py', ['attention_block', 'CoT'],
        {**state, 'MultiheadAttention': helpers.MultiheadAttention},
        replacements=[*replacements, (code.splitlines()[match[0]], lines[match[0]])])


class CoDReference(nn.Module):
    def __init__(self, dimension=1024, text_dimension=1024, classes=2, hidden=192):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in [dimension, text_dimension, classes, hidden]) or classes < 2:
            raise ValueError('CoD requires positive complete feature/text dimensions and at least two classes')
        self.dimension, self.text_dimension, self.classes = dimension, text_dimension, classes
        self.network = native_classes().CoT(SimpleNamespace(input_size=dimension,
            text_dimension=text_dimension, hidden_size=hidden), num_classes=classes)
        self.network.num_classes = classes

    def forward(self, coarse, fine, text, mapping, label=None):
        for name, value, dimension in [('coarse', coarse, self.dimension), ('fine', fine, self.dimension),
                                       ('text', text, self.text_dimension)]:
            if value.ndim != 2 or not len(value) or value.shape[1] != dimension or not torch.isfinite(value).all():
                raise ValueError('CoD complete '+name+' input differs from its frozen declared dimension')
        if text.shape[0] < 2*self.classes+2:
            raise ValueError('CoD requires ordered low/high C prompts, normal concepts and the source excluded-last prompt')
        if (mapping.ndim != 2 or mapping.shape[0] != len(coarse) or mapping.shape[1] < 1
                or mapping.dtype != torch.int64 or (mapping < -1).any() or (mapping >= len(fine)).any()):
            raise ValueError('CoD physical map must contain actual fine row IDs or -1, bound to all coarse rows')
        if any(value.device != coarse.device for value in [fine, text, mapping]):
            raise ValueError('CoD feature/text/map device mismatch')
        if label is None:
            target = torch.zeros(1, device=coarse.device, dtype=torch.long)
        else:
            target = torch.as_tensor(label, device=coarse.device).reshape(-1)
            if target.shape != (1,) or target.dtype != torch.int64 or not 0 <= int(target[0]) < self.classes:
                raise ValueError('CoD requires one actual bag class ID')
        result = self.network(coarse, None, fine, None, None, target, text, None, mapping=mapping)
        logits, probability, prediction, patches, attention, loss = result
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError('CoD nonfinite complete source objective')
        return {'logits': logits, 'probabilities': probability, 'prediction': prediction,
                'patch_prediction': patches, 'attention': attention, 'loss': loss if label is not None else None}

    def protocol(self):
        return {'source': COMMIT, 'feature_dimension': self.dimension, 'text_dimension': self.text_dimension,
            'classes': self.classes, 'gate_hidden': self.network.attention_block.D,
            'complete_objective': 'low+high text logits CE plus masked-low class/normal-text CE',
            'selection': 'top10percent coarse masking, top16 coarse-to-fine mapping; duplicate fine rows retained',
            'source_ports': 'Explicit full-D visual input, raw text TtoDtoD, C/2C class slices, supplied physical map instead of hidden path, singleton coarse mask reshape; modern Linear alias for old unused MHA output projection',
            'inactive_source_heads': ['classifier', 'self_attention'],
            'core_implementation': 'actual author complete numerical kernel with named input/class ports',
            'acceptance_evidence': 'tracked separately by release validation reports',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run'}

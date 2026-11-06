"""Pinned MMIL-Transformer with group masking and message-token communication."""
import random
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .stochastic import inference_rng
from .upstream import load_definitions


def load_mmil():
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'random': random, 'rearrange': rearrange, 'KMeans': KMeans}
    utils = load_definitions('mmil', 'utils/utils.py', ['cat_msg2cluster_group', 'split_array'], scope)
    scope.update({name: getattr(utils, name) for name in ('cat_msg2cluster_group', 'split_array')})
    core = load_definitions('mmil', 'utils/core.py', ['grouping'], scope, replacements=(
        ('np.argwhere(labels==i).squeeze()', 'np.argwhere(labels==i).reshape(-1)'),
        ('for i in range(groups_num):', 'for i in np.unique(labels):'),
        ('features = features.squeeze()', 'features = features.squeeze(0) if features.ndim == 3 else features'),
        ('coords = coords.squeeze()', 'coords = coords.squeeze(0) if coords.ndim == 3 else coords')))
    scope['grouping'] = core.grouping
    return load_definitions('mmil', 'models/models.py',
        ['Attention', 'AttenLayer', 'NyAttenLayer', 'GroupsAttenLayer', 'GroupsMSGAttenLayer',
         'BasicLayer', 'MultipleMILTransformer'], scope,
        replacements=(("print('try to group seq to ',self.args.num_subbags)", 'pass'),
            ('r = int(len(x_groups) * (1-mask_ratio))',
             'r = int(len(x_groups) * (1-mask_ratio))\n        if r < 1:\n            raise ValueError("MMIL masking must retain an actual nonempty group")')))


class MMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, groups=4, message_tokens=1, layers=2, grouping='random',
                 mask_ratio=.6, absolute_position=True, max_group_size=4300, evaluation_seed=2026):
        super().__init__()
        if min(hidden, groups, message_tokens, layers, max_group_size) < 1:
            raise ValueError('MMIL dimensions, groups, message tokens, layers and group budget must be positive')
        if grouping not in {'random', 'seq', 'embed', 'coords'}:
            raise ValueError('MMIL grouping must be random, seq, embed or coords')
        if not 0 <= mask_ratio < 1 or int(groups * (1-mask_ratio)) < 1:
            raise ValueError('MMIL masking must retain at least one group')
        args = SimpleNamespace(in_chans=dim, embed_dim=hidden, n_classes=classes, num_msg=message_tokens,
                               num_subbags=groups, num_layers=layers, mode=grouping, ape=absolute_position,
                               type='camelyon16')
        # Author projection route is selected for all FMs; this field does not set a dataset label.
        self.network = load_mmil().MultipleMILTransformer(args)
        self.network.grouping.max_size = max_group_size
        self.groups, self.grouping = groups, grouping
        self.mask_ratio, self.evaluation_seed = mask_ratio, evaluation_seed

    def forward(self, x, coords=None, label=None):
        if len(x) < self.groups:
            raise ValueError('MMIL requires at least as many instances as groups')
        if self.grouping == 'coords' and coords is None:
            raise ValueError('MMIL coordinate grouping requires real aligned coordinates')
        if self.training:
            result = self.network(x.unsqueeze(0), coords=coords, mask_ratio=self.mask_ratio)
        else:
            with inference_rng(x.device, self.evaluation_seed):
                result = self.network(x.unsqueeze(0), coords=coords, mask_ratio=0)
        return {'logits': result['logits'], 'probabilities': result['Y_prob']}

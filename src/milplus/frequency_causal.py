"""MFC-MIL author frequency, multi-resolution and memory intervention paths."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def load_mfc(backbone='abmil', flexible=True):
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math}
    memory = load_definitions('mfcmil', 'modules/causal.py', ['InstanceMemory', 'InterventionOnline'], scope,
        replacements=[('nn.Parameter(torch.randn(memory_size, d_model)).cuda()',
                       'nn.Parameter(torch.randn(memory_size, d_model))')])
    scale = load_definitions('mfcmil', 'modules/multi_level.py', ['DownSample', 'MultiLevelFuse'], scope)
    frequency = load_definitions('mfcmil', 'modules/fre_domain.py', ['FeatureFreProcessing'], scope)
    scope.update(InstanceMemory=memory.InstanceMemory, InterventionOnline=memory.InterventionOnline,
                 DownSample=scale.DownSample, MultiLevelFuse=scale.MultiLevelFuse,
                 FeatureFreProcessing=frequency.FeatureFreProcessing)
    if backbone == 'abmil':
        patches = [('reshape([1, -1, 512])', 'reshape([1, -1, self.embed_dim])'),
                   ('reshape([1, 1, 512])', 'reshape([1, 1, self.embed_dim])'),
                   ('reshape(1, 512)', 'reshape(1, self.embed_dim)'),
                   ('nn.Linear(self.L*self.K, 2)', 'nn.Linear(self.L*self.K, cfgs["model"]["num_classes"])')] if flexible else []
        model = load_definitions('mfcmil', 'models/abmil.py', ['Attention'], scope, replacements=patches)
    elif backbone == 'transmil':
        from nystrom_attention import NystromAttention
        scope['NystromAttention'] = NystromAttention
        patches = [('self.cls_token.expand(B, -1, -1).cuda()', 'self.cls_token.expand(B, -1, -1)')]
        if flexible:
            patches.append(('nn.Linear(self.embed_dim, 2)', 'nn.Linear(self.embed_dim, cfgs["model"]["num_classes"])'))
        model = load_definitions('mfcmil', 'models/transmil.py', ['TransLayer', 'PPEG', 'TransMIL'], scope,
                                replacements=patches)
    else:
        raise ValueError('MFC-MIL backbone must be abmil or transmil')
    return model, memory, scale, frequency


class FrequencyCausalMIL(nn.Module):
    def __init__(self, dim, classes, backbone='abmil', hidden=512, warmup_epochs=10):
        super().__init__()
        if any(not isinstance(v, int) or v < 2 for v in [dim, classes, hidden]):
            raise ValueError('MFC-MIL input/hidden widths and classes must be >=2')
        if backbone == 'transmil' and hidden % 8:
            raise ValueError('MFC TransMIL hidden width must be divisible by eight')
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError('MFC warmup_epochs must be a nonnegative integer')
        model, *_ = load_mfc(backbone)
        cfg = {'model': {'input_dim': dim, 'embed_dim': hidden, 'num_classes': classes}}
        self.network = (model.Attention if backbone == 'abmil' else model.TransMIL)(cfg)
        self.backbone, self.warmup_epochs = backbone, warmup_epochs
        self.total_epochs, self.steps_per_epoch, self.steps, self.base_lr = 100, 0, 0, None
        self.optimizer = None

    def configure_training(self, config):
        self.total_epochs = config['epochs']
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError('MFC training must include epochs after learning-rate warmup')

    def prepare(self, bundle, seed, device):
        from .engine import units
        self.steps_per_epoch = len(units(bundle, 'train'))

    def lr_factor(self):
        warmup, total = self.warmup_epochs*self.steps_per_epoch, self.total_epochs*self.steps_per_epoch
        if self.steps < warmup:
            return self.steps/max(1, warmup)
        progress = (self.steps-warmup)/max(1, total-warmup)
        return max(0., .5*(1+math.cos(math.pi*progress))*.999+.001)

    def make_optimizer(self, lr, weight_decay):
        self.base_lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr*self.lr_factor(),
                                          weight_decay=weight_decay, betas=(.95, .99), amsgrad=True)
        return self.optimizer

    def after_step(self):
        self.steps += 1
        for group in self.optimizer.param_groups:
            group['lr'] = self.base_lr*self.lr_factor()

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or len(x) < 16 or not torch.isfinite(x).all():
            raise ValueError('MFC-MIL requires finite [N,D] features with at least 16 instances for stride-16 branches')
        return self.network(x[None])

    def get_extra_state(self):
        return {'steps': self.steps, 'steps_per_epoch': self.steps_per_epoch}

    def set_extra_state(self, state):
        self.steps, self.steps_per_epoch = state['steps'], state['steps_per_epoch']

    def protocol(self):
        return {'backbone': self.backbone, 'source': 'pinned author MFC-MIL',
                'objective': 'CE; Adam betas=(.95,.99), AMSGrad; step linear warmup then cosine to .001 base lr',
                'warmup_epochs': self.warmup_epochs,
                'multi_scale': 'sequence square depthwise convolutions plus three dilated stride-16 filters and maxpool',
                'coordinates': 'author sequence-derived grid; does not represent measured physical multi-magnification tiles',
                'frequency': 'last source class definition; projected Hilbert analytic transform then real part',
                'memory': 'registered trainable slots plus functional per-bag write/read; no cross-bag mutation in forward',
                'intervention': 'both scales share self.causal as in source; self.causal_ll and self.fuse are unused',
                'compatibility': 'device-neutral registered memory, dimension-aware reshapes and class head',
                'selection': 'validation only; author per-epoch test logging not copied',
                'scope': 'full ABMIL and TransMIL source paths; other backbones not implied'}

"""Pinned MIL-Dropout networks and explicit CE/optimizer recipes."""
from math import ceil

import numpy as np
import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def retained_scale(mask):
    kept = mask.sum()
    if kept <= 0:
        raise ValueError('MIL-Dropout masked every instance; lower topk/kernel explicitly or use a larger bag')
    return mask.numel()/kept


def load_dropout_sources(safe=True):
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'ceil': ceil,
             'einsum': torch.einsum, 'rearrange': rearrange, 'reduce': reduce,
             'retained_scale': retained_scale}
    patches = [('torch.ones(length,1).cuda()', 'input.new_ones((length, 1))')]
    if safe:
        patches.append(('ratio = (mask.numel() / mask.sum())', 'ratio = retained_scale(mask)'))
    drop = load_definitions('mil_dropout', 'dropout.py', ['mildropout'], scope, replacements=patches)
    scope['mildropout'] = drop.mildropout
    attention = load_definitions('mil_dropout', 'abmil.py', ['AttentionMIL'], scope)
    dsmil = load_definitions('mil_dropout', 'dsmil.py', ['FCLayer', 'IClassifier', 'BClassifier', 'MILNet'], scope)
    nystrom = load_definitions('mil_dropout', 'nystrom_attention.py',
                               ['exists', 'moore_penrose_iter_pinv', 'NystromAttention'], scope)
    scope['NystromAttention'] = nystrom.NystromAttention
    transmil = load_definitions('mil_dropout', 'transmil_mildropout.py', ['TransLayer', 'PPEG', 'TransMIL'], scope,
        replacements=[('self.cls_token.expand(B, -1, -1).cuda()', 'self.cls_token.expand(B, -1, -1).to(h.device)')])
    return drop, attention, dsmil, transmil


class MILDropout(nn.Module):
    def __init__(self, dim, classes, backbone='abmil', topk=3, kernel=4, hidden=None,
                 lookahead_steps=5, lookahead_alpha=.5):
        super().__init__()
        if dim < 2 or classes < 2 or backbone not in {'abmil', 'gated', 'dsmil', 'transmil'}:
            raise ValueError('Invalid MIL-Dropout input width, classes or backbone')
        if not isinstance(topk, int) or topk < 1 or not isinstance(kernel, int) or kernel < 0:
            raise ValueError('MIL-Dropout requires integer topk>=1 and kernel>=0; topk=0 is a separate ablation')
        if not isinstance(lookahead_steps, int) or lookahead_steps < 1 or not 0 < lookahead_alpha <= 1:
            raise ValueError('Invalid Lookahead interval or interpolation coefficient')
        _, attn, dsmil, trans = load_dropout_sources()
        if backbone in {'abmil', 'gated'}:
            if hidden not in {None, 128}:
                raise ValueError('Author MIL-Dropout attention backbone has fixed 128-dimensional output')
            self.network = attn.AttentionMIL(dim, classes, attn_mode='gated' if backbone == 'gated' else 'linear',
                                             topk=topk, kernel=kernel)
        elif backbone == 'dsmil':
            hidden = 256 if hidden is None else hidden
            if not isinstance(hidden, int) or hidden < 1:
                raise ValueError('DSMIL hidden width must be positive')
            self.network = dsmil.MILNet(dsmil.FCLayer(hidden, classes),
                dsmil.BClassifier(hidden, 128, classes), L=hidden, in_features=dim, topk=topk, kernel=kernel)
        else:
            hidden = 512 if hidden is None else hidden
            if not isinstance(hidden, int) or hidden < 8 or hidden % 8:
                raise ValueError('TransMIL hidden width must be a positive multiple of eight')
            self.network = trans.TransMIL(dim, classes, mDim=hidden, topk=topk, kernel=kernel)
        self.backbone, self.topk, self.kernel = backbone, topk, kernel
        self.lookahead_steps, self.lookahead_alpha = lookahead_steps, lookahead_alpha
        self.slow, self.steps, self.scheduler, self.scheduler_state = {}, 0, None, None
        self.total_epochs = 200

    def configure_training(self, config):
        self.total_epochs = config['epochs']

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('MIL-Dropout requires finite nonempty [N,D] features')
        logits, attention, embedding = self.network(x)
        return {'logits': logits, 'embedding': embedding, **(
            {'attention': attention} if self.backbone != 'transmil' else {})}

    def make_optimizer(self, lr, weight_decay):
        optimizer = (torch.optim.RAdam if self.backbone == 'transmil' else torch.optim.AdamW)(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        if self.backbone == 'dsmil':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_epochs)
            if self.scheduler_state is not None:
                self.scheduler.load_state_dict(self.scheduler_state)
        elif not self.slow:
            self.slow = {name: p.detach().clone() for name, p in self.network.named_parameters()}
        return optimizer

    @torch.no_grad()
    def after_step(self):
        self.steps += 1
        if self.backbone != 'dsmil' and self.steps % self.lookahead_steps == 0:
            for name, p in self.network.named_parameters():
                if p.grad is not None:
                    slow = self.slow[name].to(p.device)
                    slow.add_(p-slow, alpha=self.lookahead_alpha)
                    p.copy_(slow)
                    self.slow[name] = slow

    def after_validation(self, metrics):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_extra_state(self):
        return {'steps': self.steps, 'slow': {k: v.cpu() for k, v in self.slow.items()},
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        self.steps, self.slow = state['steps'], state['slow']
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def protocol(self):
        return {'backbone': self.backbone, 'source': 'MILDropout locked author models',
                'topk': self.topk, 'kernel': self.kernel,
                'mask': 'sigmoid channel mean ranking, top-k plus union of cosine-nearest neighbors',
                'placement': 'three author shallow MLP ReLU outputs; zero rows retained, scale N/retained',
                'all_masked': 'explicit failure; no silent reduction or random fallback',
                'objective': 'bag cross entropy as paper Table 3; DSMIL does not add original dual-stream BCE',
                'optimizer': 'RAdam' if self.backbone == 'transmil' else 'AdamW',
                'update_rule': 'epoch cosine' if self.backbone == 'dsmil' else 'Lookahead fast weights for evaluation',
                'lookahead_steps': self.lookahead_steps, 'lookahead_alpha': self.lookahead_alpha,
                'recipe_limit': 'published optimizer families retained; unspecified Lookahead settings are explicit PULS defaults',
                'model_limit': 'uses published code widths; paper prose widths differ; original DTFD dropout variant not included',
                'dsmil_instance_branch': 'author FCLayer+ReLU used for discrete critical-index selection; bag-only CE leaves it without gradients'}

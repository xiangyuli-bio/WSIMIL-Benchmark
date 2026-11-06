"""S-DFP shared-prompt warmup and slide-aware joint training."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def load_sdfp_backbone(backbone, prune_unused=True):
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np}
    if backbone in {'abmil', 'gated'}:
        patches = [('self.size = self.size_dict[size]',
                    'self.size = size if isinstance(size, (list, tuple)) else self.size_dict[size]'),
                   (('H = self.feature_extractor_part1(x)\n        H = H.view(-1, 50 * 4 * 4)\n'
                    '        H = self.feature_extractor_part2(H)  # KxM'),
                    '# Unused RGB computation removed: active attention operates on x.')]
        if prune_unused:
            patches += [(
                ('self.feature_extractor_part1 = nn.Sequential(\n'
                '            nn.Conv2d(3, 20, kernel_size=5),\n'
                '            nn.ReLU(),\n'
                '            nn.MaxPool2d(2, stride=2),\n'
                '            nn.Conv2d(20, 50, kernel_size=5),\n'
                '            nn.ReLU(),\n'
                '            nn.MaxPool2d(2, stride=2)\n'
                '        )'), 'self.feature_extractor_part1 = nn.Identity()')]
            for width in ('140450', '50 * 4 * 4'):
                patches.append((('self.feature_extractor_part2 = nn.Sequential(\n'
                    f'            nn.Linear({width}, self.M),\n'
                    '            nn.ReLU(),\n        )'), 'self.feature_extractor_part2 = nn.Identity()'))
        return load_definitions('sdfp', 'model/ABMIL/model_abmil.py', ['Attention', 'GatedAttention'],
                                scope, replacements=patches)
    if backbone == 'dsmil':
        return load_definitions('sdfp', 'model/DSMIL/dsmil.py', ['FCLayer', 'BClassifier', 'MILNet'], scope)
    if backbone == 'transmil':
        from nystrom_attention import NystromAttention
        scope['NystromAttention'] = NystromAttention
        return load_definitions('sdfp', 'model/TransMIL/model_transmil.py', ['TransLayer', 'PPEG', 'TransMIL'], scope,
            replacements=[('self.size = self.size_dict[model_size]',
                'self.size = model_size if isinstance(model_size, (list, tuple)) else self.size_dict[model_size]'),
                ('self.cls_token.expand(B, -1, -1).cuda()', 'self.cls_token.expand(B, -1, -1)')])
    raise ValueError('S-DFP backbone must be abmil, gated, dsmil or transmil')


class SlidePromptMIL(nn.Module):
    def __init__(self, dim, classes, backbone='abmil', hidden=None, num_prompts=1,
                 aggregation='multiply', initialization='gaussian', warmup_epochs=20,
                 prompt_lr=.002, prompt_weight_decay=1e-5):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in [dim, num_prompts]) or classes < 2:
            raise ValueError('S-DFP needs positive input width/prompt count and at least two classes')
        if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
            raise ValueError('S-DFP requires a positive shared-prompt warmup stage')
        if not math.isfinite(prompt_lr) or prompt_lr <= 0 or not math.isfinite(prompt_weight_decay) or prompt_weight_decay < 0:
            raise ValueError('S-DFP prompt learning rate and regularization must be valid')
        if backbone == 'dsmil' and hidden is not None:
            raise ValueError('Author S-DFP DSMIL uses the original feature width; hidden is not a configurable projection')
        hidden = {192: 128, 384: 192}.get(dim, 512) if hidden is None else hidden
        if not isinstance(hidden, int) or hidden < 1 or (backbone == 'transmil' and hidden % 8):
            raise ValueError('Invalid S-DFP hidden width; TransMIL requires a multiple of eight')
        module = load_sdfp_backbone(backbone)
        if backbone in {'abmil', 'gated'}:
            self.network = (module.Attention if backbone == 'abmil' else module.GatedAttention)([dim, hidden], classes)
        elif backbone == 'dsmil':
            self.network = module.MILNet(module.FCLayer(dim, classes), module.BClassifier(dim, classes))
        else:
            self.network = module.TransMIL([dim, hidden], classes)
        prompt = load_definitions('sdfp', 'model/s_dfp.py', ['SDFP'], {'torch': torch, 'nn': nn})
        self.prompt = prompt.SDFP(dim, num_prompts, aggregation=aggregation, init=initialization)
        self.backbone, self.warmup_epochs = backbone, warmup_epochs
        self.minimum_selection_epoch = warmup_epochs
        self.prompt_lr, self.prompt_weight_decay = prompt_lr, prompt_weight_decay
        self.epoch, self.steps, self.steps_per_epoch, self.total_epochs = 0, 0, 0, 50
        self.optimizer = None

    def configure_training(self, config):
        self.total_epochs = config['epochs']
        if self.total_epochs <= self.warmup_epochs:
            raise ValueError('S-DFP requires both shared-prompt and slide-aware training epochs')

    def prepare(self, bundle, seed, device):
        from .engine import units
        self.steps_per_epoch = len(units(bundle, 'train'))

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.prompt.parameters(), 'lr': self.prompt_lr, 'weight_decay': self.prompt_weight_decay},
            {'params': self.network.parameters(), 'lr': lr, 'weight_decay': weight_decay}])
        return self.optimizer

    def step_lr(self):
        total = self.total_epochs*self.steps_per_epoch
        if not total:
            raise ValueError('S-DFP schedule needs the training bag count before optimizer steps')
        warmup = total//5
        if self.steps < warmup:
            factor = (self.steps+1)/warmup
        else:
            factor = .5*(1+math.cos(math.pi*(self.steps-warmup)/(total-warmup)))
        # This shared rate is the actual author runner behavior, even if the
        # classifier group had a different initial rate in _build_optimizer.
        for group in self.optimizer.param_groups:
            group['lr'] = self.prompt_lr*factor

    def start_epoch(self, epoch):
        self.epoch = epoch

    def after_step(self):
        self.steps += 1

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('S-DFP requires finite nonempty [N,D] features')
        if self.training and self.optimizer is not None:
            self.step_lr()
        h = self.prompt(x, slide_aware=not self.training or self.epoch >= self.warmup_epochs)
        if self.backbone == 'transmil':
            return self.network(data=h[None])
        if self.backbone == 'dsmil':
            instances, logits, attention, embedding = self.network(h)
            out = {'logits': logits, 'attention': attention.T, 'embedding': embedding}
            if self.training and label is not None:
                out['total_loss'] = .5*(F.cross_entropy(instances, label.expand(len(instances)))+
                                        F.cross_entropy(logits, label))
            return out
        logits, _, _, attention = self.network(h)
        return {'logits': logits, 'attention': attention}

    def get_extra_state(self):
        return {'epoch': self.epoch, 'steps': self.steps, 'steps_per_epoch': self.steps_per_epoch}

    def set_extra_state(self, state):
        self.epoch, self.steps, self.steps_per_epoch = state['epoch'], state['steps'], state['steps_per_epoch']

    def protocol(self):
        return {'source': 'pinned author S-DFP', 'backbone': self.backbone,
                'aggregation': self.prompt.aggregation, 'num_prompts': self.prompt.num_prompts,
                'warmup_epochs': self.warmup_epochs, 'stages': 'joint shared-prompt + MIL, then joint slide-modulation + prompt + MIL',
                'classifier_frozen': False, 'modulation': 'two-layer MLP of mean patch features, shared delta across prompt tokens',
                'objective': 'half all-instance CE plus half bag CE' if self.backbone == 'dsmil' else 'bag CE',
                'inference': 'always slide-aware, as author runner; only stage-2 checkpoints eligible for selection',
                'optimizer': 'Adam; per-step cosine with first fifth linear warmup; all groups use scheduled prompt_lr',
                'prompt_lr': self.prompt_lr, 'prompt_weight_decay': self.prompt_weight_decay,
                'compatibility': 'arbitrary feature widths, device-neutral CLS; unused RGB extractors removed from attention models',
                'unused_author_behavior': 'SDFP defines dropout but never calls it; preserved',
                'selection_fix': 'fixed validation only; author test-based checkpoint selection not copied',
                'scope': 'four explicitly named backbones; CLAM/FRMIL/ACMIL prompt recipes remain separate work'}

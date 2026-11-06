"""Channel-attention CAMIL: author MCAB blocks and Lookahead/RAdam recipe."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def load_channel_camil(hidden=512):
    from nystrom_attention import NystromAttention
    patches = [] if hidden == 512 else [
        ('nn.Linear(input_dim, 512)', f'nn.Linear(input_dim, {hidden})'),
        ('Block(dim=512,temperature=', f'Block(dim={hidden},temperature='),
        ('self.norm = nn.LayerNorm(512)', f'self.norm = nn.LayerNorm({hidden})'),
        ('L=512, D=256, dropout=False, n_classes=1)', f'L={hidden}, D={hidden//2}, dropout=False, n_classes=1)'),
        ('nn.Linear(512, self.n_classes)', f'nn.Linear({hidden}, self.n_classes)')]
    return load_definitions('camil_channel', 'models/CAMIL.py',
        ['TransLayer', 'MCAB', 'CAM', 'Block', 'initialize_weights', 'Attn_Net', 'Attn_Net_Gated', 'CAMIL'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'NystromAttention': NystromAttention}, patches)


def author_radam():
    return load_definitions('camil_channel', 'MyOptimizer/radam.py', ['RAdam'],
        {'torch': torch, 'math': math, 'Optimizer': torch.optim.Optimizer}).RAdam


class ChannelCAMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, layers=4, temperature=1.2, dropout=.15,
                 gated=True, accumulation=2):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in [dim, classes, hidden, layers, accumulation]):
            raise ValueError('CAMIL channel dimensions, layers and accumulation must be positive integers')
        if classes < 2 or hidden % 8 or hidden < 8 or not 0 <= dropout < 1:
            raise ValueError('CAMIL channel requires classes>=2, hidden multiple of eight and dropout in [0,1)')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('CAMIL channel temperature must be finite and positive')
        self.network = load_channel_camil(hidden).CAMIL(classes, input_dim=dim,
            temperature=temperature, dropout=dropout, n_layers=layers, gate=gated)
        self.accumulation, self.steps, self.slow = accumulation, 0, {}

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('CAMIL channel requires finite nonempty [N,D] features')
        return self.network(data=x[None])

    def make_optimizer(self, lr, weight_decay):
        groups = [{'params': [p for n, p in self.network.named_parameters() if p.ndim == 1 or n.endswith('.bias')],
                       'weight_decay': 0.},
                  {'params': [p for n, p in self.network.named_parameters() if p.ndim != 1 and not n.endswith('.bias')],
                       'weight_decay': weight_decay}]
        # Author RAdam has decoupled decay and an N_sma >= 5 boundary. Modern
        # torch.optim.RAdam is not substituted for this published optimizer.
        return author_radam()(groups, lr=lr, weight_decay=0.)

    @torch.no_grad()
    def after_step(self):
        self.steps += 1
        if self.steps % 6 == 0:
            for name, p in self.network.named_parameters():
                if p.grad is not None:
                    # Author Lookahead first creates slow weights at step six,
                    # rather than at initialization. Preserve that behavior.
                    slow = self.slow.get(name, p.detach().clone()).to(p.device)
                    slow.add_(p-slow, alpha=.5)
                    p.copy_(slow)
                    self.slow[name] = slow

    def get_extra_state(self):
        return {'steps': self.steps, 'slow': {k: v.cpu() for k, v in self.slow.items()}}

    def set_extra_state(self, state):
        self.steps, self.slow = state['steps'], state['slow']

    def protocol(self):
        return {'source': 'pinned channel-attention CAMIL; not context-aware CAMIL',
                'architecture': 'four default Nystrom/MCAB blocks, LayerNorm, gated attention and class head',
                'spatial': 'zero-padded sequence square; no claim of true physical coordinates or magnifications',
                'channel': 'shared max/mean MLP, three independent sigmoid branch gates; multiply temperature',
                'objective': 'cross entropy', 'optimizer': 'author RAdam, bias/1D no decay; Lookahead k=6 alpha=.5',
                'accumulation': self.accumulation,
                'tail': 'loss divided by configured accumulation even for short final window, as Lightning recipe',
                'lookahead_fix': 'slow state keyed by parameter names rather than transient Python object IDs',
                'precision': 'FP32; author AMP setting not claimed reproduced',
                'selection': 'fixed validation metric; no test-based selection'}


def channel_epoch(model, bundle, groups, optimizer, device, config):
    from .engine import load_unit
    optimizer.zero_grad(set_to_none=True)
    result = []
    for i, rows in enumerate(groups):
        x, coords = load_unit(bundle, rows, device, config['max_instances'], train=True)
        loss = F.cross_entropy(model(x, coords=coords)['logits'], torch.tensor([rows[0]['target']], device=device))
        (loss/model.accumulation).backward()
        result.append({'loss': float(loss.detach())})
        if (i+1) % model.accumulation == 0 or i+1 == len(groups):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            model.after_step()
            optimizer.zero_grad(set_to_none=True)
    return result

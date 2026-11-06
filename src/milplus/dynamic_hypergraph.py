"""Pinned DyHG hyperedge sampling and attention aggregation."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .stochastic import inference_rng
from .upstream import load_definitions


def load_dyhg(singleton_fix=True):
    patches = [('features = torch.sum(features, dim=1).squeeze(1)',
                'features = torch.sum(features, dim=1)')] if singleton_fix else []
    return load_definitions('dyhg', 'DyHG.py', ['Attn_Net_Gated', 'DyHG'],
                            {'torch': torch, 'nn': nn, 'F': F}, replacements=patches)


class DynamicHypergraphMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, hyperedges=4, temperature=.05, dropout=.25,
                 inference_seed=2026):
        super().__init__()
        if any(not isinstance(x, int) or x < 2 for x in [dim, classes, hidden, hyperedges]):
            raise ValueError('DyHG dimensions, classes and hyperedge count must be integers >=2')
        if not math.isfinite(temperature) or temperature <= 0 or not 0 <= dropout < 1:
            raise ValueError('DyHG requires positive temperature and dropout in [0,1)')
        self.network = load_dyhg().DyHG(in_dim=dim, emb_dim=hidden, n_classes=classes,
            dropout=dropout, hyper_num=hyperedges, tau=temperature, num_layers=1, device='cpu')
        self.inference_seed = inference_seed

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('DyHG requires finite nonempty [N,D] features')
        if self.training:
            logits = self.network(x[None])
        else:
            # Author gumbel_softmax remains stochastic in eval. Retain the draw,
            # isolate its seed, and never replace it by a deterministic softmax.
            with inference_rng(x.device, self.inference_seed):
                logits = self.network(x[None])
        return {'logits': logits}

    def protocol(self):
        return {'source': 'pinned author DyHG', 'objective': 'cross entropy; Adam',
                'hyperedges': self.network.hyper_num, 'temperature': self.network.tau,
                'structure': 'learned low-rank incidence, soft Gumbel assignment, node-edge-node messages',
                'normalization': 'author L2 across nodes, one message pass plus residual divided by two',
                'inference': 'one author Gumbel draw with isolated fixed seed', 'inference_seed': self.inference_seed,
                'coordinates': 'not required: hypergraph derived from features',
                'unused_author_parameters': ['fc2'],
                'compatibility_fix': 'preserve singleton node axis before gated attention',
                'selection': 'one fixed validation metric; no selection among test metrics'}

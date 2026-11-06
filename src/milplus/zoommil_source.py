"""The full pinned three-level ZoomMIL and its actual perturbed Top-K gradient.

Only device allocation is ported: the released model guesses CUDA at creation,
which breaks later CPU placement on CUDA hosts. All levels, original Kronecker
selection,100 Gaussian samples, auxiliary attentions and sum head remain source.
The paper's two-level efficient/online-FM conditions are separate unimplemented
conditions, not silently replaced by this released cached-three-level graph.
"""
import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, verified_source

COMMIT = 'da7bb7fa71629d34e9b04859c1db791ac8330f5d'


@lru_cache(maxsize=1)
def classes():
    _, revision = verified_source('zoommil')
    if revision != COMMIT:
        raise ValueError('ZoomMIL source differs from the reviewed fixed implementation')
    scope = {'torch': torch, 'nn': nn, 'F': F}
    topk = load_definitions('zoommil', 'zoommil/utils/perturbedtopk.py',
        ['PerturbedTopK', 'PerturbedTopKFunction'], scope)
    model = load_definitions('zoommil', 'zoommil/models/zoommil.py',
        ['GatedAttention', 'ZoomMIL'], {**scope, 'PerturbedTopK': topk.PerturbedTopK},
        [('device=self.device', 'device=x2.device')])
    return model, topk


def build_model(dimension, classes_count, k=12, sigma=.002, dropout=.25):
    if (type(dimension) is not int or dimension < 1 or type(classes_count) is not int or classes_count < 2
            or type(k) is not int or k < 1 or not isinstance(sigma, (float, int)) or not math.isfinite(sigma) or sigma <= 0
            or not isinstance(dropout, (float, int)) or not 0 <= dropout < 1):
        raise ValueError('ZoomMIL needs full positive dimension/classes/K/sigma and valid dropout')
    model, _ = classes()
    return model.ZoomMIL(in_feat_dim=dimension, hidden_feat_dim=256, out_feat_dim=512,
                         dropout=dropout, k_sample=k, k_sigma=sigma, n_cls=classes_count)

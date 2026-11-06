"""Pinned full ConMIL KAN/semantic-attention/source training operators."""
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, verified_source

COMMIT = 'aabb5386d15a6002a3a39437ea46c0d99d29a2f3'
PROFILES = {'conmil_pos_source', 'conmil_posneg_source', 'conmil_posneg_multiclass_repaired'}


def definitions():
    _, sha = verified_source('conmil')
    if sha != COMMIT:
        raise ValueError('ConMIL requires its audited original author source')
    return load_definitions('conmil', 'text_guided_mil_kan.py',
        ['KANLinear', 'KAN', 'TextGuidedMIL_KAN', 'build_text_guided_mil_kan'],
        {'torch': torch, 'nn': nn, 'F': F})


def build_model(classes, profile, kan_grid=5):
    if profile not in PROFILES or type(classes) is not int or classes < 2:
        raise ValueError('ConMIL needs an explicit complete semantic-attention/KAN route')
    if profile != 'conmil_posneg_multiclass_repaired' and classes != 2:
        raise ValueError('Original ConMIL trainer and label/evaluation branch are binary; select the named C-way repair')
    if type(kan_grid) is not int or kan_grid < 1:
        raise ValueError('ConMIL B-spline grid size must be a positive integer')
    source = definitions()
    model = source.build_text_guided_mil_kan(in_dim=512, hidden_dim=256,
        n_classes=classes, dropout=.25, alpha_init=.5,
        use_neg=profile != 'conmil_pos_source', neg_weight=.5, kan_grid=kan_grid)
    # The author registers an expanded zero-stride grid buffer. Its values
    # serialize, but ordinary state_dict restoration fails on overlapping
    # destination memory. Materialize the identical immutable grid; retain all
    # knot values, spline equations, initialization draws and trainable tensors.
    for layer in model.modules():
        if isinstance(layer, source.KANLinear):
            layer.grid = layer.grid.clone()
    return model

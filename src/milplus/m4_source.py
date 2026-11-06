"""Complete original M4 network used by the registered development profile.

All multi-proxy experts, spatial convolutions, four task-gate slices, sigmoid
heads and original unused expert classifiers remain present. The complete trainer and immutable target binding live in m4.py; native and
release acceptance remain separate gates.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, verified_source

COMMIT = '2b0f92ef872fcd9997f1ae6b4ba6565fc53eb332'
NAMES = ['M4', 'MP_AMIL', 'Attn_Net_Gated', 'Tower', 'initialize_weights',
         'Aggregator', 'SeparableConv2d', 'ConvPosEnc']


def author_classes():
    _, revision = verified_source('m4')
    if revision != COMMIT:
        raise ValueError('M4 must use the reviewed complete original source revision')
    return load_definitions('m4', 'models/M4.py', NAMES,
                            {'np': np, 'torch': torch, 'nn': nn, 'F': F})


def build_original(dimension=2048, tasks=10, experts=5):
    for name, value in [('feature dimension', dimension), ('endpoint count', tasks), ('expert count', experts)]:
        if type(value) is not int or value < 1:
            raise ValueError('M4 '+name+' must be a positive explicit integer')
    # Input dimension is the explicit author feat_input parameter. Native source
    # uses2048D RetCCL; all internal512D/4gate/32tower widths stay unchanged.
    return author_classes().M4(feat_input=dimension, experts_out=512,
        towers_out=1, towers_hidden=32, tasks=tasks, num_expert=experts)

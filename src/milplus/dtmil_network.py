"""Complete pinned DT-MIL; exact author Torch deformable sampling reference."""
import argparse
import ast
import copy
import math
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import constant_, normal_, xavier_uniform_

from .upstream import load_definitions, pinned_text, verified_source

COMMIT = 'f52530811d0b1b591e10eab2fad9fd2ac1573b98'
SOURCE = 'dtmil_source_crop_reference_train'
REPAIRED = 'dtmil_repaired_reference_train'


@lru_cache(maxsize=1)
def author_modules():
    root, sha = verified_source('dtmil')
    if sha != COMMIT:
        raise ValueError('DT-MIL source differs from reviewed fixed version')
    scope = {'torch': torch, 'Tensor': Tensor, 'Optional': Optional, 'nn': nn, 'F': F,
        'math': math, 'np': np, 'copy': copy, 'warnings': warnings, 'Dict': dict, 'List': list,
        'constant_': constant_, 'normal_': normal_, 'xavier_uniform_': xavier_uniform_}

    def bind(relative, names=None, replacements=()):
        if names is None:
            names = [n.name for n in ast.parse(pinned_text(root, sha, relative)).body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        module = load_definitions('dtmil', relative, names, scope, replacements)
        scope.update({name: getattr(module, name) for name in names})

    bind('util/misc.py', ['NestedTensor', 'inverse_sigmoid'])
    bind('models/ops/functions/ms_deform_attn_func.py', ['ms_deform_attn_core_pytorch'])
    operator = scope['ms_deform_attn_core_pytorch']

    class AuthorTorchReference:
        @staticmethod
        def apply(value, shapes, starts, locations, weights, im2col_step):
            expected = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])
            if not torch.equal(starts, expected):
                raise ValueError('DT-MIL deformable level offsets are inconsistent')
            return operator(value, shapes, locations, weights)

    scope['MSDeformAttnFunction'] = AuthorTorchReference
    bind('models/ops/modules/ms_deform_attn.py')
    bind('models/position_encoding.py')
    bind('models/backbone.py', ['FeatureMapDrBackbone', 'FeatureMapJoiner', 'build_WSIFeatureMapBackbone'])
    # A zero valid ratio arises from the author's symmetric padding. For the
    # single source level, normalized grid/ratio*ratio cancels identically.
    # Replace only zero denominators, preserving every positive source ratio.
    bind('models/deformable_transformer.py', replacements=[(
        '        return valid_ratio\n',
        '        return torch.where(valid_ratio > 0, valid_ratio, torch.ones_like(valid_ratio))\n')])
    bind('models/deformable_detr_wsi_feat.py')
    parser = load_definitions('dtmil', 'main.py', ['get_args_parser'],
                              {'argparse': argparse, 'np': np}).get_args_parser
    return scope, parser


def network(dimension, classes, hidden=256, encoder_layers=6, decoder_layers=6,
            heads=8, feedforward=1024, dropout=.1, queries=100, points=4):
    scope, parser = author_modules()
    args = parser().parse_args([])
    args.device, args.num_input_channels, args.num_class = 'cpu', dimension, classes
    args.hidden_dim, args.out_channels, args.enc_layers, args.dec_layers = hidden, hidden, encoder_layers, decoder_layers
    args.nheads, args.dim_feedforward, args.dropout = heads, feedforward, dropout
    args.num_queries, args.enc_n_points, args.dec_n_points = queries, points, points
    # The published DT-MIL path is one level, ordinary decoder self/cross
    # attention following deformable encoder; unrelated DETR flags not exposed.
    args.num_feature_levels, args.two_stage, args.drop_decoder = 1, False, False
    return scope['build'](args)

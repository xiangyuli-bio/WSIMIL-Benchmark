"""Pinned KMG-Mamba graph and cross-scale scan; explicit numerical backends."""
import importlib
import math
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from .io import digest_file
from .mamba_mil import selective_scan_torch
from .upstream import load_definitions, verified_source

COMMIT = '4b5075bb2d3eddf46f9bc4f46175d250b134e27c'
PROFILES = {'kmgmamba_source', 'kmgmamba_multiclass_repaired'}


class LegacyDropPath(nn.Module):
    """timm0.4.12 uniform+floor implementation, including its RNG consumption."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep = 1-self.drop_prob
        mask = keep+torch.rand((x.shape[0],)+(1,)*(x.ndim-1), dtype=x.dtype, device=x.device)
        return x.div(keep)*mask.floor_()


def source_svm():
    scope = dict(torch=torch, nn=nn, np=np, math=math, ag=torch.autograd)
    helpers = load_definitions('kmgmamba', 'models/topk/utils.py', ['delta', 'detect_large'], scope)
    lse = load_definitions('kmgmamba', 'models/topk/polynomial/sp.py', ['log_sum_exp'], scope)
    scope.update(delta=helpers.delta, detect_large=helpers.detect_large, log_sum_exp=lse.log_sum_exp)
    functions = load_definitions('kmgmamba', 'models/topk/functional.py',
                                 ['Top1_Hard_SVM', 'Top1_Smooth_SVM'], scope)
    scope.update(Top1_Hard_SVM=functions.Top1_Hard_SVM, Top1_Smooth_SVM=functions.Top1_Smooth_SVM)
    return load_definitions('kmgmamba', 'models/topk/svm.py', ['_SVMLoss', 'SmoothTop1SVM'], scope).SmoothTop1SVM


def scan_backend(backend, chunk_size):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('KMG-Mamba scan_chunk must be a positive integer')
    if backend == 'torch':
        def scan(*args, **kwargs):
            return selective_scan_torch(*args, **kwargs, chunk_size=chunk_size)
        return scan, {'name': 'torch', 'algorithm': 'exact variable-B/C selective recurrence with retained state',
                      'chunk_size': chunk_size, 'torch': torch.__version__}
    if backend != 'cuda':
        raise ValueError('KMG-Mamba scan_backend must explicitly be torch or cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('KMG-Mamba CUDA scan requested without CUDA; no backend substitution')
    try:
        operators = importlib.import_module('mamba_ssm.ops.selective_scan_interface')
        binary = importlib.import_module('selective_scan_cuda')
    except ImportError as error:
        raise RuntimeError('KMG-Mamba CUDA scan requires the installed selective_scan operator; no fallback') from error
    return operators.selective_scan_fn, {'name': 'cuda', 'binary_sha256': digest_file(binary.__file__),
        'interface_sha256': digest_file(operators.__file__), 'torch': torch.__version__, 'cuda': torch.version.cuda}


def source_definitions(scan, classes, repaired=False, detailed=True):
    _, commit = verified_source('kmgmamba')
    if commit != COMMIT:
        raise ValueError('KMG-Mamba source is not the reviewed fixed commit')
    if type(classes) is not int or classes < 2 or (not repaired and classes != 2):
        raise ValueError('The released KMG-Mamba attention and prototype heads are binary; use the explicit multiclass repair')
    scope = dict(torch=torch, nn=nn, F=F, np=np, math=math, repeat=repeat, rearrange=rearrange,
                 partial=partial, Any=Any, Callable=Callable, checkpoint=checkpoint,
                 DropPath=LegacyDropPath, selective_scan_fn_v1=scan)
    cross = load_definitions('kmgmamba', 'models/vmamba.py',
        ['Cross_Mamba_Attention_SSM', 'CrossMambaFusion_SS2D_SSM', 'Mlp', 'CrossMambaFusionBlock'], scope)
    attention = load_definitions('kmgmamba', 'models/datten.py',
        ['Attention', 'AttentionGated', 'DAttention'], scope,
        replacements=[('self.K = 2', f'self.K = {classes}')] if repaired else [])
    scope.update(CrossMambaFusionBlock=cross.CrossMambaFusionBlock, DAttention=attention.DAttention,
                 SmoothTop1SVM=source_svm())
    changes = [('SmoothTop1SVM(2).cuda()', f'SmoothTop1SVM({classes})'),
               ('torch.empty(1, self.n_classes).float().cuda()', 'Z.new_empty(1, self.n_classes)')]
    if repaired:
        changes.append(('self.instance_classifier = nn.Linear(config.hidden_size, 2)',
                        f'self.instance_classifier = nn.Linear(config.hidden_size, {classes})'))
    if detailed:
        changes.append(('return Y_prob, Y_hat, loss',
                        'return Y_prob, Y_hat, loss, logits, loss_ce, loss_ins, loss_ins_l'))
    definitions = load_definitions('kmgmamba', 'models/model_MG_Mamba.py',
        ['initialize_weights', 'Embeddings', 'WiKG', 'GraphRepresentation', 'CrossMamba', 'MG_Mamba_main'],
        scope, replacements=changes)
    return definitions


def build_model(dimension, classes, profile='kmgmamba_source', backend='torch', scan_chunk=64,
                sequence_length_min=0):
    if profile not in PROFILES or type(dimension) is not int or dimension < 1:
        raise ValueError('Invalid KMG-Mamba profile or encoder dimension')
    if type(sequence_length_min) is not int or sequence_length_min < 0:
        raise ValueError('KMG-Mamba source repeat threshold must be nonnegative')
    if profile == 'kmgmamba_multiclass_repaired' and classes < 3:
        raise ValueError('The multiclass repair is reserved for tasks with at least three classes')
    scan, identity = scan_backend(backend, scan_chunk)
    definitions = source_definitions(scan, classes, repaired=profile.endswith('_repaired'))
    model = definitions.MG_Mamba_main(SimpleNamespace(input_size=dimension, hidden_size=96),
        img_size=3000, num_classes=classes, sequence_length_min=sequence_length_min,
        smoothing_value=0., zero_head=True)
    return model, identity

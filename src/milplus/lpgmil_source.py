"""Private pinned LPGMIL and its environment's complete Mamba1.1.1 graph.

The released LPGMIL cannot run: its trainable1024D queries enter512D attention.
The named reference projects both frozen and learnable prototypes through the
same published1024->512 projection. No attention branch or SSM is replaced.
"""
import importlib
import importlib.metadata
import math
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from .io import digest_file
from .mamba_mil import inner_torch, selective_scan_torch
from .upstream import load_definitions

COMMIT = '6b14bb01dff042a09bd34ebe956d696e665a7cd0'
MAMBA_COMMIT = 'fb7b5310fa865dbd62aa059b1e26f2b431363e2a'


def native_mamba(backend='torch', scan_chunk=64):
    if type(scan_chunk) is not int or scan_chunk < 1:
        raise ValueError('LPGMIL scan_chunk must be a positive integer')
    if backend == 'torch':
        def inner(xz, cw, cb, xw, dw, ow, ob, A, *args, **kwargs):
            y = inner_torch(xz, cw, cb, xw, dw, A, *args, **kwargs, chunk_size=scan_chunk)
            return F.linear(rearrange(y, 'b d l -> b l d'), ow, ob)

        def scan(*args, **kwargs):
            if kwargs.pop('return_last_state', False):
                raise ValueError('LPGMIL full-bag execution has no recurrent inference cache')
            return selective_scan_torch(*args, **kwargs, chunk_size=scan_chunk)

        identity = {'backend':'torch_exact_recurrence', 'scan_chunk':scan_chunk,
                    'mamba_commit':MAMBA_COMMIT, 'torch':torch.__version__}
    elif backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Explicit LPGMIL CUDA backend needs CUDA; no fallback')
        versions = {p:importlib.metadata.version(p) for p in ['mamba-ssm','causal-conv1d']}
        if versions != {'mamba-ssm':'1.1.1','causal-conv1d':'1.1.3.post1'}:
            raise ValueError('Build source-pinned mamba-ssm1.1.1 and causal-conv1d1.1.3.post1; no newer-model substitution')
        ops = importlib.import_module('mamba_ssm.ops.selective_scan_interface')
        inner, scan = ops.mamba_inner_fn, ops.selective_scan_fn
        identity = {'backend':'cuda', 'versions':versions, 'mamba_commit':MAMBA_COMMIT,
            'binaries':{n:digest_file(importlib.import_module(n).__file__)
                        for n in ['selective_scan_cuda','causal_conv1d_cuda']}, 'torch':torch.__version__}
    else:
        raise ValueError('LPGMIL scan_backend must explicitly be torch or cuda')
    ns = {'torch':torch,'nn':nn,'F':F,'Tensor':Tensor,'Optional':Optional,'math':math,
          'rearrange':rearrange,'repeat':repeat,'mamba_inner_fn':inner,'selective_scan_fn':scan,
          'causal_conv1d_fn':None,'causal_conv1d_update':None,'selective_state_update':None}
    cls = load_definitions('lpgmil_mamba111','mamba_ssm/modules/mamba_simple.py',['Mamba'],ns).Mamba
    return cls, identity


def native_network(feature_dim=1024, backend='torch', scan_chunk=64, repaired=True):
    mamba, identity = native_mamba(backend,scan_chunk)
    ns = {'torch':torch,'nn':nn,'F':F,'Tensor':Tensor,'math':math,'Mamba':mamba}
    replacements = []
    if repaired:
        replacements = [
            ('self.projection = SingleLayerMLP(1024, 512)',
             f'self.projection = SingleLayerMLP({feature_dim}, 512)'),
            ('self.prototype_selector = prototypes.unsqueeze(0) if use_topk else None',
             'self.register_buffer("prototype_selector", prototypes.unsqueeze(0).detach().clone() if use_topk else None)'),
            ('token_query = self.prototypes[:, idx:idx+1]',
             'token_query = self.projection(self.prototypes[:, idx:idx+1])')]
    names = ['initialize_weights','Classifier1FC','SingleLayerMLP','MultiHeadAttention','Aggratt','LPGMIL']
    mod = load_definitions('lpgmil','Model/LPGMIL.py',names,ns,replacements=replacements)
    l1 = load_definitions('lpgmil','utils/core_utils.py',['l1_reg_all'],{'torch':torch}).l1_reg_all
    return mod.LPGMIL, l1, identity


def source_inner_reference():
    """Independent published Mamba reference equations, not the adapter implementation."""
    def causal(x, weight, bias, activation):
        assert activation == 'silu'
        return F.silu(F.conv1d(x,weight.unsqueeze(1),bias,padding=weight.shape[-1]-1,
                               groups=x.shape[1])[...,:x.shape[-1]])
    ns = {'torch':torch,'F':F,'rearrange':rearrange,'repeat':repeat,'causal_conv1d_fn':causal}
    ref = load_definitions('lpgmil_mamba111','mamba_ssm/ops/selective_scan_interface.py',
                          ['selective_scan_ref'],ns).selective_scan_ref
    ns['selective_scan_fn'] = ref
    return load_definitions('lpgmil_mamba111','mamba_ssm/ops/selective_scan_interface.py',
                            ['mamba_inner_ref'],ns).mamba_inner_ref

"""Original MGGM-MIL graph and complete sequence-reordering Mamba operators."""
import math
import importlib.metadata
import importlib
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat

from .io import digest_file
from .mamba_mil import backend_functions
from .upstream import verified_source, load_definitions, load_single

COMMIT = 'dfbe91120e0c7c822abdbbba758c307ddabd2826'


def checked_source():
    root, sha = verified_source('mggmmil')
    if sha != COMMIT:
        raise ValueError('MGGM-MIL requires its locked original graph/SRMamba source')
    return root, sha


def author_classes(backend='torch', scan_chunk=64):
    root, sha = checked_source()
    if type(scan_chunk) is not int or scan_chunk < 1:
        raise ValueError('Declare positive exact-recurrence scan chunk size')
    if backend == 'torch':
        no_out, with_out, identity = backend_functions('torch', scan_chunk)
    elif backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested original fused MGGM-MIL CUDA backend unavailable; no fallback')
        expected = {'mamba-ssm': '1.1.2', 'causal-conv1d': '1.1.1'}
        if {k: importlib.metadata.version(k) for k in expected} != expected:
            raise ValueError('Original MGGM-MIL fused operators require pinned mamba1.1.2/causal-conv1d1.1.1')
        ops = load_single('mggmmil', 'models/mamba/mamba_ssm/ops/selective_scan_interface.py')
        no_out, with_out = ops.mamba_inner_fn_no_out_proj, ops.mamba_inner_fn
        identity = {'name': 'cuda', 'versions': expected, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'binaries': {k: digest_file(importlib.import_module(k).__file__)
                    for k in ['selective_scan_cuda', 'causal_conv1d_cuda']}}
    else:
        raise ValueError('Choose explicit torch recurrence or original fused cuda backend')

    def forbidden(*args, **kwargs):
        raise RuntimeError('Original MGGM-MIL fast graph required; no broken slow/plain-Mamba replacement')

    ns = {'math': math, 'torch': torch, 'nn': nn, 'F': F, 'Tensor': Tensor,
          'Optional': Optional, 'rearrange': rearrange, 'repeat': repeat,
          'mamba_inner_fn_no_out_proj': no_out, 'mamba_inner_fn': with_out,
          'selective_scan_fn': forbidden}
    srm = load_definitions('mggmmil', 'models/mamba/mamba_ssm/modules/srmamba.py',
                          ['TransposeTokenReEmbedding', 'SRMamba'], ns)
    # Import the actual GENConv; ordinary adjacency averaging does not implement
    # its learnable softmax aggregation, message ReLU and residual MLP.
    from torch_geometric.nn import GENConv
    classes = load_definitions('mggmmil', 'models/mggm_mil.py',
        ['GCNBlock', 'GABMIL', 'MGGMMIL'],
        {'torch': torch, 'nn': nn, 'F': F, 'SRMamba': srm.SRMamba, 'GENConv': GENConv})
    return classes, srm, {'source': sha, 'scan_backend': identity,
        'torch_geometric': importlib.metadata.version('torch-geometric'),
        'source_files': {f: digest_file(root/f) for f in [
            'models/mggm_mil.py', 'models/mamba/mamba_ssm/modules/srmamba.py', 'train_mil.py']}}


def original_model(dimension=512, backend='torch', scan_chunk=64):
    if dimension != 512:
        raise ValueError('Released ResNet18/PLIP MGGM-MIL condition requires actual512D features')
    classes, _, identity = author_classes(backend, scan_chunk)
    # Exact README condition, including expand1 and default rate10 inside each
    # complete SRMamba, not the distinct512D/expand2 GMMamba method.
    args = SimpleNamespace(KNN='4+8+12')
    return classes.MGGMMIL(args=args, I=512, L=384, D=256,
                          n_classes=2, layer=4, mil_head='GABMIL'), identity

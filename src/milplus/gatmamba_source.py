"""Pinned GAT-Mamba and the exact declared Mamba1.2.0.post1 dependency."""
import hashlib
import importlib
import importlib.metadata
import math
import types
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from .io import digest_file
from .upstream import load_definitions, pinned_text, verified_source

MAMBA_COMMIT = '34076d664838588a3c97727b263478ab9f621a07'


def mamba_class(backend):
    root, sha = verified_source('gatmamba_ssm')
    if sha != MAMBA_COMMIT:
        raise ValueError('GAT-Mamba requires its actual declared Mamba1.2.0.post1 source')
    scope = {'torch': torch, 'Tensor': Tensor, 'nn': nn, 'F': F, 'math': math, 'Optional': Optional,
             'rearrange': rearrange, 'repeat': repeat, 'selective_state_update': None,
             'causal_conv1d_update': None}
    identity = {'name': backend, 'dependency_version': '1.2.0.post1', 'source_commit': sha,
                'torch': torch.__version__}
    if backend == 'torch_reference':
        scan = load_definitions('gatmamba_ssm', 'mamba_ssm/ops/selective_scan_interface.py',
            ['selective_scan_ref'], scope).selective_scan_ref
        scope.update(selective_scan_fn=scan, causal_conv1d_fn=None)
        identity.update(scan='actual same-version author selective_scan_ref and original Mamba slow branch',
                        native_cuda=False)
    elif backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('GAT-Mamba native CUDA explicitly requires an accessible GPU; no fallback')
        try:
            versions = {name: importlib.metadata.version(name) for name in ['mamba-ssm', 'causal-conv1d']}
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Build mamba-ssm==1.2.0.post1 and causal-conv1d==1.0.2 in a compatible CUDA environment') from error
        if versions != {'mamba-ssm': '1.2.0.post1', 'causal-conv1d': '1.0.2'}:
            raise ValueError('GAT-Mamba CUDA dependencies differ from the author-declared pinned versions')
        ops = importlib.import_module('mamba_ssm.ops.selective_scan_interface')
        causal = importlib.import_module('causal_conv1d')
        actual = digest_file(ops.__file__)
        expected = hashlib.sha256(pinned_text(root, sha, 'mamba_ssm/ops/selective_scan_interface.py').encode()).hexdigest()
        if actual != expected:
            raise ValueError('Installed GAT-Mamba native scan wrapper differs from the verified author source')
        scope.update(selective_scan_fn=ops.selective_scan_fn, mamba_inner_fn=ops.mamba_inner_fn,
                     causal_conv1d_fn=causal.causal_conv1d_fn)
        identity.update(versions=versions, native_cuda=True, cuda=torch.version.cuda,
            wrapper_sha256=actual, binaries={n: digest_file(importlib.import_module(n).__file__)
                for n in ['selective_scan_cuda', 'causal_conv1d_cuda']})
    else:
        raise ValueError('GAT-Mamba backend must be explicitly torch_reference or cuda')
    cls = load_definitions('gatmamba_ssm', 'mamba_ssm/modules/mamba_simple.py', ['Mamba'], scope).Mamba
    return cls, identity


def network_class(dimension, edge_conditioned, backend):
    root, sha = verified_source('gatmamba')
    mamba, identity = mamba_class(backend)
    text = pinned_text(root, sha, 'modeling/model.py')
    replacements = [('from mamba_ssm import Mamba', ''),
                    ('self.num_uni_features = 1024', f'self.num_uni_features = {dimension}'),
                    ('feature_start_index+1024', f'feature_start_index+{dimension}')]
    if edge_conditioned:
        replacements.append(('heads = self.num_heads_gnn, dropout = gnn_dropout)',
                             'heads = self.num_heads_gnn, dropout = gnn_dropout, edge_dim=self.edge_embedding_size)'))
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('GAT-Mamba explicit import/input/edge repair no longer matches the source')
        text = text.replace(old, new)
    module = types.ModuleType('_milplus_gatmamba_'+sha+'_'+str(dimension)+'_'+str(edge_conditioned))
    module.__file__ = str(root/'modeling/model.py')
    module.Mamba = mamba
    exec(compile(text, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- complete verified Git blob, named dimension/edge wiring
    return module.GATMamba, identity


def source_trainer():
    # Only mathematical method definitions are loaded; no data paths, CUDA
    # setup, plotting or executable CLI side effects are imported.
    import numpy as np
    return load_definitions('gatmamba', 'modeling/main_gat_mamba.py', ['TrainModel'],
                            {'torch': torch, 'np': np}).TrainModel

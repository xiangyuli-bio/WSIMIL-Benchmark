"""Private, pinned C²MIL source imports; no alternate aggregator implementation."""
import ast
import sys
import types

import numpy as np
import torch
from torch.nn import functional as F

from .upstream import load_definitions, pinned_text, verified_source

COMMIT = 'fd4d54fc115466e43019c77015cdef01d90c85c3'


def modules():
    root, sha = verified_source('c2mil')
    if sha != COMMIT:
        raise ValueError('C²MIL source does not match the reviewed commit')
    prefix = '_milplus_c2mil_'+sha
    if prefix+'.network' in sys.modules:
        return sys.modules[prefix+'.network'], sys.modules[prefix+'.objective']
    skipped = {'softKmeans', 'graph_transformer_pytorch_v', 'Graphtransformer', 'VGAE',
               'bias_removal', 'rotary_embedding_torch', 'torch_scatter', 'timm.models.layers'}

    def original(key, filename, namespace=None):
        tree = ast.parse(pinned_text(root, sha, filename))
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and n.module in skipped) and not (
            isinstance(n, ast.Import) and any(a.name == 'matplotlib.pyplot' for a in n.names))]
        module = types.ModuleType(prefix+'.'+key)
        module.__file__ = str(root/filename)
        module.__dict__.update(namespace or {})
        exec(compile(tree, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- pinned original Git blobs, import-only changes
        sys.modules[module.__name__] = module
        return module

    soft = original('soft', 'softKmeans.py')
    bias = original('bias', 'bias_removal.py',
                    {'SoftCluster': soft.SoftCluster, 'SoftCluster_adaptiveK': soft.SoftCluster_adaptiveK})
    vgae = original('vgae', 'VGAE.py')
    transformer = original('transformer', 'graph_transformer_pytorch_v.py')
    encoder = original('encoder', 'Graphtransformer.py', {'GraphTransformer': transformer.GraphTransformer})
    network = original('network', 'model_causal_double.py', {
        'DeepVGAE': vgae.DeepVGAE, 'DeepVGAE_mse': vgae.DeepVGAE_mse,
        'DeepGraphTransformer_mse': encoder.DeepGraphTransformer_mse, 'BiasRemoval': bias.BiasRemoval})
    objective = load_definitions('c2mil', 'utils.py', ['CoxSurvLoss', 'causal_contrastive_loss',
        'adjust_learning_rate'], {'torch': torch, 'np': np, 'F': F}, replacements=(
        ('train_R = train_R.cuda()', 'train_R = train_R.to(hazards.device)'),
        ('dtype=torch.float).to(device)', 'dtype=torch.float).to(hazards.device)')))
    sys.modules[prefix+'.objective'] = objective
    return network, objective

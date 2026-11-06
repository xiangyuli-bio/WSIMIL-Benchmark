"""Private pinned MG-Trans definitions, with enumerated compatibility repairs."""
import math
import threading
import warnings
from types import SimpleNamespace
from typing import Optional

import nmslib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch_geometric.nn import DeepGCNLayer, GENConv

from .upstream import load_definitions, verified_source

COMMIT = 'a18d886b89ff0eddf5ae0b4ae16058623fe34dbc'


def thread_seeded_hnsw(original):
    """Keep actual NMSLIB HNSW while resetting its native thread-local RNG.

    A fresh native thread starts NMSLIB's documented default generator anew.
    Reusing the caller thread makes identical coordinates produce different
    graph realizations after a PyTorch-only RNG restore. No other user's or
    method's native RNG is modified, and the search algorithm/budget is intact.
    """
    class ThreadSeededHnsw(original):
        def fit(self, values):
            self._points = np.asarray(values)
            errors = []
            def build():
                try:
                    original.fit(self, values)
                except BaseException as error:
                    errors.append(error)
            worker = threading.Thread(target=build)
            worker.start(); worker.join()
            if errors:
                raise errors[0]
            return self

        def query(self, vector, topn, self_index=None):
            indices = np.asarray(original.query(self, vector, topn))
            if self_index is None:
                return indices
            if (self.space != 'l2' or self._points.ndim != 2 or self._points.shape[1] != 2
                    or not 0 <= self_index < len(self._points)
                    or not np.array_equal(self._points[self_index], vector)):
                raise ValueError('MG-Trans spatial query must identify its actual measured node')
            # NMSLIB leaves equal-distance membership/order unspecified. Source
            # padding repeats real coordinates and assumes query[0] is self.
            # Keep actual HNSW search/ranks, choose exact geometric ties by node
            # ID, and make that existing source self-first assumption explicit.
            distances = np.square(self._points.astype(np.float64)-np.asarray(vector, dtype=np.float64)).sum(1)
            indices = indices.copy()
            if self_index not in indices:
                indices[np.argmax(distances[indices])] = self_index
            selected = [self_index]
            for distance in np.unique(distances[indices]):
                count = int(np.count_nonzero(distances[indices] == distance)) - int(distance == 0)
                candidates = np.flatnonzero(distances == distance)
                candidates = candidates[candidates != self_index]
                selected.extend(candidates[:count].tolist())
            if len(selected) != topn or len(set(selected)) != topn:
                raise ValueError('MG-Trans HNSW geometric tie refinement changed the requested neighborhood size')
            return np.asarray(selected, dtype=indices.dtype)
    return ThreadSeededHnsw


def configuration(dim=1024):
    if type(dim) is not int or dim < 1:
        raise ValueError('MG-Trans input dimension must be a positive integer')
    return SimpleNamespace(input_size=dim, hidden_size=192,
        transformer={'num_heads': 3, 'mlp_dim': 768,
                     'attention_dropout_rate': 0., 'dropout_rate': .1})


def source_definitions():
    _, sha = verified_source('mgtrans')
    if sha != COMMIT:
        raise ValueError('MG-Trans source differs from the audited fixed commit')
    namespace = {name: value for name, value in vars(F).items() if not name.startswith('__')}
    namespace.update(torch=torch, nn=nn, F=F, np=np, math=math, warnings=warnings,
        Tensor=torch.Tensor, Optional=Optional, Module=nn.Module, Parameter=nn.Parameter,
        _LinearWithBias=NonDynamicallyQuantizableLinear, nmslib=nmslib,
        xavier_uniform_=nn.init.xavier_uniform_, constant_=nn.init.constant_,
        xavier_normal_=nn.init.xavier_normal_)
    # The released latent index is built in hidden space but queried with 2D
    # coordinates. Querying mismatched native dimensions is unsafe even though
    # the model currently consumes the spatial edges only.
    old = '''    model.fit(features)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)'''
    new = old.replace('model.query(coords[v_idx]', 'model.query(features[v_idx]')
    util = load_definitions('mgtrans', 'models/model_utils.py',
        ['multi_head_attention_forward', 'MultiheadAttention', 'Hnsw', 'pt2graph',
         'pairwise_distances', 'calculate_gram_mat', 'reyi_entropy', 'joint_entropy', 'calculate_MI'],
        namespace, replacements=[(old, new),
            ('model.query(coords[v_idx], topn=radius)', 'model.query(coords[v_idx], topn=radius, self_index=v_idx)'),
            ("index_params = {'M': 16, 'post': 0, 'efConstruction': 400}",
             "index_params = {'M': 16, 'post': 0, 'efConstruction': 400, 'indexThreadQty': 1}"),
            ('torch.symeig(k, eigenvectors=True)[0]', "torch.linalg.eigh(k, UPLO='U')[0]")])
    util.Hnsw = thread_seeded_hnsw(util.Hnsw)
    bindings = {name: getattr(nn, name) for name in
                ['CrossEntropyLoss', 'Dropout', 'Softmax', 'Linear', 'Conv2d', 'LayerNorm', 'ReLU']}
    bindings.update({name: value for name, value in vars(util).items() if not name.startswith('__')})
    bindings.update(GENConv=GENConv, DeepGCNLayer=DeepGCNLayer,
                    ACT2FN={'gelu': F.gelu, 'relu': F.relu, 'swish': lambda x: x*torch.sigmoid(x)})
    model = load_definitions('mgtrans', 'models/model_MG_Trans.py',
        ['np2th', 'swish', 'LabelSmoothing', 'Attention', 'Mlp', 'Embeddings', 'Block',
         'Part_GCN', 'Block_graph', 'ATTShort', 'Encoder', 'Transformer', 'MG_Trans_main'],
        bindings, replacements=[(".to('cuda:0')", '.to(x.device)')])
    return model, util


def build_model(dim, classes):
    if type(classes) is not int or classes < 2:
        raise ValueError('MG-Trans requires at least two explicit classes')
    model, _ = source_definitions()
    return model.MG_Trans_main(configuration(dim), img_size=2500, num_classes=classes,
                               smoothing_value=0., zero_head=True)

"""Pinned MicroMIL with complete BCE/Adam training on frozen FM bags.

This is an explicit WSI-feature transfer of a microscopy method. Its frozen
RGB encoder is replaced at the feature boundary; no claim of paper scores.
"""
import importlib.metadata
import math
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, RandomSampler

from .io import digest_object
from .loss_attention import check_update_finite
from .smmile_data import positive_int
from .stochastic import inference_rng
from .upstream import load_definitions


def author_network():
    try:
        import dgl
        from dgl.nn import GATConv
    except ImportError as error:
        raise ImportError('MicroMIL requires the declared dgl extra: pip install "mil-baseline-puls[dgl]"') from error
    if importlib.metadata.version('dgl') != '1.1.3':
        raise ValueError('MicroMIL currently validates DGL 1.1.3; other backends need separate verification')
    return load_definitions('micromil', 'model.py', ['DEC', 'ClusterAttentionModel'],
        {'torch': torch, 'nn': nn, 'F': F, 'Parameter': Parameter, 'Optional': Optional,
         'dgl': dgl, 'GATConv': GATConv}, [
            ('self.feature_extractor.fc = nn.Identity()', 'self.feature_extractor = nn.Identity()'),
            ('g = dgl.graph((x, nonzero_indices[2]))', 'g = dgl.graph((x, nonzero_indices[2]), num_nodes=grid)'),
            ('.to(self.args.device)', '.to(rep_features.device)')]).ClusterAttentionModel


def source_loader(groups):
    # Source DataLoader's iterator seed and RandomSampler are part of the RNG contract.
    return DataLoader(groups, batch_size=None, sampler=RandomSampler(groups), num_workers=0,
                      collate_fn=lambda rows: rows)


class MicroMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, hidden=128, clusters=36, layers=2, dropout=.5,
                 prediction_seed=2026, prediction_samples=1, max_graph_nodes=1024):
        super().__init__()
        if bundle.task_type != 'classification' or len(bundle.classes) != 2:
            raise ValueError('MicroMIL retains the binary classification/BCE contract')
        for name, value in [('hidden', hidden), ('clusters', clusters), ('layers', layers),
                            ('prediction_samples', prediction_samples), ('max_graph_nodes', max_graph_nodes)]:
            positive_int(value, name)
        if type(prediction_seed) is not int or prediction_seed < 0:
            raise ValueError('prediction_seed must be a nonnegative integer')
        if clusters > max_graph_nodes:
            raise ValueError('MicroMIL dense similarity graph exceeds max_graph_nodes')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite in [0,1)')
        self.dim, self.bundle = bundle.dim, bundle
        self.hidden, self.clusters, self.layers, self.dropout = hidden, clusters, layers, dropout
        self.prediction_seed, self.prediction_samples = prediction_seed, prediction_samples
        args = SimpleNamespace(model_name='frozen_fm', device='cpu', cluster_number=clusters,
                               layer=layers, shuffle=False)
        self.network = author_network()(args, in_dim=self.dim, hidden_dim=hidden, n_classes=2, drop_rate=dropout)
        self.optimizer_steps = 0

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('MicroMIL preserves full bag sampling and unclipped source updates')
        if cfg['selection_metric'] != 'micromil_f1':
            raise ValueError('MicroMIL uses validation F1, replacing original test-side selection')

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) == 0 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('MicroMIL needs finite, nonempty [N,D] frozen features')
        logits = self.network(x.unsqueeze(0)).reshape(1, 2)
        if not torch.isfinite(logits).all():
            raise ValueError('MicroMIL produced nonfinite logits')
        # No gradient is invented through the source's discrete .nonzero graph edges.
        return {'logits': logits, 'cluster_assignments': torch.stack(self.network.cluster_assignments, dim=1),
                'adjacency': self.network.attention_scores.squeeze(0), 'node_features': self.network.h.reshape(self.clusters, -1)}

    def forward_unit(self, bundle, rows, device, config, train=False):
        x = torch.cat([bundle.load(row)[0] for row in rows]).to(device)
        if train:
            return self(x)
        # The original evaluates stochastic hard-Gumbel graphs. Keep stochastic
        # inference, but fix a label-independent per-bag stream and restore TRAIN RNG.
        identity = digest_object({'seed': self.prediction_seed,
                                  'slides': sorted(row['slide_id'] for row in rows)})
        probabilities = []
        for sample in range(self.prediction_samples):
            with inference_rng(device, (int(identity[:15], 16)+sample) % (2**63-1)):
                result = self(x)
                probabilities.append(result['logits'].softmax(-1))
        result['logits'] = torch.stack(probabilities).mean(0).clamp_min(torch.finfo(x.dtype).tiny).log()
        return result

    def validation_metrics(self, frame):
        # Match the declared evaluation unit, including patient mean-probability aggregation.
        from sklearn.metrics import f1_score
        if self.bundle.config.get('evaluation_unit', 'slide') == 'patient':
            groups = frame.groupby('global_patient_id')
            if (groups.target.nunique() != 1).any():
                raise ValueError('MicroMIL patient F1 needs consistent labels')
            target = groups.target.first()
            predicted = groups[['p_0', 'p_1']].mean().to_numpy().argmax(1)
        else:
            target, predicted = frame.target, frame[['p_0', 'p_1']].to_numpy().argmax(1)
        return {'micromil_f1': float(f1_score(target, predicted, zero_division=0))}

    def get_extra_state(self):
        return {'optimizer_steps': self.optimizer_steps}

    def set_extra_state(self, state):
        if set(state) != {'optimizer_steps'} or type(state['optimizer_steps']) is not int or state['optimizer_steps'] < 0:
            raise ValueError('Invalid MicroMIL checkpoint state')
        self.optimizer_steps = state['optimizer_steps']

    def protocol(self):
        return {'method': 'micromil_frozen', 'family': 'micromil',
            'input': 'frozen FM features; explicit transfer from microscopy patient bags to WSI bags',
            'training_unit': self.bundle.config.get('training_unit', 'slide'),
            'dimension': self.dim, 'hidden': self.hidden, 'clusters': self.clusters,
            'layers': self.layers, 'dropout': self.dropout, 'optimizer_steps': self.optimizer_steps,
            'recipe': 'source BCE(softmax(logits)[1], bag label), Adam continuous state; no auxiliary KL invented',
            'graph': 'source full cosine matrix, stochastic hard Gumbel per row, directed edges and added self loops; duplicate loops retained',
            'gradient': 'straight-through representative selection; edge indices are discrete, as in author source',
            'prediction': {'samples': self.prediction_samples, 'seed': self.prediction_seed,
                           'rule': 'label-independent fixed per-bag Gumbel streams; mean probability; TRAIN RNG restored'},
            'selection': 'validation binary F1 at declared evaluation unit; first best; test locked',
            'repairs': ['explicit graph node count and feature device',
                        'restore training mode every epoch', 'fix source dataset/trainer tuple mismatch via registered data contract',
                        'validation replaces test selection; configurable hidden/layer/dropout honored'],
            'source_discrepancies': ['source constructs full similarity, not the upper-triangular prose in paper',
                                     'source lr=1e-4, paper lr=1e-3; default follows source',
                                     'DGL 1.1.3 validated against original 0.8.1 GAT equations; GCNConv import unused'],
            'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'gpu_validation': 'not_run'}


def micromil_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    for rows in source_loader(groups):
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_unit(bundle, rows, device, cfg, train=True)
        target = torch.tensor(float(rows[0]['target']), device=device)
        loss = F.binary_cross_entropy(result['logits'].softmax(-1)[0, 1], target)
        if not torch.isfinite(loss):
            raise ValueError('MicroMIL objective is nonfinite')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('MicroMIL gradient is nonfinite')
        check_update_finite(model, optimizer)
        optimizer.step()
        check_update_finite(model, optimizer)
        model.optimizer_steps += 1
        losses.append({'loss': float(loss.detach()), 'bce': float(loss.detach())})
    return losses

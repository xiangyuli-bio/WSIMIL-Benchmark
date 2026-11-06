"""Pinned ReCo-MIL networks and complete prototype/rarity/annealing training recipe."""
import math
from typing import Any, Literal, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.nn import functional as F

from .data import resolve
from .io import digest_file, digest_object
from .metrics import classification_metrics
from .upstream import load_definitions

PROFILES = {'recomil_single', 'recomil_graph', 'recomil_dual', 'recomil_dual_graph'}


def author_definitions(deferred=True):
    # The source's required KMeans asset loads without consuming Torch RNG. Defer only its
    # tensor values until prepare(), retaining exactly that initialization RNG path.
    patches = ()
    if deferred:
        from .upstream import pinned_text, verified_source
        root, sha = verified_source('recomil')
        code = pinned_text(root, sha, 'recomil_model.py')
        start, end = code.index('        if proto_init_path:'), code.index('        self.patch_proj')
        patches = ((code[start:end], '        self.prototypes = nn.Parameter(torch.zeros(num_protos, dim_in), requires_grad=True)\n\n'),)
    ns = {'torch': torch, 'nn': nn, 'F': F, 'Optional': Optional}
    names = ['GatedAttention', 'Mlp', 'ProtoReducer', 'GraphProtoReducer', 'ReCoMIL']
    model = load_definitions('recomil', 'recomil_model.py', names, ns, patches)
    dual = load_definitions('recomil', 'dual_recomil.py', ['DualReCoMIL'],
        {**ns, 'ReCoMIL': model.ReCoMIL, 'Literal': Literal, 'Dict': dict, 'Any': Any}).DualReCoMIL
    gate = load_definitions('recomil', 'rarity_gate.py', ['_sim', 'rarity_gate'], ns).rarity_gate
    orth = load_definitions('recomil', 'orth_regularizer.py', ['orth_loss'], ns).orth_loss
    return model.ReCoMIL, dual, gate, orth


def fit_prototypes(arrays, prototypes=128, seed=2025, sample_ratio=1., max_per_slide=0,
                   batch_size=4096, max_iter=200):
    """Author stacked MiniBatchKMeans algorithm; caller supplies only training slides."""
    rng, sampled = np.random.RandomState(seed), []
    for array in arrays:
        x = np.asarray(array, dtype=np.float32)
        if sample_ratio < 1.:
            x = x[rng.choice(len(x), max(1, int(len(x)*sample_ratio)), replace=False)]
        if max_per_slide and len(x) > max_per_slide:
            x = x[rng.choice(len(x), max_per_slide, replace=False)]
        sampled.append(x)
    if not sampled or sum(map(len, sampled)) < prototypes:
        raise ValueError('ReCo-MIL needs at least K sampled training instances; no random prototype fallback')
    x = np.concatenate(sampled)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('ReCo-MIL prototype inputs must be finite matrices')
    km = MiniBatchKMeans(n_clusters=prototypes, random_state=seed, batch_size=batch_size,
                        max_iter=max_iter, verbose=0).fit(x)
    centers = km.cluster_centers_.astype(np.float32)
    frequencies = np.bincount(km.predict(x), minlength=prototypes).astype(np.int64)
    if not np.isfinite(centers).all():
        raise ValueError('ReCo-MIL produced nonfinite prototypes')
    return centers, frequencies, list(map(len, sampled))


def branch_settings(values):
    integers = ['hidden', 'prototypes', 'enhancers', 'route_topk', 'freeze_epochs', 'tau_epochs',
                'clustering_seed', 'cluster_max_per_slide', 'cluster_batch_size', 'cluster_iterations']
    positives = {'hidden', 'prototypes', 'enhancers', 'cluster_batch_size', 'cluster_iterations'}
    for key in integers:
        if type(values[key]) is not int or values[key] < (1 if key in positives else 0):
            raise ValueError('Invalid ReCo-MIL integer setting '+key)
    if values['prototypes'] < 2**values['enhancers']:
        raise ValueError('ReCo-MIL prototype reduction would produce an empty prototype set')
    for key in ['tau_start', 'tau_end', 'graph_threshold', 'rare_beta', 'outlier_top_p', 'outlier_alpha',
                'lambda_ortho', 'sample_ratio']:
        if isinstance(values[key], bool) or not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
            raise ValueError('Invalid ReCo-MIL finite setting '+key)
    if not 0 < values['sample_ratio'] <= 1 or values['outlier_top_p'] > 1 or min(values['tau_start'], values['tau_end']) <= 0:
        raise ValueError('ReCo-MIL sampling/outlier/tau settings are out of range')
    if values['similarity'] not in {'l2', 'cosine', 'dot'}:
        raise ValueError('Invalid ReCo-MIL similarity')
    for key in ['gate_in_embed', 'use_frequencies', 'graph_reducer']:
        if type(values[key]) is not bool:
            raise ValueError('ReCo-MIL '+key+' must be boolean')
    return values


class ReCoMIL(nn.Module):
    def __init__(self, bundle, profile='recomil_single', hidden=512, prototypes=128, enhancers=3,
                 dropout=.25, hard=False, similarity='l2', route_topk=3, graph_threshold=.15,
                 tau_start=1.2, tau_end=.7, tau_epochs=35, freeze_epochs=12,
                 outlier_top_p=.015, outlier_alpha=.2, gate_in_embed=False, use_frequencies=False,
                 rare_beta=.3, lambda_ortho=.02, ortho_projected=False, class_weight=False,
                 label_smoothing=.03, warmup_epochs=12, clustering_seed=2025, sample_ratio=1.,
                 cluster_max_per_slide=0, cluster_batch_size=4096, cluster_iterations=200, low=None):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification' or len(bundle.classes) != 2:
            raise ValueError('ReCo-MIL source recipe currently requires binary classification')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('ReCo-MIL requires individual slide bags')
        self.profile, self.dual = profile, profile.startswith('recomil_dual')
        if self.dual and bundle.dual_scale is None:
            raise ValueError('ReCo-MIL dual profiles require explicit dual_scale features')
        if low is not None and (not self.dual or not isinstance(low, dict)):
            raise ValueError('ReCo-MIL low overrides require a dual profile and mapping')
        if not math.isfinite(dropout) or not 0 <= dropout < 1 or not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
            raise ValueError('ReCo-MIL dropout/label_smoothing must be in[0,1)')
        if type(warmup_epochs) is not int or warmup_epochs < 0 or any(type(v) is not bool for v in [hard, ortho_projected, class_weight]):
            raise ValueError('Invalid ReCo-MIL warmup or boolean option')
        settings = {key: value for key, value in locals().copy().items() if key in {
            'hidden', 'prototypes', 'enhancers', 'similarity', 'route_topk', 'graph_threshold',
            'tau_start', 'tau_end', 'tau_epochs', 'freeze_epochs', 'outlier_top_p', 'outlier_alpha',
            'gate_in_embed', 'use_frequencies', 'rare_beta', 'lambda_ortho', 'clustering_seed',
            'sample_ratio', 'cluster_max_per_slide', 'cluster_batch_size', 'cluster_iterations'}}
        settings['graph_reducer'] = profile.endswith('graph')
        self.settings = {'high': branch_settings(settings)}
        if self.dual:
            if set(low or {})-set(settings) or 'enhancers' in (low or {}) and low['enhancers'] != enhancers:
                raise ValueError('Unknown ReCo-MIL low option or unequal shared enhancer count')
            self.settings['low'] = branch_settings({**settings, **(low or {})})
        self.dims = {'high': bundle.dual_scale.dim if self.dual else bundle.dim}
        if self.dual:
            self.dims['low'] = bundle.dim
        single, dual, self.gate, self.orth = author_definitions()
        def arguments(branch):
            s = self.settings[branch]
            return {'dim_in': self.dims[branch], 'embed_dim': s['hidden'], 'num_protos': s['prototypes'],
                    'similarity_method': s['similarity'], 'tau': s['tau_start'], 'route_topk': s['route_topk'],
                    'use_graph_reducer': s['graph_reducer'], 'graph_thresh': s['graph_threshold']}
        common = {'num_classes': 2, 'num_enhancers': enhancers, 'drop': dropout, 'hard': hard}
        if self.dual:
            self.network = dual(**{key+'_'+branch: value for branch in self.settings for key, value in arguments(branch).items()}, **common)
        else:
            self.network = single(**arguments('high'), **common)
        for branch, s in self.settings.items():
            self.register_buffer('initial_'+branch, torch.zeros(s['prototypes'], self.dims[branch]))
            self.register_buffer('frequency_'+branch, torch.zeros(s['prototypes'], dtype=torch.long))
            self.register_buffer('rare_weight_'+branch, torch.zeros(s['prototypes']))
        self.register_buffer('class_weights', torch.ones(2))
        self.ortho_projected, self.class_weight = ortho_projected, class_weight
        self.label_smoothing, self.warmup_epochs = label_smoothing, warmup_epochs
        self.prepared, self.ledger, self.optimizer_steps, self.epoch = False, {}, 0, 0
        self.epochs, self.base_lr = 50, 1e-4
        self.evaluation_unit = bundle.config.get('evaluation_unit', 'slide')

    def branches(self):
        return {'high': self.network.high, 'low': self.network.low} if self.dual else {'high': self.network}

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('ReCo-MIL rejects global max_instances; configure prototype sampling explicitly')
        self.epochs, self.base_lr = config['epochs'], config['lr']

    def bags(self, bundle, rows, device):
        if len(rows) != 1:
            raise ValueError('ReCo-MIL requires one slide per bag')
        values = {'low' if self.dual else 'high': bundle.load(rows[0])[0].to(device)}
        if self.dual:
            values['high'] = bundle.dual_scale.load(rows[0]['slide_id']).to(device)
        return values

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        rows = sorted(bundle.records('train'), key=lambda row: row['slide_id'])
        for branch, network in self.branches().items():
            s = self.settings[branch]; arrays, ledger = [], []
            for row in rows:
                arrays.append(self.bags(bundle, [row], 'cpu')[branch].numpy())
                path = bundle.dual_scale.rows[row['slide_id']]['path'] if self.dual and branch == 'high' else row['path']
                ledger.append({'slide_id': row['slide_id'], 'global_patient_id': row['global_patient_id'],
                               'feature_sha256': digest_file(resolve(bundle.root, path))})
            centers, frequency, counts = fit_prototypes(arrays, s['prototypes'], s['clustering_seed'], s['sample_ratio'],
                s['cluster_max_per_slide'], s['cluster_batch_size'], s['cluster_iterations'])
            network.prototypes.copy_(torch.from_numpy(centers).to(device))
            getattr(self, 'initial_'+branch).copy_(network.prototypes)
            getattr(self, 'frequency_'+branch).copy_(torch.from_numpy(frequency).to(device))
            weights = (frequency.astype(np.float32)+1.)**(-s['rare_beta'])
            weights = weights/(weights.mean()+1e-6)
            getattr(self, 'rare_weight_'+branch).copy_(torch.from_numpy(weights).to(device))
            self.ledger[branch] = {'fit_scope': 'training_patients', 'slides': ledger, 'sample_counts': counts,
                                  'centers_sha256': digest_object(centers.tolist()), 'frequencies': frequency.tolist()}
        counts = np.bincount([row['target'] for row in rows], minlength=2)
        if not counts.all():
            raise ValueError('ReCo-MIL training split is missing a class')
        pos = float(counts[1]/counts.sum())
        self.class_weights.copy_(torch.tensor([.5/max(1e-6, 1-pos), .5/max(1e-6, pos)], dtype=torch.float32, device=device))
        self.prepared = True

    def make_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def start_epoch(self, epoch):
        self.epoch = epoch+1
        for branch, network in self.branches().items():
            s = self.settings[branch]
            fraction = min(1., self.epoch/max(1, s['tau_epochs']))
            network.set_tau(s['tau_start'] if s['tau_epochs'] <= 0 else s['tau_start']+(s['tau_end']-s['tau_start'])*fraction)
            network.prototypes.requires_grad_(self.epoch > s['freeze_epochs'])

    def lr_factor(self):
        if self.epoch <= self.warmup_epochs:
            return self.epoch/max(1, self.warmup_epochs)
        t = (self.epoch-self.warmup_epochs)/max(1, self.epochs-self.warmup_epochs)
        return .5*(1+math.cos(math.pi*t))

    def get_extra_state(self):
        return {'prepared': self.prepared, 'ledger': self.ledger, 'optimizer_steps': self.optimizer_steps,
                'epoch': self.epoch, 'epochs': self.epochs, 'base_lr': self.base_lr}

    def set_extra_state(self, state):
        self.prepared, self.ledger, self.optimizer_steps = state['prepared'], state['ledger'], state['optimizer_steps']
        self.epochs, self.base_lr = state['epochs'], state['base_lr']
        self.start_epoch(state['epoch']-1)

    def forward(self, high, low=None, train=False):
        if not self.prepared:
            raise RuntimeError('ReCo-MIL requires training-only prototype preparation before use')
        values = {'high': high, **({'low': low} if self.dual else {})}
        kwargs = {}
        for branch, network in self.branches().items():
            x, s = values[branch], self.settings[branch]
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or not len(x) or x.shape[1] != self.dims[branch] or not torch.isfinite(x).all():
                raise ValueError('ReCo-MIL requires finite nonempty features at each registered scale dimension')
            gate, weights = None, None
            if train:
                if s['outlier_top_p'] > 0 and s['outlier_alpha'] > 0:
                    with torch.no_grad():
                        p, c = (network.patch_proj(x), network.proto_proj(network.prototypes)) if s['gate_in_embed'] else (x, network.prototypes.detach())
                        gate = self.gate(p, c, s['outlier_top_p'], s['outlier_alpha'], method=s['similarity'], in_embed=s['gate_in_embed'])
                if s['use_frequencies'] and s['rare_beta'] > 0:
                    weights = getattr(self, 'rare_weight_'+branch)
            suffix = '_'+branch if self.dual else ''
            kwargs.update({'data'+suffix: x, 'instance_gate'+suffix: gate, 'proto_weights'+suffix: weights})
        result = self.network(**kwargs)
        if not torch.isfinite(result['logits']).all():
            raise ValueError('Nonfinite ReCo-MIL logits')
        return result

    def forward_unit(self, bundle, rows, device, config, train=False):
        return self(**self.bags(bundle, rows, device), train=train)

    def objective(self, result, label):
        ce = F.cross_entropy(result['logits'], label, label_smoothing=self.label_smoothing,
                             weight=self.class_weights if self.class_weight else None)
        orth = ce.new_zeros(())
        for branch, network in self.branches().items():
            weight = self.settings[branch]['lambda_ortho']
            if weight > 0:
                protos = network.get_projected_protos() if self.ortho_projected else network.prototypes
                orth = orth+weight*self.orth(protos)
        return {'loss': ce+orth, 'ce': ce, 'orth': orth}

    def validation_metrics(self, predictions):
        metrics = classification_metrics(predictions, 2, self.evaluation_unit)
        return {'reco_joint': (metrics['macro_auc']+metrics['accuracy'])/2 if metrics['macro_auc'] is not None else None}

    def protocol(self):
        return {'profile': self.profile, 'branches': self.settings, 'dimensions': self.dims,
                'prototype_preparation': self.ledger, 'prepared': self.prepared,
                'optimizer': 'Adam', 'warmup_epochs': self.warmup_epochs, 'total_epochs': self.epochs,
                'epoch_1based': self.epoch, 'lr_factor': self.lr_factor(), 'optimizer_steps': self.optimizer_steps,
                'label_smoothing': self.label_smoothing, 'class_weight': self.class_weight,
                'ortho_projected': self.ortho_projected, 'source_inactive_similarity_scale': True,
                'source_projected_orth_is_detached': True, 'fusion': 'sigmoid(alpha), alpha_init=.5' if self.dual else None,
                'inference': 'author evaluate: no rarity gate or frequency weights',
                'repairs': 'explicit slide loaders; strict train-only prototype fit, no random asset fallback',
                'selection': 'source (AUC+accuracy)/2 using the task evaluation unit; validation only',
                'gpu_amp_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}


def recomil_epoch(model, bundle, groups, optimizer, device, config):
    model.train(); losses = []
    for group in optimizer.param_groups:
        group['lr'] = model.base_lr*model.lr_factor()
    for rows in groups:
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_unit(bundle, rows, device, config, train=True)
        objective = model.objective(result, torch.tensor([rows[0]['target']], device=device))
        if not torch.isfinite(objective['loss']):
            raise ValueError('Nonfinite ReCo-MIL loss')
        objective['loss'].backward()
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite ReCo-MIL gradients')
        optimizer.step(); model.optimizer_steps += 1
        losses.append({key: float(value.detach()) for key, value in objective.items()})
    return losses

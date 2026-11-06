"""Public cDP-MIL training path, including per-slide Bayesian DP preprocessing.

The released main.py trains DP_Classifier.infer with CE; its variational forward
and gamma updates are not called. This source-path profile records that limit.
"""
import hashlib
import math
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats import beta
from sklearn.mixture import BayesianGaussianMixture
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def author_classifier():
    return load_definitions('cdpmil', 'model/dpmil.py',
        ['Distribution', 'ReparametrizedGaussian_VI', 'DirichletProcess_VI', 'DP_Classifier'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math,
         'ABCMeta': ABCMeta, 'abstractmethod': abstractmethod, 'beta': beta},
        replacements=[('torch.from_numpy(samples).cuda()', 'torch.from_numpy(samples).to(self.mu[0].device)')])


class CDPMILSource(nn.Module):
    def __init__(self, dim, classes, clusters=10, concentration=.1, classifier_concentration=10.,
                 cluster_iterations=30, cache_bags=128, inference_seed=2026,
                 aggregation_dtype='float32'):
        super().__init__()
        if any(not isinstance(v, int) for v in (dim, classes, clusters, cluster_iterations, cache_bags)):
            raise ValueError('cDP-MIL dimensions, iterations, clusters and cache_bags must be integers')
        if dim < 1 or classes < 2 or clusters < 1 or cluster_iterations < 1 or cache_bags < 0:
            raise ValueError('cDP-MIL requires positive dimensions/iterations/clusters, >=2 classes and cache_bags>=0')
        if any(not math.isfinite(v) or v <= 0 for v in (concentration, classifier_concentration)):
            raise ValueError('cDP-MIL concentration must be finite positive')
        if not isinstance(inference_seed, int) or not 0 <= inference_seed < 2**32:
            raise ValueError('inference_seed must be an integer in [0, 2**32)')
        if aggregation_dtype not in {'float32', 'float64'}:
            raise ValueError('cDP-MIL aggregation_dtype must explicitly be float32 or float64')
        self.dim, self.classes = dim, classes
        self.clusters, self.concentration = clusters, concentration
        self.cluster_iterations, self.cache_bags = cluster_iterations, cache_bags
        self.inference_seed = inference_seed
        self.aggregation_dtype = aggregation_dtype
        self.network = author_classifier().DP_Classifier(
            trunc=classes, eta=classifier_concentration, batch_size=1, dim=dim, n_sample=1)
        self.cache = OrderedDict()
        self.scheduler = None
        self.loaded_scheduler = None

    def configure_training(self, cfg):
        clip = cfg['grad_clip']
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError('grad_clip must be finite positive or null')
        self.epochs = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(.5, .9), weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=.000005)
        return optimizer

    def finish_epoch(self, bundle, device, cfg):
        self.scheduler.step()

    def get_extra_state(self):
        return {'scheduler': self.scheduler.state_dict() if self.scheduler else self.loaded_scheduler}

    def set_extra_state(self, state):
        self.loaded_scheduler = state['scheduler']
        if self.scheduler is not None and self.loaded_scheduler is not None:
            self.scheduler.load_state_dict(self.loaded_scheduler)

    def aggregate(self, x):
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) < max(2, self.clusters):
            raise ValueError('cDP-MIL requires [N,D] with N >= max(2, clusters); no silent component reduction')
        if not torch.isfinite(x).all():
            raise ValueError('cDP-MIL features must be finite')
        array = x.detach().cpu().float().contiguous().numpy()
        key = (self.aggregation_dtype, array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            # Full covariance and hard-assignment means exactly follow DP_feats_aggr.py.
            # Fit independently per slide, with no labels or other slides involved.
            mixture = BayesianGaussianMixture(n_components=self.clusters, random_state=0,
                max_iter=self.cluster_iterations, weight_concentration_prior=self.concentration)
            # Optional, explicit numeric repair for full-covariance failures on
            # high-dimensional float32 bags. Do not change the default source
            # path, regularization, component budget, iterations or features.
            clustering = array if self.aggregation_dtype == 'float32' else array.astype(np.float64)
            assignments = mixture.fit(clustering).predict(clustering)
            # Preserve source float32 hard-assignment feature means and the
            # classifier's dtype. Only covariance/assignment arithmetic opts in.
            centroids = np.array([np.mean(array[assignments == i], axis=0) for i in np.unique(assignments)])
            if self.cache_bags:
                self.cache[key] = centroids
                while len(self.cache) > self.cache_bags:
                    self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
            centroids = self.cache[key]
        return torch.from_numpy(centroids).to(x.device)

    def forward(self, x, coords=None, label=None):
        centroids = self.aggregate(x)
        if self.training:
            logits = self.network(centroids).mean(0, keepdim=True)
        else:
            # One author posterior draw, but isolated/fixed for repeatable checkpoint
            # selection and evaluation; validation must not consume training RNG.
            state = np.random.get_state()
            try:
                np.random.seed(self.inference_seed)
                logits = self.network(centroids).mean(0, keepdim=True)
            finally:
                np.random.set_state(state)
        return {'logits': logits}

    def protocol(self):
        return {'method': 'cDP-MIL public source path', 'profile': 'cdpmil_source',
                'feature_aggregation': 'per-slide BayesianGaussianMixture, full covariance, hard-assignment means',
                'clusters': self.clusters, 'concentration': self.concentration,
                'cluster_iterations': self.cluster_iterations, 'cluster_seed': 0,
                'aggregation_dtype': self.aggregation_dtype,
                'aggregation_numeric_condition': 'literal source float32' if self.aggregation_dtype == 'float32' else
                    'explicit float64 mixture/covariance/assignment arithmetic; unchanged full covariance, priors, components and iterations; original float32 hard means/classifier; no seedwise float32-equivalence claim',
                'classifier_concentration': self.network.dp_process.eta,
                'objective': 'CE(mean instance log densities + sampled stick weights); public main.py',
                'gamma_updates': 'not called by public main.py; fixed gamma1=1, gamma2=eta',
                'variational_objective': 'not called by public main.py; not claimed implemented here',
                'inference': 'one posterior draw with isolated fixed NumPy seed', 'inference_seed': self.inference_seed,
                'cache': 'bounded disposable feature-content cache; recomputed identically after reload',
                'selection': 'held-out validation, replacing source test evaluation every epoch; public main.py saves no selected checkpoint',
                'scope': 'faithful released training path with explicit device/evaluation fixes; not full paper reproduction'}


def cdpmil_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    results = []
    for rows in groups:
        x, coords = load_unit(bundle, rows, device, cfg['max_instances'], train=True)
        target = torch.tensor([rows[0]['target']], device=device)
        loss = F.cross_entropy(model(x, coords=coords)['logits'], target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite cDP-MIL training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'] or float('inf'), error_if_nonfinite=True)
        optimizer.step()
        results.append({'loss': float(loss.detach())})
    return results

"""MiCo's author cluster routes/reducers, with train-only FAISS initialization."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .discrete_survival import DiscreteSurvival
from .upstream import load_definitions


def load_mico():
    return load_definitions('mico', 'models/MiCo.py',
        ['GatedAttention', 'Mlp', 'ClusterReducer', 'MiCo'],
        {'torch': torch, 'nn': nn, 'F': F}, replacements=[
            ("{'hazards': hazards, 'S': S, 'Y_hat': Y_hat}",
             "{'logits': logits, 'hazards': hazards, 'S': S, 'Y_hat': Y_hat}")]).MiCo


def cluster_centers(features, clusters, iterations=20, seed=1234):
    import faiss
    values = np.ascontiguousarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) < clusters or not np.isfinite(values).all():
        raise ValueError('MiCo clustering requires finite [N,D] input with N >= clusters')
    if len(np.unique(values, axis=0)) < clusters:
        raise ValueError('MiCo clustering requires at least clusters distinct feature vectors')
    author = load_definitions('mico', 'run_clustering.py', ['FaissKmeans'],
        {'faiss': faiss, 'np': np}, replacements=[
            ('kmeans.niter = self.niter', 'kmeans.niter = self.niter\n        kmeans.seed = self.seed'),
            ('kmeans.verbose = True', 'kmeans.verbose = False')])
    backend = author.FaissKmeans(clusters, niter=iterations, gpu=False)
    backend.seed = seed
    threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        centers, _ = backend.train(values)
    finally:
        faiss.omp_set_num_threads(threads)
    if not np.isfinite(centers).all():
        raise FloatingPointError('MiCo clustering returned nonfinite centers')
    return centers


class ContextClusters(nn.Module):
    def __init__(self, dim, outputs, survival=False, hidden=512, clusters=64, enhancers=3,
                 dropout=.25, hard=False, similarity='l2', cluster_slides=20,
                 cluster_iterations=20, cluster_seed=1234, max_clustering_bytes=1073741824):
        super().__init__()
        if min(dim, outputs, hidden, clusters, cluster_slides, cluster_iterations) < 1 or hidden < 2:
            raise ValueError('MiCo dimensions/budgets must be positive and hidden >= 2')
        if enhancers < 1 or clusters < 2**enhancers or not 0 <= dropout < 1:
            raise ValueError('MiCo needs clusters >= 2**enhancers, enhancers >= 1 and valid dropout')
        if similarity not in {'l2', 'dot'} or max_clustering_bytes < 1:
            raise ValueError('Invalid MiCo similarity or clustering memory budget')
        self.network = load_mico()(dim_in=dim, embedding_dim=hidden, num_clusters=clusters,
            num_classes=outputs, survival=survival, num_enhancers=enhancers, drop=dropout,
            hard=hard, similarity_method=similarity)
        self.clusters, self.cluster_slides = clusters, cluster_slides
        self.cluster_iterations, self.cluster_seed = cluster_iterations, cluster_seed
        self.max_clustering_bytes = max_clustering_bytes
        self.initialization_slides = []
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        rows = sorted(bundle.records('train'), key=lambda row: row['slide_id'])[:self.cluster_slides]
        samples, used = [], 0
        for row in rows:
            x, _ = bundle.load(row)
            used += x.numel()*x.element_size()
            if used > self.max_clustering_bytes:
                raise ValueError('MiCo clustering input exceeds max_clustering_bytes; choose an explicit smaller initialization set')
            samples.append(x.numpy())
        centers = cluster_centers(np.concatenate(samples), self.clusters,
                                  self.cluster_iterations, self.cluster_seed)
        self.network.cluster_centers.copy_(torch.from_numpy(centers).to(device))
        self.initialization_slides = [row['slide_id'] for row in rows]
        self.initialized.fill_(True)

    def get_extra_state(self):
        return {'initialization_slides': self.initialization_slides}

    def set_extra_state(self, state):
        self.initialization_slides = state['initialization_slides']

    def forward(self, bags):
        if not self.initialized:
            raise RuntimeError('MiCo requires training-only cluster initialization')
        if not torch.isfinite(self.network.similarity_scale).all() or (self.network.similarity_scale == 0).any():
            raise FloatingPointError('MiCo author similarity scale is zero or nonfinite')
        if any(x.ndim != 2 or len(x) == 0 for x in bags):
            raise ValueError('MiCo expects nonempty per-slide bags [N,D]')
        # Keep the batch dimension: author squeeze(0) then also supports N=1.
        data = [x[None] for x in bags] if self.network.survival else bags[0][None]
        if not self.network.survival and len(bags) != 1:
            raise ValueError('MiCo classification uses a single slide bag')
        return self.network(data=data)

    def protocol(self):
        return {'initialization': 'author FAISS L2 KMeans, CPU backend; first sorted training slides only',
                'initialization_slides': self.initialization_slides, 'clusters': self.clusters,
                'cluster_seed': self.cluster_seed, 'cluster_iterations': self.cluster_iterations,
                'max_clustering_bytes': self.max_clustering_bytes,
                'clustering_memory_scope': 'input feature bytes only, not total peak RAM',
                'similarity': self.network.similarity_method, 'hard_assignment': self.network.hard_assignment,
                'architecture': 'author per-slide cluster routes and reducers before shared attention pooling',
                'unused_author_parameter': 'similarity_bias is defined but not used by author forward',
                'optimizer': 'Adam with author no-decay grouping for biases and 1D parameters'}


def mico_optimizer(model, lr, weight_decay):
    group = load_definitions('mico', 'MyOptimizer/optim_factory.py', ['add_weight_decay'], {}).add_weight_decay
    return torch.optim.Adam(group(model, weight_decay), lr=lr)


class MiCoClassification(nn.Module):
    def __init__(self, dim, classes, **options):
        super().__init__()
        self.core = ContextClusters(dim, classes, **options)

    def prepare(self, bundle, seed, device):
        self.core.prepare(bundle, seed, device)

    def make_optimizer(self, lr, weight_decay):
        return mico_optimizer(self, lr, weight_decay)

    def forward(self, x, coords=None, label=None):
        return {'logits': self.core([x])['logits']}

    def protocol(self):
        return {**self.core.protocol(), 'objective': 'cross entropy'}


def patient_bags(bundle, rows, device, max_instances=None, train=False):
    bags = [bundle.load(row)[0] for row in rows]
    total = sum(map(len, bags))
    if max_instances and total > max_instances:
        indices = (torch.randperm(total)[:max_instances].sort().values if train else
                   torch.linspace(0, total-1, max_instances).long())
        sampled, offset = [], 0
        for bag in bags:
            keep = indices[(indices >= offset) & (indices < offset+len(bag))]-offset
            offset += len(bag)
            if len(keep):
                sampled.append(bag[keep])
        bags = sampled
    return [bag.to(device) for bag in bags]


class MiCoSurvival(DiscreteSurvival):
    def __init__(self, dim, bins=4, alpha=.15, accumulation_steps=1, **options):
        super().__init__(bins, alpha, accumulation_steps)
        self.core = ContextClusters(dim, bins, survival=True, **options)

    def prepare(self, bundle, seed, device):
        super().prepare(bundle, seed, device)
        self.core.prepare(bundle, seed, device)

    def make_optimizer(self, lr, weight_decay):
        return mico_optimizer(self, lr, weight_decay)

    def forward(self, bags):
        return self.result(self.core(bags)['logits'])

    def forward_unit(self, bundle, rows, device, config, train=False):
        return self(patient_bags(bundle, rows, device, config['max_instances'], train))

    def protocol(self):
        return {**super().protocol(), **self.core.protocol(),
                'patient_input': 'separate slide feature lists; context enhancement per slide; joint attention pool'}

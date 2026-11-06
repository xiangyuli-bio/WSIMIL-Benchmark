"""SlideGraph+ hierarchical clustering, author EdgeConv and pair-ranking recipe.

Author code is loaded from the pinned, separately obtained GPL source. Numerical
preprocessing and split corrections are enumerated in docs/SPATIAL.md.
"""
from collections import OrderedDict
from itertools import combinations, pairwise

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Delaunay, cKDTree
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import EdgeConv, GINConv, global_add_pool, global_max_pool, global_mean_pool

from .spatial_attention import checked_points
from .upstream import load_definitions


def author_graph_class():
    return load_definitions('slidegraph', 'GNN_pr.py', ['GNN'], {
        'torch': torch, 'Sequential': nn.Sequential, 'Linear': nn.Linear,
        'BatchNorm1d': nn.BatchNorm1d, 'ReLU': nn.ReLU, 'F': F,
        'GINConv': GINConv, 'EdgeConv': EdgeConv, 'global_add_pool': global_add_pool,
        'global_max_pool': global_max_pool, 'global_mean_pool': global_mean_pool}).GNN


def author_platt_class():
    return load_definitions('slidegraph', 'platt.py', ['PlattScaling'], {'np': np},
                            replacements=(('np.float(', 'float('),)).PlattScaling


def connect_centroids(points, threshold):
    """Delaunay adjacency; explicitly extended to one-dimensional/short bags."""
    n = len(points)
    if len(np.unique(points, axis=0)) != n:
        raise ValueError('Rounded cluster centers collide; review coordinate scale/cluster parameters')
    if n <= 1:
        return np.empty((2, 0), dtype=np.int64)
    centered = points-points.mean(0)
    if n == 2 or np.linalg.matrix_rank(centered) < 2:
        # Delaunay in the affine span: adjacent vertices along the occupied line.
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        order = np.argsort(points@direction)
        pairs = set(pairwise(order))
    else:
        pairs = {tuple(sorted(p)) for simplex in Delaunay(points).simplices for p in combinations(simplex, 2)}
    pairs = sorted((int(a), int(b)) for a, b in pairs if np.linalg.norm(points[a]-points[b]) <= threshold)
    directed = sorted(pairs+[(b, a) for a, b in pairs])
    return np.asarray(directed, dtype=np.int64).reshape(-1, 2).T


def cluster_graph(features, coords, lambda_d=.003, lambda_f=.001,
                  cluster_threshold=.8, edge_distance=4000., max_pair_bytes=1073741824):
    points = checked_points(coords)
    if len(features) != len(points) or not torch.isfinite(features).all():
        raise ValueError('SlideGraph requires finite aligned feature/coordinate arrays')
    if lambda_d <= 0 or lambda_f < 0 or not 0 < cluster_threshold < 1 or edge_distance <= 0:
        raise ValueError('Invalid SlideGraph similarity or graph-distance parameters')
    # Preserve source integer pixel coordinates and rounded cluster centers.
    if not np.equal(points, np.round(points)).all():
        raise ValueError('SlideGraph source protocol requires integer coordinate units')
    raw = features.detach().cpu().numpy()
    active = np.ptp(raw, axis=0) > 1e-4
    distances_features = raw[:, active]
    n = len(points)
    pair_count = n*(n-1)//2
    if pair_count*8 > max_pair_bytes:
        raise ValueError('SlideGraph condensed distances exceed max_pair_bytes; hierarchical clustering is quadratic')
    if n == 1:
        clusters = np.ones(1, dtype=int)
    else:
        tree = cKDTree(points)
        condensed = np.ones(pair_count, dtype=np.float64)
        offset = 0
        for i in range(n-1):
            near = np.asarray(tree.query_ball_point(points[i], 6/lambda_d), dtype=int)
            near = near[near > i]
            d = np.linalg.norm(points[near]-points[i], axis=1)
            f = np.linalg.norm(distances_features[near]-distances_features[i], axis=1)
            condensed[offset+near-i-1] = 1-np.exp(-lambda_f*f)*np.exp(-lambda_d*d)
            offset += n-i-1
        clusters = fcluster(linkage(condensed, method='average'), cluster_threshold, criterion='distance')
    indices = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    centers = np.asarray([np.round(points[idx].mean(0)) for idx in indices])
    # Keep a common input width across slides; distance computation above still
    # uses exactly the author's per-slide nonconstant feature subset.
    pooled = np.asarray([raw[idx].mean(0) for idx in indices], dtype=np.float32)
    edges = connect_centroids(centers, edge_distance)
    return Data(x=torch.from_numpy(pooled), edge_index=torch.from_numpy(edges),
                coords=torch.from_numpy(centers), num_nodes=len(pooled))


def ranking_loss(scores, labels):
    scores, labels = scores.flatten(), labels.flatten()
    a, b = torch.triu_indices(len(labels), len(labels), offset=1, device=scores.device)
    keep = labels[a] != labels[b]
    if not keep.any():
        raise ValueError('SlideGraph ranking batches require both binary classes')
    a, b = a[keep], b[keep]
    return F.relu(1-(labels[a]-labels[b])*(scores[a]-scores[b])).mean()


def ranking_batches(groups, batch_size, seed):
    labels = np.asarray([g[0]['target'] for g in groups])
    counts = np.bincount(labels, minlength=2)
    if len(counts) != 2 or counts.min() == 0:
        raise ValueError('SlideGraph requires two training classes')
    folds = min(max(1, len(groups)//batch_size), int(counts.min()))
    if folds == 1:
        return [groups]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [[groups[i] for i in idx] for _, idx in splitter.split(labels, labels)]


class SlideGraphMIL(nn.Module):
    def __init__(self, dim, classes, layers=(16, 16, 8), dropout=.5, batch_size=8,
                 lambda_d=.003, lambda_f=.001, cluster_threshold=.8, edge_distance=4000.,
                 max_pair_bytes=1073741824, cache_graphs=32):
        super().__init__()
        if classes != 2 or batch_size < 2 or cache_graphs < 0 or not layers or min(layers) < 1:
            raise ValueError('SlideGraph+ requires binary classes and a ranking batch size >= 2')
        self.network = author_graph_class()(dim_features=dim, dim_target=1, layers=list(layers),
            dropout=dropout, pooling='mean', conv='EdgeConv', gembed=False, aggr='max')
        self.batch_size, self.cache_limit = batch_size, cache_graphs
        self.graph_args = {'lambda_d': lambda_d, 'lambda_f': lambda_f, 'cluster_threshold': cluster_threshold,
                           'edge_distance': edge_distance, 'max_pair_bytes': max_pair_bytes}
        self.graph_cache = OrderedDict()
        self.register_buffer('feature_mean', torch.zeros(dim))
        self.register_buffer('feature_scale', torch.ones(dim))
        self.register_buffer('calibration', torch.tensor([-1., 0.], dtype=torch.float64))

    def graph(self, bundle, rows):
        if len(rows) != 1:
            raise ValueError('SlideGraph uses single-slide coordinate frames')
        row = rows[0]
        key = row['slide_id']
        if key in self.graph_cache:
            self.graph_cache.move_to_end(key)
            return self.graph_cache[key]
        x, coords = bundle.load(row)
        graph = cluster_graph(torch.as_tensor(x), torch.as_tensor(coords) if coords is not None else None,
                              **self.graph_args)
        if self.cache_limit:
            self.graph_cache[key] = graph
            if len(self.graph_cache) > self.cache_limit:
                self.graph_cache.popitem(last=False)
        return graph

    def prepare(self, bundle, seed, device):
        from .engine import units
        # Merge per-graph statistics in float64; no validation/test access.
        count, mean, m2 = 0, torch.zeros_like(self.feature_mean, dtype=torch.float64, device='cpu'), 0.
        for rows in units(bundle, 'train'):
            x = self.graph(bundle, rows).x.double()
            n, current = len(x), x.mean(0)
            delta = current-mean
            m2 = m2+(x-current).square().sum(0)+delta.square()*count*n/(count+n)
            mean = mean+delta*n/(count+n)
            count += n
        if count < 2:
            raise ValueError('SlideGraph standardization needs at least two training nodes')
        scale = (m2/(count-1)).clamp_min(0).sqrt()
        scale[scale <= 1e-8] = 1.
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def batch(self, bundle, groups, device):
        graphs = [self.graph(bundle, rows).clone() for rows in groups]
        graph = Batch.from_data_list(graphs).to(device)
        graph.x = (graph.x-self.feature_mean)/self.feature_scale
        score, node_scores, _ = self.network(graph)
        score = score.flatten()
        p = torch.sigmoid(-(score.double()*self.calibration[0]+self.calibration[1])).to(score.dtype)
        return {'logits': torch.stack([torch.zeros_like(score), score], 1),
                'probabilities': torch.stack([1-p, p], 1), 'score': score, 'node_scores': node_scores}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if config.get('max_instances') is not None:
            raise ValueError('SlideGraph clusters the full bag; max_instances would change graph topology')
        return self.batch(bundle, [rows], device)

    @torch.no_grad()
    def finish_epoch(self, bundle, device, config):
        from .engine import units
        was_training = self.training
        self.eval()
        groups = units(bundle, 'train')
        scores = [float(self.batch(bundle, [rows], device)['score'].item()) for rows in groups]
        labels = [rows[0]['target'] for rows in groups]
        fitted = author_platt_class()().fit(labels, scores)
        values = torch.tensor([fitted.A, fitted.B], dtype=self.calibration.dtype, device=device)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite training-only Platt coefficients')
        self.calibration.copy_(values)
        self.train(was_training)

    def protocol(self):
        return {'graph': self.graph_args, 'normalization': 'train nodes only, sample standard deviation',
                'calibration': 'author Platt fit on training scores only; coefficients checkpointed each epoch',
                'objective': 'mean pairwise hinge margin 1 over opposite-class graph pairs',
                'batch_size': self.batch_size, 'selection': 'one validation-selected checkpoint, no snapshot ensemble',
                'degenerate_graphs': '1D affine-span adjacency extension; one node has no edges',
                'feature_width': 'retain input dimensions; per-slide constant columns excluded from distance only'}


def slidegraph_epoch(model, bundle, groups, optimizer, device, cfg):
    if cfg.get('max_instances') is not None:
        raise ValueError('SlideGraph full-bag topology is incompatible with max_instances')
    # Seed is drawn from the checkpointed NumPy state, not wall clock/global entropy.
    batches = ranking_batches(groups, model.batch_size, int(np.random.randint(2**31-1)))
    losses = []
    for batch in batches:
        result = model.batch(bundle, batch, device)
        labels = torch.tensor([rows[0]['target'] for rows in batch], device=device)
        loss = ranking_loss(result['score'], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite graph ranking loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append({'loss': float(loss.detach())})
    return losses

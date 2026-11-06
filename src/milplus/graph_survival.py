"""Patch-GCN: author residual GENConv network on per-slide physical kNN graphs."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from .discrete_survival import DiscreteSurvival
from .upstream import load_survival


def spatial_edges(coords, neighbors=8):
    if neighbors < 1 or not isinstance(neighbors, int):
        raise ValueError('neighbors must be a positive integer')
    points = coords.detach().cpu().numpy()
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite physical coordinates [N,2]')
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Duplicate patch coordinates within a slide')
    if len(points) <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    k = min(neighbors, len(points)-1)
    _, indices = cKDTree(points).query(points, k=k+1, workers=1)
    # Directed kNN: neighboring node -> center node. Never connect different slide frames.
    source = np.concatenate([row[row != i][:k] for i, row in enumerate(indices)])
    return torch.tensor(np.stack([source, np.repeat(np.arange(len(points)), k)]), dtype=torch.long)


def patient_graph(bundle, rows, device, neighbors=8, max_instances=None, train=False):
    from torch_geometric.data import Data
    bags = [bundle.load(row, require_coords=True) for row in rows]
    counts = [len(x) for x, _ in bags]
    total = sum(counts)
    selected = None
    if max_instances and total > max_instances:
        selected = torch.randperm(total)[:max_instances].sort().values if train else torch.linspace(0, total-1, max_instances).long()
    xs, edges, slide_ids = [], [], []
    source_offset = output_offset = 0
    for slide_index, (x, coords) in enumerate(bags):
        if selected is not None:
            local = selected[(selected >= source_offset) & (selected < source_offset+len(x))] - source_offset
            source_offset += len(x)
            x, coords = x[local], coords[local]
        if len(x) == 0:
            continue
        edges.append(spatial_edges(coords, neighbors)+output_offset)
        xs.append(x)
        slide_ids.append(torch.full((len(x),), slide_index, dtype=torch.long))
        output_offset += len(x)
    return Data(x=torch.cat(xs).to(device), edge_index=torch.cat(edges, dim=1).to(device),
                batch=torch.zeros(output_offset, dtype=torch.long, device=device),
                slide_index=torch.cat(slide_ids).to(device))


class PatchGCNSurvival(DiscreteSurvival):
    def __init__(self, dim, bins=4, alpha=0., accumulation_steps=1, l1_weight=0.,
                 hidden=128, layers=4, neighbors=8, dropout=.25):
        super().__init__(bins, alpha, accumulation_steps, l1_weight)
        if layers < 2 or hidden < 1 or neighbors < 1:
            raise ValueError('Patch-GCN needs layers >= 2, hidden >= 1 and neighbors >= 1')
        self.neighbors = neighbors
        self.network = load_survival('patchgcn').PatchGCN_Surv(
            num_features=dim, hidden_dim=hidden, num_layers=layers, dropout=dropout, n_classes=bins)

    def forward(self, graph):
        return self.result(self.network(x_path=graph)[0])

    def forward_unit(self, bundle, rows, device, config, train=False):
        graph = patient_graph(bundle, rows, device, self.neighbors, config['max_instances'], train)
        return self(graph)

    def protocol(self):
        return {**super().protocol(), 'graph': 'exact directed physical kNN per slide; patient disjoint union',
                'neighbors': self.neighbors, 'cross_slide_edges': False,
                'recipe_scope': 'author GENConv architecture; PULS exact kNN, patient splits, optimizer and budget'}

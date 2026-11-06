"""CBMIL: pinned selection, phenotype/patch pooling and two-view contrastive training."""
import math
from typing import Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from .upstream import load_definitions, load_single


def load_cbmil():
    # The pinned author model imports nonexistent models.mil; the actual class is in models.clam.
    clam = load_single('cbmil', 'models/clam.py')
    scope = {'torch': torch, 'np': np, 'math': math, 'nn': nn, 'F': torch.nn.functional,
             'CLAM_SB': clam.CLAM_SB, 'Tuple': tuple, 'Union': Union}
    return load_definitions('cbmil', 'models/cbmil.py',
        ['NonLocalRanking', 'AdaptiveSelection', 'PPMIL', 'NT_Xent', 'CBMIL', 'mixup'], scope,
        replacements=(('sim[self.mask_correlated_samples(batch_size)]',
                       'sim[self.mask_correlated_samples(batch_size).to(sim.device)]'),
                      ('sim[self.mask]', 'sim[self.mask.to(sim.device)]')))


def contrastive_batches(groups, batch_size):
    if batch_size < 2 or len(groups) < 2:
        raise ValueError('CBMIL contrastive training requires at least two bags and batch_size >= 2')
    batches = [groups[start:start+batch_size] for start in range(0, len(groups), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


class ClusterContrastiveMIL(nn.Module):
    def __init__(self, dim, classes, clusters=10, selected_instances=1024, batch_size=16,
                 contrastive_dim=128, temperature=1., cl_weight=.1, mixup_alpha=.8,
                 clustering='kmeans', clustering_seed=2026, clustering_iterations=100,
                 view_instances=None):
        super().__init__()
        if min(clusters, contrastive_dim, clustering_iterations) < 1 or selected_instances < 8:
            raise ValueError('CBMIL needs positive cluster/dimension settings and selected_instances >= 8')
        if batch_size < 2 or temperature <= 0 or not 0 < cl_weight < 1:
            raise ValueError('CBMIL requires batch_size >= 2, positive temperature and 0 < cl_weight < 1')
        if not 0 < mixup_alpha < 1 or clustering not in {'kmeans', 'provided'}:
            raise ValueError('CBMIL requires 0 < mixup_alpha < 1 and clustering kmeans/provided')
        if view_instances is not None and view_instances < clusters:
            raise ValueError('CBMIL view_instances must cover at least one instance per cluster')
        # Author ranking keys and CLAM embeddings have width 512; no silent dimension mismatch for other FMs.
        self.input_projection = nn.Identity() if dim == 512 else nn.Linear(dim, 512)
        self.network = load_cbmil().CBMIL(dim_feature=512, num_sample_feature=selected_instances,
            dim_output=classes, dim_cl=contrastive_dim, mixup_alpha=mixup_alpha,
            bach_size=batch_size, temperature=temperature, cl_weight=cl_weight,
            selecting='adaptive', top_fuse=True)
        self.clusters, self.batch_size, self.clustering = clusters, batch_size, clustering
        self.clustering_seed, self.clustering_iterations = clustering_seed, clustering_iterations
        self.view_instances = view_instances
        self.assignments = {}

    def get_extra_state(self):
        return {'assignments': {k: v.cpu() for k, v in self.assignments.items()}}

    def set_extra_state(self, state):
        self.assignments = {k: v.cpu() for k, v in state['assignments'].items()}

    def assign(self, x, labels=None):
        if labels is None:
            if self.clustering == 'provided':
                raise ValueError('CBMIL provided clustering requires a bundle cluster_labels_path')
            if len(x) < self.clusters:
                raise ValueError('CBMIL cluster count exceeds the number of instances')
            labels = torch.from_numpy(KMeans(n_clusters=self.clusters, n_init=10,
                random_state=self.clustering_seed, max_iter=self.clustering_iterations)
                .fit_predict(x.detach().cpu().numpy()).astype(np.int64))
        labels = labels.detach().cpu()
        if labels.ndim != 1 or len(labels) != len(x) or labels.unique().numel() != self.clusters:
            raise ValueError('CBMIL requires exactly the configured number of nonempty aligned clusters')
        return labels

    @staticmethod
    def check_unit(rows):
        if len(rows) != 1:
            raise ValueError('CBMIL uses individual slide bags; patient evaluation can aggregate slide probabilities')

    def raw_unit(self, bundle, rows):
        self.check_unit(rows)
        x, _ = bundle.load(rows[0])
        labels = bundle.cluster_labels(rows[0], len(x)) if self.clustering == 'provided' else None
        return x, labels

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        from .engine import units
        for rows in units(bundle, 'train'):
            x, labels = self.raw_unit(bundle, rows)
            self.assignments[rows[0]['slide_id']] = self.assign(x, labels)

    def grouped(self, x, assignment):
        return [x[(assignment == label).to(x.device)] for label in assignment.unique(sorted=True)]

    def views(self, x, assignment):
        projected = self.input_projection(x)
        full = self.grouped(projected, assignment)
        ratio = 1. if self.view_instances is None else self.view_instances / len(x)
        if ratio >= 1:
            return full, full
        subset = [group[torch.from_numpy(np.random.choice(len(group),
            size=max(1, int(np.rint(len(group)*ratio))), replace=False)).to(group.device)] for group in full]
        return full, subset

    def batch_loss(self, views, labels):
        if len(labels) < 2:
            raise ValueError('CBMIL NT-Xent needs negative pairs; singleton batches are not admitted')
        outputs, z_full, instance = self.network.batch_forward(views[0], labels=labels,
            selecting=self.network.select_method, top_fuse=self.network.top_fuse)
        z_subset = self.network.batch_forward(views[1], selecting='all', top_fuse=False)
        bag = self.network.ce_loss(outputs, labels)
        contrastive = self.network.cl_loss(z_full, z_subset)
        objective = (1-self.network.cl_weight) * (.7*bag + .3*instance) + self.network.cl_weight*contrastive
        return {'loss': objective, 'bag_loss': bag, 'instance_loss': instance,
                'contrastive_loss': contrastive, 'logits': outputs}

    def forward(self, x, coords=None, label=None):
        if label is not None:
            raise ValueError('CBMIL training uses paired views and batches; use the dedicated epoch recipe')
        assignment = self.assign(x)
        groups = self.grouped(self.input_projection(x), assignment)
        return {'logits': self.network.eval_forward(groups)}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if config['max_instances'] is not None:
            raise ValueError('CBMIL uses selected_instances and view_instances; max_instances would break cluster alignment')
        x, labels = self.raw_unit(bundle, rows)
        assignment = self.assign(x, labels)
        groups = self.grouped(self.input_projection(x.to(device)), assignment)
        return {'logits': self.network.eval_forward(groups)}


def cbmil_epoch(model, bundle, groups, optimizer, device, cfg):
    if cfg['max_instances'] is not None:
        raise ValueError('CBMIL uses selected_instances and view_instances; max_instances would break cluster alignment')
    records = []
    for batch in contrastive_batches(groups, model.batch_size):
        optimizer.zero_grad(set_to_none=True)
        first, second = [], []
        for rows in batch:
            x, _ = model.raw_unit(bundle, rows)
            a, b = model.views(x.to(device), model.assignments[rows[0]['slide_id']])
            first.append(a)
            second.append(b)
        labels = torch.tensor([rows[0]['target'] for rows in batch], device=device)
        output = model.batch_loss([first, second], labels)
        if not torch.isfinite(output['loss']):
            raise FloatingPointError('Nonfinite CBMIL objective')
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        records.append({key: float(output[key].detach()) for key in
                        ('loss', 'bag_loss', 'instance_loss', 'contrastive_loss')})
    return records

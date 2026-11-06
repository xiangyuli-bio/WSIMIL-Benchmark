"""Pinned cross-scale and hierarchical attention with their distinct training recipes."""
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .stochastic import inference_rng
from .upstream import load_definitions


def load_csmil():
    return load_definitions('csmil', 'Train_Test_Code/DeepAttnMISL_model_no21.py',
        ['DeepAttnMIL_Surv'], {'torch': torch, 'nn': nn}, replacements=(
            ('def __init__(self, cluster_num):', 'def __init__(self, cluster_num, input_dim=2048):'),
            ('nn.Conv2d(2048, 64, 1)', 'nn.Conv2d(input_dim, 64, 1)'),
            ('hh = x[i].type(torch.FloatTensor).to("cuda")',
             'hh = x[i].to(device=self.embedding_net[0].weight.device, dtype=self.embedding_net[0].weight.dtype)'),
            ('Y_pred = self.fc6(M)', 'Y_pred = self.fc6[:-1](M)'),
        ))


def stable_masked_softmax(x, mask=None):
    if mask is not None:
        mask = mask.to(device=x.device, dtype=torch.bool).reshape_as(x)
        if not mask.any(1).all():
            raise ValueError('CS-MIL requires at least one unmasked instance')
        x = x.masked_fill(~mask, -torch.inf)
    return x.softmax(1)


def multiscale_unit(bundle, rows, device, cfg, kind):
    if len(rows) != 1:
        raise ValueError('Multiscale methods require individual slide bags')
    if cfg.get('max_instances') is not None:
        raise ValueError('Use method-specific multiscale sampling; max_instances would break scale alignment')
    if bundle.multiscale is None or bundle.multiscale.kind != kind:
        raise ValueError(f'Method requires multiscale kind {kind}')
    features, _, parents = bundle.multiscale.load(rows[0])
    return [x.to(device) for x in features], [None if p is None else p.to(device) for p in parents]


class CrossScaleMIL(nn.Module):
    def __init__(self, dim, classes, clusters=8, bag_mean=3., bag_std=1., accumulation_steps=4,
                 clustering='kmeans', clustering_seed=0, inference_seed=2026):
        super().__init__()
        if classes != 2:
            raise ValueError('CS-MIL pinned sigmoid head requires two classes')
        if clusters < 1 or accumulation_steps < 1 or bag_mean < 1 or bag_std < 0:
            raise ValueError('Invalid CS-MIL clustering/sampling/accumulation settings')
        if clustering not in {'kmeans', 'provided'}:
            raise ValueError('CS-MIL clustering must be kmeans or provided')
        self.network = load_csmil().DeepAttnMIL_Surv(clusters, input_dim=dim)
        self.network.masked_softmax = stable_masked_softmax
        self.clusters, self.bag_mean, self.bag_std = clusters, bag_mean, bag_std
        self.accumulation_steps, self.clustering = accumulation_steps, clustering
        self.clustering_seed, self.inference_seed = clustering_seed, inference_seed
        self.assignments = {}

    def get_extra_state(self):
        return {'assignments': {k: v.cpu() for k, v in self.assignments.items()}}

    def set_extra_state(self, state):
        self.assignments = {k: v.cpu() for k, v in state['assignments'].items()}

    def assign(self, features, labels=None):
        if labels is None:
            if self.clustering == 'provided':
                raise ValueError('CS-MIL provided clustering needs cluster_labels_path')
            if len(features) < self.clusters:
                raise ValueError('CS-MIL cluster count exceeds available aligned centers')
            labels = torch.from_numpy(KMeans(n_clusters=self.clusters, n_init=10,
                random_state=self.clustering_seed).fit_predict(features.detach().cpu().numpy()).astype(np.int64))
        labels = labels.detach().cpu()
        if (labels.shape != (len(features),) or labels.dtype != torch.int64
                or (labels < 0).any() or (labels >= self.clusters).any()):
            raise ValueError('CS-MIL needs aligned integer cluster labels in [0, clusters)')
        return labels

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        for row in bundle.records('train'):
            features, _ = multiscale_unit(bundle, [row], 'cpu', {}, 'aligned_centers')
            labels = bundle.cluster_labels(row, len(features[-1])) if self.clustering == 'provided' else None
            self.assignments[row['slide_id']] = self.assign(features[-1], labels)

    def sample(self, features, labels):
        if len(features) != 3 or any(x.shape != features[0].shape for x in features):
            raise ValueError('CS-MIL needs three aligned feature matrices')
        n = max(1, int(np.random.normal(self.bag_mean, self.bag_std)))
        stacked = torch.stack(list(reversed(features)), dim=2).unsqueeze(-1)  # fine, middle, coarse
        bags, masks, selected = [], [], []
        for cluster in range(self.clusters):
            candidates = torch.where(labels.cpu() == cluster)[0].numpy()
            if len(candidates):
                indices = candidates[np.random.randint(0, len(candidates), min(n, len(candidates)))]
                bags.append(stacked[torch.as_tensor(indices, device=stacked.device)])
                masks.extend([True]*len(indices))
                selected.append(indices.tolist())
            else:
                bags.append(stacked.new_zeros(1, stacked.shape[1], 3, 1))
                masks.append(False)
                selected.append([])
        return bags, torch.tensor([masks], device=stacked.device), selected

    def forward(self, features, labels):
        labels = self.assign(features[-1], labels)
        bags, mask, selected = self.sample(features, labels)
        logit = self.network(bags, mask)
        return {'logits': torch.cat([torch.zeros_like(logit), logit], dim=1),
                'binary_logit': logit, 'selected_indices': selected}

    def forward_unit(self, bundle, rows, device, config, train=False):
        context = nullcontext() if train else inference_rng(device, self.inference_seed)
        with context:
            features, _ = multiscale_unit(bundle, rows, device, config, 'aligned_centers')
            row = rows[0]
            labels = self.assignments.get(row['slide_id'])
            if labels is None:
                supplied = bundle.cluster_labels(row, len(features[-1])) if self.clustering == 'provided' else None
                labels = self.assign(features[-1], supplied)
            return self(features, labels)

    def protocol(self):
        return {'kind': 'aligned_centers', 'scale_order': 'fine_middle_coarse',
                'scale_attention': 'raw scores, not softmax; preserved from pinned author code',
                'clustering': self.clustering, 'clusters': self.clusters, 'kmeans_n_init': 10,
                'clustering_seed': self.clustering_seed, 'bag_mean': self.bag_mean, 'bag_std': self.bag_std,
                'sampling': 'with replacement per cluster, empty clusters masked',
                'accumulation_steps': self.accumulation_steps, 'accumulation_reduction': 'sum',
                'inference_seed': self.inference_seed, 'feature_training': 'frozen; SimSiam not reproduced'}


def csmil_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step, rows in enumerate(groups):
        result = model.forward_unit(bundle, rows, device, cfg, train=True)
        target = result['binary_logit'].new_tensor([[rows[0]['target']]])
        loss = F.binary_cross_entropy_with_logits(result['binary_logit'], target)
        loss.backward()  # author sums per-bag gradients, including the partial tail
        if (step+1) % model.accumulation_steps == 0 or step+1 == len(groups):
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append({'loss': float(loss.detach())})
    return losses


def load_hagmil():
    import admin_torch
    from nystrom_attention import NystromAttention
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np,
                 'admin_torch': admin_torch, 'NystromAttention': NystromAttention}
    helper = load_definitions('hagmil', 'models/model_utils.py',
                             ['Attn_Net', 'Attn_Net_Gated', 'TransLayer'], namespace)
    namespace.update({name: getattr(helper, name) for name in ('Attn_Net', 'Attn_Net_Gated', 'TransLayer')})
    return load_definitions('hagmil', 'models/model.py', ['create_attention_net', 'IAMBlock', 'IATModel'],
        namespace, replacements=(('norm_func(h).squeeze()', 'norm_func(h).squeeze(0)'),
                                 ('h.squeeze().T, A.squeeze()', 'h.squeeze(0).T, A.squeeze(0)'),
                                 ('logits.cpu(), all_targets.cpu()', 'logits, all_targets'),
                                 ('logits.cpu(), p_targets.cpu()', 'logits, p_targets')))


def child_indices(parents, selected):
    """Preserve ranked parent order; use explicit parents instead of guessing row*4."""
    order = torch.argsort(parents, stable=True).reshape(-1, 4)
    return order[selected].flatten()


class HierarchicalAttentionMIL(nn.Module):
    def __init__(self, dim, classes, hidden_dims=(1024, 1536, 512, 1024), dropout=.25,
                 k_sample=8, gate=True, subtyping=False, train_parents=7000, eval_parents=10000,
                 bag_weight=.5, instance_weight=.5):
        super().__init__()
        if len(hidden_dims) != 4 or any(d < 8 or d % 8 for d in hidden_dims):
            raise ValueError('HAG-MIL uses four IAM layers with widths divisible by eight')
        if min(k_sample, train_parents, eval_parents) < 1 or min(bag_weight, instance_weight) <= 0:
            raise ValueError('HAG-MIL requires active instance/bag losses and positive sampling settings')
        self.k_sample, self.train_parents, self.eval_parents = k_sample, train_parents, eval_parents
        self.bag_weight, self.instance_weight = bag_weight, instance_weight
        author = load_hagmil()
        self.branches = nn.ModuleList([author.IATModel(feat_in=dim, n_classes=classes,
            hidden_dims=list(hidden_dims), dropout=dropout, k_sample=k_sample, gate=gate,
            subtyping=subtyping, instance_loss_fn=nn.CrossEntropyLoss()) for _ in range(3)])

    def branch(self, level, x, label=None):
        if label is not None and len(x) < 2*self.k_sample:
            raise ValueError('HAG-MIL needs at least 2*k_sample instances at each selected level')
        logits, _, _, attention, extra = self.branches[level](x, label=label, instance_eval=label is not None)
        return {'logits': logits, 'attention': attention, **extra}

    def select(self, attention, current, parents, train):
        budget = self.train_parents if train else self.eval_parents
        ranked = attention.detach().flatten().argsort(descending=True)[:budget]
        return child_indices(parents, current[ranked])

    def forward(self, features, parents):
        current = torch.arange(len(features[0]), device=features[0].device)
        trace = []
        for level in range(3):
            trace.append(current)
            result = self.branch(level, features[level][current])
            if level < 2:
                current = self.select(result['attention'], current, parents[level+1], self.training)
        return {**result, 'selected_indices': trace}

    def forward_unit(self, bundle, rows, device, config, train=False):
        features, parents = multiscale_unit(bundle, rows, device, config, 'quadtree')
        return self(features, parents)

    def objective(self, result, label, level):
        bag = F.cross_entropy(result['logits'], label, label_smoothing=.1 if level < 2 else 0.)
        return self.bag_weight*bag + self.instance_weight*result['instance_loss']

    def protocol(self):
        return {'kind': 'quadtree', 'levels': 'coarse_middle_fine', 'instance_loss': 'cross_entropy',
                'optimizer_steps_per_slide': 3, 'bag_weight': self.bag_weight,
                'instance_weight': self.instance_weight, 'k_sample': self.k_sample,
                'train_parents': self.train_parents, 'eval_parents': self.eval_parents,
                'selection': 'detached pre-update attention; explicit four-child parent mapping',
                'coarse_middle_label_smoothing': .1, 'fine_label_smoothing': 0.}


def hagmil_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    for rows in groups:
        features, parents = multiscale_unit(bundle, rows, device, cfg, 'quadtree')
        current = torch.arange(len(features[0]), device=device)
        label = torch.tensor([rows[0]['target']], device=device)
        record = {}
        for level in range(3):
            optimizer.zero_grad(set_to_none=True)
            result = model.branch(level, features[level][current], label)
            loss = model.objective(result, label, level)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            record[f'level_{level}_loss'] = float(loss.detach())
            record[f'level_{level}_instance_loss'] = float(result['instance_loss'].detach())
            if level < 2:
                current = model.select(result['attention'], current, parents[level+1], True)
        record['loss'] = record['level_2_loss']
        losses.append(record)
    return losses

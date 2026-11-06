"""DeepAttnMISL phenotype pooling and the author's batch Cox/L1 recipe."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from .upstream import load_definitions


def patient_cox_mean(risks, times, events):
    """Author mean-over-patients partial likelihood, with stable risk-set sums."""
    risks, times, events = risks.flatten(), times.flatten(), events.flatten()
    if not (len(risks) == len(times) == len(events)) or len(risks) == 0:
        raise ValueError('Cox risk, time and event arrays must be aligned and nonempty')
    if not torch.isfinite(risks).all() or not torch.isfinite(times).all() or (times <= 0).any():
        raise ValueError('Cox risks/times must be finite and times positive')
    if not torch.isin(events, events.new_tensor([0, 1])).all():
        raise ValueError('event must be 0=censored or 1=observed')
    mask = times[None, :] >= times[:, None]
    denominator = risks[None].expand(len(risks), -1).masked_fill(~mask, -torch.inf).logsumexp(1)
    return (events*(denominator-risks)).mean()


def load_deepattn(dim):
    return load_definitions('deepattnmisl', 'DeepAttnMISL_model.py', ['DeepAttnMIL_Surv'],
        {'torch': torch, 'nn': nn}, replacements=[
            ('nn.Conv2d(4096, 64, 1)', f'nn.Conv2d({dim}, 64, 1)'),
            ('x_exp = (x - x_max.unsqueeze(-1)).exp()',
             'x_exp = (x - x_max.unsqueeze(-1)).masked_fill(mask == 0, -torch.inf).exp() if mask is not None else (x - x_max.unsqueeze(-1)).exp()')]).DeepAttnMIL_Surv


class DeepAttentionSurvival(nn.Module):
    survival_objective = 'batch_cox_patient_mean'

    def __init__(self, dim, clusters=10, clustering='kmeans', batch_size=16, l1_weight=1e-5,
                 clustering_seed=2026, clustering_iterations=100, plateau_patience=20):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (dim, clusters, batch_size, clustering_iterations)):
            raise ValueError('DeepAttnMISL dimensions and budgets must be positive integers')
        if batch_size < 2 or l1_weight < 0 or clustering not in {'provided', 'kmeans'} or plateau_patience < 0:
            raise ValueError('Invalid DeepAttnMISL training or clustering settings')
        self.network = load_deepattn(dim)(clusters)
        self.clusters, self.clustering, self.batch_size = clusters, clustering, batch_size
        self.clustering_seed, self.clustering_iterations = clustering_seed, clustering_iterations
        self.l1_weight, self.plateau_patience = l1_weight, plateau_patience
        self.assignments, self.scheduler, self.scheduler_state = {}, None, None

    def raw_unit(self, bundle, rows):
        bags = [bundle.load(row)[0] for row in rows]
        x = torch.cat(bags)
        if self.clustering == 'provided':
            meta = bundle.encoder.get('phenotype_clusters', {})
            if meta.get('scope') != 'patient' or meta.get('count') != self.clusters or not meta.get('provenance'):
                raise ValueError('Provided DeepAttnMISL clusters require patient scope, count and provenance in encoder.phenotype_clusters')
            labels = torch.cat([bundle.cluster_labels(row, len(bag)) for row, bag in zip(rows, bags)])
        else:
            labels = self.assignments.get(rows[0]['global_patient_id'])
            if labels is None:
                if len(x) < self.clusters or len(torch.unique(x, dim=0)) < self.clusters:
                    raise ValueError('DeepAttnMISL KMeans requires at least clusters distinct patient patches')
                labels = torch.from_numpy(KMeans(n_clusters=self.clusters, n_init=10,
                    random_state=self.clustering_seed, max_iter=self.clustering_iterations)
                    .fit_predict(x.numpy()).astype(np.int64))
        if len(labels) != len(x) or (labels < 0).any() or (labels >= self.clusters).any():
            raise ValueError('DeepAttnMISL cluster labels must align and lie in [0, clusters)')
        return x, labels

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        from .engine import units
        for rows in units(bundle, 'train'):
            _, labels = self.raw_unit(bundle, rows)
            if self.clustering == 'kmeans':
                self.assignments[rows[0]['global_patient_id']] = labels

    def forward(self, x, cluster_labels):
        if x.ndim != 2 or not len(x) or cluster_labels.shape != (len(x),):
            raise ValueError('DeepAttnMISL requires features [N,D] and aligned phenotype IDs [N]')
        labels = cluster_labels.to(x.device)
        if labels.dtype not in {torch.int32, torch.int64} or (labels < 0).any() or (labels >= self.clusters).any():
            raise ValueError('Invalid DeepAttnMISL phenotype IDs')
        groups, mask = [], x.new_zeros(1, self.clusters)
        for i in range(self.clusters):
            bag = x[labels == i]
            mask[0, i] = bool(len(bag))
            if not len(bag):
                bag = x.new_zeros(1, x.shape[1])
            groups.append(bag.T[None, :, None, :])
        return {'logits': self.network(groups, mask)}

    def forward_unit(self, bundle, rows, device, config, train=False):
        x, labels = self.raw_unit(bundle, rows)
        limit = config['max_instances']
        if limit and len(x) > limit:
            indices = torch.randperm(len(x))[:limit] if train else torch.linspace(0, len(x)-1, limit).long()
            x, labels = x[indices], labels[indices]
        return self(x.to(device), labels.to(device))

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=.5, patience=self.plateau_patience)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def validation_metrics(self, predictions):
        device = next(self.parameters()).device
        risk, time, event = [torch.tensor(predictions[k].to_numpy(), device=device, dtype=torch.float32)
                             for k in ('risk', 'time', 'event')]
        loss = patient_cox_mean(risk, time, event)+self.l1_weight*sum(p.abs().sum() for p in self.parameters())
        return {'survival_loss': float(loss.detach())}

    def after_validation(self, metrics):
        self.scheduler.step(metrics['survival_loss'])

    def get_extra_state(self):
        return {'assignments': {k: v.cpu() for k, v in self.assignments.items()},
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        self.assignments = {k: v.cpu() for k, v in state['assignments'].items()}
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def protocol(self):
        return {'objective': 'author batch Cox averaged over patients, plus L1', 'batch_size': self.batch_size,
                'l1_weight': self.l1_weight, 'clustering': self.clustering, 'clusters': self.clusters,
                'clustering_seed': self.clustering_seed, 'clustering_iterations': self.clustering_iterations,
                'clustering_scope': 'provided patient phenotype labels or explicit per-patient sklearn KMeans extension',
                'architecture': 'shared 1x1 Conv/ReLU/mean per phenotype, attention over nonempty phenotypes, risk MLP',
                'input_adaptation': 'author 4096-channel first Conv generalized to declared FM dimension',
                'numerical_repair': 'mask exponent before overflow; stable risk-set logsumexp',
                'all_censored_batch': 'skipped as author; tail patients retained',
                'scheduler': 'ReduceLROnPlateau on validation Cox+L1',
                'missing_scope': 'original patch extraction and unpublished phenotype preprocessing not reproduced'}


def deepattn_epoch(model, bundle, groups, optimizer, device, config):
    losses, skipped = [], 0
    for start in range(0, len(groups), model.batch_size):
        batch = groups[start:start+model.batch_size]
        events = torch.tensor([r[0]['event'] for r in batch], dtype=torch.float32, device=device)
        if not events.any():
            skipped += len(batch)
            continue
        predictions = torch.cat([model.forward_unit(bundle, rows, device, config, True)['logits'] for rows in batch])
        times = predictions.new_tensor([r[0]['time'] for r in batch])
        loss = patient_cox_mean(predictions, times, events)+model.l1_weight*sum(p.abs().sum() for p in model.parameters())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError('DeepAttnMISL epoch has no observed-event batch')
    return float(np.mean(losses))

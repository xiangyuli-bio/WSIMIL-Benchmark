"""Patient-level discrete hazards, censoring likelihood, and train-only time intervals."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def discrete_nll(logits, labels, events, alpha=0.):
    if logits.ndim != 2 or not 0 <= alpha < 1:
        raise ValueError('Expected [patients, bins] logits and 0 <= alpha < 1')
    if not torch.isfinite(labels).all() or not (labels == labels.long()).all():
        raise ValueError('Survival interval indices must be finite integers')
    labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
    events = events.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if len(labels) != len(logits) or len(events) != len(logits):
        raise ValueError('Survival label batch mismatch')
    if (labels < 0).any() or (labels >= logits.shape[1]).any() or not torch.isin(events, events.new_tensor([0, 1])).all():
        raise ValueError('Invalid time interval or event indicator')
    log_no_event = F.logsigmoid(-logits)
    inclusive = log_no_event.cumsum(1).gather(1, labels[:, None]).squeeze(1)
    current = log_no_event.gather(1, labels[:, None]).squeeze(1)
    observed = -(inclusive - current + F.logsigmoid(logits).gather(1, labels[:, None]).squeeze(1))
    # Equivalent to the authors' (1-alpha)*(censor+event) + alpha*event objective.
    return (events * observed - (1-events) * (1-alpha) * inclusive).mean()


class DiscreteSurvival(nn.Module):
    survival_objective = 'discrete_nll'

    def __init__(self, bins=4, alpha=0., accumulation_steps=1, l1_weight=0.):
        super().__init__()
        if not isinstance(bins, int) or bins < 2 or not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError('bins >= 2 and accumulation_steps >= 1 must be integers')
        if not 0 <= alpha < 1 or not np.isfinite(l1_weight) or l1_weight < 0:
            raise ValueError('Invalid censor weighting or L1 penalty')
        self.bins, self.alpha = bins, alpha
        self.accumulation_steps, self.l1_weight = accumulation_steps, l1_weight
        self.register_buffer('bin_edges', torch.zeros(bins-1, dtype=torch.float64))
        self.register_buffer('prepared', torch.tensor(False))

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        patients = bundle.rows[bundle.rows.subset == 'train'].drop_duplicates('global_patient_id')
        times = patients.loc[patients.event == 1, 'time'].to_numpy(dtype=np.float64)
        if len(np.unique(times)) < self.bins:
            raise ValueError(f'Need at least {self.bins} distinct observed training event times; reduce bins explicitly')
        edges = np.quantile(times, np.arange(1, self.bins)/self.bins)
        if not np.all(np.diff(edges) > 0):
            raise ValueError('Tied training quantiles; choose fewer survival bins')
        self.bin_edges.copy_(torch.tensor(edges, dtype=torch.float64, device=device))
        self.prepared.fill_(True)

    def time_label(self, time):
        if not self.prepared:
            raise RuntimeError('Survival intervals have not been fitted')
        return torch.bucketize(torch.as_tensor(time, dtype=torch.float64, device=self.bin_edges.device),
                               self.bin_edges, right=True)

    def result(self, hazard_logits):
        if hazard_logits.shape != (1, self.bins) or not torch.isfinite(hazard_logits).all():
            raise ValueError('Expected finite one-patient hazard logits with the configured bin count')
        hazards = hazard_logits.sigmoid()
        survival = F.logsigmoid(-hazard_logits).cumsum(1).exp()
        return {'logits': -survival.sum(1, keepdim=True), 'hazard_logits': hazard_logits,
                'hazards': hazards, 'survival': survival}

    def objective(self, result, time, event):
        return discrete_nll(result['hazard_logits'], self.time_label([time]),
                            self.bin_edges.new_tensor([event]), self.alpha)

    def training_losses(self, bundle, rows, device, config):
        """Yield additive patient-loss terms, allowing micro-batch graphs to be freed separately."""
        from .engine import forward_unit
        result = forward_unit(self, bundle, rows, device, config, train=True)
        yield self.objective(result, rows[0]['time'], rows[0]['event'])

    def protocol(self):
        return {'objective': self.survival_objective, 'event_definition': '1=observed; 0=censored',
                'bin_fit': 'unique training patients with observed events; linear quantiles',
                'interior_edges': self.bin_edges.tolist(), 'intervals': 'left closed; tails unbounded',
                'risk': '-sum(survival probabilities)', 'alpha': self.alpha,
                'accumulation_steps': self.accumulation_steps, 'l1_weight': self.l1_weight}


def discrete_epoch(model, bundle, groups, optimizer, device, config):
    losses = []
    for start in range(0, len(groups), model.accumulation_steps):
        window = groups[start:start+model.accumulation_steps]
        optimizer.zero_grad(set_to_none=True)
        for rows in window:
            patient_loss = 0.
            for loss in model.training_losses(bundle, rows, device, config):
                (loss / len(window)).backward()
                patient_loss += float(loss.detach())
            losses.append(patient_loss)
        if model.l1_weight:
            penalty = model.l1_weight * sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            penalty.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
    return float(np.mean(losses))

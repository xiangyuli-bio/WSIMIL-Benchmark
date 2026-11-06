"""Pinned TPMIL model, prototype objective and class-balanced slide training."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def normalized_attention(scores):
    """Author min/max rule; neutral soft assignment for an uninformative constant bag."""
    low, high = scores.min(), scores.max()
    if high == low:
        return torch.full_like(scores, .5)
    return ((scores - low) / (high - low)).clamp(1e-5, 1 - 1e-5)


def author_model(dim, hidden, attention_dim):
    return load_definitions('tpmil', 'model_tpmil.py',
        ['initialize_weights', 'Dist', 'GCPLoss', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, '_norm_attn_to_prob': normalized_attention},
        replacements=[("{'small': [1024, 512, 256], 'big': [1024, 512, 384]}",
                       repr({'small': [dim, hidden, attention_dim], 'big': [dim, hidden, attention_dim]}))])


class TPMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, attention_dim=256, gate=True,
                 dropout=True, weighted_sampling=True):
        super().__init__()
        if min(dim, hidden, attention_dim) < 1 or classes < 2:
            raise ValueError('TPMIL requires positive dimensions and at least two classes')
        if any(not isinstance(v, bool) for v in (gate, dropout, weighted_sampling)):
            raise ValueError('gate, dropout and weighted_sampling must be booleans; dropout=True means p=.25')
        self.dim, self.classes = dim, classes
        self.weighted_sampling = weighted_sampling
        self.network = author_model(dim, hidden, attention_dim).CLAM_SB(
            gate=gate, dropout=dropout, n_classes=classes)

    def configure_training(self, cfg):
        clip = cfg['grad_clip']
        if clip is not None and (not np.isfinite(clip) or clip <= 0):
            raise ValueError('grad_clip must be finite positive or null')

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x):
            raise ValueError('TPMIL requires a nonempty [instances, feature_dim] bag')
        if not torch.isfinite(x).all():
            raise ValueError('TPMIL features must be finite')
        if self.training:
            if label is None or label.numel() != 1 or label.dtype != torch.long or not 0 <= label.item() < self.classes:
                raise ValueError('TPMIL training requires one integer slide label in range')
            logits, probability, _, prototype_loss, extra = self.network(
                x, label=label.reshape(1), instance_eval=False, return_features=True)
            ce = F.cross_entropy(logits, label.reshape(1))
            return {'logits': logits, 'probabilities': probability, 'embedding': extra['features'],
                    'prototype_loss': prototype_loss, 'bag_loss': ce,
                    'total_loss': .5 * ce + .5 * prototype_loss}
        # Labels only construct the auxiliary training targets. Prediction never needs labels.
        raw, h = self.network.attention_net(x)
        attention = raw.T.softmax(1)
        embedding = attention @ h
        return {'logits': self.network.classifiers(embedding), 'attention': attention, 'embedding': embedding}

    def protocol(self):
        return {'method': 'TPMIL', 'input_dim': self.dim,
                'prototypes': self.classes + 1, 'prototype_distance': 'squared_l2 / projected_feature_dim',
                'objective': '.5 CE + .5 KLDiv(batchmean); no CLAM instance clustering loss',
                'pseudo_targets': 'detached minmax raw attention to slide class; complement to extra background prototype',
                'constant_attention_policy': 'explicit repair: neutral .5/.5 assignment (author code divides by zero)',
                'weighted_sampling': self.weighted_sampling,
                'prediction': 'label-free author attention bag classifier',
                'selection': 'validation log_loss by default; no test labels used',
                'scope': 'public single-branch TPMIL; paper numerical reproduction not run'}


def sampled_groups(groups, weighted):
    if not weighted:
        return groups
    labels = torch.tensor([rows[0]['target'] for rows in groups])
    counts = labels.bincount()
    # Equivalent weights to CLAM's N / class_count, with replacement, N draws.
    weights = (len(groups) / counts[labels].double())
    return [groups[i] for i in torch.multinomial(weights, len(groups), replacement=True).tolist()]


def tpmil_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    results = []
    for rows in sampled_groups(groups, model.weighted_sampling):
        x, coords = load_unit(bundle, rows, device, cfg['max_instances'], train=True)
        target = torch.tensor([rows[0]['target']], device=device)
        result = model(x, coords=coords, label=target)
        loss = result['total_loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite TPMIL training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'] or float('inf'), error_if_nonfinite=True)
        optimizer.step()
        results.append({k: float(result[v].detach()) for k, v in
                        [('loss', 'total_loss'), ('bag_loss', 'bag_loss'), ('prototype_loss', 'prototype_loss')]})
    return results

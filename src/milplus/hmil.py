"""HMIL released training recipe with an explicitly restored scalar class head.

The current author model produces C×C logits where its trainer expects 1×C.
Restore Linear(D,1) + transpose, present in author commit af6f85a; keep the
current source objective (which differs from the paper). This is deliberately
named hmil_source_repaired, not a claim of numerical paper reproduction.
"""
import math
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import classification_metrics
from .upstream import load_definitions


def author_network(dim, hidden, attention_dim):
    return load_definitions('hmil', 'src/models.py', ['initialize_weights', 'GatedAttention', 'HMIL'],
        {'torch': torch, 'nn': nn, 'F': F, 'List': list, 'Tuple': tuple}, replacements=[
            ('self.feature_dim = 1024', f'self.feature_dim = {dim}'),
            ('self.reduced_dim = 256', f'self.reduced_dim = {hidden}'),
            ('self.attention_output_dim = 64', f'self.attention_output_dim = {attention_dim}'),
            ('x = x.squeeze()', 'x = x.reshape(-1, self.feature_dim)'),
            ('nn.Linear(self.feature_dim, n_classes[0])', 'nn.Linear(self.feature_dim, 1)'),
            ('nn.Linear(self.reduced_dim, n_classes[1])', 'nn.Linear(self.reduced_dim, 1)'),
            ('self.coarse_classifier(M1)', 'self.coarse_classifier(M1).T'),
            ('self.fine_classifier(M2)', 'self.fine_classifier(M2).T')]).HMIL


def author_losses():
    loss = load_definitions('hmil', 'src/loss.py', ['SupConLoss'],
                           {'torch': torch, 'nn': nn, 'Optional': Optional})
    return load_definitions('hmil', 'src/trainer.py',
        ['compute_hierarchical_loss', 'compute_attention_matching_loss', 'compute_temperature',
         'compute_contrastive_loss'],
        {'torch': torch, 'F': F, 'math': math, 'List': list, 'Tuple': tuple, 'Any': Any,
         'SupConLoss': loss.SupConLoss, 'gen_mapping_dict': lambda cfg: dict(enumerate(cfg.mapping))},
        replacements=[
            ('torch.zeros(preds_fine.shape[0], cfg.n_class[0]).cuda()',
             'preds_fine.new_zeros((preds_fine.shape[0], cfg.n_class[0]))'),
            ('torch.zeros(cfg.n_class[0], attention_fine.shape[1]).cuda()',
             'attention_fine.new_zeros((cfg.n_class[0], attention_fine.shape[1]))'),
            ('SupConLoss(temperature=tau).cuda()', 'SupConLoss(temperature=tau).to(semantics_features.device)')])


class HierarchicalMIL(nn.Module):
    def __init__(self, dim, classes, hierarchy, hidden=None, attention_dim=None,
                 dropout=False, batch_size=64):
        super().__init__()
        if hierarchy is None or len(hierarchy.fine) != classes:
            raise ValueError('HMIL requires explicit hierarchy labels matching the task classes')
        hidden = max(1, dim // 4) if hidden is None else hidden
        attention_dim = max(1, hidden // 4) if attention_dim is None else attention_dim
        if any(type(x) is not int or x < 1 for x in (dim, hidden, attention_dim, batch_size)):
            raise ValueError('HMIL dimensions and batch_size must be positive integers')
        if not isinstance(dropout, bool):
            raise TypeError('HMIL dropout is boolean; true means author p=.25')
        self.dim, self.classes, self.hierarchy = dim, classes, hierarchy
        self.batch_size = batch_size
        self.network = author_network(dim, hidden, attention_dim)([len(hierarchy.coarse), classes], dropout)
        self.losses = author_losses()
        self.register_buffer('mapping', torch.tensor(hierarchy.mapping, dtype=torch.long))
        self.epoch, self.total_epochs = 0, 200

    def configure_training(self, cfg):
        self.total_epochs = cfg['epochs']
        clip = cfg['grad_clip']
        if clip is not None and (not np.isfinite(clip) or clip <= 0):
            raise ValueError('grad_clip must be finite positive or null')

    def start_epoch(self, epoch):
        if not 0 <= epoch < self.total_epochs:
            raise ValueError('HMIL epoch outside declared curriculum')
        self.epoch = epoch

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('HMIL requires finite nonempty [instances, feature_dim] features')
        attention, logits, embeddings = self.network(x)
        coarse, fine = logits
        probability = fine.softmax(1)
        mapped = probability.new_zeros((1, len(self.hierarchy.coarse)))
        mapped.scatter_add_(1, self.mapping.unsqueeze(0), probability)
        return {'logits': fine, 'coarse_logits': coarse, 'attention_coarse': attention[0],
                'attention': attention[1], 'embedding': embeddings,
                'mapped_coarse_probabilities': mapped}

    def batch_loss(self, outputs, targets):
        from types import SimpleNamespace
        if (not outputs or targets.shape != (len(outputs),) or targets.dtype != torch.long or
                (targets < 0).any() or (targets >= self.classes).any()):
            raise ValueError('HMIL needs one integer fine label per bag')
        cfg = SimpleNamespace(mapping=self.hierarchy.mapping,
                              n_class=[len(self.hierarchy.coarse), self.classes], num_epochs=self.total_epochs)
        coarse = torch.cat([r['coarse_logits'] for r in outputs])
        fine = torch.cat([r['logits'] for r in outputs])
        ce_coarse, ce_fine, alignment = self.losses.compute_hierarchical_loss(
            coarse, fine, self.mapping[targets], targets, cfg)
        matching = self.losses.compute_attention_matching_loss(
            [r['attention_coarse'] for r in outputs], [r['attention'] for r in outputs], cfg)
        contrastive, tau = self.losses.compute_contrastive_loss(
            torch.stack([r['embedding'] for r in outputs]), targets, self.epoch, cfg)
        alpha = 1 - (self.epoch / self.total_epochs)**2
        total = alpha * (ce_coarse + matching) + ce_fine + alignment + (1-alpha) * contrastive
        return {'loss': total, 'coarse_ce': ce_coarse, 'fine_ce': ce_fine,
                'bag_alignment': alignment, 'attention_matching': matching, 'contrastive': contrastive,
                'alpha': fine.new_tensor(alpha), 'temperature': fine.new_tensor(tau)}

    def prediction_fields(self, result, row):
        record = {'coarse_target': self.hierarchy.mapping[row['target']]}
        for prefix, values in [('coarse_p', result['coarse_logits'].softmax(1)),
                               ('mapped_coarse_p', result['mapped_coarse_probabilities'])]:
            record.update({f'{prefix}_{i}': float(v) for i, v in enumerate(values[0])})
        return record

    def validation_metrics(self, frame):
        result = {}
        for prefix in ('coarse', 'mapped_coarse'):
            view = frame[['global_patient_id', 'coarse_target']].rename(columns={'coarse_target': 'target'}).copy()
            for i in range(len(self.hierarchy.coarse)):
                view[f'p_{i}'] = frame[f'{prefix}_p_{i}']
            values = classification_metrics(view, len(self.hierarchy.coarse), self.hierarchy.evaluation_unit)
            result.update({f'{prefix}_{key}': value for key, value in values.items()})
        return result

    def protocol(self):
        return {'method': 'HMIL source recipe with scalar-head repair', 'hierarchy': self.hierarchy.protocol(),
                'classifier_repair': 'Linear(D,1)+transpose restored from author af6f85ae38303fa0afb14ea479c47355fb3eebbe',
                'feature_dim': self.dim, 'fine_dim': self.network.reduced_dim,
                'batch_size': self.batch_size, 'sampling': 'shuffle slide bags; retain final incomplete batch',
                'objective': 'alpha*(CE_coarse+attention_matching)+CE_fine+CE(mapped_fine_probs)+(1-alpha)*SupCon/batch_size',
                'curriculum': 'alpha=1-(epoch/epochs)^2; epoch starts at zero',
                'contrastive': 'class-wise normalized fine embeddings as views; all anchors; cosine-cycle temperature .1..1',
                'optimizer': 'Adam; no scheduler; optional explicit gradient clip',
                'selection': 'validation fine macro_auc by default; test only after training',
                'scope': 'released source recipe; bag CE, weighting and temperature differ from paper equations; no numerical paper reproduction'}


def hmil_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    history = []
    for offset in range(0, len(groups), model.batch_size):
        batch = groups[offset:offset + model.batch_size]
        outputs = [model(load_unit(bundle, rows, device, cfg['max_instances'], train=True)[0]) for rows in batch]
        target = torch.tensor([rows[0]['target'] for rows in batch], device=device)
        terms = model.batch_loss(outputs, target)
        if not torch.isfinite(terms['loss']):
            raise FloatingPointError('Nonfinite HMIL loss')
        optimizer.zero_grad(set_to_none=True)
        terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'] or float('inf'), error_if_nonfinite=True)
        optimizer.step()
        history.append({key: float(value.detach()) for key, value in terms.items()})
    return history

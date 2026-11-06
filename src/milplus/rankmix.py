"""RankMix author backbones, supervised initialization and binary feature mixing.

The literal source indexes the argsort *positions* of selected patch IDs. The
default profiles repair that indexing and exclude FRMIL padding from selection.
The `_source` profiles deliberately retain both behaviors for audit comparisons.
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


def author_definitions():
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'Tuple': tuple}
    dsmil = load_definitions('rankmix', 'dsmil.py', ['FCLayer', 'BClassifier', 'MILNet'], namespace)
    frmil = load_definitions('rankmix', 'frmil.py', ['MAB', 'FRMIL'], namespace)
    mix = load_definitions('rankmix', 'train_mil_rankmix.py',
        ['mixup_features', 'mixup_label', 'mixup', 'mixup_by_rank'], namespace)
    loss = load_definitions('rankmix', 'train_mil.py', ['FeatMag'], namespace)
    return dsmil, frmil, mix, loss


def selected_ids(scores, count, descending=True, literal=False):
    ranked = torch.sort(scores.flatten(), descending=descending).indices[:count]
    # Preserve source patch ordering after selecting the high/low score subset.
    ordered = torch.sort(ranked)
    return ordered.indices if literal else ordered.values


class RankMix(nn.Module):
    def __init__(self, dim, classes, backbone='dsmil', literal=False, warmup_epochs=20,
                 alpha=.1, heads=8, margin=10.17, rank_direction='rank', keep_percent=100):
        super().__init__()
        if classes != 2 or backbone not in {'dsmil', 'frmil'}:
            raise ValueError('RankMix admitted author recipe requires binary classification')
        if (not isinstance(warmup_epochs, int) or warmup_epochs < 1 or
                not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(margin) or margin <= 0):
            raise ValueError('RankMix requires positive warmup_epochs, alpha and margin')
        if rank_direction not in {'rank', 'rank2'} or not isinstance(keep_percent, int) or not 1 <= keep_percent <= 100:
            raise ValueError('RankMix requires rank/rank2 and integer keep_percent in [1,100]')
        if heads < 1 or (backbone == 'frmil' and dim % heads):
            raise ValueError('FRMIL feature dimension must be divisible by attention heads')
        dsmil, frmil, mix, loss = author_definitions()
        self.backbone, self.literal = backbone, literal
        self.warmup_epochs, self.alpha, self.margin = warmup_epochs, alpha, margin
        self.rank_direction, self.keep_percent = rank_direction, keep_percent
        self.minimum_selection_epoch = warmup_epochs
        if backbone == 'dsmil':
            self.network = dsmil.MILNet(dsmil.FCLayer(dim, 1), dsmil.BClassifier(dim, 1))
        else:
            self.network = frmil.FRMIL(SimpleNamespace(num_classes=1, feats_size=dim, n_heads=heads))
        self.magnitude = loss.FeatMag(margin)
        self.author_mix = mix
        self.register_buffer('padding_length', torch.tensor(0))
        self.epoch = -1
        self.warmup_best = None
        self.warmup_score = -float('inf')
        self.stage_transitions = 0
        self.scheduler = None
        self.optimizer = None
        self.loaded_scheduler = None

    def configure_training(self, cfg):
        if cfg['epochs'] <= self.warmup_epochs:
            raise ValueError('RankMix epochs must include supervised warmup AND RankMix finetuning')
        if cfg['grad_clip'] is not None and cfg['grad_clip'] <= 0:
            raise ValueError('grad_clip must be positive or null')
        self.total_epochs = cfg['epochs']
        self.selection_metric = cfg['selection_metric']

    def prepare(self, bundle, seed, device):
        from .engine import load_unit, units
        training = units(bundle, 'train')
        if {g[0]['target'] for g in training} != {0, 1}:
            raise ValueError('RankMix requires both classes in training')
        if self.backbone == 'frmil':
            self.padding_length.fill_(max(len(load_unit(bundle, rows, 'cpu')[0]) for rows in training))

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr,
                                          betas=(.5, .9), weight_decay=weight_decay)
        self.initial_lr = lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.warmup_epochs, eta_min=min(lr, .000005))
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch = epoch
        if epoch == self.warmup_epochs and self.stage_transitions == 0:
            if self.warmup_best is None:
                raise ValueError('RankMix has no validation-selected supervised initialization')
            self.network.load_state_dict(self.warmup_best)
            self.optimizer.state.clear()
            for group in self.optimizer.param_groups:
                group['lr'] = self.initial_lr
                group['initial_lr'] = self.initial_lr
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.total_epochs - self.warmup_epochs,
                eta_min=min(self.initial_lr, .000005))
            self.stage_transitions += 1

    def finish_epoch(self, bundle, device, cfg):
        self.scheduler.step()

    def after_validation(self, metrics):
        if self.epoch < self.warmup_epochs:
            value = metrics[self.selection_metric]
            value = -value if self.selection_metric == 'log_loss' else value
            if np.isfinite(value) and value > self.warmup_score:
                self.warmup_score = value
                self.warmup_best = copy.deepcopy(self.network.state_dict())

    def get_extra_state(self):
        return {'epoch': self.epoch, 'warmup_best': self.warmup_best,
                'warmup_score': self.warmup_score, 'stage_transitions': self.stage_transitions,
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.loaded_scheduler}

    def set_extra_state(self, state):
        self.epoch, self.warmup_best = state['epoch'], state['warmup_best']
        self.warmup_score, self.stage_transitions = state['warmup_score'], state['stage_transitions']
        self.loaded_scheduler = state['scheduler']
        if self.scheduler is not None and self.loaded_scheduler is not None:
            self.scheduler.load_state_dict(self.loaded_scheduler)

    def protocol(self):
        return {'source': 'willpower057/RankMix', 'backbone': self.backbone,
                'output': 'one positive logit; bag sigmoid only at inference',
                'ranking': 'literal author argsort-position indexing, including padding' if self.literal else
                           'repaired selected patch IDs in source order; padded positions excluded',
                'warmup': 'supervised author backbone on training bags, held-out validation-selected checkpoint',
                'warmup_epochs': self.warmup_epochs, 'minimum_selection_epoch': self.minimum_selection_epoch,
                'finetune': 'joint score and bag network; reset Adam and cosine schedule',
                'lambda': f'Beta({self.alpha}, {1 if self.backbone == "dsmil" else self.alpha})',
                'sampler': 'natural-frequency bag pass + inverse-class-frequency replacement' if self.backbone == 'dsmil'
                           else 'one bag per class per episode; N_train episodes',
                'rank_direction': self.rank_direction, 'keep_percent': self.keep_percent,
                'padding': 'none' if self.backbone == 'dsmil' else 'maximum training bag; retained in author backbone/loss',
                'input_dropout': .2, 'margin': self.margin,
                'scope': 'binary frozen-feature adaptation; no paper score reproduction or multiclass claim'}

    @torch.no_grad()
    def mix_pair(self, bags, labels, lam, lengths=None):
        if not 0 <= lam <= 1 or len(bags) != 2 or len(labels) != 2:
            raise ValueError('RankMix needs two bags/labels and lambda in [0,1]')
        lengths = [len(b) for b in bags] if lengths is None else list(lengths)
        count = int(min(lengths) * self.keep_percent / 100)
        if count < 1:
            raise ValueError('RankMix keep_percent selects no instances')
        previous_mode = self.network.training
        self.network.eval()
        try:
            if self.literal:
                strategy = self.rank_direction + (f'_{self.keep_percent}' if self.keep_percent != 100 else '')
                features = bags if self.backbone == 'dsmil' else torch.stack(tuple(bags))
                valid_lengths = None if self.backbone == 'dsmil' else torch.tensor(lengths, device=labels.device)
                return self.author_mix.mixup_by_rank(features, labels.reshape(2, 1), valid_lengths,
                    self.network, strategy, self.author_mix.mixup, lam)
            if self.backbone == 'dsmil':
                scores = [self.network(bag)[0].flatten() for bag in bags]
            else:
                scores = self.network(torch.stack(tuple(bags)))[1]
            selected = []
            for i, bag in enumerate(bags):
                descending = self.rank_direction == 'rank' or bool(labels[i] == 1)
                ids = selected_ids(scores[i][:lengths[i]], count, descending=descending)
                selected.append(bag.index_select(0, ids))
            if self.backbone == 'frmil':
                selected = [F.pad(b, (0, 0, 0, len(bags[i])-len(b))) for i, b in enumerate(selected)]
            return (1-lam)*selected[0] + lam*selected[1], (1-lam)*labels[0] + lam*labels[1]
        finally:
            self.network.train(previous_mode)

    def forward(self, x, coords=None, label=None):
        if self.backbone == 'dsmil':
            positive = self.network(x)[1].reshape(1, 1)
        else:
            positive = self.network(x[None])[0].reshape(1, 1)
        return {'logits': torch.cat((torch.zeros_like(positive), positive), dim=1)}

    def dsmil_objective(self, x, label):
        instances, bag, _, _ = self.network(F.dropout(x, p=.2, training=True))
        target = label.reshape(1, 1)
        bag_loss = F.binary_cross_entropy_with_logits(bag, target)
        instance_loss = F.binary_cross_entropy_with_logits(instances.max(0).values.reshape(1, 1), target)
        return {'loss': (bag_loss + instance_loss)/2, 'bag_loss': bag_loss, 'instance_loss': instance_loss}

    def frmil_objective(self, features, labels, mixed=False):
        logits, query, scores = self.network(F.dropout(features, p=.2, training=True))
        select = slice(2, 3) if mixed else slice(0, 2)
        targets = labels[select].reshape(-1)
        bag_loss = F.binary_cross_entropy_with_logits(logits[select].flatten(), targets)
        instance_loss = F.binary_cross_entropy(scores.max(1).values[select], targets)
        positive = int(labels[:2].argmax())
        negative = 1-positive
        magnitude = self.magnitude(query[positive:positive+1], query[negative:negative+1], w_scale=query.shape[1])
        return {'loss': (bag_loss + instance_loss + magnitude)*(1./3),
                'bag_loss': bag_loss, 'instance_loss': instance_loss, 'magnitude_loss': magnitude}


def rankmix_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    finetune = model.epoch >= model.warmup_epochs
    class_indices = [[i for i, g in enumerate(groups) if g[0]['target'] == c] for c in range(2)]
    if not all(class_indices):
        raise ValueError('RankMix requires both classes in training')
    if model.backbone == 'dsmil':
        weights = torch.tensor([1/len(class_indices[g[0]['target']]) for g in groups])
        pairs = torch.multinomial(weights, len(groups), replacement=True).tolist() if finetune else None
        episodes = [[i, pairs[i]] if finetune else [i] for i in range(len(groups))]
    else:
        episodes = []
        for _ in groups:
            order = torch.randperm(2).tolist()
            episodes.append([class_indices[c][int(torch.randint(len(class_indices[c]), (1,)))] for c in order])
    losses = []
    for indices in episodes:
        rows = [groups[i] for i in indices]
        bags = [load_unit(bundle, r, device, cfg['max_instances'], train=True)[0] for r in rows]
        labels = bags[0].new_tensor([r[0]['target'] for r in rows])
        if model.backbone == 'dsmil':
            if finetune:
                lam = float(np.random.beta(model.alpha, 1))
                x, y = model.mix_pair(bags, labels, lam)
            else:
                x, y = bags[0], labels[0]
            result = model.dsmil_objective(x, y)
        else:
            lengths = [len(b) for b in bags]
            size = int(model.padding_length)
            if cfg['max_instances']:
                size = min(size, cfg['max_instances'])
            features = torch.stack([F.pad(b, (0, 0, 0, size-len(b))) for b in bags])
            if finetune:
                lam = float(np.random.beta(model.alpha, model.alpha))
                mixed, target = model.mix_pair(features, labels, lam, lengths)
                features = torch.cat((features, mixed[None]))
                labels = torch.cat((labels, target.reshape(1)))
            result = model.frmil_objective(features, labels, mixed=finetune)
        if not torch.isfinite(result['loss']):
            raise FloatingPointError('Nonfinite RankMix objective')
        optimizer.zero_grad(set_to_none=True)
        result['loss'].backward()
        if cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite RankMix gradient')
        optimizer.step()
        losses.append({**{k: float(v.detach()) for k, v in result.items()}, 'rankmix_stage': float(finetune)})
    return losses

"""Complete INS frozen-feature source repair with an explicit input condition.

The published feature script really maps each full feature vector to a repeated
3x32x(D/32) pseudo-image and trains ResNet18. Keep that input condition explicit.
Original source momentum/prototype/queue/partial/SupCon kernels are used. Fix
the unused two-class bag head and derive all weak labels from bag labels only.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models

from .io import digest_object
from .upstream import load_definitions, verified_source

COMMIT = 'cd97ba2fa7df2a951f218e6de405f664129931c8'


def author_kernels():
    _, sha = verified_source('ins')
    if sha != COMMIT:
        raise ValueError('INS source differs')
    backbone = load_definitions('ins', 'resnet.py',
        ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet18', 'LinearBatchNorm', 'Identity', 'SupConResNet'],
        {'torch': torch, 'nn': nn, 'F': F, 'models': models},
        replacements=[('print(x.shape)', 'pass # omit source shape logging')])
    backbone.__dict__['model_dict'] = {'resnet18': (backbone.resnet18, 512)}
    heads = load_definitions('ins', 'alexnet.py',
        ['AlexNet_CIFAR10_attention', 'AlexNet_CIFAR10_dsmil', 'teacher_Attention_head', 'teacher_DSMIL_head'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math})
    original = load_definitions('ins', 'model.py', ['INS'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np,
         'teacher_Attention_head': heads.teacher_Attention_head,
         'teacher_DSMIL_head': heads.teacher_DSMIL_head,
         'concat_all_gather': lambda tensor: tensor.detach()},
        replacements=[('pretrained = True', 'pretrained = False # paper states no encoder pretraining'),
            ('bag_prediction = torch.softmax(bag_prediction, dim=1)',
             'bag_prediction = self.bag_classifier(bag_prediction) # restore declared Cway bag head'),
            ("print('val:',args.moco_queue, batch_size)", 'pass # omit source queue logging')])
    losses = load_definitions('ins', 'utils/utils_loss.py', ['partial_loss', 'SupConLoss'],
        {'torch': torch, 'nn': nn, 'F': F},
        replacements=[('F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()',
                       'F.one_hot(prot_pred, batchY.shape[1]).float().to(batchY.device).detach()'),
            ("torch.zeros(logits.shape[0], dtype=torch.long).cuda()",
             "torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)")])
    return backbone.SupConResNet, original.INS, losses


def source_single_rank_network(args):
    encoder, original, losses = author_kernels()

    class SingleRankINS(original):
        @torch.no_grad()
        def _batch_shuffle_ddp(self, x, y, partial):
            order = torch.randperm(len(x)).to(x.device)
            return x[order], y[order], partial[order], order.argsort()

        @torch.no_grad()
        def _batch_unshuffle_ddp(self, x, y, partial, order):
            return x[order], y[order], partial[order]

    return SingleRankINS(args, encoder), losses


class INSFrozenReference(nn.Module):
    def __init__(self, dimension, bag_labels, queue_size=8192, projection=128,
                 momentum=.999, prototype_momentum=.9, warmup_epochs=20,
                 confidence_start=.95, confidence_end=.8, instance_weight=1.,
                 contrastive_weight=.5, bag_weight=1.):
        super().__init__()
        if type(dimension) is not int or dimension < 32 or dimension % 32:
            raise ValueError('Actual source frozen-feature layout needs complete D divisible by32, without crop/pad')
        if any(type(v) is not int or v < 1 for v in [queue_size, projection]) or type(warmup_epochs) is not int or warmup_epochs < 1:
            raise ValueError('INS needs positive queue/projection/warmup settings')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 < v < 1
               for v in [momentum, prototype_momentum, confidence_start, confidence_end]):
            raise ValueError('INS EMA coefficients must be finite and in (0,1)')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0
               for v in [instance_weight, contrastive_weight, bag_weight]):
            raise ValueError('All three INS objectives must remain active')
        y = torch.as_tensor(bag_labels)
        if y.ndim != 1 or not len(y) or y.dtype == torch.bool or not torch.equal(y, y.long()) or not set(y.tolist()) <= {0, 1}:
            raise ValueError('Each actual TRAIN patch needs its binary bag label, not hidden instance truth')
        self.args = SimpleNamespace(num_class=2, low_dim=projection, arch='resnet18', dataset='cam16_mil',
            moco_queue=queue_size, moco_m=momentum, proto_m=prototype_momentum,
            conf_ema_range=[confidence_start, confidence_end])
        self.network, kernels = source_single_rank_network(self.args)
        self.register_buffer('training_bag_labels', y.long())
        self.register_buffer('confidence', torch.stack([torch.where(y == 0, 1., .5), torch.where(y == 0, 0., .5)], 1))
        self.register_buffer('queue_valid', torch.zeros(queue_size, dtype=torch.bool))
        self.register_buffer('queue_indices', torch.full((queue_size,), -1, dtype=torch.long))
        self.partial_loss = kernels.partial_loss(self.confidence)
        self.contrastive_loss = kernels.SupConLoss()
        self.dimension, self.warmup_epochs = dimension, warmup_epochs
        self.instance_weight, self.contrastive_weight, self.bag_weight = instance_weight, contrastive_weight, bag_weight
        self.confidence_momentum = .8  # Literal initial partial_loss default; scheduled after every epoch.
        self.steps = self.bag_steps = 0
        self.contract = digest_object({'source': COMMIT, 'dimension': dimension, 'bag_labels': y.tolist(),
            'settings': vars(self.args), 'warmup_epochs': warmup_epochs,
            'weights': [instance_weight, contrastive_weight, bag_weight],
            'queue_policy': 'only real TRAIN keys; original random slots masked until written',
            'encoder': 'actual author ResNet18 from scratch on full source feature pseudo-image',
            'bag_head': 'declared original512to2 linear head on mean backbone features, explicitly repaired'})

    def image(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimension or not len(features) or not torch.isfinite(features).all():
            raise ValueError('INS full feature input changed')
        return features.reshape(len(features), 32, -1)[:, None].repeat(1, 3, 1, 1)

    def forward(self, features):
        return self.network(self.image(features), args=self.args, bag_flag=True)

    def instance_logits(self, features):
        return self.network(self.image(features), args=self.args, eval_only=True)

    def instance_objective(self, weak, strong, index, epoch):
        index = torch.as_tensor(index, dtype=torch.long, device=self.confidence.device)
        if index.ndim != 1 or len(index) != len(weak) or not len(index) or index.min() < 0 or index.max() >= len(self.confidence):
            raise ValueError('INS confidence indices must be actual TRAIN patch identities')
        if self.args.moco_queue % len(index):
            raise ValueError('Original INS queue requires queue_size divisible by full minibatch size')
        bag_y = self.training_bag_labels[index]
        if epoch < self.warmup_epochs and not (bag_y == 1).all():
            raise ValueError('Source frozen-feature warmup samples positive-bag patches only')
        partial = torch.ones((len(index), 2), device=weak.device)
        partial[bag_y == 0, 1] = 0
        prior_valid = self.queue_valid.clone()
        pointer = int(self.network.queue_ptr)
        output, features, pseudo, _candidate, proto_scores = self.network(
            self.image(weak), self.image(strong), partial, self.args)
        # Source returns previous queue snapshots before enqueuing. Drop only
        # slots that were not actual observed TRAIN keys at that instant.
        positions = torch.cat([torch.arange(2*len(index), device=weak.device),
                               2*len(index)+prior_valid.nonzero(as_tuple=True)[0]])
        features, pseudo = features[positions], pseudo[positions]
        self.queue_valid[pointer:pointer+len(index)] = True
        self.queue_indices[pointer:pointer+len(index)] = index
        self.partial_loss.confidence = self.confidence
        self.partial_loss.conf_ema_m = self.confidence_momentum
        if epoch >= self.warmup_epochs:
            # point=0 is known only for a negative BAG. No patch annotations.
            self.partial_loss.confidence_update(proto_scores, index, partial, point=bag_y)
            labels = pseudo.argmax(1)
            mask = labels[:len(index), None].eq(labels[None]).float()
        else:
            mask = None
        contrast = self.contrastive_loss(features, mask=mask, batch_size=len(index))
        classification = self.partial_loss(output, index)
        loss = self.instance_weight*classification+self.contrastive_weight*contrast
        self.steps += 1
        return {'loss': loss, 'classification': classification, 'contrastive': contrast,
            'output': output, 'features': features, 'pseudo_scores': pseudo, 'prototype_scores': proto_scores,
            'mask': mask, 'prior_valid_keys': int(prior_valid.sum()), 'index': index.clone()}

    def finish_epoch(self, epoch, total_epochs):
        self.confidence_momentum = self.args.conf_ema_range[0]+epoch/total_epochs*(self.args.conf_ema_range[1]-self.args.conf_ema_range[0])

    def get_extra_state(self):
        return {'contract': self.contract, 'confidence_momentum': self.confidence_momentum,
                'steps': self.steps, 'bag_steps': self.bag_steps}

    def set_extra_state(self, state):
        if state['contract'] != self.contract:
            raise ValueError('INS source/feature/weak-label/recipe contract changed')
        self.confidence_momentum, self.steps, self.bag_steps = state['confidence_momentum'], state['steps'], state['bag_steps']


def frozen_epoch(model, features, bags, optimizer, epoch, total_epochs, batch_size=64):
    """Actual source phase order, complete instance minibatches then all bag CE."""
    model.train()
    eligible = (model.training_bag_labels == 1).nonzero(as_tuple=True)[0] if epoch < model.warmup_epochs else torch.arange(len(features), device=features.device)
    if len(eligible) < batch_size:
        raise ValueError('Real eligible TRAIN instances do not fill a source minibatch; no fabricated duplicates')
    if model.args.moco_queue % batch_size:
        raise ValueError('Native source queue/full batch divisibility required')
    class Indices(Dataset):
        def __len__(self):
            return len(eligible)

        def __getitem__(self, i):
            return eligible[i]

    dataset = Indices()
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=0)
    sampler.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
    results = []
    for ids in loader:
        # Literal frozen source dropout uses training=False for both views.
        result = model.instance_objective(features[ids], features[ids], ids, epoch)
        optimizer.zero_grad()
        result['loss'].backward()
        optimizer.step()
        results.append({'loss': float(result['loss'].detach()),
            'instance': float(result['classification'].detach()), 'contrastive': float(result['contrastive'].detach()),
            'prior_valid_keys': result['prior_valid_keys']})
    for ids in bags:
        ids = torch.as_tensor(ids, device=features.device, dtype=torch.long)
        y = model.training_bag_labels[ids]
        if not len(ids) or not (y == y[0]).all():
            raise ValueError('Each indexed bag requires one actual bag label')
        logits = model(features[ids])
        loss = F.cross_entropy(logits, y[:1])*model.bag_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.bag_steps += 1
        results.append({'loss': float(loss.detach()), 'bag': float(loss.detach())})
    model.finish_epoch(epoch, total_epochs)
    return results

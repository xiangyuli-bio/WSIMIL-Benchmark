"""Pinned SMMILe numerical networks shared by the full recipes and source audits."""
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .smmile_data import positive_int
from .upstream import load_definitions, pinned_text, verified_source

VARIANTS = {'binary', 'multiclass', 'multilabel'}


def stable_span(value):
    """Keep all nonzero spans literal; constant inputs normalize to zero."""
    return torch.where(value == 0, torch.ones_like(value), value)


def author_network(variant='multiclass', dimension=1024, classes=3, size='small',
                   dropout=True, dropout_rate=.25, refs=3, repair=True):
    if variant not in VARIANTS or size not in {'small', 'big'}:
        raise ValueError('Unknown SMMILe variant or source size')
    positive_int(dimension, 'dimension')
    positive_int(classes, 'classes')
    positive_int(refs, 'refs')
    if variant == 'binary' and classes != 2 or variant != 'binary' and classes < 2:
        raise ValueError('SMMILe binary needs two task classes; other routes need at least two')
    if type(dropout) is not bool or not isinstance(dropout_rate, (int, float)) or isinstance(dropout_rate, bool) or not np.isfinite(dropout_rate) or not 0 <= dropout_rate < 1:
        raise ValueError('Invalid SMMILe dropout options')
    root, sha = verified_source('smmile')
    branch = 'multi' if variant == 'multilabel' else 'single'
    init = load_definitions('smmile', branch+'/utils/utils.py', ['initialize_weights'],
                            {'torch': torch, 'nn': nn}).initialize_weights
    relative = branch+'/models/model_smmile.py'
    code = pinned_text(root, sha, relative)
    patches = []
    if '.cuda()' in code:
        patches.append(('.cuda()', '.to(device)'))
    if repair:
        patches.extend([
            ('(final_score[:,i].max()-final_score[:,i].min())',
             'stable_span(final_score[:,i].max()-final_score[:,i].min())'),
            ('(cls_final_score.max()-cls_final_score.min())',
             'stable_span(cls_final_score.max()-cls_final_score.min())'),
            ('(valid_loss.max() - valid_loss.min())',
             'stable_span(valid_loss.max() - valid_loss.min())'),
            ('torch.zeros((sp_list.max()+1, scores.shape[-1]))',
             'scores.new_zeros((int(sp_list.max())+1, scores.shape[-1]))')])
        line = 'adj_sp_scores = sp_scores[(one_adj==1)]'
        matches = [s for s in code.splitlines() if s.strip().startswith(line)]
        if len(matches) != 1:
            raise ValueError('Pinned SMMILe MRF adjacency statement changed')
        indent = len(matches[0])-len(matches[0].lstrip())
        patches.append((line, line+'\n'+' '*indent+'if len(adj_sp_scores) == 0:\n'+' '*(indent+4)+'continue'))
        # No empty-candidate or top-k change: the source already guards len(targets).
    names = ['Attn_Net', 'Attn_Net_Gated', 'IAMIL', 'SMMILe']
    if branch == 'single':
        names.append('SMMILe_SINGLE')
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'randint': random.randint,
             'sample': random.sample, 'random': random.random,
             'initialize_weights': init, 'stable_span': stable_span}
    module = load_definitions('smmile', relative, names, scope, patches)
    cls = module.SMMILe_SINGLE if variant == 'binary' else module.SMMILe
    args = {'size_arg': size, 'dropout': dropout, 'drop_rate': dropout_rate,
            'fea_dim': dimension, 'n_refs': refs, 'n_classes': classes}
    if variant == 'multilabel':
        args['multi_label'] = True
    network = cls(**args)
    network.milplus_variant = variant
    network.milplus_repairs = repair
    return network


def weak_supervision_loss(result, target, variant, include_refinement=True, augmented_loss=None):
    """Source BCE probability objectives; no instance annotations are accepted.

    The base probability uses BCE; augmented bags may use the author's bibce.
    Targets are one-hot/multi-hot for multiclass/multilabel, length-one for binary.
    """
    if variant not in VARIANTS:
        raise ValueError('Unknown SMMILe loss variant')
    probabilities, extras = result[1], result[4]
    if not probabilities:
        raise ValueError('SMMILe must return its basic bag probability')
    target = target.to(probabilities[0])
    if target.shape != probabilities[0].shape or not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('SMMILe requires aligned binary-valued supervision')
    if variant == 'multiclass' and target.sum() != 1:
        raise ValueError('SMMILe multiclass supervision must be one-hot')
    augmentation = F.binary_cross_entropy if augmented_loss is None else augmented_loss
    if variant == 'binary':
        loss = F.binary_cross_entropy(probabilities[0], target)
        for p in probabilities[1:]:
            loss = loss + augmentation(p, target)/len(probabilities[1:])
    else:
        loss = F.binary_cross_entropy(probabilities[0], target)/len(probabilities)
        for p in probabilities[1:]:
            loss = loss + augmentation(p, target)/len(probabilities)
    for name in ['instance_loss', 'mrf_loss', 'consist_loss']:
        if name == 'instance_loss' and not include_refinement:
            continue
        value = extras.get(name, 0)
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError('Nonfinite SMMILe auxiliary loss: '+name)
        if value > 0:
            weight = .1 if name == 'consist_loss' and variant == 'multilabel' else 1.
            loss = loss + weight*value
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite SMMILe objective')
    return loss


def predict_instances(network, features, layout, refinement=True):
    """Deterministic label-free numerical prediction in original input row order.

    Call in eval mode. This returns native bag scores; class-dependent detector
    softmax means multiclass scores need not sum to one. No reporting softmax is
    silently applied. Instance attention products are not class probabilities.
    """
    if network.training:
        raise ValueError('SMMILe deterministic prediction requires eval mode')
    variant = network.milplus_variant
    dense = layout.scatter(features)
    h = network.conv_nic(dense.unsqueeze(0)).squeeze(0)
    ix, iy = np.where(layout.mask == 1)
    ix, iy = torch.as_tensor(ix, device=h.device), torch.as_tensor(iy, device=h.device)
    hidden = h[:, ix, iy].T
    detection, _ = network.det_net(hidden)
    classifier = network.cls_net(hidden)
    classification = classifier.softmax(1) if variant == 'multiclass' else classifier.sigmoid()
    attention = detection.softmax(0)
    instances = classification*attention
    bag = instances.sum(0).clamp(1e-10, 1-1e-10)
    heads = [head(hidden).softmax(1) for head in network.ref_net] if refinement else []
    if not torch.isfinite(bag).all() or any(not torch.isfinite(x).all() for x in [instances, *heads]):
        raise ValueError('Nonfinite SMMILe prediction')
    return {'bag_scores': bag, 'instance_attention_product': layout.restore_rows(instances),
            'instance_class_scores': layout.restore_rows(classification),
            'attention': layout.restore_rows(attention),
            'refined_scores': [layout.restore_rows(head) for head in heads]}

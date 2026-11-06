"""Pinned SAM-MIL attention backbone and explicitly enumerated source repairs.

The released train loop and its gradient paths are retained as a named control.
The repaired route keeps all objectives but corrects the documented indexing
and student/teacher wiring. Neither route claims historical paper scores.
"""
import copy
import math

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, verified_source

COMMIT = 'ccff640825321a12658daa0a85c03390bc9d367b'
PROFILES = {'sammil_attn_source_control', 'sammil_attn_objective_repaired'}


def definitions(repaired=False):
    _, commit = verified_source('sammil')
    if commit != COMMIT:
        raise ValueError('SAM-MIL requires the audited author commit')
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math,
                     repeat=repeat, rearrange=rearrange, copy=copy)
    attention = load_definitions('sammil', 'modules/datten.py',
        ['Attention', 'AttentionGated', 'DAttention', 'Backbone'], namespace)
    namespace.update({name: getattr(attention, name) for name in
                      ['Attention', 'AttentionGated', 'DAttention', 'Backbone']})
    patches = []
    if repaired:
        patches = [
            ('torch.where(is_group_feat == 0)[0]', 'torch.where(is_group_feat[0] == 0)[0]'),
            ('torch.where(is_group_feat == 1)[0]', 'torch.where(is_group_feat[0] == 1)[0]'),
            ('attn = attn[:, non_group_indices]',
             'attn = attn[:, non_group_indices]\n            relative_area = relative_area[:, non_group_indices]'),
            ('mask_ids = torch.cat([group_indices.unsqueeze(0), mask_ids], dim=1)',
             'mask_ids = torch.cat([group_indices.unsqueeze(0), non_group_indices[mask_ids]], dim=1)')]
    model = load_definitions('sammil', 'modules/sam.py',
        ['get_cam_1d', 'get_cam_1d_trans', 'initialize_weights',
         'SoftTargetCrossEntropy_v2', 'MCA', 'sam_mil'], namespace, replacements=patches)
    utils = load_definitions('sammil', 'utils.py',
        ['cosine_scheduler', 'calculate_consistency_loss', 'ema_update'], namespace)
    return model, utils


def build_model(dim, classes, repaired=False):
    if type(dim) is not int or dim < 1 or type(classes) is not int or classes < 2:
        raise ValueError('SAM-MIL needs positive native feature width and explicit classes')
    if not repaired and classes != 2:
        raise ValueError('SAM-MIL released evaluation is binary; C-way belongs to the named repaired route')
    source, _ = definitions(repaired)
    # Literal README attention route. No width, branch or objective reduction.
    return source.sam_mil(input_dim=dim, mlp_dim=512, n_classes=classes,
        baseline='attn', dropout=.25, act='relu', da_act='relu', head=8,
        temp_t=.1, temp_s=1., mask_ratio=.9, select_mask=True,
        mask_non_group_feat=True, mask_by_seg_area=True,
        mask_ratio_h=0., mask_ratio_l=0., mask_ratio_hr=1.,
        sigmoid_k=.0005, sigmoid_A0=5000)


def full_step(student, teacher, bag, target, flags, areas, *, repaired,
              groups=5, cls_alpha=1., cl_alpha=0., group_alpha=.5,
              consistency_alpha=1000., consistency_batch=2048, step=0, utility=None):
    """Exact released calls/order, plus the separately named objective repair.

    No backward/step here: callers can compare every gradient with the author
    train_loop and can retain the source's accumulated, unoptimized teacher
    gradients. The consistency kernel itself is always the original function.
    """
    if (bag.ndim != 3 or bag.shape[0] != 1 or bag.shape[1] < groups
            or flags.shape != bag.shape[:2] or areas.shape != bag.shape[:2]
            or target.shape != (1,) or groups < 2
            or min(cls_alpha, group_alpha, consistency_alpha) <= 0 or cl_alpha < 0):
        raise ValueError('SAM-MIL requires all global/group/consistency objectives and complete aligned single-slide inputs')
    if utility is None:
        _, utility = definitions(repaired)
    teacher_feature, attention, _ = teacher.forward_teacher(bag, return_attn=True, label=target)
    teacher_feature = None if cl_alpha == 0. else teacher_feature
    global_logits, global_distill, patch_count, kept = student(
        bag, attention, teacher_feature, i=step, is_group_feat=flags, relative_area=areas)
    indices = torch.randperm(bag.shape[1])
    shuffled = bag[:, indices, :]
    size = bag.shape[1] // groups
    parts = [shuffled[:, i*size:(i+1)*size, :] for i in range(groups)]
    if bag.shape[1] % groups:
        parts[-1] = torch.cat((parts[-1], shuffled[:, groups*size:, :]), dim=1)
    group_logits, group_distill = [], []
    for part in parts:
        target_feature, attention, _ = teacher.forward_teacher(part, return_attn=True, label=target)
        logits, distill, _, _ = student(part, attention,
            None if cl_alpha == 0. else target_feature, i=step)
        group_logits.append(logits); group_distill.append(distill)
    group_losses = [F.cross_entropy(logits.view(1, -1), target) for logits in group_logits]
    bag_loss = sum(group_losses) / len(group_losses)
    # Source overwrites train_logits/cls_loss in its group loop. Its main CE
    # therefore selects the last group, and consistency gradients go only to
    # the never-optimized, never-EMA-updated teacher. Keep both facts visible.
    main_loss = (F.cross_entropy(global_logits.view(1, -1), target)
                 if repaired else group_losses[-1])
    distill_loss = global_distill if repaired else group_distill[-1]
    attention = (student if repaired else teacher).compute_attn_with_grad(bag, label=target)
    consistency = utility.calculate_consistency_loss(attention, areas, consistency_batch)
    loss = cls_alpha*main_loss + cl_alpha*distill_loss + group_alpha*bag_loss + consistency_alpha*consistency
    return {'loss': loss, 'global_ce': main_loss, 'global_forward_ce': F.cross_entropy(global_logits, target),
        'distillation': distill_loss, 'group_ce': bag_loss, 'consistency': consistency,
        'patch_count': patch_count, 'kept': kept, 'group_sizes': [p.shape[1] for p in parts],
        'group_logits': group_logits, 'global_logits': global_logits, 'shuffle': indices}

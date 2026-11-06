"""Full fixed-source AHDMIL experts and asymmetric joint objectives."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hdmil_source import RECIPES as PRETRAIN_RECIPES
from .mammoth import BinarySmoothTop1
from .upstream import load_definitions, verified_source

COMMIT = '2f38354386165167815eb2998424ad63792a65bf'
RECIPES = {
    'ahdmil_camelyon16_source_repaired': dict(task='camelyon16', attn_ratio=.4, LIPN_ema=.2),
    'ahdmil_nsclc_source_repaired': dict(task='nsclc', attn_ratio=.1, LIPN_ema=0.),
    'ahdmil_brca_source_repaired': dict(task='brca', attn_ratio=0., LIPN_ema=0.),
    'ahdmil_rcc_source_repaired': dict(task='rcc', attn_ratio=0., LIPN_ema=0.),
}
for _recipe in RECIPES.values():
    _recipe['pretrain_profile'] = 'hdmil_'+_recipe['task']+'_source'
    _recipe.update(PRETRAIN_RECIPES[_recipe['pretrain_profile']])


def arguments(profile, dimension):
    if profile not in RECIPES or type(dimension) is not int or dimension < 1:
        raise ValueError('AHDMIL requires an explicit full task recipe and feature dimension')
    recipe = RECIPES[profile]
    return SimpleNamespace(**recipe, feature_dim=dimension, n_classes=recipe['classes'],
        init_type='xaiver', lwc='mbv4t', distill_loss='l1', num_moe=2,
        LIPN_lr=1e-5, DMIN_lr=1e-5, wd=1e-5)


def definitions():
    if verified_source('ahdmil')[1] != COMMIT:
        raise ValueError('AHDMIL source differs from the reviewed full joint implementation')
    scope = dict(torch=torch, nn=nn, F=F, np=np, math=math, copy=copy)
    mobile = load_definitions('ahdmil', 'models/mobilenetv4.py',
        ['make_divisible', 'ConvBN', 'UniversalInvertedBottleneck', 'MobileNetV4', 'mobilenetv4_conv_tiny'], scope)
    scope['mobilenetv4_conv_tiny'] = mobile.mobilenetv4_conv_tiny
    return load_definitions('ahdmil', 'models/hdmil.py',
        ['initialize_weights', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'CLAM_MB',
         'ChebyKANLayer', 'ChebyKAN', '_gumbel_sigmoid', 'KAN_CLAM_MB_v5', 'ImageProcessor_MoE'], scope)


def build_joint(profile, dimension):
    args = arguments(profile, dimension); source = definitions()
    dmin = source.KAN_CLAM_MB_v5(I=dimension, dropout=True, n_classes=args.n_classes,
        subtyping=args.subtyping, instance_loss_fn=BinarySmoothTop1(), k_sample=args.k_sample, args=args)
    return dmin, source.ImageProcessor_MoE(args)


def lipn_objective(dmin, lipn, features, images, attn_ratio):
    """Literal Joint.HADMIL.train_LIPN, preserving NumPy draw and all classes."""
    if attn_ratio == 1.:
        manner = 'attn_train'
    elif attn_ratio == 0.:
        manner = 'hard_train'
    else:
        manner = 'attn_train' if np.random.uniform() <= attn_ratio else 'hard_train'
    with torch.no_grad():
        if manner == 'attn_train':
            target, mask = dmin(features, only_attn=True)
        else:
            _, mask = dmin(features); target = mask
    # Explicit sequential additions reproduce the binary and RCC author paths.
    added = mask[:, :, 0]+mask[:, :, 1]
    if mask.shape[-1] == 3:
        added = added+mask[:, :, 2]
    mask_rate = (added != 0).sum()/(mask.shape[0]*mask.shape[1])
    output = lipn(images, phase=manner)
    raw, discrete = output if manner == 'attn_train' else (output, output)
    total, avg_rate, avg_l1, avg_rate_loss = 0, 0, 0, 0
    for scores, binary in zip(raw, discrete):
        added = binary[:, :, 0]+binary[:, :, 1]
        if binary.shape[-1] == 3:
            added = added+binary[:, :, 2]
        union = torch.zeros_like(added, memory_format=torch.legacy_contiguous_format).masked_fill(added != 0, 1.)
        rate = (union-added.detach()+added).sum()/(binary.shape[0]*binary.shape[1])
        l1, rate_loss = F.l1_loss(scores, target), F.mse_loss(rate, mask_rate.detach())
        total, avg_rate = total+(l1+rate_loss), avg_rate+rate
        avg_l1, avg_rate_loss = avg_l1+l1, avg_rate_loss+rate_loss
    n = len(discrete)
    return total/n, dict(distillation=avg_l1/n, retention=avg_rate_loss/n,
                        mask_rate=mask_rate, score_rate=avg_rate/n), manner


def dmin_objective(dmin, lipn, features, images, labels):
    # LIPN remains in train mode: its batchnorm buffers update on this second
    # per-bag forward even though no gradient is requested through the experts.
    with torch.no_grad():
        scores = lipn(images, phase='infer')
    return F.cross_entropy(dmin(features, scores)[0], labels)


def inference(dmin, lipn, features, images):
    scores = lipn(images, phase='infer')
    union = scores[:, :, 0]+scores[:, :, 1]
    if scores.shape[-1] == 3:
        union = union+scores[:, :, 2]
    rate = (union > 0).sum()/(scores.shape[0]*scores.shape[1])
    return dmin(features, scores)[0], rate

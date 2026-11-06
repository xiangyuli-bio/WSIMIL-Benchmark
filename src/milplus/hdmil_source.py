"""Pinned HDMIL definitions and its two released training objectives.

Author code is loaded from verified Git blobs, never redistributed or replaced
with an attention-only substitute. The historical two-class union and global
training attention normalization are deliberately preserved (also for RCC).
"""
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mammoth import BinarySmoothTop1
from .upstream import load_definitions, verified_source

COMMIT = 'ef3d4681054970208e94f8205bd96b3180c00926'
RECIPES = {
    'hdmil_camelyon16_source': dict(degree=12, mask_ratio=.6, subtyping=False, k_sample=32, lr=3e-4, classes=2),
    'hdmil_nsclc_source': dict(degree=16, mask_ratio=.7, subtyping=True, k_sample=32, lr=3e-5, classes=2),
    'hdmil_brca_source': dict(degree=12, mask_ratio=.7, subtyping=True, k_sample=32, lr=3e-4, classes=2),
    'hdmil_rcc_source': dict(degree=12, mask_ratio=.6, subtyping=True, k_sample=8, lr=3e-4, classes=3),
}


def definitions():
    _, commit = verified_source('hdmil')
    if commit != COMMIT:
        raise ValueError('HDMIL source differs from the reviewed complete two-stage release')
    scope = dict(torch=torch, nn=nn, F=F, np=np, math=math)
    mobile = load_definitions('hdmil', 'models/mobilenetv4.py',
        ['make_divisible', 'ConvBN', 'UniversalInvertedBottleneck', 'MobileNetV4', 'mobilenetv4_conv_tiny'], scope)
    scope['mobilenetv4_conv_tiny'] = mobile.mobilenetv4_conv_tiny
    return load_definitions('hdmil', 'models/hdmil.py',
        ['initialize_weights', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'CLAM_MB',
         'ChebyKANLayer', 'ChebyKAN', '_gumbel_sigmoid', 'KAN_CLAM_MB_v4', 'KAN_CLAM_MB_v5', 'ImageProcessor'], scope)


def arguments(profile, dim):
    if profile not in RECIPES or type(dim) is not int or dim < 1:
        raise ValueError('HDMIL requires an explicit released task recipe and positive feature dimension')
    recipe = RECIPES[profile]
    return SimpleNamespace(**recipe, feature_dim=dim, n_classes=recipe['classes'],
        init_type='xaiver', lwc='mbv4t', distill_loss='l1', use_random_inst='False')


def build_dmin(profile, dim, stage='dmin'):
    args = arguments(profile, dim)
    if stage not in {'dmin', 'lipn'}:
        raise ValueError('HDMIL stage must be dmin or lipn')
    source = definitions()
    factory = source.KAN_CLAM_MB_v4 if stage == 'dmin' else source.KAN_CLAM_MB_v5
    return factory(I=dim, dropout=True, n_classes=args.n_classes, subtyping=args.subtyping,
                   instance_loss_fn=BinarySmoothTop1(), k_sample=args.k_sample, args=args)


def build_lipn(profile, dim):
    return definitions().ImageProcessor(arguments(profile, dim))


def dmin_loss(model, features, labels, mask_ratio):
    """Literal DMINMIL.train_inference, with explicit formerly implicit dim=1."""
    logits, _, result, representation, selected_logits, selected, rate = model(
        features, labels, instance_eval=True)
    ce = .7*F.cross_entropy(logits, labels)+.3*result['instance_loss'].mean()
    kl = F.kl_div(F.log_softmax(selected_logits, dim=1), F.log_softmax(logits, dim=1),
                  reduction='batchmean', log_target=True)
    distance = F.mse_loss(selected, representation)
    retention = F.mse_loss(rate, rate.new_tensor([mask_ratio]))
    return ce+.5*kl+.5*distance+2*retention, dict(classification=ce, kl=kl,
        representation=distance, retention=retention, mask_rate=rate)


def lipn_loss(teacher, processor, features, images):
    """Literal LIPNMIL.train_inference: frozen TRAIN-mode teacher, L1 + rate MSE."""
    with torch.no_grad():
        _, mask = teacher(features)
    mask_add = mask[:, :, 0]+mask[:, :, 1]
    rate = (mask_add != 0).sum()/(mask.shape[0]*mask.shape[1])
    scores = processor(images)
    added = scores[:, :, 0]+scores[:, :, 1]
    union = torch.zeros_like(added, memory_format=torch.legacy_contiguous_format).masked_fill(added != 0, 1.)
    score_rate = (union-added.detach()+added).sum()/(mask.shape[0]*mask.shape[1])
    l1, rate_loss = F.l1_loss(scores, mask), F.mse_loss(score_rate, rate.detach())
    return l1+rate_loss, dict(distillation=l1, retention=rate_loss, mask_rate=rate, score_rate=score_rate)

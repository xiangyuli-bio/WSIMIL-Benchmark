"""Original full C2C network/loss/FAISS operator with historical BN semantics."""
import copy
import random

import faiss
import numpy as np
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.nn import functional as F

from .c2c_assets import backbone_factory, components
from .upstream import load_definitions, verified_source

COMMIT = 'd53b6350372117bf3e5251044a856e0251d810b2'
PROFILES = {'c2c_cluster_source', 'c2c_cluster_default_bn_source', 'c2c_cluster_multiclass_repaired'}


def definitions(assets):
    _, commit = verified_source('c2c')
    if commit != COMMIT:
        raise ValueError('C2C requires the audited author commit')
    namespace = dict(torch=torch, nn=nn, F=F, np=np, copy=copy)
    network = load_definitions('c2c', 'C2C/models/resnet.py',
        ['WSIClassifier', 'Enc', 'PatchClassifier', 'EncAttn'],
        {**namespace, 'models': backbone_factory(assets)})
    losses = load_definitions('c2c', 'C2C/loss.py', ['KLDLoss'], namespace)
    clustering = load_definitions('c2c', 'C2C/cluster.py', ['cluster_representation'],
        {**namespace, 'faiss': faiss, 'normalize': normalize}, replacements=[
            ('return label, label_metric, path_list',
             'return label, label_metric, path_list, kmeans.centroids.copy()')])
    predict = load_definitions('c2c', 'C2C/eval_model.py', ['compute_attn_df'],
        {**namespace, 'EncAttn': network.EncAttn}, replacements=[
            ('sample[0].cuda()', 'sample[0].to(next(model.parameters()).device)'),
            ('return torch.max(Y_prob, 1)[1].item(), attn_rep.cpu().numpy().flatten()',
             'return Y_prob, attn_rep.cpu().numpy().flatten()')])
    return network, losses.KLDLoss(), clustering.cluster_representation, predict.compute_attn_df


def build_model(assets, classes, profile):
    if profile not in PROFILES or type(classes) is not int or classes < 2:
        raise ValueError('C2C needs explicit classes and a complete cluster-route profile')
    if profile != 'c2c_cluster_multiclass_repaired' and classes != 2:
        raise ValueError('C2C original weak patch labels are binary; use the explicitly named C-way repair')
    source, criterion, clustering, predict = definitions(assets)
    # Notebook passes True: author sets track_running_stats=False on existing
    # pretrained BN modules. Default constructor False instead resets stats and
    # uses momentum.9. Both operate through the exact historical BN forward.
    model = source.WSIClassifier(classes, bn_track_running_stats=profile != 'c2c_cluster_default_bn_source')
    return model, source, criterion, clustering, predict


class ToTensor:
    """Literal fixed Compose([ToTensor()]) pipeline, including its RNG draw."""
    def __init__(self, assets):
        _, _, self.convert = components(assets)

    def __call__(self, image):
        # Albumentations0.4.6 Compose consumes one Python random draw even at
        # p=1. Its ToTensor override adds no further draw. This affects the
        # subsequent Python-shuffled patch selection, so cannot be omitted.
        random.random()
        return self.convert(image, normalize=None)


def objective(logits, patch_logits, attention, target, clusters, criterion, repaired=False):
    if target.numel() != 1 or target.dtype != torch.long:
        raise ValueError('C2C released bag batch size is one')
    if not repaired and int(target.item()) not in {0, 1}:
        raise ValueError('Source C2C weak patch CE must retain binary0/1 labels')
    patch_target = target.expand(len(patch_logits))
    bag_ce = F.cross_entropy(logits, target)
    patch_ce = F.cross_entropy(patch_logits, patch_target)
    # The source loss consumes UNNORMALIZED attention, and skips <=4-member
    # clusters. Preserve that exact threshold and its original KL reduction.
    kl = criterion(attention, np.asarray(clusters))
    return {'loss': bag_ce+.01*patch_ce+.01*kl, 'bag_ce': bag_ce,
            'patch_ce': patch_ce, 'cluster_kl': kl}

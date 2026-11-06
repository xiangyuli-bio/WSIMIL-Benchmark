"""Full original ResNet34/FPN/bag-CNN and center operators, privately loaded."""
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .center_embeddings_assets import verify
from .upstream import load_definitions, verified_source

COMMIT = '871b1ecad3f0480c526536b11fafe4d759273a25'
PROFILES = {'center_embeddings_bag_classifier_repaired', 'center_embeddings_soft_assignment_repaired'}


@lru_cache(maxsize=1)
def definitions():
    _, sha = verified_source('center_embeddings')
    if sha != COMMIT:
        raise ValueError('Center Embeddings requires its reviewed original author commit')
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    backbone = load_definitions('center_embeddings', 'research_mil/models/backbones/resnetv2.py',
        ['BasicBlock', 'Bottleneck', 'ResNet'], namespace)
    pooling = load_definitions('center_embeddings', 'research_mil/models/pooling/methods.py',
        ['BagClassifier', 'bagAverage', 'BagNetwork'], namespace)
    center = load_definitions('center_embeddings', 'research_mil/trainers/globaltaskmil_trainer.py',
        ['CenterLoss'], namespace, [('classes = classes.to(self.device)', 'classes = classes.to(self.centers.device)')])
    selector = load_definitions('center_embeddings', 'research_mil/utils/mil.py',
        ['group_argtopk', 'group_max', 'calc_err'], {'np': np})
    return backbone, pooling, center, selector


def build_model(root):
    root = verify(root)
    source, pooling, center, _ = definitions()
    features = source.ResNet(source.BasicBlock, [3, 4, 6, 3])
    weights = torch.load(root/'resnet34-333f7ec4.pth', map_location='cpu', weights_only=True)
    result = features.load_state_dict(weights, strict=False)
    if result.missing_keys or set(result.unexpected_keys) != {'fc.weight', 'fc.bias'}:
        raise ValueError('Center Embeddings full original ImageNet ResNet34 weights no longer match')
    model = pooling.BagNetwork(features, pooling.bagAverage(512, 2, 512), classes=2, embed_size=512, k=32)
    center_model = center.CenterLoss(num_classes=2, feat_dim=512, device='cpu')
    return model, center_model


def staircase(param_groups, epoch):
    if type(epoch) is not int or epoch < 0 or len(param_groups) != 2:
        raise ValueError('Center Embeddings requires two source learning-rate groups')
    # Actual source resets to these bases only at epochs10,15,17. In particular
    # its first base differs from cfg.lr, so using a generic MultiStepLR changes
    # the public recipe. Do not alter intervening rates.
    if epoch+1 in [10, 15, 17]:
        factor = .1**([10, 15, 17].index(epoch+1)+1)
        for group, base in zip(param_groups, [.001, .01]):
            group['lr'] = base*factor

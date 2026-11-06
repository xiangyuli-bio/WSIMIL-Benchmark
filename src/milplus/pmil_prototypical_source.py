"""Pinned PMIL (Yu et al., MedIA2023), including its actual patch encoder.

These operators are components of the full raw-image pipeline. They do not
register a feature-only replacement as the complete Prototypical PMIL method.
Author code remains in its separately obtained, immutable source checkout.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file
from .upstream import load_definitions, verified_source

COMMIT = 'f9b241a30ef90235da2c57814408da5eedb368e0'
IMAGENET_SHA256 = '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'


def _check():
    root, commit = verified_source('pmil_prototypical')
    if commit != COMMIT:
        raise ValueError('Use the locked Yu et al. Prototypical PMIL source, not the Shapley PMIL or ProtoMIL')
    return root


@lru_cache(maxsize=1)
def encoder_definitions():
    _check()
    # No import-time example, torchvision download or global CUDA mutation is
    # executed. These are the complete author ResNet definitions, not a tiny
    # substitute or an untrained frozen-feature projection.
    return load_definitions('pmil_prototypical', 'model/model_maxmil.py',
        ['conv3x3', 'conv1x1', 'BasicBlock', 'Bottleneck', 'ResNet'],
        {'torch': torch, 'nn': nn})


def encoder(weights):
    if digest_file(weights) != IMAGENET_SHA256:
        raise ValueError('PMIL requires actual original resnet34-333f7ec4 ImageNet weights')
    source = encoder_definitions()
    model = source.ResNet(source.BasicBlock, [3, 4, 6, 3])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    # Exactly the native max-mil.py construction: load 1000-way ImageNet first,
    # then initialize the binary patch classifier with the caller's Torch RNG.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def bag_definitions(weights):
    _check()
    if digest_file(weights) != IMAGENET_SHA256:
        raise ValueError('PMIL source constructor requires the original ResNet34 initializer')
    source = encoder_definitions()

    def resnet34(pretrained=False, progress=True, **kwargs):
        model = source.ResNet(source.BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        return model

    # The author's model imports resnet34 from the otherwise empty model/
    # package. Supply its own model_maxmil.py implementation explicitly.
    return load_definitions('pmil_prototypical', 'model/model_pmil.py',
        ['IClassifier', 'BClassifier', 'PBMIL', 'Euclidean_Similarity'],
        {'torch': torch, 'nn': nn, 'F': F, 'resnet34': resnet34})


def bag_model(weights, selected_encoder, prototypes):
    if (not torch.is_tensor(prototypes) or prototypes.ndim != 2
            or prototypes.shape[1] != 512 or prototypes.shape[0] < 1
            or prototypes.dtype != torch.float32 or not torch.isfinite(prototypes).all()):
        raise ValueError('Original PMIL requires real finite512D float32 TRAIN-discovered prototypes')
    state = torch.load(selected_encoder, map_location='cpu', weights_only=False)
    # Do not silently accept an unrelated learned patch head. This full source
    # constructor also retains its originally registered (unused here) CNN.
    if set(state) != {'state_dict'}:
        raise ValueError('Pass the explicit selected native max-MIL state_dict checkpoint')
    source = bag_definitions(weights)
    model = source.PBMIL(source.IClassifier(str(selected_encoder), feature_size=512, out_class=2),
        source.BClassifier(len(prototypes), feature_size=512, input_size=128, output_class=2, dropout=0))
    return model


def bag_objective(model, features, prototypes, target):
    if (features.ndim != 2 or features.shape[1] != 512 or len(features) < 5
            or features.dtype != torch.float32 or not torch.isfinite(features).all()
            or target.dtype != torch.long or target.shape != (1,)
            or int(target.item()) not in {0, 1}):
        raise ValueError('PMIL requires one binary bag with at least five actual512D patch features')
    # Literal source train(): eval mode during optimizer updates; do not
    # "correct" this to training dropout or replace the instance CE with ABMIL.
    model.eval()
    instance, bag, coding = model(features, prototypes)
    scores = F.softmax(instance, dim=1)
    _, maximum = torch.max(scores[:, 1], dim=0)
    bag_ce = F.cross_entropy(bag.view(1, -1), target)
    max_ce = F.cross_entropy(instance[maximum, :].view(1, -1), target)
    return {'loss': bag_ce+max_ce, 'bag_ce': bag_ce, 'max_ce': max_ce,
        'bag_logits': bag, 'instance_logits': instance, 'coding': coding, 'maximum_index': maximum}


def bag_step(model, features, prototypes, target, optimizer):
    values = bag_objective(model, features, prototypes, target)
    if not torch.isfinite(values['loss']):
        raise FloatingPointError('Original PMIL dual CE is nonfinite')
    optimizer.zero_grad()
    values['loss'].backward(retain_graph=True)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Original PMIL dual CE gradient is nonfinite')
    optimizer.step()
    return {key: float(values[key].detach()) for key in ['loss', 'bag_ce', 'max_ce']}


@lru_cache(maxsize=1)
def selection_definitions():
    _check()
    # The published max-mil.py has a real IndentationError in a feature-export
    # block. Whitespace-only repair permits parsing its unaffected definitions.
    # The unsafe main() (including validation optimizer updates) is never run.
    return load_definitions('pmil_prototypical', 'max-mil.py', ['group_argtopk', 'group_max'],
        {'np': np}, replacements=[(
            '                    for i, slide in enumerate(train_dset.slidenames):\n'
            '                    slidename =',
            '                    for i, slide in enumerate(train_dset.slidenames):\n'
            '                        slidename =')])


def select_indices(scores, count):
    values = np.asarray(scores)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or type(count) is not int or count < 1):
        raise ValueError('PMIL source selection requires finite one-slide probabilities and positive count')
    return selection_definitions().group_argtopk(np.zeros(len(values), dtype=np.int64), values, count)


def patch_tensor(pixels):
    # Exact original ToTensor + Normalize(mean=.5, std=.1), without resize,
    # ImageNet mean/std, FM preprocessing or synthesized RGB from features.
    from PIL import Image
    from torchvision import transforms
    a = np.asarray(pixels)
    if a.shape != (256, 256, 3) or a.dtype != np.uint8:
        raise ValueError('Original PMIL needs actual uint8 RGB256 patch pixels')
    return transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.1, .1, .1])])(Image.fromarray(a))

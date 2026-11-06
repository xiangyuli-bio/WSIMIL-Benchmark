"""Exact historical C2C backbone/preprocessing assets; no implicit substitutes."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file

ASSETS = {
    'resnet18-5c106cde.pth': ('https://download.pytorch.org/models/resnet18-5c106cde.pth',
        '5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8'),
    'torchvision_v061_resnet.py': ('https://raw.githubusercontent.com/pytorch/vision/v0.6.1/torchvision/models/resnet.py',
        '8c27d6ea12e675320481292d67479179718df406e3442365af369be5d3852019'),
    'pytorch_v151_batchnorm.py': ('https://raw.githubusercontent.com/pytorch/pytorch/v1.5.1/torch/nn/modules/batchnorm.py',
        'e83afbb78231916297e30beb3ebd51a0885ddb509b1a041213b6cb0f91498d7c'),
    'albumentations_v046_transforms.py': ('https://raw.githubusercontent.com/albumentations-team/albumentations/0.4.6/albumentations/pytorch/transforms.py',
        '3e8e1de31900df786d0e59e4ebef05f40c5c004f2ff3d8dc42b4be5d53dc0c47'),
    'albumentations_v046_composition.py': ('https://raw.githubusercontent.com/albumentations-team/albumentations/0.4.6/albumentations/core/composition.py',
        'b6328154f988b12b8c6c350a648b66b4508b50f9722057338b268a42d75d9bba'),
}


def default_root():
    return Path(os.environ.get('MILPLUS_C2C_ASSETS', Path.home()/'.cache/milplus/assets/c2c'))


def verify(root):
    root = Path(root)
    for name, (_, sha256) in ASSETS.items():
        if not (root/name).is_file() or digest_file(root/name) != sha256:
            raise ValueError(f'C2C requires exact historical pretrained/source asset {name}; run setup on a networked CPU host')
    return root


def fetch(root=None):
    root = Path(root or default_root()); root.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in ASSETS.items():
        path = root/name
        if path.exists():
            if digest_file(path) != sha256:
                raise ValueError(f'C2C existing asset changed: {path}; preserve it and choose a new asset root')
            continue
        temporary = path.with_suffix(path.suffix+'.download')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as output:
            for chunk in iter(lambda: response.read(8*1024**2), b''):
                output.write(chunk)
        if digest_file(temporary) != sha256:
            raise ValueError(f'C2C asset checksum mismatch: {temporary}')
        temporary.replace(path)
    return verify(root)


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('Historical C2C support source definitions changed')
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def components(root):
    root = verify(root)
    # Changing track_running_stats on an already constructed BatchNorm has
    # different semantics in PyTorch1.5 and2.5. Execute the exact old forward;
    # merely setting that flag on a modern BN silently changes C2C inference.
    tree = ast.parse((root/'pytorch_v151_batchnorm.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_BatchNorm')
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    forward.name = '_source_forward'
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(root/'pytorch_v151_batchnorm.py'), 'exec'), namespace)
    LegacyBatchNorm2d = type('C2CTorch151BatchNorm2d', (nn.BatchNorm2d,), {'forward': namespace['_source_forward']})
    vision = _definitions(root/'torchvision_v061_resnet.py',
        ['conv3x3', 'conv1x1', 'BasicBlock', 'Bottleneck', 'ResNet'], {'torch': torch, 'nn': nn})
    transforms = _definitions(root/'albumentations_v046_transforms.py', ['img_to_tensor'],
                             {'torch': torch, 'np': np})
    return LegacyBatchNorm2d, vision, transforms.img_to_tensor


def backbone_factory(root):
    root = verify(root)
    batchnorm, vision, _ = components(root)

    def resnet18(pretrained=True):
        if pretrained is not True:
            raise ValueError('Complete source C2C requires its actual original ImageNet initialization')
        network = vision.ResNet(vision.BasicBlock, [2, 2, 2, 2], norm_layer=batchnorm)
        network.load_state_dict(torch.load(root/'resnet18-5c106cde.pth', map_location='cpu', weights_only=True), strict=True)
        return network
    return SimpleNamespace(resnet18=resnet18)

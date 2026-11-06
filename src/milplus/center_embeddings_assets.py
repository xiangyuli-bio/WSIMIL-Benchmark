"""Exact original ImageNet initialization and torchvision0.7 RGB transforms."""
import ast
from collections.abc import Iterable
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np
from PIL import Image
import torch

from .io import digest_file

ASSETS = {
    'resnet34-333f7ec4.pth': ('https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'),
    'torchvision_v070_transforms.py': ('https://raw.githubusercontent.com/pytorch/vision/v0.7.0/torchvision/transforms/transforms.py',
        '98745994aa5e8d96ab80a2c0ba345ac6d63f061ab9e4b6db91893c11d81162f8'),
    'torchvision_v070_functional.py': ('https://raw.githubusercontent.com/pytorch/vision/v0.7.0/torchvision/transforms/functional.py',
        '21cd3a4c8b5620ed79106b2f9333114a00b14677bd89a124adf2f8f8ef354827'),
    'torchvision_v070_functional_pil.py': ('https://raw.githubusercontent.com/pytorch/vision/v0.7.0/torchvision/transforms/functional_pil.py',
        'e19337cdfc5ba9df57df4013f595e365562f65ac7e301919397611f072e48663'),
}


def default_root():
    return Path(os.environ.get('MILPLUS_CENTER_EMBEDDINGS_ASSETS',
        Path.home()/'.cache/milplus/assets/center_embeddings'))


def verify(root):
    root = Path(root).expanduser().resolve()
    for name, (_, expected) in ASSETS.items():
        if not (root/name).is_file() or digest_file(root/name) != expected:
            raise ValueError('Center Embeddings requires exact original asset '+name)
    return root


def fetch(root=None):
    root = Path(root or default_root()); root.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        p = root/name
        if p.exists():
            if digest_file(p) != expected:
                raise ValueError('Preserve changed Center Embeddings assets and select a new version')
            continue
        temporary = p.with_suffix(p.suffix+'.download')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as output:
            for chunk in iter(lambda: response.read(4*1024**2), b''):
                output.write(chunk)
        if digest_file(temporary) != expected:
            raise ValueError('Center Embeddings original asset checksum failed')
        temporary.replace(p)
    return verify(root)


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Original torchvision0.7 transform definitions changed')
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return SimpleNamespace(**{n: scope[n] for n in names})


def transforms(root, training):
    root = verify(root)
    common = {'torch': torch, 'Tensor': torch.Tensor, 'np': np, 'Image': Image,
              'accimage': None, 'Iterable': Iterable}
    pil = definitions(root/'torchvision_v070_functional_pil.py', ['_is_pil_image', 'hflip'], common)
    functional = definitions(root/'torchvision_v070_functional.py',
        ['_is_pil_image', '_is_numpy', '_is_numpy_image', 'to_tensor', 'normalize', 'resize', 'hflip'],
        {**common, 'F_pil': pil})
    module = definitions(root/'torchvision_v070_transforms.py',
        ['Compose', 'Resize', 'RandomHorizontalFlip', 'ToTensor', 'Normalize'],
        {**common, 'F': functional})
    # Input is the original already-read256x256 RGB tile. Resize256 is still
    # executed; no interpolation, crop, reduced image or frozen FM substitute.
    sequence = [module.Resize(256)]
    if training:
        sequence.append(module.RandomHorizontalFlip(.5))
    sequence += [module.ToTensor(), module.Normalize([.5]*3, [.1]*3)]
    return module.Compose(sequence)

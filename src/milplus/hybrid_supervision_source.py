"""Full author DLA34/DLA34Up and disclosed paper/source training conditions."""
import sys
import hashlib
import types
from pathlib import Path
from functools import lru_cache

import torch
from torch import nn

from .upstream import verified_source, package, pinned_text, load_definitions
from .io import digest_file

COMMIT = '1f54478812cd057ad3ddc70543bed503ba39f800'
WEIGHT_NAME = 'dla34-ba72cf86.pth'
WEIGHT_SHA384 = '38c035736538b6c67f2539bf1ca1cc1349c3691f41321bd58de9e97aa9eb83ea6f52c4aeef7c4cd54c92b6f8c3f3e89b'


def check_source():
    root, sha = verified_source('hybrid_supervision')
    if sha != COMMIT:
        raise ValueError('Use the locked original Hybrid Supervision Learning source')
    return root, sha


@lru_cache(maxsize=1)
def modules():
    root, sha = check_source()
    name = '_milplus_hybrid_'+sha
    parent = package(name, root)
    package(name+'.sync_batchnorm', root/'sync_batchnorm')
    for suffix, relative in [
        ('dataset', 'dataset.py'),
        ('sync_batchnorm.comm', 'sync_batchnorm/comm.py'),
        ('sync_batchnorm.replicate', 'sync_batchnorm/replicate.py'),
        ('sync_batchnorm.batchnorm', 'sync_batchnorm/batchnorm.py'),
        ('sync_batchnorm', 'sync_batchnorm/__init__.py'),
        ('dla_bn', 'dla_bn.py'), ('dla_up_bn', 'dla_up_bn.py')]:
        text = pinned_text(root, sha, relative)
        text = text.replace('import dataset\n', 'from . import dataset\n')
        text = text.replace('from sync_batchnorm import ', 'from .sync_batchnorm import ')
        text = text.replace('import dla_bn as dla', 'from . import dla_bn as dla')
        module = types.ModuleType(name+'.'+suffix)
        module.__file__ = str(root/relative)
        module.__package__ = name+'.sync_batchnorm' if suffix.startswith('sync_batchnorm') else name
        if suffix == 'sync_batchnorm':
            module.__path__ = [str(root/'sync_batchnorm')]
        sys.modules[module.__name__] = module
        exec(compile(text, module.__file__, 'exec'), module.__dict__)
        if '.' not in suffix:
            setattr(parent, suffix, module)
    return parent


def validate_weights(path):
    path = Path(path)
    sha = digest_file(path)
    h=hashlib.sha384()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    if not sha.startswith('ba72cf86') or h.hexdigest()!=WEIGHT_SHA384 or path.stat().st_size!=63228658:
        raise ValueError('Hybrid supervision requires the actual original DLA34 ImageNet initializer')
    return path, sha


def build_model(weights, role):
    path, sha = validate_weights(weights)
    source = modules()
    # Preserve the author's constructor/init/load order. Supply the published
    # exact checkpoint locally rather than downloading inside GPU training.
    expected_url = 'http://dl.yf.io/dla/models/imagenet/'+WEIGHT_NAME
    def local_load(url, *args, **kwargs):
        if url != expected_url or digest_file(path) != sha:
            raise ValueError('Original hybrid DLA34 initializer identity changed')
        return torch.load(path, map_location='cpu', weights_only=True)
    original = source.dla_bn.model_zoo
    source.dla_bn.model_zoo = types.SimpleNamespace(load_url=local_load)
    try:
        if role == 'segmentation':
            return source.dla_up_bn.dla34up_bn(1, 'imagenet', down_ratio=2)
        if role == 'classification':
            model = source.dla_bn.dla34(pretrained='imagenet')
            model.fc = nn.Conv2d(model.channels[-1], 1, kernel_size=1, bias=True)
            return model
        raise ValueError('Choose actual DLA34Up segmentation or DLA34 image classification')
    finally:
        source.dla_bn.model_zoo = original


def source_stage2():
    import numpy as np
    from skimage import measure
    check_source()
    return load_definitions('hybrid_supervision', 'Stage2_classifier.py',
        ['feature_extract', '_feature_extract_thre', 'node2patient'],
        {'np': np, 'measure': measure})


@lru_cache(maxsize=1)
def source_transforms():
    import math
    import numbers
    import random
    import collections
    import numpy as np
    import PIL
    from PIL import Image, ImageOps, ImageEnhance
    check_source()
    # np.int was removed after NumPy1.23; use the exact int64 label tensor
    # conversion without changing the source joint RGB/mask transform graph.
    root, sha = check_source()
    name = '_milplus_hybrid_transforms_'+sha
    module = types.ModuleType(name)
    module.__file__ = str(root/'data_transforms.py')
    text = pinned_text(root, sha, 'data_transforms.py').replace('dtype=np.int)', 'dtype=np.int64)')
    exec(compile(text, module.__file__, 'exec'), module.__dict__)
    return module

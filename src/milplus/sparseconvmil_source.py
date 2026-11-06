"""Full original SparseConvMIL and compiled SparseConvNet; no dense substitute."""
import importlib
import math
import random
from types import SimpleNamespace

import torch
from torch import nn

from .io import digest_file, digest_object
from .upstream import load_definitions, pinned_text, verified_source

COMMIT = 'd7dd705bc3dd4ebfb6a36f0589698400c9fa4c15'
SCN_COMMIT = 'cf251d058959a9dbaccb25fc919dc4f4548be232'
PROFILES = {'sparseconvmil_raw_source_c32', 'sparseconvmil_raw_source_c128'}


def backend():
    """Require the actual, independently compiled author C++/autograd backend."""
    root, commit = verified_source('sparseconvnet')
    if commit != SCN_COMMIT:
        raise ValueError('SparseConvMIL requires its recorded SparseConvNet source')
    module = importlib.import_module('sparseconvnet')
    extension = importlib.import_module('sparseconvnet.SCN')
    if not extension.__file__.endswith(('.so', '.pyd')):
        raise ValueError('SparseConvMIL needs the actual compiled sparse backend, not an emulation')
    installed = __import__('pathlib').Path(module.__file__).parent
    files = {}
    for path in sorted((root/'sparseconvnet').glob('*.py')):
        text = pinned_text(root, commit, 'sparseconvnet/'+path.name)
        actual = installed/path.name
        if not actual.is_file() or actual.read_text() != text:
            raise ValueError('SparseConvNet Python/autograd source changed: '+path.name)
        files[path.name] = digest_file(actual)
    return module, {'source_commit': commit, 'python_sha256': digest_object(files),
                    'extension_sha256': digest_file(extension.__file__), 'backend': 'original_compiled_sparseconvnet'}


def definitions():
    _, commit = verified_source('sparseconvmil')
    if commit != COMMIT:
        raise ValueError('SparseConvMIL requires its audited original author commit')
    sparse, identity = backend()
    # The source uses a positional pretrained argument; torchvision's modern
    # API makes it keyword-only. This bridge preserves the complete ResNet18
    # architecture and the released default random initialization, with no
    # guessed/pre-downloaded checkpoint or change to the tile encoder.
    vision = importlib.import_module('torchvision.models.resnet')

    def resnet18(pretrained=False, progress=True):
        if pretrained is not False:
            raise ValueError('This released-source route has no pretrained weights; declare a separate weight-bound protocol')
        return vision.resnet18(weights=None, progress=progress)

    factory = SimpleNamespace(__all__=['resnet18'], resnet18=resnet18)
    source = load_definitions('sparseconvmil', 'model.py',
        ['SparseConvMIL', 'SparseAdaptiveAvgPool', 'get_classifier', 'get_resnet_model',
         'get_two_layers_sparse_cnn', 'instantiate_sparseconvmil'],
        {'torch': torch, 'nn': nn, 'math': math, 'random': random,
         'resnet_factory': factory, 'sparseconvnet': sparse})
    return source, identity


def build_model(classes, profile, downsample=10):
    if profile not in PROFILES or type(classes) is not int or classes < 2:
        raise ValueError('SparseConvMIL needs explicit complete end-to-end source profiles')
    if type(downsample) is not int or downsample < 1:
        raise ValueError('SparseConvMIL spatial downsampling must be a positive integer')
    source, identity = definitions()
    channels = 32 if profile.endswith('_c32') else 128
    model = source.instantiate_sparseconvmil('resnet18', False, channels, channels,
                                            3, 3, downsample, 32, classes)
    return model, identity

"""Load the complete pinned author MIL-Adapter in a private namespace.

The original source has no license file; do not vendor it. Only import paths,
device placement, the undefined ZS temperature and two ABMIL tuple returns are
ported. All seven native aggregators and all actual adaptation equations stay.
"""
import importlib.metadata
import sys
import types
from functools import lru_cache

import numpy as np

from .upstream import package, pinned_text, verified_source

COMMIT = 'ade11bb19c3f09a35da9e8255a87a079217aef48'
AGGREGATORS = ('BGAP', 'BGMP', 'ABMIL', 'TransMIL', 'WIKGMIL', 'ILRAMIL', 'RRTMIL')
CONDITIONS = [('ZSMIL', a, init) for a in AGGREGATORS for init in ('ZS', 'random')]
CONDITIONS += [(route, a, None) for route in ('TaskRes', 'CLIPAdapter', 'TIPAdapter')
               for a in AGGREGATORS[:4]]
PROFILES = {'miladapter_'+r.lower()+'_'+a.lower()+('_'+i.lower() if i else ''): (r, a, i)
            for r, a, i in CONDITIONS}
TEMPERATURE = 56.347694396972656


def source_code(relative, *, repaired=True):
    root, sha = verified_source('miladapter')
    if sha != COMMIT:
        raise ValueError('MIL-Adapter requires the reviewed original author commit')
    code = pinned_text(root, sha, relative)
    # Isolate every source import; neither an unrelated installed utils package
    # nor another original MIL checkout may change the executed definitions.
    prefix = '_milplus_miladapter_'+sha
    code = code.replace('from utils.MIL.', 'from '+prefix+'.MIL.')
    code = code.replace('from utils.MIL import', 'from '+prefix+'.MIL import')
    if repaired:
        if relative == 'utils/MIL/TransMIL.py':
            code = code.replace('cls_tokens = self.cls_token.expand(B, -1, L).cuda()',
                                'cls_tokens = self.cls_token.expand(B, -1, L).to(h.device)')
            code = code.replace(').cuda()', ')')
        if relative == 'utils/adapters.py':
            if code.count(' * logit_scale') != 1 or code.count('embedding = self.ABMIL(features)\n') != 2:
                raise ValueError('Pinned MIL-Adapter declared repairs no longer match')
            code = code.replace(' * logit_scale', ' * self.logit_scale')
            code = code.replace('embedding = self.ABMIL(features)\n',
                                'embedding = self.ABMIL(features)[0]\n')
            old = 'self.c_in, self.reduction, self.ratio = 512, 4, 0.2'
            if code.count(old) != 1:
                raise ValueError('MIL-Adapter CLIP source input-width port no longer matches')
            code = code.replace(old, 'self.c_in, self.reduction, self.ratio = text_embeddings.shape[0], 4, 0.2')
            code = code.replace('.cuda()', '')
    return code


@lru_cache(maxsize=1)
def definitions():
    for name, expected in {'timm': '0.9.12', 'torch-geometric': '2.6.1', 'nystrom-attention': '0.0.14', 'transformers': '4.51.3'}.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('MIL-Adapter requires its audited dependency '+name+'=='+expected)
    root, _ = verified_source('miladapter')
    prefix = '_milplus_miladapter_'+COMMIT
    package(prefix, root/'utils')
    parent = package(prefix+'.MIL', root/'utils/MIL')
    for filename in ('mil_template', 'nystrom_attention', 'ABMIL', 'TransMIL',
                     'WIKGMIL', 'ILRAMIL', 'RRTMIL'):
        relative = 'utils/MIL/'+filename+'.py'
        code = source_code(relative)
        obj = types.ModuleType(prefix+'.MIL.'+filename)
        obj.__file__, obj.__package__ = str(root/relative), prefix+'.MIL'
        sys.modules[obj.__name__] = obj
        exec(compile(code, obj.__file__, 'exec'), obj.__dict__)
        if filename in ('ABMIL', 'TransMIL', 'WIKGMIL'):
            setattr(parent, filename, getattr(obj, filename))
    relative = 'utils/adapters.py'
    obj = types.ModuleType(prefix+'.adapters')
    obj.__file__, obj.__package__ = str(root/relative), prefix
    sys.modules[obj.__name__] = obj
    exec(compile(source_code(relative), obj.__file__, 'exec'), obj.__dict__)
    return obj


def build_model(profile, prototypes, support_features, support_labels):
    if profile not in PROFILES:
        raise ValueError('Unknown or unsupported original MIL-Adapter configuration')
    route, aggregator, init = PROFILES[profile]
    if (not isinstance(prototypes, np.ndarray) or prototypes.dtype != np.float32
            or prototypes.ndim != 2 or prototypes.shape[0] not in {512, 768}
            or prototypes.shape[1] < 2 or not np.isfinite(prototypes).all()
            or not np.all(np.linalg.norm(prototypes, axis=0) > 0)):
        raise ValueError('MIL-Adapter requires actual native512D CONCH/768D KEEP float32 text prototype columns')
    source = definitions()
    kwargs = {'text_embeddings': prototypes, 'aggregator': aggregator}
    if route == 'ZSMIL':
        kwargs['init'] = init
    if route == 'TIPAdapter':
        if (not support_features or len(support_features) != len(support_labels)
                or set(support_labels) != set(range(prototypes.shape[1]))):
            raise ValueError('TIP-Adapter needs original TRAIN-only support bags from every class')
        kwargs.update(train_data=support_features, train_labels=support_labels)
    model = getattr(source, route)(**kwargs)
    # Source plain text tensors are constants, but neither move with .to nor
    # serialize. Register the identical tensors; no values, RNG or formula change.
    value = model.text_embeddings
    del model.text_embeddings
    model.register_buffer('text_embeddings', value)
    return model


def fewshot_indices(labels, k_shots, seed):
    """Author classwise NumPy choice, including its global RNG transition."""
    labels = np.asarray(labels, dtype=np.int64)
    if (type(k_shots) is not int or k_shots not in {2, 4, 8, 16}
            or set(labels) != set(range(len(set(labels))))
            or any(np.sum(labels == c) < k_shots for c in set(labels))):
        raise ValueError('MIL-Adapter needs at least k=2/4/8/16 eligible TRAIN slides in each class')
    np.random.seed(seed)
    ids = np.arange(len(labels)); chosen = []
    for cls in range(len(set(labels))):
        chosen.extend(np.random.choice(ids[labels == cls], size=k_shots, replace=False))
    return [int(i) for i in chosen]

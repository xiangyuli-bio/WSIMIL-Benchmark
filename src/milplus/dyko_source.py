"""Pinned DyKo mechanisms, without claiming a validated executable profile.

The complete native model still requires authorized TITAN assets. This module
loads the author's actual definitions; it never supplies a substitute language
tower or substitutes CONCH-v1 for the released TITAN text branch.
"""
import ast
import math
import types
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file
from .upstream import load_definitions, pinned_text, verified_source

COMMIT = 'a8b43645840cf283c334b07c42cf4ddf230f644a'
NAMES = ['TextEncoder', 'PromptLearner', '_no_grad_trunc_normal_',
         'trunc_normal_', 'encode_text', 'GatedAttention', 'Adapter',
         'ClusterHead', 'DyKo']
COHORTS = {'CAMELYON16': 2, 'NSCLC': 2, 'RCC': 3, 'UBC': 5}


def checked_source():
    root, commit = verified_source('dyko')
    if commit != COMMIT:
        raise ValueError('DyKo requires the reviewed original author revision')
    return root, commit


class FaissBackend:
    """Explicit GPU native route or explicitly named CPU numerical reference.

    FAISS CPU installations can silently accept gpu=True. That behavior must
    not turn a CPU calculation into evidence for the author's CUDA recipe.
    """
    def __init__(self, backend):
        if backend not in {'cuda_faiss', 'cpu_faiss_reference'}:
            raise ValueError('DyKo requires an explicit FAISS execution backend')
        import faiss
        self.backend = backend
        self.faiss = faiss
        self.calls = []
        if backend == 'cuda_faiss' and (
                not torch.cuda.is_available() or not hasattr(faiss, 'StandardGpuResources')
                or faiss.get_num_gpus() < 1):
            raise RuntimeError('DyKo native route requires real CUDA-enabled FAISS; no CPU fallback')

    def Kmeans(self, **kwargs):  # noqa: N802 -- actual author API
        if (set(kwargs) != {'d', 'k', 'niter', 'min_points_per_centroid', 'verbose', 'gpu', 'seed'}
                or kwargs['niter'] != 20 or kwargs['min_points_per_centroid'] != 5
                or kwargs['verbose'] is not False or kwargs['gpu'] is not True or kwargs['seed'] != 42):
            raise ValueError('DyKo original per-forward FAISS clustering recipe changed')
        requested = dict(kwargs)
        if self.backend == 'cpu_faiss_reference':
            kwargs = dict(kwargs, gpu=False)
        self.calls.append({'requested': requested, 'actual_backend': self.backend})
        return self.faiss.Kmeans(**kwargs)


def author_classes(backend='cuda_faiss', titan_loader=None):
    """Actual original architecture with only explicit dependency routing.

    titan_loader is an internal loader interface, not a registered substitute
    model profile. Full model validation must use authorized original weights.
    """
    checked_source()
    faiss = FaissBackend(backend)

    class OfflineTITAN:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code):
            if model_id != 'MahmoodLab/TITAN' or trust_remote_code is not True:
                raise ValueError('DyKo changed its original TITAN model identity')
            if titan_loader is None:
                raise FileNotFoundError('DyKo requires authorized local TITAN weights; no replacement text model')
            return titan_loader()

    module = load_definitions('dyko', 'models/DyKo.py', NAMES,
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'pd': pd, 'math': math,
         'warnings': warnings, 'types': types, 'AutoModel': OfflineTITAN, 'faiss': faiss})
    module.backend = faiss
    return module


def source_assets(cohort):
    root, commit = checked_source()
    cohort = str(cohort).upper()
    if cohort not in COHORTS:
        raise ValueError('DyKo needs an explicit released concept/prompt cohort')
    concepts = root/f'concepts/text_embeddings_{cohort}.pt'
    prompts = root/f'text_prompt/Cluade-3.5-Sonnet/{cohort}_text_prompt.csv'
    # Binary concepts are checked against the Git blob, not just a mutable
    # filename. They are provided by the author; no concepts are fabricated.
    import hashlib
    import subprocess
    original = subprocess.check_output(['git', 'show', f'{commit}:{concepts.relative_to(root)}'], cwd=root)
    if digest_file(concepts) != hashlib.sha256(original).hexdigest():
        raise ValueError('DyKo released concept bank bytes changed')
    if prompts.read_text() != pinned_text(root, commit, str(prompts.relative_to(root))):
        raise ValueError('DyKo full ordered original prompt text changed')
    bank = torch.load(concepts, map_location='cpu', weights_only=True)
    frame = pd.read_csv(prompts)
    if (not isinstance(bank, torch.Tensor) or bank.ndim != 2 or bank.shape[1] != 768
            or not bank.is_floating_point() or not torch.isfinite(bank).all()
            or len(bank) < 10 or 'Prompt' not in frame or frame.Prompt.isna().any()):
        raise ValueError('DyKo original full native768 concept/prompt assets are invalid')
    return {'cohort': cohort, 'classes': COHORTS[cohort], 'concepts': concepts,
            'prompts': prompts, 'bank': bank, 'prompt_text': frame.Prompt.tolist(),
            'concepts_sha256': digest_file(concepts), 'prompts_sha256': digest_file(prompts)}


@lru_cache(maxsize=1)
def classification_training_step():
    """Compile the actual source training_step once, without changing its body."""
    root, commit = checked_source()
    tree = ast.parse(pinned_text(root, commit, 'models/model_interface.py'))
    interface = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelInterfaceCls')
    method = next(n for n in interface.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
    scope = {'F': F, 'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(root/'models/model_interface.py'), 'exec'), scope)
    return scope['training_step']


def classification_loss(results, label):
    """Actual source CE plus unweighted KL, including dim1 instance softmax."""
    step = classification_training_step()

    class Interface:
        model = staticmethod(lambda **kwargs: results)
        loss = nn.CrossEntropyLoss()
        training_step_outputs = None

    interface = Interface()
    interface.training_step_outputs = []
    return step(interface, (None, label), 0)


def source_adam(model, lr=.001, weight_decay=.00001):
    """Original Adam, including the source bias/one-dimensional zero decay."""
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('DyKo needs finite positive LR and nonnegative weight decay')
    root, commit = checked_source()
    function = next(n for n in ast.parse(pinned_text(root, commit, 'MyOptimizer/optim_factory.py')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'add_weight_decay')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root/'MyOptimizer/optim_factory.py'), 'exec'), scope)
    # Match create_optimizer: weight_decay=0 bypasses filtering; otherwise
    # decay is set on the second explicit group, not on optimizer defaults.
    parameters = scope['add_weight_decay'](model, weight_decay) if weight_decay else model.parameters()
    return torch.optim.Adam(parameters, lr=lr, weight_decay=0.)


def original_shot_selection(frame, n_shot, seed):
    """Author per-class pandas sample, retaining the original TRAIN order.

    Validation/test are supplied independently. This function never writes the
    author's reusable shot cache (whose identity does not bind the split CSV).
    """
    if type(n_shot) is not int or n_shot < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('DyKo shot count/seed must be explicit positive/in-range integers')
    columns = ['train_slide_id', 'train_label']
    if list(frame.columns) != columns or frame.isna().any().any() or frame.empty:
        raise ValueError('DyKo requires an explicit complete TRAIN-only table')
    if frame.train_slide_id.astype(str).duplicated().any():
        raise ValueError('DyKo TRAIN slide IDs must be unique')
    labels = pd.to_numeric(frame.train_label, errors='raise')
    if not np.isfinite(labels).all() or (labels % 1 != 0).any() or (labels < 0).any():
        raise ValueError('DyKo requires nonnegative integer class IDs')
    # Preserve pandas groupby sorted-class order and one RandomState per group.
    return pd.concat([group.sample(n=min(n_shot, len(group)), random_state=seed)
                      for _, group in frame.groupby('train_label')]).reset_index(drop=True)

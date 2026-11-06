"""Pinned MoMIL three-order SSD network with explicit construction and survival adaptations."""
import importlib
import importlib.metadata
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from .discrete_survival import DiscreteSurvival
from .io import digest_file, digest_object
from .ssd_scan import gated_rms, split_conv_scan
from .upstream import load_definitions, load_single

PROFILES = {'momil', 'momil_survival_extension'}


class NoHubMixin:
    """Unused Hub serialization API is not part of the numerical adaptation."""


def backend_namespace(backend='torch', scan_chunk=64):
    if backend == 'torch':
        def kernel(*args, **kwargs):
            if kwargs.get('ngroups', 1) != 1:
                raise ValueError('MoMIL torch backend admits the source ngroups=1 configuration only')
            return split_conv_scan(*args, **kwargs, checkpoint_chunk=scan_chunk)
        norm = gated_rms
        identity = {'name': 'torch', 'torch': torch.__version__, 'scan_chunk': scan_chunk,
                    'scan': 'explicit full-SSM SSD recurrence with differentiable cross-chunk state'}
    elif backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('MoMIL CUDA backend requires a GPU; no automatic fallback')
        try:
            versions = {p: importlib.metadata.version(p) for p in ('mamba-ssm', 'causal-conv1d', 'triton')}
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Build pinned MoMIL Mamba2.2.4 and the explicit causal-conv1d1.5.0 admission environment') from error
        if versions['mamba-ssm'] != '2.2.4' or versions['causal-conv1d'] != '1.5.0':
            raise ValueError('MoMIL fused environment requires mamba-ssm2.2.4/causal-conv1d1.5.0; author causal range is >=1.2.0')
        kernel = load_single('momil', 'mamba_latest/mamba_ssm/ops/triton/ssd_combined.py').mamba_split_conv1d_scan_combined
        norm = load_single('momil', 'mamba_latest/mamba_ssm/ops/triton/layernorm_gated.py').rmsnorm_fn
        package = Path(importlib.import_module('mamba_ssm').__file__).parent
        identity = {'name': 'cuda', 'versions': versions, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'package_python_sha256': digest_object({str(p.relative_to(package)): digest_file(p) for p in sorted(package.rglob('*.py'))}),
                    'causal_binary_sha256': digest_file(importlib.import_module('causal_conv1d_cuda').__file__)}
    else:
        raise ValueError('MoMIL backend must be explicitly torch or cuda')
    root = 'mamba_latest/mamba_ssm/'
    norm_cls = load_definitions('momil', root+'ops/triton/layernorm_gated.py', ['RMSNorm'],
                                {'torch': torch, 'rmsnorm_fn': norm}).RMSNorm
    def unavailable(*args, **kwargs):
        raise RuntimeError('MoMIL uses fixed whole-bag single-process SSD; no implicit streaming/parallel/slow path')
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'rearrange': rearrange, 'repeat': repeat,
        'RMSNormGated': norm_cls, 'PyTorchModelHubMixin': NoHubMixin, 'mamba_split_conv1d_scan_combined': kernel,
        'ColumnParallelLinear': unavailable, 'RowParallelLinear': unavailable,
        'all_reduce': unavailable, 'reduce_scatter': unavailable, 'causal_conv1d_fn': None,
        'causal_conv1d_update': None, 'selective_state_update': None, 'mamba_chunk_scan_combined': unavailable}
    cls = load_definitions('momil', root+'modules/mamba2.py', ['Mamba2'], namespace).Mamba2
    return cls, identity


def author_network(hidden=512, state_size=64, backend='torch', scan_chunk=64):
    mamba, identity = backend_namespace(backend, scan_chunk)
    def factory(d_model, d_state, d_conv, expand):
        if (d_model, d_state, d_conv, expand) != (hidden, 64, 4, 2):
            raise ValueError('Pinned MoMIL Mamba2 constructor contract changed')
        return mamba(d_model=hidden, d_state=state_size, d_conv=d_conv, expand=expand)
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'ceil': math.ceil,
                 'Mamba2': factory, 'rearrange': rearrange}
    patches = [('super(SpeMIL, self).__init__()', 'super(MoMIL, self).__init__()'), ('512', str(hidden))]
    model = load_definitions('momil', 'models/MoMIL.py',
        ['initialize_weights', 'TransposeTokenReEmbedding', 'MoMIL'], namespace, patches).MoMIL
    return model, identity


class MoMIL(DiscreteSurvival):
    def __init__(self, bundle, profile='momil', hidden=512, layers=2, state_size=64, dropout=.25,
                 activation='relu', backend='torch', scan_chunk=64, bins=4, alpha=0.,
                 accumulation_steps=None, weighted_sampling=None, early_stopping=None, l1_weight=0.):
        survival = profile == 'momil_survival_extension'
        if profile not in PROFILES or bundle.task_type != ('survival' if survival else 'classification'):
            raise ValueError('MoMIL requires its classification or explicitly named survival-extension profile')
        if not survival and bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('MoMIL classification uses individual slide bags')
        if any(type(x) is not int or x < 1 for x in [hidden, layers, state_size, scan_chunk]) or hidden % 32:
            raise ValueError('MoMIL needs positive integer widths/layers/state/chunk and hidden divisible by32')
        if not math.isfinite(dropout) or not 0 <= dropout < 1 or activation not in {'relu', 'gelu'}:
            raise ValueError('Invalid MoMIL dropout or activation')
        steps = (32 if survival else 1) if accumulation_steps is None else accumulation_steps
        if type(steps) is not int or steps < 1 or not survival and steps != 1:
            raise ValueError('MoMIL classification updates every bag; survival accumulation must be positive')
        if any(value is not None and type(value) is not bool for value in [weighted_sampling, early_stopping]):
            raise ValueError('MoMIL sampling and early_stopping options must be boolean')
        if not survival and l1_weight:
            raise ValueError('MoMIL classification source has no L1 objective')
        super().__init__(bins, alpha, steps, l1_weight)
        self.profile, self.survival, self.dim = profile, survival, bundle.dim
        self.hidden, self.layers, self.state_size = hidden, layers, state_size
        self.weighted_sampling = survival if weighted_sampling is None else weighted_sampling
        self.early_stopping = survival if early_stopping is None else early_stopping
        self.stop_after_epoch = 41 if survival else 51
        cls, self.backend_identity = author_network(hidden, state_size, backend, scan_chunk)
        self.network = cls(bundle.dim, bins if survival else len(bundle.classes), dropout, activation,
                           survival=False, layer=layers)
        self.author_nll = load_definitions('momil', 'utils/survival_utils.py', ['nll_loss'], {'torch': torch}).nll_loss
        self.optimizer_steps, self.epoch = 0, 0

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('MoMIL preserves the registered sequence; max_instances truncation is not supported')

    def prepare(self, bundle, seed, device):
        if self.survival:
            super().prepare(bundle, seed, device)

    def make_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def start_epoch(self, epoch):
        self.epoch = epoch

    def selection_objective(self, metrics, epoch, objective):
        return objective if self.early_stopping else float(epoch)

    def get_extra_state(self):
        return {'backend': self.backend_identity, 'optimizer_steps': self.optimizer_steps, 'epoch': self.epoch}

    def set_extra_state(self, state):
        if state['backend'] != self.backend_identity:
            raise ValueError('MoMIL backend identity differs from its checkpoint')
        self.optimizer_steps, self.epoch = state['optimizer_steps'], state['epoch']

    def forward(self, x, coords=None):
        if x.ndim != 2 or not len(x) or x.shape[1] != self.dim or not torch.isfinite(x).all():
            raise ValueError('MoMIL requires finite nonempty [instances,registered FM dimension] input')
        logits = self.network(x)[0]
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite MoMIL logits')
        if not self.survival:
            return {'logits': logits}
        hazards = logits.sigmoid(); survival = torch.cumprod(1-hazards, 1)
        return {'logits': -survival.sum(1, keepdim=True), 'hazard_logits': logits,
                'hazards': hazards, 'survival': survival}

    def objective(self, result, time, event):
        return self.author_nll(result['hazards'], result['survival'], self.time_label([time]),
                               self.bin_edges.new_tensor([1-event]), alpha=self.alpha)

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        x, _ = load_unit(bundle, sorted(rows, key=lambda row: row['slide_id']), device)
        return self(x)

    def protocol(self):
        return {**(super().protocol() if self.survival else {}), 'profile': self.profile,
                'hidden': self.hidden, 'layers_per_order': self.layers, 'state_size': self.state_size,
                'backend': self.backend_identity, 'constructor_repair': 'SpeMIL->MoMIL only',
                'orders': ['stored token order', 'literal feature-channel reverse', 'square token transpose'],
                'padding': 'repeat prefix to ceil(sqrt(N)) square; all padded tokens retained in three-order attention',
                'coordinate_claim': 'sequence layout only; does not fabricate physical coordinates',
                'optimizer': 'author Adam all parameters; source weight-decay exclusion tags unused',
                'initialization': 'literal repeated recursive author initializer',
                'selection': 'validation selected metric' if self.early_stopping else 'last epoch, matching source no-early-stop classification',
                'early_stopping': self.early_stopping, 'weighted_sampling': self.weighted_sampling,
                'optimizer_steps': self.optimizer_steps, 'epoch': self.epoch,
                'survival_extension': self.survival,
                'survival_boundary': 'source head+generic author trainer composition; MoMIL absent from published survival factory',
                'survival_repairs': 'fit bins on unique observed training patients; flush/average final short window; preserve per-patient optional L1',
                'patient_order': 'slide_id sorted concatenation, stored patch order within each slide',
                'gpu_validation': 'not_run', 'paper_reproduction': 'not_run'}


def sampled_groups(model, groups):
    if not model.weighted_sampling:
        return groups
    labels = [(int(model.time_label([g[0]['time']]).item()), 1-g[0]['event']) if model.survival else g[0]['target'] for g in groups]
    counts = Counter(labels)
    weights = torch.tensor([len(labels)/counts[value] for value in labels], dtype=torch.double)
    return [groups[index] for index in torch.utils.data.WeightedRandomSampler(weights, len(groups), replacement=True)]


def momil_epoch(model, bundle, groups, optimizer, device, config):
    model.train(); groups = sampled_groups(model, groups); losses = []
    for start in range(0, len(groups), model.accumulation_steps):
        window = groups[start:start+model.accumulation_steps]
        optimizer.zero_grad(set_to_none=True)
        for rows in window:
            result = model.forward_unit(bundle, rows, device, config, train=True)
            loss = model.objective(result, rows[0]['time'], rows[0]['event']) if model.survival else F.cross_entropy(result['logits'], torch.tensor([rows[0]['target']], device=device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MoMIL objective')
            objective = loss/len(window)
            if model.l1_weight:
                objective = objective+model.l1_weight*sum(p.abs().sum() for p in model.network.parameters())
            objective.backward(); losses.append({'loss': float(loss.detach())})
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite MoMIL gradients')
        optimizer.step(); model.optimizer_steps += 1
    return losses

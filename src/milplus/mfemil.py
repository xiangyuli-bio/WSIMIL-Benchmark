"""Pinned MFE-MIL reconstruction/aggregation graphs and complete task-specific recipes."""
import contextlib
import io
import math
from collections import Counter
from functools import partial
from types import SimpleNamespace
from typing import Final

import numpy as np
import torch
from einops import rearrange, reduce
from torch import einsum, nn
from torch.nn import functional as F
from torch.utils import checkpoint

from milplus.discrete_survival import DiscreteSurvival
from milplus.upstream import load_definitions

HEADS = {'mfemil_mean': 'mean', 'mfemil_max': 'max', 'mfemil_abmil': 'att',
         'mfemil_clam_sb': 'clam_sb', 'mfemil_clam_mb': 'clam_mb', 'mfemil_transmil': 'trans'}


def timm_block(attention_backend='sdpa'):
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'partial': partial, 'Final': Final,
                 'to_2tuple': lambda value: value if isinstance(value, (tuple, list)) else (value, value),
                 'use_fused_attn': lambda: attention_backend == 'sdpa'}
    namespace['Mlp'] = load_definitions('timm_mfemil', 'timm/layers/mlp.py', ['Mlp'], namespace).Mlp
    # MFE always uses drop_path=0 and init_values=None; unsupported optional paths must fail.
    def disabled(*args, **kwargs):
        raise ValueError('Nonzero timm DropPath is outside the released MFE decoder configuration')
    namespace['DropPath'] = disabled
    return load_definitions('timm_mfemil', 'timm/models/vision_transformer.py',
                            ['Attention', 'LayerScale', 'Block'], namespace).Block


def author_model(decoder_dim=512, decoder_depth=4, decoder_heads=16, attention_backend='sdpa'):
    namespace = {'torch': torch, 'nn': nn, 'np': np, 'F': F, 'math': math, 'ceil': math.ceil,
                 'checkpoint': checkpoint, 'rearrange': rearrange, 'reduce': reduce, 'einsum': einsum,
                 'Block': timm_block(attention_backend)}
    for path, names, patches in [
        ('models/Mean_Max_MIL.py', ['initialize_weights', 'MeanMIL', 'MaxMIL'], []),
        ('models/ABMIL.py', ['DAttention'], [('feature.squeeze()', 'feature.squeeze(0)')]),
        ('models/CLAM.py', ['initialize_weights', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'CLAM_MB'],
         [('self.attention_net(h.squeeze())', 'self.attention_net(h.squeeze(0))'),
          ('self.instance_loss_fn = instance_loss_fn.cuda()', 'self.instance_loss_fn = instance_loss_fn')]),
        ('models/TransMIL.py', ['exists', 'moore_penrose_iter_pinv', 'NystromAttention', 'initialize_weights',
                              'TransLayer', 'PPEG', 'TransMIL'], [])]:
        source = load_definitions('mfemil', path, names, namespace, patches)
        # Preserve each module's own initializer in its private function globals.
        namespace.update({name: getattr(source, name) for name in names if name != 'initialize_weights'})
    patches = [
        ('from models.Mean_Max_MIL import MeanMIL', ''), ('from models.Mean_Max_MIL import MaxMIL', ''),
        ('from models.ABMIL import DAttention', ''), ('from models.CLAM import CLAM_SB', ''),
        ('from models.CLAM import CLAM_MB', ''), ('from models.TransMIL import TransMIL', ''),
        ('self.decoder = MAEFeatureDecoder(embed_dim=args.in_dim)',
         f'self.decoder = MAEFeatureDecoder(embed_dim=args.in_dim, decoder_embed_dim={decoder_dim}, decoder_depth={decoder_depth}, decoder_num_heads={decoder_heads})'),
        ('loss_mae = torch.tensor(0.0, device=data.device)',
         "raise ValueError('Nonfinite MFE reconstruction loss; source zero replacement disabled')")]
    return load_definitions('mfemil', 'mfe_mil.py',
        ['get_2d_sincos_pos_embed', 'MAEFeatureDecoder', 'MFE_MIL'], namespace, patches).MFE_MIL


class MFEMIL(DiscreteSurvival):
    def __init__(self, bundle, profile='mfemil_abmil', alpha=.3, mask_ratio=.75, shared_layers=2,
                 sub_batch=8, decoder_dim=512, decoder_depth=4, decoder_heads=16,
                 attention_backend='sdpa', precision='fp32', bins=4, survival_alpha=0.,
                 accumulation_steps=None, weighted_sampling=False):
        survival = bundle.task_type == 'survival'
        if profile not in HEADS or bundle.task_type not in {'classification', 'survival'}:
            raise ValueError('MFE-MIL requires a registered classification or survival profile')
        if profile == 'mfemil_transmil' and not survival:
            raise ValueError('The released MFE classification sweep does not include TransMIL')
        if not all(math.isfinite(v) and 0 < v < 1 for v in (alpha, mask_ratio)):
            raise ValueError('MFE alpha and mask_ratio must be finite and strictly between zero and one')
        if any(type(v) is not int or v < 1 for v in (shared_layers, sub_batch, decoder_dim, decoder_depth, decoder_heads)):
            raise ValueError('MFE layer counts, chunk size and decoder dimensions must be positive integers')
        if decoder_dim % 4 or decoder_dim % decoder_heads:
            raise ValueError('MFE decoder width must be divisible by four and its head count')
        if attention_backend not in {'sdpa', 'math'} or precision not in {'fp32', 'amp'}:
            raise ValueError('Explicit MFE attention backend and precision are required')
        if precision == 'amp' and not torch.cuda.is_available():
            raise RuntimeError('MFE AMP requires a CUDA GPU; FP32 fallback is not automatic')
        if not survival and bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('Classification MFE uses individual slide bags')
        if type(weighted_sampling) is not bool:
            raise ValueError('weighted_sampling must be boolean')
        if weighted_sampling and not survival:
            raise ValueError('MFE classification released recipe uses unweighted slide sampling')
        steps = (32 if survival else 1) if accumulation_steps is None else accumulation_steps
        if type(steps) is not int or steps < 1 or (not survival and steps != 1):
            raise ValueError('MFE classification updates once per bag; survival accumulation must be positive')
        super().__init__(bins, survival_alpha, steps)
        self.survival, self.profile, self.dim = survival, profile, bundle.dim
        self.mfe_alpha, self.mask_ratio, self.sub_batch = alpha, mask_ratio, sub_batch
        self.shared_layers, self.weighted_sampling = shared_layers, weighted_sampling
        self.precision, self.attention_backend = precision, attention_backend
        self.decoder_config = {'width': decoder_dim, 'depth': decoder_depth, 'heads': decoder_heads}
        classes = bins if survival else len(bundle.classes)
        args = SimpleNamespace(task=bundle.config.get('dataset_id', 'registered_task'), in_dim=bundle.dim,
                               no_existing=True, stop_grad=False, dropout=.25, n_classes=classes)
        with contextlib.redirect_stdout(io.StringIO()):
            self.network = author_model(decoder_dim, decoder_depth, decoder_heads, attention_backend)(
                args, feat_dim=bundle.dim, shared_layers=shared_layers, n_classes=classes,
                mask_ratio=mask_ratio, sub_batch=sub_batch, mil_head_type=HEADS[profile], survival=survival)
        self.scaler = torch.amp.GradScaler('cuda', enabled=precision == 'amp')
        self.optimizer_steps = 0
        self.source_nll = load_definitions('mfemil', 'utils/survival_utils.py', ['nll_loss'], {'torch': torch}).nll_loss

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('MFE-MIL preserves the complete registered sequence; max_instances is not supported')

    def prepare(self, bundle, seed, device):
        if self.precision == 'amp' and torch.device(device).type != 'cuda':
            raise ValueError('MFE AMP must be paired with --device cuda')
        if self.survival:
            bundle.check(deep=True, require_coords=True)
            super().prepare(bundle, seed, device)

    def make_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_extra_state(self):
        return {'optimizer_steps': self.optimizer_steps, 'scaler': self.scaler.state_dict(),
                'precision': self.precision, 'attention_backend': self.attention_backend}

    def set_extra_state(self, state):
        if state['precision'] != self.precision or state['attention_backend'] != self.attention_backend:
            raise ValueError('MFE numerical backend differs from the recorded checkpoint')
        self.optimizer_steps = state['optimizer_steps']
        self.scaler.load_state_dict(state['scaler'])

    def forward(self, x, coords=None, joint=False):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('MFE-MIL requires finite nonempty [instances, registered FM width] features')
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=self.precision == 'amp'):
            reconstruction = self.network.forward_mae(x) if joint else None
            first, second, _ = self.network.forward_mil(x)
        if not torch.isfinite(first).all() or (joint and not torch.isfinite(reconstruction)):
            raise ValueError('Nonfinite MFE output or reconstruction objective')
        if self.survival:
            result = {'logits': -second.sum(1, keepdim=True), 'hazards': first, 'survival': second}
        else:
            result = {'logits': first}
        if joint:
            result['mfe_loss'] = reconstruction
        return result

    def forward_unit(self, bundle, rows, device, config, train=False):
        bags = []
        for row in sorted(rows, key=lambda r: r['slide_id']):
            x, coords = bundle.load(row, require_coords=self.survival)
            if self.survival:
                # Native sort key assumes bounded pixel x. Stable lexicographic sorting
                # avoids collisions and sorts each physical slide frame independently.
                indices = np.lexsort((coords[:, 0].numpy(), coords[:, 1].numpy()))
                x = x[indices]
            bags.append(x)
        return self(torch.cat(bags).to(device), joint=train)

    def objective(self, result, time, event):
        return self.source_nll(result['hazards'], result['survival'], self.time_label([time]),
                               self.bin_edges.new_tensor([1-event]), alpha=self.alpha)

    def protocol(self):
        return {**(super().protocol() if self.survival else {}), 'profile': self.profile,
                'alpha_mfe': self.mfe_alpha, 'mask_ratio': self.mask_ratio, 'sub_batch': self.sub_batch,
                'shared_layers': self.shared_layers, 'decoder': self.decoder_config,
                'precision': self.precision, 'attention_backend': self.attention_backend,
                'optimizer_steps': self.optimizer_steps, 'optimizer': 'Adam; no scheduler',
                'objective': '(1-alpha_mfe)*task_loss + alpha_mfe*masked normalized-feature MSE',
                'order': 'classification stored patch order; survival per-slide stable (y,x), then slide-ID concatenation',
                'grid': 'square-packed ordered features, zero padding; no physical adjacency claim',
                'target': 'L2-normalized frozen raw features, matching source stop_grad=False',
                'head': HEADS[self.profile], 'clam': 'MFE released bag-only recipe; instance classifiers unused, not standalone full CLAM',
                'unused_source_parameters': 'feat_scale, norm, and CLAM instance classifiers',
                'inference': 'shared adapter and MIL head only; no masking, reconstruction or decoder call',
                'weighted_sampling': self.weighted_sampling,
                'repairs': 'singleton-safe attention squeeze, device-neutral CE, fail on NaN, train-only time bins, per-slide sort frames, actual tail-window averaging',
                'gpu_validation': 'not_run', 'paper_reproduction': 'not_run'}


def mfemil_epoch(model, bundle, groups, optimizer, device, config):
    if model.survival and model.weighted_sampling:
        labels = [(int(model.time_label([g[0]['time']]).item()), 1-g[0]['event']) for g in groups]
        counts = Counter(labels)
        weights = torch.tensor([len(groups)/counts[label] for label in labels], dtype=torch.double)
        groups = [groups[index] for index in torch.utils.data.WeightedRandomSampler(weights, len(groups))]
    model.train()
    losses = []
    for start in range(0, len(groups), model.accumulation_steps):
        window = groups[start:start+model.accumulation_steps]
        optimizer.zero_grad(set_to_none=True)
        for rows in window:
            result = model.forward_unit(bundle, rows, device, config, train=True)
            task_loss = model.objective(result, rows[0]['time'], rows[0]['event']) if model.survival else F.cross_entropy(
                result['logits'].float(), torch.tensor([rows[0]['target']], device=device))
            loss = (1-model.mfe_alpha)*task_loss + model.mfe_alpha*result['mfe_loss']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MFE joint training loss')
            model.scaler.scale(loss/len(window)).backward()
            losses.append({'loss': float(loss.detach()), 'task': float(task_loss.detach()),
                           'mfe': float(result['mfe_loss'].detach())})
        model.scaler.unscale_(optimizer)
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite MFE parameter gradients')
        model.scaler.step(optimizer)
        model.scaler.update()
        model.optimizer_steps += 1
    return losses

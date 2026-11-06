"""Pinned Wibawa multi-depth residual attention on frozen WSI feature bags.

The released source is a MNIST supplementary demo; this is its explicit frozen
WSI aggregator adaptation, not reproduction of the private tumour-filter model.
"""
import math
from pathlib import Path

import h5py
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import digest_file, digest_object
from .stochastic import inference_rng
from .upstream import load_definitions

PROFILES = {'msattention_source', 'msattention_repaired'}


def author_network(repaired=False):
    replacements = [(f'x{i} = nn.Dropout(self.n_dropout)(x{i})',
                     f'x{i} = F.dropout(x{i}, p=self.n_dropout, training=self.training)')
                    for i in range(1, 4)] if repaired else []
    return load_definitions('msattention', 'model.py', ['MultiAttentionMIL'],
                            {'torch': torch, 'nn': nn, 'F': F}, replacements).MultiAttentionMIL


class MultiAttentionMIL(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'three separate source attention distributions at learned feature depths; not a spatial pyramid or a causal explanation'

    def __init__(self, bundle, profile='msattention_repaired', dropout=.5,
                 prediction_seed=2026, prediction_samples=1):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification':
            raise ValueError('Wibawa MultiAttentionMIL requires a registered classification route')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('Wibawa attention preserves individual full slide bags')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('Dropout must be finite in [0,1)')
        if type(prediction_seed) is not int or prediction_seed < 0 or type(prediction_samples) is not int or prediction_samples < 1:
            raise ValueError('Prediction seed must be nonnegative and sample count positive')
        if profile.endswith('repaired') and prediction_samples != 1:
            raise ValueError('Deterministic repaired evaluation uses prediction_samples=1')
        self.bundle, self.dim, self.profile = bundle, bundle.dim, profile
        self.dropout, self.prediction_seed, self.prediction_samples = dropout, prediction_seed, prediction_samples
        self.network = author_network(profile.endswith('repaired'))(
            num_classes=len(bundle.classes), num_features=bundle.dim,
            use_dropout=dropout > 0, n_dropout=dropout)
        self.optimizer_steps = 0

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('Wibawa attention uses complete bags and unclipped Adam')
        if cfg['selection_metric'] != 'log_loss':
            raise ValueError('Wibawa adaptation selects validation bag CE, with TEST locked')
        if cfg['weight_decay'] != 0:
            raise ValueError('The released supplementary Adam uses zero weight decay')
        self.stop_after_epoch = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Wibawa attention requires finite nonempty [N,D] patch features')
        logits, a1, a2, a3 = self.network(x.unsqueeze(0))
        if not torch.isfinite(logits).all():
            raise ValueError('Wibawa attention produced nonfinite logits')
        return {'logits': logits, 'attention': torch.cat([a1, a2, a3], dim=0).T}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:
            raise ValueError('Wibawa attention preserves individual full slide bags')
        x = bundle.load(rows[0])[0].to(device)
        if train or self.profile.endswith('repaired'):
            return self(x)
        key = digest_object({'seed': self.prediction_seed, 'slide': rows[0]['slide_id']})
        logits, attention = [], []
        for sample in range(self.prediction_samples):
            with inference_rng(device, (int(key[:15], 16)+sample) % (2**63-1)):
                result = self(x)
                logits.append(result['logits'].softmax(-1))
                attention.append(result['attention'])
        return {'logits': torch.stack(logits).mean(0).clamp_min(torch.finfo(x.dtype).tiny).log(),
                'attention': torch.stack(attention).mean(0)}

    def export_instances(self, result, row, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object({'slide_id': row['slide_id']})[:24]+'.h5')
        _, coords = self.bundle.load(row)
        with h5py.File(path, 'w') as h:
            h['attention'] = result['attention'].detach().cpu().numpy()
            if coords is not None:
                h['coords'] = coords.detach().cpu().numpy()
            h.attrs['method'] = self.profile
            h.attrs['level_order'] = 'first,second,third learned feature depth'
            h.attrs['score_semantics'] = self.instance_score_semantics
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path),
                'instances': len(result['attention']), 'coordinates': coords is not None}

    def get_extra_state(self):
        return {'optimizer_steps': self.optimizer_steps}

    def set_extra_state(self, state):
        if set(state) != {'optimizer_steps'} or type(state['optimizer_steps']) is not int or state['optimizer_steps'] < 0:
            raise ValueError('Invalid Wibawa attention checkpoint state')
        self.optimizer_steps = state['optimizer_steps']

    def protocol(self):
        return {'method': self.profile, 'family': 'msattention', 'dimension': self.dim,
                'hidden': 128, 'depths': 3, 'dropout': self.dropout,
                'input': 'frozen FM features per slide; single optical feature stream',
                'architecture': 'three learned feature depths, tanh attention, cumulative residual pooled vectors; no fake spatial grid',
                'recipe': 'supplementary batch1 shuffled full bags, CE, Adam betas(.9,.999), eps1e-8, zero decay, no clipping/scheduler',
                'selection': 'independent validation log-loss at declared evaluation unit, first best; replaces source epoch-local training-accuracy save',
                'evaluation_unit': self.bundle.config.get('evaluation_unit', 'slide'),
                'optimizer_steps': self.optimizer_steps,
                'prediction': {'rule': 'fixed per-slide dropout streams and mean probabilities with TRAIN RNG restored' if self.profile.endswith('source') else 'named repair: dropout obeys train/eval mode',
                               'seed': self.prediction_seed, 'samples': self.prediction_samples if self.profile.endswith('source') else 1},
                'source_scope': 'MNIST supplementary implementation; explicit frozen WSI aggregator transfer. Private tissue/tumour selection pipeline not released or reproduced.',
                'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'gpu_validation': 'not_run'}


def msattention_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    loader = DataLoader(groups, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda batch: batch[0])
    for rows in loader:
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_unit(bundle, rows, device, cfg, train=True)
        loss = F.cross_entropy(result['logits'], torch.tensor([rows[0]['target']], device=device))
        if not torch.isfinite(loss):
            raise ValueError('Wibawa attention objective is nonfinite')
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('Wibawa attention gradient is nonfinite')
        optimizer.step(); model.optimizer_steps += 1
        losses.append({'loss': float(loss.detach())})
    return losses

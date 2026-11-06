"""Pinned HEALNet fusion with its weighted survival objective and step-wise recipe."""
import copy
import math
from functools import wraps
from typing import Optional, Union

import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch import nn
from torch.nn import functional as F

from .multimodal import MolecularSurvival
from .upstream import load_definitions


def load_healnet(strict=True):
    patch = [(("                try:\n                    x = cross_attn(x, context = tensors[i], mask = mask) + x\n"
               "                    x =  cross_ff(x) + x\n                except:\n                    pass"),
              ("                x = cross_attn(x, context=tensors[i], mask=mask) + x\n"
               "                x = cross_ff(x) + x"))] if strict else []
    if strict:
        patch.append(('                        continue\n                cross_attn=layer[i*2]',
                      '                    continue\n                cross_attn=layer[i*2]'))
    names = ['HealNet', 'exists', 'default', 'cache_fn', 'fourier_encode', 'PreNorm',
             'GELU', 'SELU', 'RELU', 'FeedForward', 'temperature_softmax', 'Attention']
    return load_definitions('healnet', 'healnet/models/healnet.py', names,
        {'torch': torch, 'nn': nn, 'F': F, 'einsum': torch.einsum, 'pi': math.pi, 'log': math.log,
         'wraps': wraps, 'List': list, 'Optional': Optional, 'Union': Union,
         'rearrange': rearrange, 'repeat': repeat, 'Reduce': Reduce}, patch)


class HEALNetSurvival(MolecularSurvival):
    def __init__(self, dim, omics, bins=4, alpha=.4, batch_size=4, loss_divisor=16.,
                 l1_weight=1e-4, bin_weighting='inverse', omics_layout='vector',
                 depth=2, num_latents=4, latent_dim=16, num_freq_bands=2, max_freq=2.,
                 cross_heads=1, latent_heads=8, cross_dim_head=16, latent_dim_head=16,
                 attn_dropout=.5, ff_dropout=.5, fourier_encode_data=True,
                 self_per_cross_attn=0, weight_tie_layers=False, snn=True,
                 eval_missing=None):
        if omics is None:
            raise ValueError('HEALNet requires explicit patient-aligned omics')
        super().__init__('healnet', dim, omics, bins, alpha, batch_size, l1_weight)
        for name, value in {'depth': depth, 'num_latents': num_latents, 'latent_dim': latent_dim,
                            'cross_heads': cross_heads, 'latent_heads': latent_heads,
                            'cross_dim_head': cross_dim_head, 'latent_dim_head': latent_dim_head}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'HEALNet {name} must be a positive integer')
        if self_per_cross_attn not in {0, 1}:
            raise ValueError('Pinned HEALNet supports self_per_cross_attn 0 or 1; larger values fail upstream')
        if not isinstance(num_freq_bands, int) or num_freq_bands < 1 or not math.isfinite(max_freq) or max_freq <= 0:
            raise ValueError('Invalid Fourier frequency configuration')
        if not 0 <= attn_dropout < 1 or not 0 <= ff_dropout < 1:
            raise ValueError('HEALNet dropout must be in [0,1)')
        if not math.isfinite(loss_divisor) or loss_divisor <= 0:
            raise ValueError('loss_divisor must be finite and positive')
        if omics_layout not in {'vector', 'gene_tokens'} or bin_weighting not in {'inverse', 'inverse_root', 'none'}:
            raise ValueError('Unknown HEALNet omics layout or time-bin weighting')
        if eval_missing not in {None, 'omics', 'wsi'}:
            raise ValueError('eval_missing must be null, omics or wsi')
        self.dim, self.omics_layout, self.eval_missing = dim, omics_layout, eval_missing
        self.bin_weighting, self.batch_size, self.loss_divisor = bin_weighting, batch_size, loss_divisor
        self.register_buffer('bin_weights', torch.ones(bins))
        self.network = load_healnet().HealNet(n_modalities=2,
            channel_dims=[len(omics.columns) if omics_layout == 'vector' else 1, dim],
            num_spatial_axes=[1, 1], out_dims=bins, depth=depth, l_c=num_latents, l_d=latent_dim,
            num_freq_bands=num_freq_bands, max_freq=max_freq, x_heads=cross_heads, l_heads=latent_heads,
            cross_dim_head=cross_dim_head, latent_dim_head=latent_dim_head, attn_dropout=attn_dropout,
            ff_dropout=ff_dropout, fourier_encode_data=fourier_encode_data, self_per_cross_attn=self_per_cross_attn,
            weight_tie_layers=weight_tie_layers, snn=snn)
        self.author_nll = load_definitions('healnet', 'healnet/models/survival_loss.py', ['nll_loss'],
                                          {'torch': torch}).nll_loss
        self.epochs, self.steps_per_epoch, self.optimizer, self.scheduler = 50, 0, None, None
        self.saved_scheduler = None

    def configure_training(self, config):
        self.epochs = config['epochs']

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        super().prepare(bundle, seed, device)
        patients = bundle.rows.query("subset == 'train'").drop_duplicates('global_patient_id')
        labels = self.time_label(patients.time.to_numpy())
        counts = torch.bincount(labels, minlength=self.bins).to(self.bin_weights)
        if self.bin_weighting != 'none':
            if (counts == 0).any():
                raise ValueError('HEALNet time-bin class weights need a training patient in every bin')
            self.bin_weights.copy_(1 / (counts if self.bin_weighting == 'inverse' else counts.sqrt()))
        self.steps_per_epoch = math.ceil(len(patients) / self.batch_size)

    def make_optimizer(self, lr, weight_decay):
        if weight_decay != 0:
            raise ValueError('HEALNet author Adam has no L2 decay; use model.l1_weight')
        # lr is exposed as the maximum cycle learning rate. OneCycle starts at lr/25.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.max_lr = lr
        return self.optimizer

    def ensure_scheduler(self):
        if self.scheduler is not None:
            return
        if self.steps_per_epoch < 1:
            raise RuntimeError('HEALNet training steps must be prepared or restored')
        # OneCycleLR's constructor mutates lr/betas; preserve resumed optimizer values.
        saved_groups = [dict(g) for g in self.optimizer.param_groups] if self.saved_scheduler else None
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.max_lr,
                            epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)
        if self.saved_scheduler:
            self.scheduler.load_state_dict(self.saved_scheduler)
            for group, saved in zip(self.optimizer.param_groups, saved_groups):
                group.update(saved)

    def forward(self, x, omics, coords=None):
        if x is None and omics is None:
            raise ValueError('HEALNet needs at least one available modality')
        if x is not None and (x.ndim != 2 or x.shape[1] != self.dim or len(x) < 1 or not torch.isfinite(x).all()):
            raise ValueError('HEALNet WSI must be finite nonempty [N,D] features')
        molecular = None if omics is None else self.normalize_omics(omics)
        if molecular is not None:
            molecular = molecular[None, None, :] if self.omics_layout == 'vector' else molecular[None, :, None]
        # Preserve modality slots even for missing-modality inference; source ablation collapsed them.
        logits = self.network([molecular, None if x is None else x[None]])
        return self.result(logits)

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        missing = None if train else self.eval_missing
        x = None if missing == 'wsi' else load_unit(bundle, rows, device, config['max_instances'], train)[0]
        molecular = None if missing == 'omics' else bundle.omics.tensor(rows[0]['global_patient_id']).to(device)
        return self(x, molecular)

    def weighted_loss(self, hazards, labels, events):
        return self.author_nll(hazards, None, labels, 1-events,
            weights=None if self.bin_weighting == 'none' else self.bin_weights.to(hazards), alpha=self.alpha)

    def objective(self, result, time, event):
        return self.weighted_loss(result['hazards'], self.time_label([time]),
                                  self.bin_edges.new_tensor([event]))

    def validation_metrics(self, predictions):
        hazards = torch.tensor(predictions[[f'hazards_{i}' for i in range(self.bins)]].to_numpy(),
                               dtype=torch.float64, device=self.bin_edges.device)
        labels = self.time_label(predictions.time.to_numpy())
        events = torch.tensor(predictions.event.to_numpy(), device=hazards.device)
        # Preserve upstream mean-of-batch-means selection, including its last short batch.
        values = [self.weighted_loss(hazards[i:i+self.batch_size], labels[i:i+self.batch_size],
                                     events[i:i+self.batch_size]) for i in range(0, len(events), self.batch_size)]
        return {'survival_loss': float(torch.stack(values).mean())}

    def get_extra_state(self):
        return {'steps_per_epoch': self.steps_per_epoch, 'epochs': self.epochs,
                'scheduler': copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else self.saved_scheduler}

    def set_extra_state(self, state):
        self.steps_per_epoch, self.epochs = state['steps_per_epoch'], state['epochs']
        self.saved_scheduler = copy.deepcopy(state['scheduler'])
        if self.scheduler and self.saved_scheduler:
            self.scheduler.load_state_dict(self.saved_scheduler)

    def protocol(self):
        return {**super().protocol(), 'recipe_scope': 'author fusion and weighted NLL; patient-level PULS data protocol',
                'bin_weighting': self.bin_weighting, 'bin_weights': self.bin_weights.tolist(),
                'loss_divisor': self.loss_divisor, 'batch_size': self.batch_size,
                'optimizer': 'Adam, zero L2; OneCycleLR every patient-batch update; lr is max_lr',
                'omics_layout': self.omics_layout, 'modality_order': ['omics', 'wsi'],
                'eval_missing': self.eval_missing,
                'positions': 'Fourier encoding of sequence indices, not physical coordinates',
                'multi_slide': 'concatenate patient slide bags in manifest order; explicit PULS adaptation',
                'compatibility': 'remove exception swallowing; retain missing modality slots; reject all missing',
                'preprocessing': 'training-only normalization and event bins; no global mean imputation',
                'selection': 'validation weighted survival loss; no test-set selection'}


def healnet_epoch(model, bundle, groups, optimizer, device, config):
    model.ensure_scheduler()
    losses = []
    for start in range(0, len(groups), model.batch_size):
        window = groups[start:start+model.batch_size]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.
        for rows in window:
            result = model.forward_unit(bundle, rows, device, config, train=True)
            loss = model.objective(result, rows[0]['time'], rows[0]['event'])
            (loss / len(window) / model.loss_divisor).backward()
            batch_loss += float(loss.detach()) / len(window)
        penalty = model.l1_weight * sum(p.abs().sum() for p in model.network.parameters())
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        model.scheduler.step()
        losses.append({'loss': batch_loss/model.loss_divisor + float(penalty.detach()),
                       'survival_loss': batch_loss, 'l1': float(penalty.detach())})
    return losses

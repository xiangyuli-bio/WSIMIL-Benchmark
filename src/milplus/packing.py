"""Author PackMIL: packed bags, hyperslide targets and attention downsampling."""
import math
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .discrete_survival import DiscreteSurvival, discrete_nll
from .upstream import load_definitions


class UnselectedAuthorBranch:
    def __init__(self, *args, **kwargs):
        raise ValueError('This PackMIL profile does not select the asymmetric-loss/other-backbone branch')


def load_pack_sources():
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'partial': partial,
             'Callable': Callable, 'List': list, 'Tensor': Tensor,
             'rearrange': rearrange, 'repeat': repeat, 'orig_pad_sequence': pad_sequence,
             'SAttention': UnselectedAuthorBranch, 'DSAttention': UnselectedAuthorBranch,
             'AsymmetricLossSingleLabel': UnselectedAuthorBranch}
    util_names = ['initialize_weights', 'one_hot', 'apply_label_activation', 'mixup_target',
                  '_argmin', 'mixup_target_batched', 'get_dropout', 'group_seqs']
    util = load_definitions('packmil', 'modules/pack/pack_util.py', util_names, scope)
    scope.update({name: getattr(util, name) for name in util_names})
    ads = load_definitions('packmil', 'modules/pack/ads.py', ['DimensionDrop', 'ADS'], scope)
    scope['ADS'] = ads.ADS
    pack = load_definitions('packmil', 'modules/pack/packing.py',
                           ['get_packs', '_process_kept_part', '_process_res_part'], scope)
    scope['get_packs'] = pack.get_packs
    base = load_definitions('packmil', 'modules/pack/pack_baseline.py',
                           ['DAttention', 'apply_function_nonpad', 'MILBase'], scope)
    scope.update(DAttention=base.DAttention, MILBase=base.MILBase)
    losses = load_definitions('packmil', 'modules/pack/pack_loss.py',
                             ['FocalLoss', 'nll_loss_mixup', 'NLLSurvMulLoss'], scope)
    scope.update(FocalLoss=losses.FocalLoss, NLLSurvMulLoss=losses.NLLSurvMulLoss)
    network = load_definitions('packmil', 'modules/packmil.py', ['PackMIL'], scope)
    return network, pack, util, ads, losses, base


class PackedMIL(DiscreteSurvival):
    def __init__(self, dim, classes=2, survival=False, hidden=512, batch_size=32, bins=4,
                 token_dropout=.5, pack_length=None, min_sequence=None, downsample_ratio=None,
                 auxiliary_weight=None, dropout=.25, min_lr=1e-7):
        super().__init__(bins=bins)
        if any(not isinstance(v, int) or v < 1 for v in (dim, hidden, batch_size)):
            raise ValueError('PackMIL widths and batch size must be positive integers')
        if batch_size < 2 or not 0 < token_dropout < 1 or dropout not in {0., .25}:
            raise ValueError('PackMIL requires batch_size>=2, 0<token_dropout<1, and author dropout=0 or .25')
        pack_length = (3200 if survival else 2400) if pack_length is None else pack_length
        min_sequence = (256 if survival else 128) if min_sequence is None else min_sequence
        downsample_ratio = (3 if survival else 4) if downsample_ratio is None else downsample_ratio
        auxiliary_weight = (.5 if survival else .2) if auxiliary_weight is None else auxiliary_weight
        if (any(not isinstance(v, int) for v in (pack_length, min_sequence, downsample_ratio))
                or min_sequence < 1 or pack_length < min_sequence or downsample_ratio < 2):
            raise ValueError('PackMIL requires pack_length>=min_sequence>=1 and ADS downsample_ratio>=2')
        if not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0 or not math.isfinite(min_lr) or min_lr < 0:
            raise ValueError('PackMIL auxiliary weight must be positive and minimum lr nonnegative')
        self.is_survival, self.batch_size, self.auxiliary_weight = survival, batch_size, auxiliary_weight
        network, *_ = load_pack_sources()
        self.network = network.PackMIL(mil='abmil', task_type='surv' if survival else 'sub',
            token_dropout=token_dropout, group_max_seq_len=pack_length, min_seq_len=min_sequence,
            pack_residual=True, downsample_mode='ads', residual_loss='nll' if survival else 'focal',
            residual_downsample_r=downsample_ratio, input_dim=dim, inner_dim=hidden,
            n_classes=bins if survival else classes, dropout=dropout, act='gelu', mil_bias=True)
        self.min_lr, self.total_epochs = min_lr, 75
        self.optimizer, self.base_lr, self.epoch = None, None, 0

    def configure_training(self, config):
        self.total_epochs = config['epochs']

    def prepare(self, bundle, seed, device):
        if self.is_survival:
            super().prepare(bundle, seed, device)

    def validate_bags(self, bags):
        if not bags or any(x.ndim != 2 or len(x) < 1 or not torch.isfinite(x).all() for x in bags):
            raise ValueError('PackMIL requires finite nonempty [N,D] bags')
        # Source int(N*rate) partition can be empty with extreme rates. Do not fabricate instances.
        threshold = self.network.min_seq_len*self.network.downsample_r
        if self.training and any(len(x) >= threshold and
                not 1 <= int(len(x)*self.network.token_dropout) < len(x) for x in bags):
            raise ValueError('PackMIL sampling would create an empty branch; change token_dropout/min_sequence explicitly')

    def batch_objective(self, bags, targets, events=None):
        self.validate_bags(bags)
        label = (targets, 1-events) if self.is_survival else targets
        logits, auxiliary, original, kept, pad, length_std = self.network(bags, label=label)
        main = discrete_nll(logits, targets, events) if self.is_survival else F.cross_entropy(logits, targets)
        return {'loss': main+self.auxiliary_weight*auxiliary, 'main_loss': main,
                'auxiliary_loss': auxiliary, 'logits': logits,
                'mean_original_patches': original, 'mean_kept_patches': kept,
                'padding_percent': pad, 'kept_length_std': length_std}

    def forward(self, x, coords=None, label=None):
        if self.training:
            raise ValueError('PackMIL training requires batch_objective to preserve its residual hyperslide branch')
        self.validate_bags([x])
        # Explicit B axis avoids source in-place unsqueeze on the caller's tensor.
        logits = self.network(x.unsqueeze(0))
        return self.result(logits) if self.is_survival else {'logits': logits}

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.base_lr = lr
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch = epoch
        if self.optimizer is not None:
            lr = self.min_lr+(self.base_lr-self.min_lr)*.5*(1+math.cos(math.pi*epoch/self.total_epochs))
            for group in self.optimizer.param_groups:
                group['lr'] = lr

    def protocol(self):
        return {**(super().protocol() if self.is_survival else {}),
                'backbone': 'pinned author packed ABMIL', 'batch_size': self.batch_size,
                'token_dropout': self.network.token_dropout,
                'branch_fraction': 'source assigns int(N*token_dropout) to main branch; name does not mean removed fraction',
                'pack_length': self.network.group_max_seq_len, 'min_sequence': self.network.min_seq_len,
                'downsample_ratio': self.network.downsample_r, 'auxiliary_weight': self.auxiliary_weight,
                'main_objective': 'discrete censored NLL alpha=0' if self.is_survival else 'cross entropy',
                'auxiliary_objective': 'author hyperslide mixed-censor NLL' if self.is_survival else 'author max-count-normalized targets and focal BCE',
                'downsampler': 'author ADS attention residual, MLP and channel selection after token grouping',
                'inference': 'all tokens through ADS attention+MLP; no channel drop/token reduction in author eval',
                'packing': 'author sequential bag grouping, original slide order, masked independent main attention',
                'short_bags': 'author reuses full short bag in both branches below minimum threshold',
                'patient_rule': 'survival concatenates same-patient slides before packing; no spatial coordinates used',
                'optimizer': 'Adam; epoch cosine min_lr='+str(self.min_lr)+' without warmup',
                'scope': 'ABMIL classification/survival recipe; no PANDA-specific grade or other backbone claim',
                'selection': 'PULS fixed validation set; test not used to select checkpoint',
                'tail_batch': 'retained, including singleton tail; author optional no_drop_last behavior'}


def packed_epoch(model, bundle, groups, optimizer, device, config):
    from .engine import load_unit
    losses = []
    for start in range(0, len(groups), model.batch_size):
        batch = groups[start:start+model.batch_size]
        bags = [load_unit(bundle, rows, device, config['max_instances'], True)[0] for rows in batch]
        if model.is_survival:
            target = model.time_label([rows[0]['time'] for rows in batch])
            events = torch.tensor([rows[0]['event'] for rows in batch], device=device)
        else:
            target = torch.tensor([rows[0]['target'] for rows in batch], device=device)
            events = None
        out = model.batch_objective(bags, target, events)
        if not torch.isfinite(out['loss']):
            raise FloatingPointError('PackMIL produced nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append({k: float(v.detach()) if isinstance(v, Tensor) else float(v)
                       for k, v in out.items() if k != 'logits'})
    return losses

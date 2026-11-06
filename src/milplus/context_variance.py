"""Author CAPRMIL context tokenization and DeepSMILE's frozen-feature VarMIL head."""
import math
import warnings

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions

CAPR_METHODS = {'caprmil_mean', 'caprmil_attention', 'caprmil_gated'}


def load_capr():
    return load_definitions('caprmil', 'src/model_caprmil.py',
        ['MILAggregator', 'Attn_Net', 'DAttn_Net_Gated', 'TransolverAttention', 'TransolverBlock', 'CAPRMIL'],
        {'torch': torch, 'nn': nn, 'rearrange': rearrange}).CAPRMIL


def capr_scheduler():
    return load_definitions('caprmil', 'schedulers/LinearCosine.py', ['LinearWarmupCosineAnnealingLR'],
        {'math': math, 'warnings': warnings, 'List': list, 'Optimizer': torch.optim.Optimizer,
         '_LRScheduler': torch.optim.lr_scheduler._LRScheduler}).LinearWarmupCosineAnnealingLR


class ContextPatchMIL(nn.Module):
    def __init__(self, dim, classes, profile='caprmil_mean', hidden=128, heads=8, clusters=4,
                 depth=1, mlp_ratio=4, dropout=.1, use_temperature=True,
                 scheduler='cosine', min_lr=1e-7, warmup_epochs=6, warmup_lr=1e-5):
        super().__init__()
        if profile not in CAPR_METHODS or min(dim, hidden, heads, clusters, depth, mlp_ratio) < 1:
            raise ValueError('Invalid CAPRMIL profile or architecture dimensions')
        if hidden % heads or not 0 <= dropout < 1 or scheduler not in {'cosine', 'linearcosine'}:
            raise ValueError('CAPRMIL requires divisible head width, valid dropout and explicit scheduler')
        mode = {'caprmil_mean': 'mean', 'caprmil_attention': 'attn', 'caprmil_gated': 'gated_attn'}[profile]
        self.network = load_capr()(input_dim=dim, hidden_dim=hidden, num_heads=heads,
            cluster_num=clusters, T_depth=depth, mlp_ratio=mlp_ratio, dropout=dropout,
            aggregator=mode, num_classes=classes, use_temperature=use_temperature)
        self.profile, self.schedule_name = profile, scheduler
        self.min_lr, self.warmup_epochs, self.warmup_lr = min_lr, warmup_epochs, warmup_lr
        self.total_epochs, self.scheduler, self.scheduler_state = 30, None, None

    def configure_training(self, config):
        self.total_epochs = config['epochs']
        if self.schedule_name == 'linearcosine' and not 2 <= self.warmup_epochs < self.total_epochs:
            raise ValueError('CAPRMIL linearcosine requires 2 <= warmup_epochs < total epochs')
        if not 0 <= self.min_lr <= config['lr'] or not 0 <= self.warmup_lr <= config['lr']:
            raise ValueError('CAPRMIL scheduler bounds must be between zero and the base learning rate')

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        if self.schedule_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.total_epochs, eta_min=self.min_lr)
        else:
            self.scheduler = capr_scheduler()(optimizer, self.warmup_epochs, self.total_epochs,
                                              warmup_start_lr=self.warmup_lr, eta_min=self.min_lr)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def after_validation(self, metrics):
        self.scheduler.step()

    def get_extra_state(self):
        return {'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or len(x) == 0:
            raise ValueError('CAPRMIL requires a nonempty feature bag [N,D]')
        return {'logits': self.network(x.unsqueeze(0))}

    def protocol(self):
        return {'profile': self.profile, 'context': 'author soft cluster tokens, token self-attention, de-slicing',
                'objective': 'cross entropy', 'optimizer': 'AdamW', 'scheduler': self.schedule_name,
                'epochs': self.total_epochs, 'min_lr': self.min_lr,
                'source_entry_fix': 'LitCAPRMIL refers to undefined TSMIL; call the author CAPRMIL class directly',
                'head_dimension': 'hidden // heads as in author class; no separately effective head_dim parameter'}


def load_varmil():
    # Only logging/obsolete Lightning metrics initialization is removed. The
    # Attention/VarAttention equations and original Adam trainer are retained.
    replacements = [
        ('self.auroc = torchmetrics.AUROC()', 'pass'),
        ('self.f1 = torchmetrics.F1()', 'pass'),
        ('self.pr_curve = torchmetrics.PrecisionRecallCurve()', 'pass'),
        ('self.save_hyperparameters()', 'pass'),
        ('S = (upper / lower)', "\n        if (lower <= 0).any():\n            raise ValueError('VarMIL variance requires at least two positive attention weights')\n        S = (upper / lower)"),
    ]
    return load_definitions('varmil', 'dlup_lightning_mil/models/heads/deepmil.py',
        ['Attention', 'VarAttention'], {'torch': torch, 'F': F, 'nn': nn, 'LightningModule': nn.Module},
        replacements=replacements).VarAttention


class VarianceMIL(nn.Module):
    def __init__(self, dim, classes, hidden=128):
        super().__init__()
        if min(dim, classes, hidden) < 1:
            raise ValueError('VarMIL dimensions must be positive')
        self.network = load_varmil()(lr=.001, hidden_features=hidden, in_features=dim,
                                     num_classes=classes, weight_decay=0.)

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or len(x) < 2:
            raise ValueError('VarMIL weighted variance needs at least two instances')
        logits, attention = self.network(x[None])
        return {'logits': logits, 'attention': attention}

    def protocol(self):
        return {'aggregation': 'author attention-weighted raw-feature mean and variance concatenation',
                'variance_correction': 'N_nonzero / (N_nonzero-1), not 1/(1-sum(weights**2))',
                'objective': 'cross entropy', 'optimizer': 'Adam',
                'scope': 'frozen-feature VarMIL classifier; SimCLR encoder pretraining is separate',
                'source_limit': 'author reproduction shell scripts are empty at this commit'}

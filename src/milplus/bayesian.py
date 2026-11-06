"""Pinned Bayes-MIL: variational weights, SDPR, spatial APCRF and MC prediction."""
import math
import numbers

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .spatial_attention import checked_points, spatial_metadata
from .stochastic import inference_rng
from .upstream import load_definitions

BAYES_METHODS = {'bayesmil_vis', 'bayesmil_sdpr', 'bayesmil_apcrf'}


def load_bayes(dim=1024, hidden=512, attention_dim=256, patch_size=256, corrected_stride=True):
    vdo = load_definitions('bayesmil', 'models/linear_vdo.py', ['LinearVDO', 'Conv2dVDO'],
        {'torch': torch, 'nn': nn, 'F': F, 'Parameter': nn.Parameter, 'eps': 1e-8},
        replacements=[('si = torch.sqrt(var)',
            ('positive = var > 0\n        si = torch.where(positive, '
             'torch.sqrt(torch.where(positive, var, torch.ones_like(var))), torch.zeros_like(var))'))])
    init = load_definitions('bayesmil', 'utils/utils.py', ['initialize_weights'], {'nn': nn}).initialize_weights
    patches = [
        ('[1024, 512, 256]', repr([dim, hidden, attention_dim])),
        ('[1024, 512, 384]', repr([dim, hidden, attention_dim])),
        ('self.temperature = torch.tensor([1.0])', "self.register_buffer('temperature', torch.tensor([1.0]))"),
        ('self.fixed_b = torch.tensor([5.], requires_grad=False)', "self.register_buffer('fixed_b', torch.tensor([5.]))"),
        ('self.prior_mu = torch.tensor([-5., 0.])', "self.register_buffer('prior_mu', torch.tensor([-5., 0.]))"),
        ('self.prior_logvar = torch.tensor([-1., 3.])', "self.register_buffer('prior_logvar', torch.tensor([-1., 3.]))"),
    ]
    if corrected_stride:
        patches.append(('coord[:, 0] + coord[:, 1] * (width // PATCH_SIZE)',
                        'coord[:, 0] + coord[:, 1] * (width // PATCH_SIZE + 1)'))
    return load_definitions('bayesmil', 'models/model_bmil.py',
        ['Attn_Net', 'Attn_Net_Gated', 'DAttn_Net_Gated', 'GaussianSmoothing',
         'probabilistic_MIL_Bayes_vis', 'probabilistic_MIL_Bayes_enc',
         'probabilistic_MIL_Bayes_spvis', 'get_ard_reg_vdo'],
        {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'numbers': numbers, 'np': np,
         'LinearVDO': vdo.LinearVDO, 'Conv2dVDO': vdo.Conv2dVDO,
         'initialize_weights': init, 'PATCH_SIZE': patch_size}, replacements=patches)


def canvas_shape(row):
    values = []
    for name in ('width_level0', 'height_level0'):
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('APCRF requires actual width_level0 and height_level0 in slides.csv') from e
        if not math.isfinite(value) or value <= 0 or int(value) != value:
            raise ValueError('Slide canvas dimensions must be positive integer level-0 pixels')
        values.append(int(value))
    return tuple(values)


def validate_canvas(coords, width, height, patch_size, max_cells):
    points = checked_points(coords)
    if not np.equal(points, np.round(points)).all() or (points < 0).any():
        raise ValueError('APCRF requires nonnegative integer top-left coordinates')
    if (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
        raise ValueError('Patch coordinates lie outside the declared WSI canvas')
    grid = np.floor_divide(points, patch_size).astype(np.int64)
    if len(np.unique(grid, axis=0)) != len(grid):
        raise ValueError('Multiple patches occupy the same APCRF raster cell')
    if (width//patch_size+1)*(height//patch_size+1) > max_cells:
        raise ValueError('APCRF full-slide raster exceeds max_grid_cells')
    return points.astype(np.int64)


def check_bayes_bundle(bundle):
    meta = spatial_metadata(bundle)
    if meta['unit'] != 'level0_pixels' or meta['origin'] != 'top_left' or int(meta['patch_size']) != meta['patch_size']:
        raise ValueError('APCRF requires integer level0_pixels patch_size and top_left coordinates')
    for row in bundle.rows.to_dict('records'):
        canvas_shape(row)
    return int(meta['patch_size'])


class BayesMIL(nn.Module):
    def __init__(self, dim, classes, profile='bayesmil_vis', hidden=512, attention_dim=256,
                 dropout=True, model_kl_weight=1e-8, data_kl_weight=1e-6,
                 mc_samples=16, inference_seed=7919, weighted_sampling=False,
                 patch_size=256, max_grid_cells=4000000, plateau_patience=10, plateau_factor=.1):
        super().__init__()
        if classes != 2 or profile not in BAYES_METHODS:
            raise ValueError('Original Bayes-MIL priors and head require a binary classification profile')
        if min(dim, hidden, attention_dim, mc_samples, patch_size, max_grid_cells) < 1:
            raise ValueError('Bayes-MIL dimensions and sampling budgets must be positive')
        if model_kl_weight <= 0 or (profile != 'bayesmil_vis' and data_kl_weight <= 0):
            raise ValueError('Bayes-MIL requires active weight and applicable patch regularizers')
        self.author = load_bayes(dim, hidden, attention_dim, patch_size)
        cls = {'bayesmil_vis': self.author.probabilistic_MIL_Bayes_vis,
               'bayesmil_sdpr': self.author.probabilistic_MIL_Bayes_enc,
               'bayesmil_apcrf': self.author.probabilistic_MIL_Bayes_spvis}[profile]
        self.network = cls(dropout=dropout, n_classes=classes)
        self.profile, self.model_kl_weight, self.data_kl_weight = profile, model_kl_weight, data_kl_weight
        self.mc_samples, self.inference_seed = mc_samples, inference_seed
        self.weighted_sampling, self.patch_size, self.max_grid_cells = weighted_sampling, patch_size, max_grid_cells
        self.plateau_patience, self.plateau_factor = plateau_patience, plateau_factor
        self.scheduler, self.scheduler_state = None, None

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            patience=self.plateau_patience, factor=self.plateau_factor)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def after_validation(self, metrics):
        # PULS uses the declared evaluation unit and the MC mean probability.
        self.scheduler.step(metrics['log_loss'])

    def get_extra_state(self):
        return {'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def sample(self, x, coords=None, label=None, canvas=None):
        if self.profile == 'bayesmil_vis':
            result = self.network(x, validation=label is None)
        elif self.profile == 'bayesmil_sdpr':
            result = self.network(x, slide_label=label, validation=label is None)
        else:
            if canvas is None:
                raise ValueError('APCRF needs the real WSI canvas, not inferred feature extents')
            width, height = canvas
            points = validate_canvas(coords, width, height, self.patch_size, self.max_grid_cells)
            result = self.network(x, points, height, width, slide_label=label, validation=label is None)
        output = {'logits': result[0], 'probabilities': result[1], 'attention': result[-1]}
        if label is not None:
            weight_kl = self.author.get_ard_reg_vdo(self.network)
            patch_regularizer = x.new_zeros(()) if self.profile == 'bayesmil_vis' else result[3].mean()
            output.update(weight_kl=weight_kl, patch_regularizer=patch_regularizer,
                total_loss=F.cross_entropy(result[0], label)+self.model_kl_weight*weight_kl+self.data_kl_weight*patch_regularizer)
        return output

    def forward(self, x, coords=None, label=None, canvas=None):
        if label is not None:
            return self.sample(x, coords, label, canvas)
        # Reproducible posterior prediction that leaves subsequent training RNG intact.
        with inference_rng(x.device, self.inference_seed):
            results = [self.sample(x, coords, None, canvas) for _ in range(self.mc_samples)]
        probabilities = torch.stack([r['probabilities'] for r in results])
        mean = probabilities.mean(0)
        entropy = lambda p: -(p*torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))).sum(-1)
        total = entropy(mean)
        aleatoric = entropy(probabilities).mean(0)
        return {'logits': torch.stack([r['logits'] for r in results]).mean(0), 'probabilities': mean,
                'attention': torch.stack([r['attention'] for r in results]).mean(0),
                'predictive_entropy': total, 'expected_entropy': aleatoric,
                'mutual_information': total-aleatoric}

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        if self.profile == 'bayesmil_apcrf' and len(rows) != 1:
            raise ValueError('APCRF cannot combine independent WSI coordinate frames')
        x, coords = load_unit(bundle, rows, device, config.get('max_instances'), train)
        canvas = canvas_shape(rows[0]) if self.profile == 'bayesmil_apcrf' else None
        label = torch.tensor([rows[0]['target']], device=device) if train else None
        return self(x, coords, label, canvas)

    def protocol(self):
        return {'profile': self.profile, 'model_kl_weight': self.model_kl_weight,
                'data_kl_weight': self.data_kl_weight, 'patch_regularizer': 'literal author log-variance formula',
                'spatial_stride': 'allocated grid width, corrected from author stride',
                'patch_size': self.patch_size, 'mc_samples': self.mc_samples,
                'inference': 'mean probabilities; fixed isolated RNG, dropout disabled in eval',
                'weighted_sampling': self.weighted_sampling,
                'scheduler': 'ReduceLROnPlateau on declared-unit validation MC probability log_loss',
                'zero_variance_fix': 'exact zero forward with zero subgradient; positive variances unchanged',
                'max_grid_cells': self.max_grid_cells}


def bayes_epoch(model, bundle, groups, optimizer, device, config):
    if model.weighted_sampling:
        targets = np.asarray([rows[0]['target'] for rows in groups])
        counts = np.bincount(targets)
        indices = torch.multinomial(torch.as_tensor(1/counts[targets]), len(groups), replacement=True)
        groups = [groups[i] for i in indices.tolist()]
    losses = []
    for rows in groups:
        result = model.forward_unit(bundle, rows, device, config, train=True)
        loss = result['total_loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite Bayes-MIL variational objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append({k: float(result[v].detach()) for k, v in
                       [('loss', 'total_loss'), ('weight_kl', 'weight_kl'), ('patch_regularizer', 'patch_regularizer')]})
    return losses

"""SGPMIL author variational GP attention and inducing-distribution KL training."""
import torch
from torch import nn

from .stochastic import inference_rng
from .upstream import load_single


class SparseGPMIL(nn.Module):
    optimizer_name = 'adamw'

    def __init__(self, dim, classes, hidden=768, hidden1=384, hidden2=256, attention_dim=64,
                 inducing_points=80, mc_samples=30, sampling='var', prior_mean=0., prior_variance=1.,
                 jitter=1e-4, kl_factor=1., attention_activation='sigmoid', post_activation='sigmoid',
                 evaluation_seed=2026):
        super().__init__()
        if min(hidden, attention_dim, inducing_points) < 1 or min(hidden1, hidden2) < 0 or mc_samples < 2:
            raise ValueError('GP dimensions/inducing points must be positive and mc_samples >= 2')
        if sampling not in {'var', 'cov'} or prior_variance <= 0 or jitter <= 0 or kl_factor <= 0:
            raise ValueError('Invalid GP sampling, prior, jitter or KL weight')
        if attention_activation not in {'sigmoid', 'tanh'} or post_activation not in {'sigmoid', 'softmax'}:
            raise ValueError('Unsupported SGPMIL attention activation')
        cfg = {'data': {'data_dims': dim, 'num_classes': classes}, 'model': {
            'hidden_layer_size_0': hidden, 'hidden_layer_size_1': hidden1, 'hidden_layer_size_2': hidden2,
            'hidden_layer_size_att': attention_dim, 'attention': 'sgpmil',
            'attention_multiplication_type': 'elementwise', 'attn_hl_activation': attention_activation,
            'post_attention_activation': post_activation, 'inducing_points': inducing_points,
            'mc_samples': mc_samples, 'sampling': sampling, 'prior_mean': prior_mean,
            'prior_variance': prior_variance, 'jitter': jitter}}
        self.network = load_single('sgpmil', 'src/gp_models.py').SGPMIL(cfg)
        self.kl_factor, self.evaluation_seed = kl_factor, evaluation_seed
        self.register_buffer('training_units', torch.tensor(0, dtype=torch.long))

    def prepare(self, bundle, seed, device):
        from .engine import units
        self.training_units.fill_(len(units(bundle, 'train')))

    def forward(self, x, coords=None, label=None):
        if self.training:
            out = self.network(x.unsqueeze(0))
        else:
            with inference_rng(x.device, self.evaluation_seed):
                out = self.network(x.unsqueeze(0))
        result = {'logits': out['logits'], 'probabilities': out['y_hat'], 'attention': out['attention'],
                  'predictive_std': out['y_hat_se']}
        if self.training:
            if self.training_units < 1:
                raise RuntimeError('SGPMIL KL scaling requires the number of training bags; call prepare')
            kl = self.network.sgp.variational_strategy.kl_divergence().sum()
            result['regularization_loss'] = self.kl_factor * kl / self.training_units
        return result

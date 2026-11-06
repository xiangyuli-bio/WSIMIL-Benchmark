"""SCMIL sparse contexts and its continuous softplus-Gaussian mixture likelihood."""
import math

import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions


class CPUClusterLabels:
    def __init__(self, labels):
        self.labels = labels

    def get(self):
        return self.labels


class CPUKMeans:
    """Explicit CPU backend; same L2 problem, not a cuML numerical-equivalence claim."""
    def __init__(self, **options):
        self.options = options

    def fit_predict(self, x):
        values = x.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('SCMIL clustering received nonfinite normalized features/coordinates')
        return CPUClusterLabels(KMeans(n_init=1, **self.options).fit_predict(values))


def load_scmil(backend='sklearn'):
    if backend == 'sklearn':
        cluster = CPUKMeans
    elif backend == 'cuml':
        from cuml import KMeans as cluster
    else:
        raise ValueError('SCMIL clustering backend must be explicitly sklearn or cuml')
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math,
             'rearrange': rearrange, 'KMeans': cluster}
    scope['SoftFilterLayer'] = load_definitions('scmil', 'models/SoftFilter.py',
                                              ['SoftFilterLayer'], scope).SoftFilterLayer
    scope['AMIL_layer'] = load_definitions('scmil', 'models/AMIL.py', ['AMIL_layer'], scope).AMIL_layer
    model = load_definitions('scmil', 'models/SCMIL.py', ['SelfAttention', 'ClusterLocalAttention', 'SCMIL'], scope)
    # Exact algebra for the author's epsilon-smoothed inverse; finite at long follow-up times.
    mdn = load_definitions('scmil', 'models/MDN.py',
        ['safe_inverse_softplus', 'inverse_softplus_grad', 'logsumexp', 'MLP', 'SurvivalMDN'], scope,
        replacements=[('torch.log(torch.exp(x) - 1 + 1e-6)', 'x + torch.log(-torch.expm1(-x) + 1e-6*torch.exp(-x))'),
                      ('torch.exp(x) / (torch.exp(x) - 1 + 1e-6)', '1 / (-torch.expm1(-x) + 1e-6*torch.exp(-x))')])
    return model, mdn


def spatial_patient_bags(bundle, rows, device, max_instances=None, train=False):
    bags = [bundle.load(row, require_coords=True) for row in rows]
    total = sum(len(x) for x, _ in bags)
    if max_instances and total > max_instances:
        if max_instances < len(rows):
            raise ValueError('SCMIL patient budget must retain at least one patch per slide')
        # Allocate at least one per slide, then distribute remaining budget proportionally.
        sizes = np.array([len(x) for x, _ in bags]); capacities = sizes-1
        fractional = capacities/capacities.sum()*(max_instances-len(bags))
        counts = 1+np.floor(fractional).astype(int)
        remainder = max_instances-counts.sum()
        counts[np.argsort(-(fractional-np.floor(fractional)), kind='stable')[:remainder]] += 1
        sampled = []
        for (x, c), count in zip(bags, counts):
            idx = torch.randperm(len(x))[:count].sort().values if train else torch.linspace(0, len(x)-1, int(count)).long()
            sampled.append((x[idx], c[idx]))
        bags = sampled
    return [(x.to(device), coords.to(device)) for x, coords in bags]


class SparseContextSurvival(nn.Module):
    survival_objective = 'continuous_mdn'

    def __init__(self, dim, hidden=256, components=100, cluster_size=64, feature_weight=.8,
                 dropout=.25, clustering_backend='sklearn', time_unit='months',
                 time_divisor=None, curve_points=64, accumulation_steps=1):
        super().__init__()
        if dim < 8 or dim % 8:
            raise ValueError('Author SCMIL eight-head backbone requires FM dimension divisible by 8')
        if any(not isinstance(v, int) or v < 1 for v in (hidden, components, cluster_size, accumulation_steps)):
            raise ValueError('SCMIL dimensions and budgets must be positive integers')
        if not 0 <= feature_weight <= 1 or not 0 <= dropout < 1 or curve_points < 2:
            raise ValueError('Invalid SCMIL feature weighting, dropout or curve grid')
        if time_unit not in {'months', 'days'}:
            raise ValueError('SCMIL task must declare time_unit: months or days')
        if time_unit == 'days' and (time_divisor is None or not math.isfinite(time_divisor) or time_divisor <= 0):
            raise ValueError('SCMIL day-valued input requires an explicit positive days-per-model-month time_divisor')
        if time_unit == 'months' and time_divisor not in {None, 1}:
            raise ValueError('Month-valued SCMIL input uses time_divisor=1')
        model, mdn = load_scmil(clustering_backend)
        backbone = model.SCMIL(input_size=dim, hidden_size=dim, deep=1, cluster_size=cluster_size,
            feature_weight=feature_weight, as_backbone=True, dropout_rate=dropout, with_softfilter=True)
        # Author constructor hardcodes softfilter hidden 256 and attention hidden 256; retain them.
        self.network = mdn.SurvivalMDN(backbone=backbone, input_size=dim, hidden_size=hidden,
            K=components, param_share=['mu', 'sigma'])
        self.time_unit, self.time_divisor = time_unit, float(time_divisor or 1)
        self.clustering_backend, self.feature_weight = clustering_backend, feature_weight
        self.accumulation_steps, self.l1_weight = accumulation_steps, 0.
        self.register_buffer('curve_times', torch.zeros(curve_points))
        self.register_buffer('prepared', torch.tensor(False))

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        if bundle.config.get('time_unit') != self.time_unit:
            raise ValueError('SCMIL requires the declared task time_unit to match its model time unit')
        maximum = bundle.rows[bundle.rows.subset == 'train'].drop_duplicates('global_patient_id').time.max()
        self.curve_times.copy_(torch.linspace(0, maximum/self.time_divisor, len(self.curve_times), device=device))
        self.prepared.fill_(True)

    def slide_parameters(self, x, coords):
        if x.ndim != 2 or not len(x) or coords is None or coords.shape != (len(x), 2):
            raise ValueError('SCMIL requires a nonempty per-slide [N,D] bag and real aligned [N,2] coordinates')
        if not torch.isfinite(x).all() or not torch.isfinite(coords).all():
            raise ValueError('SCMIL features and coordinates must be finite')
        # The source filters first. Only a selected group of size >1 enters clustering.
        # A zero std is undefined there; validate inside the backend, after source normalization.
        params, _ = self.network(x[None], coords=coords[None])
        return params

    def forward(self, bags):
        if not self.prepared:
            raise RuntimeError('SCMIL requires a training-derived evaluation grid')
        if not bags:
            raise ValueError('SCMIL needs at least one slide per patient')
        params = [self.slide_parameters(x, c) for x, c in bags]
        curves = [1-self.network.cdf(p, self.curve_times[None]) for p in params]
        survival = torch.stack(curves).mean(0).clamp(0, 1)
        risk = -torch.trapezoid(survival, self.curve_times, dim=1).reshape(1, 1)
        return {'logits': risk, 'survival': survival, 'params': params}

    def objective(self, result, time, event):
        time = self.curve_times.new_tensor([time/self.time_divisor])
        if not torch.isfinite(time).all() or (time <= 0).any() or event not in {0, 1}:
            raise ValueError('SCMIL needs positive finite times and binary event indicators')
        pdfs = torch.stack([self.network.log_prob(p, time).exp().reshape(1) for p in result['params']])
        survivals = torch.stack([1-self.network.cdf(p, time) for p in result['params']])
        # Patient-level equal mixture of slide distributions. One slide equals author MDNML.
        pdf, survival = pdfs.mean(0), survivals.mean(0).clamp(0, 1)
        loss = -event*torch.log((pdf+1e-6)/(survival+1e-6))-torch.log(survival+1e-6)
        return loss.mean()

    def forward_unit(self, bundle, rows, device, config, train=False):
        return self(spatial_patient_bags(bundle, rows, device, config['max_instances'], train))

    def training_losses(self, bundle, rows, device, config):
        out = self.forward_unit(bundle, rows, device, config, True)
        yield self.objective(out, rows[0]['time'], rows[0]['event'])

    def make_optimizer(self, lr, weight_decay):
        grouping = load_definitions('scmil', 'optimizers/optim_factory.py', ['add_weight_decay'], {}).add_weight_decay
        return torch.optim.Adam(grouping(self, weight_decay), lr=lr)

    def prediction_fields(self, result, row):
        return {'survival_loss': float(self.objective(result, row['time'], row['event'])),
                **{f'survival_time_{i}': float(t) for i, t in enumerate(self.curve_times)}}

    def validation_metrics(self, predictions):
        return {'survival_loss': float(predictions.survival_loss.mean())}

    def protocol(self):
        return {'objective': 'author epsilon-smoothed continuous MDN censor likelihood',
                'distribution': 'softplus-transformed Gaussian mixture; shared trainable mu/sigma registers',
                'components': self.network.K, 'softfilter': True, 'context_depth': 1,
                'cluster_size': self.network.backbone.cluster_size, 'feature_weight': self.feature_weight,
                'clustering_backend': self.clustering_backend,
                'clustering': 'per-slide filtered contexts, global scalar normalization, L2 KMeans max_iter=5 seed=1',
                'time_unit': self.time_unit, 'model_time_unit': 'months', 'time_divisor': self.time_divisor,
                'curve_times_months': self.curve_times.tolist(),
                'risk': 'negative trapezoidal restricted mean survival over training-derived grid',
                'patient_rule': 'equal mixture of independently processed slide distributions; no merged coordinate frames',
                'optimizer': 'Adam with author bias/1D zero-decay grouping',
                'numeric_repair': 'algebraically stable epsilon-smoothed inverse softplus and Jacobian',
                'unused_author_modules': ['backbone.fc1', 'backbone.attens.0.atten'],
                'evaluation_scope': 'PULS patient Harrell C-index and censor likelihood; not author IPCW/Brier protocol'}

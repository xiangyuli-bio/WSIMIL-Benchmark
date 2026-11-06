"""HE2RNA shared tile MLP, random top-k training and gene-expression regression."""
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans
from torch import nn

from .io import digest_file, digest_object
from .upstream import load_definitions


def load_he2rna():
    return load_definitions('he2rna', 'model.py', ['HE2RNA'], {'np': np, 'torch': torch, 'nn': nn})


def spatial_supertiles(x, coords, count=100, seed=2026):
    """Cluster measured coordinates, then average features; no feature-derived spatial proxy."""
    if (coords is None or coords.shape != (len(x), 2) or not torch.isfinite(coords).all() or
            x.ndim != 2 or not torch.isfinite(x).all()):
        raise ValueError('HE2RNA supertiles require finite features and aligned physical coordinates')
    if len(x) < count or len(torch.unique(coords, dim=0)) < count:
        raise ValueError('Not enough distinct physical patches for the requested HE2RNA supertile count')
    a, c = x.detach().cpu().numpy(), coords.detach().cpu().numpy()
    key = digest_object({'features': hashlib.sha256(a.tobytes()).hexdigest(), 'shape': list(a.shape),
                         'coords': hashlib.sha256(c.tobytes()).hexdigest(), 'count': count, 'seed': seed,
                         'sklearn': sklearn.__version__, 'code': digest_file(Path(__file__))})
    cache = Path(os.environ.get('MILPLUS_HE2RNA_CACHE', Path.home()/'.cache/milplus/he2rna_supertiles'))
    path = cache/(key+'.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            value, checksum = stored['features'], str(stored['sha256'])
        if value.shape != (count, x.shape[1]) or hashlib.sha256(value.tobytes()).hexdigest() != checksum:
            raise ValueError('HE2RNA supertile cache checksum/shape mismatch')
        return torch.from_numpy(value).to(x)
    labels = KMeans(n_clusters=count, random_state=seed, n_init=10, algorithm='lloyd').fit_predict(c)
    value = np.stack([a[labels == i].mean(0) for i in range(count)])
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite HE2RNA supertile features')
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as stream:
        tmp = Path(stream.name)
        np.savez(stream, features=value, sha256=hashlib.sha256(value.tobytes()).hexdigest())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return torch.from_numpy(value).to(x)


class HE2RNARegression(nn.Module):
    def __init__(self, dim, targets, layers=None, ks=None, dropout=.25, batch_size=16,
                 input_mode='supertiles', supertile_count=100, cluster_seed=2026):
        super().__init__()
        layers = [1024, 1024] if layers is None else layers
        ks = [1, 2, 5, 10, 20, 50, 100] if ks is None else ks
        if not isinstance(layers, list) or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in layers):
            raise ValueError('HE2RNA layers must be an explicit list of positive widths')
        if (not isinstance(ks, list) or not ks or len(ks) != len(set(ks)) or
                any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in ks)):
            raise ValueError('HE2RNA ks must be unique positive integer pooling counts')
        if not 0 <= dropout < 1 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Invalid HE2RNA dropout or batch_size')
        if input_mode not in {'tiles', 'supertiles'}:
            raise ValueError('HE2RNA input_mode must be tiles or supertiles')
        if not isinstance(supertile_count, int) or supertile_count < max(ks):
            raise ValueError('HE2RNA supertile_count must be at least max(ks)')
        if not isinstance(cluster_seed, int) or not 0 <= cluster_seed < 2**32:
            raise ValueError('HE2RNA cluster_seed must be a uint32 integer')
        if targets is None or (targets.values.to_numpy() < 0).any():
            raise ValueError('HE2RNA requires nonnegative transformed expression targets')
        self.outputs, self.transform = targets.columns, targets.transform
        self.dim, self.ks, self.batch_size = dim, ks, batch_size
        self.input_mode, self.supertile_count, self.cluster_seed = input_mode, supertile_count, cluster_seed
        self.network = load_he2rna().HE2RNA(input_dim=dim, output_dim=len(self.outputs), layers=layers,
                                           ks=ks, dropout=dropout, device='cpu')
        self.register_buffer('prepared', torch.tensor(False))

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        from .engine import units
        groups = units(bundle, 'train')
        mean = torch.stack([bundle.regression_targets.tensor(rows[0]['global_patient_id']) for rows in groups]).mean(0)
        self.network.layers[-1].bias.copy_(mean.to(device))
        self.prepared.fill_(True)
        # Check the pooling contract for all declared inputs before any optimization.
        for subset in ['train', 'val', 'test']:
            for rows in units(bundle, subset):
                x = self.load_features(bundle, rows)
                if len(x) < max(self.ks):
                    raise ValueError('HE2RNA max(ks) exceeds the available instances; set the protocol explicitly')

    def load_features(self, bundle, rows):
        parts = []
        for row in rows:
            x, coords = bundle.load(row, require_coords=self.input_mode == 'supertiles')
            if self.input_mode == 'supertiles':
                x = spatial_supertiles(x, coords, self.supertile_count, self.cluster_seed)
            parts.append(x)
        return torch.cat(parts)

    def training_k(self):
        return int(np.random.choice(self.ks))

    def forward(self, x, k=None):
        return self.forward_batch([x], k=k)

    def forward_batch(self, bags, k=None):
        """Run the author's actual batched Conv1d, retaining measured bag lengths.

        Accumulating separate per-bag backwards changes FP32 reductions and the
        dropout stream. Padding is an input layout only: an explicit length mask
        excludes padded scores before top-k while retaining signed and zero
        embeddings from real patches.
        """
        if not isinstance(bags, (list, tuple)) or not bags:
            raise ValueError('HE2RNA requires a nonempty batch of feature bags')
        for x in bags:
            if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.dim or
                    len(x) < max(self.ks) or not torch.isfinite(x).all()):
                raise ValueError('HE2RNA requires finite [N,D] with N >= max(ks)')
            if x.device != bags[0].device or x.dtype != bags[0].dtype:
                raise ValueError('HE2RNA batch features require one device and dtype')
        if k is not None and k not in self.ks:
            raise ValueError('HE2RNA pooling count is outside the declared ks')
        choices = [self.training_k() if k is None else k] if self.training else self.ks
        lengths = torch.tensor([len(x) for x in bags], device=bags[0].device)
        batch = nn.utils.rnn.pad_sequence(bags, batch_first=True).transpose(1, 2).contiguous()
        valid = torch.arange(batch.shape[2], device=batch.device)[None, None, :] < lengths[:, None, None]
        # At eval recompute conv for each k as the author does (dropout disabled).
        raw = sum(self.network.conv(batch).masked_fill(~valid, -torch.inf)
                  .topk(v, dim=2).values.sum(2)/v/len(choices) for v in choices)
        return {'raw': raw, 'prediction': raw.relu()}

    def forward_unit(self, bundle, rows, device, config, train=False, k=None):
        return self.forward_batch_units(bundle, [rows], device, config, train=train, k=k)

    def forward_batch_units(self, bundle, groups, device, config, train=False, k=None):
        if config['max_instances'] is not None:
            raise ValueError('HE2RNA defines its own tile/supertile protocol; max_instances must be null')
        bags = [self.load_features(bundle, rows).to(device) for rows in groups]
        return self.forward_batch(bags, k=k)

    def protocol(self):
        return {'source': 'pinned Owkin HE2RNA', 'task': 'gene expression regression; RNA is target, never input',
                'outputs': self.outputs, 'target_transform': self.transform,
                'objective': 'MSE on raw output; final bias initialized from training-unit target means',
                'pooling': 'one random k per training batch; average all ks at inference; ReLU prediction only',
                'ks': self.ks, 'batch_size': self.batch_size, 'input_mode': self.input_mode,
                'supertile_count': self.supertile_count, 'cluster_seed': self.cluster_seed,
                'supertiles': 'per-slide coordinate KMeans then mean features; sklearn Lloyd n_init=10, not libKMCUDA',
                'padding': 'actual batched Conv1d with explicit measured-length mask before top-k; signed and zero real rows remain valid',
                'batch_update': 'one shared random k, one batch/gene mean MSE and backward per optimizer step',
                'scope': 'expression prediction training; no MSI transfer classifier or significance-testing reproduction',
                'selection': 'validation project/gene mean Pearson; undefined pairs explicitly reported',
                'multi_slide': 'one target per global patient; configured slide/patient training and evaluation units'}

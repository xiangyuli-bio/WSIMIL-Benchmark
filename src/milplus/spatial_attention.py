"""Author SmMIL and PSA-MIL: physical graphs, smoothing and learned distance priors."""
import math
import sys
import types

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, package, pinned_text, verified_source

SM_METHODS = {'smmil_early', 'smmil_mid', 'smmil_late', 'smmil_transformer'}


def spatial_metadata(bundle):
    meta = bundle.encoder.get('spatial', {})
    if (meta.get('unit') not in {'level0_pixels', 'micrometers', 'tile_grid'}
            or meta.get('origin') not in {'top_left', 'center'} or not meta.get('provenance')):
        raise ValueError('Spatial methods require encoder.spatial unit, origin, patch_size and provenance')
    scale = meta.get('patch_size', 0)
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Spatial patch_size must be positive, in the coordinate unit')
    if meta['unit'] == 'tile_grid' and scale != 1:
        raise ValueError('tile_grid coordinates require patch_size: 1')
    return meta


def checked_points(coords):
    if coords is None or coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError('Spatial methods require nonempty coordinates [N,2]')
    points = coords.detach().cpu().numpy()
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Coordinates must be finite and distinct within each slide')
    return points


def smooth_adjacency(x, coords, patch_size, feature_weighted=True):
    points = checked_points(coords)
    if len(x) != len(points):
        raise ValueError('Feature/coordinate row count mismatch')
    n = len(x)
    if n == 1:
        return torch.sparse_coo_tensor(torch.tensor([[0], [0]], device=x.device),
                                      x.new_ones(1), (1, 1)).coalesce()
    pairs = cKDTree(points).query_pairs(np.sqrt(2)*patch_size, output_type='ndarray')
    edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    values = np.ones(edges.shape[1], dtype=np.float64)
    if feature_weighted and edges.shape[1]:
        raw = x.detach().cpu().numpy()
        values = np.exp(-np.linalg.norm(raw[edges[0]]-raw[edges[1]], axis=1)/x.shape[1])
    degrees = np.bincount(edges[1], weights=values, minlength=n).astype(np.float32)
    inv = np.zeros(n, dtype=np.float32)
    np.power(degrees, -.5, out=inv, where=degrees > 0)
    values = inv[edges[0]]*values*inv[edges[1]]
    return torch.sparse_coo_tensor(torch.as_tensor(edges, device=x.device),
        torch.as_tensor(values, dtype=x.dtype, device=x.device), (n, n)).coalesce()


def load_smmil():
    smooth = load_definitions('smmil', 'code/models/modules/Sm.py', ['ApproxSm', 'ExactSm'], {'torch': torch})
    scope = {'torch': torch, 'ApproxSm': smooth.ApproxSm, 'ExactSm': smooth.ExactSm}
    pool = load_definitions('smmil', 'code/models/modules/MILAttentionPool.py', ['MILAttentionPool'], scope,
        replacements=(('ExactSm(alpha=self.alpha)', 'ExactSm(alpha=self.sm_alpha)'),
                      ('exp_f = torch.exp(f)*mask', 'exp_f = torch.exp(f.masked_fill(~mask.bool(), -torch.inf) - f.masked_fill(~mask.bool(), -torch.inf).amax(1, keepdim=True))')))
    from torch.nn.attention import SDPBackend
    scope['SDP_BACKEND'] = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    transformer = load_definitions('smmil', 'code/models/modules/MILTransformer.py',
        ['MultiheadSelfAttention', 'Identity', 'TransformerEncoderLayer', 'TransformerEncoder'], scope,
        replacements=(('query, key, value, mask, self.dropout)',
                       'query, key, value, mask, self.dropout if self.training else 0.0)'),))
    return load_definitions('smmil', 'code/models/DeepMILAttModel.py', ['DeepMILAttModel'],
        {'torch': torch, 'TransformerEncoder': transformer.TransformerEncoder,
         'MILAttentionPool': pool.MILAttentionPool})


def spatial_bags(bundle, groups, device, cfg, train):
    from .engine import load_unit
    if any(len(rows) != 1 for rows in groups):
        raise ValueError('Spatial classification uses individual slide bags; do not merge coordinate frames')
    bags = [load_unit(bundle, rows, device, cfg.get('max_instances'), train) for rows in groups]
    for x, coords in bags:
        if len(checked_points(coords)) != len(x):
            raise ValueError('Coordinate/feature alignment mismatch')
    return bags


def padded_bags(bags):
    if not bags:
        raise ValueError('At least one spatial bag is required')
    for x, c in bags:
        if x.ndim != 2 or not torch.isfinite(x).all() or len(checked_points(c)) != len(x):
            raise ValueError('Spatial bags require finite, aligned feature/coordinate matrices')
    length = max(len(x) for x, _ in bags)
    dim = bags[0][0].shape[1]
    features = bags[0][0].new_zeros(len(bags), length, dim)
    coords = bags[0][0].new_zeros(len(bags), length, 2)
    mask = torch.zeros(len(bags), length, dtype=torch.bool, device=features.device)
    for i, (x, c) in enumerate(bags):
        features[i, :len(x)], coords[i, :len(x)], mask[i, :len(x)] = x, c, True
    return features, coords, mask


class SmoothMIL(nn.Module):
    def __init__(self, dim, classes, patch_size, profile='smmil_early', hidden=512, attention_dim=128,
                 alpha='trainable', mode='approx', steps=10, spectral_norm=False, feature_weighted=True,
                 batch_size=1, balance_loss=False, transformer_heads=8, transformer_layers=2,
                 transformer_dropout=0., max_dense_bytes=1073741824):
        super().__init__()
        if classes != 2 or profile not in SM_METHODS:
            raise ValueError('SmMIL author head requires binary classification and an explicit Sm profile')
        if alpha != 'trainable' and (not isinstance(alpha, (float, int)) or not 0 < alpha < 1):
            raise ValueError('Sm alpha must be trainable or strictly between zero and one')
        if min(batch_size, steps, hidden, attention_dim) < 1 or mode not in {'approx', 'exact'}:
            raise ValueError('Invalid SmMIL dimensions, batch size, steps or smoothing mode')
        if profile == 'smmil_transformer' and hidden % transformer_heads:
            raise ValueError('Transformer width must divide evenly across heads')
        if alpha != 'trainable':
            alpha = float(alpha)
        where = profile.rsplit('_', 1)[-1] if profile != 'smmil_transformer' else 'early'
        transformer = {} if profile != 'smmil_transformer' else {
            'att_dim': hidden, 'num_heads': transformer_heads, 'num_layers': transformer_layers,
            'use_ff': False, 'dropout': transformer_dropout, 'use_sm': True,
            'sm_alpha': alpha, 'sm_mode': mode, 'sm_steps': steps}
        self.network = load_smmil().DeepMILAttModel(input_dim=dim, emb_dim=hidden,
            transformer_encoder_kwargs=transformer,
            pool_kwargs={'att_dim': attention_dim, 'sm_alpha': alpha, 'sm_mode': mode,
                         'sm_where': where, 'sm_steps': steps, 'spectral_norm': spectral_norm})
        self.patch_size, self.feature_weighted, self.batch_size = patch_size, feature_weighted, batch_size
        self.balance_loss, self.mode, self.max_dense_bytes = balance_loss, mode, max_dense_bytes
        self.register_buffer('pos_weight', torch.ones(1))
        self.epoch = 0

    def prepare(self, bundle, seed, device):
        labels = bundle.rows.query("subset == 'train'").target.to_numpy()
        if self.balance_loss:
            self.pos_weight.fill_(float((labels == 0).sum()/(labels == 1).sum()))

    def start_epoch(self, epoch):
        self.epoch = epoch

    def batch(self, bags):
        features, _, mask = padded_bags(bags)
        indices, values = [], []
        for i, (x, coords) in enumerate(bags):
            adj = smooth_adjacency(x, coords, self.patch_size, self.feature_weighted)
            indices.append(torch.cat([torch.full((1, adj._nnz()), i, device=x.device, dtype=torch.long), adj.indices()]))
            values.append(adj.values())
        length = features.shape[1]
        adj = torch.sparse_coo_tensor(torch.cat(indices, 1), torch.cat(values),
                                      (len(bags), length, length)).coalesce()
        if self.mode == 'exact':
            if len(bags)*length*length*features.element_size() > self.max_dense_bytes:
                raise ValueError('Exact Sm exceeds max_dense_bytes; choose an explicit feasible protocol')
            adj = adj.to_dense()
        logit, attention = self.network(features, adj, mask, return_att=True)
        return {'logits': torch.stack([torch.zeros_like(logit), logit], 1),
                'binary_logit': logit, 'attention': attention}

    def forward(self, x, coords=None, label=None):
        out = self.batch([(x, coords)])
        if label is not None:
            out['total_loss'] = self.objective(out, label)
        return out

    def objective(self, result, labels):
        return F.binary_cross_entropy_with_logits(result['binary_logit'], labels.float(), pos_weight=self.pos_weight)

    def forward_unit(self, bundle, rows, device, config, train=False):
        return self.batch(spatial_bags(bundle, [rows], device, config, train))

    def protocol(self):
        return {'graph': 'sqrt(2)*patch_size radius, feature weights exp(-L2/D), symmetric degree normalization',
                'patch_size': self.patch_size, 'feature_weighted': self.feature_weighted, 'mode': self.mode,
                'batch_size': self.batch_size, 'pos_weight': self.pos_weight.item(),
                'scheduler': 'epoch linear warmup 0.1 to 1 over max(floor(epochs*0.1),5)',
                'source_behavior': 'ApproxSm alpha coefficient convention preserved; see SPATIAL.md'}


def load_psamil():
    root, sha = verified_source('psamil')
    name = '_milplus_psamil_' + sha
    if name+'.psa' in sys.modules:
        return sys.modules[name+'.psa'], sys.modules[name+'.schedule']
    package(name, root/'psa_mil')
    for key in ('decay', 'diversity', 'attention', 'schedule', 'psa'):
        relative = f'psa_mil/{"training" if key == "schedule" else "models"}/{key}.py'
        code = pinned_text(root, sha, relative).replace('from psa_mil.models.', 'from .')
        module = types.ModuleType(name+'.'+key)
        module.__file__, module.__package__ = str(root/relative), name
        sys.modules[module.__name__] = module
        try:
            exec(compile(code, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified author Git blob
        except BaseException:
            del sys.modules[module.__name__]
            raise
    return sys.modules[name+'.psa'], sys.modules[name+'.schedule']


class ProbabilisticSpatialMIL(nn.Module):
    def __init__(self, dim, classes, patch_size, hidden=96, heads=3, depth=1, residual_layers=2,
                 pool='attention', dropout=0., attention_chunk=256, decay_type='Gaussian',
                 decay_clip=.001, min_local_k=1., max_local_k=25., init_k=7.,
                 diversity='gaussian_binning', alpha=.001, batch_size=8, class_balance=True,
                 decay_lr_scale=100., max_distance_bytes=4294967296):
        super().__init__()
        if min(hidden, depth, batch_size, attention_chunk) < 1 or heads < 2 or alpha <= 0 or decay_lr_scale <= 0:
            raise ValueError('PSA-MIL requires positive dimensions/budgets and an active multi-head diversity term')
        if not 0 < decay_clip < 1 or not 0 < min_local_k < max_local_k or not min_local_k+1 <= init_k <= max_local_k-1:
            raise ValueError('Invalid PSA-MIL decay threshold or radius range/initialization')
        author, _ = load_psamil()
        self.network = author.PSAMIL(embed_dim=dim, num_classes=classes, num_heads=heads, depth=depth,
            attn_dim=hidden, num_residual_layers=residual_layers, pool_type=pool, dropout=dropout,
            attention_chunk=attention_chunk, decay_type=decay_type, decay_clip=decay_clip,
            min_local_k=min_local_k, max_local_k=max_local_k, init_k=init_k, div_loss=diversity, alpha=alpha)
        self.patch_size, self.batch_size, self.class_balance = patch_size, batch_size, class_balance
        self.decay_lr_scale, self.max_distance_bytes = decay_lr_scale, max_distance_bytes
        self.register_buffer('class_weights', torch.ones(classes))
        self.epoch = 0

    def prepare(self, bundle, seed, device):
        if self.class_balance:
            counts = np.bincount(bundle.rows.query("subset == 'train'").target.to_numpy(), minlength=len(self.class_weights))
            self.class_weights.copy_(torch.as_tensor(1/counts, device=device))

    def start_epoch(self, epoch):
        self.epoch = epoch

    def make_optimizer(self, lr, weight_decay):
        decay, other = [], []
        for name, param in self.network.named_parameters():
            (decay if name.endswith('lambda_p') else other).append(param)
        return torch.optim.Adam([{'params': other}, {'params': decay, 'lr': lr*self.decay_lr_scale}],
                                lr=lr, weight_decay=weight_decay)

    def batch(self, bags):
        features, coords, mask = padded_bags(bags)
        if features.shape[0]*features.shape[1]**2*4 > self.max_distance_bytes:
            raise ValueError('PSA-MIL distance matrix exceeds max_distance_bytes; set an explicit tile budget')
        return {'logits': self.network(features, coords/self.patch_size, mask)}

    def forward(self, x, coords=None, label=None):
        out = self.batch([(x, coords)])
        if label is not None:
            out['total_loss'] = self.objective(out, label)
        return out

    def objective(self, result, labels):
        loss = F.cross_entropy(result['logits'], labels, reduction='none')
        weights = self.class_weights[labels]
        task_loss = (loss*weights).sum()/weights.sum()
        return task_loss-self.network.alpha*self.network.diversity_loss()

    def forward_unit(self, bundle, rows, device, config, train=False):
        return self.batch(spatial_bags(bundle, [rows], device, config, train))

    def protocol(self):
        return {'patch_size': self.patch_size, 'coordinate_conversion': 'coords/patch_size -> tile units',
                'batch_size': self.batch_size, 'class_weights': self.class_weights.cpu().tolist(),
                'diversity': self.network.div_loss, 'alpha': self.network.alpha, 'decay_lr_scale': self.decay_lr_scale,
                'author_revision': '2026-09-28 refactored classifier; not historical survival interface',
                'distance_memory': 'dense B*N*N distances, query chunking bounds attention working memory'}


def spatial_epoch(model, bundle, groups, optimizer, device, cfg):
    batches = [groups[i:i+model.batch_size] for i in range(0, len(groups), model.batch_size)]
    if isinstance(model, ProbabilisticSpatialMIL):
        schedule = load_psamil()[1].piecewise_linear
        total = len(batches)*cfg['epochs']
        lr = schedule([{'from': cfg['lr']*.01, 'to': cfg['lr'], 'steps': .1},
                       {'from': cfg['lr'], 'to': cfg['lr'], 'steps': .9},
                       {'from': cfg['lr'], 'to': cfg['lr']*.01, 'steps': -1}], total)
        wd = schedule([{'from': cfg['weight_decay']*.1, 'to': cfg['weight_decay'], 'steps': .1},
                       {'from': cfg['weight_decay'], 'to': cfg['weight_decay']*5, 'steps': .9},
                       {'from': cfg['weight_decay']*5, 'to': cfg['weight_decay']*10, 'steps': -1}], total)
    else:
        warmup = max(int(cfg['epochs']*.1), 5)
        for group in optimizer.param_groups:
            group['lr'] = cfg['lr']*(.1 + .9*min(model.epoch/warmup, 1.))
    losses = []
    for step, batch in enumerate(batches):
        if isinstance(model, ProbabilisticSpatialMIL):
            index = model.epoch*len(batches)+step
            for i, group in enumerate(optimizer.param_groups):
                group['lr'] = float(lr[index])*(model.decay_lr_scale if i else 1.)
                group['weight_decay'] = float(wd[index])
        result = model.batch(spatial_bags(bundle, batch, device, cfg, True))
        labels = torch.tensor([rows[0]['target'] for rows in batch], device=device)
        loss = model.objective(result, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite spatial objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append({'loss': float(loss.detach()), 'lr': optimizer.param_groups[0]['lr']})
    return losses

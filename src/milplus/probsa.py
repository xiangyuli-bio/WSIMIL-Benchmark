"""Author-library ProbSA: sampled Gaussian attention, graph KL and full recipe."""
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.spatial import KDTree, cKDTree
from torch import Tensor, nn
from torch.nn.attention import SDPBackend
from torch.utils.data import DataLoader

from .io import digest_file, digest_object
from .loss_attention import check_update_finite, finite
from .smmile_data import load_spatial_inputs, positive_int
from .spatial_attention import checked_points, spatial_metadata
from .stochastic import inference_rng
from .upstream import load_definitions

PROFILES = {'probsa_abmil_source', 'probsa_abmil_probmean', 'probsa_transformer_source', 'probsa_transformer_probmean'}


def graph_product(adjacency, values):
    """Preserve sparse A@v while aligning CUDA cuSPARSE batch pointers.

    Torch2.5.1/cu124 can fail with ``misaligned address`` for a sparse
    [B,N,N] product and an odd/large [B,N,1] right operand. Pad only the
    independent output-channel workspace to16 bytes, never the graph, bag,
    sampling population or KL denominator. Padding consumes no RNG state.
    """
    if adjacency.is_sparse and values.device.type == 'cuda':
        columns = values.shape[-1]
        alignment = max(1, 16 // values.element_size())
        padding = (-columns) % alignment
        aligned = torch.nn.functional.pad(values, (0, padding)).contiguous()
        return torch.bmm(adjacency, aligned)[..., :columns]
    return torch.bmm(adjacency, values)


def definitions():
    scope = {'torch': torch, 'np': np, 'Tensor': Tensor, 'Any': Any, 'TensorDict': dict,
             'SDP_BACKEND': [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]}
    def load(path, names, replacements=()):
        module = load_definitions('probsamil', path, names, scope, replacements)
        scope.update({name: getattr(module, name) for name in names})
        return module
    load('torchmil/nn/utils.py', ['LazyLinear', 'masked_softmax', 'get_feat_dim'])
    load('torchmil/models/mil_model.py', ['get_args_names', 'MILModel', 'MILModelWrapper'])
    pool = load('torchmil/nn/attention/prob_smooth_attention_pool.py', ['ProbSmoothAttentionPool'])
    scope['ProbSmoothAttentionPool'] = sparse_pool_class(pool.ProbSmoothAttentionPool)
    load('torchmil/nn/attention/multihead_self_attention.py', ['MultiheadSelfAttention'], [
        ('query, key, value, mask, self.dropout', 'query, key, value, mask, self.dropout if self.training else 0.0'),
        ('qk.masked_fill_(mask,', 'qk.masked_fill_(~mask,')])
    load('torchmil/nn/transformers/encoder.py', ['Encoder'])
    load('torchmil/nn/transformers/layer.py', ['Layer'])
    load('torchmil/nn/transformers/conventional_transformer.py', ['TransformerLayer', 'TransformerEncoder'])
    load('torchmil/models/prob_smooth_abmil.py', ['ProbSmoothABMIL'])
    load('torchmil/models/transformer_prob_smooth_abmil.py', ['TransformerProbSmoothABMIL'])
    load('torchmil/utils/annealing_scheduler.py', ['AnnealingScheduler', 'ConstantAnnealingScheduler', 'CyclicalAnnealingScheduler'])
    return scope


def sparse_pool_class(base):
    class SparsePool(base):
        def forward(self, X, adj=None, mask=None, return_att_samples=False, return_att_dist=False,
                    return_kl_div=False, n_samples=None):
            self.valid_mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device) if mask is None else mask.bool()
            if self.valid_mask.shape != X.shape[:2] or not self.valid_mask.any(1).all():
                raise ValueError('ProbSA masks require at least one instance per bag')
            X = X.masked_fill(~self.valid_mask.unsqueeze(-1), 0)
            result = super().forward(X, adj, self.valid_mask, return_att_samples, True, return_kl_div, n_samples)
            # Original sampling, MLP, clipping and classifier paths are unchanged.
            offset = 2 if return_att_samples else 1
            self.posterior_mean, self.posterior_log_variance = result[offset:offset+2]
            if return_att_dist:
                return result
            result = result[:offset]+result[offset+2:]
            return result[0] if len(result) == 1 else result

        def _kl_div(self, mu_f, log_diag_Sigma_f, adj_mat):
            if adj_mat is None or tuple(adj_mat.shape) != (len(mu_f), mu_f.shape[1], mu_f.shape[1]):
                raise ValueError('ProbSA KL requires aligned normalized adjacency')
            mask = self.valid_mask.unsqueeze(-1)
            mu = torch.where(mask, mu_f, 0)
            product = graph_product(adj_mat, mu)
            count = mask.sum((1, 2))
            inverse = 1.0/count.to(mu.dtype)
            quadratic = inverse*(mu.square().sum((1, 2))-torch.bmm(mu.transpose(1, 2), product).squeeze(1, 2))
            if log_diag_Sigma_f is None:
                return quadratic.mean()
            log_variance = torch.where(mask, log_diag_Sigma_f, 0)
            variance = torch.where(mask, log_variance.exp(), 0)
            if adj_mat.is_sparse:
                adj = adj_mat.coalesce();i = adj.indices();same = i[1] == i[2]
                diag = mu.new_zeros(mu.shape[0]*mu.shape[1])
                diag.scatter_add_(0, i[0, same]*mu.shape[1]+i[1, same], adj.values()[same])
                diag = diag.reshape(mu.shape[0], mu.shape[1], 1)
            else:
                diag = adj_mat.diagonal(dim1=1, dim2=2).unsqueeze(-1)
            trace = inverse*variance.sum((1, 2))
            trace_adj = inverse*(diag*variance).sum((1, 2))
            log_det = inverse*log_variance.sum((1, 2))
            return (quadratic+trace-trace_adj-.5*log_det).mean()
    return SparsePool


def graph_functions():
    return load_definitions('probsamil', 'torchmil/utils/graph_utils.py', ['degree', 'normalize_adj', 'build_adj'],
                            {'np': np, 'KDTree': KDTree})


def build_graph(x, coords, patch_size, feature_weighted=True, max_edges=2_000_000):
    points = checked_points(coords)
    if len(points) != len(x):
        raise ValueError('ProbSA feature/coordinate row mismatch')
    tree = cKDTree(points);radius = math.sqrt(2)*patch_size
    if int(tree.count_neighbors(tree, radius))-len(points) > max_edges:
        raise ValueError('ProbSA physical graph exceeds max_graph_edges before allocation')
    source = graph_functions()
    indices, values = source.build_adj(points, x.detach().cpu().numpy() if feature_weighted else None, radius, False)
    values = source.normalize_adj(indices, values, len(x))
    return torch.sparse_coo_tensor(torch.as_tensor(indices), torch.as_tensor(values, dtype=x.dtype),
                                   (len(x), len(x))).coalesce()


def pad_batch(bags, graphs, device):
    n = max(len(x) for x in bags);features = bags[0].new_zeros(len(bags), n, bags[0].shape[1], device=device)
    mask = torch.zeros(len(bags), n, device=device, dtype=torch.bool)
    indices, values = [], []
    for b, (x, graph) in enumerate(zip(bags, graphs)):
        features[b, :len(x)] = x.to(device);mask[b, :len(x)] = True
        i = graph.indices().to(device)
        indices.append(torch.cat([torch.full((1, i.shape[1]), b, device=device, dtype=torch.long), i]))
        values.append(graph.values().to(device))
    adj = torch.sparse_coo_tensor(torch.cat(indices, 1), torch.cat(values), (len(bags), n, n), device=device).coalesce()
    return features, mask, adj


class ProbSA(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'posterior attention Gaussian parameters and Monte Carlo normalized attention; these are not patch cancer probabilities'

    def __init__(self, bundle, profile='probsa_abmil_probmean', hidden=512, attention_dim=128,
                 train_samples=1000, test_samples=5000, batch_size=None, transformer_layers=2,
                 transformer_heads=8, transformer_dropout=0., cycles=5, kl_weight='cyclical',
                 prediction_seed=2026, feature_weighted=True, balance_loss=False, max_graph_edges=2_000_000,
                 max_sample_bytes=1024**3, max_attention_bytes=1024**3, max_graph_cache_bytes=256*1024**2):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification' or len(bundle.classes) != 2:
            raise ValueError('ProbSA author-library routes require binary classification')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('ProbSA requires separate slide coordinate frames')
        self.profile, self.bundle, self.dim = profile, bundle, bundle.dim
        self.transformer = 'transformer' in profile
        # These four routes share a training objective, but distinguish source versus Eq.19 prediction.
        self.batch_size = (1 if self.transformer else 4) if batch_size is None else batch_size
        for name, value in [('hidden', hidden), ('attention_dim', attention_dim), ('train_samples', train_samples),
                            ('test_samples', test_samples), ('batch_size', self.batch_size), ('cycles', cycles),
                            ('transformer_layers', transformer_layers), ('transformer_heads', transformer_heads),
                            ('max_graph_edges', max_graph_edges), ('max_graph_cache_bytes', max_graph_cache_bytes), ('max_sample_bytes', max_sample_bytes), ('max_attention_bytes', max_attention_bytes)]:
            positive_int(value, name)
        if self.transformer and hidden % transformer_heads:
            raise ValueError('ProbSA hidden width must divide transformer heads')
        finite(transformer_dropout, 'transformer_dropout', 0, 1)
        if type(prediction_seed) is not int or prediction_seed < 0 or type(feature_weighted) is not bool or type(balance_loss) is not bool:
            raise ValueError('Invalid prediction seed or feature weighting')
        if kl_weight != 'cyclical' and finite(kl_weight, 'kl_weight', 0, 1.000000001) <= 0:
            raise ValueError('A ProbSA route requires active KL regularization')
        self.hidden, self.attention_dim, self.train_samples, self.test_samples = hidden, attention_dim, train_samples, test_samples
        self.cycles, self.kl_weight, self.prediction_seed = cycles, kl_weight, prediction_seed
        self.feature_weighted, self.max_graph_edges = feature_weighted, max_graph_edges
        self.max_sample_bytes, self.max_attention_bytes = max_sample_bytes, max_attention_bytes
        self.transformer_layers, self.transformer_heads, self.transformer_dropout = transformer_layers, transformer_heads, transformer_dropout
        self.balance_loss = balance_loss
        self.patch_size = spatial_metadata(bundle)['patch_size']
        source = definitions();self.scheduler_class = source['CyclicalAnnealingScheduler']
        feature_extractor = nn.Sequential(nn.Linear(bundle.dim, hidden), nn.ReLU())
        common = {'in_shape': (bundle.dim,), 'feat_ext': feature_extractor, 'n_samples_train': train_samples, 'n_samples_test': test_samples}
        self.network = (source['TransformerProbSmoothABMIL'](pool_att_dim=attention_dim,
            transf_att_dim=hidden, transf_n_layers=transformer_layers, transf_n_heads=transformer_heads,
            transf_dropout=transformer_dropout, **common) if self.transformer else
            source['ProbSmoothABMIL'](att_dim=attention_dim, **common))
        if balance_loss:self.network.criterion.pos_weight = torch.ones(1)
        self.graphs, self.optimizer, self.lr_scheduler, self.annealing = OrderedDict(), None, None, None
        self.optimizer_steps, self.epoch, self.lr_state, self.cycle_state = 0, -1, None, None
        self.total_steps = None
        self.max_graph_cache_bytes = max_graph_cache_bytes

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'probsa_bag_auc':
            raise ValueError('ProbSA preserves full bags, unclipped Adam and validation AUROC selection')
        if cfg['weight_decay'] != 0:
            raise ValueError('ProbSA paper Adam uses default zero weight decay')
        self.stop_after_epoch = cfg['epochs']
        self.total_steps = math.ceil(len(self.bundle.records('train'))/self.batch_size)*cfg['epochs']
        if self.kl_weight == 'cyclical':
            length = self.total_steps/self.cycles
            if int(length*.8) < 1:
                raise ValueError('ProbSA training budget too small for declared KL cycles')
            self.annealing = self.scheduler_class(length, cycle_prop=.8)

    def prepare(self, bundle, seed, device):
        if self.balance_loss:
            targets = [r['target'] for r in bundle.records('train')]
            positive, negative = targets.count(1), targets.count(0)
            if min(positive, negative) == 0:raise ValueError('ProbSA balanced BCE requires both TRAIN classes')
            self.network.criterion.pos_weight.fill_(negative/positive)

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.1, total_iters=10)
        if self.lr_state is not None:self.lr_scheduler.load_state_dict(self.lr_state)
        return self.optimizer

    def start_epoch(self, epoch):self.epoch = epoch

    def finish_epoch(self, bundle, device, cfg):
        self.lr_scheduler.step()

    def coefficient(self):
        if self.kl_weight != 'cyclical':return float(self.kl_weight)
        if self.annealing is None:raise ValueError('ProbSA training schedule not configured')
        return float(self.annealing())

    def reserve(self, batch, n, samples):
        if batch*samples*(n+self.hidden)*4*4 > self.max_sample_bytes:
            raise ValueError('ProbSA Monte Carlo working set exceeds max_sample_bytes before allocation')
        if self.transformer and batch*n*n*self.transformer_heads*4*3*self.transformer_layers > self.max_attention_bytes:
            raise ValueError('ProbSA transformer working set exceeds max_attention_bytes')

    def load(self, row):
        x, coords = load_spatial_inputs(self.bundle, row)
        key = row['slide_id']
        if key in self.graphs:
            graph = self.graphs.pop(key);self.graphs[key] = graph
        else:
            graph = build_graph(x, coords, self.patch_size, self.feature_weighted, self.max_graph_edges)
            size = graph._nnz()*20
            while self.graphs and sum(g._nnz()*20 for g in self.graphs.values())+size > self.max_graph_cache_bytes:
                self.graphs.popitem(last=False)
            if size <= self.max_graph_cache_bytes:self.graphs[key] = graph
        return x, graph

    def batch(self, rows, device):
        items = [self.load(row) for row in rows]
        self.reserve(len(rows), max(len(x) for x, _ in items), self.train_samples if self.training else self.test_samples)
        return pad_batch([x for x, _ in items], [g for _, g in items], device)

    def forward(self, x, coords=None, label=None):
        return self.prediction(x, self.prediction_seed)

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:raise ValueError('ProbSA cannot merge different slide coordinate frames')
        row = rows[0];x, _ = bundle.load(row)
        seed = int(digest_object({'seed': self.prediction_seed, 'slide': row['slide_id']})[:15], 16)
        return self.prediction(x.to(device), seed)

    def prediction(self, x, seed):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('ProbSA requires finite nonempty [N,D] features')
        device = x.device
        self.reserve(1, len(x), self.train_samples if self.training else self.test_samples)
        with inference_rng(device, seed):
            logits, f = self.network(x.to(device).unsqueeze(0), adj=None, return_att=True, return_samples=True)
        p = logits.sigmoid();mean_p = p.mean(1)
        value = logits.mean(1) if self.profile.endswith('_source') else torch.logit(mean_p.clamp(1e-7, 1-1e-7))
        entropy = lambda q: -(q.clamp(1e-7, 1-1e-7)*q.clamp(1e-7, 1-1e-7).log()+(1-q).clamp(1e-7, 1-1e-7)*(1-q).clamp(1e-7, 1-1e-7).log())
        total_entropy, expected_entropy = entropy(mean_p), entropy(p).mean(1)
        result = {'logits': torch.stack([torch.zeros_like(value), value], 1),
                'probabilities': torch.stack([1-value.sigmoid(), value.sigmoid()], 1) if self.profile.endswith('_source') else torch.stack([1-mean_p, mean_p], 1),
                'attention': f.softmax(1).mean(2).squeeze(0),
                'posterior_mean': self.network.pool.posterior_mean.squeeze(0).squeeze(-1),
                'posterior_variance': self.network.pool.posterior_log_variance.squeeze(0).squeeze(-1).exp(),
                'predictive_entropy': total_entropy, 'expected_entropy': expected_entropy,
                'mutual_information': (total_entropy-expected_entropy).clamp_min(0)}
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise ValueError('ProbSA prediction or posterior is nonfinite')
        return result

    def validation_metrics(self, frame):
        from sklearn.metrics import roc_auc_score
        return {'probsa_bag_auc': float(roc_auc_score(frame.target, frame.p_1))}

    def get_extra_state(self):
        return {'optimizer_steps': self.optimizer_steps, 'epoch': self.epoch, 'total_steps': self.total_steps,
                'lr_state': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else self.lr_state,
                'cycle_state': vars(self.annealing).copy() if self.annealing is not None else self.cycle_state}

    def set_extra_state(self, state):
        expected = {'optimizer_steps', 'epoch', 'total_steps', 'lr_state', 'cycle_state'}
        if set(state) != expected or type(state['optimizer_steps']) is not int or state['optimizer_steps'] < 0:
            raise ValueError('Invalid ProbSA checkpoint counters')
        if self.total_steps is not None and state['total_steps'] != self.total_steps:
            raise ValueError('ProbSA checkpoint training budget changed')
        if self.annealing is not None and (not isinstance(state['cycle_state'], dict) or
                set(state['cycle_state']) != set(vars(self.annealing))):
            raise ValueError('Invalid ProbSA KL schedule checkpoint')
        for key in ['optimizer_steps', 'epoch', 'total_steps', 'lr_state', 'cycle_state']:setattr(self, key, state[key])
        if self.lr_scheduler is not None:self.lr_scheduler.load_state_dict(self.lr_state)
        if self.annealing is not None:self.annealing.__dict__.update(self.cycle_state)

    def protocol(self):
        return {'method': self.profile, 'family': 'probsamil', 'source': 'explicit author-maintained torchmil version, not unpublished original experiment scripts',
            'hidden': self.hidden, 'attention_dim': self.attention_dim, 'train_samples': self.train_samples, 'test_samples': self.test_samples,
            'batch_size': self.batch_size, 'optimizer_steps': self.optimizer_steps, 'kl_weight': self.kl_weight, 'cycles': self.cycles,
            'objective': 'mean sampled BCEWithLogits + coefficient * normalized graph-Gaussian KL; padded instances excluded',
            'optimizer': 'continuous Adam, zero weight decay; LinearLR(.1,10) per epoch',
            'coefficient': self.coefficient() if self.annealing is not None or self.kl_weight != 'cyclical' else None,
            'graph': 'static raw-FM spatial radius sqrt(2)*patch_size, exp(-L2/D) edge weights, symmetric normalization, no added selfloops',
            'inference': 'sigmoid(mean logits)' if self.profile.endswith('_source') else 'mean sigmoid(logits), paper Eq.19',
            'balance_loss': self.balance_loss, 'pos_weight': float(self.network.criterion.pos_weight) if self.balance_loss else 1.,
            'transformer_layers': self.transformer_layers, 'transformer_heads': self.transformer_heads, 'transformer_dropout': self.transformer_dropout,
            'repairs': ['padding excluded from KL and its N denominator', 'sparse trace/diagonal without dense CPU NxN materialization',
                        'CUDA sparse matvec output-channel workspace aligned to16 bytes; no graph/instance padding',
                        'SDPA evaluation dropout and diagnostic mask polarity'],
            'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'gpu_validation': 'not_run'}

    def export_instances(self, result, row, directory):
        directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object(row['slide_id'])[:24]+'.h5')
        _, coords = load_spatial_inputs(self.bundle, row)
        with h5py.File(path, 'w') as h:
            for key in ['attention', 'posterior_mean', 'posterior_variance']:h[key] = result[key].detach().cpu().numpy()
            h['coords'] = coords.numpy();h.attrs['score_semantics'] = self.instance_score_semantics
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path), 'instances': len(result['attention']), 'coordinates': True}


def probsa_epoch(model, bundle, groups, optimizer, device, cfg):
    if any(len(g) != 1 for g in groups):raise ValueError('ProbSA trains individual slide bags')
    rows = [g[0] for g in groups];losses = []
    loader = DataLoader(rows, batch_size=model.batch_size, shuffle=True, num_workers=0, collate_fn=lambda value: value)
    for batch in loader:
        x, mask, adj = model.batch(batch, device)
        target = torch.tensor([row['target'] for row in batch], device=device)
        optimizer.zero_grad(set_to_none=True)
        _, terms = model.network.compute_loss(target, x, adj, mask)
        coefficient = model.coefficient();total = terms['BCEWithLogitsLoss']+coefficient*terms['KLDiv']
        if not torch.isfinite(total):raise ValueError('ProbSA objective is nonfinite')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('ProbSA gradient is nonfinite')
        check_update_finite(model, optimizer);optimizer.step();check_update_finite(model, optimizer)
        model.optimizer_steps += 1
        if model.annealing is not None:model.annealing.step()
        losses.append({'loss': float(total.detach()), 'bce': float(terms['BCEWithLogitsLoss'].detach()),
                       'kl': float(terms['KLDiv'].detach()), 'kl_coefficient': coefficient})
    return losses

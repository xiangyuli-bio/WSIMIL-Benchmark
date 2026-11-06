"""Pinned GTP: GCN, differentiable min-cut pooling, ViT and complete recipe."""
import importlib.metadata
import sys
import types
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import dense_mincut_pool

from .io import digest_file, digest_object
from .loss_attention import check_update_finite, finite
from .smmile_data import load_spatial_inputs, positive_int
from .spatial_attention import checked_points, spatial_metadata
from .upstream import load_definitions, package, pinned_text, verified_source

PROFILES = {'gtp_source', 'gtp_repaired'}


def repaired_pool(x, adj, s, mask):
    output, coarse, mincut, orthogonality = dense_mincut_pool(x, adj, s, mask)
    positive = adj.sum((1, 2)) > 0
    if not positive.all():
        # The unused original 0/0 scalar is not attached to the returned objective.
        mincut = (dense_mincut_pool(x[positive], adj[positive], s[positive], mask[positive])[2]*positive.float().mean()
                  if positive.any() else x.new_zeros(()))
    return output, coarse, mincut, orthogonality


def load_network(repaired=False):
    if importlib.metadata.version('torch-geometric') != '2.6.1':
        raise ValueError('GTP min-cut backend currently validates torch-geometric2.6.1')
    root, sha = verified_source('gtp');name = '_milplus_gtp_'+sha
    if name+'.ViT' not in sys.modules:
        package(name, root/'models')
        for filename in ['layers', 'ViT']:
            code = pinned_text(root, sha, 'models/'+filename+'.py')
            module = types.ModuleType(name+'.'+filename);module.__file__ = str(root/'models'/(filename+'.py'))
            module.__package__ = name;module.__dict__['warnings'] = warnings
            sys.modules[module.__name__] = module
            exec(compile(code, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified pinned Git blobs
    gcn = load_definitions('gtp', 'models/gcn.py', ['GCNBlock'], {'torch': torch, 'nn': nn, 'F': F},
                           [('.cuda()', '')]).GCNBlock
    if repaired:
        class SafeBatchNorm(gcn):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Override just the one-observation BN operation; source behavior otherwise.
                self.bn_layer.register_forward_pre_hook(self.single_bn_before)
                self.bn_layer.register_forward_hook(self.single_bn_after)
                self._bn_training = None
            def single_bn_before(self, layer, inputs):
                if layer.training and len(inputs[0]) == 1:
                    self._bn_training = True;layer.training = False
            def single_bn_after(self, layer, inputs, output):
                if self._bn_training:
                    layer.training = True;self._bn_training = None
        gcn = SafeBatchNorm
    return load_definitions('gtp', 'models/GraphTransformer.py', ['Classifier'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'VisionTransformer': sys.modules[name+'.ViT'].VisionTransformer,
         'GCNBlock': gcn, 'Linear': nn.Linear, 'dense_mincut_pool': repaired_pool if repaired else dense_mincut_pool}, [
        ('n_features: int = 512):', 'n_features: int = 512, hidden=64, clusters=100, depth=3, heads=8):'),
        ('self.embed_dim = 64', 'self.embed_dim = hidden'), ('self.node_cluster_num = 100', 'self.node_cluster_num = clusters'),
        ('embed_dim=self.embed_dim)', 'embed_dim=self.embed_dim, depth=depth, num_heads=heads)'),
        ('return pred,labels,loss', "return {'logits': out, 'total_loss': loss, 'mincut': mc1, 'orthogonality': o1, 'assignment_logits': s}")]).Classifier


def lattice_edges(coords, stride):
    points = checked_points(coords)
    lattice = (points-points.min(0))/stride
    if (abs(lattice) > 2**52).any() or not np.allclose(lattice, np.rint(lattice), rtol=0, atol=1e-6):
        raise ValueError('GTP coordinates must lie on the declared physical patch_stride lattice')
    lattice = np.rint(lattice).astype(np.int64)
    lookup = {tuple(point): i for i, point in enumerate(lattice)}
    edges = []
    for i, (x, y) in enumerate(lattice):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                j = lookup.get((x+dx, y+dy))
                if j is not None and i != j:edges.append((i, j))
    return torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()


class GTP(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'author gradient/LRP transformer GraphCAM mapped by soft min-cut assignments; display scores include probability floor .4 and are not patch cancer probabilities'

    def __init__(self, bundle, profile='gtp_repaired', hidden=64, clusters=100, depth=3, heads=8,
                 batch_size=8, drop_last=True, max_dense_bytes=1024**3):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('GTP requires separate slide classification bags')
        for name, value in [('hidden', hidden), ('clusters', clusters), ('depth', depth), ('heads', heads),
                            ('batch_size', batch_size), ('max_dense_bytes', max_dense_bytes)]:positive_int(value, name)
        if hidden % heads or type(drop_last) is not bool:raise ValueError('Invalid GTP heads or drop_last')
        metadata = spatial_metadata(bundle)
        self.stride = finite(metadata.get('patch_stride', metadata['patch_size']), 'patch_stride')
        if self.stride <= 0:raise ValueError('GTP patch_stride must be positive')
        self.bundle, self.dim, self.profile = bundle, bundle.dim, profile
        self.hidden, self.clusters, self.depth, self.heads = hidden, clusters, depth, heads
        self.batch_size, self.drop_last, self.max_dense_bytes = batch_size, drop_last, max_dense_bytes
        self.reserve(1, 1)
        self.network = load_network(profile.endswith('repaired'))(len(bundle.classes), bundle.dim, hidden, clusters, depth, heads)
        self.optimizer, self.scheduler, self.scheduler_state = None, None, None
        self.optimizer_steps, self.epoch = 0, -1

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'gtp_bag_accuracy':
            raise ValueError('GTP retains complete graphs, unclipped source Adam and validation bag accuracy')
        if self.drop_last and len(self.bundle.records('train')) < self.batch_size:
            raise ValueError('GTP TRAIN bags cannot form one full drop_last batch')
        self.stop_after_epoch = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 100], gamma=.1)
        return self.optimizer

    def start_epoch(self, epoch):self.epoch = epoch

    def reserve(self, count, n):
        # Includes adjacency/degree intermediates, padded features and pooled transformer buffers.
        estimate = count*(20*n*n+4*n*(self.dim+self.hidden+self.clusters)+16*self.heads*(self.clusters+1)**2*self.depth)
        estimate += 16*(self.dim*self.hidden+self.hidden*self.clusters+8*self.depth*self.hidden*self.hidden+self.hidden*len(self.bundle.classes))
        if estimate > self.max_dense_bytes:raise ValueError('GTP dense graph/transformer exceeds max_dense_bytes before allocation')

    def inputs(self, rows, device):
        bags = [load_spatial_inputs(self.bundle, row) for row in rows]
        self.reserve(len(rows), max(len(x) for x, _ in bags))
        n = max(len(x) for x, _ in bags)
        x = torch.zeros(len(rows), n, self.dim, device=device)
        adj = torch.zeros(len(rows), n, n, device=device)
        mask = torch.zeros(len(rows), n, device=device)
        for b, (features, coords) in enumerate(bags):
            indices = lattice_edges(coords, self.stride).to(device)
            x[b, :len(features)] = features.to(device);mask[b, :len(features)] = 1
            if indices.shape[1]:adj[b, indices[0], indices[1]] = 1
        return x, adj, mask

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) == 0 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('GTP requires finite nonempty [N,D] features')
        self.reserve(1, len(x));indices = lattice_edges(coords, self.stride).to(x.device)
        adj = x.new_zeros(1, len(x), len(x));adj[0, indices[0], indices[1]] = 1
        labels = torch.zeros(1, dtype=torch.long, device=x.device) if label is None else label
        return self.batch_forward(x.unsqueeze(0), labels, adj, x.new_ones(1, len(x)))

    def batch_forward(self, x, labels, adj, mask):
        if x.ndim != 3 or x.shape[-1] != self.dim or min(x.shape[:2]) < 1:
            raise ValueError('GTP requires nonempty [B,N,D] padded features')
        b, n = x.shape[:2]
        if adj.shape != (b, n, n) or mask.shape != (b, n) or labels.shape != (b,):
            raise ValueError('GTP feature, adjacency, mask and label shapes do not align')
        if not all(torch.isfinite(v).all() for v in [x, adj, mask]) or (adj < 0).any():
            raise ValueError('GTP inputs must be finite with nonnegative adjacency')
        if not ((mask == 0) | (mask == 1)).all() or not mask.sum(1).gt(0).all():
            raise ValueError('GTP masks require valid binary nodes in every bag')
        mask = mask.to(x.dtype)
        count = mask.sum(1).long()
        if not torch.equal(mask.bool(), torch.arange(n, device=x.device)[None] < count[:, None]):
            raise ValueError('GTP source BN requires prefix-aligned valid nodes')
        for i, length in enumerate(count.tolist()):
            if adj[i, length:].count_nonzero() or adj[i, :length, length:].count_nonzero():
                raise ValueError('GTP padded graph nodes cannot have edges')
        if labels.dtype != torch.long or (labels < 0).any() or (labels >= len(self.bundle.classes)).any():
            raise ValueError('GTP requires valid integer class targets')
        if self.profile == 'gtp_source' and ((adj.sum((1, 2)) == 0).any() or self.training and mask.sum() < 2):
            raise ValueError('Source GTP requires graph edges and at least two TRAIN observations for BN; use explicit repaired route')
        result = self.network(x, labels, adj, mask)
        if not all(torch.isfinite(value).all() for value in result.values()):raise ValueError('GTP output or regularizer is nonfinite')
        return result

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:raise ValueError('GTP cannot merge different slide coordinate frames')
        x, adj, mask = self.inputs(rows, device)
        # Query truth is not passed even though the source computes an unused CE.
        return self.batch_forward(x, torch.zeros(1, dtype=torch.long, device=device), adj, mask)

    def validation_metrics(self, frame):
        return {'gtp_bag_accuracy': float((frame[['p_'+str(c) for c in range(len(self.bundle.classes))]].to_numpy().argmax(1) == frame.target).mean())}

    def get_extra_state(self):
        return {'optimizer_steps': self.optimizer_steps, 'epoch': self.epoch,
                'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        if set(state) != {'optimizer_steps', 'epoch', 'scheduler_state'} or type(state['optimizer_steps']) is not int or state['optimizer_steps'] < 0:
            raise ValueError('Invalid GTP checkpoint state')
        self.optimizer_steps, self.epoch, self.scheduler_state = state['optimizer_steps'], state['epoch'], state['scheduler_state']
        if self.scheduler is not None:self.scheduler.load_state_dict(self.scheduler_state)

    def protocol(self):
        return {'method': self.profile, 'family': 'gtp', 'input': 'explicit frozen-FM transfer of author SimCLR-feature graph classifier',
            'hidden': self.hidden, 'clusters': self.clusters, 'depth': self.depth, 'heads': self.heads,
            'batch_size': self.batch_size, 'drop_last': self.drop_last, 'patch_stride': self.stride,
            'graph': 'binary eight-neighbor physical patch lattice, no input selfloops; GCN adds self branch',
            'objective': 'author CE + differentiable mincut + orthogonality', 'optimizer_steps': self.optimizer_steps,
            'optimizer': 'Adam continuous state; MultiStepLR20/100 gamma.1 called with epoch before each batch, as source',
            'selection': 'validation individual-bag accuracy, first best; fixed declared training budget',
            'repairs': ['device allocation and exposed true logits', 'undefined constant GraphCAM display emits zero with explicit defined flag']+(['zero-edge graphs contribute zero mincut; source orthogonality retained',
                        'singleton TRAIN BN uses running statistics without advancing its counter'] if self.profile.endswith('repaired') else []),
            'graphcam': 'author LRP/attention-gradient rollout and soft assignment mapping; per-class fresh forward, no file side effects',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

    def graphcam(self, row, device):
        if self.training:raise ValueError('GTP GraphCAM requires eval mode')
        x, adj, mask = self.inputs([row], device)
        saved = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters()]
        raw, display, pooled, defined = [], [], [], []
        try:
            for c in range(len(self.bundle.classes)):
                with torch.enable_grad():
                    self.zero_grad(set_to_none=True)
                    result = self.batch_forward(x, torch.zeros(1, dtype=torch.long, device=device), adj, mask)
                    logits = result['logits'];seed = torch.zeros_like(logits);seed[0, c] = logits[0, c].detach()
                    (seed*logits).sum().backward(retain_graph=True)
                    cam = self.network.transformer.relprop(seed, method='transformer_attribution', start_layer=0, alpha=1)
                    mapped = result['assignment_logits'][0].softmax(1)@cam.T
                    mapped = mapped.detach().squeeze(1);span = mapped.max()-mapped.min()
                    if span == 0:
                        normalized = torch.zeros_like(mapped)
                    else:normalized = (mapped-mapped.min())/span
                    defined.append(bool(span > 0))
                    display.append((normalized*logits.softmax(1)[0, c].detach().clamp(.4, 1)).clamp(0, 1))
                    raw.append(mapped);pooled.append(cam.detach().squeeze(0))
        finally:
            for p, gradient in zip(self.parameters(), saved):p.grad = gradient
        result = {'graphcam_raw': torch.stack(raw, 1), 'graphcam_display': torch.stack(display, 1), 'pooled_graphcam': torch.stack(pooled, 1),
                  'graphcam_display_defined': torch.tensor(defined, device=device)}
        if not all(torch.isfinite(value).all() for value in result.values()):raise ValueError('GTP GraphCAM is nonfinite')
        return result

    def export_instances(self, result, row, directory):
        maps = self.graphcam(row, result['logits'].device)
        directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object(row['slide_id'])[:24]+'.h5');_, coords = load_spatial_inputs(self.bundle, row)
        with h5py.File(path, 'w') as h:
            for key, value in maps.items():h[key] = value.cpu().numpy()
            h['coords'] = coords.numpy();h.attrs['score_semantics'] = self.instance_score_semantics
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path), 'instances': len(coords), 'coordinates': True}


def gtp_epoch(model, bundle, groups, optimizer, device, cfg):
    if any(len(g) != 1 for g in groups):raise ValueError('GTP trains separate slide graphs')
    loader = DataLoader([g[0] for g in groups], batch_size=model.batch_size, shuffle=True, drop_last=model.drop_last,
                        num_workers=0, collate_fn=lambda rows: rows)
    losses = []
    for rows in loader:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*')
            warnings.filterwarnings('ignore', message='.*epoch parameter.*')
            model.scheduler.step(model.epoch)
        x, adj, mask = model.inputs(rows, device)
        target = torch.tensor([row['target'] for row in rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        result = model.batch_forward(x, target, adj, mask);loss = result['total_loss']
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('GTP gradient is nonfinite')
        check_update_finite(model, optimizer);optimizer.step();check_update_finite(model, optimizer)
        model.optimizer_steps += 1
        losses.append({'loss': float(loss.detach()), 'mincut': float(result['mincut'].detach()), 'orthogonality': float(result['orthogonality'].detach())})
    if not losses:raise ValueError('GTP produced no complete TRAIN batch')
    return losses

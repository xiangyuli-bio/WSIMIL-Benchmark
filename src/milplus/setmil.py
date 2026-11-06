"""Pinned SETMIL: physical PPE, pyramid T2T, spatial iRPE and complete training."""
import ast
import math
import sys
import types

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import resolve
from .loss_attention import check_update_finite, finite
from .smmile_data import array_identity, load_spatial_inputs, positive_int
from .spatial_attention import checked_points, spatial_metadata
from .upstream import load_definitions, load_single, package, pinned_text, verified_source

PROFILES = {'setmil_source', 'setmil_repaired'}


class AttrDict(dict):
    """Data-only EasyDict compatibility for the author's RPE configuration."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_network():
    root, sha = verified_source('setmil');name = '_milplus_setmil_'+sha
    if name+'.irpe_t2t' not in sys.modules:
        package(name, root/'trans/models');package(name+'.t2t_module', root/'trans/models/t2t_module')
        for module_name, path in [('t2t_module.irpe', 'trans/models/t2t_module/irpe.py'),
                                  ('t2t_module.transformer_block', 'trans/models/t2t_module/transformer_block.py'),
                                  ('t2t_module.token_transformer', 'trans/models/t2t_module/token_transformer.py'),
                                  ('t2t_module.token_performer', 'trans/models/t2t_module/token_performer.py'),
                                  ('irpe_t2t', 'trans/models/irpe_t2t.py')]:
            code = pinned_text(root, sha, path)
            code = code.replace('from easydict import EasyDict as edict', 'from milplus.setmil import AttrDict as edict')
            code = code.replace('from timm.models.layers import DropPath', 'from milplus.layers import DropPath')
            code = code.replace('from timm.models.layers import trunc_normal_', 'from torch.nn.init import trunc_normal_')
            code = code.replace('from timm.models.helpers import load_pretrained', '')
            code = code.replace('from timm.models.registry import register_model', '')
            tree = ast.parse(code)
            if module_name == 't2t_module.irpe':
                # Select the existing official pure-Torch backend, never an unrelated installed rpe_ops.
                candidates = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Try)]
                if len(candidates) != 1:raise ValueError('SETMIL RPE fallback patch no longer matches')
                tree.body[candidates[0]] = ast.parse('RPEIndexFunction = None').body[0]
            module = types.ModuleType(name+'.'+module_name)
            module.__file__, module.__package__ = str(root/path), name+('.t2t_module' if module_name.startswith('t2t_module.') else '')
            sys.modules[module.__name__] = module
            exec(compile(ast.fix_missing_locations(tree), module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified Git blobs
    return sys.modules[name+'.irpe_t2t'].T2T_ViT_iRPE


def augmentation():
    tensor_aug = load_single('setmil', 'trans/util/numpy_img_aug.py')
    transforms = load_definitions('setmil', 'trans/datasets/wsi_feat_dataset.py', ['TrFeatureMapAug', 'ValFeatureMapAug'],
                                  {'np': np, 'tensor_aug': tensor_aug})
    return tensor_aug, transforms


def physical_indices(coords, stride):
    if not isinstance(coords, torch.Tensor) or coords.is_complex() or coords.dtype == torch.bool:
        raise ValueError('SETMIL coordinates must be real numeric tensors')
    points = checked_points(coords)
    shifted = (points-points.min(0))/stride
    if (abs(shifted) > 2**52).any() or not np.allclose(shifted, np.rint(shifted), rtol=0, atol=1e-6):
        raise ValueError('SETMIL requires coordinates aligned to the declared physical patch_stride')
    grid = np.rint(shifted).astype(np.int64)[:, ::-1].copy()  # canonical x,y -> source row,col
    if len(np.unique(grid, axis=0)) != len(grid):raise ValueError('SETMIL physical grid cells must be distinct')
    return grid


def encode_map(features, coords, stride, size, repaired, train=False, views=None, max_bytes=4*1024**3):
    """Actual WSIFeatDataset path, including its source crop and NumPy augmentation."""
    positive_int(size, 'map_size');positive_int(max_bytes, 'max_dense_bytes')
    if (features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all()
            or min(features.shape) == 0 or coords is None or len(features) != len(coords)):
        raise ValueError('SETMIL needs finite aligned nonempty patch features and real coordinates')
    if not math.isfinite(stride) or stride <= 0:raise ValueError('SETMIL patch_stride must be positive')
    grid = physical_indices(coords, stride);height, width = map(int, grid.max(0)+1)
    dim = features.shape[1]
    # Both source float64 canvas and row-ID provenance are allocated only after this check.
    estimate = height*width*(dim+1)*8 + size*size*(dim*32+16)
    if estimate > max_bytes:raise ValueError('SETMIL physical canvas exceeds max_dense_bytes before allocation')
    canvas = np.zeros((height, width, dim), dtype=np.float64)
    identities = np.full((height, width, 1), -1, dtype=np.int64)
    values = features.detach().cpu().numpy()
    if views is not None:
        views = np.asarray(views)
        if (views.ndim != 3 or views.shape[0] != len(values) or views.shape[2] != dim or views.shape[1] < 1
                or not np.issubdtype(views.dtype, np.floating) or not np.isfinite(views).all()):
            raise ValueError('SETMIL cached views require finite floating [N,V,D] aligned to features')
        # This is also active at VAL/TEST in the author's source route when tr exists.
        values = np.stack([v[np.random.choice(v.shape[0], 1).item()] for v in views])
    for i, (r, c) in enumerate(grid):
        canvas[r, c] = values[i];identities[r, c, 0] = i
    aug, transforms = augmentation()
    if height > size:
        canvas = aug.center_crop(canvas, (size, width));identities = aug.center_crop(identities, (size, width))
    if width > size:
        crop_height = canvas.shape[0] if repaired else height
        canvas = aug.center_crop(canvas, (crop_height, size));identities = aug.center_crop(identities, (crop_height, size))
    cropped_shape = list(canvas.shape[:2]);retained = identities[identities >= 0].reshape(-1)
    metadata = {'source_canvas_shape': [height, width], 'cropped_shape': cropped_shape, 'map_size': size,
                'total_patches': len(features), 'retained_before_augmentation': len(retained),
                'retained_rows_sha256': array_identity(retained)['sha256'],
                'coordinate_sha256': array_identity(coords.detach().cpu().numpy())['sha256'],
                'cached_views_used': views is not None, 'augmented': bool(train)}
    mask = np.zeros(canvas.shape[:2])
    canvas, mask = aug.pad_image_and_mask(canvas, mask, size, image_fill_value=0, mask_fill_value=1)
    transform = transforms.TrFeatureMapAug() if train else transforms.ValFeatureMapAug()
    canvas, mask = transform(canvas, mask)
    tensor = torch.from_numpy(canvas.copy()).float().permute(2, 0, 1)
    mask = torch.from_numpy(mask.copy()).bool()
    if tensor.shape != (dim, size, size) or not torch.isfinite(tensor).all():raise ValueError('SETMIL PPE produced invalid features')
    return tensor, mask, metadata


class SETMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile='setmil_repaired', map_size=48, reduction=7, token_dim=128, embed_dim=64,
                 depth=6, heads=16, dropout=.2, attention_dropout=.2, batch_size=4,
                 lr_drop=100, max_dense_bytes=4*1024**3):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide' or len(bundle.classes) < 2:
            raise ValueError('SETMIL requires individual-slide classification bags')
        for name, value in [('map_size', map_size), ('reduction', reduction), ('token_dim', token_dim), ('embed_dim', embed_dim),
                            ('depth', depth), ('heads', heads), ('batch_size', batch_size), ('lr_drop', lr_drop), ('max_dense_bytes', max_dense_bytes)]:positive_int(value, name)
        if reduction < 3 or reduction % 2 != 1 or map_size < 8 or map_size % (reduction//2) or 3*embed_dim % heads:
            raise ValueError('SETMIL requires odd reduction>=3, map_size>=8 divisible by stride, and decoder width divisible by heads')
        for name, value in [('dropout', dropout), ('attention_dropout', attention_dropout)]:
            if not 0 <= finite(value, name) < 1:raise ValueError('SETMIL dropout must lie in [0,1)')
        metadata = spatial_metadata(bundle)
        self.stride = finite(metadata.get('patch_stride', metadata['patch_size']), 'patch_stride')
        if self.stride <= 0:raise ValueError('SETMIL patch_stride must be positive')
        self.bundle, self.dim, self.profile = bundle, bundle.dim, profile
        self.map_size, self.reduction, self.token_dim, self.embed_dim = map_size, reduction, token_dim, embed_dim
        self.depth, self.heads, self.batch_size = depth, heads, batch_size
        self.dropout, self.attention_dropout, self.lr_drop, self.max_dense_bytes = dropout, attention_dropout, lr_drop, max_dense_bytes
        self.reserve(batch_size)
        self.cached = bundle.encoder.get('setmil_cached_views')
        if self.cached is not None and (not isinstance(self.cached, dict) or set(self.cached) != {'dataset', 'provenance'}
                or not all(isinstance(self.cached[k], str) and self.cached[k].strip() for k in self.cached)):
            raise ValueError('SETMIL cached views require dataset and provenance in the same hashed feature HDF5')
        self.network = load_network()(img_size=map_size, in_chans=bundle.dim, num_classes=len(bundle.classes),
            embed_dim=embed_dim, depth=depth, num_heads=heads, mlp_ratio=4., drop_rate=dropout, attn_drop_rate=attention_dropout,
            token_dim=token_dim, channel_reduce_rate=reduction, aspp_flag=1,
            irpe={'rpe': True, 'method': 'euc', 'mode': 'bias', 'shared_head': False, 'rpe_on': 'q'},
            token_t={'rpe': False, 'drop': dropout, 'attn_drop': attention_dropout, 'method': 'euc', 'mode': 'bias', 'shared_head': False, 'rpe_on': 'q'})
        self.optimizer, self.scheduler, self.scheduler_state = None, None, None
        self.optimizer_steps, self.epoch = 0, -1

    def reserve(self, count):
        m, d, t, e, r = self.map_size, self.dim, self.token_dim, self.embed_dim, self.reduction
        side = m//(r//2);n = side**2
        # Conservative parameter/gradient/Adam and retained unfold, attention and FFN buffers.
        parameters = d*r*r*(3*t+2)+12*t*t+3*sum(t*k*k*(3*e+2)+4*e*e for k in [3,5,7])+12*(3*e)**2*self.depth
        estimate = 20*parameters+count*(16*d*r*r*n+16*t*83*n+64*e*n+32*self.heads*(n+1)**2*self.depth)
        if estimate > self.max_dense_bytes:raise ValueError('SETMIL model/attention exceeds max_dense_bytes before allocation')

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['selection_metric'] != 'setmil_bag_auc':
            raise ValueError('SETMIL preserves full physical maps and validation individual-bag AUC')
        if cfg['grad_clip'] is None or finite(cfg['grad_clip'], 'grad_clip') <= 0:
            raise ValueError('SETMIL requires its declared positive gradient clipping')
        if set(self.bundle.rows.query("subset == 'val'").target) != set(range(len(self.bundle.classes))):
            raise ValueError('SETMIL validation AUC requires all declared classes')

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_drop)
        return self.optimizer

    def start_epoch(self, epoch):self.epoch = epoch

    def finish_epoch(self, bundle, device, cfg):self.scheduler.step()

    def map_row(self, row, train):
        x, coords = load_spatial_inputs(self.bundle, row);views = None
        if self.cached is not None and (train or self.profile == 'setmil_source'):
            path = resolve(self.bundle.root, row['path'])
            if path.suffix not in {'.h5', '.hdf5'}:raise ValueError('SETMIL cached views must share the hashed HDF5 feature file')
            with h5py.File(path, 'r') as h:
                if self.cached['dataset'] not in h:raise ValueError('SETMIL declared cached views dataset is missing')
                dataset = h[self.cached['dataset']]
                if dataset.ndim != 3 or dataset.shape[0] != len(x) or dataset.shape[2] != self.dim or dataset.shape[1] < 1 or not np.issubdtype(dataset.dtype, np.floating):
                    raise ValueError('SETMIL cached views require floating [N,V,D] aligned to features')
                if dataset.size*dataset.dtype.itemsize > self.max_dense_bytes:raise ValueError('SETMIL cached views exceed max_dense_bytes')
                views = dataset[:]
        return encode_map(x, coords, self.stride, self.map_size, self.profile == 'setmil_repaired', train, views, self.max_dense_bytes)

    def inputs(self, rows, device, train=False):
        self.reserve(len(rows));maps = [self.map_row(row, train) for row in rows]
        return (types.SimpleNamespace(tensors=torch.stack([x for x, _, _ in maps]).to(device),
                                      mask=torch.stack([mask for _, mask, _ in maps]).to(device)), [meta for _, _, meta in maps])

    def forward(self, x, coords=None, label=None):
        tensor, mask, meta = encode_map(x, coords, self.stride, self.map_size, self.profile == 'setmil_repaired',
                                         self.training, max_bytes=self.max_dense_bytes)
        return self.batch_forward(types.SimpleNamespace(tensors=tensor.unsqueeze(0).to(x.device), mask=mask.unsqueeze(0).to(x.device)), [meta], label)

    def batch_forward(self, sample, metadata=None, labels=None):
        x = sample.tensors
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.dim, self.map_size, self.map_size) or len(x) < 1 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('SETMIL requires finite nonempty [B,D,map_size,map_size] spatial maps')
        if labels is not None and (labels.shape != (len(x),) or labels.dtype != torch.long or (labels < 0).any() or (labels >= len(self.bundle.classes)).any()):
            raise ValueError('SETMIL requires aligned integer class labels')
        self.reserve(len(x));logits = self.network(sample)
        if not torch.isfinite(logits).all():raise ValueError('SETMIL logits are nonfinite')
        result = {'logits': logits, 'ppe': metadata}
        if labels is not None:result['total_loss'] = F.cross_entropy(logits, labels)
        return result

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1 or config['max_instances'] is not None:raise ValueError('SETMIL cannot merge slide coordinate frames or sample away patches')
        sample, metadata = self.inputs(rows, device, train)
        return self.batch_forward(sample, metadata)

    def prediction_fields(self, result, row):
        p = result['ppe'][0]
        return {'ppe_total_patches': p['total_patches'], 'ppe_retained_before_augmentation': p['retained_before_augmentation'],
                'ppe_retained_rows_sha256': p['retained_rows_sha256'], 'ppe_coordinate_sha256': p['coordinate_sha256']}

    def validation_metrics(self, frame):
        y = frame.target.to_numpy();p = frame[['p_'+str(c) for c in range(len(self.bundle.classes))]].to_numpy()
        if len(np.unique(y)) != len(self.bundle.classes):raise ValueError('SETMIL bag AUC requires all declared classes')
        return {'setmil_bag_auc': float(roc_auc_score(y, p[:, 1]) if len(self.bundle.classes) == 2 else
                np.mean([roc_auc_score(y == c, p[:, c]) for c in range(len(self.bundle.classes))]))}

    def get_extra_state(self):
        return {'epoch': self.epoch, 'optimizer_steps': self.optimizer_steps,
                'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        if set(state) != {'epoch', 'optimizer_steps', 'scheduler_state'} or type(state['optimizer_steps']) is not int or state['optimizer_steps'] < 0:
            raise ValueError('Invalid SETMIL checkpoint extra state')
        self.epoch, self.optimizer_steps, self.scheduler_state = state['epoch'], state['optimizer_steps'], state['scheduler_state']
        if self.scheduler is not None:self.scheduler.load_state_dict(self.scheduler_state)

    def protocol(self):
        return {'method': self.profile, 'family': 'setmil', 'input': 'physical frozen-FM PPE; optical patch encoder training is outside this adapter',
                'map_size': self.map_size, 'reduction': self.reduction, 'token_dim': self.token_dim, 'embed_dim': self.embed_dim,
                'decoder_width': 3*self.embed_dim, 'depth': self.depth, 'heads': self.heads, 'dropout': self.dropout,
                'attention_dropout': self.attention_dropout, 'batch_size': self.batch_size, 'drop_last': False,
                'num_workers': 0, 'prefetch': 'one batch ahead, source consumption order with synchronous device transfer',
                'patch_stride': self.stride, 'PPE': 'real row/column lattice, subtract minima, zero holes, center crop and symmetric padding; no pseudo-grid',
                'crop': 'source stale-height second crop' if self.profile == 'setmil_source' else 'recompute height after first crop',
                'cached_views': self.cached, 'cached_views_scope': 'all subsets as source' if self.profile == 'setmil_source' else 'TRAIN only; validation/test use registered val features',
                'augmentation': 'source flipH/flipV, nearest scale/crop/pad, mean cutout, rot90; TRAIN only',
                'objective': 'CrossEntropyLoss', 'optimizer': 'AdamW; StepLR after each epoch, gamma.1', 'lr_drop': self.lr_drop,
                'optimizer_steps': self.optimizer_steps, 'selection': 'first best validation individual-bag AUC, default patience20; test only explicit evaluate',
                'checkpoint': 'coherent last model/AdamW/scheduler/epoch/RNG separated from best, unlike source mixed checkpoint',
                'mask': 'network ignores source mask; not silently changed', 'iRPE': 'official pure-Torch, euc bias on queries, per head; original lookup preserved',
                'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}


def setmil_epoch(model, bundle, groups, optimizer, device, cfg):
    if any(len(g) != 1 for g in groups):raise ValueError('SETMIL requires separate slide bags')
    def collate(rows):
        sample, metadata = model.inputs(rows, device, train=True)
        return sample, metadata, torch.tensor([r['target'] for r in rows], device=device)
    loader = DataLoader([g[0] for g in groups], batch_size=model.batch_size, shuffle=True, drop_last=False,
                        num_workers=0, collate_fn=collate)
    iterator = iter(loader);current = next(iterator, None);losses = []
    while current is not None:
        # Author CUDA prefetcher advances before the current model forward, including the last sampler exhaustion.
        upcoming = next(iterator, None)
        sample, metadata, targets = current
        optimizer.zero_grad(set_to_none=True)
        result = model.batch_forward(sample, metadata, targets);loss = result['total_loss'];loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        check_update_finite(model, optimizer);optimizer.step();check_update_finite(model, optimizer)
        model.optimizer_steps += 1
        losses.append({'loss': float(loss.detach()), 'gradient_norm': float(norm), 'lr': optimizer.param_groups[0]['lr']})
        current = upcoming
    if not losses:raise ValueError('SETMIL has no TRAIN batches')
    return losses

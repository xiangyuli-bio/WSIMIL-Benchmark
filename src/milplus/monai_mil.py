"""Full trainable-ResNet MONAI MIL models and the published ordinal tutorial recipe."""
import importlib.metadata
from contextlib import nullcontext
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .io import digest_object
from .loss_attention import check_update_finite, finite
from .raw_images import positive_integer
from .upstream import load_definitions

MODES = {'monai_mean_tutorial': 'mean', 'monai_max_tutorial': 'max', 'monai_attention_tutorial': 'att',
         'monai_transformer_tutorial': 'att_trans', 'monai_pyramid_tutorial': 'att_trans_pyramid'}


def author_model():
    if importlib.metadata.version('monai') != '1.5.1' or importlib.metadata.version('torchvision').split('+')[0] != '0.20.1':
        raise ValueError('MONAI raw-image route requires monai1.5.1 and torchvision0.20.1')
    from torchvision import models
    return load_definitions('monai_mil', 'monai/networks/nets/milmodel.py', ['MILModel'],
                            {'torch': torch, 'nn': nn, 'cast': cast, 'models': models}).MILModel


def ordinal_targets(target, thresholds, device):
    y = torch.as_tensor(target, device=device)
    if y.ndim != 1 or y.dtype == torch.bool or y.is_complex() or not torch.isfinite(y).all() or not torch.equal(y, y.long()) or (y < 0).any() or (y > thresholds).any():
        raise ValueError('Ordinal label must be an integer grade in the registered class range')
    return (torch.arange(thresholds, device=device)[None] < y[:, None]).float()


def image_transform(tile_size, tile_count, training):
    from monai.transforms import (
        Compose,
        GridPatchd,
        RandFlipd,
        RandGridPatchd,
        RandRotate90d,
        ScaleIntensityRanged,
        SplitDimd,
        ToTensord,
    )
    grid = (RandGridPatchd(keys=['image'], patch_size=(tile_size, tile_size), num_patches=tile_count,
                          sort_fn='min', pad_mode=None, constant_values=255) if training else
            GridPatchd(keys=['image'], patch_size=(tile_size, tile_size), threshold=.999*3*255*tile_size**2,
                       pad_mode=None, constant_values=255))
    items = [grid, SplitDimd(keys=['image'], dim=0, keepdim=False, list_output=True)]
    if training:
        items += [RandFlipd(keys=['image'], spatial_axis=0, prob=.5), RandFlipd(keys=['image'], spatial_axis=1, prob=.5),
                  RandRotate90d(keys=['image'], prob=.5)]
    return Compose(items+[ScaleIntensityRanged(keys=['image'], a_min=np.float32(0), a_max=np.float32(255)), ToTensord(keys=['image'])])


class ImageBags(Dataset):
    def __init__(self, model, rows):
        self.model, self.rows = model, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        images, _ = self.model.images(row, training=True)
        return {'image': images, 'label': ordinal_targets([row['target']], self.model.thresholds, 'cpu')[0]}


class MONAIMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile='monai_transformer_tutorial', trans_blocks=4, trans_dropout=0., batch_size=4,
                 tile_size=256, tile_count=44, eval_chunk=44, amp=False, max_dense_bytes=16*1024**3):
        super().__init__()
        if profile not in MODES or bundle.task_type != 'ordinal' or bundle.raw_images is None:
            raise ValueError('MONAI requires an explicit full RGB image bundle and ordered grade labels')
        if bundle.config.get('training_unit', 'slide') != 'slide' or bundle.config.get('evaluation_unit', 'slide') != 'slide':
            raise ValueError('Published MONAI recipe trains and evaluates individual WSI grades; register patient fusion separately')
        for name, value in [('trans_blocks', trans_blocks), ('batch_size', batch_size), ('tile_size', tile_size),
                            ('tile_count', tile_count), ('eval_chunk', eval_chunk), ('max_dense_bytes', max_dense_bytes)]:
            positive_integer(value, name)
        if tile_size < 32 or type(amp) is not bool or not 0 <= finite(trans_dropout, 'trans_dropout') < 1:
            raise ValueError('MONAI requires RGB patches at least32px and explicit AMP/dropout settings')
        self.bundle, self.profile, self.mode = bundle, profile, MODES[profile]
        self.trans_blocks, self.trans_dropout = trans_blocks, trans_dropout
        self.batch_size, self.tile_size, self.tile_count, self.eval_chunk = batch_size, tile_size, tile_count, eval_chunk
        self.thresholds, self.amp, self.max_dense_bytes = len(bundle.classes)-1, amp, max_dense_bytes
        self.reserve(1, 1)
        self.network = author_model()(self.thresholds, self.mode, pretrained=False, trans_blocks=trans_blocks, trans_dropout=trans_dropout)
        if bundle.raw_images.weights is not None:
            state = torch.load(bundle.raw_images.weights, weights_only=True, map_location='cpu')
            # The known ImageNet classifier is removed by the native MILModel; no other keys are discarded.
            if not {'fc.weight', 'fc.bias'} <= set(state):
                raise ValueError('ImageNet ResNet checkpoint lacks its expected classification layer')
            state = {k: v for k, v in state.items() if k not in {'fc.weight', 'fc.bias'}}
            self.network.net.load_state_dict(state, strict=True)
        self.train_transform = image_transform(tile_size, tile_count, True)
        self.eval_transform = image_transform(tile_size, tile_count, False)
        self.epoch, self.optimizer_steps, self.training_bags = -1, 0, 0
        self.optimizer, self.scheduler, self.scaler = None, None, None
        self.scheduler_state, self.scaler_state = None, None

    def reserve(self, b, n, raw_bytes=0, training=True):
        d = 2304 if self.mode == 'att_trans_pyramid' else 2048
        layer = 4*d*d+2*2048*d+12*d+2048
        params = 24_000_000+2048*d+d*self.thresholds
        if self.mode.startswith('att_trans'):
            params += self.trans_blocks*layer
        if self.mode == 'att_trans_pyramid':
            params += 3*self.trans_blocks*(4*256**2+2*256*2048+12*256+2048)+768*256+1280*256
        # Whole TRAIN minibatch keeps backbone BN/activation graphs; no silent train chunking or freezing.
        active = n if training else min(n, self.eval_chunk)
        estimate = 20*params+8*raw_bytes+768*b*active*self.tile_size**2+64*self.trans_blocks*b*n*n+32*b*n*d
        if estimate > self.max_dense_bytes:
            raise ValueError('MONAI raw CNN/transformer exceeds max_dense_bytes before allocation')

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'qwk':
            raise ValueError('MONAI retains full RGB/own tile sampling, no clipping and native validation QWK')
        if self.amp and next(self.parameters()).device.type != 'cuda':
            raise ValueError('AMP MONAI training requires an actual CUDA device')
        self.total_epochs = cfg['epochs']
        self.stop_after_epoch = cfg['epochs']

    def images(self, row, training=False):
        info = self.bundle.raw_images.describe(row)
        if min(info['shape'][1:]) < self.tile_size:
            raise ValueError('Image level is smaller than the real patch size; no resize/upscale substitute')
        grid_count = (info['shape'][1]//self.tile_size)*(info['shape'][2]//self.tile_size)
        self.reserve(1, self.tile_count if training else grid_count, int(np.prod(info['shape'])), training=training)
        image, info = self.bundle.raw_images.load(row)
        pipeline = self.train_transform if training else self.eval_transform
        if training:
            # A declared one-worker stream, fully determined by checkpointed CPU Torch RNG.
            seed = int(torch.randint(0, 2**32-1, (1,), dtype=torch.int64).item())
            pipeline.set_random_state(seed)
        try:
            pieces = pipeline({'image': image})
        except (RuntimeError, ValueError, IndexError) as error:
            raise ValueError('Native MONAI grid extraction failed; check image dimensions and foreground tiles') from error
        if not pieces or pieces[0]['image'].numel() == 0:
            raise ValueError('Raw WSI has no foreground validation tiles under the native threshold')
        x = torch.stack([p['image'].as_subclass(torch.Tensor) for p in pieces])
        from monai.utils.enums import PatchKeys
        # SplitDim keeps the complete location matrix in each patch's metadata. Read it once, not N times.
        locations = np.asarray(pieces[0]['image'].meta[PatchKeys.LOCATION]).T
        offset = tuple(int(v) for v in pieces[0]['image'].meta['offset'])
        if locations.shape != (len(x), 2):
            raise ValueError('Native raw tile location metadata is not aligned with tile rows')
        available = int(np.prod([(s-o)//self.tile_size for s, o in zip(info['shape'][1:], offset)]))
        padded = max(0, self.tile_count-available) if training else 0
        geometry = {'row_column_at_level': locations.tolist(), 'offset': offset, 'padded_white_tiles': padded,
                    'level': info['level'], 'downsample': info['downsample']}
        info = {**info, 'tiles': len(x), 'tile_size': self.tile_size, 'padded_white_tiles': padded,
                'locations_sha256': digest_object(geometry)}
        if x.shape[1:] != (3, self.tile_size, self.tile_size) or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('Native raw RGB tile pipeline produced invalid input')
        return x, info

    def forward(self, x, chunked=False):
        if x.ndim != 5 or not x.is_floating_point() or x.shape[2:] != (3, self.tile_size, self.tile_size) or min(x.shape[:2]) < 1 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('MONAI expects finite [B,N,3,H,W] RGB in[0,1]')
        b, n = x.shape[:2]
        self.reserve(b, n, training=self.training or not chunked)
        if chunked and self.training:
            raise ValueError('Chunking TRAIN changes native backbone BN; it is permitted only in eval')
        if not chunked or n <= self.eval_chunk:
            logits = self.network(x)
        else:
            features, pyramid = [], []
            for start in range(0, n, self.eval_chunk):
                part = x[:, start:start+self.eval_chunk]
                features.append(self.network(part, no_head=True))
                if self.mode == 'att_trans_pyramid':
                    # Native head uses only spatial means. Pool per chunk before retaining feature maps.
                    # Reshape B and N explicitly so B>1 keeps bag order across chunks.
                    pyramid.append({key: value.mean((2, 3), keepdim=True).reshape(b, len(part[0]), -1, 1, 1)
                                    for key, value in self.network.extra_outputs.items()})
            if pyramid:
                self.network.extra_outputs.update({key: torch.cat([p[key] for p in pyramid], dim=1).flatten(0, 1)
                                                   for key in pyramid[0]})
            logits = self.network.calc_head(torch.cat(features, dim=1))
        if not torch.isfinite(logits).all():
            raise ValueError('MONAI produced nonfinite ordinal logits')
        return {'ordinal_logits': logits}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:
            raise ValueError('MONAI cannot merge several independent full slides into one image bag')
        context = self.prediction_precision(torch.device(device))
        images, info = self.images(rows[0], training=train)
        with context:
            result = self(images.unsqueeze(0).to(device), chunked=not train)
        result['raw_input'] = info
        return result

    def prediction_precision(self, device):
        if not self.amp:
            return nullcontext()
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('AMP MONAI prediction requires an actual CUDA device; no silent CPU precision substitution')
        return torch.autocast(device_type='cuda', enabled=True)

    def prediction_fields(self, result, row):
        logits = result['ordinal_logits']
        with self.prediction_precision(logits.device):
            loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets([row['target']], self.thresholds, logits.device))
        return {'monai_bce': float(loss),
                'raw_tiles': result['raw_input']['tiles'], 'raw_level': result['raw_input']['level'],
                'raw_padded_white_tiles': result['raw_input']['padded_white_tiles'],
                'raw_locations_sha256': result['raw_input']['locations_sha256']}

    def validation_metrics(self, frame):
        return {'monai_bce': float(frame.monai_bce.mean())}

    def make_optimizer(self, lr, weight_decay):
        params = self.network.parameters()
        if self.mode.startswith('att_trans'):
            params = [{'params': list(self.network.attention.parameters())+list(self.network.myfc.parameters())+list(self.network.net.parameters())},
                      {'params': list(self.network.transformer.parameters()), 'lr': 6e-6, 'weight_decay': .1}]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_epochs, eta_min=0)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch = epoch

    def finish_epoch(self, bundle, device, config):
        self.scheduler.step()

    def get_extra_state(self):
        return {'optimizer_steps': self.optimizer_steps, 'training_bags': self.training_bags, 'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict() if self.scheduler else self.scheduler_state,
                'scaler': self.scaler.state_dict() if self.scaler else self.scaler_state}

    def set_extra_state(self, state):
        if set(state) != {'optimizer_steps', 'training_bags', 'epoch', 'scheduler', 'scaler'} or any(type(state[k]) is not int or state[k] < 0 for k in ['optimizer_steps', 'training_bags']):
            raise ValueError('Invalid MONAI checkpoint state')
        self.optimizer_steps, self.training_bags, self.epoch = state['optimizer_steps'], state['training_bags'], state['epoch']
        self.scheduler_state, self.scaler_state = state['scheduler'], state['scaler']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
            self.scaler.load_state_dict(self.scaler_state)

    def protocol(self):
        return {'method': self.profile, 'family': 'monai_mil', 'mode': self.mode, 'backbone': 'full trainable ResNet50',
                'initialization': self.bundle.raw_images.backbone, 'trans_blocks': self.trans_blocks, 'trans_dropout': self.trans_dropout,
                'batch_size': self.batch_size, 'tile_size': self.tile_size, 'tile_count': self.tile_count, 'eval_chunk': self.eval_chunk,
                'task': 'cumulative ordinal BCE, K-1 sigmoid thresholds for K grades; round(sum(sigmoid))',
                'optimizer': 'AdamW3e-5; Transformer6e-6/decay.1; backbone/attention/head decay0; epoch-end cosine to0',
                'optimizer_steps': self.optimizer_steps, 'training_bags': self.training_bags,
                'scheduler': self.scheduler.state_dict() if self.scheduler else self.scheduler_state,
                'amp': self.amp, 'data': 'entire declared image level; native random-remainder offset/min-intensity grid and per-tile flips/rot90; native white count padding; eval all foreground grid tiles',
                'normalization': 'RGBuint8 to[0,1], no ImageNet mean/std or stain conversion',
                'random_stream': 'one synchronous worker, each bag source Compose seeded by checkpointed CPU Torch RNG',
                'selection': 'strict validation slide QWK; full budget; first finite best saved even if QWK<=0; test sealed',
                'scope': 'complete published MONAI tutorial baseline; paper five-model ensemble/pseudo-label extension not claimed implemented',
                'repairs': ['coherent best/last/optimizer/scheduler/scaler/RNG; save first finite best rather than require QWK>0',
                            'eval retains only spatially pooled hook maps, same native pyramid head; B/N chunk order preserved'],
                'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}


def monai_epoch(model, bundle, groups, optimizer, device, cfg):
    if any(len(g) != 1 for g in groups):
        raise ValueError('MONAI requires independent whole-slide RGB bags')
    loader = DataLoader(ImageBags(model, [g[0] for g in groups]), batch_size=model.batch_size, shuffle=True, num_workers=0)
    losses = []
    for item in loader:
        images, target = item['image'].to(device), item['label'].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=model.amp):
            result = model(images)
            loss = F.binary_cross_entropy_with_logits(result['ordinal_logits'], target)
        if not torch.isfinite(loss):
            raise ValueError('MONAI ordinal BCE is nonfinite')
        model.scaler.scale(loss).backward()
        if not model.amp and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('MONAI nonfinite gradient')
        old_scale = model.scaler.get_scale()
        model.scaler.step(optimizer)
        model.scaler.update()
        check_update_finite(model, optimizer)
        model.optimizer_steps += int(model.scaler.get_scale() >= old_scale)
        model.training_bags += len(images)
        losses.append({'loss': float(loss.detach())})
    if not losses:
        raise ValueError('MONAI has no TRAIN bags')
    return losses

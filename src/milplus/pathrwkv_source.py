"""Native PathRWKV blocks and heads, with an exact factorized position cache."""
import contextlib
import functools
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from .pathrwkv_backend import COMMIT, HEAD_SIZE
from .upstream import load_definitions, verified_source


@functools.lru_cache(None)
def position_source():
    return load_definitions('pathrwkv', 'DownStream/model/pe.py',
        ['get_1d_sincos_pos_embed_from_grid', 'get_2d_sincos_pos_embed_from_grid',
         'get_2d_sincos_pos_embed', 'SlidePE'], {'torch': torch, 'nn': nn})


class FactorizedSlidePE(nn.Module):
    """The source2D table separates into two identical1D tables, without approximation."""
    def __init__(self, embed_dim, slide_ngrids, MTL_token_num=0):
        super().__init__()
        if type(slide_ngrids) is not int or slide_ngrids < 1 or embed_dim % 4:
            raise ValueError('PathRWKV positions require positive integer grid and a width divisible by4')
        if type(MTL_token_num) is not int or MTL_token_num < 0:
            raise ValueError('Position prefix count must be nonnegative')
        self.slide_ngrids, self.MTL_token_num = slide_ngrids, MTL_token_num
        values = position_source().get_1d_sincos_pos_embed_from_grid(
            embed_dim//2, torch.arange(slide_ngrids, dtype=torch.float32))
        # Compute on CPU, like the author's constructor, then transfer with the
        # model. CUDA trig is not silently substituted for the CPU position table.
        self.register_buffer('one_dim', values, persistent=False)

    def indices(self, coords):
        if coords.ndim != 3 or coords.shape[-1] != 2 or not torch.isfinite(coords).all():
            raise ValueError('PathRWKV positions require finite B,N,2 coords_yx')
        ids = torch.floor(coords/224)
        if (ids < 0).any() or (ids >= self.slide_ngrids).any():
            raise ValueError('PathRWKV coordinate lies outside its declared physical position grid')
        return ids.long()

    def coords_to_pos(self, coords):
        ids = self.indices(coords)
        return ids[..., 0]*self.slide_ngrids+ids[..., 1]+self.MTL_token_num

    def forward(self, x, coords, MTL_tokens=None):
        ids = self.indices(coords)
        if x.shape[:-1] != ids.shape[:-1]:
            raise ValueError('PathRWKV coordinate/token alignment mismatch')
        value = torch.cat([self.one_dim[ids[..., 1]], self.one_dim[ids[..., 0]]], -1)
        result = x+value
        if MTL_tokens is not None:
            if MTL_tokens.shape[1:] != (self.MTL_token_num, x.shape[-1]):
                raise ValueError('PathRWKV position prefix shape mismatch')
            result = torch.cat([MTL_tokens.expand(x.shape[0], -1, -1), result], dim=1)
        return result


def source_modules(backend, inference_chunk):
    _, sha = verified_source('pathrwkv')
    if sha != COMMIT:
        raise ValueError('PathRWKV source version mismatch')
    scope = {'torch': torch, 'nn': nn, 'HEAD_SIZE': HEAD_SIZE,
             'CHUNK_LEN_INFERENCE': inference_chunk, 'SlidePE': FactorizedSlidePE,
             'cuda_wkv_parallel': lambda r, k, v, w, u: backend(r, k, v, w, u),
             'cuda_wkv_state': lambda r, k, v, w, u, s: backend(r, k, v, w, u, s)}
    backbone = load_definitions('pathrwkv', 'DownStream/model/pathrwkv.py',
        ['TimeMix', 'ChannelMix', 'Block', 'PathRWKVv6'], scope)
    pipeline = load_definitions('pathrwkv', 'DownStream/utils/pipeline.py', ['Head', 'WSIPipeline'],
        {'torch': torch, 'nn': nn, 'optim': optim, 'math': math,
         'LambdaLR': LambdaLR, 'LightningModule': nn.Module, 'Path': Path,
         'PathRWKV': backbone.PathRWKVv6})
    cox = load_definitions('pathrwkv', 'DownStream/utils/utils.py', ['CoxPHLoss'],
                           {'torch': torch, 'nn': nn})
    return backbone, pipeline, cox


class NativeNetwork(nn.Module):
    def __init__(self, args, backend, hidden=768, depth=2, grids=1200, inference_chunk=2000):
        super().__init__()
        self.args = args
        self.source_backbone, self.source_pipeline, cox = source_modules(backend, inference_chunk)
        self.backbone = self.source_backbone.PathRWKVv6(args, embed_dim=hidden,
                                                       n_layers=depth, slide_ngrids=grids)
        self.input_projector = nn.Sequential(nn.Linear(args.input_dim, hidden), nn.ReLU(), nn.Dropout(.25))
        self.head = self.source_pipeline.Head(args, hidden)
        self.source_pipeline.WSIPipeline.init_weights(self)
        functions = {'cls': nn.CrossEntropyLoss, 'reg': nn.SmoothL1Loss, 'surv': cox.CoxPHLoss}
        self.loss_functions = nn.ModuleDict({name: functions[spec['type']]() for name, spec in args.tasks.items()})

    def forward(self, x, coords):
        return self.source_pipeline.WSIPipeline.forward(self, x, coords)

    def source_loss(self, predictions, targets):
        return self.source_pipeline.WSIPipeline.compute_loss(self, predictions, targets)

    def optimizer(self):
        return self.source_pipeline.WSIPipeline.configure_optimizers(self)


@contextlib.contextmanager
def carry_token_shifts(backbone):
    """Explicit repair: carry normalized TimeMix AND ChannelMix shift inputs."""
    previous, handles = {}, []

    def shift(module, args, output):
        value = output.clone()
        if module in previous:
            value[:, 0] = previous[module]
        previous[module] = args[0][:, -1].clone()
        return value

    try:
        for block in backbone.blocks:
            for module in [block.time_mix.time_shift, block.channel_mix.time_shift]:
                handles.append(module.register_forward_hook(shift))
        yield previous
    finally:
        for handle in handles:
            handle.remove()

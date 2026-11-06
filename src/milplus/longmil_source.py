"""Pinned public LongMIL-v1 and explicit paper-v2 reference primitives.

The public v1 ALiBi model is not the unpublished NeurIPS local/chunked model.
No generic dense-attention fallback is used by a public CUDA execution.
"""
import copy
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import autograd, nn
from torch.nn import functional as F

from .io import digest_file
from .upstream import load_definitions, verified_source

COMMIT = '457b1cf6e33a05b3c8359e1fe98114cd8241f772'
ALIBI_FILE_SHA256 = 'ad94b933bcbb1f01671a96fb78380e6b48e99be50a40a9ad328a74922030269a'
ALIBI_TENSOR_SHA256 = '137574db94efa7d3336d91052f34390903b5f62252e095f8e2cb2d66324bdb8a'


def checked_source():
    root, commit = verified_source('longmil')
    if commit != COMMIT:
        raise ValueError('LongMIL requires the pinned original public v1 source')
    return root


def cuda_attention(q, k, v, bias=None, p=0.):
    """Actual original xFormers kernel; missing/unsupported kernels fail closed."""
    if q.device.type != 'cuda':
        raise ValueError('Full LongMIL execution requires actual CUDA xFormers')
    from xformers.ops import memory_efficient_attention
    return memory_efficient_attention(q, k, v, attn_bias=bias, p=p)


def dense_oracle(q, k, v, bias=None, p=0.):
    """Independent small-input equation oracle; never a runnable profile backend."""
    if p != 0:
        raise ValueError('The dense equation oracle excludes stochastic dropout')
    weights = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(q.shape[-1])
    if bias is not None:
        weights = weights + bias
    return torch.einsum('bhqk,bkhd->bqhd', weights.softmax(-1), v)


def aligned_bias(value):
    """xFormers requires aligned key strides, including for odd bag lengths."""
    size = value.shape[-1]
    padding = 8 - size % 8
    return F.pad(value, (0, padding))[..., :size]


def grid_positions(coords, step, origin=(0., 0.)):
    if (not torch.is_tensor(coords) or coords.ndim != 2 or coords.shape[1] != 2
            or len(coords) == 0 or not torch.isfinite(coords).all()
            or type(step) not in (int, float) or not math.isfinite(step) or step <= 0):
        raise ValueError('LongMIL needs real aligned finite level-0 XY coordinates and a declared physical patch step')
    offset = torch.as_tensor(origin, dtype=torch.float64, device=coords.device)
    if offset.shape != (2,) or not torch.isfinite(offset).all():
        raise ValueError('Declare the actual level-0 grid origin XY')
    cells = (coords.to(torch.float64) - offset) / step
    integer = cells.round()
    if (cells - integer).abs().max() > 1e-6 or (integer < 0).any():
        raise ValueError('Coordinates do not match the declared physical patch grid; no invented/pseudo grid')
    result = integer.to(torch.int64)
    if len(torch.unique(result, dim=0)) != len(result):
        raise ValueError('LongMIL physical patch positions must be unique within a slide')
    return result


def local_plan(positions, chunk=128, radius=10):
    """Exact 2D Euclidean neighbourhoods, without constructing an N by N mask."""
    from scipy.spatial import cKDTree
    if type(chunk) is not int or chunk < 1 or radius != 10:
        raise ValueError('Paper reference uses native radius10 and positive query chunks')
    xy = positions.detach().cpu().numpy().astype(np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Provide actual two-dimensional positions')
    tree = cKDTree(xy)
    blocks = []
    for start in range(0, len(xy), chunk):
        stop = min(start + chunk, len(xy))
        neighbours = tree.query_ball_point(xy[start:stop], r=radius, p=2, eps=0)
        keys = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in neighbours]))
        if not len(keys):
            raise ValueError('Every local query must retain at least itself')
        blocks.append((start, stop, torch.from_numpy(keys)))
    return blocks


def chunked_attention(q, k, v, positions, *, local, chunk=128, operator=cuda_attention):
    """Exact masked local attention, or global ALiBi2D, in query/key chunks.

    Sparse local key unions omit all disallowed pairs. The global stage still
    evaluates every retained pair; query chunking changes memory, not context.
    Paper-v2 uses one head, no dropout and radius10 (Appendix A.5.3).
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError('LongMIL requires one complete bag of aligned Q/K/V')
    if positions.shape != (q.shape[1], 2) or type(chunk) is not int or chunk < 1:
        raise ValueError('Attention positions and actual query chunk are invalid')
    n, heads = q.shape[1], q.shape[2]
    blocks = local_plan(positions, chunk) if local else [
        (a, min(a + chunk, n), torch.arange(n)) for a in range(0, n, chunk)]
    xy = positions.to(device=q.device, dtype=torch.float32)
    outputs = []
    for start, stop, cpu_keys in blocks:
        keys = cpu_keys.to(q.device)
        delta = xy[start:stop, None, :] - xy[None, keys, :]
        squared = delta.square().sum(-1)
        if local:
            value = torch.zeros_like(squared).masked_fill(squared > 100, -torch.inf)
            bias = value[None, None].expand(1, heads, -1, -1)
        else:
            # Original author's ALiBi slope convention; true Euclidean float
            # distance in the paper reference, not v1's int8 truncation.
            slopes = torch.tensor([2. ** (-8. * (h + 1) / heads) for h in range(heads)],
                                  dtype=torch.float32, device=q.device)
            bias = -squared.sqrt()[None, None] * slopes[None, :, None, None]
        bias = aligned_bias(bias.contiguous())
        outputs.append(operator(q[:, start:stop], k[:, keys], v[:, keys], bias, p=0.))
    return torch.cat(outputs, dim=1)


def spatial_pool2(features, positions):
    """Nonoverlapping real 2x2 cell mean pooling; holes are never fake tokens.

    The paper leaves the reduction and incomplete-cell policy unspecified.
    The reference explicitly averages observed patches and their coordinates.
    """
    if features.ndim != 3 or features.shape[0] != 1 or positions.shape != (features.shape[1], 2):
        raise ValueError('Pool actual aligned features and physical positions')
    cells = positions.detach().cpu().to(torch.int64)
    if len(torch.unique(cells, dim=0)) != len(cells):
        raise ValueError('Cannot pool duplicate physical patches')
    groups = {}
    for i, pair in enumerate((cells // 2).tolist()):
        groups.setdefault(tuple(pair), []).append(i)
    ordered = [groups[pair] for pair in sorted(groups)]
    if any(len(group) > 4 for group in ordered):
        raise ValueError('A real 2x2 cell contains at most four unique patch positions')
    # No scatter-add atomic reductions and no insertion of missing tissue.
    indices = torch.tensor([g + [g[0]] * (4 - len(g)) for g in ordered],
                           dtype=torch.long, device=features.device)
    mask = torch.tensor([[1.] * len(g) + [0.] * (4 - len(g)) for g in ordered],
                        dtype=features.dtype, device=features.device)
    counts = mask.sum(-1, keepdim=True)
    pooled = (features[0, indices] * mask[..., None]).sum(1) / counts
    xy = positions.to(features.device, dtype=torch.float32)
    centres = (xy[indices] * mask[..., None]).sum(1) / counts
    return pooled.unsqueeze(0), centres, ordered


def public_definitions(load_alibi, operator=cuda_attention):
    """Execute every original used v1 definition, with audited device/IO fixes."""
    checked_source()
    rotary = load_definitions('longmil', 'rotary.py',
        ['position_encoding_1d', 'apply_rotary_position_embeddings', 'Rotary2D'],
        {'torch': torch, 'math': math})
    return load_definitions('longmil', 'LongMIL.py',
        ['exists', 'Attention', 'Mlp', 'drop_path', 'DropPath', 'TransBlock', 'get_slopes', 'LongMIL'],
        {'copy': copy, 'math': math, 'np': np, 'torch': torch, 'nn': nn, 'F': F,
         'autograd': autograd, 'Rotary2D': rotary.Rotary2D,
         'apply_rotary_position_embeddings': rotary.apply_rotary_position_embeddings,
         'memory_efficient_attention': operator, '_load_native_alibi': load_alibi},
        replacements=[("torch.load('./alibi_tensor_core.pt').cuda()", '_load_native_alibi()'),
                      ('self.slope_m[None, None, :].cuda()', 'self.slope_m[None, None, :].to(x.device)'),
                      ('.cuda()', '.to(x.device)'),
                      ("            except:\n                print('xformer error')",
                       "            except Exception as error:\n                raise RuntimeError('Original LongMIL xFormers attention failed; no silent identity fallback') from error")])


def load_full_alibi(path, expected_file_sha256=ALIBI_FILE_SHA256):
    path = Path(path)
    if digest_file(path) != expected_file_sha256:
        raise ValueError('Use the fully verified original 90000x90000 int8 ALiBi table')
    return _loaded_native_alibi(str(path), expected_file_sha256)


@lru_cache(maxsize=2)
def _loaded_native_alibi(path, expected_file_sha256):
    import hashlib
    table = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if table.shape != (90000, 90000) or table.dtype != torch.int8:
        raise ValueError('Full original table dimensions/dtype are mandatory')
    digest = hashlib.sha256()
    array = table.numpy()
    for start in range(0, len(table), 128):
        digest.update(array[start:start+128].tobytes())
    if digest.hexdigest() != ALIBI_TENSOR_SHA256:
        raise ValueError('Every original ALiBi tensor byte must match the full native source formula')
    return table


class PaperAttention(nn.Module):
    def __init__(self, local, chunk=128, operator=cuda_attention):
        super().__init__()
        self.qkv = nn.Linear(384, 384 * 3, bias=False)
        self.proj = nn.Linear(384, 384)
        self.local, self.chunk, self.operator = local, chunk, operator

    def forward(self, value, positions):
        q, k, v = self.qkv(value).reshape(1, len(positions), 3, 1, 384).unbind(2)
        result = chunked_attention(q, k, v, positions, local=self.local,
                                   chunk=self.chunk, operator=self.operator)
        return self.proj(result.reshape(1, len(positions), 384))


class PaperBlock(nn.Module):
    def __init__(self, local, chunk=128, operator=cuda_attention):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.attn = PaperAttention(local, chunk, operator)
        self.norm2 = nn.LayerNorm(384)
        self.mlp = nn.Sequential(nn.Linear(384, 1536), nn.GELU(), nn.Linear(1536, 384))

    def forward(self, value, positions):
        value = value + self.attn(self.norm(value), positions)
        return value + self.mlp(self.norm2(value))


class PaperLongMIL(nn.Module):
    """Full 2-local / actual2x2-pool / 1-global NeurIPS reference, not author v2 code."""
    def __init__(self, dim, outputs, chunk=128, operator=cuda_attention):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, 384), nn.ReLU())
        self.local1 = PaperBlock(True, chunk, operator)
        self.local2 = PaperBlock(True, chunk, operator)
        self.global_block = PaperBlock(False, chunk, operator)
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, outputs)
        self.last_geometry = None

    def forward(self, features, positions):
        value = self.project(features).unsqueeze(0)
        value = self.local1(value, positions)
        value = self.local2(value, positions)
        pooled, centres, groups = spatial_pool2(value, positions)
        value = self.global_block(pooled, centres)
        self.last_geometry = {'input_patches': len(features), 'pooled_cells': len(groups),
                              'observed_patches_per_cell': [len(g) for g in groups]}
        embedding = self.norm(value.mean(1))
        return {'logits': self.head(embedding), 'embedding': embedding}

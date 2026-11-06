"""Explicit WKV6 tensor reference and the pinned author's native CUDA kernels."""
import functools
import os
import threading
from pathlib import Path

import torch

from .io import digest_file
from .upstream import load_definitions, pinned_text, verified_source

COMMIT = '8818fa5711666bd8d30672f02bdcd98634389b70'
HEAD_SIZE = 64
NATIVE_SEQUENCE_LIMIT = 2000
_BUILD_LOCK = threading.Lock()


def recurrence(r, k, v, w, u, state=None):
    """B,T,H,N inputs; state axes B,H,value,key. Functional, differentiable."""
    if r.ndim != 4 or any(x.shape != r.shape for x in [k, v, w]):
        raise ValueError('WKV6 requires matching B,T,H,N inputs')
    b, t, h, n = r.shape
    if min(b, t, h, n) < 1 or u.shape != (h, n):
        raise ValueError('WKV6 dimensions must be positive and bonus must be H,N')
    if state is None:
        state = r.new_zeros((b, h, n, n))
    if state.shape != (b, h, n, n):
        raise ValueError('WKV6 state axes must be B,H,value,key')
    if any(x.device != r.device or x.dtype != r.dtype for x in [k, v, w, u, state]):
        raise ValueError('WKV6 input dtype/device mismatch')
    values = []
    decay = (-w.exp()).exp()
    for i in range(t):
        kv = v[:, i, :, :, None]*k[:, i, :, None, :]
        values.append(((state+u[None, :, None, :]*kv)*r[:, i, :, None, :]).sum(-1))
        state = state*decay[:, i, :, None, :]+kv
    return torch.stack(values, dim=1), state


@functools.lru_cache(None)
def native_definitions():
    _, sha = verified_source('pathrwkv')
    if sha != COMMIT:
        raise ValueError('PathRWKV source differs from the reviewed commit')
    return load_definitions('pathrwkv', 'DownStream/model/pathrwkv.py',
        ['WKV_6_Parallel', 'WKV_6_State', 'cuda_wkv_parallel', 'cuda_wkv_state'],
        {'torch': torch, 'HEAD_SIZE': HEAD_SIZE})


@functools.lru_cache(None)
def load_native(device_index):
    """Compile Git objects, not unchecked worktree files; no automatic fallback."""
    if not torch.cuda.is_available():
        raise ValueError('PathRWKV native CUDA backend requires CUDA; select explicit torch reference for CPU')
    root, sha = verified_source('pathrwkv')
    if sha != COMMIT:
        raise ValueError('PathRWKV native source mismatch')
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('Pinned native WKV6 requires the default float32 allocation dtype')
    from torch.utils.cpp_extension import get_default_build_root, load
    directory = Path(get_default_build_root())/('milplus_pathrwkv_'+COMMIT)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with _BUILD_LOCK, torch.cuda.device(device_index):
        major, minor = torch.cuda.get_device_capability(device_index)
        arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
        try:
            os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
            for stem in ['wkv6', 'wkv6state']:
                paths = []
                for suffix in ['.cu', '_op.cpp']:
                    name = stem+suffix
                    path = directory/name
                    body = pinned_text(root, sha, 'DownStream/model/cuda/'+name)
                    if not path.is_file() or path.read_text() != body:
                        path.write_text(body)
                    paths.append(str(path)); hashes[name] = digest_file(path)
                flags = ['-res-usage', '--use_fast_math', '-O3', '-Xptxas', '-O3',
                    '--extra-device-vectorization', '-D_N_=64', '-D_T_=2000']
                load(name='milplus_pathrwkv_'+stem, is_python_module=False,
                     sources=paths, extra_cuda_cflags=flags, verbose=False)
        finally:
            if arch is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    return {'source_commit': sha, 'cuda_sources_sha256': hashes,
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'architecture': [major, minor], 'sequence_limit': NATIVE_SEQUENCE_LIMIT}


class Backend:
    def __init__(self, name='torch'):
        if name not in {'torch', 'native'}:
            raise ValueError('PathRWKV backend must be explicit torch or native; no auto fallback')
        self.name = name
        self.receipt = None

    def __call__(self, r, k, v, w, u, state=None):
        if r.ndim != 3 or any(x.shape != r.shape for x in [k, v, w]):
            raise ValueError('PathRWKV kernel expects matching B,T,C tensors')
        b, t, c = r.shape
        if min(b, t, c) < 1 or c % HEAD_SIZE or u.shape != (c//HEAD_SIZE, HEAD_SIZE):
            raise ValueError('PathRWKV source uses64-channel heads and a matching bonus')
        if any(x.device != r.device for x in [k, v, w, u]) or (state is not None and state.device != r.device):
            raise ValueError('PathRWKV kernel tensors must share one device')
        if state is not None and state.shape != (b, c//HEAD_SIZE, HEAD_SIZE, HEAD_SIZE):
            raise ValueError('PathRWKV matrix state shape mismatch')
        # The source autograd operator always casts its inputs tofloat32.
        values = [x.float().contiguous() for x in [r, k, v, w, u]]
        if state is not None:
            state = state.float().contiguous()
        if self.name == 'native':
            if not r.is_cuda:
                raise ValueError('Native WKV6 cannot execute on CPU; no fallback')
            if t > NATIVE_SEQUENCE_LIMIT:
                raise ValueError('Pinned native WKV6 supports at most2000 tiles per kernel call')
            if state is not None and torch.is_grad_enabled():
                raise ValueError('Native stateful WKV6 is registered for no-grad inference; source state-backward is not certified')
            self.receipt = load_native(r.device.index)
            source = native_definitions()
            return source.cuda_wkv_parallel(*values) if state is None else source.cuda_wkv_state(*values, state)
        x, new = recurrence(*[z.reshape(b, t, -1, HEAD_SIZE) for z in values[:4]], values[4], state)
        x = x.reshape(b, t, c)
        return x if state is None else (x, new)

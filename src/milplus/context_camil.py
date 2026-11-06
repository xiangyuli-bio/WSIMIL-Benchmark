"""TensorFlow CAMIL equations ported with their literal axis conventions."""
import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F

from .io import digest_file, digest_object
from .upstream import verified_source


def camil_neighbors(features, coords, chunk=128):
    """Author 16 spatial candidates -> cosine distance weights -> first 8 edges.

    Sorting and normalization precede truncation, including the original
    max-neighbor self weight. Work in bounded row chunks instead of N*N storage.
    """
    x = np.asarray(features, dtype=np.float64)
    xy = np.asarray(coords, dtype=np.float64)
    if x.ndim != 2 or not len(x) or xy.shape != (len(x), 2) or not np.isfinite(xy).all():
        raise ValueError('Context CAMIL requires real aligned [N,2] coordinates and nonempty [N,D] features')
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('Context CAMIL requires distinct patch coordinates')
    if not np.isfinite(x).all():
        raise ValueError('Context CAMIL features must be finite')
    if len(x) == 1:
        return np.zeros((1, 1), dtype=np.int64), np.ones((1, 1), dtype=np.float32)
    count = min(16, len(x))
    norms = np.linalg.norm(x, axis=1)
    ids, weights = [], []
    for begin in range(0, len(x), chunk):
        end = min(begin+chunk, len(x))
        indices = np.argsort(cdist(xy[begin:end], xy), axis=1)[:, :count]
        # Do not project or learn these features: author preprocessing computes
        # cosine distances before the trainable network is invoked.
        dot = np.einsum('bd,bkd->bk', x[begin:end], x[indices])
        norm = norms[begin:end, None]*norms[indices]
        distance = np.clip(1-np.divide(dot, norm, out=np.ones_like(dot), where=norm > 0), 0, 2)
        distance = distance[:, 1:]
        length = np.linalg.norm(distance, axis=1, keepdims=True)
        similarity = np.exp(-np.divide(distance, length, out=np.zeros_like(distance), where=length > 0))
        value = np.concatenate([similarity.max(1, keepdims=True), similarity], axis=1)
        ids.append(indices[:, :8]); weights.append(value[:, :8])
    return np.concatenate(ids), np.concatenate(weights).astype(np.float32)


def cached_neighbors(features, coords, directory):
    """Content-addressed graph preprocessing; cached arrays are never trusted unchecked."""
    x, xy = np.ascontiguousarray(features), np.ascontiguousarray(coords)
    key = digest_object({'implementation': digest_file(Path(__file__)),
                         'x': [x.shape, str(x.dtype), hashlib.sha256(x.tobytes()).hexdigest()],
                         'xy': [xy.shape, str(xy.dtype), hashlib.sha256(xy.tobytes()).hexdigest()]})
    directory = Path(directory).expanduser()
    path = directory/(key+'.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            indices, weights = data['indices'], data['weights']
            checksum = hashlib.sha256(indices.tobytes()+weights.tobytes()).hexdigest()
            if (str(data['key']) != key or str(data['checksum']) != checksum or
                    indices.dtype != np.int64 or weights.dtype != np.float32 or
                    indices.shape != (len(x), min(8, len(x))) or weights.shape != indices.shape or
                    (indices < 0).any() or (indices >= len(x)).any() or not np.isfinite(weights).all()):
                raise ValueError(f'Context CAMIL graph cache is invalid: {path}')
            return indices, weights
    indices, weights = camil_neighbors(x, xy)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, indices=indices, weights=weights, key=np.array(key),
                 checksum=np.array(hashlib.sha256(indices.tobytes()+weights.tobytes()).hexdigest()))
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return indices, weights


class ContextNystrom(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64, landmarks=256, iterations=6, kernel=33):
        super().__init__()
        self.heads, self.dim_head, self.landmarks, self.iterations = heads, dim_head, landmarks, iterations
        self.qkv = nn.Linear(dim, heads*dim_head*3, bias=False)
        self.output = nn.Linear(heads*dim_head, dim)
        # In the author TF code v is [B,heads,N,dim_head] and Conv2D is NHWC.
        # Its kernel acts on the heads axis; groups split dim_head channels.
        # A conventional PyTorch Nystrom depthwise token convolution differs.
        self.residual = nn.Conv2d(dim_head, dim_head, (kernel, 1),
                                  groups=heads, padding=(kernel//2, 0), bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        pad = (-n) % self.landmarks
        if pad:
            x = F.pad(x, (0, 0, pad, 0))
        q, k, v = [a.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)
                   for a in self.qkv(x).chunk(3, -1)]
        q = q * self.dim_head**-.5
        # Padded length is a multiple of landmarks, so TF Addons adaptive
        # average pooling is exactly this grouping, not unpadded pooling.
        ql, kl = [a.reshape(b, self.heads, self.landmarks, -1, self.dim_head).mean(3) for a in (q, k)]
        a1, a2, a3 = [(a @ z.transpose(-2, -1)).softmax(-1) for a, z in [(q, kl), (ql, kl), (ql, k)]]
        z = a2.transpose(-2, -1)/(a2.abs().sum(-1).max()*a2.abs().sum(-2).max())
        eye = torch.eye(self.landmarks, dtype=x.dtype, device=x.device)
        for _ in range(self.iterations):
            az = a2 @ z
            z = .25*z @ (13*eye-az @ (15*eye-az @ (7*eye-az)))
        out = (a1 @ z) @ (a3 @ v)
        out = out + self.residual(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.output(out.transpose(1, 2).reshape(b, -1, self.heads*self.dim_head))
        return out[:, -n:]


class KerasAdam(torch.optim.Optimizer):
    """Keras Adam epsilon position, used to preserve the TF training equation."""
    def __init__(self, params, lr=.0002, betas=(.9, .999), eps=1e-7):
        super().__init__(params, {'lr': lr, 'betas': betas, 'eps': eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError('KerasAdam adapter expects dense model gradients')
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state['step'] += 1
                state['m'].add_(p.grad-state['m'], alpha=1-b1)
                state['v'].add_(p.grad.square()-state['v'], alpha=1-b2)
                alpha = group['lr']*math.sqrt(1-b2**state['step'])/(1-b1**state['step'])
                p.addcdiv_(state['m'], state['v'].sqrt()+group['eps'], value=-alpha)
        return loss


class ContextCAMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, heads=8, dim_head=64, landmarks=256,
                 attention_dim=256, pool_dim=128, kernel=33, subtyping=None, graph_cache=True):
        super().__init__()
        verified_source('camil_context')
        if any(not isinstance(v, int) or v < 1 for v in
               [dim, classes, hidden, heads, dim_head, landmarks, attention_dim, pool_dim, kernel]):
            raise ValueError('Context CAMIL sizes must be positive integers')
        if classes < 2 or dim_head % heads or kernel % 2 == 0:
            raise ValueError('Context CAMIL requires classes>=2, dim_head divisible by heads and odd kernel')
        self.subtyping = classes > 2 if subtyping is None else subtyping
        if not self.subtyping and classes != 2:
            raise ValueError('Binary CAMIL head requires exactly two classes')
        self.project = nn.Identity() if dim == hidden else nn.Linear(dim, hidden)
        self.nystrom = ContextNystrom(hidden, heads, dim_head, landmarks, kernel=kernel)
        self.query = nn.Linear(hidden, attention_dim, bias=False)
        self.key = nn.Linear(hidden, attention_dim, bias=False)
        self.value = nn.Linear(hidden, hidden)
        self.pool_v = nn.Linear(hidden, pool_dim, bias=False)
        self.pool_u = nn.Linear(hidden, pool_dim, bias=False)
        self.pool_w = nn.Linear(pool_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden, classes if self.subtyping else 1)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # TF 2.15 reuses unseeded initializer objects for equal-shaped V/U and Q/K.
        with torch.no_grad():
            self.pool_u.weight.copy_(self.pool_v.weight)
            self.key.weight.copy_(self.query.weight)
        self.optimizer, self.best_loss, self.plateau_wait = None, float('inf'), 0
        self.graph_cache = graph_cache
        self.sampled_training = False

    def configure_training(self, config):
        self.sampled_training = config['max_instances'] is not None

    def make_optimizer(self, lr, weight_decay):
        if weight_decay != 0:
            raise ValueError('Official context CAMIL tape omits layer regularizers; weight_decay must be zero')
        self.optimizer = KerasAdam(self.parameters(), lr=lr)
        return self.optimizer

    def after_validation(self, metrics):
        current = metrics['log_loss']
        if current < self.best_loss-1e-4:
            self.best_loss, self.plateau_wait = current, 0
        else:
            self.plateau_wait += 1
            if self.plateau_wait >= 10:
                for group in self.optimizer.param_groups:
                    group['lr'] *= .2
                self.plateau_wait = 0

    def forward_edges(self, x, indices, weights):
        dense = self.project(x)
        global_features = self.nystrom(dense[None]).squeeze(0)+dense
        q, k = self.query(global_features), self.key(global_features)
        # Equivalent to sparse adjacency times dense N*N attention, but only
        # materialize selected edges. Chunk to bound [N,8,attention_dim] memory.
        raw_parts = []
        for begin in range(0, len(x), 1024):
            idx = indices[begin:begin+1024]
            scores = (q[begin:begin+1024, None]*k[idx]).sum(-1)/math.sqrt(k.shape[-1])
            raw_parts.append((scores*weights[begin:begin+1024]).sum(-1))
        raw = torch.cat(raw_parts)
        alpha = raw.softmax(0)
        local = alpha[:, None]*self.value(dense)
        gate = torch.sigmoid(-local).square()
        combined = 2*local*gate+2*global_features*(1-gate)
        attention = self.pool_w(self.pool_v(combined).tanh()*self.pool_u(combined).sigmoid()).softmax(0)
        logits = self.classifier((attention*combined).sum(0, keepdim=True))
        if not self.subtyping:
            logits = torch.cat([torch.zeros_like(logits), logits], -1)
        return {'logits': logits, 'attention': attention.T, 'neighbor_scores': raw, 'local_attention': alpha}

    def forward(self, x, coords=None, label=None):
        if coords is None:
            raise ValueError('Context CAMIL requires physical coordinates; sequence-square coordinates are not accepted')
        values, xy = x.detach().cpu().numpy(), coords.detach().cpu().numpy()
        if self.graph_cache and not (self.training and self.sampled_training):
            directory = os.environ.get('MILPLUS_CAMIL_GRAPH_CACHE', '~/.cache/milplus/camil_graphs')
            indices, weights = cached_neighbors(values, xy, directory)
        else:
            indices, weights = camil_neighbors(values, xy)
        return self.forward_edges(x, torch.as_tensor(indices, device=x.device),
                                  torch.as_tensor(weights, device=x.device, dtype=x.dtype))

    def get_extra_state(self):
        return {'best_loss': self.best_loss, 'plateau_wait': self.plateau_wait}

    def set_extra_state(self, state):
        self.best_loss, self.plateau_wait = state['best_loss'], state['plateau_wait']

    def protocol(self):
        return {'source': 'equation port of pinned TensorFlow context-aware CAMIL; not channel CAMIL',
                'graph': '16 coordinate-nearest candidates; cosine distances normalized over 15 nonself candidates; first 8 weighted edges',
                'graph_cache': 'content-addressed features/coordinates/implementation; checksummed; random sampled training bypasses disk cache',
                'global': 'literal TF Nystrom with leading zero padding; grouped NHWC convolution across heads',
                'local': 'neighbor-constrained raw QK sum -> node softmax -> projected input -> squared sigmoid fusion',
                'pool': 'gated attention; binary sigmoid or multiclass softmax',
                'optimizer': 'Keras Adam epsilon position, no weight decay, validation-loss plateau factor .2 patience 10',
                'regularization': 'author tape does not add model.losses; declared Keras L2 is not active',
                'feature_adapter': 'identity at native width; explicit learned linear FM projection otherwise',
                'compatibility': 'integer Conv2D filters; 1D node softmax; singleton batch squeeze fixed; sparse edges replace dense QK',
                'singleton': 'self edge weight one, explicit extension where author graph preprocessing is undefined',
                'precision': 'FP32; no claim of TensorFlow/PyTorch bitwise or GPU equivalence',
                'selection': 'fixed validation metric; default log_loss, patient probability aggregation when configured'}

"""Explicit Mamba2/SSD head-shared recurrence; all cross-chunk states remain differentiable."""
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _chunk(state, x, dt, A, B, C):
    output = []
    decay = (dt*A[None, None]).exp()
    for index in range(x.shape[1]):
        state = state*decay[:, index, :, None, None] + (
            dt[:, index, :, None, None]*x[:, index, :, :, None]*B[:, index, :, None, :])
        output.append((state*C[:, index, :, None, :]).sum(-1))
    return torch.stack(output, 1), state


def ssd_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=True,
             dt_limit=(0., float('inf')), checkpoint_chunk=64):
    if type(checkpoint_chunk) is not int or checkpoint_chunk < 1:
        raise ValueError('SSD checkpoint_chunk must be a positive integer')
    if x.ndim != 4 or not x.shape[1] or dt.shape != x.shape[:3] or A.shape != (x.shape[2],):
        raise ValueError('SSD expects nonempty [batch,length,heads,headdim], per-head dt/A')
    if B.ndim != 4 or C.shape != B.shape or B.shape[:2] != x.shape[:2] or x.shape[2] % B.shape[2]:
        raise ValueError('SSD grouped B/C must align with the sequence and divide the head count')
    dtype = x.dtype
    x, dt, A, B, C = [v.float() for v in (x, dt, A, B, C)]
    if dt_bias is not None:
        dt = dt+dt_bias.float()[None, None]
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    repeats = x.shape[2]//B.shape[2]
    B, C = [v.repeat_interleave(repeats, dim=2) for v in (B, C)]
    state = x.new_zeros((x.shape[0], x.shape[2], x.shape[3], B.shape[-1]))
    outputs = []
    for start in range(0, x.shape[1], checkpoint_chunk):
        sl = slice(start, start+checkpoint_chunk)
        args = (state, x[:, sl], dt[:, sl], A, B[:, sl], C[:, sl])
        if torch.is_grad_enabled() and any(v.requires_grad for v in args):
            y, state = checkpoint(_chunk, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            y, state = _chunk(*args)
        outputs.append(y)
    output = torch.cat(outputs, dim=1)
    if D is not None:
        skip = D[:, None] if D.ndim == 1 else D
        output = output+x*skip.float()[None, None]
    if z is not None:
        output = output*F.silu(z.float())
    return output.to(dtype)


def gated_rms(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
    dtype = x.dtype
    value = x.float()
    if z is not None and not norm_before_gate:
        value = value*F.silu(z.float())
    size = value.shape[-1] if group_size is None else group_size
    groups = value.reshape(*value.shape[:-1], -1, size)
    groups = groups*torch.rsqrt(groups.square().mean(-1, keepdim=True)+eps)
    output = groups.reshape_as(value)*weight.float()
    if bias is not None:
        output = output+bias.float()
    if z is not None and norm_before_gate:
        output = output*F.silu(z.float())
    return output.to(dtype)


def split_conv_scan(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
                    seq_idx=None, dt_limit=(0., float('inf')), activation='silu',
                    rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None,
                    outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True,
                    checkpoint_chunk=64):
    if seq_idx is not None or activation not in {'silu', 'swish'}:
        raise ValueError('SSD bag backend requires fixed bags and SiLU, without packed seq_idx')
    if D.ndim == 2:
        heads, headdim = D.shape
    else:
        heads = len(D)
    if type(headdim) is not int or headdim < 1 or heads % ngroups:
        raise ValueError('Invalid SSD head/group dimensions')
    dim = heads*headdim
    state = (zxbcdt.shape[-1]-2*dim-heads)//(2*ngroups)
    if state < 1 or zxbcdt.shape[-1] != 2*dim+2*ngroups*state+heads:
        raise ValueError('Invalid SSD z/x/B/C/dt projection layout')
    z, xbc, dt = torch.split(zxbcdt, [dim, dim+2*ngroups*state, heads], dim=-1)
    length = xbc.shape[1]
    xbc = F.silu(F.conv1d(xbc.transpose(1, 2), conv1d_weight[:, None], conv1d_bias,
                          padding=conv1d_weight.shape[-1]-1, groups=xbc.shape[-1])[..., :length]).transpose(1, 2)
    x, B, C = torch.split(xbc, [dim, ngroups*state, ngroups*state], dim=-1)
    result = ssd_scan(rearrange(x, 'b l (h p) -> b l h p', h=heads), dt, A,
                       rearrange(B, 'b l (g n) -> b l g n', g=ngroups),
                       rearrange(C, 'b l (g n) -> b l g n', g=ngroups), D,
                       None if rmsnorm_weight is not None else rearrange(z, 'b l (h p) -> b l h p', h=heads),
                       dt_bias, True, dt_limit, checkpoint_chunk)
    result = rearrange(result, 'b l h p -> b l (h p)')
    if rmsnorm_weight is not None:
        result = gated_rms(result, rmsnorm_weight, z=z, eps=rmsnorm_eps, norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
        result = F.linear(result, outproj_weight, outproj_bias)
    return result

"""Small dependency-free tensor utilities with explicit semantics."""
import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * (mask / keep if self.scale_by_keep else mask)


def spatial_neighbors(coords, k=4, chunk=1024):
    """Physical nearest neighbours; never infer a square grid from token order."""
    if coords is None or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("This method requires real patch coordinates [N, 2]")
    if len(coords) < 2 or not torch.isfinite(coords).all():
        raise ValueError("Spatial methods require at least two finite coordinates")
    result = []
    c = coords.float()
    for start in range(0, len(c), chunk):
        d = torch.cdist(c[start:start + chunk], c)
        i = torch.arange(len(d), device=c.device)
        d[i, i + start] = torch.inf
        result.append(d.topk(min(k, len(c) - 1), largest=False).indices)
    return torch.cat(result)

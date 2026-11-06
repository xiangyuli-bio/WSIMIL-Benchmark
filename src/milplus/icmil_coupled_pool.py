"""Exact output-size1 source max pooling with a deterministic CUDA backward."""
import torch
from torch import nn
from torch.nn import functional as F


class SingleAdaptiveMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        if x.ndim!=2 or x.shape[-1]<1:raise ValueError('Coupled source MaxPool operates on a nonempty channel-by-instance matrix')
        # Retain the actual source forward and its original tie indices.
        values,indices=F.adaptive_max_pool1d(x,1,return_indices=True)
        ctx.save_for_backward(indices);ctx.shape=x.shape
        return values

    @staticmethod
    def backward(ctx,gradient):
        (indices,)=ctx.saved_tensors
        # One selected instance per distinct channel: no duplicate write or
        # atomic accumulation. This is the exact source K=1 derivative.
        result=gradient.new_zeros(ctx.shape)
        result.scatter_(-1,indices,gradient)
        return result


class SourceMaxPool(nn.Module):
    def __init__(self):super().__init__();self.original=nn.AdaptiveMaxPool1d(1)

    def forward(self,x):
        return SingleAdaptiveMax.apply(x) if x.is_cuda else self.original(x)

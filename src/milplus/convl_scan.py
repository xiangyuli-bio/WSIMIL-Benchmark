"""Full four-bin survival cumulative product with deterministic CUDA backward.

PyTorch2.5.1 cumprod_backward calls a CUDA cumsum which rejects strict
DeterministicAlgorithms. Forward retains the original native cumprod bytes.
Backward evaluates the same four-bin derivative without an atomic scan, with
explicit zero handling. CPU stays on unmodified native autograd.
Reference: pytorch/v2.5.1 aten/src/ATen/native/ReduceOps.cpp cumprod_backward.
"""
import torch


class FourBinCumprod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        if value.ndim != 2 or value.shape[1] != 4 or not value.is_floating_point():
            raise ValueError('ConVL deterministic survival scan requires its full four-bin real patient matrix')
        output = torch.cumprod(value, dim=1)
        ctx.save_for_backward(value, output)
        return output

    @staticmethod
    def backward(ctx, gradient):
        value, output = ctx.saved_tensors
        # Same native zero-free formula: reverse-sum(output * grad) / input.
        # The four-term sums use a fixed order and accumulator precision, not
        # CUDA cumsum. Preserve native intermediate scalar dtype for w/division.
        weight = output*gradient
        accumulation = weight.float() if weight.dtype in {torch.float16, torch.bfloat16} else weight
        suffixes = []
        for index in range(4):
            total = accumulation[:, 3]
            for tail in range(2, index-1, -1):
                total = total+accumulation[:, tail]
            suffixes.append(total.to(weight.dtype))
        no_zero = torch.stack(suffixes, dim=1)/value
        # Direct derivative sum_i>=j grad_i * prod_k<=i,k!=j x_k handles
        # one or multiple exact zeros, without dropping a survival bin.
        columns = []
        for column in range(4):
            terms = []
            for index in range(column, 4):
                product = torch.ones_like(value[:, 0])
                for factor in range(index+1):
                    if factor != column:
                        product = product*value[:, factor]
                terms.append(gradient[:, index]*product)
            total = terms[-1]
            for term in reversed(terms[:-1]):
                total = total+term
            columns.append(total)
        with_zero = torch.stack(columns, dim=1)
        return torch.where((value != 0).all(dim=1, keepdim=True), no_zero, with_zero)


def survival_cumprod(value, dim=1):
    if dim != 1:
        raise ValueError('ConVL source survival axis must remain dim1')
    if value.device.type != 'cuda':
        return torch.cumprod(value, dim=dim)
    return FourBinCumprod.apply(value)

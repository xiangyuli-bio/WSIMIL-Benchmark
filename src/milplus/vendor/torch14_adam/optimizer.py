"""Host optimizer state serialization with the original zero_grad default."""
from torch.optim import Optimizer as HostOptimizer


class Optimizer(HostOptimizer):
    def zero_grad(self, set_to_none=False):
        return super().zero_grad(set_to_none=set_to_none)

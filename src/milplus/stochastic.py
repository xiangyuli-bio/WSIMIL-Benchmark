"""Reproducible stochastic inference without consuming the subsequent training RNG stream."""
import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def inference_rng(device, seed):
    state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(state)
        np.random.set_state(numpy_state)

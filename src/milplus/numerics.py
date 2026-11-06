"""Initialize CPU vector math serially before training or evaluation.

Some statically linked MKL builds can return low-accuracy results on their
first concurrent VML call despite requesting high accuracy. A scalar PyTorch
call initializes the same dispatcher without using random input, changing
thread counts, replacing an operator, or modifying accuracy settings.
"""
import os
import threading


_ready_pid = None
_initialization_lock = threading.Lock()


def _reset_in_child():
    global _ready_pid, _initialization_lock
    _ready_pid = None
    _initialization_lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_in_child)


def initialize_cpu_vector_math():
    """Perform one serialized float32/float64 scalar initialization per process."""
    global _ready_pid
    pid = os.getpid()
    if _ready_pid == pid:
        return
    with _initialization_lock:
        if _ready_pid == pid:
            return
        import torch

        if torch.backends.mkl.is_available():
            with torch.no_grad():
                for dtype in [torch.float32, torch.float64]:
                    value = torch.tensor(4.0, dtype=dtype, device='cpu')
                    if torch.sqrt(value).item() != 2.0:
                        raise RuntimeError('CPU vector math initialization returned an invalid scalar square root')
        _ready_pid = pid

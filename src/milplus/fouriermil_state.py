"""Portable binding around the unchanged author Lookahead/RAdam step.

Original Lookahead omits current Optimizer hooks and serializes its slow state
by id(parameter). Restore the original RAdam loader and explicitly bind the
unchanged Lookahead slow tensors to their stable model parameter names.
"""

from collections import defaultdict, OrderedDict
import copy

import torch


def initialize_lookahead_hooks(optimizer):
    # Original Lookahead never calls Optimizer.__init__. Current zero_grad()
    # lazily wraps step() and then expects these registries to exist. Empty
    # hook registries repair that interface without changing any update math.
    for name in (
        "_optimizer_step_pre_hooks", "_optimizer_step_post_hooks",
        "_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks",
        "_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks",
    ):
        if not hasattr(optimizer, name):
            setattr(optimizer, name, OrderedDict())


def binding(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        current = []
        for parameter in group["params"]:
            name = names[id(parameter)]
            if name in seen:
                raise ValueError("Fourier duplicate optimizer parameter")
            seen.add(name)
            current.append({"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)})
        groups.append(current)
    if seen != set(names.values()):
        raise ValueError("Fourier optimizer omits model parameters")
    return groups


def optimizer_snapshot(model, optimizer):
    groups, slow = binding(model, optimizer), {}
    for group, records in zip(optimizer.param_groups, groups):
        for parameter, record in zip(group["params"], records):
            if parameter in optimizer.state:
                if set(optimizer.state[parameter]) != {"slow_buffer"}:
                    raise ValueError("Unexpected Fourier Lookahead state")
                slow[record["name"]] = copy.deepcopy(optimizer.state[parameter]["slow_buffer"])
    return {
        "schema": "fourier_original_optimizer_named_state_v1", "parameter_binding": groups,
        "fast_state": copy.deepcopy(optimizer.base_optimizer.state_dict()),
        "original_RAdam_ten_slot_cache": copy.deepcopy(optimizer.base_optimizer.buffer),
        "slow_buffers_by_parameter_name": slow,
    }


def restore_optimizer(model, optimizer, state):
    if (state["schema"] != "fourier_original_optimizer_named_state_v1"
            or state["parameter_binding"] != binding(model, optimizer)):
        raise ValueError("Fourier optimizer checkpoint parameter binding mismatch")
    names = dict(model.named_parameters())
    slow, cache = state["slow_buffers_by_parameter_name"], state["original_RAdam_ten_slot_cache"]
    if set(slow) - set(names) or not isinstance(cache, list) or len(cache) != 10:
        raise ValueError("Fourier slow state or RAdam cache mismatch")
    for name, value in slow.items():
        if value.shape != names[name].shape or value.dtype != names[name].dtype:
            raise ValueError("Fourier slow tensor shape/dtype mismatch")
    optimizer.base_optimizer.load_state_dict(copy.deepcopy(state["fast_state"]))
    optimizer.base_optimizer.buffer = copy.deepcopy(cache)
    optimizer.param_groups = optimizer.base_optimizer.param_groups
    optimizer.defaults = optimizer.base_optimizer.defaults
    optimizer.state = defaultdict(dict)
    for name, value in slow.items():
        optimizer.state[names[name]]["slow_buffer"] = value.detach().clone().to(names[name].device)


def gradients(model):
    # This includes scaled/infinite gradients before a pending GradScaler step.
    return {n: None if p.grad is None else p.grad.detach().clone()
            for n, p in model.named_parameters()}


def restore_gradients(model, saved):
    if set(saved) != set(dict(model.named_parameters())):
        raise ValueError("Fourier gradient parameter binding mismatch")
    for name, parameter in model.named_parameters():
        value = saved[name]
        if value is not None and (value.shape != parameter.shape or value.dtype != parameter.dtype):
            raise ValueError("Fourier gradient shape/dtype mismatch")
        parameter.grad = None if value is None else value.detach().clone().to(parameter.device)

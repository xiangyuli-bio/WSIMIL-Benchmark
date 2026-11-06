"""nnMIL classification recipe: balanced batches and masked fixed-length bags."""
import random

import numpy as np
import torch
from torch.nn import functional as F


def balanced_batches(groups, batch_size):
    pools = {}
    for rows in groups:
        pools.setdefault(rows[0]["target"], []).append(rows)
    classes = sorted(pools)
    if batch_size < len(classes):
        raise ValueError("nnMIL batch_size must cover all classes")
    for pool in pools.values():
        random.shuffle(pool)
    step = 0
    while True:
        counts = {c: batch_size // len(classes) for c in classes}
        for j in range(batch_size % len(classes)):
            counts[classes[(step + j) % len(classes)]] += 1
        if any(len(pools[c]) < counts[c] for c in classes):
            break
        batch = [pools[c].pop() for c in classes for _ in range(counts[c])]
        random.shuffle(batch)
        yield batch
        step += 1


def padded_subbags(bags, size):
    if size < 1:
        raise ValueError("fixed_bag_size must be positive")
    x = bags[0].new_zeros((len(bags), size, bags[0].shape[-1]))
    valid = torch.zeros(len(bags), size, dtype=torch.bool, device=x.device)
    for i, bag in enumerate(bags):
        indices = torch.randperm(len(bag), device=bag.device)[:size]
        x[i, :len(indices)] = bag[indices]
        valid[i, :len(indices)] = True
    return x, valid


def nnmil_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    size = model.fixed_bag_size
    if size == "auto":
        # Planning uses only training bags, including all slides in a patient bag.
        counts = [sum(len(bundle.load(row)[0]) for row in rows) for rows in groups]
        size = max(1, int(np.median(counts) / 2))
    losses = []
    for batch in balanced_batches(groups, model.batch_size):
        bags = [load_unit(bundle, rows, device, cfg["max_instances"], train=True)[0] for rows in batch]
        x, mask = padded_subbags(bags, int(size))
        labels = torch.tensor([rows[0]["target"] for rows in batch], device=device)
        output = model(x, valid_mask=mask)
        loss = F.cross_entropy(output["logits"], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite nnMIL loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        losses.append({"loss": float(loss.detach()), "fixed_bag_size": size, "batch_size": len(batch)})
    if not losses:
        raise ValueError("Not enough training bags per class for nnMIL balanced batch; lower batch_size")
    return losses

"""Instance scoring methods with binary-specific objectives."""
import random
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class FocusMIL(nn.Module):
    """Published binary variational max-pooling route, deterministic mean at inference."""
    def __init__(self, dim, classes, hidden=128, kl_weight=1., classification_weight=1.):
        super().__init__()
        if classes != 2 or min(kl_weight, classification_weight) <= 0:
            raise ValueError("FocusMIL requires binary labels and active KL/classification objectives")
        self.encoder = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 200))
        self.mean, self.logvar = nn.Linear(200, hidden), nn.Linear(200, hidden)
        self.head = nn.Linear(hidden, 1)
        self.kl_weight, self.classification_weight = kl_weight, classification_weight

    def forward(self, x, coords=None, label=None):
        encoded = self.encoder(x)
        mean, logvar = self.mean(encoded), self.logvar(encoded)
        std = (logvar * .5).exp()
        z = mean + torch.randn_like(std) * std if self.training else mean
        positive = self.head(z).max(0).values.reshape(1, 1)
        logits = torch.cat((torch.zeros_like(positive), positive), -1)
        out = {"logits": logits}
        if self.training and label is not None:
            kl = .5 * (mean.square() + std.square() - 2 * (std + 1e-8).log() - 1).mean()
            out["total_loss"] = self.classification_weight * F.cross_entropy(logits, label) + self.kl_weight * kl
            out["kl_loss"] = kl
        return out


class FRMIL(nn.Module):
    """Original MICCAI source and its paired-bag three-term training recipe."""
    def __init__(self, dim, classes, heads=1, margin=8.48, feature_dropout=None, mode="cm16"):
        super().__init__()
        if classes != 2 or dim % heads or mode not in {"cm16", "msi"}:
            raise ValueError("FRMIL original recipe is binary, with feature width divisible by heads")
        from .upstream import load_single
        author = load_single("frmil", "models/mil_ss.py")
        self.model = author.FRMIL(SimpleNamespace(model_ext=mode, num_class=classes, num_feats=dim, n_heads=heads))
        self.margin = margin
        self.feature_dropout = (.2 if mode == "cm16" else 0.) if feature_dropout is None else feature_dropout
        self.register_buffer("padding_length", torch.tensor(0))

    def prepare(self, bundle, seed, device):
        from .engine import load_unit, units
        self.padding_length.fill_(max(len(load_unit(bundle, rows, "cpu")[0]) for rows in units(bundle, "train")))

    def forward(self, x, coords=None, label=None):
        out = self.model(x.unsqueeze(0))
        return {"logits": out[0] if self.training else out}


def frmil_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    counts = torch.bincount(torch.tensor([g[0]["target"] for g in groups]), minlength=2).float().to(device)
    losses = []
    classes = [[g for g in groups if g[0]["target"] == c] for c in range(2)]
    if not all(classes):
        raise ValueError("FRMIL requires both positive and negative training bags")
    for _ in range(len(groups)):
        # Author CategoriesSampler samples one bag per class for each of N episodes.
        batch = [random.choice(pool) for pool in classes]
        bags = [load_unit(bundle, rows, device, cfg["max_instances"], train=True)[0] for rows in batch]
        # Author dataset pads to the largest TRAIN bag, not the batch maximum.
        size = int(model.padding_length)
        if cfg["max_instances"]:
            size = min(size, cfg["max_instances"])
        x = bags[0].new_zeros(2, size, bags[0].shape[-1])
        for i, bag in enumerate(bags):
            x[i, :len(bag)] = bag
        x = F.dropout(x, p=model.feature_dropout, training=True)
        targets = torch.tensor([rows[0]["target"] for rows in batch], device=device)
        logits, shifted, instance_probs = model.model(x)
        bag_loss = F.cross_entropy(logits, targets, weight=counts.reciprocal())
        # The author's BCE uses a scalar negative/positive ratio on BOTH observations.
        instance_loss = F.binary_cross_entropy(instance_probs.max(1).values, targets.float(),
                                               weight=counts[0] / counts[1])
        norms = shifted.mean(1).norm(dim=-1)
        magnitude_loss = (F.relu(model.margin - norms[targets == 1]) + norms[targets == 0]).square().mean() / x.shape[1]
        loss = (bag_loss + instance_loss + magnitude_loss) / 3
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite FRMIL loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        losses.append({"loss": float(loss.detach()), "instance_loss": float(instance_loss.detach()),
                       "magnitude_loss": float(magnitude_loss.detach())})
    if not losses:
        raise ValueError("FRMIL requires both positive and negative training bags")
    return losses

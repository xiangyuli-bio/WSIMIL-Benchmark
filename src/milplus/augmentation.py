"""Train-only PseMix/ProtoDiv and ReMix preparation and optimization.

Sources and deliberate adaptations are recorded in docs/EXPANSION.md.
"""
import math
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .models import ABMIL


def unit_key(rows):
    return tuple(row["slide_id"] for row in rows)


def phenotype_partition(x, parts, phenotypes, refinements=8):
    """PseMix ProtoDiv: cosine bins -> centroid refinement -> balanced phenotype sampling."""
    if len(x) < parts:
        raise ValueError("Bag is shorter than pseudo_bags; lower pseudo_bags explicitly")
    unit = F.normalize(x, dim=-1)
    score = unit @ F.normalize(x.mean(0), dim=0)
    labels = ((score + 1) * phenotypes / 2).long().clamp(0, phenotypes - 1)
    for _ in range(refinements):
        centers = torch.stack([x[labels == label].mean(0) for label in labels.unique(sorted=True)])
        labels = (unit @ F.normalize(centers, dim=-1).T).argmax(-1)
    assignment = torch.zeros(len(x), dtype=torch.long)
    for c in labels.unique(sorted=True):
        chosen = labels == c
        # Same balanced random assignment as the author's uniform_assign.
        shuffled_instances = torch.randperm(int(chosen.sum())) % parts
        shuffled_labels = torch.randperm(parts)
        assignment[chosen] = shuffled_labels[shuffled_instances]
    if len(assignment.unique()) < parts:
        raise ValueError("Empty pseudo-bags after phenotype division; reduce pseudo_bags")
    return assignment


class PseudoBagMIL(nn.Module):
    def __init__(self, dim, classes, mode, hidden=512, pseudo_bags=30, phenotypes=8,
                 refinements=8, batch_size=16, alpha=1., mix_probability=.4, loss="bce"):
        super().__init__()
        if min(pseudo_bags, phenotypes, batch_size) < 1 or alpha <= 0 or not 0 <= mix_probability <= 1:
            raise ValueError("Invalid pseudo-bag configuration")
        if loss not in {"bce", "soft_ce"}:
            raise ValueError("Pseudo-bag loss must be bce or soft_ce")
        self.model = ABMIL(dim, classes, hidden=hidden)
        self.mode, self.classes, self.parts, self.phenotypes = mode, classes, pseudo_bags, phenotypes
        self.refinements, self.batch_size, self.alpha = refinements, batch_size, alpha
        self.mix_probability, self.loss = mix_probability, loss
        self.partitions = {}

    def get_extra_state(self):
        return {"partitions": {k: v.cpu() for k, v in self.partitions.items()}}

    def set_extra_state(self, state):
        self.partitions = {k: v.cpu() for k, v in state["partitions"].items()}

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        from .engine import load_unit, units
        for rows in units(bundle, "train"):
            x, _ = load_unit(bundle, rows, "cpu")
            self.partitions[unit_key(rows)] = phenotype_partition(x, self.parts, self.phenotypes, self.refinements)

    def forward(self, x, coords=None, label=None):
        return self.model(x, coords=coords, label=label)


def fetch_parts(x, assignment, total, number):
    chosen = torch.randperm(total)[:number].to(x.device)
    return x[torch.isin(assignment.to(x.device), chosen)]


def pseudo_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    if cfg["max_instances"] is not None:
        raise ValueError("Pseudo-bag profiles need the registered full patch order; do not use max_instances")
    records = []
    for start in range(0, len(groups), model.batch_size):
        batch = groups[start:start + model.batch_size]
        bags = [load_unit(bundle, rows, device)[0] for rows in batch]
        assignment = [model.partitions[unit_key(rows)] for rows in batch]
        labels = torch.tensor([rows[0]["target"] for rows in batch], device=device)
        permutation = torch.randperm(len(batch))
        lam = np.random.beta(model.alpha, model.alpha)
        count = min(model.parts, int(lam * (model.parts + 1)))
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for i, x in enumerate(bags):
            target = F.one_hot(labels[i:i+1], model.classes).float()
            if model.mode == "prodiv":
                mixed = fetch_parts(x, assignment[i], model.parts, 1)
            else:
                other = int(permutation[i])
                a = fetch_parts(x, assignment[i], model.parts, count)
                b = fetch_parts(bags[other], assignment[other], model.parts, model.parts - count)
                ratio = count / model.parts
                if len(a) and len(b) and np.random.rand() > model.mix_probability:
                    mixed, ratio = a, 1.
                else:
                    mixed = torch.cat((a, b))
                target = ratio * target + (1-ratio) * F.one_hot(labels[other:other+1], model.classes).float()
            logits = model(mixed)["logits"]
            loss = (F.binary_cross_entropy_with_logits(logits, target) if model.loss == "bce"
                    else -(target * logits.log_softmax(-1)).sum(-1).mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pseudo-bag objective")
            # Accumulating gradients avoids retaining all variable-length bag graphs.
            (loss / len(batch)).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        records.append({"loss": total / len(batch), "mixed_bags": len(batch)})
    return records


class ReMix(nn.Module):
    def __init__(self, dim, classes, hidden=None, prototypes=8, shifts=200, rate=.3, mode="joint"):
        super().__init__()
        if prototypes < 1 or shifts < 1 or not 0 < rate <= 1 or mode not in {"replace", "append", "interpolate", "cov", "joint"}:
            raise ValueError("Invalid ReMix prototype or augmentation configuration")
        # Author's ABMIL baseline uses ReLU attention on unprojected features.
        self.attention = nn.Sequential(nn.Linear(dim, hidden or dim), nn.ReLU(), nn.Linear(hidden or dim, 1))
        self.classifier = nn.Linear(dim, classes)
        self.prototypes, self.shifts, self.rate, self.mode = prototypes, shifts, rate, mode
        self.cache = {}

    def get_extra_state(self):
        return {"cache": {k: (a.cpu(), b.cpu()) for k, (a,b) in self.cache.items()}}

    def set_extra_state(self, state):
        self.cache = {k: (a.cpu(), b.cpu()) for k, (a,b) in state["cache"].items()}

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        from .engine import load_unit, units
        for rows in units(bundle, "train"):
            x, _ = load_unit(bundle, rows, "cpu")
            if self.prototypes > len(x):
                raise ValueError("ReMix prototypes exceed number of patches")
            labels = KMeans(n_clusters=self.prototypes, n_init=10, max_iter=20, random_state=seed).fit_predict(F.normalize(x,dim=-1).numpy())
            centers, shifts = [], []
            for i in range(self.prototypes):
                cluster = x[labels == i]
                if not len(cluster):
                    raise ValueError("Empty ReMix cluster; reduce prototypes")
                center = cluster.mean(0)
                centers.append(center)
                # X-centered is a covariance factor; this samples N(0, sample covariance) without a D x D decomposition.
                shift = torch.randn(self.shifts, len(cluster)) @ (cluster - center) / math.sqrt(max(1,len(cluster)-1))
                shifts.append(shift)
            self.cache[unit_key(rows)] = (torch.stack(centers), torch.stack(shifts))

    def forward(self, x, coords=None, label=None):
        a = self.attention(x).T.softmax(-1)
        logits = self.classifier(a @ x)
        p = logits.sigmoid()
        return {"logits": logits, "attention": a, "probabilities": p/p.sum(-1,keepdim=True)}


def remix_augment(source, target, shifts, rate, strength, mode):
    nearest = torch.cdist(source, target).argmin(-1)
    result = list(source.unbind())
    for i in range(len(source)):
        j = int(nearest[i])
        for action in (["replace","append","interpolate","cov"] if mode == "joint" else [mode]):
            if np.random.rand() > rate:
                continue
            if action == "replace":
                result[i] = target[j]
            elif action == "append":
                result.append(target[j])
            elif action == "interpolate":
                result.append((1-strength)*result[i] + strength*target[j])
            else:
                result.append(result[i] + strength*shifts[j, np.random.randint(shifts.shape[1])])
    return torch.stack(result)


def remix_epoch(model, bundle, groups, optimizer, device, cfg):
    records = []
    pools = {}
    for rows in groups:
        pools.setdefault(rows[0]["target"], []).append(rows)
    for rows in groups:
        label = rows[0]["target"]
        donor = random.choice(pools[label])
        source = model.cache[unit_key(rows)][0]
        target, shifts = model.cache[unit_key(donor)]
        x = remix_augment(source, target, shifts, model.rate, np.random.uniform(), model.mode).to(device)
        logits = model(x)["logits"]
        y = F.one_hot(torch.tensor([label], device=device), logits.shape[-1]).float()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ReMix objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
        optimizer.step()
        records.append({"loss":float(loss.detach()),"augmented_instances":len(x)})
    return records

"""MHIM v1 attention variant: supervised initialization, masking, feature distillation and EMA."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionStudent(nn.Module):
    def __init__(self, dim, classes, hidden=512, dropout=0.25):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention = nn.Sequential(nn.Linear(hidden, 128, bias=False), nn.ReLU(), nn.Linear(128, 1, bias=False))
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x, keep=None):
        h = self.project(x)
        if keep is not None:
            h = h[keep]
        a = self.attention(h).T.softmax(-1)
        embedding = a @ h
        return {"logits": self.classifier(embedding), "attention": a, "embedding": embedding}


class MHIM(nn.Module):
    """PULS explicit warm-start recipe; not a claim of the original released checkpoint."""
    def __init__(self, dim, classes, hidden=512, dropout=0.25, warmup_epochs=5,
                 high_mask=0.1, low_mask=0.1, random_mask=0.0, high_random_fraction=1.0,
                 teacher_temperature=0.1, student_temperature=1.0, distillation_weight=0.1,
                 momentum=0.9999, high_mask_decay=True):
        super().__init__()
        if warmup_epochs < 1 or not 0 <= momentum < 1 or teacher_temperature <= 0 or student_temperature <= 0:
            raise ValueError("Invalid MHIM teacher initialization/temperature/momentum")
        if not all(0 <= r < 1 for r in (high_mask, low_mask, random_mask)) or high_mask <= 0:
            raise ValueError("MHIM requires positive high-attention mining and mask ratios in [0,1)")
        if distillation_weight <= 0 or not 0 < high_random_fraction <= 1:
            raise ValueError("MHIM requires feature distillation and a valid high candidate fraction")
        self.student = AttentionStudent(dim, classes, hidden, dropout)
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.register_buffer("teacher_initialized", torch.tensor(False))
        self.warmup_epochs, self.high_mask, self.low_mask, self.random_mask = warmup_epochs, high_mask, low_mask, random_mask
        self.teacher_temperature, self.student_temperature = teacher_temperature, student_temperature
        self.distillation_weight, self.momentum = distillation_weight, momentum
        self.high_mask_decay, self.high_random_fraction = high_mask_decay, high_random_fraction
        self.epoch, self.total_epochs, self.step_in_epoch, self.steps_per_epoch = 0, warmup_epochs + 1, 0, 1

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def initialize_teacher(self):
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_initialized.fill_(True)

    @torch.no_grad()
    def update_teacher(self):
        if self.teacher_initialized:
            for t, s in zip(self.teacher.parameters(), self.student.parameters()):
                t.mul_(self.momentum).add_(s, alpha=1 - self.momentum)

    def selection(self, attention):
        a = attention.flatten()
        n = len(a)
        mask = torch.zeros(n, dtype=torch.bool, device=a.device)
        if self.random_mask:
            mask[torch.randperm(n, device=a.device)[:math.ceil(n * self.random_mask)]] = True
        if self.low_mask:
            mask[a.topk(math.ceil(n * self.low_mask), largest=False).indices] = True
        high = self.high_mask
        if self.high_mask_decay:
            progress = ((self.epoch - self.warmup_epochs) * self.steps_per_epoch + self.step_in_epoch) / max(
                1, (self.total_epochs - self.warmup_epochs) * self.steps_per_epoch - 1)
            high *= .5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        count = math.ceil(n * high)
        if count:
            candidates = a.topk(min(n, math.ceil(count / self.high_random_fraction))).indices
            mask[candidates[torch.randperm(len(candidates), device=a.device)[:count]]] = True
        if mask.all():
            raise ValueError("MHIM masking removed the entire bag; adjust protocol for small bags")
        return (~mask).nonzero(as_tuple=False).flatten()

    def forward(self, x, coords=None, label=None):
        if not self.training or self.epoch < self.warmup_epochs:
            return self.student(x)
        if not self.teacher_initialized:
            raise RuntimeError("MHIM teacher initialization phase was not executed")
        with torch.no_grad():
            target = self.teacher(x)
            keep = self.selection(target["attention"])
        out = self.student(x, keep)
        distill = -(F.softmax(target["embedding"] / self.teacher_temperature, dim=-1) *
                     F.log_softmax(out["embedding"] / self.student_temperature, dim=-1)).sum(-1).mean()
        out["regularization_loss"] = self.distillation_weight * distill
        out["kept_instances"] = len(keep)
        return out

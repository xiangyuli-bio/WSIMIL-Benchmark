"""Explicit adaptations with source equations and differences in docs/METHODS.md."""
import torch
from torch import nn
from torch.nn import functional as F

from .models import ABMIL


class AEM(ABMIL):
    def __init__(self, dim, classes, hidden=512, attention_dim=128, dropout=0.0, entropy_weight=0.1):
        super().__init__(dim, classes, hidden, attention_dim, dropout=dropout)
        if entropy_weight <= 0:
            raise ValueError("AEM entropy_weight must be positive")
        self.entropy_weight = entropy_weight

    def forward(self, x, coords=None, label=None):
        out = super().forward(x, coords, label)
        a = out["attention"]
        out["regularization_loss"] = self.entropy_weight * (a * a.clamp_min(1e-12).log()).sum(-1).mean()
        return out


class ACMIL(nn.Module):
    """ACMIL-GA: gated branches, stochastic top-k masking, branch supervision and diversity."""
    def __init__(self, dim, classes, hidden=512, attention_dim=128, branches=5,
                 mask_topk=10, mask_ratio=0.6, dropout=0.0):
        super().__init__()
        if branches < 2 or not 0 < mask_ratio < 1 or mask_topk < 1:
            raise ValueError("ACMIL profile requires >1 branch and active partial top-k masking")
        self.reduce = nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.ReLU())
        self.v, self.u = nn.Linear(hidden, attention_dim), nn.Linear(hidden, attention_dim)
        self.w = nn.Linear(attention_dim, branches)
        self.heads = nn.ModuleList(nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, classes)) for _ in range(branches))
        self.slide = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, classes))
        self.branches, self.mask_topk, self.mask_ratio = branches, mask_topk, mask_ratio

    def forward(self, x, coords=None, label=None):
        h = self.reduce(x)
        logits = self.w(self.v(h).tanh() * self.u(h).sigmoid()).T
        if self.training:
            k = min(self.mask_topk, len(x))
            count = int(k * self.mask_ratio)
            if count:
                top = logits.topk(k, dim=-1).indices
                chosen = torch.rand_like(top, dtype=torch.float32).argsort(-1)[:, :count]
                mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top.gather(1, chosen), True)
                logits = logits.masked_fill(mask, -1e9)
        a = logits.softmax(-1)
        pooled = a @ h
        heads = torch.stack([head(pooled[i]) for i, head in enumerate(self.heads)])
        similarity = F.normalize(a, dim=-1) @ F.normalize(a, dim=-1).T
        diversity = similarity.triu(diagonal=1).sum() / (self.branches * (self.branches - 1) / 2)
        return {"logits": self.slide(pooled.mean(0, keepdim=True)), "branch_logits": heads,
                "regularization_loss": diversity, "attention": a}


class LowRankAttention(nn.Module):
    def __init__(self, qdim, kdim, dim, heads, gated, layernorm):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(qdim, dim), nn.Linear(kdim, dim), nn.Linear(kdim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.gate = nn.Sequential(nn.Linear(qdim, dim), nn.SiLU()) if gated else None

    def forward(self, q, k):
        query = self.q(q)
        value, _ = self.attn(query, self.k(k), self.v(k), need_weights=False)
        h = self.norm1(query + value)
        h = self.norm2(h + self.out(h).relu())
        return h if self.gate is None else h * self.gate(q)


class LowRankBlock(nn.Module):
    def __init__(self, dim_in, hidden, heads, rank, layernorm):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(1, rank, hidden))
        nn.init.xavier_uniform_(self.latent)
        self.compress = LowRankAttention(hidden, dim_in, hidden, heads, True, layernorm)
        self.expand = LowRankAttention(dim_in, hidden, hidden, heads, True, layernorm)

    def forward(self, x):
        return self.expand(x, self.compress(self.latent, x))


class ILRA(nn.Module):
    def __init__(self, dim, classes, hidden=256, heads=8, layers=2, rank=1, layernorm=False):
        super().__init__()
        if hidden % heads or layers < 1 or rank < 1:
            raise ValueError("Invalid ILRA dimensions")
        self.blocks = nn.ModuleList(LowRankBlock(dim if i == 0 else hidden, hidden, heads, rank, layernorm)
                                   for i in range(layers))
        self.seed = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.xavier_uniform_(self.seed)
        self.pool = LowRankAttention(hidden, hidden, hidden, heads, False, layernorm)
        self.head = nn.Linear(hidden, classes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, x, coords=None, label=None):
        h = x.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        return {"logits": self.head(self.pool(self.seed, h)[:, 0])}


class WiKG(nn.Module):
    """Feature-derived graph. Query chunks avoid materializing the full N-by-N matrix."""
    def __init__(self, dim, classes, hidden=512, topk=6, dropout=0.3, chunk_size=256):
        super().__init__()
        if topk < 1 or chunk_size < 1:
            raise ValueError("topk and chunk_size must be positive")
        self.project = nn.Linear(dim, hidden)
        self.head, self.tail = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.sum_layer, self.product_layer = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.attention = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.LeakyReLU(), nn.Linear(hidden // 2, 1))
        self.norm, self.classifier = nn.LayerNorm(hidden), nn.Linear(hidden, classes)
        self.drop = nn.Dropout(dropout)
        self.topk, self.chunk_size, self.scale = topk, chunk_size, hidden ** -.5

    def forward(self, x, coords=None, label=None):
        if len(x) < self.topk:
            raise ValueError("WiKG bag smaller than topk")
        x = F.leaky_relu(self.project(x))
        x = (x + x.mean(0, keepdim=True)) * .5
        head, tail = self.head(x), self.tail(x)
        outputs = []
        for start in range(0, len(x), self.chunk_size):
            h = head[start:start+self.chunk_size]
            weights, indices = (h * self.scale @ tail.T).topk(self.topk, dim=-1)
            neighbours = tail[indices]
            p = weights.softmax(-1).unsqueeze(-1)
            relation = p * neighbours + (1-p) * h.unsqueeze(1)
            gate = (h.unsqueeze(1) + relation).tanh()
            # Preserve official einsum('ijkl,ijkm->ijk') semantics: independent feature sums.
            knowledge = (neighbours.sum(-1) * gate.sum(-1)).softmax(-1)
            neighbour = (knowledge.unsqueeze(-1) * neighbours).sum(1)
            outputs.append(F.leaky_relu(self.sum_layer(h + neighbour)) +
                           F.leaky_relu(self.product_layer(h * neighbour)))
        embeddings = self.drop(torch.cat(outputs))
        a = self.attention(embeddings).T.softmax(-1)
        return {"logits": self.classifier(self.norm(a @ embeddings)), "attention": a}

"""Train-only prototype preparation and published histogram/mixture representations."""
import math

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .models import Attention


def training_patch_sample(bundle, maximum=50000):
    """Uniform priority reservoir over TRAIN patches; bounded retained memory."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    retained, priorities = None, None
    for row in bundle.records("train"):
        bag, _ = bundle.load(row)
        for block in bag.split(min(maximum, 4096)):
            keys = torch.rand(len(block))
            retained = block if retained is None else torch.cat((retained, block))
            priorities = keys if priorities is None else torch.cat((priorities, keys))
            if len(retained) > maximum:
                keep = priorities.topk(maximum).indices
                retained, priorities = retained[keep], priorities[keep]
    return retained


def centroids(features, number, seed, normalized=False):
    if number > len(features):
        raise ValueError(f"{number} prototypes requested for only {len(features)} training observations")
    values = F.normalize(features, dim=-1) if normalized else features
    labels = KMeans(n_clusters=number, n_init=10, max_iter=100, random_state=seed).fit_predict(values.numpy())
    if len(set(labels)) != number:
        raise ValueError("Empty prototype clusters; reduce prototypes or check feature diversity")
    # IBMIL clusters normalized bags but averages ORIGINAL embeddings.
    return torch.stack([features[torch.as_tensor(labels == i)].mean(0) for i in range(number)])


class PrototypeMIL(nn.Module):
    """H2T mean-of-normalized-clusters or PANTHER MAP-EM allcat + linear head."""
    def __init__(self, dim, classes, mode, prototypes=8, sample_size=50000,
                 em_steps=3, tau=10., prior_variance=.1, variance_floor=1e-7):
        super().__init__()
        if min(prototypes, sample_size, em_steps, tau, prior_variance, variance_floor) <= 0:
            raise ValueError("Prototype and MAP-EM parameters must be positive")
        self.mode, self.sample_size = mode, sample_size
        self.steps, self.tau, self.prior_variance = em_steps, tau, prior_variance
        self.variance_floor = variance_floor
        self.register_buffer("prototypes", torch.zeros(prototypes, dim))
        self.register_buffer("prepared", torch.tensor(False))
        width = prototypes * dim if mode == "h2t" else prototypes * (2 * dim + 1)
        self.classifier = nn.Linear(width, classes, bias=False)

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        sample = training_patch_sample(bundle, self.sample_size)
        centers = centroids(sample, len(self.prototypes), seed, normalized=self.mode == "h2t")
        if self.mode == "h2t":
            centers = F.normalize(centers, dim=-1)
        self.prototypes.copy_(centers.to(device))
        self.prepared.fill_(True)

    def representation(self, x):
        if not self.prepared:
            raise ValueError("Train-only prototype preparation has not run")
        if self.mode == "h2t":
            h = F.normalize(x, dim=-1)
            assignments = torch.cdist(h, self.prototypes).argmin(-1)
            output = h.new_zeros(self.prototypes.shape)
            output.index_add_(0, assignments, h)
            counts = torch.bincount(assignments, minlength=len(output)).clamp_min(1)
            return (output / counts[:, None]).reshape(1, -1)
        # Equivalent diagonal normal-inverse-Wishart MAP updates, with variance floor for round-off.
        means = self.prototypes
        var = torch.full_like(means, self.prior_variance)
        pi = x.new_full((len(means),), 1 / len(means))
        for _ in range(self.steps):
            log_density = -.5 * (x.shape[-1] * math.log(2 * math.pi) + var.log().sum(-1)
                                + (x.square() @ var.reciprocal().T)
                                + (means.square() / var).sum(-1)
                                - 2 * x @ (means / var).T)
            posterior = (log_density + pi.log()).softmax(-1)
            weight = posterior.sum(0) + self.tau
            pi = weight / weight.sum()
            means = (posterior.T @ x + self.tau * self.prototypes) / weight[:, None]
            second = (posterior.T @ x.square() + self.tau *
                      (self.prior_variance + self.prototypes.square())) / weight[:, None]
            var = (second - means.square()).clamp_min(self.variance_floor)
        return torch.cat((pi, means.flatten(), var.flatten())).unsqueeze(0)

    def forward(self, x, coords=None, label=None):
        with torch.no_grad():
            embedding = self.representation(x)
        return {"logits": self.classifier(embedding), "embedding": embedding}


class IBMIL(nn.Module):
    """ABMIL warm-up -> train-bag clustering -> interventional attention (concat)."""
    def __init__(self, dim, classes, hidden=None, prototypes=8, confounder_dim=128, warmup_epochs=5):
        super().__init__()
        if warmup_epochs < 1 or prototypes < 1:
            raise ValueError("IBMIL requires supervised initialization and positive prototype count")
        self.attention = Attention(dim, hidden or dim)
        self.base_classifier = nn.Linear(dim, classes)
        self.classifier = nn.Linear(2 * dim, classes)
        self.query = nn.Linear(dim, confounder_dim)
        self.key = nn.Linear(dim, confounder_dim)
        self.register_buffer("confounders", torch.zeros(prototypes, dim))
        self.register_buffer("prepared", torch.tensor(False))
        self.warmup_epochs = warmup_epochs

    @torch.no_grad()
    def initialize_confounders(self, bundle, seed, device, config):
        from .engine import load_unit, units
        training = self.training
        self.eval()
        bags = []
        for rows in units(bundle, "train"):
            x, _ = load_unit(bundle, rows, device, config["max_instances"])
            bags.append((self.attention(x) @ x).squeeze(0).cpu())
        self.confounders.copy_(centroids(torch.stack(bags), len(self.confounders), seed, normalized=True).to(device))
        self.prepared.fill_(True)
        self.train(training)

    def forward(self, x, coords=None, label=None):
        a = self.attention(x)
        pooled = a @ x
        if not self.prepared:
            return {"logits": self.base_classifier(pooled), "attention": a}
        score = self.query(pooled) @ self.key(self.confounders).T / math.sqrt(self.query.out_features)
        intervened = score.softmax(-1) @ self.confounders
        return {"logits": self.classifier(torch.cat((pooled, intervened), -1)), "attention": a}


class Chowder(nn.Module):
    """Owkin's later multiclass CHOWDER: per-class extremes + shared sigmoid MLP."""
    def __init__(self, dim, classes, hidden=128, extremes=10):
        super().__init__()
        self.extremes = extremes
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid(), nn.Linear(hidden, classes))
        self.head = nn.Sequential(nn.Linear(2 * extremes, hidden), nn.Sigmoid(),
                                  nn.Linear(hidden, hidden // 2), nn.Sigmoid(), nn.Linear(hidden // 2, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, coords=None, label=None):
        if len(x) < 2 * self.extremes:
            raise ValueError("CHOWDER needs disjoint top/bottom extreme instances")
        values = self.score(x)
        extremes = torch.cat((values.topk(self.extremes, dim=0).values,
                              values.topk(self.extremes, dim=0, largest=False).values), 0)
        return {"logits": self.head(extremes.T).T, "instance_scores": values}


class VarPool(nn.Module):
    """Projected attention-weighted variances, separate mean/variance attention optional."""
    def __init__(self, dim, classes, hidden=512, projections=100, attention_dim=256,
                 separate_attention=False, variance_activation="sqrt", dropout=0.):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention = Attention(hidden, attention_dim, gated=True)
        self.var_attention = Attention(hidden, attention_dim, gated=True) if separate_attention else None
        self.projections = nn.Linear(hidden, projections, bias=False)
        self.activation = variance_activation
        if variance_activation not in {"sqrt", "log", "sigmoid", "identity"}:
            raise ValueError("Unknown variance activation")
        self.head = nn.Sequential(nn.Linear(hidden + projections, 256), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(256, classes))

    def forward(self, x, coords=None, label=None):
        h = self.project(x)
        a = self.attention(h)
        v_a = self.var_attention(h) if self.var_attention is not None else a
        p = self.projections(h)
        variance = v_a @ (p - v_a @ p).square()
        if self.activation == "sqrt":
            variance = variance.clamp_min(1e-12).sqrt()
        elif self.activation == "log":
            variance = (variance + .01).log()
        elif self.activation == "sigmoid":
            variance = variance.sigmoid()
        embedding = torch.cat((a @ h, variance), -1)
        return {"logits": self.head(embedding), "attention": a, "embedding": embedding}

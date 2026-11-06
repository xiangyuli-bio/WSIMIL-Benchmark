"""Method-specific wrappers; objectives and state are part of each adapter."""
import importlib

import torch
from torch import nn

from .layers import spatial_neighbors
from .models import Attention

BASELINE_MODELS = {
    "additive_mil": ("ADD_MIL", "hidden_dim"),
    "linmil": ("LIN_MIL", "latent_dim"),
    "fouriermil": ("FOURIER_MIL", "hidden_dim"),
    "retmil": ("RET_MIL", "embed_dim"),
    "pneamil": ("PNEA_MIL", "evidence_dim"),
    "mrblock": ("MR_AB_MIL", "attention_dim"),
    "carmil": ("CAR_MIL", "hidden_dim"),
    "fcmil": ("FC_MIL", "hidden_dim"),
    "asmil": ("AS_MIL", "hidden_dim"),
    "attrimil": ("ATTRI_MIL", "attention_dim"),
    "nnmil": ("NN_MIL", "hidden_dim"),
}


class BaselineAdapter(nn.Module):
    def __init__(self, method, dim, classes, hidden=None, **config):
        super().__init__()
        self.method, self.classes = method, classes
        module = importlib.import_module(f".vendor.baseline_expansion.{method}", "milplus")
        cls, width_key = BASELINE_MODELS[method]
        if hidden is not None:
            config[width_key] = hidden
        self.loss_config = {}
        if method == "carmil":
            self.loss_config = {k: config.pop(k, default) for k, default in
                                (("alpha_effect", .2), ("alpha_attention", .2), ("attention_loss", "cosine"))}
        if method == "fcmil":
            self.loss_config = {k: config.pop(k, default) for k, default in
                                (("topk_ratio", .03), ("lam", .5), ("causal_weight", .1))}
            if not 0 < self.loss_config["topk_ratio"] < .5 or self.loss_config["causal_weight"] <= 0:
                raise ValueError("FC-MIL requires an active causal regularizer and topk_ratio in (0,.5)")
        if method == "attrimil":
            self.spatial_weight = config.pop("spatial_weight", .1)
            self.rank_weight = config.pop("rank_weight", .1)
            self.neighbors = config.pop("neighbors", 4)
            self.memory = module.AttributeMemory(classes, config.pop("queue_size", 4))
            if min(self.spatial_weight, self.rank_weight) <= 0:
                raise ValueError("AttriMIL requires both spatial and ranking constraints")
        if method == "nnmil":
            self.fixed_bag_size = config.pop("fixed_bag_size", "auto")
            self.batch_size = config.pop("batch_size", 4)
            if self.batch_size < 2:
                raise ValueError("nnMIL requires batched training")
        if method == "pneamil" and config.get("lambda_l1", 5.) <= 0:
            raise ValueError("PNEA requires positive evidence sparsity weight")
        self.model = getattr(module, cls)(in_dim=dim, num_classes=classes, **config)

    def train(self, mode=True):
        super().train(mode)
        if self.method == "asmil":
            self.model.anchor.eval()
        return self

    def get_extra_state(self):
        if self.method != "attrimil":
            return {}
        return {key: [[x.detach().cpu() for x in queue] for queue in getattr(self.memory, key)]
                for key in ("positive", "negative")}

    def set_extra_state(self, state):
        if self.method == "attrimil":
            for key in ("positive", "negative"):
                for queue, saved in zip(getattr(self.memory, key), state[key]):
                    queue.clear()
                    queue.extend(saved)

    def after_step(self):
        if self.method == "asmil":
            self.model.update_anchor()

    def forward(self, x, coords=None, label=None, **kwargs):
        out = self.model(x, **kwargs)
        if not self.training:
            if self.method == "nnmil":
                out["probabilities"] = out["chunk_probabilities"].mean(0)
            return out
        if self.method == "pneamil":
            out["regularization_loss"] = out["sparsity_loss"]
        if self.method == "asmil":
            out["regularization_loss"] = out["consistency_loss"]
        if label is not None and self.method == "carmil":
            from .vendor.baseline_expansion.carmil import car_mil_loss
            out["total_loss"] = car_mil_loss(out, label, nn.CrossEntropyLoss(), **self.loss_config)
        if label is not None and self.method == "fcmil":
            from .vendor.baseline_expansion.fcmil import causal_mil_loss
            cfg = {k: v for k, v in self.loss_config.items() if k != "causal_weight"}
            out["regularization_loss"] = self.loss_config["causal_weight"] * causal_mil_loss(
                self.model, out["logits"], out["sampled_features"], out["sampled_attention"], **cfg)
        if label is not None and self.method == "attrimil":
            from .vendor.baseline_expansion.attrimil import rank_constraint, spatial_constraint
            for key in ("positive", "negative"):
                for queue in getattr(self.memory, key):
                    for i in range(len(queue)):
                        queue[i] = queue[i].to(x.device)
            spatial = spatial_constraint(out["attribute_scores"], spatial_neighbors(coords, self.neighbors), self.classes)
            rank = rank_constraint(x, label, self.model, out["attribute_scores"], self.memory, self.classes)
            out["regularization_loss"] = self.spatial_weight * spatial + self.rank_weight * rank
        return out


class MADMIL(nn.Module):
    """Author's split-feature multihead gated attention, generalized input width."""
    def __init__(self, dim, classes, hidden=512, heads=2, dropout=0., gated=True):
        super().__init__()
        if hidden % heads or hidden // heads < 2:
            raise ValueError("MAD-MIL hidden must be divisible by heads; each head >=2 channels")
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([Attention(hidden // heads, hidden // heads // 2, gated=gated, dropout=dropout)
                                    for _ in range(heads)])
        self.classifier = nn.Linear(hidden, classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, coords=None, label=None):
        chunks = self.project(x).chunk(len(self.heads), -1)
        attention = [head(chunk) for head, chunk in zip(self.heads, chunks)]
        embedding = torch.cat([a @ h for a, h in zip(attention, chunks)], -1)
        return {"logits": self.classifier(embedding), "attention": torch.cat(attention), "embedding": embedding}


def build_expansion(method, dim, classes, config):
    if method == "rrt":
        from .upstream import load_rrt
        author = load_rrt()

        class RRTAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = config.pop("hidden", 512)
                self.model = author.RRTMIL(input_dim=dim, mlp_dim=hidden, n_classes=classes, **config)

            def forward(self, x, coords=None, label=None):
                return {"logits": self.model(x.unsqueeze(0))}

        return RRTAdapter()
    if method in BASELINE_MODELS:
        return BaselineAdapter(method, dim, classes, **config)
    if method == "madmil":
        return MADMIL(dim, classes, **config)
    if method == "litemil":
        from .vendor.litemil.model import LiteMIL

        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                options = {"hidden_dim": config.pop("hidden", 256), "dropout": .25,
                           "num_heads": 4, "num_queries": 1, **config}
                self.model = LiteMIL(input_dim=dim, num_classes=classes, **options)

            def forward(self, x, coords=None, label=None):
                logits, extra = self.model(x.unsqueeze(0))
                return {"logits": logits, "attention": extra["attn"]}

        return Adapter()
    raise ValueError(f"No expansion adapter {method}")


EXPANSION_METHODS = {*BASELINE_MODELS, "madmil", "litemil", "rrt"}

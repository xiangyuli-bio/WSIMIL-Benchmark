"""DGR author's architecture with lesion diversity and class-conditional center loss."""
import torch
from torch import nn
from torch.nn import functional as F


class DGRMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, lesions=11, warmup_epochs=5,
                 diversity_weight=.1, similarity_weight=.1, jitter=1e-10,
                 dropout_node=0., dropout_patch=0.):
        super().__init__()
        if classes != 2 or hidden % 8 or lesions > hidden or min(diversity_weight, similarity_weight, jitter) <= 0:
            raise ValueError("DGR requires binary normal/abnormal classes, hidden divisible by 8 and active lesion objectives")
        from .upstream import load_dgr
        self.model = load_dgr().DGRMIL(dim, num_classes=classes, L=hidden, n_lesion=lesions,
                                     dropout_node=dropout_node, dropout_patch=dropout_patch)
        self.epoch, self.warmup_epochs = 0, warmup_epochs
        self.diversity_weight, self.similarity_weight, self.jitter = diversity_weight, similarity_weight, jitter

    def forward(self, x, coords=None, label=None):
        if self.training and label is None:
            raise ValueError("DGR training requires class identity for center updates")
        mode = "normal" if label is None or int(label.item()) == 0 else "abnormal"
        result = self.model(x.unsqueeze(0), bag_mode=mode)
        logits = result[0]
        sigmoid = logits.sigmoid()
        out = {"logits": logits, "probabilities": sigmoid / sigmoid.sum(-1, keepdim=True), "attention": result[1]}
        if self.training:
            target = F.one_hot(label, logits.shape[-1]).to(logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if self.epoch >= self.warmup_epochs:
                positive, negative, lesion = result[3:]
                vectors = lesion.squeeze(0)
                normalized = F.normalize(vectors.double(), dim=-1)
                gram = normalized @ normalized.T + self.jitter * torch.eye(len(vectors), device=x.device, dtype=torch.float64)
                sign, determinant = torch.linalg.slogdet(gram)
                if sign <= 0:
                    raise FloatingPointError("DGR lesion Gram matrix is not positive definite")
                diversity = -determinant.to(x.dtype)
                similarity = F.triplet_margin_with_distance_loss(vectors, positive.expand_as(vectors),
                    negative.expand_as(vectors), distance_function=lambda a, b: 1 - F.cosine_similarity(a, b), margin=1.)
                loss = loss + self.diversity_weight * diversity + self.similarity_weight * similarity
                out.update(diversity_loss=diversity, similarity_loss=similarity)
            out["total_loss"] = loss
        return out

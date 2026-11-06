"""Training mechanisms are explicit and checkpointed, never inferred from a class name."""
import torch
from torch.nn import functional as F


def classification_loss(method, output, target, bag_weight=0.7):
    if "total_loss" in output:
        return output["total_loss"]
    if method == "dsmil":
        onehot = F.one_hot(target, output["logits"].shape[-1]).float()
        return (F.binary_cross_entropy_with_logits(output["logits"], onehot)
                + F.binary_cross_entropy_with_logits(output["max_logits"], onehot)) / 2
    bag = F.cross_entropy(output["logits"], target)
    if method.startswith("clam_"):
        return bag_weight * bag + (1 - bag_weight) * output["instance_loss"]
    if method == "acmil_ga":
        bag = bag + F.cross_entropy(output["branch_logits"], target.expand(len(output["branch_logits"])))
    if "regularization_loss" in output:
        bag = bag + output["regularization_loss"]
    return bag


def probabilities(method, output):
    if "probabilities" in output:
        return output["probabilities"]
    if method == "dsmil":
        # Original dual-stream sigmoid ensemble, normalized for exclusive-class reporting.
        p = (output["logits"].sigmoid() + output["max_logits"].sigmoid()) / 2
        return p / p.sum(-1, keepdim=True)
    return output["logits"].softmax(-1)


def make_optimizers(model, method, lr, weight_decay):
    if hasattr(model, 'make_optimizers'):
        return model.make_optimizers(lr, weight_decay)
    if hasattr(model, 'make_optimizer'):
        return [model.make_optimizer(lr, weight_decay)]
    if getattr(model, 'optimizer_name', None) == 'adamw':
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)]
    if method == "dtfd":
        first = list(model.reduce.parameters()) + list(model.attention.parameters()) + list(model.tier1.parameters())
        second = list(model.tier2.parameters()) + list(model.tier2_attention.parameters())
        return [torch.optim.Adam(first, lr=lr, weight_decay=weight_decay),
                torch.optim.Adam(second, lr=lr, weight_decay=weight_decay)]
    return [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]


def classification_step(model, method, x, coords, target, optimizers, grad_clip=5.0, bag_weight=0.7):
    if method == "dtfd":
        group_logits, pseudo = model.first_tier(x)
        # Original optimizer ownership: tier 2 does not update tier 1 parameters.
        # Detach before optimizer0.step to avoid retaining a graph with mutated parameters.
        distilled = pseudo.detach()
        first_loss = F.cross_entropy(group_logits, target.expand(len(group_logits)))
        if not torch.isfinite(first_loss):
            raise FloatingPointError("Nonfinite DTFD tier 1 loss")
        optimizers[0].zero_grad(set_to_none=True)
        first_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizers[0].param_groups for p in g["params"]], grad_clip, error_if_nonfinite=True)
        optimizers[0].step()
        second_loss = F.cross_entropy(model.second_tier(distilled), target)
        if not torch.isfinite(second_loss):
            raise FloatingPointError("Nonfinite DTFD tier 2 loss")
        optimizers[1].zero_grad(set_to_none=True)
        second_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizers[1].param_groups for p in g["params"]], grad_clip, error_if_nonfinite=True)
        optimizers[1].step()
        return {"loss": float((first_loss + second_loss).detach()),
                "tier1_loss": float(first_loss.detach()), "tier2_loss": float(second_loss.detach())}
    out = model(x, coords=coords, label=target)
    loss = classification_loss(method, out, target, bag_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    optimizers[0].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizers[0].step()
    if method == "mhim_abmil":
        model.update_teacher()
    if hasattr(model, "after_step"):
        model.after_step()
    result = {"loss": float(loss.detach())}
    if "instance_loss" in out:
        result["instance_loss"] = float(out["instance_loss"].detach())
    if "regularization_loss" in out:
        result["regularization_loss"] = float(out["regularization_loss"].detach())
    if "kept_instances" in out:
        result["kept_instances"] = out["kept_instances"]
    return result


def cox_breslow_loss(risk, time, event):
    """Full risk set Cox partial likelihood, Breslow handling of tied event times."""
    risk, time, event = risk.flatten(), time.flatten(), event.flatten().bool()
    if not event.any():
        raise ValueError("Cox training requires observed events")
    # Entry j is in patient i's risk set iff time_j >= time_i.
    log_denom = torch.logsumexp(risk.unsqueeze(0).expand(len(risk), -1).masked_fill(
        time.unsqueeze(0) < time.unsqueeze(1), -torch.inf), dim=1)
    return -(risk[event] - log_denom[event]).mean()

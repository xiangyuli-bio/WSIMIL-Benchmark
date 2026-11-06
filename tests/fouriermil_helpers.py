"""Independent fixed author model/dataset/steps and framework compatibility loop."""

import ast
from collections import defaultdict, OrderedDict
from contextlib import nullcontext
import copy
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

from milplus.data import Bundle
from milplus.upstream import pinned_text, verified_source
from attrimil_helpers import make_task
from pneamil_helpers import equal_tree


def literal_sources():
    root, pin = verified_source("fouriermil_author")
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", pin], cwd=root, text=True).splitlines()
    model_path, = [p for p in tracked if p.startswith("models/") and p.endswith(".py")
                   and p not in {"models/__init__.py", "models/model_interface.py"}]

    def selected(relative, names=None, additions=None, device_rewrite=False):
        source = pinned_text(root, pin, relative)
        if device_rewrite:
            assert source.count("self.cls_token.expand(B, -1, -1).cuda()") == 1
            source = source.replace("self.cls_token.expand(B, -1, -1).cuda()",
                                    "self.cls_token.expand(B, -1, -1).to(h.device)")
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                     and (names is None or n.name in names)]
        scope = {
            "__name__": "_independent_literal_fourier_" + Path(relative).stem,
            "torch": torch, "nn": torch.nn, "F": torch.nn.functional, "np": np,
            "math": math, "Optimizer": torch.optim.Optimizer, "defaultdict": defaultdict,
            "optim": torch.optim, "has_apex": False, **(additions or {}),
        }
        exec(compile(tree, str(root / relative), "exec"), scope)
        return SimpleNamespace(**scope)

    models = selected(model_path, device_rewrite=True)
    class_name = [n.name for n in ast.parse(pinned_text(root, pin, model_path)).body
                  if isinstance(n, ast.ClassDef)][-1]
    radam = selected("MyOptimizer/radam.py", {"RAdam"})
    lookahead = selected("MyOptimizer/lookahead.py", {"Lookahead"})
    factory = selected("MyOptimizer/optim_factory.py", {"create_optimizer", "add_weight_decay"}, {
        "RAdam": radam.RAdam, "Lookahead": lookahead.Lookahead,
    })
    interface = selected("models/model_interface.py", {"ModelInterface"}, {
        "pl": SimpleNamespace(LightningModule=torch.nn.Module),
    })
    dataset = selected("datasets/camel_data.py", {"CamelData"}, {
        "pd": pd, "Path": Path, "data": torch.utils.data,
    })
    val = selected("utils/utils.py", {"cross_entropy_torch"})
    return SimpleNamespace(
        Model=getattr(models, class_name), factory=factory.create_optimizer,
        Interface=interface.ModelInterface, Dataset=dataset.CamelData,
        val_loss=val.cross_entropy_torch,
    )


def named_optimizer(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return {
        "fast_state": copy.deepcopy(optimizer.base_optimizer.state_dict()),
        "original_RAdam_ten_slot_cache": copy.deepcopy(optimizer.base_optimizer.buffer),
        "slow_buffers_by_parameter_name": {
            names[id(p)]: copy.deepcopy(value["slow_buffer"])
            for p, value in optimizer.state.items()
        },
    }


def compat_optimizer(author, model, recipe):
    optimizer = author.factory(SimpleNamespace(**recipe), model)
    # Same explicitly disclosed framework interface repair. The author
    # Lookahead step and RAdam definitions remain untouched.
    for operation in ("step", "state_dict", "load_state_dict"):
        for when in ("pre", "post"):
            setattr(optimizer, f"_optimizer_{operation}_{when}_hooks", OrderedDict())
    return optimizer


def original_datasets(task, author, out):
    bundle = Bundle(task)
    features = out / "original_features"
    features.mkdir()
    columns = {}
    for subset in ("train", "val"):
        rows = bundle.records(subset)
        columns[subset] = pd.Series([r["slide_id"] for r in rows])
        columns[subset + "_label"] = pd.Series([int(r["target"]) for r in rows])
        for row in rows:
            tensor, _ = bundle.load(row)
            torch.save(tensor, features / (row["slide_id"] + ".pt"))
    pd.DataFrame(columns).to_csv(out / "fold0.csv")
    cfg = SimpleNamespace(nfold=4, fold=0, data_dir=str(features),
                          label_dir=str(out) + "/", data_shuffle=False)
    return bundle, {s: author.Dataset(cfg, s) for s in ("train", "val")}


def independent_complete(task, profile, out, *, device):
    """Original dataset and step bodies, full200 maximum, literal VAL early stop."""
    from milplus.engine import rng_state, seed_everything
    out.mkdir()
    author = literal_sources()
    bundle, datasets = original_datasets(task, author, out)
    dev, amp = torch.device(device), profile["author_recipe"]["precision"] == 16
    seed_everything(2026)
    model = author.Model(n_classes=len(bundle.classes)).to(dev)
    interface = SimpleNamespace(model=model, loss=torch.nn.CrossEntropyLoss(),
                                data=[], n_classes=len(bundle.classes))
    optimizer = compat_optimizer(author, model, profile["author_recipe"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, batch_count, optimizer_steps, skipped_steps = [], 0, 0, 0
    best, wait = float("inf"), 0

    class Tap(Sampler):
        def __init__(self, data):
            self.sampler = RandomSampler(data)
            self.order = []

        def __iter__(self):
            for index in self.sampler:
                self.order.append(index)
                yield index

        def __len__(self):
            return len(self.sampler)

    def capture():
        return copy.deepcopy({
            "model": model.state_dict(), "optimizer": named_optimizer(model, optimizer),
            "gradients": {n: None if p.grad is None else p.grad for n, p in model.named_parameters()},
            "scaler": scaler.state_dict(), "rng": rng_state(), "model_training": model.training,
        })

    for epoch in range(200):
        model.train()
        interface.data = [{"count": 0, "correct": 0} for _ in bundle.classes]
        sampler = Tap(datasets["train"])
        loader = DataLoader(datasets["train"], batch_size=1, sampler=sampler, num_workers=8)
        traces = []
        for index, batch in enumerate(loader):
            batch = [v.to(dev) for v in batch]
            if index % 2 == 0:
                optimizer.zero_grad(set_to_none=False)
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                raw = author.Interface.training_step(interface, batch, index)["loss"]
                loss = raw / 2
            assert torch.isfinite(raw)
            scaler.scale(loss).backward()
            boundary = (index + 1) % 2 == 0 or index + 1 == len(loader)
            before = scaler.get_scale()
            if boundary:
                if amp:
                    scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            skipped = scaler.get_scale() < before
            optimizer_steps += int(boundary and not skipped)
            skipped_steps += int(skipped)
            traces.append({
                "loss": float(raw.detach()), "normalized_loss": float(loss.detach()),
                "target": batch[1].cpu().tolist(), "patch_count": batch[0].shape[1],
                "accumulation_index": index % 2, "optimizer_boundary": boundary,
                "optimizer_step_skipped": skipped, "grad_scale_before": before,
                "grad_scale_after": scaler.get_scale(),
                "slide_id": bundle.records("train")[sampler.order[index]]["slide_id"],
            })
            batch_count += 1
        train_statistics = copy.deepcopy(interface.data)
        model.eval()
        interface.data = [{"count": 0, "correct": 0} for _ in bundle.classes]
        outputs = []
        with torch.no_grad():
            for index, batch in enumerate(DataLoader(
                    datasets["val"], batch_size=1, shuffle=False, num_workers=8)):
                batch = [v.to(dev) for v in batch]
                with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                    outputs.append(author.Interface.validation_step(interface, batch, index))
            logits = torch.cat([row["logits"] for row in outputs], dim=0)
            targets = torch.stack([row["label"] for row in outputs], dim=0)
            val_loss = author.val_loss(logits, targets)
            assert torch.isfinite(val_loss)
        value = float(val_loss)
        improved = value < best
        best, wait = (value, 0) if improved else (best, wait + 1)
        history.append({
            "epoch": epoch + 1, "val": {
                "loss": value, "literal_loss_dtype": str(val_loss.dtype),
                "forward_logits_dtype": str(logits.dtype),
                "class_statistics": copy.deepcopy(interface.data),
                "probabilities": torch.cat([r["Y_prob"] for r in outputs]).cpu().tolist(),
                "targets": torch.cat([r["label"] for r in outputs]).cpu().tolist(),
            },
            "train_class_statistics": train_statistics, "train_trace": traces,
            "train_order": [bundle.records("train")[i]["slide_id"] for i in sampler.order],
            "improved": improved, "best_val_loss": best, "wait_count": wait,
            "optimizer_steps": optimizer_steps,
        })
        last = capture()
        if improved:
            selected = copy.deepcopy(last)
        print(f"Independent Fourier epoch={epoch+1}/200 val={value} wait={wait}", flush=True)
        if wait >= 10:
            break
    result = {
        **last, "selected": selected, "history": history, "train_batches": batch_count,
        "optimizer_steps": optimizer_steps, "skipped_steps": skipped_steps,
        "actual_original_dataset_and_step_bodies": True, "maximum_epoch_budget": 200,
        "original_Lightning_environment_reproduced": False,
    }
    torch.save(result, out / "independent_complete_reference.pt")
    # Compare complete serialized states on the same device. The adapter
    # checkpoints are read with map_location="cpu" by the test harness.
    # Preserve every tensor bit, dtype, shape, gradient and numerical assertion.
    return torch.load(out / "independent_complete_reference.pt",
                      map_location="cpu", weights_only=False)

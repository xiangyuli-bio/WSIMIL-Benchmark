"""Independent unchanged author model files, datasets and literal MR trainer."""

import ast
import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
import types

import numpy as np
import pandas as pd
import torch
import yaml

from milplus.data import Bundle
from milplus.upstream import verified_source
from carmil_helpers import make_task as car_task
from pneamil_helpers import equal_tree


def module(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def literal_sources():
    root, _ = verified_source("mrblock")
    models = module("_independent_mr_original_models", root / "models/__init__.py", package=True)
    data = module("_independent_mr_original_data", root / "data.py")
    metrics = module("_independent_mr_original_metrics", root / "metrics.py")
    optimizer = module("_independent_mr_original_optim", root / "optim.py")
    text = (root / "engine.py").read_text()
    tree = ast.parse(text)
    selected = {"EarlyStopping", "train_loop", "validate"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in selected]
    assert {node.name for node in nodes} == selected
    scope = {
        "torch": torch, "nn": torch.nn, "np": np, "os": os, "DEVICE": torch.device("cpu"),
        "AccuracyLogger": metrics.AccuracyLogger, "calculate_error": metrics.calculate_error,
        "validation_auc_auprc": metrics.validation_auc_auprc,
        "is_plateau_scheduler": optimizer.is_plateau_scheduler,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / "engine.py"), "exec"), scope)
    seed_tree = ast.parse((root / "train.py").read_text())
    seed_node = next(n for n in seed_tree.body if isinstance(n, ast.FunctionDef) and n.name == "seed_torch")
    seed_scope = {"torch": torch, "np": np, "random": random, "os": os, "DEVICE": torch.device("cpu")}
    exec(compile(ast.Module(body=[seed_node], type_ignores=[]),
                 str(root / "train.py"), "exec"), seed_scope)
    return types.SimpleNamespace(
        models=models, data=data, metrics=metrics, optimizer=optimizer,
        trainer=types.SimpleNamespace(**{name: scope[name] for name in selected}),
        seed=seed_scope["seed_torch"], block=sys.modules[models.__name__ + ".mr_block"].ManifoldResidualBlock)


def make_task(root, *, dimension=512, concepts=False):
    task = car_task(root, dimension=dimension)
    if concepts:
        values = [torch.randn(3, 512, generator=torch.Generator().manual_seed(31)),
                  torch.randn(4, 512, generator=torch.Generator().manual_seed(32))]
        checkpoint = task / "nsclc_concepts.pt"
        torch.save(values, checkpoint)
        config = yaml.safe_load((task / "task.yaml").read_text())
        config["mrblock_concepts"] = {"path": checkpoint.name, "task": "nsclc"}
        config["label_provenance"] += "; synthetic concept matrices, never real original assets"
        (task / "task.yaml").write_text(yaml.safe_dump(config))
    return task


def args(task):
    return types.SimpleNamespace(
        opt="adamw", lr=5e-4, reg=1e-5, scheduler="linear", max_epochs=200,
        task="task_1_tcga_nsclc_4_shot", concept_root=str(task))


def native_model(original, profile, task, dimension=512):
    return original.models.build_model(
        profile["native_recipe"]["model_type"], dimension, 2, .25, "small", args(task))


def mode_state(model, original):
    return {
        "training": {name: value.training for name, value in model.named_modules()},
        "merged": {name: value.merged for name, value in model.named_modules()
                   if isinstance(value, original.block)},
    }


def literal_datasets(task, original):
    bundle = Bundle(task)
    root = task / "independent_mr_features"
    (root / "pt_files").mkdir(parents=True)
    result = {}
    for subset in ("train", "val"):
        rows = bundle.records(subset)
        for row in rows:
            features, _ = bundle.load(row)
            torch.save(features, root / "pt_files" / (row["slide_id"] + ".pt"))
        frame = pd.DataFrame([{
            "slide_id": row["slide_id"], "case_id": row["patient_id"],
            "label": int(row["target"])} for row in rows])
        result[subset] = original.data.GenericSplit(frame, str(root), num_classes=2)
    return result


class MeasuredCE(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.values = []

    def forward(self, logits, target):
        result = super().forward(logits, target)
        self.values.append(float(result.detach()))
        return result


def independent_complete(task, profile, output):
    from milplus.engine import rng_state
    original = literal_sources()
    datasets = literal_datasets(task, original)
    original.seed(2026)
    model = native_model(original, profile, task)
    optimizer = original.optimizer.get_optim(model, args(task))
    scheduler = original.optimizer.get_scheduler(optimizer, args(task))
    train_data, val_data = datasets["train"], datasets["val"]

    class TracedTrain(original.data.GenericSplit):
        def __getitem__(self, index):
            self.access.append(index)
            return super().__getitem__(index)

    train_data.__class__ = TracedTrain
    train_loader = original.data.get_split_loader(train_data, training=True)
    val_loader = original.data.get_split_loader(val_data)
    early = original.trainer.EarlyStopping(20, 50, True) if profile["native_recipe"]["early_stopping"] else None
    history, best = [], None

    def snapshot():
        return copy.deepcopy({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "early_stopping": vars(early).copy() if early else None,
            "rng": rng_state(), "mode_state": mode_state(model, original),
        })

    for epoch in range(200):
        train_data.access = []
        original.trainer.train_loop(epoch, model, train_loader, optimizer, scheduler, 2,
                                    torch.nn.CrossEntropyLoss())
        criterion = MeasuredCE()
        before = early.best_score if early else None
        stop = original.trainer.validate(
            0, epoch, model, val_loader, 2, early, criterion, str(output), scheduler)
        val_loss = sum(criterion.values) / len(criterion.values)
        improved = bool(early and (before is None or -val_loss >= before))
        history.append({
            "epoch": epoch + 1, "val_loss": val_loss, "improved": improved,
            "best_loss": early.val_loss_min if early else None,
            "retain": early.counter if early else None, "train_order": list(train_data.access),
            "scheduler_steps": scheduler.last_epoch, "lr_next": optimizer.param_groups[0]["lr"],
        })
        if improved:
            best = snapshot()
        if stop:
            break
    if early is None:
        best = snapshot()
    assert best is not None
    return {**snapshot(), "best": best, "history": history,
            "author_stopped": bool(early and early.early_stop)}

"""Independent original FC-MIL files and its literal repaired manager train body."""

import ast
import copy
import hashlib
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.upstream import pinned_text, verified_source
from carmil_helpers import make_task
from pneamil_helpers import equal_tree


def module(name, path, text=None, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    if text is None:
        spec.loader.exec_module(value)
    else:
        exec(compile(text, str(path), "exec"), value.__dict__)
    return value


def literal_sources():
    root, pin = verified_source("fcmil")
    name = "_independent_fcmil_original"
    package = SimpleNamespace()
    parent = __import__("types").ModuleType(name)
    parent.__path__ = [str(root)]
    sys.modules[name] = parent
    package.modules = module(name + ".modules", root / "models/modules.py")
    text = pinned_text(root, pin, "models/MIL_models.py")
    assert text.count("from models.modules import FAA") == 1
    text = text.replace("from models.modules import FAA", f"from {name}.modules import FAA")
    package.models = module(name + ".models", root / "models/MIL_models.py", text)
    package.loss = module(name + ".causal_mil_loss", root / "src/causal_mil_loss.py")
    text = pinned_text(root, pin, "src/trainer.py")
    assert text.count("from src.causal_mil_loss import causal_mil_loss") == 1
    text = text.replace("from src.causal_mil_loss import causal_mil_loss",
                        f"from {name}.causal_mil_loss import causal_mil_loss")
    package.trainer = module(name + ".trainer", root / "src/trainer.py", text)
    package.datasets = module(name + ".dataset", root / "src/dataset.py")
    tree = ast.parse(pinned_text(root, pin, "src/utils.py"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collate_fn")
    scope = {"torch": torch, "List": List, "Tuple": Tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(root / "src/utils.py"), "exec"), scope)
    package.collate = scope["collate_fn"]
    tree = ast.parse(pinned_text(root, pin, "src/trainer_manager.py"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingManager")
    scope = {
        "torch": torch, "optim": torch.optim, "nn": torch.nn,
        "lr_scheduler": torch.optim.lr_scheduler, "logger": logging.getLogger("FC-native-reference"),
        "Tuple": Tuple, "List": List, "Dict": Dict, "Any": Any, "Optional": Optional,
        "_run_epoch": package.trainer._run_epoch,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(root / "src/trainer_manager.py"), "exec"), scope)
    package.manager = scope["TrainingManager"]
    package.manager_scope = scope
    return package


def independent_complete(task, profile, output):
    from milplus.engine import rng_state, seed_everything
    output.mkdir()
    author = literal_sources()
    bundle = Bundle(task)
    cfg = SimpleNamespace(**profile["benchmark_recipe"])
    cfg.seed, cfg.label_list, cfg.mapping, cfg.data_root = 2026, bundle.classes, {}, str(output)
    holder = {"history": [], "traces": [], "updates": 0}
    split_data = {}
    (output / "actual_original_features").mkdir()
    for subset in ("train", "val"):
        split_data[subset] = []
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            path = output / "actual_original_features" / (row["slide_id"] + ".pt")
            torch.save(features, path)
            split_data[subset].append((row["slide_id"], str(row["label"]), str(path)))

    class ObservedDataset(author.datasets.MILDataset):
        def __init__(self, split_data, cfg):
            # Original BaseDataset parses before assigning its logger. Initialize
            # only that logging field; keep the actual dataset/parser/getitem.
            self.logger = logging.getLogger("FC-native-reference-dataset")
            super().__init__(split_data, cfg)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            self.access.append(self.list_sample[index][0])
            self.last_target = result[1]
            return result

    datasets = {
        subset: ObservedDataset(split_data[subset], cfg) for subset in ("train", "val")}
    for dataset in datasets.values():
        dataset.access = []

    def snapshot(manager):
        return copy.deepcopy({
            "model": manager.model.state_dict(), "optimizer": manager.optimizer.state_dict(),
            "scheduler": manager.scheduler.state_dict(), "best_auc": manager.best_auc,
            "model_training": manager.model.training, "rng": rng_state(),
        })

    def observed_causal(model, y_orig, bag_feats, attention_weights, topk_ratio=.03, lam=.5):
        cr = author.loss.causal_mil_loss(
            model, y_orig, bag_feats, attention_weights, topk_ratio=topk_ratio, lam=lam)
        target = torch.tensor([datasets["train"].last_target], device=bag_feats.device)
        ce = torch.nn.functional.cross_entropy(y_orig, target)
        holder["traces"].append({
            "features_sha256": hashlib.sha256(bag_feats.detach().cpu().numpy().tobytes()).hexdigest(),
            "target": target.cpu().tolist(), "patch_count": bag_feats.shape[1],
            "loss": (ce + cfg.alpha_cr * cr).item(),
            "loss_ce": ce.item(), "loss_cr": cr.item(),
            "correct": int(y_orig.argmax(1).eq(target).sum()),
            "logits": y_orig.detach().cpu().tolist(),
            "lr_used": holder["manager"].optimizer.param_groups[0]["lr"],
        })
        holder["updates"] += 1
        return cr

    author.trainer.causal_mil_loss = observed_causal

    def repaired_epoch(*, writer=None, **kwargs):
        # Exact original epoch body, only repair the unsupported logging keyword
        # and append real validation AUC from observed outputs and actual labels.
        model = kwargs["model"]
        captured = []
        hook = None
        if not kwargs.get("is_training", True):
            hook = model.register_forward_hook(
                lambda module, args, result: captured.append(
                    torch.softmax(result[0], dim=1).detach().cpu().numpy()[0]))
        try:
            result = author.trainer._run_epoch(**kwargs)
        finally:
            if hook is not None:
                hook.remove()
        if captured:
            probabilities = np.stack(captured)
            targets = np.asarray([
                bundle.classes.index(record[1]) for record in split_data["val"]], dtype=int)
            result["auc"] = float(np.mean([
                roc_auc_score(targets == i, probabilities[:, i])
                for i in range(len(bundle.classes))]))
        return result

    class ObservedPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def step(self, metrics, epoch=None):
            super().step(metrics, epoch)
            manager = holder["manager"]
            holder["history"].append({
                "epoch": manager.current_epoch + 1, "train": holder["train"], "val": holder["val"],
                "improved": holder["improved"], "best_auc": manager.best_auc,
                "lr_next": [g["lr"] for g in manager.optimizer.param_groups],
                "train_order": list(datasets["train"].access),
                "train_trace": copy.deepcopy(holder["traces"]), "updates": holder["updates"],
            })
            if holder["improved"]:
                holder["best"] = snapshot(manager)
            holder["last"] = snapshot(manager)
            datasets["train"].access.clear()
            holder["traces"].clear()

    author.manager_scope["_run_epoch"] = repaired_epoch
    author.manager_scope["lr_scheduler"] = SimpleNamespace(ReduceLROnPlateau=ObservedPlateau)

    class ObservedManager(author.manager):
        def _log_metrics(self, train_metrics, val_metrics):
            super()._log_metrics(train_metrics, val_metrics)
            holder["train"], holder["val"] = copy.deepcopy(train_metrics), copy.deepcopy(val_metrics)

        def _save_best_model(self, val_metrics):
            before = self.best_auc
            super()._save_best_model(val_metrics)
            holder["improved"] = self.best_auc > before

    seed_everything(2026)
    manager = ObservedManager.__new__(ObservedManager)
    manager.cfg, manager.device, manager.current_epoch = cfg, torch.device("cpu"), 0
    manager._set_random_seed()
    manager.model = author.models.FCMIL(n_classes=len(bundle.classes))
    manager.best_auc, manager.best_loss = 0., float("inf")
    manager.checkpoint_path = str(output / "literal_original_selected_weights.pt")
    manager.writer = None
    manager.dataloaders = {
        subset: DataLoader(dataset, batch_size=1, shuffle=subset == "train",
                           num_workers=0, collate_fn=author.collate)
        for subset, dataset in datasets.items()}
    holder["manager"] = manager
    manager._build_optimizer()
    manager.train()  # complete unchanged original manager body, explicit100epoch benchmark
    assert len(holder["history"]) == 100
    result = {
        **holder["last"], "selected": holder["best"], "history": holder["history"],
        "complete_original_manager_train_body_executed": True,
        "literal_original_selected_weights": manager.checkpoint_path,
        "author_training_configuration_recovered": False,
        "benchmark_configuration": profile["benchmark_recipe"],
    }
    torch.save(result, output / "independent_complete_reference.pt")
    return result

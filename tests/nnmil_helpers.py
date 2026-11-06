"""Independent original nnMIL trainer with observation-only subclasses/hooks."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from milplus.data import Bundle
from milplus.upstream import verified_source
from pneamil_helpers import equal_tree


def make_task(root, dimension=768, train_count=32):
    root.mkdir()
    (root / "features").mkdir()
    generator = np.random.default_rng(42)
    slides, splits, index = [], [], []
    for i in range(train_count + 16):
        subset = "train" if i < train_count else "val" if i < train_count + 8 else "test"
        slide, patient, target = f"s{i:03}", f"p{i:03}", i % 2
        values = generator.normal(0, .2, (7 + i % 27, dimension)).astype("f4")
        values[:, :min(256, dimension)] += .8 if target else -.8
        np.save(root / "features" / f"{slide}.npy", values)
        slides.append({"slide_id": slide, "patient_id": patient, "label": str(target)})
        splits.append({"patient_id": patient, "subset": subset})
        index.append({"slide_id": slide, "path": f"features/{slide}.npy"})
    pd.DataFrame(slides).to_csv(root / "slides.csv", index=False)
    pd.DataFrame(splits).to_csv(root / "split.csv", index=False)
    pd.DataFrame(index).to_csv(root / "features.csv", index=False)
    (root / "task.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "dataset_id": "synthetic_nnmil", "task_id": "synthetic_nnmil",
        "task_type": "classification", "classes": ["0", "1"],
        "synthetic": True, "label_provenance": "synthetic class signal for software checks only",
        "defaults": {"fm": "toy", "split": "fixed"},
        "encoders": {"toy": {"dimension": dimension, "index": "features.csv"}},
        "splits": {"fixed": "split.csv"},
    }))
    return root


def original_package():
    root, commit = verified_source("nnmil")
    if "nnMIL" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "nnMIL", root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["nnMIL"] = module
        spec.loader.exec_module(module)
    assert Path(sys.modules["nnMIL"].__file__).resolve() == (root / "__init__.py").resolve()
    return root


def sampler_state(loader):
    value = loader.batch_sampler
    state = {
        "kind": type(value).__name__,
        "state": copy.deepcopy({
            key: item for key, item in vars(value).items()
            if key not in {"dataset", "sampler", "generator", "_rng"}}),
        "generator": loader.generator.get_state() if loader.generator is not None else None,
    }
    if hasattr(value, "_rng"):
        state["sampler_rng"] = copy.deepcopy(value._rng.get_state())
    if hasattr(value, "sampler"):
        state["sub_sampler_state"] = copy.deepcopy({
            key: item for key, item in vars(value.sampler).items()
            if key not in {"data_source", "generator"}})
    return state


def stopping_state(value):
    return copy.deepcopy({key: item for key, item in vars(value).items()
                          if key not in {"logger", "save_dir", "model_type", "best_model_state"}})


def independent_complete(task, plan, output):
    from milplus.engine import rng_state
    original_package()
    import nnMIL.training.trainers.classification_trainer as original
    from nnMIL.training.callbacks.early_stopping import EarlyStopping
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    output.mkdir()
    h5 = output / "actual_original_features"
    h5.mkdir()
    bundle = Bundle(task)
    native_plan = copy.deepcopy(plan)
    native_plan["feature_dir"] = str(h5)
    native_plan["data_splits"] = {"official_split": {}}
    for subset in ("train", "val", "test"):
        rows = bundle.records(subset)
        native_plan["data_splits"]["official_split"][subset] = {"slide_info": [{
            "slide_id": row["slide_id"], "patient_id": row["patient_id"],
            "label": int(row["target"])} for row in rows]}
        for row in rows:
            features, coords = bundle.load(row)
            with h5py.File(h5 / (row["slide_id"] + ".h5"), "x") as handle:
                handle.create_dataset("features", data=features.numpy())
                handle.create_dataset("coords", data=(
                    coords.numpy() if coords is not None else np.zeros((len(features), 2), "f4")))
    plan_path = output / "unique_native_plan.json"
    plan_path.write_text(json.dumps(native_plan, indent=2) + "\n")
    import tempfile
    previous_temp = tempfile.tempdir
    # torch's default file-descriptor sharing creates AF_UNIX sockets here.
    # A pytest output path can exceed the kernel's socket-address limit.
    # Keep the original four workers and sharing strategy; shorten only the
    # independently owned transport directory, never the data or train recipe.
    transport = tempfile.TemporaryDirectory(
        prefix="nnmil-", dir=os.environ.get("MILPLUS_TEST_SOCKET_TMPDIR"))
    tempfile.tempdir = transport.name
    holder, history, traces = {}, [], []

    class ObservedTrainer(original.ClassificationTrainer):
        def forward_model(self, features, bag_sizes=None, **kwargs):
            result = super().forward_model(features, bag_sizes, **kwargs)
            if self.model.training:
                if traces and "lr_next" not in traces[-1]:
                    traces[-1]["lr_next"] = [g["lr"] for g in holder["optimizer"].param_groups]
                traces.append({
                    "features_sha256": hashlib.sha256(
                        features.detach().cpu().numpy().tobytes()).hexdigest(),
                    "bag_sizes": bag_sizes.cpu().tolist(),
                    "sampled_channels": self.model.last_train_idx.detach().cpu().tolist(),
                    "logits": result["logits"].detach().float().cpu().tolist(),
                })
            return result

        def evaluate(self, split="test"):
            result = super().evaluate(split)
            if split == "val":
                holder["metrics"] = result
            return result

    class ObservedStopping(EarlyStopping):
        def __call__(self, *args, **kwargs):
            before = self.best_score
            super().__call__(*args, **kwargs)
            trainer, optimizer = holder["trainer"], holder["optimizer"]
            traces[-1]["lr_next"] = [g["lr"] for g in optimizer.param_groups]
            history.append({
                "epoch": len(history) + 1, "metrics": holder["metrics"],
                "improved": before is None or self.best_score > before,
                "early_stopping": stopping_state(self),
                "train_trace": copy.deepcopy(traces),
                "total_updates": holder["updates"],
                "lr_next": [g["lr"] for g in optimizer.param_groups],
            })
            traces.clear()
            holder["last"] = copy.deepcopy({
                "model": trainer.model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": None, "early_stopping": stopping_state(self),
                "loader": sampler_state(trainer.train_loader), "rng": rng_state(),
            })

    def observe_optimizer(optimizer, args, kwargs):
        holder["optimizer"] = optimizer
        holder["updates"] = holder.get("updates", 0) + 1
        traces[-1]["lr_used"] = [g["lr"] for g in optimizer.param_groups]

    saved_callback = original.EarlyStopping
    original.EarlyStopping = ObservedStopping
    hook = register_optimizer_step_pre_hook(observe_optimizer)
    try:
        trainer = ObservedTrainer(str(plan_path), "nnmil", save_dir=str(output / "native"),
                                  seed=plan["random_seed"])
        holder["trainer"] = trainer
        trainer.train()  # complete literal native train method, no shortened budget
        assert history and (len(history) == 100 or holder["last"]["early_stopping"]["early_stop"])
        result = {
            **holder["last"], "selected_model": copy.deepcopy(trainer.model.state_dict()),
            "history": history,
            "original_best_file": str(output / "native/best_nnmil.pth"),
            "complete_native_train_body_executed": True,
        }
        torch.save(result, output / "independent_complete_original_reference.pt")
        return result
    finally:
        hook.remove()
        original.EarlyStopping = saved_callback
        tempfile.tempdir = previous_temp
        transport.cleanup()

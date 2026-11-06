"""Independent full author modules and original HDF5/neighbor preprocessing."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from milplus.data import Bundle
from milplus.upstream import pinned_text, verified_source
from pneamil_helpers import equal_tree


def module_from_file(name, path, text=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if text is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def literal_sources(*, repair_axis=True, repair_logging=True):
    root, commit = verified_source("attrimil")
    model = module_from_file("_attri_literal_model", root / "models/AttriMIL.py")
    utils = module_from_file("_attri_literal_utils", root / "utils.py")
    datasets = module_from_file("_attri_literal_dataset", root / "dataloader.py")
    package = types.ModuleType("models")
    package.__path__ = []
    constraint_text = pinned_text(root, commit, "constraints.py")
    if repair_axis:
        assert constraint_text.count("score = A[:, c]") == 1
        constraint_text = constraint_text.replace("score = A[:, c]", "score = A[0, c]")
    trainer_text = pinned_text(root, commit, "trainer_attrimil_abmil.py")
    if repair_logging:
        assert trainer_text.count("loss_ins_value = loss_ins.item()") == 1
        trainer_text = trainer_text.replace("loss_ins_value = loss_ins.item()", "loss_ins_value = 0.0")
    with patch.dict(sys.modules, {"utils": utils, "models": package,
                                 "models.AttriMIL": model, "dataloader": datasets}):
        constraints = module_from_file(
            "_attri_literal_constraints", root / "constraints.py", constraint_text)
        with patch.dict(sys.modules, {"constraints": constraints}):
            trainer = module_from_file(
                "_attri_literal_training", root / "trainer_attrimil_abmil.py", trainer_text)
    return model, constraints, utils, datasets, trainer


def original_neighbors(coords, output):
    root, commit = verified_source("attrimil")
    tree = ast.parse(pinned_text(root, commit, "create_3coords.py"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "find_nearest"]
    scope = {"np": np, "h5py": h5py, "name": "synthetic_original_geometry"}
    exec(compile(tree, str(root / "create_3coords.py"), "exec"), scope)
    original = output.with_name(output.stem + "_input.h5")
    with h5py.File(original, "w") as handle:
        handle["coords"] = coords
    scope["find_nearest"](str(original), str(output), patch_size=(256, 256))
    with h5py.File(output, "r") as handle:
        return handle["nearest"][:]


def make_task(root, *, dimension=512, classes=2, zero=False, train_slides=5):
    root = Path(root)
    root.mkdir()
    (root / "features").mkdir()
    rng = np.random.default_rng(3901)
    slides, index, splits = [], [], []
    for subset, count in [("train", train_slides), ("val", classes), ("test", classes)]:
        for i in range(count):
            slide = f"{subset}_{i:03d}"
            n = 17 + i
            features = (np.zeros((n, dimension), dtype=np.float32) if zero
                        else rng.normal(size=(n, dimension)).astype(np.float32))
            places = np.delete(np.arange(n + 3), [1, 7, 13])
            coords = np.stack([places % 5, places // 5], -1).astype(np.int64) * 256 + 1024
            path, cp = f"features/{slide}.pt", f"features/{slide}_coords.npy"
            torch.save(torch.from_numpy(features), root / path)
            np.save(root / cp, coords)
            slides.append({"slide_id": slide, "patient_id": slide, "label": f"c{i % classes}"})
            index.append({"slide_id": slide, "path": path, "coords_path": cp})
            splits.append({"patient_id": slide, "subset": subset})
    pd.DataFrame(slides).to_csv(root / "slides.csv", index=False)
    pd.DataFrame(index).to_csv(root / "features.csv", index=False)
    pd.DataFrame(splits).to_csv(root / "split.csv", index=False)
    config = {
        "schema_version": 1, "task_id": "synthetic_complete_author_attrimil",
        "synthetic": True, "label_provenance": "synthetic software fixture",
        "task_type": "classification", "classes": [f"c{i}" for i in range(classes)],
        "training_unit": "slide", "evaluation_unit": "patient",
        "slides": "slides.csv", "defaults": {"fm": "synthetic", "split": "fixed"},
        "splits": {"fixed": "split.csv"},
        "encoders": {"synthetic": {
            "dimension": dimension, "index": "features.csv",
            "weights": "synthetic", "preprocessing": "synthetic, no clinical interpretation",
            "spatial": {"unit": "level0_pixels", "origin": "top_left",
                        "patch_size": 256, "provenance": "synthetic physical-grid software fixture"},
        }},
    }
    (root / "task.yaml").write_text(yaml.safe_dump(config))
    return root


def literal_datasets(task, original):
    bundle = Bundle(task)
    data = task / "independent_original_h5"
    (data / "h5_coords_files").mkdir(parents=True)
    datasets = {}
    for subset in ["train", "val"]:
        rows = bundle.records(subset)
        for row in rows:
            x, c = bundle.load(row, require_coords=True)
            nearest = original_neighbors(c.numpy().astype(np.int64), data / (row["slide_id"] + "_neighbors.h5"))
            with h5py.File(data / "h5_coords_files" / (row["slide_id"] + ".h5"), "w") as handle:
                handle["features"] = x.numpy()
                handle["coords"] = c.numpy().astype(np.int64)
                handle["nearest"] = nearest
        frame = pd.DataFrame([{"slide_id": r["slide_id"], "label": int(r["target"])} for r in rows])
        datasets[subset] = original.Generic_Split(frame, str(data), num_classes=len(bundle.classes))
    return datasets

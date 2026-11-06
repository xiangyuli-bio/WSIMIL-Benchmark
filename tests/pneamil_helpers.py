"""Independent, explicitly synthetic fixtures for the complete PNEA recipe."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from milplus.upstream import verified_source


def literal_sources():
    root, _ = verified_source("pneamil")
    spec = importlib.util.spec_from_file_location("_pnea_literal_model", root / "PNEA_MIL.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    spec = importlib.util.spec_from_file_location("_pnea_literal_training", root / "train_PNEA-MIL.py")
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    with patch.dict(sys.modules, {"PNEA_MIL": model}):
        spec.loader.exec_module(training)
    return model, training


def make_task(root, *, dimension=1024, classes=2, zero=False, train_slides=10):
    """Original .pt bags, patient-disjoint labels and a partial final minibatch."""
    root = Path(root)
    root.mkdir()
    (root / "features").mkdir()
    rng = np.random.default_rng(3301)
    rows, index, splits = [], [], []
    for subset, count in [("train", train_slides), ("val", classes), ("test", classes)]:
        for i in range(count):
            slide = f"{subset}_{i:03d}"
            n = 1041 if subset == "val" and i == 0 else 3 + i
            values = np.zeros((n, dimension), dtype=np.float32) if zero else rng.normal(
                size=(n, dimension)).astype(np.float32)
            path = f"features/{slide}.pt"
            torch.save(torch.from_numpy(values), root / path)
            rows.append({"slide_id": slide, "patient_id": slide, "label": f"c{i % classes}"})
            index.append({"slide_id": slide, "path": path})
            splits.append({"patient_id": slide, "subset": subset})
    pd.DataFrame(rows).to_csv(root / "slides.csv", index=False)
    pd.DataFrame(index).to_csv(root / "features.csv", index=False)
    pd.DataFrame(splits).to_csv(root / "split.csv", index=False)
    task = {
        "schema_version": 1, "task_id": "synthetic_complete_pneamil",
        "synthetic": True, "label_provenance": "synthetic software fixture",
        "task_type": "classification", "classes": [f"c{i}" for i in range(classes)],
        "training_unit": "slide", "evaluation_unit": "patient",
        "slides": "slides.csv", "defaults": {"fm": "synthetic", "split": "fixed"},
        "splits": {"fixed": "split.csv"},
        "encoders": {"synthetic": {
            "dimension": dimension, "index": "features.csv",
            "weights": "synthetic", "preprocessing": "synthetic; no clinical interpretation",
        }},
    }
    (root / "task.yaml").write_text(yaml.safe_dump(task))
    return root


def equal_tree(left, right):
    """Exact tensor/RNG/optimizer comparisons, including every nested entry."""
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right

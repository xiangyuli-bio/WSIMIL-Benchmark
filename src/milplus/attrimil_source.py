"""Private loading of fixed author definitions; every necessary repair is named."""

import ast
import hashlib
import queue

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from .upstream import load_definitions, pinned_text, verified_source

MODEL_REPAIRS = (
    ("A_raw = torch.empty(self.n_classes, h.size(0), )",
     "A_raw = torch.empty(self.n_classes, h.size(0), device=h.device)"),
)
SPATIAL_REPAIRS = (("score = A[:, c]", "score = A[0, c]"),)
LOGGING_REPAIRS = (("loss_ins_value = loss_ins.item()", "loss_ins_value = 0.0"),)


def definitions(device="cpu"):
    model = load_definitions(
        "attrimil", "models/AttriMIL.py", ["Attn_Net_Gated", "AttriMIL"],
        {"torch": torch, "nn": nn, "F": nn.functional, "np": np},
        replacements=MODEL_REPAIRS,
    )
    constraints = load_definitions(
        "attrimil", "constraints.py", ["spatial_constraint", "rank_constraint"],
        {"torch": torch, "np": np, "device": torch.device(device)},
        replacements=SPATIAL_REPAIRS,
    )
    utils = load_definitions(
        "attrimil", "utils.py", ["make_weights_for_balanced_classes_split"],
        {"torch": torch},
    )
    validation = load_definitions(
        "attrimil", "trainer_attrimil_abmil.py", ["Accuracy_Logger", "validate"],
        {"torch": torch, "nn": nn, "np": np, "device": torch.device(device),
         "roc_auc_score": roc_auc_score},
    )
    return model, constraints, utils, validation


def contract():
    root, commit = verified_source("attrimil")
    paths = ["models/AttriMIL.py", "constraints.py", "utils.py",
             "trainer_attrimil_abmil.py", "create_3coords.py", "dataloader.py"]
    texts = {p: pinned_text(root, commit, p) for p in paths}
    tree = ast.parse(texts["trainer_attrimil_abmil.py"])
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_abmil")
    defaults = {a.arg: ast.literal_eval(d)
                for a, d in zip(train.args.args[-len(train.args.defaults):], train.args.defaults)}
    assert defaults["max_epoch"] == 200 and defaults["early_stopping"] is True
    assert "early_stopping = False" in texts["trainer_attrimil_abmil.py"]
    for source, replacements in [
        ("models/AttriMIL.py", MODEL_REPAIRS), ("constraints.py", SPATIAL_REPAIRS),
        ("trainer_attrimil_abmil.py", LOGGING_REPAIRS),
    ]:
        for old, _ in replacements:
            if texts[source].count(old) != 1:
                raise ValueError("Fixed AttriMIL repair no longer matches exactly once")
    if "loss_ins" in texts["utils.py"]:
        raise ValueError("Actual AttriMIL wildcard utility namespace changed")
    return {
        "commit": commit, "function_defaults": defaults,
        "files_sha256": {p: hashlib.sha256(t.encode()).hexdigest() for p, t in texts.items()},
        "model_replacements": [list(p) for p in MODEL_REPAIRS],
        "spatial_replacements": [list(p) for p in SPATIAL_REPAIRS],
        "logging_replacements_in_reference": [list(p) for p in LOGGING_REPAIRS],
        "original_summary": "DSMIL-style4-output summary conflicts with actual5-output AttriMIL",
        "optimizer": "SGD lr2e-4/momentum0.9/wd1e-5; no scheduler or clipping",
        "queue_update": "original rank_constraint; max4, rebuilt every epoch",
        "source_seed": "no explicit model RNG seeding in original main; unified seed disclosed",
    }

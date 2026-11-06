"""Load original PNEA definitions from the fixed external author repository."""

import ast
import hashlib

import numpy as np
import torch
from torch import nn

from .upstream import load_definitions, pinned_text, verified_source


def original_definitions():
    model = load_definitions(
        "pneamil", "PNEA_MIL.py", ["PNEA_MIL"],
        {"np": np, "torch": torch, "nn": nn},
    )
    trainer = load_definitions(
        "pneamil", "train_PNEA-MIL.py",
        ["train_one_epoch", "evaluate_loss"],
        {"torch": torch},
    )
    return model.PNEA_MIL, trainer


def source_contract():
    root, commit = verified_source("pneamil")
    text = pinned_text(root, commit, "train_PNEA-MIL.py")
    tree = ast.parse(text)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    defaults = {}
    for node in ast.walk(main):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    expected = {
        "--D_in": 1024, "--D": 64, "--C": 2, "--num_train_patches": 1024,
        "--batch_size": 8, "--epochs": 1000, "--early_stop_patience": 10,
        "--early_stop_threshold": 0.0005, "--lr": 1e-4,
        "--weight_decay": 0.0, "--lambda_l1": 5.0, "--random_seed": 0,
    }
    if any(defaults.get(key) != value for key, value in expected.items()):
        raise ValueError("Fixed PNEA author defaults differ from the reviewed complete recipe")
    return {
        "commit": commit, "author_defaults": expected,
        "trainer_blob_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "literal_selector_population": "TRAIN loader despite validation_loss variable name",
        "source_definition_replacements": [],
        "extra_model_initialization_or_pretrained_weights": None,
    }

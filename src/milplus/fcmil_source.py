"""Privately load unchanged FC-MIL definitions from the exact registered Git pin."""

import ast
import hashlib
import logging
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .upstream import load_definitions, pinned_text, verified_source


def definitions():
    modules = load_definitions("fcmil", "models/modules.py", ["DABlock", "FAA"], {
        "torch": torch, "nn": torch.nn, "F": torch.nn.functional,
    })
    models = load_definitions("fcmil", "models/MIL_models.py", ["FCMIL"], {
        "torch": torch, "nn": torch.nn, "FAA": modules.FAA,
    })
    loss = load_definitions("fcmil", "src/causal_mil_loss.py", ["causal_mil_loss"], {
        "torch": torch, "F": torch.nn.functional,
    })
    trainer = load_definitions("fcmil", "src/trainer.py", ["_run_epoch"], {
        "torch": torch, "nn": torch.nn, "F": torch.nn.functional, "tqdm": tqdm,
        "logging": logging, "Dict": Dict, "Optional": Optional, "Any": Any,
        "causal_mil_loss": loss.causal_mil_loss,
    })
    manager = load_definitions("fcmil", "src/trainer_manager.py", ["TrainingManager"], {
        "torch": torch, "optim": torch.optim, "nn": torch.nn,
        "lr_scheduler": torch.optim.lr_scheduler, "logger": logging.getLogger("FCMIL-native"),
        "Tuple": Tuple, "List": List, "Dict": Dict, "Any": Any, "Optional": Optional,
        "_run_epoch": trainer._run_epoch,
    })
    return SimpleNamespace(modules=modules, models=models, loss=loss,
                           trainer=trainer, manager=manager)


def contract():
    root, commit = verified_source("fcmil")
    names = ["LICENSE", "README.md", "src/main.py", "src/trainer_manager.py",
             "src/trainer.py", "src/causal_mil_loss.py", "src/dataset.py",
             "models/MIL_models.py", "models/modules.py"]
    texts = {name: pinned_text(root, commit, name) for name in names}
    tree = ast.parse(texts["src/trainer.py"])
    epoch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_epoch")
    assert "writer" not in [arg.arg for arg in epoch.args.args]
    assert "writer=self.writer" in texts["src/trainer_manager.py"]
    returns = [n for n in ast.walk(epoch) if isinstance(n, ast.Dict)]
    assert not any(isinstance(k, ast.Constant) and k.value == "auc" for n in returns for k in n.keys)
    assert "self.best_auc = 0.0" in texts["src/trainer_manager.py"]
    assert "current_auc > self.best_auc" in texts["src/trainer_manager.py"]
    assert "Apache License" in texts["LICENSE"] and "Version 2.0" in texts["LICENSE"]
    dataset_tree = ast.parse(texts["src/dataset.py"])
    base = next(n for n in dataset_tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseDataset")
    init = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    parse_index = next(i for i, n in enumerate(init.body)
                       if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Attribute)
                       and n.value.func.attr == "parse_input_list")
    logger_index = next(i for i, n in enumerate(init.body)
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Attribute) and t.attr == "logger" for t in n.targets))
    assert parse_index < logger_index
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True).splitlines()
    configurations = [
        name for name in tracked
        if name.lower().endswith((".yaml", ".yml", ".json", ".toml"))
        or "config" in name.rsplit("/", 1)[-1].lower()
    ]
    assert configurations == [], configurations
    return {
        "commit": commit, "file_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()},
        "original_writer_keyword_mismatch": True, "original_epoch_auc_missing": True,
        "original_dataset_logger_initialization_order_failure": True,
        "original_configuration_files": configurations,
        "original_training_configuration_recovered": False,
        "license": "Apache-2.0 fixed-tree text read; author source is privately loaded",
        "actual_attention_used_for_intervention": "mean-query attention returned by frequency FAA",
        "original_quadratic_spatial_and_frequency_attention_preserved": True,
        "patch_order_preserved_for_FFT": True,
    }

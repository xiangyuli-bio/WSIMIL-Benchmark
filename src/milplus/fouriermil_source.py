"""Load Fourier author Git objects privately; keep math and initialization."""

import ast
from collections import defaultdict
import hashlib
import math
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .fouriermil_spec import SOURCE_COMMIT, SOURCE_ID, RECIPE
from .upstream import load_definitions, pinned_text, verified_source


def locations():
    root, commit = verified_source(SOURCE_ID)
    if commit != SOURCE_COMMIT:
        raise ValueError("Fourier original source pin changed")
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True
    ).splitlines()
    models = [p for p in tracked if p.startswith("models/") and p.endswith(".py")
              and p not in {"models/__init__.py", "models/model_interface.py"}]
    configs = [p for p in tracked if p.startswith("Camelyon/") and p.endswith(".yaml")]
    if len(models) != 1 or len(configs) != 1:
        raise ValueError("Fourier fixed tree model/configuration layout changed")
    return root, commit, models[0], configs[0]


def definitions():
    root, commit, model_path, _ = locations()
    tree = ast.parse(pinned_text(root, commit, model_path))
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    model = load_definitions(SOURCE_ID, model_path, classes, {
        "torch": torch, "nn": torch.nn, "F": torch.nn.functional, "np": np,
    }, replacements=[
        ("self.cls_token.expand(B, -1, -1).cuda()",
         "self.cls_token.expand(B, -1, -1).to(h.device)"),
    ])
    radam = load_definitions(SOURCE_ID, "MyOptimizer/radam.py", ["RAdam"], {
        "torch": torch, "math": math, "Optimizer": torch.optim.Optimizer,
    })
    lookahead = load_definitions(SOURCE_ID, "MyOptimizer/lookahead.py", ["Lookahead"], {
        "torch": torch, "Optimizer": torch.optim.Optimizer, "defaultdict": defaultdict,
    })
    factory = load_definitions(
        SOURCE_ID, "MyOptimizer/optim_factory.py", ["add_weight_decay", "create_optimizer"], {
            "torch": torch, "optim": torch.optim, "RAdam": radam.RAdam,
            "Lookahead": lookahead.Lookahead, "has_apex": False,
        })
    loss = load_definitions(SOURCE_ID, "MyLoss/loss_factory.py", ["create_loss"], {"nn": torch.nn})
    val = load_definitions(SOURCE_ID, "utils/utils.py", ["cross_entropy_torch"], {
        "torch": torch, "F": torch.nn.functional,
    })
    interface = load_definitions(SOURCE_ID, "models/model_interface.py", ["ModelInterface"], {
        "pl": SimpleNamespace(LightningModule=torch.nn.Module), "torch": torch,
        "create_optimizer": factory.create_optimizer, "create_loss": loss.create_loss,
    })
    return SimpleNamespace(
        Model=getattr(model, classes[-1]), create_optimizer=factory.create_optimizer,
        create_loss=loss.create_loss, Interface=interface.ModelInterface,
        val_loss=val.cross_entropy_torch, model_module=model,
    )


def carrier(author, model, classes):
    # The constructor initializes unrelated old logging/metric imports. Its
    # unchanged step methods only need these explicit training attributes.
    return SimpleNamespace(
        model=model, n_classes=classes,
        loss=author.create_loss(SimpleNamespace(base_loss="CrossEntropyLoss")),
        data=[{"count": 0, "correct": 0} for _ in range(classes)],
    )


def contract():
    root, commit, model_path, config_path = locations()
    files = [
        model_path, config_path, "LICENSE", "train.py", "models/model_interface.py",
        "MyOptimizer/radam.py", "MyOptimizer/lookahead.py", "MyOptimizer/optim_factory.py",
        "MyLoss/loss_factory.py", "utils/utils.py", "datasets/camel_data.py",
        "datasets/data_interface.py",
    ]
    text = {p: pinned_text(root, commit, p) for p in files}
    cfg = yaml.safe_load(text[config_path])
    general = cfg["General"]
    expected = {"epochs": 200, "patience": 10, "precision": 16, "grad_acc": 2, "seed": 2021}
    if any(general[k] != value for k, value in expected.items()):
        raise ValueError("Fourier original training budget changed")
    if cfg["Optimizer"] != {k: RECIPE[k] for k in cfg["Optimizer"]}:
        raise ValueError("Fourier optimizer source configuration changed")
    if (cfg["Data"]["data_shuffle"] is not False
            or cfg["Data"]["train_dataloader"] != {"batch_size": 1, "num_workers": 8}
            or cfg["Data"]["test_dataloader"] != {"batch_size": 1, "num_workers": 8}
            or cfg["Loss"] != {"base_loss": "CrossEntropyLoss"}):
        raise ValueError("Fourier data/loss configuration changed")
    if "MIT License" not in text["LICENSE"]:
        raise ValueError("Fourier fixed-tree license differs")
    return {
        "commit": commit, "model_path": model_path, "config_path": config_path,
        "file_sha256": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in text.items()},
        "original_configuration": cfg, "license": "MIT original fixed-tree text read",
        "original_environment": RECIPE["original_environment"],
        "author_source_redistributed": False,
        "native_optimizer_step_and_model_initialization_unchanged": True,
        "original_Lookahead_loader_incompatible_with_current_torch": True,
        "current_zero_grad_lazily_wraps_step_requiring_empty_Lookahead_hook_registries": True,
        "optimizer_checkpoint_repair": "stable names, fast RAdam, ten-slot cache and Lookahead slow state",
        "current_framework_is_original_environment": False,
    }

"""Private loading of pinned original MR model, optimizer and validator definitions."""

import ast
import hashlib
import os
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from .upstream import load_definitions, package, pinned_text, verified_source


def runtime_identity():
    import nystrom_attention
    import timm
    from importlib.metadata import version
    if timm.__version__ != "0.9.12":
        raise ValueError("MR reference runtime explicitly requires timm0.9.12")
    nystrom_version = version("nystrom-attention")
    if nystrom_version != "0.0.14":
        raise ValueError("MR reference runtime explicitly requires nystrom-attention0.0.14")
    return {"timm": timm.__version__, "nystrom_attention": nystrom_version,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "pin_origin": "explicit benchmark runtime choice; author dependency versions unpinned"}


def definitions():
    runtime_identity()
    root, commit = verified_source("mrblock")
    name = "_milplus_mrblock_native_" + commit
    if not hasattr(sys.modules.get(name + ".models"), "build_model"):
        package(name, root)
        models = package(name + ".models", root / "models")
        for filename in ("mr_block", "abmil", "transmil", "multiheadatt", "dgrmil", "rrtmil", "cate"):
            module = types.ModuleType(name + ".models." + filename)
            module.__file__ = str(root / "models" / (filename + ".py"))
            module.__package__ = name + ".models"
            sys.modules[module.__name__] = module
            setattr(models, filename, module)
            code = pinned_text(root, commit, "models/" + filename + ".py")
            exec(compile(code, module.__file__, "exec"), module.__dict__)
        code = pinned_text(root, commit, "models/__init__.py")
        exec(compile(code, str(root / "models/__init__.py"), "exec"), models.__dict__)
    metrics = load_definitions(
        "mrblock", "metrics.py", ["AccuracyLogger", "calculate_error",
                                  "validation_auc_auprc", "summary_scores"],
        _metric_namespace())
    optimizer = load_definitions(
        "mrblock", "optim.py", ["get_optim", "get_scheduler", "is_plateau_scheduler"],
        {"torch": torch, "optim": torch.optim})
    seed = load_definitions(
        "mrblock", "train.py", ["seed_torch"],
        {"torch": torch, "random": random, "np": np, "os": os,
         "DEVICE": torch.device("cpu")})
    scope = {"torch": torch, "nn": torch.nn, "np": np, "os": os,
             "DEVICE": torch.device("cpu"), "AccuracyLogger": metrics.AccuracyLogger,
             "calculate_error": metrics.calculate_error,
             "validation_auc_auprc": metrics.validation_auc_auprc,
             "is_plateau_scheduler": optimizer.is_plateau_scheduler}
    training = load_definitions(
        "mrblock", "engine.py", ["EarlyStopping", "train_loop", "validate"], scope)
    return SimpleNamespace(models=sys.modules[name + ".models"],
                           training=training, optimizer=optimizer, seed=seed,
                           block=sys.modules[name + ".models.mr_block"].ManifoldResidualBlock)


def _metric_namespace():
    from sklearn.metrics import (accuracy_score, average_precision_score, auc, f1_score,
                                 roc_auc_score, roc_curve)
    from sklearn.preprocessing import label_binarize
    return {"np": np, "accuracy_score": accuracy_score,
            "average_precision_score": average_precision_score, "calc_auc": auc,
            "f1_score": f1_score, "roc_auc_score": roc_auc_score,
            "roc_curve": roc_curve, "label_binarize": label_binarize}


def contract():
    root, commit = verified_source("mrblock")
    paths = ["README.md", "LICENSE", "train.py", "config.py", "data.py", "engine.py",
             "optim.py", "metrics.py", "models/__init__.py", "models/mr_block.py",
             "models/abmil.py", "models/transmil.py", "models/multiheadatt.py",
             "models/dgrmil.py", "models/rrtmil.py", "models/cate.py"]
    texts = {path: pinned_text(root, commit, path) for path in paths}
    tree = ast.parse(texts["train.py"])
    parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "build_parser")
    defaults = {}
    for node in ast.walk(parser):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for kw in node.keywords:
                if kw.arg == "default":
                    defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(kw.value)
    assert defaults["--max_epochs"] == 200 and defaults["--lr"] == 5e-4
    assert defaults["--scheduler"] == "linear" and defaults["--opt"] == "adamw"
    return {
        "commit": commit, "original_CLI_defaults": defaults,
        "files_sha256": {path: hashlib.sha256(text.encode()).hexdigest()
                         for path, text in texts.items()},
        "scheduler": "original LinearLR total_iters=200; called after every optimizer update",
        "merge": "native linear delta merge/GELU active residual preserved literally",
        "selection": "author VAL CE earlystop or explicit last-weight baseline",
        "scope": "all five MR backbones and five corresponding author controls",
    }

"""Execute unmodified necessary definitions from the fixed ASMIL Git objects."""

import ast
from collections import defaultdict, deque
import datetime
import hashlib
import math
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import yaml

from .upstream import load_definitions, pinned_text, verified_source


def definitions():
    network = load_definitions(
        "asmil", "architecture/network.py",
        ["Classifier_1fc", "residual_block", "DimReduction"],
        {"torch": torch, "nn": nn},
    )
    architecture = load_definitions(
        "asmil", "architecture/mytransformer.py",
        ["sigmax", "MutiHeadAttention", "ACMIL_MYMHA", "ACMIL_MYMHA_EMA"],
        {
            "torch": torch, "nn": nn, "Tensor": Tensor, "F": F, "math": math,
            "Classifier_1fc": network.Classifier_1fc, "DimReduction": network.DimReduction,
        },
    )
    return architecture


def author_utilities():
    return load_definitions(
        "asmil", "utils/utils.py",
        ["SmoothedValue", "MetricLogger", "is_dist_avail_and_initialized",
         "adjust_learning_rate"],
        {
            "torch": torch, "dist": torch.distributed, "deque": deque,
            "defaultdict": defaultdict, "time": time, "datetime": datetime, "math": math,
        },
    )


def contract(profile):
    root, commit = verified_source("asmil")
    texts = {
        path: pinned_text(root, commit, path)
        for path in [
            "architecture/network.py", "architecture/mytransformer.py",
            "Step3_WSI_classification_ASMIL.py", "utils/utils.py", profile["author_config"],
        ]
    }
    config = yaml.safe_load(texts[profile["author_config"]])
    expected = {
        "train_epoch": profile["native_recipe"]["epochs"], "B": 1,
        "warmup_epoch": 0, "wd": 1e-5, "lr": 1e-4, "min_lr": 0,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Fixed ASMIL author configuration differs from reviewed recipe")
    tree = ast.parse(texts["Step3_WSI_classification_ASMIL.py"])
    args = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_arguments")
    defaults = {}
    for node in ast.walk(args):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            for kw in node.keywords:
                if kw.arg == "default":
                    defaults[node.args[0].value] = ast.literal_eval(kw.value)
    if any(defaults.get(k) != v for k, v in {
        "--n_token": 1, "--n_drop": 0, "--lr": 1e-4, "--temp": 1.,
        "--seed": 1, "--pretrain": "medical_ssl",
    }.items()):
        raise ValueError("Fixed ASMIL author CLI defaults changed")
    return {
        "commit": commit, "config": expected, "cli_defaults": defaults,
        "files_sha256": {
            k: hashlib.sha256(v.encode()).hexdigest() for k, v in texts.items()
        },
        "architecture_definition_replacements": [],
        "excluded_imports": [
            "unused NystromAttention, emb_position, einops and DimReduction1",
            "wandb/timm/slide HDF5 trainer imports not needed by architecture",
        ],
        "teacher_initialization": "independent sequential constructor; no online copy",
        "teacher_train_mode": True,
        "literal_teacher_condition": "torch.isnan(msd[k]).any(), BEFORE optimizer step",
        "metric_runtime_reference": profile["metric_runtime"],
        "repairs": profile["repairs"],
    }

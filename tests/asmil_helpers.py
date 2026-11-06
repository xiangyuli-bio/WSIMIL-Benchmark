"""Independent literal author module execution and synthetic native-width bags."""

import ast
from collections import defaultdict, deque
import datetime
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torchmetrics
from torch import Tensor, nn
from torch.nn import functional as F

from milplus.upstream import pinned_text, verified_source
from pneamil_helpers import equal_tree, make_task as make_pnea_task


def literal_source(*, repair_teacher=False):
    """Compile complete model files without unused imports; never call port loaders."""
    root, sha = verified_source("asmil")
    network = {"torch": torch, "nn": nn}
    exec(compile(pinned_text(root, sha, "architecture/network.py"),
                 str(root / "architecture/network.py"), "exec"), network)
    ns = {
        "torch": torch, "nn": nn, "F": F, "Tensor": Tensor, "math": math,
        **{name: network[name] for name in ["Classifier_1fc", "DimReduction", "DimReduction1"]},
    }
    tree = ast.parse(pinned_text(root, sha, "architecture/mytransformer.py"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(root / "architecture/mytransformer.py"), "exec"), ns)
    utils = {
        "torch": torch, "dist": torch.distributed, "defaultdict": defaultdict,
        "deque": deque, "datetime": datetime, "time": time, "math": math,
    }
    tree = ast.parse(pinned_text(root, sha, "utils/utils.py"))
    names = {"SmoothedValue", "MetricLogger", "is_dist_avail_and_initialized", "adjust_learning_rate"}
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in names]
    exec(compile(tree, str(root / "utils/utils.py"), "exec"), utils)

    def accuracy(output, target, topk=(1,)):
        assert topk == (1,)
        # Only the unavailable timm logging helper is replaced, never a selector metric.
        return [(output.argmax(1) == target).float().sum() * (100. / len(target))]

    training = {
        "torch": torch, "nn": nn, "F": F, "torchmetrics": torchmetrics,
        "accuracy": accuracy, "sigmax": ns["sigmax"],
        **{n: utils[n] for n in names if n != "is_dist_avail_and_initialized"},
    }
    text = pinned_text(root, sha, "Step3_WSI_classification_ASMIL.py")
    if repair_teacher:
        old = "if torch.isnan(msd[k]).any():"
        assert text.count(old) == 1
        text = text.replace(old, "if torch.isfinite(msd[k]).all():")
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"train_one_epoch", "evaluate"}]
    exec(compile(tree, str(root / "Step3_WSI_classification_ASMIL.py"), "exec"), training)
    return SimpleNamespace(**ns), SimpleNamespace(**training)


def make_task(root, *, dimension=384, classes=2, train_slides=4):
    # Existing fixture retains one1041-patch validation bag: no silent patch cap.
    return make_pnea_task(root, dimension=dimension, classes=classes, train_slides=train_slides)

"""Independent complete CAR modules, original HDF5 bags and literal trainer."""

import ast
from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import time
import types

import h5py
import numpy as np
import torch
from timm.models import model_parameters
from timm.utils import AverageMeter, dispatch_clip_grad

from milplus.data import Bundle
from milplus.upstream import pinned_text, verified_source
from attrimil_helpers import make_task as attri_task
from pneamil_helpers import equal_tree


def module(name, path, text=None):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    if text is None:
        spec.loader.exec_module(value)
    else:
        exec(compile(text, str(path), "exec"), value.__dict__)
    return value


def literal_sources(*, repair_numpy=True, repair_CPU_timing=True):
    root, commit = verified_source("carmil")
    name = "_carmil_independent_full_models"
    package = types.ModuleType(name)
    package.__path__ = [str(root / "src/modules")]
    sys.modules[name] = package
    modules = {}
    for key in ["nystrom_attention", "attmil", "attmil_cf", "dsmil", "transmil"]:
        path = root / f"src/modules/{key}.py"
        modules[key] = module(name + "." + key, path)
    text = pinned_text(root, commit, "src/utils.py")
    if repair_numpy:
        assert text.count("self.val_loss_min = np.Inf") == 1
        text = text.replace("self.val_loss_min = np.Inf", "self.val_loss_min = np.inf")
    utils = module("_carmil_independent_full_utils", root / "src/utils.py", text)
    datasets = module("_carmil_independent_full_datasets", root / "src/dataloader.py")
    text = pinned_text(root, commit, "src/main_car.py")
    if repair_CPU_timing:
        for old, new in [
            ("    torch.cuda.synchronize()\n", ""),
            ("    start_cuda = torch.cuda.Event(enable_timing=True)\n", ""),
            ("    end_cuda = torch.cuda.Event(enable_timing=True)\n", ""),
            ("    start_cuda.record()\n", ""),
            ("    end_cuda.record()\n", ""),
            ("    time_cuda = start_cuda.elapsed_time(end_cuda)\n", "    time_cuda = 0.0\n"),
        ]:
            assert old in text
            text = text.replace(old, new)
    tree = ast.parse(text)
    names = {"CustomLoss_L1", "CustomLoss_COS", "train_loop", "val_loop"}
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
    assert {node.name for node in tree.body} == names
    scope = {
        "torch": torch, "nn": torch.nn, "F": torch.nn.functional, "np": np,
        "time": time, "AverageMeter": AverageMeter, "OrderedDict": OrderedDict,
        "model_parameters": model_parameters, "dispatch_clip_grad": dispatch_clip_grad,
        "five_scores": utils.five_scores, "one_hot": torch.nn.functional.one_hot,
    }
    exec(compile(tree, str(root / "src/main_car.py"), "exec"), scope)
    return types.SimpleNamespace(**modules, utils=utils, datasets=datasets,
                                 trainer=types.SimpleNamespace(**{name: scope[name] for name in names}))


def native_model(author, profile, dimension=1024):
    backbone = profile["native_recipe"]["backbone"]
    if backbone == "attmil":
        return author.attmil_cf.DAttention_CF(dimension, 2, .25, "relu")
    if backbone == "gattmil":
        return author.attmil_cf.AttentionGated_CF(dimension, n_classes=2, dropout=.25)
    if backbone == "transmil":
        return author.transmil.TransMIL_CF(dimension, 2, .25, "relu")
    return author.dsmil.MILNet_CF(2, .25, "relu", input_dim=dimension)


def native_args(profile):
    backbone = profile["native_recipe"]["backbone"]
    return types.SimpleNamespace(
        model=backbone, loss="ce", datasets="camelyon16", cls_alpha=.5 if backbone == "dsmil" else 1.,
        aux_alpha=.5 if backbone == "dsmil" else 1., patch_shuffle=False,
        group_shuffle=False, accumulation_steps=1, clip_grad=0., lr_supi=False, log_iter=100,
    )


def make_task(root, *, dimension=1024):
    task = attri_task(root, dimension=dimension, train_slides=3)
    # A disclosed class signal makes this a software trajectory fixture.
    # It cannot be interpreted as clinical performance or paper reproduction.
    bundle = Bundle(task)
    for row in [row for subset in ("train", "val", "test") for row in bundle.records(subset)]:
        path = task / "features" / (row["slide_id"] + ".pt")
        value = torch.load(path, weights_only=False)
        value[:, :64] += (2 * int(row["target"]) - 1) * 4
        torch.save(value, path)
    return task


def literal_datasets(task, original):
    bundle = Bundle(task)
    root = task / "original_car_h5"
    (root / "h5_files").mkdir(parents=True)
    result = {}
    for subset in ["train", "val"]:
        rows = bundle.records(subset)
        for row in rows:
            x, _ = bundle.load(row)
            with h5py.File(root / "h5_files" / (row["slide_id"] + ".h5"), "w") as handle:
                handle["features"] = x.numpy()
        result[subset] = original.C16Dataset(
            [row["slide_id"] for row in rows],
            ["normal" if int(row["target"]) == 0 else "tumor" for row in rows], str(root))
    return result

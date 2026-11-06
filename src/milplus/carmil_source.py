"""Original CAR-MIL definitions, without copying third-party source into package."""

import ast
import hashlib
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import torch
from torch import nn

from .upstream import load_definitions, load_single, pinned_text, verified_source


def runtime_identity():
    import timm
    if timm.__version__ != "0.9.12":
        raise ValueError("CAR reference runtime explicitly requires timm0.9.12")
    return {"timm": timm.__version__, "torch": str(torch.__version__),
            "numpy": np.__version__,
            "runtime_pin_origin": "explicit benchmark choice; original repository leaves timm unpinned"}


def definitions():
    runtime_identity()
    scope = {"torch": torch, "nn": nn, "F": nn.functional, "np": np}
    attention = load_definitions(
        "carmil", "src/modules/attmil_cf.py",
        ["initialize_weights", "DAttention_CF", "AttentionGated_CF"], scope)
    dsmil = load_definitions(
        "carmil", "src/modules/dsmil.py",
        ["initialize_weights", "BClassifier_CF", "MILNet_CF"], scope)
    nystrom = load_single("carmil", "src/modules/nystrom_attention.py")
    transformer = load_definitions(
        "carmil", "src/modules/transmil.py",
        ["initialize_weights", "TransLayer", "PPEG", "TransMIL_CF"],
        {**scope, "NystromAttention": nystrom.NystromAttention})
    utils = load_definitions(
        "carmil", "src/utils.py", ["seed_torch", "optimal_thresh", "five_scores", "EarlyStopping"],
        {**scope, "roc_curve": roc_curve, "roc_auc_score": roc_auc_score,
         "precision_recall_fscore_support": precision_recall_fscore_support,
         "accuracy_score": accuracy_score},
        replacements=[("self.val_loss_min = np.Inf", "self.val_loss_min = np.inf")])
    # Original val_loop only needs a running average with the native scalar
    # update arithmetic. Its definition is loaded from the installed timm below.
    from timm.utils import AverageMeter
    training = load_definitions(
        "carmil", "src/main_car.py", ["CustomLoss_L1", "CustomLoss_COS", "val_loop"],
        {**scope, "five_scores": utils.five_scores, "AverageMeter": AverageMeter})
    return SimpleNamespace(attention=attention, dsmil=dsmil, transformer=transformer,
                           utils=utils, training=training)


def arguments(profile):
    recipe = profile["native_recipe"]
    return SimpleNamespace(
        model=recipe["backbone"], loss="ce", datasets="camelyon16",
        cls_alpha=recipe["cls_alpha"], aux_alpha=recipe["aux_alpha"],
        patch_shuffle=False, group_shuffle=False, accumulation_steps=1,
        clip_grad=0., lr_supi=False, log_iter=100,
    )


def model_and_loss(author, profile, dimension, device):
    backbone = profile["native_recipe"]["backbone"]
    kwargs = {"input_dim": dimension, "n_classes": 2, "dropout": .25, "act": "relu"}
    if backbone == "attmil":
        model = author.attention.DAttention_CF(**kwargs)
    elif backbone == "gattmil":
        # Original one_fold does not pass act for this branch.
        kwargs.pop("act")
        model = author.attention.AttentionGated_CF(**kwargs)
    elif backbone == "transmil":
        model = author.transformer.TransMIL_CF(**kwargs)
    else:
        model = author.dsmil.MILNet_CF(**kwargs)
    loss = (author.training.CustomLoss_L1 if profile["native_recipe"]["reg_dist"] == "L1"
            else author.training.CustomLoss_COS)("ce", .2, .2)
    return model.to(device), loss


def contract():
    root, commit = verified_source("carmil")
    paths = ["README.md", "src/main_car.py", "src/utils.py", "src/dataloader.py",
             "src/modules/attmil_cf.py", "src/modules/dsmil.py", "src/modules/transmil.py",
             "src/modules/nystrom_attention.py"]
    texts = {path: pinned_text(root, commit, path) for path in paths}
    tree = ast.parse(texts["src/main_car.py"])
    main = next(n for n in tree.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")
    defaults = {}
    for node in main.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "add_argument"):
            call = node.value
            values = {kw.arg: kw.value for kw in call.keywords}
            if "default" in values:
                defaults[ast.literal_eval(call.args[0])] = ast.literal_eval(values["default"])
    assert defaults["--num_epoch"] == 100 and defaults["--reg_dist"] == "L1"
    assert defaults["--model"] == "attmil" and defaults["--input_dim"] == 1024
    assert defaults["--lr"] == 2e-4 and defaults["--opt"] == "adam"
    return {
        "commit": commit, "original_CLI_defaults": defaults,
        "files_sha256": {path: hashlib.sha256(text.encode()).hexdigest() for path, text in texts.items()},
        "selection": "strict pre-TRAIN VAL AUC, initial0; no post-final-training selector",
        "fixed_train_loader_seed": 7784414403328510413,
        "native_transmil_flattened_attention_preserved": True,
        "native_DSMIL_half_bag_half_instance_loss_preserved": True,
        "initialization": "original model seed followed by original post-model training reseed",
        "scope": "original camelyon16 recipe with explicit patient/VAL/locked-TEST adaptation",
        "warm_start": "optional original model mapping, strict=False as author CLI; exact file bound",
    }

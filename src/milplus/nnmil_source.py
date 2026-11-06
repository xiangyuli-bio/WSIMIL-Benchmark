"""Private pinned native nnMIL definitions without redistributing author code."""

from abc import ABC, abstractmethod
import hashlib
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .upstream import load_definitions, load_single, package, pinned_text, verified_source


def runtime_identity():
    if str(torch.__version__).split("+")[0] != "2.5.1" or np.__version__ != "2.2.6":
        raise ValueError("nnMIL reference runtime explicitly requires torch2.5.1/numpy2.2.6")
    try:
        tensorboard = version("tensorboard")
    except PackageNotFoundError:
        tensorboard = None
    return {"torch": str(torch.__version__), "numpy": np.__version__,
            "tensorboard": tensorboard,
            "pin_origin": (
                "same benchmark environment as completed native diagnostic; "
                "author environment lists torch2.3.0/numpy1.26")}


def native_model_factory():
    root, commit = verified_source("nnmil")
    name = "_milplus_nnmil_native_" + commit
    if name + ".model_factory" not in sys.modules:
        package(name, root / "network_architecture")
        package(name + ".models", root / "network_architecture/models")
        for suffix, path in [
            ("models.simple_mil", "network_architecture/models/simple_mil.py"),
            ("models.ab_mil", "network_architecture/models/ab_mil.py"),
            ("model_factory", "network_architecture/model_factory.py"),
        ]:
            spec = importlib.util.spec_from_file_location(name + "." + suffix, root / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module.__name__] = module
            # Execute the pinned Git blob, never an untracked replacement.
            exec(compile(pinned_text(root, commit, path), str(root / path), "exec"), module.__dict__)
    return sys.modules[name + ".model_factory"]


def definitions(dataset_factory):
    runtime_identity()
    model_factory = native_model_factory()
    utilities = load_single("nnmil", "utilities/utils.py")
    masking = load_single("nnmil", "utilities/masking.py")
    samplers = load_single("nnmil", "training/samplers/classification_sampler.py")
    callbacks = load_single("nnmil", "training/callbacks/early_stopping.py")
    planner = load_single("nnmil", "preprocessing/experiment_planner.py")
    loader = load_definitions(
        "nnmil", "utilities/plan_loader.py",
        ["load_plan", "get_config_from_plan", "get_dataset_info_from_plan"],
        {"json": json, "Dict": Dict})
    base = load_definitions(
        "nnmil", "training/trainers/base_trainer.py",
        ["set_random_seeds", "setup_logging", "BaseTrainer"],
        {
            "ABC": ABC, "abstractmethod": abstractmethod, "Optional": Optional,
            "Dict": Dict, "Any": Any, "torch": torch, "np": np, "os": os,
            "sys": sys, "random": random, "warnings": warnings, "logging": logging,
            "json": json, "DataLoader": DataLoader,
            "load_plan": loader.load_plan, "get_config_from_plan": loader.get_config_from_plan,
            "get_dataset_info_from_plan": loader.get_dataset_info_from_plan,
            "create_dataset_from_plan": dataset_factory,
            "is_simple_mil_family": model_factory.is_simple_mil_family,
            "storage_model_type": model_factory.storage_model_type,
            "valid_mask_from_bag_sizes": masking.valid_mask_from_bag_sizes,
        })
    trainer = load_definitions(
        "nnmil", "training/trainers/classification_trainer.py", ["ClassificationTrainer"],
        {
            "BaseTrainer": base.BaseTrainer, "os": os, "sys": sys, "time": time,
            "torch": torch, "nn": torch.nn, "F": torch.nn.functional, "np": np,
            "tqdm": tqdm, "DataLoader": DataLoader,
            "create_mil_model": model_factory.create_mil_model,
            "get_eval_metrics": utilities.get_eval_metrics, "cosine_lr": utilities.cosine_lr,
            "create_dataset_from_plan": dataset_factory,
            "BalancedBatchSampler": samplers.BalancedBatchSampler,
            "AUCBatchSampler": samplers.AUCBatchSampler,
            "EarlyStopping": callbacks.EarlyStopping,
        })
    return SimpleNamespace(
        base=base, trainer=trainer, planner=planner, callbacks=callbacks,
        samplers=samplers, utilities=utilities, masking=masking, models=model_factory)


def contract():
    root, commit = verified_source("nnmil")
    paths = [
        "README.md", "environment.yml", "preprocessing/experiment_planner.py",
        "data/dataset.py", "utilities/plan_loader.py", "utilities/utils.py",
        "utilities/masking.py", "network_architecture/model_factory.py",
        "network_architecture/models/simple_mil.py",
        "training/trainers/base_trainer.py", "training/trainers/classification_trainer.py",
        "training/samplers/classification_sampler.py", "training/callbacks/early_stopping.py",
    ]
    return {
        "commit": commit,
        "files_sha256": {
            path: hashlib.sha256(pinned_text(root, commit, path).encode()).hexdigest() for path in paths},
        "definition_loading": "unchanged named function/class bodies with explicit dependency injection",
        "source_license": "No root LICENSE observed at pinned tree; private loading, no source redistribution",
        "scope": "original classification trainer; survival/regression have separate pending native gates",
        "planner_repair": "statistics and minority proportions use TRAIN only, TEST locked",
    }

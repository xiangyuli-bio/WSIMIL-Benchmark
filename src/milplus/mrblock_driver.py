"""Complete MR recipes with native mode/merge state and exact update recovery."""

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .attrimil_data import RecordingWeightedSampler
from .attrimil_driver import numerical_identity
from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .mrblock_source import contract, definitions, runtime_identity
from .mrblock_spec import PROFILES
from .pneamil_driver import RecordingRandomSampler, RemainingOrder, own_lock
from .provenance import software_identity


BACKEND = "mrblock_complete_author_merge_aware_locked_test"
SPECS = {profile["id"]: profile for profile in PROFILES}


class SlideBags(Dataset):
    def __init__(self, bundle, subset):
        self.bundle, self.rows = bundle, bundle.records(subset)
        self.slide_cls_ids = [
            [i for i, row in enumerate(self.rows) if int(row["target"]) == label]
            for label in range(len(bundle.classes))]

    def __len__(self):
        return len(self.rows)

    def getlabel(self, index):
        return int(self.rows[index]["target"])

    def __getitem__(self, index):
        row = self.rows[index]
        features, _ = self.bundle.load(row)
        return features, int(row["target"])


def collate(batch):
    return torch.cat([item[0] for item in batch]), torch.tensor(
        [item[1] for item in batch], dtype=torch.long)


class MeasuredCE(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.values = []

    def forward(self, logits, target):
        loss = super().forward(logits, target)
        self.values.append(float(loss.detach()))
        return loss


def options(config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("MR preserves the complete200epoch original recipe")
    if "epochs" in config and config["epochs"] != 200:
        raise ValueError("MR maximum200epochs cannot be shortened; use recoverable interruption")
    model = config.get("model", {})
    if not isinstance(model, dict) or set(model) - {"stop_after_train_batches", "weighted_sample"}:
        raise ValueError("MR native widths/rank64/AdamW/scheduler/merge behavior cannot be replaced")
    threads = config.get("threads", 2)
    stop_batch = model.get("stop_after_train_batches")
    weighted = model.get("weighted_sample", False)
    if type(threads) is not int or threads < 1 or type(weighted) is not bool:
        raise ValueError("MR requires positive threads and a Boolean weighted_sample")
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError("MR update interruption must be a positive integer")
    return {"threads": threads, "weighted_sample": weighted}, stop_batch


def concept_asset(bundle, profile):
    if not profile["native_recipe"]["model_type"].endswith("cate"):
        return None
    config = bundle.config.get("mrblock_concepts")
    if not isinstance(config, dict) or set(config) != {"path", "task"}:
        raise ValueError("Native MR-CATE requires mrblock_concepts path/task and aligned real concepts")
    filenames = {"camelyon": "brca_concepts.pt", "nsclc": "nsclc_concepts.pt",
                 "rcc": "rcc_concepts.pt"}
    if config["task"] not in filenames:
        raise ValueError("Native MR-CATE concept task must be camelyon, nsclc or rcc")
    path = Path(config["path"]).expanduser()
    path = (bundle.root / path).resolve() if not path.is_absolute() else path.resolve()
    if path.name != filenames[config["task"]]:
        raise ValueError("Native MR-CATE keeps the author's actual concept filename mapping")
    if bundle.dim != 512:
        raise ValueError("Native MR-CATE512 internal concept/projection widths require aligned512D input")
    values = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(values, (list, tuple)) or len(values) < 2
            or any(not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] < 1
                   or value.shape[1] != 512 or not torch.isfinite(value).all()
                   for value in values)):
        raise ValueError("Native MR-CATE concept checkpoint must contain finite aligned512D concept matrices")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task": config["task"], "shapes": [list(value.shape) for value in values],
            "alignment": "caller-provided original embedding alignment; not inferred by reshaping"}


def check_bundle(bundle):
    if (bundle.task_type != "classification" or len(bundle.classes) < 2
            or bundle.config.get("training_unit", "slide") != "slide"):
        raise ValueError("MR original recipes require individual-slide classification")
    inventory = bundle.check(deep=True)
    for subset in ("train", "val"):
        if {int(row["target"]) for row in bundle.records(subset)} != set(range(len(bundle.classes))):
            raise ValueError("Native MR TRAIN/VAL requires all declared classes for original validation metrics")
    return inventory


def make_components(bundle, profile, author, asset, device):
    args = SimpleNamespace(
        opt="adamw", lr=5e-4, reg=1e-5, scheduler="linear", max_epochs=200,
        task=asset["task"] if asset else "task_1_tcga_nsclc",
        concept_root=str(Path(asset["path"]).parent) if asset else None)
    model = author.models.build_model(
        profile["native_recipe"]["model_type"], embed_dim=bundle.dim,
        n_classes=len(bundle.classes), dropout=.25, size_arg="small", args=args).to(device)
    optimizer = author.optimizer.get_optim(model, args)
    scheduler = author.optimizer.get_scheduler(optimizer, args)
    early = (author.training.EarlyStopping(patience=20, stop_epoch=50, verbose=True)
             if profile["native_recipe"]["early_stopping"] else None)
    return model, optimizer, scheduler, early


def mode_state(model, author):
    return {
        "training": {name: module.training for name, module in model.named_modules()},
        "merged": {name: module.merged for name, module in model.named_modules()
                   if isinstance(module, author.block)},
    }


def restore_mode_state(model, author, saved):
    modules = dict(model.named_modules())
    assert set(saved["training"]) == set(modules)
    assert set(saved["merged"]) == {name for name, module in modules.items()
                                   if isinstance(module, author.block)}
    # Direct flags are essential: train()/eval() would merge/unmerge again.
    for name, value in saved["training"].items():
        modules[name].training = value
    for name, value in saved["merged"].items():
        modules[name].merged = value


def snapshot(model, optimizer, scheduler, early, progress, request, author):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "early_stopping": vars(early).copy() if early else None,
        "mode_state": mode_state(model, author), "rng": rng_state(), "progress": progress,
    }


def restore(state, model, optimizer, scheduler, early, request, author):
    from .engine import restore_rng
    if state["request_sha256"] != digest_object(request):
        raise ValueError("MR checkpoint request changed")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if (early is None) != (state["early_stopping"] is None):
        raise ValueError("MR original early-stopping configuration changed")
    if early is not None:
        vars(early).update(state["early_stopping"])
    restore_mode_state(model, author, state["mode_state"])
    restore_rng(state["rng"])
    return state["progress"]


def train_step(model, optimizer, scheduler, batch, device):
    features, target = (value.to(device) for value in batch)
    logits, _, _, _, _ = model(features)
    loss = nn.functional.cross_entropy(logits, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Original MR CE became nonfinite")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Original MR gradient became nonfinite")
    used_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()  # literal original scheduling is per optimizer update
    optimizer.zero_grad()
    return {"loss": loss.item(), "lr_used": used_lr, "lr_next": optimizer.param_groups[0]["lr"],
            "patch_count": len(features)}


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import initialize_cpu_vector_math, restore_rng, rng_state
    initialize_cpu_vector_math()
    if method not in SPECS:
        raise ValueError("Choose an explicit complete MR author profile")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("MR requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("MR epoch interruption must be a positive integer")
    cfg, stop_batch = options(config)
    profile, dev = SPECS[method], torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("Native MR supports CPU/CUDA")
    bundle = Bundle(data, fm, split)
    inventory = check_bundle(bundle)
    asset = concept_asset(bundle, profile)
    torch.set_num_threads(cfg["threads"])
    author = definitions()
    author.seed.DEVICE = author.training.DEVICE = dev
    author.seed.seed_torch(seed)
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg, "method_spec": profile,
        "software": software_identity(), "numerical_identity": numerical_identity(),
        "runtime": runtime_identity(), "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"],
        "source_contract": contract(), "concept_asset": asset,
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    out = Path(output).expanduser().resolve()
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("MR recovery changed inputs/software/numerics/source/concepts/recipe")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("MR run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        author.seed.seed_torch(seed)
        model, optimizer, scheduler, early = make_components(bundle, profile, author, asset, dev)
        train_data, val_data = SlideBags(bundle, "train"), SlideBags(bundle, "val")
        weights = [len(train_data) / len(train_data.slide_cls_ids[row["target"]])
                   for row in train_data.rows] if cfg["weighted_sample"] else None
        progress = {"epoch": 0, "cursor": 0, "order": None, "updates": 0,
                    "history": [], "train_trace": [], "best_loss": None, "author_stopped": False}
        write_json(out / "method_protocol.json", {
            "method": method, "recipe": profile["native_recipe"], "source": request["source_contract"],
            "input_dimension": bundle.dim, "rank": 64, "hidden_dimension": 512,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "weighted_sample": cfg["weighted_sample"], "sampling_weights": weights,
            "concept_asset": asset, "native_mode_and_merge_flags_checkpointed": True,
            "original_GELU_active_and_linear_merge_difference_preserved": True,
            "selection": "author minimum VAL CE with ties" if early else "author last weights",
            "adaptations": profile["adaptations"],
        })
        if resume:
            progress = restore(torch.load(out / "last.pt", map_location=dev, weights_only=False),
                               model, optimizer, scheduler, early, request, author)
        while progress["epoch"] < 200 and not progress["author_stopped"]:
            if progress["cursor"] < len(train_data):
                recovering = progress["order"] is not None
                sampler = (
                    RemainingOrder(progress["order"][progress["cursor"]:]) if recovering
                    else RecordingWeightedSampler(torch.tensor(weights, dtype=torch.double), len(weights))
                    if weights is not None else RecordingRandomSampler(train_data))
                loader = DataLoader(train_data, batch_size=1, sampler=sampler,
                                    collate_fn=collate, num_workers=0)
                before = rng_state() if recovering else None
                # Match original train_loop: mode transition happens before
                # DataLoader iterator consumes its base seed.
                model.train()
                iterator = iter(loader)
                if recovering:
                    restore_rng(before)
                for batch in iterator:
                    values = train_step(model, optimizer, scheduler, batch, dev)
                    if progress["order"] is None:
                        progress["order"] = list(sampler.order)
                    index = progress["order"][progress["cursor"]]
                    progress["train_trace"].append({
                        "slide_id": train_data.rows[index]["slide_id"], **values})
                    progress["cursor"] += 1
                    progress["updates"] += 1
                    atomic_torch_save(snapshot(
                        model, optimizer, scheduler, early, progress, request, author), out / "last.pt")
                    if stop_batch is not None and progress["updates"] >= stop_batch:
                        write_json(out / "status.json", {
                            "state": "interrupted", "backend": BACKEND,
                            "completed_train_batches": progress["updates"],
                            "reason": "requested actual update boundary"})
                        return out
            criterion = MeasuredCE()
            old_best = early.best_score if early else None
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                                    collate_fn=collate, num_workers=0)
            # Native validate also writes the author's raw selected state dict;
            # keep it as evidence, while best.pt adds complete recoverable state.
            stopped = author.training.validate(
                0, progress["epoch"], model, val_loader, len(bundle.classes), early,
                criterion, str(out), scheduler)
            val_loss = sum(criterion.values) / len(criterion.values)
            if not math.isfinite(val_loss):
                raise FloatingPointError("Original MR validation CE became nonfinite")
            improved = bool(early is not None and (old_best is None or -val_loss >= old_best))
            progress["best_loss"] = early.val_loss_min if early else None
            progress["author_stopped"] = bool(stopped)
            progress["history"].append({
                "epoch": progress["epoch"] + 1, "val_loss": val_loss,
                "improved": improved, "best_loss": progress["best_loss"],
                "retain": early.counter if early else None, "train_order": progress["order"],
                "train_trace": progress["train_trace"], "total_updates": progress["updates"],
                "scheduler_steps": scheduler.last_epoch, "lr_next": optimizer.param_groups[0]["lr"],
            })
            progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None, train_trace=[])
            state = snapshot(model, optimizer, scheduler, early, progress, request, author)
            if improved or early is None and progress["epoch"] == 200:
                atomic_torch_save(copy.deepcopy(state), out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"epoch={progress['epoch']} val_ce={val_loss:.8f} lr={optimizer.param_groups[0]['lr']:.8g}",
                  flush=True)
            if (stop_after is not None and progress["epoch"] >= stop_after
                    and progress["epoch"] < 200 and not stopped):
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND, "completed_epochs": progress["epoch"],
                    "reason": "requested epoch boundary"})
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("Native MR did not create selected weights")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND, "completed_epochs": progress["epoch"],
            "maximum_epoch_budget": 200, "completed_train_batches": progress["updates"],
            "termination": "author_early_stop" if progress["author_stopped"] else "complete_author_epoch_budget",
            "test_evaluated": False,
        })
        return out
    except BaseException as error:
        write_json(out / "status.json", {
            "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "backend": BACKEND, "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def evaluate(run, subset="test", device="cpu"):
    from .engine import initialize_cpu_vector_math, score
    initialize_cpu_vector_math()
    out = Path(run).expanduser().resolve()
    request = read_json(out / "request.json")
    if subset not in {"val", "test"} or read_json(out / "status.json")["state"] != "completed":
        raise ValueError("MR selected evaluation requires complete training and val/test subset")
    dev = torch.device(device)
    torch.set_num_threads(request["config"]["threads"])
    author = definitions()
    author.seed.DEVICE = author.training.DEVICE = dev
    author.seed.seed_torch(request["seed"])
    if (request["software"] != software_identity() or request["device"] != str(dev)
            or request["numerical_identity"] != numerical_identity()
            or request["runtime"] != runtime_identity()):
        raise ValueError("MR selected evaluation software/device/numerics changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = check_bundle(bundle)
    profile = SPECS[request["method"]]
    asset = concept_asset(bundle, profile)
    if (request["bundle_sha256"] != bundle.fingerprint
            or inventory["feature_inventory_sha256"] != request["features_sha256"]
            or request["source_contract"] != contract() or request["concept_asset"] != asset):
        raise ValueError("MR selected evaluation changed input features/source/concepts")
    model, optimizer, scheduler, early = make_components(bundle, profile, author, asset, dev)
    restore(torch.load(out / "best.pt", map_location=dev, weights_only=False),
            model, optimizer, scheduler, early, request, author)
    model.eval()  # flags were restored; native MR does not merge twice
    records = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            probabilities = model(features.to(dev))[1].flatten().cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("MR selected probabilities became nonfinite")
            records.append({
                "slide_id": row["slide_id"], "patient_id": row["patient_id"],
                "global_patient_id": row["global_patient_id"], "subset": subset,
                "source_slides": row["slide_id"], "target": int(row["target"]),
                **{f"p_{i}": float(p) for i, p in enumerate(probabilities)},
            })
    frame = pd.DataFrame(records)
    metrics = score(bundle, frame)
    frame.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", metrics)
    return metrics

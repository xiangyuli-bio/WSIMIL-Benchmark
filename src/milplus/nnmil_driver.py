"""Resumable native nnMIL classification with original worker4 input replay."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attrimil_driver import numerical_identity
from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .nnmil_data import SlideBags, train_only_plan
from .nnmil_source import contract, definitions, runtime_identity
from .nnmil_spec import PROFILES
from .pneamil_driver import own_lock
from .provenance import software_identity

BACKEND = "nnmil_complete_author_classification_TRAIN_only_planner_repaired"
SPECS = {profile["id"]: profile for profile in PROFILES}


def options(config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("nnMIL preserves the complete native100epoch adaptive recipe")
    if "epochs" in config and config["epochs"] != 100:
        raise ValueError("nnMIL maximum100epoch budget cannot be shortened")
    model = config.get("model", {})
    if not isinstance(model, dict) or set(model) - {"stop_after_train_batches"}:
        raise ValueError("nnMIL native widths/batch/sampler/AMP/optimizer cannot be replaced")
    threads, stop_batch = config.get("threads", 2), model.get("stop_after_train_batches")
    if type(threads) is not int or threads < 1:
        raise ValueError("nnMIL threads must be a positive integer")
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError("nnMIL update interruption must be a positive integer")
    return {"threads": threads}, stop_batch


def loader_state(loader):
    sampler = loader.batch_sampler
    value = {
        "kind": type(sampler).__name__,
        "state": copy.deepcopy({
            key: item for key, item in vars(sampler).items()
            if key not in {"dataset", "sampler", "generator", "_rng"}}),
        "generator": loader.generator.get_state() if loader.generator is not None else None,
    }
    if hasattr(sampler, "_rng"):
        value["sampler_rng"] = copy.deepcopy(sampler._rng.get_state())
    if hasattr(sampler, "sampler"):
        value["sub_sampler_state"] = copy.deepcopy({
            key: item for key, item in vars(sampler.sampler).items()
            if key not in {"data_source", "generator"}})
    return value


def restore_loader(loader, value):
    sampler = loader.batch_sampler
    if type(sampler).__name__ != value["kind"]:
        raise ValueError("nnMIL native sampler type changed")
    vars(sampler).update(copy.deepcopy(value["state"]))
    if "sampler_rng" in value:
        sampler._rng.set_state(value["sampler_rng"])
    if "sub_sampler_state" in value:
        vars(sampler.sampler).update(copy.deepcopy(value["sub_sampler_state"]))
    if (loader.generator is None) != (value["generator"] is None):
        raise ValueError("nnMIL native loader generator changed")
    if loader.generator is not None:
        loader.generator.set_state(value["generator"])


def early_state(early):
    return copy.deepcopy({key: value for key, value in vars(early).items()
                          if key not in {"logger", "save_dir", "model_type", "best_model_state"}})


def make_components(bundle, profile, plan, out, device, seed):
    datasets = {subset: SlideBags(bundle, subset, plan["training_configuration"])
                for subset in ("train", "val", "test")}

    def dataset_factory(plan_path, split, fold=None):
        return datasets[split]

    author = definitions(dataset_factory)
    native = out / "native"
    native.mkdir(exist_ok=True)
    logfile = native / "nnmil_training.log"
    if logfile.exists():
        archive = native / "prior_initialization_logs"
        archive.mkdir(exist_ok=True)
        number = len(list(archive.glob("*.log")))
        logfile.rename(archive / f"initialization_{number}.log")
    trainer = author.trainer.ClassificationTrainer(
        str(out / "plan.json"), "nnmil", save_dir=str(native), seed=seed)
    trainer.device = device  # explicit caller device; native constructor auto-selects available CUDA
    torch.set_grad_enabled(True)
    trainer.create_model()
    trainer.model.train()
    for parameter in trainer.model.parameters():
        parameter.requires_grad = True
    trainer.create_data_loaders()
    if len(trainer.train_loader) < 1:
        raise ValueError("nnMIL original sampler cannot form a TRAIN batch")
    trainer.save_training_config()
    config = plan["training_configuration"]
    named = list(trainer.model.named_parameters())

    def exclude(name, parameter):
        return (parameter.ndim < 2 or "bn" in name or "ln" in name
                or "bias" in name or "logit_scale" in name)

    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in named if exclude(name, p) and p.requires_grad],
         "weight_decay": 0.},
        {"params": [p for name, p in named if not exclude(name, p) and p.requires_grad],
         "weight_decay": config["weight_decay"]},
    ], lr=config["learning_rate"])
    scheduler = author.utilities.cosine_lr(
        optimizer, config["learning_rate"], len(trainer.train_loader) * config["warmup_epochs"],
        len(trainer.train_loader) * config["num_epochs"])
    early = author.callbacks.EarlyStopping(
        patience=config["patience"], verbose=True, metric=profile["native_recipe"]["metric"],
        save_dir=str(native), model_type="nnmil", logger=trainer.logger)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    return trainer, optimizer, scheduler, early, scaler, author


def checkpoint(trainer, optimizer, early, scaler, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": trainer.model.state_dict(), "model_training": trainer.model.training,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None,
        "early_stopping": early_state(early), "progress": progress,
        "loader": loader_state(trainer.train_loader), "rng": rng_state(),
    }


def restore(state, trainer, optimizer, early, scaler, request):
    from .engine import restore_rng
    if state["request_sha256"] != digest_object(request):
        raise ValueError("nnMIL checkpoint request changed")
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.model.train(state["model_training"])
    optimizer.load_state_dict(state["optimizer"])
    if (scaler is None) != (state["scaler"] is None):
        raise ValueError("nnMIL native AMP scaler changed")
    if scaler:
        scaler.load_state_dict(state["scaler"])
    vars(early).update(state["early_stopping"])
    # Preserve the original shallow in-memory fallback alias. Selection itself
    # uses immutable serialized best.pt, as original train loads best_nnmil.pth.
    early.best_model_state = trainer.model.state_dict().copy()
    restore_loader(trainer.train_loader, state["loader"])
    restore_rng(state["rng"])
    return state["progress"]


def step(trainer, optimizer, scheduler, scaler, batch, global_step):
    features, coords, bag_sizes, targets = [value.to(trainer.device) for value in batch[:4]]
    optimizer.zero_grad()
    lr_used = [group["lr"] for group in optimizer.param_groups]
    with torch.amp.autocast(trainer.device.type, dtype=torch.bfloat16):
        logits = trainer.forward_model(features, bag_sizes)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Native nnMIL CE became nonfinite")
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in trainer.model.parameters()):
            raise FloatingPointError("Native nnMIL gradient became nonfinite")
        optimizer.step()
    scheduler(global_step)  # original step0 schedule is AFTER the first update
    return {
        "slide_ids": list(batch[4]), "bag_sizes": bag_sizes.cpu().tolist(),
        "features_sha256": hashlib.sha256(features.detach().cpu().numpy().tobytes()).hexdigest(),
        "sampled_channels": trainer.model.last_train_idx.detach().cpu().tolist(),
        "loss": loss.item(), "logits": logits.detach().float().cpu().tolist(),
        "lr_used": lr_used, "lr_next": [group["lr"] for group in optimizer.param_groups],
    }


def validate(trainer, early):
    trainer.model.eval()
    torch.set_grad_enabled(False)
    metrics = trainer.evaluate("val")
    torch.set_grad_enabled(True)
    trainer.model.train()
    values = [
        metrics.get("val_val/loss", metrics.get("val/loss", 0.)),
        metrics.get("val_val/bacc", metrics.get("val/bacc", 0.)),
        metrics.get("val_val/weighted_f1", metrics.get("val/weighted_f1", 0.)),
        metrics.get("val_val/auroc", metrics.get("val/auroc", 0.)),
    ]
    before = early.best_score
    early(*values, trainer.model,
          val_kappa=metrics.get("val_val/kappa", metrics.get("val/kappa")))
    improved = before is None or early.best_score > before
    return metrics, improved


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import initialize_cpu_vector_math, restore_rng, rng_state
    initialize_cpu_vector_math()
    if method not in SPECS:
        raise ValueError("Choose an explicit complete nnMIL classification profile")
    cfg, stop_batch = options(config)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("nnMIL requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("nnMIL epoch interruption must be a positive integer")
    bundle = Bundle(data, fm, split)
    if (bundle.task_type not in {"classification", "ordinal"}
            or bundle.config.get("training_unit", "slide") != "slide"):
        raise ValueError("Native nnMIL classification requires individual-slide classification/ordinal bags")
    inventory = bundle.check(deep=True)
    torch.set_num_threads(cfg["threads"])
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("nnMIL native autocast supports CPU/CUDA")
    profile = SPECS[method]
    plan = train_only_plan(bundle, profile["native_recipe"]["metric"],
                           definitions(lambda *args, **kwargs: None), seed)
    # Native BaseTrainer sets deterministic flags as well as RNG seeds.
    # Record the numerical environment after that initialization so a first
    # run and a fresh-process recovery bind to the same actual native state.
    definitions(lambda *args, **kwargs: None).base.set_random_seeds(seed)
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg, "method_spec": profile,
        "software": software_identity(), "numerical_identity": numerical_identity(),
        "runtime": runtime_identity(), "source_contract": contract(), "plan": plan,
        "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"],
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    out = Path(output).expanduser().resolve()
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("nnMIL recovery changed inputs/software/numerics/plan/native recipe")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("nnMIL run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
        write_json(out / "plan.json", plan)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    trainer = iterator = None
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        trainer, optimizer, scheduler, early, scaler, author = make_components(
            bundle, profile, plan, out, dev, seed)
        progress = {
            "epoch": 0, "cursor": 0, "updates": 0, "history": [], "train_trace": [],
            "epoch_start_rng": None, "epoch_start_loader": None,
        }
        write_json(out / "method_protocol.json", {
            "method": method, "plan": plan, "source": request["source_contract"],
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "native_workers": trainer.train_loader.num_workers,
            "native_train_batches": len(trainer.train_loader),
            "TEST_locked": True, "deterministic_per_index_sampling_seed": 42,
            "recovery": "replay native input iterator only; no optimizer/model update replay",
            "adaptations": profile["adaptations"],
        })
        saved = None
        if resume:
            saved = torch.load(out / "last.pt", map_location=dev, weights_only=False)
            progress = restore(saved, trainer, optimizer, early, scaler, request)
        while progress["epoch"] < 100 and not early.early_stop:
            torch.set_grad_enabled(True)
            trainer.model.train()
            recovering = progress["epoch_start_rng"] is not None
            if recovering:
                current_rng = rng_state()
                restore_loader(trainer.train_loader, progress["epoch_start_loader"])
                restore_rng(progress["epoch_start_rng"])
            else:
                progress["epoch_start_rng"] = rng_state()
                progress["epoch_start_loader"] = loader_state(trainer.train_loader)
            iterator = iter(trainer.train_loader)
            if recovering:
                # Native workers use deterministic per-index42 patch selection.
                # Replaying data iteration restores sampler pools/prefetch and
                # the dedicated generator; no forward/backward/update occurs.
                for _ in range(progress["cursor"]):
                    next(iterator)
                restore_rng(current_rng)
            for batch in iterator:
                trace = step(trainer, optimizer, scheduler, scaler, batch, progress["updates"])
                progress["train_trace"].append(trace)
                progress["cursor"] += 1
                progress["updates"] += 1
                atomic_torch_save(checkpoint(
                    trainer, optimizer, early, scaler, progress, request), out / "last.pt")
                if stop_batch is not None and progress["updates"] >= stop_batch:
                    write_json(out / "status.json", {
                        "state": "interrupted", "backend": BACKEND,
                        "completed_train_batches": progress["updates"],
                        "reason": "requested actual update boundary"})
                    return out
            iterator = None
            metrics, improved = validate(trainer, early)
            progress["history"].append({
                "epoch": progress["epoch"] + 1, "metrics": metrics, "improved": improved,
                "early_stopping": early_state(early), "train_trace": progress["train_trace"],
                "total_updates": progress["updates"],
                "lr_next": [group["lr"] for group in optimizer.param_groups],
            })
            progress.update(epoch=progress["epoch"] + 1, cursor=0, train_trace=[],
                            epoch_start_rng=None, epoch_start_loader=None)
            state = checkpoint(trainer, optimizer, early, scaler, progress, request)
            if improved:
                atomic_torch_save(copy.deepcopy(state), out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"epoch={progress['epoch']} updates={progress['updates']} "
                  f"best={early.best_score} retain={early.counter}", flush=True)
            if (stop_after is not None and progress["epoch"] >= stop_after
                    and progress["epoch"] < 100 and not early.early_stop):
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND,
                    "completed_epochs": progress["epoch"], "reason": "requested epoch boundary"})
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("Native nnMIL did not create selected weights")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND,
            "completed_epochs": progress["epoch"], "maximum_epoch_budget": 100,
            "completed_train_batches": progress["updates"],
            "termination": "author_early_stop" if early.early_stop else "complete_author_epoch_budget",
            "test_evaluated": False,
        })
        return out
    except BaseException as error:
        write_json(out / "status.json", {
            "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "backend": BACKEND, "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        if trainer is not None and trainer.writer is not None:
            trainer.writer.close()
        lock.unlink(missing_ok=True)


def evaluate(run, subset="test", device="cpu"):
    from .engine import initialize_cpu_vector_math, score
    initialize_cpu_vector_math()
    out = Path(run).expanduser().resolve()
    request = read_json(out / "request.json")
    if subset not in {"val", "test"} or read_json(out / "status.json")["state"] != "completed":
        raise ValueError("nnMIL selected evaluation requires completed training and val/test subset")
    dev = torch.device(device)
    torch.set_num_threads(request["config"]["threads"])
    author = definitions(lambda *args, **kwargs: None)
    author.base.set_random_seeds(request["seed"])
    if (request["software"] != software_identity() or request["runtime"] != runtime_identity()
            or request["numerical_identity"] != numerical_identity() or request["device"] != str(dev)
            or request["source_contract"] != contract()):
        raise ValueError("nnMIL selected evaluation software/runtime/device/source changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = bundle.check(deep=True)
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]):
        raise ValueError("nnMIL selected evaluation inputs changed")
    plan = request["plan"]["training_configuration"]
    model = author.models.create_mil_model(
        "nnmil", input_dim=bundle.dim, hidden_dim=plan["hidden_dim"],
        num_classes=len(bundle.classes), dropout=plan["dropout"]).to(dev)
    state = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    records = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            logits = model(features.to(dev).unsqueeze(0))["logits"]
            probabilities = torch.softmax(logits, dim=1).flatten().cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Native nnMIL probabilities became nonfinite")
            records.append({
                "slide_id": row["slide_id"], "patient_id": row["patient_id"],
                "global_patient_id": row["global_patient_id"], "subset": subset,
                "source_slides": row["slide_id"], "target": int(row["target"]),
                **{f"p_{i}": float(value) for i, value in enumerate(probabilities)},
            })
    frame = pd.DataFrame(records)
    metrics = score(bundle, frame)
    frame.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", metrics)
    return metrics

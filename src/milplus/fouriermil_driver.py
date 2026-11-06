"""Complete original Fourier recipe with explicit current-framework recovery."""

from contextlib import nullcontext
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import Bundle
from .fcmil_driver import SlideBags, collate
from .fouriermil_source import carrier, contract, definitions
from .fouriermil_spec import BACKEND, PROFILES
from .fouriermil_state import (
    gradients, initialize_lookahead_hooks, optimizer_snapshot, restore_gradients, restore_optimizer,
)
from .io import atomic_torch_save, digest_object, read_json, write_json
from .pneamil_driver import RecordingRandomSampler, RemainingOrder, own_lock
from .provenance import software_identity

SPECS = {p["id"]: p for p in PROFILES}


def initialize_runtime():
    from .numerics import initialize_cpu_vector_math
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    initialize_cpu_vector_math()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numerical_identity(amp):
    from .attrimil_driver import numerical_identity as base_identity
    return {**base_identity(), "amp": amp, "autocast_dtype": "torch.float16" if amp else None,
            "gradient_accumulation": 2, "loss_normalization": "every minibatch /2 including last partial",
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "modern_framework_compatibility": True}


def options(config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("Fourier author recipe accepts threads/interruption options only")
    if "epochs" in config and (type(config["epochs"]) is not int or config["epochs"] != 200):
        raise ValueError("Fourier author maximum epoch budget is200")
    settings = config.get("model", {})
    if not isinstance(settings, dict) or set(settings) - {"stop_after_train_batches"}:
        raise ValueError("Fourier native dimensions, optimizer, AMP and eight workers are fixed")
    threads, stop = config.get("threads", 2), settings.get("stop_after_train_batches")
    if type(threads) is not int or threads < 1:
        raise ValueError("Fourier threads requires positive integer")
    if stop is not None and (type(stop) is not int or stop < 1):
        raise ValueError("Fourier batch interruption requires positive integer")
    return {"threads": threads}, stop


def check_bundle(bundle):
    if (bundle.task_type != "classification" or bundle.dim != 1024
            or bundle.config.get("training_unit", "slide") != "slide"):
        raise ValueError("Original Fourier requires individual-slide classification and1024D features")
    return bundle.check(deep=True)


def policy(method, device):
    if method not in SPECS:
        raise ValueError("Select an explicit original Fourier training profile")
    amp = SPECS[method]["author_recipe"]["precision"] == 16
    dev = torch.device(device)
    if amp and (dev.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("Original Fourier FP16 profile requires actual CUDA; use explicit CPU32 reference")
    if not amp and dev.type != "cpu":
        raise ValueError("Fourier CPU32 reference requires CPU")
    if amp and dev.index not in (None, 0):
        raise ValueError("Original Fourier single-GPU profile uses visible cuda:0")
    return dev, amp


def autocast(amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()


def components(classes, device, recipe):
    author = definitions()
    model = author.Model(n_classes=classes).to(device)
    interface = carrier(author, model, classes)
    optimizer = author.create_optimizer(SimpleNamespace(**recipe), model)
    initialize_lookahead_hooks(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=recipe["precision"] == 16)
    return model, interface, optimizer, scaler, author


def minibatch(model, interface, optimizer, scaler, author, batch, index, total, device, amp):
    features, labels = [x.to(device) for x in batch]
    if index % 2 == 0:
        # Match torch1.7 default; affects parameters with missing gradients.
        optimizer.zero_grad(set_to_none=False)
    with autocast(amp):
        raw = author.Interface.training_step(interface, (features, labels), index)["loss"]
        loss = raw / 2  # PL1.2.3 also divides the final singleton group by2.
    if not torch.isfinite(raw):
        raise FloatingPointError("Original Fourier training loss became nonfinite")
    scaler.scale(loss).backward()
    boundary = (index + 1) % 2 == 0 or index + 1 == total
    scale_before, skipped = scaler.get_scale(), False
    if boundary:
        if amp:
            scaler.unscale_(optimizer)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Fourier CPU reference gradient became nonfinite")
        # Infinite scaled gradients are handled by GradScaler, not an early rejection.
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
    return {
        "loss": float(raw.detach()), "normalized_loss": float(loss.detach()),
        "target": labels.detach().cpu().tolist(), "patch_count": features.shape[1],
        "accumulation_index": index % 2, "optimizer_boundary": boundary,
        "optimizer_step_skipped": skipped,
        "grad_scale_before": scale_before, "grad_scale_after": scaler.get_scale(),
    }


def validation(model, interface, author, bundle, device, amp):
    model.eval()
    interface.data = [{"count": 0, "correct": 0} for _ in bundle.classes]
    loader = DataLoader(SlideBags(bundle, "val"), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=collate)
    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = [x.to(device) for x in batch]
            with autocast(amp):
                outputs.append(author.Interface.validation_step(interface, batch, index))
        # Native validation_epoch_end executes outside forward autocast.
        loss = author.val_loss(torch.cat([o["logits"] for o in outputs], dim=0),
                               torch.stack([o["label"] for o in outputs], dim=0))
    if not torch.isfinite(loss):
        raise FloatingPointError("Original Fourier literal VAL log(softmax) loss became nonfinite")
    return {
        "loss": float(loss), "literal_loss_dtype": str(loss.dtype),
        "forward_logits_dtype": str(outputs[0]["logits"].dtype),
        "class_statistics": copy.deepcopy(interface.data),
        "probabilities": torch.cat([o["Y_prob"] for o in outputs]).cpu().tolist(),
        "targets": torch.cat([o["label"] for o in outputs]).cpu().tolist(),
    }


def snapshot(model, interface, optimizer, scaler, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": model.state_dict(), "model_training": model.training,
        "optimizer": optimizer_snapshot(model, optimizer), "gradients": gradients(model),
        "scaler": scaler.state_dict(), "class_statistics": copy.deepcopy(interface.data),
        "progress": copy.deepcopy(progress), "rng": rng_state(),
    }


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import restore_rng, rng_state, seed_everything
    initialize_runtime()
    cfg, stop_batch = options(config)
    dev, amp = policy(method, device)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Fourier seed requires32bit integer")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("Fourier epoch interruption requires positive integer")
    bundle = Bundle(data, fm, split)
    inventory = check_bundle(bundle)
    torch.set_num_threads(cfg["threads"])
    seed_everything(seed)
    profile, out = SPECS[method], Path(output).expanduser().resolve()
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg, "method_spec": profile,
        "source_contract": contract(), "software": software_identity(),
        "numerical_identity": numerical_identity(amp),
        "bundle_sha256": bundle.fingerprint, "features_sha256": inventory["feature_inventory_sha256"],
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("Fourier recovery changed request/data/software/source/precision")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("Fourier run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        model, interface, optimizer, scaler, author = components(
            len(bundle.classes), dev, profile["author_recipe"])
        progress = {
            "epoch": 0, "train_batches": 0, "optimizer_steps": 0, "skipped_steps": 0,
            "order": [], "cursor": 0, "train_trace": [], "history": [],
            "best_val_loss": float("inf"), "wait_count": 0, "stopped_early": False,
        }
        if resume:
            state = torch.load(out / "last.pt", map_location=dev, weights_only=False)
            if state["request_sha256"] != digest_object(request):
                raise ValueError("Fourier checkpoint request mismatch")
            model.load_state_dict(state["model"], strict=True)
            model.train(state["model_training"])
            restore_optimizer(model, optimizer, state["optimizer"])
            restore_gradients(model, state["gradients"])
            scaler.load_state_dict(state["scaler"])
            interface.data, progress = state["class_statistics"], state["progress"]
            restore_rng(state["rng"])
        write_json(out / "method_protocol.json", {
            "method": method, "author_recipe": profile["author_recipe"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "TEST_locked": True, "source": request["source_contract"],
            "patch_order": "unaltered per-slide features, no patch cap or resampling",
            "current_framework_compatibility_not_identical_author_environment": True,
        })
        dataset = SlideBags(bundle, "train")
        while progress["epoch"] < 200 and not progress["stopped_early"]:
            model.train()
            if progress["order"]:
                saved_rng = rng_state()
                sampler = RemainingOrder(progress["order"][progress["cursor"]:])
                loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8,
                                    collate_fn=collate)
                iterator = iter(loader)
                restore_rng(saved_rng)  # suppress only additional recovery iterator seed draw
            else:
                interface.data = [{"count": 0, "correct": 0} for _ in bundle.classes]
                sampler = RecordingRandomSampler(dataset)
                loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8,
                                    collate_fn=collate)
                iterator = iter(loader)
            for batch in iterator:
                if not progress["order"]:
                    progress["order"] = sampler.order
                index = progress["cursor"]
                trace = minibatch(model, interface, optimizer, scaler, author, batch,
                                  index, len(dataset), dev, amp)
                trace["slide_id"] = dataset.rows[progress["order"][index]]["slide_id"]
                progress["train_trace"].append(trace)
                progress["cursor"] += 1
                progress["train_batches"] += 1
                progress["optimizer_steps"] += int(trace["optimizer_boundary"]
                                                   and not trace["optimizer_step_skipped"])
                progress["skipped_steps"] += int(trace["optimizer_step_skipped"])
                atomic_torch_save(snapshot(model, interface, optimizer, scaler, progress, request),
                                  out / "last.pt")
                if stop_batch is not None and progress["train_batches"] >= stop_batch:
                    write_json(out / "status.json", {
                        "state": "interrupted", "backend": BACKEND,
                        "completed_train_batches": progress["train_batches"]})
                    return out
            train_statistics = copy.deepcopy(interface.data)
            metrics = validation(model, interface, author, bundle, dev, amp)
            improved = metrics["loss"] < progress["best_val_loss"]
            if improved:
                progress.update(best_val_loss=metrics["loss"], wait_count=0)
            else:
                progress["wait_count"] += 1
            progress["stopped_early"] = progress["wait_count"] >= 10
            progress["history"].append({
                "epoch": progress["epoch"] + 1, "val": metrics,
                "train_class_statistics": train_statistics, "train_trace": progress["train_trace"],
                "train_order": [dataset.rows[i]["slide_id"] for i in progress["order"]],
                "improved": improved, "best_val_loss": progress["best_val_loss"],
                "wait_count": progress["wait_count"], "optimizer_steps": progress["optimizer_steps"],
            })
            progress.update(epoch=progress["epoch"] + 1, order=[], cursor=0, train_trace=[])
            interface.data = [{"count": 0, "correct": 0} for _ in bundle.classes]
            state = snapshot(model, interface, optimizer, scaler, progress, request)
            if improved:
                atomic_torch_save(state, out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"Fourier epoch={progress['epoch']}/200 VAL_loss={metrics['loss']} "
                  f"wait={progress['wait_count']}/10 steps={progress['optimizer_steps']}", flush=True)
            if (stop_after is not None and progress["epoch"] >= stop_after
                    and progress["epoch"] < 200 and not progress["stopped_early"]):
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND, "completed_epochs": progress["epoch"]})
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("Fourier literal VAL selector produced no finite selected checkpoint")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND, "completed_epochs": progress["epoch"],
            "maximum_epoch_budget": 200, "stopped_early": progress["stopped_early"],
            "completed_train_batches": progress["train_batches"],
            "optimizer_steps": progress["optimizer_steps"], "skipped_steps": progress["skipped_steps"],
            "test_evaluated": False, "precision": 16 if amp else 32,
            "original_environment_reproduced": False,
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
    from .engine import score, seed_everything
    initialize_runtime()
    out = Path(run).expanduser().resolve()
    request = read_json(out / "request.json")
    if subset not in {"val", "test"} or read_json(out / "status.json")["state"] != "completed":
        raise ValueError("Fourier selected evaluation requires complete fitting and VAL/TEST")
    dev, amp = policy(request["method"], device)
    torch.set_num_threads(request["config"]["threads"])
    seed_everything(request["seed"])
    if (request["software"] != software_identity() or request["device"] != str(dev)
            or request["numerical_identity"] != numerical_identity(amp)
            or request["source_contract"] != contract()):
        raise ValueError("Fourier selected evaluation software/device/source/precision changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = check_bundle(bundle)
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]):
        raise ValueError("Fourier selected evaluation data changed")
    state = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    if state["request_sha256"] != digest_object(request):
        raise ValueError("Fourier selected checkpoint request mismatch")
    model = definitions().Model(n_classes=len(bundle.classes)).to(dev)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    records = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            with autocast(amp):
                probabilities = model(data=features.unsqueeze(0).to(dev))["Y_prob"].flatten().cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Fourier selected probabilities became nonfinite")
            records.append({
                "slide_id": row["slide_id"], "patient_id": row["patient_id"],
                "global_patient_id": row["global_patient_id"], "subset": subset,
                "source_slides": row["slide_id"], "target": int(row["target"]),
                **{f"p_{i}": float(v) for i, v in enumerate(probabilities)},
            })
    frame = pd.DataFrame(records)
    metrics = score(bundle, frame)
    frame.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", metrics)
    return metrics

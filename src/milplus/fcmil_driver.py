"""Resumable original FC-MIL mechanism with explicit benchmark recipe and repairs."""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .attrimil_driver import numerical_identity
from .data import Bundle
from .fcmil_source import contract, definitions
from .fcmil_spec import PROFILES
from .io import atomic_torch_save, digest_object, read_json, write_json
from .pneamil_driver import RecordingRandomSampler, RemainingOrder, own_lock
from .provenance import software_identity

BACKEND = "fcmil_original_mechanism_explicit_benchmark100_auc_writer_repaired"
SPECS = {p["id"]: p for p in PROFILES}


def initialize_runtime():
    """Bind the declared benchmark policy in both library and fresh CLI calls."""
    from .numerics import initialize_cpu_vector_math
    initialize_cpu_vector_math()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlideBags(Dataset):
    def __init__(self, bundle, subset):
        self.bundle, self.rows = bundle, bundle.records(subset)
        self.access = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, _ = self.bundle.load(row)
        self.access.append(row["slide_id"])
        return features.unsqueeze(0), torch.tensor([int(row["target"])], dtype=torch.long)


def collate(batch):
    assert len(batch) == 1
    return batch[0]


def options(config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("FC-MIL explicit benchmark recipe accepts runtime/interruption options only")
    if "epochs" in config and config["epochs"] != 100:
        raise ValueError("FC-MIL benchmark100 complete epoch budget cannot be shortened")
    settings = config.get("model", {})
    if not isinstance(settings, dict) or set(settings) - {"stop_after_train_batches"}:
        raise ValueError("FC-MIL preserves original1024->512 model, FFT order and native objective")
    threads = config.get("threads", 2)
    stop_batch = settings.get("stop_after_train_batches")
    if type(threads) is not int or threads < 1:
        raise ValueError("FC-MIL threads must be a positive integer")
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError("FC-MIL interruption requires a positive actual update count")
    return {"threads": threads}, stop_batch


def check_bundle(bundle):
    if (bundle.task_type != "classification" or bundle.dim != 1024
            or bundle.config.get("training_unit", "slide") != "slide"):
        raise ValueError("Original FC-MIL requires individual-slide classification and1024D features")
    inventory = bundle.check(deep=True)
    for subset in ("train", "val", "test"):
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            if len(features) < 4:
                raise ValueError("Original FC-MIL FFT quarter-band requires at least4 patches")
    if {int(r["target"]) for r in bundle.records("val")} != set(range(len(bundle.classes))):
        raise ValueError("FC-MIL strict real VAL macro AUROC requires all declared classes")
    return inventory


def components(bundle, profile, device, out):
    author = definitions()
    manager = author.manager.TrainingManager.__new__(author.manager.TrainingManager)
    manager.cfg = SimpleNamespace(**profile["benchmark_recipe"])
    manager.model = author.models.FCMIL(n_classes=len(bundle.classes)).to(device)
    manager.best_auc, manager.current_epoch = 0., 0
    manager.checkpoint_path = str(out / "native_selected_weights.pt")
    manager._build_optimizer()
    return manager, author


@torch.enable_grad()
def step(model, optimizer, author, batch, recipe, device):
    features, targets = [v.to(device) for v in batch]
    logits, attention = model(features)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    # The literal trainer computes the counterfactuals even when alpha_cr=0.
    cr = author.loss.causal_mil_loss(
        model, logits, features, attention,
        topk_ratio=recipe["topk_ratio"], lam=recipe["lam"])
    loss = ce + recipe["alpha_cr"] * cr
    if not torch.isfinite(loss):
        raise FloatingPointError("Original FC-MIL objective became nonfinite")
    optimizer.zero_grad()
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Original FC-MIL gradient became nonfinite")
    optimizer.step()
    return {
        "features_sha256": hashlib.sha256(features.detach().cpu().numpy().tobytes()).hexdigest(),
        "target": targets.detach().cpu().tolist(), "patch_count": features.shape[1],
        "loss": loss.item(), "loss_ce": ce.item(), "loss_cr": cr.item(),
        "correct": int(logits.argmax(1).eq(targets).sum()),
        "logits": logits.detach().cpu().tolist(), "lr_used": optimizer.param_groups[0]["lr"],
    }


def validation(model, loader, device):
    model.eval()
    probabilities, targets, total_loss, correct = [], [], 0., 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits, _ = model(features)
            total_loss += torch.nn.functional.cross_entropy(logits, labels).item()
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
            targets.append(int(labels[0]))
            correct += int(logits.argmax(1).eq(labels).sum())
    probabilities = np.stack(probabilities)
    aucs = [roc_auc_score(np.asarray(targets) == i, probabilities[:, i])
            for i in range(probabilities.shape[1])]
    return {"loss": total_loss / len(loader), "loss_ce": total_loss / len(loader),
            "loss_cr": 0., "accuracy": correct / len(targets), "auc": float(np.mean(aucs))}


def snapshot(manager, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": manager.model.state_dict(), "optimizer": manager.optimizer.state_dict(),
        "scheduler": manager.scheduler.state_dict(), "best_auc": manager.best_auc,
        "model_training": manager.model.training, "progress": progress, "rng": rng_state(),
    }


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import restore_rng, rng_state, seed_everything
    initialize_runtime()
    cfg, stop_batch = options(config)
    if method not in SPECS:
        raise ValueError("Select an explicit FC-MIL original-model benchmark profile")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("FC-MIL requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("FC-MIL epoch interruption requires a positive integer")
    bundle = Bundle(data, fm, split)
    inventory = check_bundle(bundle)
    torch.set_num_threads(cfg["threads"])
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("FC-MIL original attention/FFT requires CPU/CUDA")
    seed_everything(seed)
    profile, out = SPECS[method], Path(output).expanduser().resolve()
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg, "method_spec": profile,
        "source_contract": contract(), "software": software_identity(),
        "numerical_identity": numerical_identity(),
        "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"],
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "author_training_configuration_recovered": False,
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("FC-MIL recovery changed request/data/software/native definitions")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("FC-MIL run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        manager, author = components(bundle, profile, dev, out)
        progress = {"epoch": 0, "updates": 0, "order": [], "cursor": 0,
                    "train_trace": [], "history": [], "train_access": []}
        if resume:
            state = torch.load(out / "last.pt", map_location=dev, weights_only=False)
            if state["request_sha256"] != digest_object(request):
                raise ValueError("FC-MIL saved checkpoint identity differs")
            manager.model.load_state_dict(state["model"], strict=True)
            manager.model.train(state["model_training"])
            manager.optimizer.load_state_dict(state["optimizer"])
            manager.scheduler.load_state_dict(state["scheduler"])
            manager.best_auc = state["best_auc"]
            progress = state["progress"]
            restore_rng(state["rng"])
        write_json(out / "method_protocol.json", {
            "method": method, "benchmark_recipe": profile["benchmark_recipe"],
            "parameters": sum(p.numel() for p in manager.model.parameters()),
            "source": request["source_contract"], "TEST_locked": True,
            "patch_order": "unaltered Bundle rows; no hidden cap, shuffle or resampling",
            "batch_size": 1, "workers": 0, "training_configuration_is_benchmark_choice": True,
        })
        dataset, val_dataset = SlideBags(bundle, "train"), SlideBags(bundle, "val")
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate)
        while progress["epoch"] < 100:
            manager.current_epoch = progress["epoch"]
            manager.model.train()
            if progress["order"]:
                saved_rng = rng_state()
                sampler = RemainingOrder(progress["order"][progress["cursor"]:])
                loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate)
                iterator = iter(loader)
                restore_rng(saved_rng)  # suppress only the extra recovery iterator's seed draw
            else:
                sampler = RecordingRandomSampler(dataset)
                loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate)
                iterator = iter(loader)
            for batch in iterator:
                trace = step(manager.model, manager.optimizer, author, batch,
                             profile["benchmark_recipe"], dev)
                if not progress["order"]:
                    progress["order"] = sampler.order
                progress["train_trace"].append(trace)
                progress["train_access"].append(dataset.rows[progress["order"][progress["cursor"]]]["slide_id"])
                progress["cursor"] += 1
                progress["updates"] += 1
                atomic_torch_save(snapshot(manager, progress, request), out / "last.pt")
                if stop_batch is not None and progress["updates"] >= stop_batch:
                    write_json(out / "status.json", {
                        "state": "interrupted", "backend": BACKEND,
                        "completed_train_batches": progress["updates"]})
                    return out
            traces = progress["train_trace"]
            train_metrics = {
                key: sum(row[key] for row in traces) / len(traces)
                for key in ("loss", "loss_ce", "loss_cr")}
            train_metrics["accuracy"] = sum(row["correct"] for row in traces) / len(traces)
            metrics = validation(manager.model, val_loader, dev)
            previous_auc = manager.best_auc
            manager._save_best_model(metrics)  # original strict selector, actual repaired AUC
            improved = manager.best_auc > previous_auc
            manager.scheduler.step(metrics["loss"])
            progress["history"].append({
                "epoch": progress["epoch"] + 1, "train": train_metrics, "val": metrics,
                "improved": improved, "best_auc": manager.best_auc,
                "lr_next": [g["lr"] for g in manager.optimizer.param_groups],
                "train_order": progress["train_access"], "train_trace": traces,
                "updates": progress["updates"],
            })
            progress.update(epoch=progress["epoch"] + 1, order=[], cursor=0,
                            train_trace=[], train_access=[])
            state = snapshot(manager, progress, request)
            if improved:
                atomic_torch_save(copy.deepcopy(state), out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"FC-MIL epoch={progress['epoch']}/100 updates={progress['updates']} "
                  f"VAL_AUC={metrics['auc']} best={manager.best_auc}", flush=True)
            if stop_after is not None and progress["epoch"] >= stop_after and progress["epoch"] < 100:
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND,
                    "completed_epochs": progress["epoch"]})
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("Original strict AUC selector produced no checkpoint because no VAL AUC exceeded0")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND, "completed_epochs": 100,
            "maximum_epoch_budget": 100, "completed_train_batches": progress["updates"],
            "test_evaluated": False, "original_author_training_config_recovered": False})
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
        raise ValueError("FC-MIL selected evaluation requires complete training and VAL/TEST")
    dev = torch.device(device)
    torch.set_num_threads(request["config"]["threads"])
    seed_everything(request["seed"])
    if (request["software"] != software_identity() or request["device"] != str(dev)
            or request["numerical_identity"] != numerical_identity()
            or request["source_contract"] != contract()):
        raise ValueError("FC-MIL selected evaluation software/device/source identity changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = check_bundle(bundle)
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]):
        raise ValueError("FC-MIL selected evaluation data changed")
    model = definitions().models.FCMIL(n_classes=len(bundle.classes)).to(dev)
    state = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    records = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            logits, _ = model(features.to(dev).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).flatten().cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Original FC-MIL selected probabilities became nonfinite")
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

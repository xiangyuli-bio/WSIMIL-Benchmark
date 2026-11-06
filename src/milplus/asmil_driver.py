"""Full author ASMIL budgets, explicit teacher semantics and exact recovery."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .asmil_source import author_utilities, contract, definitions
from .asmil_spec import DIMENSIONS, PROFILES
from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .pneamil_driver import RecordingRandomSampler, RemainingOrder, own_lock
from .provenance import software_identity

BACKEND = "asmil_complete_author_recipe_locked_test"
SPECS = {p["id"]: p for p in PROFILES}


class SlideBags(Dataset):
    def __init__(self, bundle, subset):
        self.bundle, self.rows = bundle, bundle.records(subset)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        features, _ = self.bundle.load(self.rows[index])
        return {"input": features, "label": torch.tensor(int(self.rows[index]["target"]))}


def options(profile, config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("ASMIL author profiles accept runtime options and actual author CLI choices only")
    recipe = dict(profile["native_recipe"])
    if "epochs" in config and config["epochs"] != recipe["epochs"]:
        raise ValueError("ASMIL full author epoch budget cannot be reduced; use stop_after")
    values = config.get("model", {})
    allowed = {"n_token", "n_drop", "temp", "lr", "stop_after_train_batches"}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("ASMIL preserves full native widths, batch1, full bags and author optimizer")
    for k in ("n_token", "n_drop", "temp", "lr"):
        if k in values:
            recipe[k] = values[k]
    if (type(recipe["n_token"]) is not int or recipe["n_token"] < 1
            or type(recipe["n_drop"]) is not int
            or not 0 <= recipe["n_drop"] < recipe["n_token"]):
        raise ValueError("ASMIL token/drop choices must leave at least one attention token")
    for k in ("lr", "temp"):
        if (type(recipe[k]) not in (int, float) or not math.isfinite(recipe[k])
                or recipe[k] <= 0):
            raise ValueError("ASMIL learning rate and temperature must be positive finite numbers")
    threads = config.get("threads", 2)
    stop = values.get("stop_after_train_batches")
    if type(threads) is not int or threads < 1:
        raise ValueError("ASMIL threads must be a positive integer")
    if stop is not None and (type(stop) is not int or stop < 1):
        raise ValueError("ASMIL minibatch interruption must be a positive integer")
    return {"threads": threads, "recipe": recipe}, stop


def metric_runtime():
    import torchmetrics
    if torchmetrics.__version__ != "1.3.0.post0":
        raise ValueError("ASMIL declared metric reference requires torchmetrics==1.3.0.post0")
    return torchmetrics


def check_bundle(bundle):
    if (bundle.task_type != "classification"
            or bundle.config.get("training_unit", "slide") != "slide"
            or bundle.dim not in DIMENSIONS):
        raise ValueError("ASMIL requires slide classification and an original384/512/768/1024/1536D input")
    return bundle.check(deep=True)


def make_models(bundle, recipe, device):
    source = definitions()
    conf = SimpleNamespace(
        D_feat=bundle.dim, D_inner=DIMENSIONS[bundle.dim], n_class=len(bundle.classes),
        train_epoch=recipe["epochs"], warmup_epoch=0, min_lr=0,
        lr=recipe["lr"], wd=recipe["weight_decay"], wandb_mode="disabled",
    )
    online = source.ACMIL_MYMHA(conf, n_token=recipe["n_token"], n_drop=recipe["n_drop"])
    # The original source constructs a separate teacher; there is NO state copy.
    teacher = source.ACMIL_MYMHA_EMA(conf, n_token=recipe["n_token"])
    online.to(device)
    teacher.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, online.parameters()),
        lr=recipe["initial_optimizer_lr"], weight_decay=recipe["weight_decay"],
    )
    return online, teacher, optimizer, conf, source


def train_step(online, teacher, optimizer, conf, source, batch, device,
               fractional_epoch, temp, teacher_update, schedule):
    """Reviewed transcription of actual train_one_epoch, tested against its AST."""
    online.train()
    teacher.train()
    features = batch["input"].to(device, dtype=torch.float32)
    targets = batch["label"].to(device)
    schedule(optimizer, fractional_epoch, conf)
    _, logits, attention = online(features)
    with torch.no_grad():
        teacher_attention = teacher(features).detach()
        msd, esd = online.state_dict(), teacher.state_dict()
        for name in msd:
            if teacher_update == "literal_nan_teacher":
                update = torch.isnan(msd[name]).any()
            elif teacher_update == "finite_ema_repaired":
                update = torch.isfinite(msd[name]).all()
            else:
                raise ValueError("Unknown ASMIL teacher condition")
            if update:
                esd[name].copy_(.999 * esd[name] + (1. - .999) * msd[name])
    ce = nn.functional.cross_entropy(logits, targets)
    teacher_prob = source.sigmax(teacher_attention / temp, dim=-1)
    attention_prob = torch.softmax(attention + 1e-8, dim=-1)
    kd = -torch.nan_to_num(teacher_prob * attention_prob.log()).mean()
    # Preserve the literal source's zero-valued auxiliary terms and arithmetic order.
    loss = (torch.tensor(0., device=device) + torch.tensor(0., device=device) + ce + kd)
    if not torch.isfinite(loss):
        raise FloatingPointError("ASMIL original objective became nonfinite")
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=5.)
    optimizer.step()
    if not torch.isfinite(norm):
        raise FloatingPointError("ASMIL original gradient norm became nonfinite")
    return {
        "loss": loss.item(), "ce": ce.item(), "kd": kd.item(),
        "lr": optimizer.param_groups[0]["lr"], "grad_norm_before_clip": float(norm),
        "patch_count": int(features.shape[1]),
    }


@torch.no_grad()
def validation(online, dataset, conf, device):
    tm = metric_runtime()
    online.eval()
    predictions, targets, losses = [], [], []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    for batch in loader:
        _, logits, _ = online(batch["input"].to(device, dtype=torch.float32))
        target = batch["label"].to(device)
        losses.append(nn.functional.cross_entropy(logits, target).item())
        predictions.append(logits.softmax(-1))
        targets.append(target)
    probabilities, labels = torch.cat(predictions), torch.cat(targets)
    # Match the author's actual unqualified defaults: AUROC macro, F1 micro.
    auc_metric = tm.AUROC(num_classes=conf.n_class, task="multiclass").to(device)
    f1_metric = tm.F1Score(num_classes=conf.n_class, task="multiclass").to(device)
    auc_metric(probabilities, labels)
    f1_metric(probabilities, labels)
    result = {
        "auc": auc_metric.compute().item(), "f1": f1_metric.compute().item(),
        "acc": (probabilities.argmax(-1) == labels).float().mean().item() * 100.,
        "loss": sum(losses) / len(losses),
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("ASMIL validation selector is nonfinite")
    return result, probabilities.cpu()


def checkpoint(online, teacher, optimizer, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": online.state_dict(), "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(), "progress": progress, "rng": rng_state(),
    }


def restore(online, teacher, optimizer, state, request):
    from .engine import restore_rng
    if state["request_sha256"] != digest_object(request):
        raise ValueError("ASMIL checkpoint is bound to another software/data/recipe request")
    online.load_state_dict(state["model"], strict=True)
    teacher.load_state_dict(state["teacher"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["progress"]


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import restore_rng, rng_state, seed_everything
    if method not in SPECS:
        raise ValueError("Choose an explicit ASMIL author profile")
    profile = SPECS[method]
    cfg, stop_batch = options(profile, config)
    recipe = cfg["recipe"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("ASMIL requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("ASMIL epoch interruption must be a positive integer")
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("ASMIL author runtime supports CPU and CUDA")
    metric_runtime()
    bundle = Bundle(data, fm, split)
    inventory = check_bundle(bundle)
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg,
        "software": software_identity(), "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"],
        "source_contract": contract(profile), "method_spec": profile,
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    out = Path(output).expanduser().resolve()
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("ASMIL recovery changed inputs, software, teacher or full recipe")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("ASMIL run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        torch.set_num_threads(cfg["threads"])
        seed_everything(seed)
        online, teacher, optimizer, conf, source = make_models(bundle, recipe, dev)
        schedule = author_utilities().adjust_learning_rate
        write_json(out / "method_protocol.json", {
            "method": method, "recipe": recipe, "teacher_update": profile["teacher_update"],
            "source_contract": request["source_contract"], "input_dimension": bundle.dim,
            "inner_dimension": DIMENSIONS[bundle.dim], "heads": 8, "dropout": .1,
            "online_parameters": sum(p.numel() for p in online.parameters()),
            "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
            "selector": "strict VAL micro-F1+macro-AUROC; initial best score0",
            "no_early_stopping": True, "test_evaluated_during_training": False,
            "checkpoint_boundaries": "every actual TRAIN batch and VAL selector",
            "source_row_order": "task manifest sorted slide_id; patient partitions fixed",
            "runtime_workers": "0 for recoverable same-process data reads; source specifies8",
        })
        progress = {
            "epoch": 0, "cursor": 0, "order": None, "updates": 0,
            "train_trace": [], "history": [], "best_score": 0.,
        }
        if resume:
            progress = restore(online, teacher, optimizer, torch.load(
                out / "last.pt", map_location=dev, weights_only=False), request)
        train_data, val_data = SlideBags(bundle, "train"), SlideBags(bundle, "val")
        while progress["epoch"] < recipe["epochs"]:
            if progress["cursor"] < len(train_data):
                recovering = progress["order"] is not None
                sampler = (RemainingOrder(progress["order"][progress["cursor"]:])
                           if recovering else RecordingRandomSampler(train_data))
                loader = DataLoader(train_data, batch_size=1, sampler=sampler,
                                    num_workers=0, drop_last=True)
                saved_rng = rng_state() if recovering else None
                iterator = iter(loader)
                if recovering:
                    restore_rng(saved_rng)
                for batch in iterator:
                    step = train_step(
                        online, teacher, optimizer, conf, source, batch, dev,
                        progress["epoch"] + progress["cursor"] / len(train_data),
                        recipe["temp"], profile["teacher_update"], schedule,
                    )
                    if progress["order"] is None:
                        progress["order"] = list(sampler.order)
                    index = progress["order"][progress["cursor"]]
                    progress["train_trace"].append({
                        "slide_id": train_data.rows[index]["slide_id"], **step,
                    })
                    progress["cursor"] += 1
                    progress["updates"] += 1
                    atomic_torch_save(
                        checkpoint(online, teacher, optimizer, progress, request), out / "last.pt")
                    if stop_batch is not None and progress["updates"] >= stop_batch:
                        write_json(out / "status.json", {
                            "state": "interrupted", "backend": BACKEND,
                            "completed_train_batches": progress["updates"],
                            "reason": "requested actual TRAIN batch boundary",
                        })
                        return out
            if progress["cursor"] != len(train_data):
                raise ValueError("ASMIL epoch did not visit every TRAIN bag exactly once")
            selected_metrics, _ = validation(online, val_data, conf, dev)
            selector = selected_metrics["f1"] + selected_metrics["auc"]
            improved = selector > progress["best_score"]
            if improved:
                progress["best_score"] = selector
            record = {
                "epoch": progress["epoch"] + 1, "validation": selected_metrics,
                "selection_score": selector, "improved": improved,
                "train_order": list(progress["order"]), "train_trace": progress["train_trace"],
                "total_updates": progress["updates"],
            }
            progress["history"].append(record)
            progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None, train_trace=[])
            state = checkpoint(online, teacher, optimizer, progress, request)
            if improved:
                atomic_torch_save(copy.deepcopy(state), out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"epoch={progress['epoch']} val_f1_auc={selector:.8f}", flush=True)
            if (stop_after is not None and progress["epoch"] >= stop_after
                    and progress["epoch"] < recipe["epochs"]):
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND,
                    "completed_epochs": progress["epoch"],
                    "completed_train_batches": progress["updates"],
                    "reason": "requested epoch boundary",
                })
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("ASMIL original strict selector never produced a checkpoint")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND,
            "completed_epochs": progress["epoch"],
            "completed_train_batches": progress["updates"],
            "maximum_epoch_budget": recipe["epochs"],
            "termination": "complete_author_epoch_budget", "test_evaluated": False,
        })
        return out
    except BaseException as error:
        write_json(out / "status.json", {
            "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "backend": BACKEND, "error_type": type(error).__name__, "error": str(error),
        })
        raise
    finally:
        lock.unlink(missing_ok=True)


def evaluate(run, subset="test", device="cpu"):
    from .engine import score, seed_everything
    out = Path(run).expanduser().resolve()
    request = read_json(out / "request.json")
    if subset not in {"val", "test"} or read_json(out / "status.json")["state"] != "completed":
        raise ValueError("ASMIL evaluation requires complete training and val/test subset")
    if request["software"] != software_identity() or request["device"] != str(torch.device(device)):
        raise ValueError("ASMIL selected evaluation software or device changed")
    profile = SPECS[request["method"]]
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = check_bundle(bundle)
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]
            or contract(profile) != request["source_contract"]):
        raise ValueError("ASMIL evaluation changed data bytes, patient split or source pin")
    metric_runtime()
    torch.set_num_threads(request["config"]["threads"])
    seed_everything(request["seed"])
    online, teacher, optimizer, conf, _ = make_models(
        bundle, request["config"]["recipe"], torch.device(device))
    selected = torch.load(out / "best.pt", map_location=device, weights_only=False)
    restore(online, teacher, optimizer, selected, request)
    native_metrics, probabilities = validation(online, SlideBags(bundle, subset), conf, device)
    predictions = [
        {
            "slide_id": row["slide_id"], "patient_id": row["patient_id"],
            "global_patient_id": row["global_patient_id"], "subset": subset,
            "source_slides": row["slide_id"], "target": int(row["target"]),
            **{f"p_{i}": float(p) for i, p in enumerate(probability)},
        }
        for row, probability in zip(bundle.records(subset), probabilities)
    ]
    frame = pd.DataFrame(predictions)
    metrics = score(bundle, frame)
    frame.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", metrics)
    write_json(out / f"{subset}_author_slide_metrics.json", native_metrics)
    return metrics

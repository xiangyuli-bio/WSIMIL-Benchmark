"""Complete PNEA author training with explicit source/VAL selection references."""

import copy
import hashlib
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .pneamil_source import original_definitions, source_contract
from .pneamil_spec import PROFILES, RECIPE
from .provenance import software_identity


BACKEND = "pneamil_complete_author_recipe"
SPECS = {row["id"]: row for row in PROFILES}


class SlideBags(Dataset):
    def __init__(self, bundle, subset, sample=False):
        self.bundle, self.rows, self.sample = bundle, bundle.records(subset), sample
        self.access = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, _ = self.bundle.load(row)
        count = len(features)
        if self.sample:
            selected = torch.randint(low=0, high=count, size=(RECIPE["train_patches"],))
            features = features[selected]
            selection = {
                "patch_indices_sha256": hashlib.sha256(selected.numpy().tobytes()).hexdigest(),
                "unique_indices": len(torch.unique(selected)), "sampled_patches": len(selected),
            }
        else:
            selection = {"sampled_patches": count, "all_original_patches": True}
        self.access.append({"slide_id": row["slide_id"], "original_patches": count, **selection})
        return features, torch.tensor(int(row["target"]), dtype=torch.long)


class RecordingRandomSampler(RandomSampler):
    """Preserve the actual RandomSampler algorithm and record its emitted order."""

    def __iter__(self):
        self.order = list(super().__iter__())
        yield from self.order


class RemainingOrder(Sampler):
    def __init__(self, order):
        self.order = list(order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def options(config):
    supplied = config or {}
    if not isinstance(supplied, dict) or set(supplied) - {"threads", "model", "epochs"}:
        raise ValueError("PNEA author profiles preserve the complete recipe; only runtime/interruption options are accepted")
    if "epochs" in supplied and supplied["epochs"] != RECIPE["max_epochs"]:
        raise ValueError("PNEA author maximum1000epoch budget cannot be reduced; use --stop-after to interrupt")
    settings = supplied.get("model", {})
    if not isinstance(settings, dict) or set(settings) - {"stop_after_train_batches"}:
        raise ValueError("PNEA preserves original64 axes, batch8,1024patches and full author optimizer")
    threads = supplied.get("threads", 2)
    stop_batch = settings.get("stop_after_train_batches")
    if type(threads) is not int or threads < 1:
        raise ValueError("PNEA threads must be a positive integer")
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError("PNEA interruption requires a positive actual TRAIN minibatch count")
    return {"threads": threads, "recipe": dict(RECIPE)}, stop_batch


def check_bundle(bundle):
    if (bundle.task_type != "classification"
            or bundle.config.get("training_unit", "slide") != "slide"):
        raise ValueError("PNEA author recipes require individual slide classification bags")
    return bundle.check(deep=True)


def finite_model(model):
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("PNEA contains a nonfinite updated model parameter")
    for state in model.optim.state.values():
        if any(torch.is_tensor(value) and not torch.isfinite(value).all()
               for value in state.values()):
            raise FloatingPointError("PNEA contains a nonfinite optimizer state")


def checkpoint(model, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": model.state_dict(), "optimizer": model.optim.state_dict(),
        "progress": progress, "rng": rng_state(),
    }


def restore(model, state, request):
    from .engine import restore_rng
    if state["request_sha256"] != digest_object(request):
        raise ValueError("PNEA checkpoint belongs to another input/software/recipe request")
    model.load_state_dict(state["model"], strict=True)
    model.optim.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["progress"]


def optimizer_groups(model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        {"names": [names[id(parameter)] for parameter in group["params"]],
         "lr": group["lr"], "weight_decay": group["weight_decay"],
         "betas": list(group["betas"]), "eps": group["eps"]}
        for group in model.optim.param_groups
    ]


def own_lock(path):
    def identity(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[19]
        except (FileNotFoundError, IndexError, ProcessLookupError):
            return None
    if path.exists():
        previous = read_json(path)
        if identity(previous["pid"]) == previous["starttime"]:
            raise ValueError("The actual PNEA owner process is still live")
        path.rename(path.with_name(path.name + f".stale.{time.time_ns()}.json"))
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        import json
        json.dump({"pid": os.getpid(), "starttime": identity(os.getpid())}, stream)


def make_model(bundle, seed, device):
    model_class, trainer = original_definitions()
    model = model_class(
        D_in=bundle.dim, D=RECIPE["evidence_axes"], C=len(bundle.classes), seed=seed,
        lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"],
        lambda_l1=RECIPE["lambda_l1"],
    )
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    group_names = [
        [names[id(parameter)] for parameter in group["params"]]
        for group in model.optim.param_groups
    ]
    model.to(device)
    # Module.to can replace Parameter objects on some supported devices/runtime
    # settings. Retain the author's exact groups, pointing at the actual model.
    if model.optim.state:
        raise ValueError("PNEA author optimizer unexpectedly initialized with state")
    current = dict(model.named_parameters())
    for group, parameter_names in zip(model.optim.param_groups, group_names):
        group["params"] = [current[name] for name in parameter_names]
    return model, trainer


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import restore_rng, rng_state, seed_everything
    if method not in SPECS:
        raise ValueError("Choose an explicit PNEA author selection reference")
    cfg, stop_batch = options(config)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("PNEA requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("PNEA epoch interruption must be a positive integer")
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("PNEA author implementation supports CPU or CUDA")
    bundle = Bundle(data, fm, split)
    inventory = check_bundle(bundle)
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg, "software": software_identity(),
        "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"],
        "source_contract": source_contract(), "method_spec": SPECS[method],
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    out = Path(output).expanduser().resolve()
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("PNEA recovery changed actual inputs, native recipe or software")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("PNEA run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    started = time.monotonic()
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        torch.set_num_threads(cfg["threads"])
        seed_everything(seed)
        model, author = make_model(bundle, seed, dev)
        protocol = {
            "method": method, "source": request["source_contract"], "recipe": RECIPE,
            "selection_population": SPECS[method]["selection_population"],
            "repairs": SPECS[method]["repairs"], "optimizer_groups": optimizer_groups(model),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "input_dimension": bundle.dim, "classes": list(bundle.classes),
            "checkpoint_boundaries": "every completed TRAIN minibatch and epoch selector",
            "sampling": "actual torch.randint with replacement; actual RandomSampler order",
            "source_row_order": "task manifest sorted slide_id; fixed patient partitions",
            "source_definitions_rewritten": False,
            "full_budget_validation": "not_established_by_software_execution_alone",
        }
        write_json(out / "method_protocol.json", protocol)
        progress = {
            "epoch": 0, "cursor": 0, "order": None, "updates": 0,
            "train_total": 0., "train_count": 0, "train_trace": [],
            "best_loss": 1e20, "bad": 0, "history": [],
        }
        if resume:
            progress = restore(
                model, torch.load(out / "last.pt", map_location=dev, weights_only=False), request,
            )
        train_data, val_data = SlideBags(bundle, "train", True), SlideBags(bundle, "val")
        while progress["epoch"] < RECIPE["max_epochs"] and progress["bad"] < RECIPE["patience"]:
            if progress["cursor"] < len(train_data):
                recovering = progress["order"] is not None
                sampler = (
                    RemainingOrder(progress["order"][progress["cursor"]:])
                    if recovering else RecordingRandomSampler(train_data)
                )
                loader = DataLoader(
                    train_data, batch_size=RECIPE["batch_size"], sampler=sampler,
                    num_workers=0, drop_last=False,
                )
                # Recreating a DataLoader iterator consumes a base seed. A
                # recovered iterator continues an existing source epoch, so
                # restore that extra draw before reading remaining patches.
                saved_rng = rng_state() if recovering else None
                iterator = iter(loader)
                if recovering:
                    restore_rng(saved_rng)
                model.train()
                train_data.access = []
                for features, target in iterator:
                    features, target = features.to(dev), target.to(dev)
                    model.optim.zero_grad()
                    loss = model.return_loss(features, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("PNEA author objective is nonfinite")
                    loss.backward()
                    model.optim.step()
                    finite_model(model)
                    if progress["order"] is None:
                        progress["order"] = list(sampler.order)
                    progress["cursor"] += len(target)
                    progress["updates"] += 1
                    progress["train_total"] += loss.item() * len(target)
                    progress["train_count"] += len(target)
                    progress["train_trace"].extend(train_data.access)
                    train_data.access = []
                    atomic_torch_save(checkpoint(model, progress, request), out / "last.pt")
                    if stop_batch is not None and progress["updates"] >= stop_batch:
                        write_json(out / "status.json", {
                            "state": "interrupted", "backend": BACKEND,
                            "epoch": progress["epoch"], "completed_train_batches": progress["updates"],
                            "reason": "requested actual TRAIN minibatch boundary",
                        })
                        return out
            if progress["cursor"] != len(train_data):
                raise ValueError("PNEA source epoch did not visit every TRAIN slide exactly once")
            source_selector = SPECS[method]["selection_population"] == "train"
            selection_data = train_data if source_selector else val_data
            selection_data.access = []
            selection_sampler = RecordingRandomSampler(train_data) if source_selector else None
            selection_loader = DataLoader(
                selection_data, batch_size=RECIPE["batch_size"] if source_selector else 1,
                sampler=selection_sampler, shuffle=False, num_workers=0, drop_last=False,
            )
            selection_loss = author.evaluate_loss(model, selection_loader, dev)
            if not math.isfinite(selection_loss):
                raise FloatingPointError("PNEA selector objective is nonfinite")
            improved = (
                selection_loss < progress["best_loss"]
                - abs(progress["best_loss"]) * RECIPE["relative_improvement"]
            )
            if improved:
                progress["best_loss"], progress["bad"] = selection_loss, 0
            else:
                progress["bad"] += 1
            record = {
                "epoch": progress["epoch"] + 1,
                "train_loss": progress["train_total"] / progress["train_count"],
                "selection_loss": selection_loss, "selection_population": SPECS[method]["selection_population"],
                "improved": improved, "bad_epochs": progress["bad"],
                "train_order": list(progress["order"]),
                "train_trace": progress["train_trace"],
                "selection_order": list(selection_sampler.order) if source_selector else list(range(len(val_data))),
                "selection_trace": list(selection_data.access), "total_updates": progress["updates"],
            }
            progress["history"].append(record)
            progress.update(
                epoch=progress["epoch"] + 1, cursor=0, order=None,
                train_total=0., train_count=0, train_trace=[],
            )
            state = checkpoint(model, progress, request)
            if improved:
                # Serialization occurs now; later updates cannot mutate a best
                # snapshot through the author's shallow state_dict reference.
                atomic_torch_save(copy.deepcopy(state), out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(
                f"epoch={progress['epoch']} train_loss={record['train_loss']:.8f} "
                f"{record['selection_population']}_selector_loss={selection_loss:.8f} "
                f"bad={progress['bad']}", flush=True,
            )
            terminal = progress["epoch"] >= RECIPE["max_epochs"] or progress["bad"] >= RECIPE["patience"]
            if stop_after is not None and progress["epoch"] >= stop_after and not terminal:
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND,
                    "completed_epochs": progress["epoch"], "completed_train_batches": progress["updates"],
                    "reason": "requested epoch boundary",
                })
                return out
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND, "completed_epochs": progress["epoch"],
            "completed_train_batches": progress["updates"],
            "termination": "author_relative_loss_patience" if progress["bad"] >= RECIPE["patience"] else "author_max1000epochs",
            "maximum_epoch_budget": RECIPE["max_epochs"], "test_evaluated": False,
            "wall_seconds": time.monotonic() - started,
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
        raise ValueError("PNEA evaluation requires completed training and a val/test subset")
    if request["software"] != software_identity() or request["device"] != str(torch.device(device)):
        raise ValueError("PNEA evaluation software or declared device changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = check_bundle(bundle)
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]
            or source_contract() != request["source_contract"]):
        raise ValueError("PNEA evaluation data, full feature bytes, or author pin changed")
    torch.set_num_threads(request["config"]["threads"])
    seed_everything(request["seed"])
    model, _ = make_model(bundle, request["seed"], torch.device(device))
    selected = torch.load(out / "best.pt", map_location=device, weights_only=False)
    restore(model, selected, request)
    model.eval()
    predictions = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            logits = model(features.unsqueeze(0).to(device))
            probabilities = logits.softmax(-1)[0].cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("PNEA selected-model predictions are nonfinite")
            predictions.append({
                "slide_id": row["slide_id"], "patient_id": row["patient_id"],
                "global_patient_id": row["global_patient_id"], "subset": subset,
                "source_slides": row["slide_id"], "target": int(row["target"]),
                **{f"p_{index}": float(value) for index, value in enumerate(probabilities)},
            })
    frame = pd.DataFrame(predictions)
    metrics = score(bundle, frame)
    frame.to_csv(out / (subset + "_predictions.csv"), index=False)
    write_json(out / (subset + "_metrics.json"), metrics)
    return metrics

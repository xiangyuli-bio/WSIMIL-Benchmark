"""Complete CAR-MIL pre-TRAIN selection, native losses and exact recovery."""

import copy
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .attrimil_driver import numerical_identity
from .carmil_source import arguments, contract, definitions, model_and_loss, runtime_identity
from .carmil_spec import PROFILES, RECIPE
from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .pneamil_driver import RemainingOrder, own_lock
from .provenance import software_identity

BACKEND = "carmil_complete_author_pretrain_selector_locked_test"
SPECS = {profile["id"]: profile for profile in PROFILES}


class SlideBags(Dataset):
    def __init__(self, bundle, subset):
        self.bundle, self.rows = bundle, bundle.records(subset)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, _ = self.bundle.load(row)
        return features, int(row["target"])


class RecordingShuffle(RandomSampler):
    def __iter__(self):
        # Peek at the first original randperm using a clone. Do not advance
        # the private generator early: the original sampler's empty-tail
        # randperm must remain after its last yielded batch.
        clone = torch.Generator().set_state(self.generator.get_state())
        self.order = torch.randperm(len(self.data_source), generator=clone).tolist()
        yield from super().__iter__()


def options(config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("CAR preserves the complete100epoch author recipe")
    if "epochs" in config and config["epochs"] != 100:
        raise ValueError("CAR maximum100epoch budget cannot be reduced; use recoverable interruption")
    settings = config.get("model", {})
    if not isinstance(settings, dict) or set(settings) - {
        "initial_checkpoint", "stop_after_train_batches", "stop_after_selectors",
    }:
        raise ValueError("CAR native dimensions/Adam/loss/schedule cannot be replaced")
    threads = config.get("threads", 2)
    if type(threads) is not int or threads < 1:
        raise ValueError("CAR threads must be a positive integer")
    stops = [settings.get("stop_after_train_batches"), settings.get("stop_after_selectors")]
    if any(value is not None and (type(value) is not int or value < 1) for value in stops):
        raise ValueError("CAR interruption counts must be positive integers")
    checkpoint = settings.get("initial_checkpoint")
    if checkpoint is not None:
        import hashlib
        path = Path(checkpoint).expanduser().resolve()
        checkpoint = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"threads": threads, "initial_checkpoint": checkpoint}, stops


def validate_bundle(bundle, profile):
    if (bundle.task_type != "classification" or len(bundle.classes) != 2
            or bundle.config.get("training_unit", "slide") != "slide"):
        raise ValueError("CAR original selection/validation requires binary individual-slide classification")
    inventory = bundle.check(deep=True)
    for subset in ("train", "val"):
        if {int(row["target"]) for row in bundle.records(subset)} != {0, 1}:
            raise ValueError("CAR native binary TRAIN/VAL requires both classes")
    if profile["native_recipe"]["backbone"] == "dsmil":
        for subset in ("train", "val", "test"):
            for row in bundle.records(subset):
                features, _ = bundle.load(row)
                if len(features) < 2:
                    raise ValueError("Original CAR DSMIL squeeze requires at least2instances per bag")
    return inventory


def construct(bundle, profile, author, device, initial):
    model, criterion = model_and_loss(author, profile, bundle.dim, device)
    warm_start = None
    if initial:
        state = torch.load(initial["path"], map_location=device, weights_only=False)
        result = model.load_state_dict(state["model"], strict=False)
        warm_start = {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys}
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 0)
    early = author.utils.EarlyStopping(patience=30, stop_epoch=130, save_best_model_stage=0.)
    return model, criterion, optimizer, scheduler, early, warm_start


def train_step(model, criterion, optimizer, args, batch, device):
    model.train()
    optimizer.zero_grad()
    features, target = (value.to(device) for value in batch)
    if args.model == "dsmil":
        logits, attention, instance_loss, patches = model(features, target, torch.nn.CrossEntropyLoss())
    else:
        logits, attention = model(features)
        instance_loss, patches = 0., features.shape[1]
    bag_loss = criterion(logits[0].view(1, -1), target, logits[1].view(1, -1),
                         attention[0], attention[1])
    loss = args.cls_alpha * bag_loss + args.aux_alpha * instance_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Original CAR loss became nonfinite")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Original CAR gradient became nonfinite")
    optimizer.step()
    return {"loss": loss.item(), "bag_loss": bag_loss.item(),
            "instance_loss": float(instance_loss), "patch_count": patches}


def snapshot(model, optimizer, scheduler, early, generator, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "early_stopping": early.state_dict(),
        "loader_generator": generator.get_state(), "rng": rng_state(), "progress": progress,
    }


def restore(state, model, optimizer, scheduler, early, generator, request):
    from .engine import restore_rng
    if state["request_sha256"] != digest_object(request):
        raise ValueError("CAR checkpoint request changed")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    early.load_state_dict(state["early_stopping"])
    generator.set_state(state["loader_generator"].cpu())
    restore_rng(state["rng"])
    return state["progress"]


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import initialize_cpu_vector_math
    initialize_cpu_vector_math()
    if method not in SPECS:
        raise ValueError("Select an explicit complete CAR author profile")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("CAR requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("CAR epoch interruption must be a positive integer")
    profile = SPECS[method]
    cfg, (stop_batch, stop_selector) = options(config)
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("CAR supports CPU/CUDA")
    bundle = Bundle(data, fm, split)
    inventory = validate_bundle(bundle, profile)
    torch.set_num_threads(cfg["threads"])
    author = definitions()
    author.utils.seed_torch(seed)
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg, "method_spec": profile,
        "software": software_identity(), "numerical_identity": numerical_identity(),
        "runtime": runtime_identity(),
        "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"], "source_contract": contract(),
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    out = Path(output).expanduser().resolve()
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("CAR recovery changed data/software/numerics/source/initial weights/recipe")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("CAR run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        author.utils.seed_torch(seed)
        model, criterion, optimizer, scheduler, early, warm_start = construct(
            bundle, profile, author, dev, cfg["initial_checkpoint"])
        train_data, val_data = SlideBags(bundle, "train"), SlideBags(bundle, "val")
        generator = torch.Generator().manual_seed(RECIPE["fixed_loader_seed"])
        author.utils.seed_torch(seed)  # original fix_train_random, after model initialization
        progress = {"epoch": 0, "phase": "validate", "cursor": 0, "order": None,
                    "sampler_exhausted": False, "updates": 0, "selectors": 0,
                    "best_auc": 0., "history": [], "author_stopped": False}
        write_json(out / "method_protocol.json", {
            "method": method, "recipe": profile["native_recipe"], "source": request["source_contract"],
            "input_dimension": bundle.dim, "hidden_dimension": 512, "parameters": sum(p.numel() for p in model.parameters()),
            "selection": "pre-TRAIN VAL AUC including epoch0, strict improvement over0",
            "no_post_final_training_selector": True, "warm_start": warm_start,
            "private_loader_seed": RECIPE["fixed_loader_seed"],
            "native_attention_shape": profile["native_attention_shape"], "repairs": profile["repairs"],
        })
        if resume:
            progress = restore(torch.load(out / "last.pt", map_location=dev, weights_only=False),
                               model, optimizer, scheduler, early, generator, request)
        args = arguments(profile)
        while progress["epoch"] < 100 and not progress["author_stopped"]:
            if progress["phase"] == "validate":
                val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
                result = author.training.val_loop(
                    args, model, val_loader, dev, criterion, early, progress["epoch"])
                stop, accuracy, auc, precision, recall, f1, val_loss = result
                if not all(math.isfinite(float(v)) for v in [accuracy, auc, precision, recall, f1, val_loss]):
                    raise FloatingPointError("CAR original binary selector became nonfinite")
                improved = auc > progress["best_auc"]
                if improved:
                    progress["best_auc"] = auc
                progress["selectors"] += 1
                progress["history"].append({
                    "epoch_zero_based": progress["epoch"], "val_auc": auc, "val_loss": float(val_loss),
                    "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                    "improved": improved, "train_order": None, "train_trace": [],
                })
                progress["phase"], progress["author_stopped"] = "train", bool(stop)
                state = snapshot(model, optimizer, scheduler, early, generator, progress, request)
                if improved:
                    atomic_torch_save(copy.deepcopy(state), out / "best.pt")
                atomic_torch_save(state, out / "last.pt")
                if stop_selector is not None and progress["selectors"] >= stop_selector:
                    write_json(out / "status.json", {
                        "state": "interrupted", "backend": BACKEND,
                        "completed_selectors": progress["selectors"], "reason": "requested pre-TRAIN selector boundary",
                    })
                    return out
                if stop:
                    break
            recovering = progress["order"] is not None
            if progress["cursor"] < len(train_data):
                sampler = (RemainingOrder(progress["order"][progress["cursor"]:])
                           if recovering else RecordingShuffle(train_data, generator=generator))
                loader = DataLoader(train_data, batch_size=1, sampler=sampler,
                                    generator=generator, num_workers=0)
                before_iter = generator.get_state() if recovering else None
                iterator = iter(loader)
                if recovering:
                    generator.set_state(before_iter)
                for batch in iterator:
                    values = train_step(model, criterion, optimizer, args, batch, dev)
                    if progress["order"] is None:
                        progress["order"] = list(sampler.order)
                        progress["history"][-1]["train_order"] = list(sampler.order)
                    index = progress["order"][progress["cursor"]]
                    progress["history"][-1]["train_trace"].append({
                        "slide_id": train_data.rows[index]["slide_id"], **values,
                    })
                    progress["cursor"] += 1
                    progress["updates"] += 1
                    atomic_torch_save(snapshot(model, optimizer, scheduler, early, generator, progress, request),
                                      out / "last.pt")
                    if stop_batch is not None and progress["updates"] >= stop_batch:
                        write_json(out / "status.json", {
                            "state": "interrupted", "backend": BACKEND,
                            "completed_train_batches": progress["updates"], "reason": "requested TRAIN batch boundary",
                        })
                        return out
                if not recovering:
                    # Native RandomSampler has now consumed its trailing
                    # empty-slice permutation, after yielding the last batch.
                    progress["sampler_exhausted"] = True
            if not progress["sampler_exhausted"]:
                # Resume uses recorded order and must still consume the
                # original RandomSampler's trailing randperm exactly once.
                torch.randperm(len(train_data), generator=generator)
                progress["sampler_exhausted"] = True
            assert progress["cursor"] == len(train_data)
            scheduler.step()
            progress.update(epoch=progress["epoch"] + 1, phase="validate", cursor=0,
                            order=None, sampler_exhausted=False)
            atomic_torch_save(snapshot(model, optimizer, scheduler, early, generator, progress, request),
                              out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"epoch={progress['epoch']} pretrain_val_auc={progress['history'][-1]['val_auc']:.8f}", flush=True)
            if stop_after is not None and progress["epoch"] >= stop_after and progress["epoch"] < 100:
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND, "completed_epochs": progress["epoch"],
                    "reason": "requested full-epoch boundary",
                })
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("Original CAR strict AUC>0 selector did not create selected weights")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND, "completed_epochs": progress["epoch"],
            "maximum_epoch_budget": 100, "completed_train_batches": progress["updates"],
            "completed_selectors": progress["selectors"], "test_evaluated": False,
            "termination": "author_early_stop" if progress["author_stopped"] else "complete_author_epoch_budget",
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
    from .engine import score
    out = Path(run).expanduser().resolve()
    request = read_json(out / "request.json")
    if subset not in {"val", "test"} or read_json(out / "status.json")["state"] != "completed":
        raise ValueError("CAR selected evaluation requires complete training and val/test subset")
    torch.set_num_threads(request["config"]["threads"])
    author = definitions()
    author.utils.seed_torch(request["seed"])
    if (request["software"] != software_identity() or request["device"] != str(torch.device(device))
            or request["numerical_identity"] != numerical_identity()
            or request["runtime"] != runtime_identity()):
        raise ValueError("CAR evaluation software/device/numerics changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = validate_bundle(bundle, SPECS[request["method"]])
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]
            or contract() != request["source_contract"]):
        raise ValueError("CAR evaluation changed data/features/source")
    initial = request["config"]["initial_checkpoint"]
    if initial:
        current, _ = options({"model": {"initial_checkpoint": initial["path"]}})
        if current["initial_checkpoint"] != initial:
            raise ValueError("CAR selected evaluation initial checkpoint changed")
    model, _, optimizer, scheduler, early, _ = construct(
        bundle, SPECS[request["method"]], author, torch.device(device), request["config"]["initial_checkpoint"])
    generator = torch.Generator()
    restore(torch.load(out / "best.pt", map_location=device, weights_only=False),
            model, optimizer, scheduler, early, generator, request)
    model.eval()
    records = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row)
            result = model(features.unsqueeze(0).to(device))
            probabilities = torch.softmax(result[0][0], dim=-1).cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("CAR selected probabilities became nonfinite")
            records.append({
                "slide_id": row["slide_id"], "patient_id": row["patient_id"],
                "global_patient_id": row["global_patient_id"], "subset": subset,
                "source_slides": row["slide_id"], "target": int(row["target"]),
                "p_0": float(probabilities[0]), "p_1": float(probabilities[1]),
            })
    frame = pd.DataFrame(records)
    metrics = score(bundle, frame)
    frame.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", metrics)
    return metrics

"""Complete AttriMIL author SGD, weighted sampler, epoch queues and recovery."""

import copy
import math
from pathlib import Path
import queue

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .attrimil_data import RecordingWeightedSampler, SlideBags, neighbor_step, single_bag
from .attrimil_source import contract, definitions
from .attrimil_spec import PROFILES, RECIPE
from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .pneamil_driver import RemainingOrder, own_lock
from .provenance import software_identity
from .spatial_attention import checked_points

BACKEND = "attrimil_complete_author_recipe_repaired"
SPECS = {p["id"]: p for p in PROFILES}


def options(profile, config):
    config = config or {}
    if not isinstance(config, dict) or set(config) - {"threads", "epochs", "model"}:
        raise ValueError("AttriMIL preserves the complete author recipe; only runtime/interruption options are accepted")
    if "epochs" in config and config["epochs"] != RECIPE["epochs"]:
        raise ValueError("AttriMIL original maximum200epoch budget cannot be reduced; use stop_after")
    model = config.get("model", {})
    if not isinstance(model, dict) or set(model) - {"stop_after_train_batches"}:
        raise ValueError("AttriMIL preserves native widths, SGD, weighted sampling, spatial1/rank5 and queues4")
    threads = config.get("threads", 2)
    stop_batch = model.get("stop_after_train_batches")
    if type(threads) is not int or threads < 1:
        raise ValueError("AttriMIL threads must be a positive integer")
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError("AttriMIL minibatch interruption must be a positive integer")
    return {"threads": threads, "recipe": dict(profile["native_recipe"])}, stop_batch


def numerical_identity():
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "amp": False, "feature_dtype": "float32",
    }


def check_bundle(bundle):
    if (bundle.task_type != "classification"
            or bundle.config.get("training_unit", "slide") != "slide"
            or bundle.dim < 2 or len(bundle.classes) < 2):
        raise ValueError("AttriMIL author recipes require individual slide classification bags")
    neighbor_step(bundle)
    inventory = bundle.check(deep=True, require_coords=True)
    # Replacement sampling can omit a particular TRAIN slide in an epoch.
    # Invalid geometry must still fail at admission, even if not drawn yet.
    for subset in ("train", "val", "test"):
        for row in bundle.records(subset):
            _, coords = bundle.load(row, require_coords=True)
            checked_points(coords)
    return inventory


def new_queues(classes):
    return ([queue.Queue(maxsize=4) for _ in range(classes)],
            [queue.Queue(maxsize=4) for _ in range(classes)])


def snapshot_queues(queues):
    return [[[value.detach().cpu().clone() for value in q.queue] for q in side]
            for side in queues]


def restore_queues(state, classes, device):
    result = new_queues(classes)
    if state is None:
        return result
    if len(state) != 2 or any(len(side) != classes for side in state):
        raise ValueError("AttriMIL saved queue/class layout changed")
    for destinations, saved in zip(result, state):
        for q, values in zip(destinations, saved):
            if len(values) > q.maxsize:
                raise ValueError("AttriMIL saved queue exceeds original capacity4")
            for value in values:
                q.put(value.to(device))
    return result


def make_model(bundle, device):
    author = definitions(device)
    model = author[0].AttriMIL(n_classes=len(bundle.classes), dim=bundle.dim).to(device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, momentum=.9, weight_decay=1e-5,
    )
    return model, optimizer, author


def train_step(model, optimizer, constraints, queues, batch, classes, device):
    model.train()
    features, target, _, nearest = batch
    features, target = features.to(device), target.to(device)
    logits, _, _, attributes, _ = model(features)
    ce = nn.functional.cross_entropy(logits, target)
    spatial = constraints.spatial_constraint(attributes, classes, nearest, ks=3)
    rank, positive, negative = constraints.rank_constraint(
        features, target, model, attributes, classes, *queues)
    if positive is not queues[0] or negative is not queues[1]:
        raise ValueError("Original AttriMIL unexpectedly replaced its queue containers")
    loss = ce + 1.0 * spatial + 5.0 * rank
    if not torch.isfinite(loss):
        raise FloatingPointError("AttriMIL original objective became nonfinite")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("AttriMIL original gradient became nonfinite")
    optimizer.step()
    # Original train_loop clears gradients AFTER each step, without clipping.
    optimizer.zero_grad()
    return {"loss": loss.item(), "ce": ce.item(), "spatial": spatial.item(),
            "rank": rank.item(), "patch_count": len(features)}


def checkpoint(model, optimizer, progress, request):
    from .engine import rng_state
    return {
        "schema_version": 1, "request_sha256": digest_object(request),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "progress": progress, "rng": rng_state(),
    }


def restore(model, optimizer, state, request):
    from .engine import restore_rng
    if state["request_sha256"] != digest_object(request):
        raise ValueError("AttriMIL checkpoint belongs to another data/software/geometry/recipe request")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["progress"]


def stop_under_author_rule(profile, epoch_zero_based, retain):
    return profile["native_recipe"]["early_stopping"] and retain > 20 and epoch_zero_based > 50


def train(data, method, output, *, fm=None, split=None, seed=2026, device="cpu",
          config=None, resume=False, stop_after=None):
    from .engine import restore_rng, rng_state, seed_everything
    if method not in SPECS:
        raise ValueError("Choose an explicit complete AttriMIL author profile")
    profile = SPECS[method]
    cfg, stop_batch = options(profile, config)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("AttriMIL requires a32bit integer seed")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("AttriMIL epoch interruption must be a positive integer")
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("AttriMIL author reference supports CPU and CUDA")
    bundle = Bundle(data, fm, split)
    inventory = check_bundle(bundle)
    torch.set_num_threads(cfg["threads"])
    seed_everything(seed)
    request = {
        "schema_version": 1, "backend": BACKEND, "method": method,
        "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
        "seed": seed, "device": str(dev), "config": cfg,
        "software": software_identity(), "numerical_identity": numerical_identity(),
        "bundle_sha256": bundle.fingerprint,
        "features_sha256": inventory["feature_inventory_sha256"],
        "source_contract": contract(), "method_spec": profile,
        "geometry": {"spatial": bundle.encoder["spatial"], "neighbor_step": neighbor_step(bundle)},
        "synthetic": bool(bundle.config.get("synthetic", False)),
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    out = Path(output).expanduser().resolve()
    if resume:
        if read_json(out / "request.json") != request:
            raise ValueError("AttriMIL recovery changed inputs, source, geometry, numerics or full recipe")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("AttriMIL run already completed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / "RUNNING.lock"
    own_lock(lock)
    try:
        write_json(out / "status.json", {"state": "running", "backend": BACKEND})
        seed_everything(seed)
        model, optimizer, author = make_model(bundle, dev)
        train_data, val_data = SlideBags(bundle, "train"), SlideBags(bundle, "val")
        if any(len(indices) == 0 for indices in train_data.slide_cls_ids):
            raise ValueError("Original AttriMIL balanced sampler requires every class in TRAIN")
        weights = author[2].make_weights_for_balanced_classes_split(train_data)
        write_json(out / "method_protocol.json", {
            "method": method, "recipe": cfg["recipe"], "source": request["source_contract"],
            "input_dimension": bundle.dim, "attention_dimension": bundle.dim // 2,
            "parameters": sum(p.numel() for p in model.parameters()),
            "geometry": request["geometry"],
            "neighbor_order": ["self", "left", "right", "up", "down",
                               "left_up", "left_down", "right_up", "right_down"],
            "missing_neighbor": "self index, exactly as original create_3coords.py",
            "spatial_classes": "indices1..C-1; original class0 excluded",
            "training_weights": weights.tolist(),
            "selection": "strict minimum full-VAL CE, initial10000",
            "original_summary_repair": "native5-output probabilities with fresh selected weights",
            "repairs": profile["repairs"],
            "checkpoint_boundaries": "every actual weighted TRAIN batch and VAL selector",
            "source_row_order": "task manifest sorted slide_id; patient-disjoint fixed splits",
            "runtime_workers": "0; original uses0 on CPU and4 on CUDA",
        })
        progress = {
            "epoch": 0, "cursor": 0, "order": None, "updates": 0, "queues": None,
            "train_trace": [], "history": [], "best_loss": 10000., "retain": 0,
            "author_stopped": False,
        }
        if resume:
            progress = restore(model, optimizer, torch.load(
                out / "last.pt", map_location=dev, weights_only=False), request)
        while progress["epoch"] < RECIPE["epochs"] and not progress["author_stopped"]:
            if progress["cursor"] < len(train_data):
                recovering = progress["order"] is not None
                queues = restore_queues(progress["queues"], len(bundle.classes), dev)
                sampler = (RemainingOrder(progress["order"][progress["cursor"]:])
                           if recovering else RecordingWeightedSampler(weights, len(weights)))
                loader = DataLoader(train_data, batch_size=1, sampler=sampler,
                                    collate_fn=single_bag, num_workers=0)
                saved_rng = rng_state() if recovering else None
                iterator = iter(loader)
                if recovering:
                    restore_rng(saved_rng)
                for batch in iterator:
                    metrics = train_step(model, optimizer, author[1], queues,
                                         batch, len(bundle.classes), dev)
                    if progress["order"] is None:
                        progress["order"] = list(sampler.order)
                    index = progress["order"][progress["cursor"]]
                    progress["train_trace"].append({
                        "slide_id": train_data.rows[index]["slide_id"], **metrics,
                    })
                    progress["cursor"] += 1
                    progress["updates"] += 1
                    progress["queues"] = snapshot_queues(queues)
                    atomic_torch_save(checkpoint(model, optimizer, progress, request), out / "last.pt")
                    if stop_batch is not None and progress["updates"] >= stop_batch:
                        write_json(out / "status.json", {
                            "state": "interrupted", "backend": BACKEND,
                            "completed_train_batches": progress["updates"],
                            "reason": "requested actual weighted TRAIN batch boundary",
                        })
                        return out
            if progress["cursor"] != len(train_data):
                raise ValueError("AttriMIL epoch did not consume the original number of weighted draws")
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                                    collate_fn=single_bag, num_workers=0)
            val_loss = author[3].validate(
                progress["epoch"], model, val_loader, len(bundle.classes), None, nn.CrossEntropyLoss())
            if not math.isfinite(val_loss):
                raise FloatingPointError("AttriMIL original VAL CE selector became nonfinite")
            improved = val_loss < progress["best_loss"]
            if improved:
                progress["best_loss"], progress["retain"] = val_loss, 0
            else:
                progress["retain"] += 1
            record = {
                "epoch": progress["epoch"] + 1, "val_loss": val_loss, "improved": improved,
                "retain": progress["retain"], "train_order": progress["order"],
                "train_trace": progress["train_trace"], "total_updates": progress["updates"],
                "epoch_final_queues": progress["queues"],
            }
            # Tensor-valued queues belong in checkpoints; JSON history records
            # their counts, and exact bytes stay available to source/wheel audits.
            record["epoch_final_queue_sizes"] = [[len(q) for q in side] for side in progress["queues"]]
            record.pop("epoch_final_queues")
            progress["history"].append(record)
            progress["author_stopped"] = stop_under_author_rule(
                profile, progress["epoch"], progress["retain"])
            progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None,
                            queues=None, train_trace=[])
            state = checkpoint(model, optimizer, progress, request)
            if improved:
                atomic_torch_save(copy.deepcopy(state), out / "best.pt")
            atomic_torch_save(state, out / "last.pt")
            write_json(out / "history.json", progress["history"])
            print(f"epoch={progress['epoch']} val_ce={val_loss:.8f} retain={progress['retain']}", flush=True)
            if (stop_after is not None and progress["epoch"] >= stop_after
                    and progress["epoch"] < RECIPE["epochs"] and not progress["author_stopped"]):
                write_json(out / "status.json", {
                    "state": "interrupted", "backend": BACKEND,
                    "completed_epochs": progress["epoch"], "completed_train_batches": progress["updates"],
                    "reason": "requested epoch boundary",
                })
                return out
        if not (out / "best.pt").is_file():
            raise ValueError("Original AttriMIL strict selector did not create best weights")
        write_json(out / "status.json", {
            "state": "completed", "backend": BACKEND,
            "completed_epochs": progress["epoch"], "completed_train_batches": progress["updates"],
            "maximum_epoch_budget": 200,
            "termination": "author_early_stop" if progress["author_stopped"] else "complete_author_epoch_budget",
            "test_evaluated": False,
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
        raise ValueError("AttriMIL evaluation requires complete training and val/test subset")
    torch.set_num_threads(request["config"]["threads"])
    seed_everything(request["seed"])
    if (request["software"] != software_identity()
            or request["device"] != str(torch.device(device))
            or request["numerical_identity"] != numerical_identity()):
        raise ValueError("AttriMIL selected evaluation software/device/numerics changed")
    bundle = Bundle(request["data"], request["fm"], request["split"])
    inventory = check_bundle(bundle)
    if (bundle.fingerprint != request["bundle_sha256"]
            or inventory["feature_inventory_sha256"] != request["features_sha256"]
            or contract() != request["source_contract"]):
        raise ValueError("AttriMIL selected evaluation changed source, features, coordinates or patient splits")
    model, optimizer, _ = make_model(bundle, torch.device(device))
    selected = torch.load(out / "best.pt", map_location=device, weights_only=False)
    restore(model, optimizer, selected, request)
    model.eval()
    records = []
    with torch.no_grad():
        for row in bundle.records(subset):
            features, _ = bundle.load(row, require_coords=True)
            _, probabilities, _, _, _ = model(features.to(device))
            probabilities = probabilities[0].cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("AttriMIL selected probabilities became nonfinite")
            records.append({
                "slide_id": row["slide_id"], "patient_id": row["patient_id"],
                "global_patient_id": row["global_patient_id"], "subset": subset,
                "source_slides": row["slide_id"], "target": int(row["target"]),
                **{f"p_{i}": float(p) for i, p in enumerate(probabilities)},
            })
    frame = pd.DataFrame(records)
    result = score(bundle, frame)
    frame.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", result)
    return result

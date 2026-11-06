from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Full50/100 author trajectories, patient/test boundary and complete recovery."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch
import yaml
from torch.utils.data import DataLoader

from milplus import engine
from milplus.asmil_driver import SlideBags, options
from milplus.asmil_spec import DIMENSIONS, PROFILES
from milplus.data import Bundle
from milplus.io import read_json
from asmil_helpers import equal_tree, literal_source, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def complete_reference(task, profile, config, seed=2026):
    """Use literal original train_one_epoch/evaluate for every configured epoch.

    The original per-epoch TEST call is deliberately excluded under the named
    locked-test repair. Only the finite-EMA variant changes one source condition.
    """
    bundle = Bundle(task)
    cfg, _ = options(profile, config)
    recipe = cfg["recipe"]
    source, original = literal_source(
        repair_teacher=profile["teacher_update"] == "finite_ema_repaired")
    conf = SimpleNamespace(
        D_feat=bundle.dim, D_inner=DIMENSIONS[bundle.dim], n_class=len(bundle.classes),
        train_epoch=recipe["epochs"], warmup_epoch=0, min_lr=0,
        lr=recipe["lr"], wd=recipe["weight_decay"], wandb_mode="disabled",
    )
    torch.set_num_threads(cfg["threads"])
    engine.seed_everything(seed)
    model = source.ACMIL_MYMHA(conf, n_token=recipe["n_token"], n_drop=recipe["n_drop"])
    teacher = source.ACMIL_MYMHA_EMA(conf, n_token=recipe["n_token"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    # Literal training functions receive the same original-style dict bags,
    # full-patch tensors, shuffle=True and batch1/drop_last=True.
    loader = DataLoader(SlideBags(bundle, "train"), batch_size=1, shuffle=True,
                        drop_last=True, num_workers=0)
    val_loader = DataLoader(SlideBags(bundle, "val"), batch_size=1, shuffle=False,
                            drop_last=False, num_workers=0)
    history, best_score, best = [], 0., None
    for epoch in range(recipe["epochs"]):
        original.train_one_epoch(model, teacher, torch.nn.CrossEntropyLoss(), loader,
                                 optimizer, "cpu", epoch, conf, recipe["temp"])
        auc, acc, f1, loss = original.evaluate(
            model, torch.nn.CrossEntropyLoss(), val_loader, "cpu", conf, "Val")
        selector = auc + f1
        improved = selector > best_score
        state = {
            "model": copy.deepcopy(model.state_dict()),
            "teacher": copy.deepcopy(teacher.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": engine.rng_state(),
        }
        if improved:
            best, best_score = copy.deepcopy(state), selector
        history.append({
            "epoch": epoch + 1, "auc": auc, "f1": f1, "loss": loss,
            "selector": selector, "improved": improved,
        })
    return {**state, "history": history, "best": best}


@pytest.fixture(scope="module", params=[p["id"] for p in PROFILES])
def full_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    profile = next(p for p in PROFILES if p["id"] == request.param)
    # Keep the source's default native384->128 for all full budgets. Separate
    # tests exercise all five original dimensions and longer TRAIN bags.
    task = make_task(root / "task", classes=3, train_slides=4)
    config = {"model": {"n_token": 3, "n_drop": 1, "temp": .5}}
    output = engine.train(task, profile["id"], root / "run", config=config)
    reference = complete_reference(task, profile, config)
    torch.save(reference, root / "independent_literal_reference.pt")
    return task, output, profile, config, reference


def test_complete_original_budget_model_teacher_optimizer_RNG_history_and_best_snapshot(full_run):
    _, output, profile, _, reference = full_run
    state = load(output / "last.pt")
    for key in ["model", "teacher", "optimizer", "rng"]:
        equal_tree(state[key], reference[key])
    budget = profile["native_recipe"]["epochs"]
    status = read_json(output / "status.json")
    assert status["completed_epochs"] == budget
    assert status["completed_train_batches"] == 4 * budget
    assert status["test_evaluated"] is False
    assert status["termination"] == "complete_author_epoch_budget"
    assert len(state["progress"]["history"]) == len(reference["history"]) == budget
    for actual, expected in zip(state["progress"]["history"], reference["history"]):
        assert actual["selection_score"] == expected["selector"]
        assert actual["improved"] == expected["improved"]
        assert actual["validation"]["loss"] == expected["loss"]
        assert actual["validation"]["auc"] == expected["auc"]
        assert actual["validation"]["f1"] == expected["f1"]
        assert len(actual["train_trace"]) == 4
    best = load(output / "best.pt")
    for key in ["model", "teacher", "optimizer", "rng"]:
        equal_tree(best[key], reference["best"][key])
    assert read_json(output / "request.json")["synthetic"] is True


@pytest.mark.parametrize("boundary", ["midbatch", "lastbatch_before_selector", "epoch"])
def test_exact_resume_full_budget_including_teacher_dropout_and_fractional_schedule(full_run, boundary):
    task, output, profile, config, _ = full_run
    resumed = output.parent / ("recovered_" + boundary)
    stopping = copy.deepcopy(config)
    stop_after = 1 if boundary == "epoch" else None
    if boundary != "epoch":
        stopping["model"]["stop_after_train_batches"] = 1 if boundary == "midbatch" else 4
    engine.train(task, profile["id"], resumed, config=stopping, stop_after=stop_after)
    assert read_json(resumed / "status.json")["state"] == "interrupted"
    checkpoint = load(resumed / "last.pt")
    assert checkpoint["progress"]["cursor"] == {
        "midbatch": 1, "lastbatch_before_selector": 4, "epoch": 0,
    }[boundary]
    # Interruption controls are excluded from the bound recipe; remove them
    # when continuing the same exact requested complete training.
    engine.train(task, profile["id"], resumed, config=config, resume=True)
    equal_tree(load(resumed / "last.pt"), load(output / "last.pt"))
    equal_tree(load(resumed / "best.pt"), load(output / "best.pt"))
    assert read_json(resumed / "history.json") == read_json(output / "history.json")


def test_two_fresh_CLI_selected_reloads_are_deterministic_with_eval_randperm(full_run):
    _, output, _, _, _ = full_run
    command = [sys.executable, "-m", "milplus", "evaluate",
               "--run", str(output), "--subset", "test", "--device", "cpu"]
    env = dict(os.environ)
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    raw = (output / "test_predictions.csv").read_bytes()
    metrics = read_json(output / "test_metrics.json")
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert raw == (output / "test_predictions.csv").read_bytes()
    assert read_json(output / "test_metrics.json") == metrics
    frame = pd.read_csv(output / "test_predictions.csv")
    assert len(frame) == 3 and set(frame["subset"]) == {"test"}
    assert frame["patient_id"].nunique() == 3


def test_public_run_CLI_accepts_data_method_config_and_completes_same_recovered_recipe(full_run):
    task, output, profile, config, _ = full_run
    config_file = output.parent / "public_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    public = output.parent / "public_cli_run"
    method = "ASMIL" if profile is PROFILES[0] else profile["id"]
    command = [
        sys.executable, "-m", "milplus", "run", "--data", str(task),
        "--method", method, "--output", str(public), "--config", str(config_file),
        "--device", "cpu", "--seed", "2026",
    ]
    interrupted = subprocess.run(command + ["--stop-after", "1"], capture_output=True, text=True)
    assert interrupted.returncode == 0, interrupted.stdout + interrupted.stderr
    assert read_json(public / "status.json")["state"] == "interrupted"
    completed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert read_json(public / "status.json")["state"] == "completed"
    equal_tree(load(public / "last.pt"), load(output / "last.pt"))
    equal_tree(load(public / "best.pt"), load(output / "best.pt"))
    (public / "public_cli_interrupted.log").write_text(interrupted.stdout + interrupted.stderr)
    (public / "public_cli_completed.log").write_text(completed.stdout + completed.stderr)


def test_train_never_reads_test_and_rejects_data_or_recipe_changes(tmp_path, monkeypatch):
    profile = PROFILES[0]["id"]
    task = make_task(tmp_path / "task")
    accessed = []
    original_dataset = SlideBags.__init__

    def recorded(self, bundle, subset):
        accessed.append(subset)
        assert subset != "test"
        original_dataset(self, bundle, subset)

    monkeypatch.setattr(SlideBags, "__init__", recorded)
    output = engine.train(task, profile, tmp_path / "run", stop_after=1)
    assert accessed == ["train", "val"]
    assert not list(output.glob("test_*"))
    with pytest.raises(ValueError, match="changed"):
        engine.train(task, profile, output, resume=True, config={"model": {"temp": .5}})
    bundle = Bundle(task)
    path = bundle.root / bundle.records("train")[0]["path"]
    values = load(path)
    values[0, 0] += 1
    torch.save(values, path)
    with pytest.raises(ValueError, match="changed"):
        engine.train(task, profile, output, resume=True)


def test_catalog_generation_preserves_all354_profile_records_and_every_original_family(tmp_path):
    from milplus.registry import methods, profiles
    before = copy.deepcopy(profiles())
    from milplus.attrimil_spec import PROFILES as new_profiles
    from milplus.carmil_spec import PROFILES as carmil_profiles
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in new_profiles + carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 358
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 360
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(before) == 415 and len(before[:394]) == 394
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert profiles() == before
    assert len(methods()) == 164
    assert sum(bool(m["profiles"]) for m in methods()) == 164

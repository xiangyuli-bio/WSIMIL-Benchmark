from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Full native author trajectories, weighted/queue recovery and public CLI."""

import copy
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch
import yaml

from milplus import engine
from milplus.attrimil_driver import SlideBags
from milplus.attrimil_spec import DEFAULT_PROFILE, PROFILES
from milplus.data import Bundle
from milplus.io import read_json
from attrimil_helpers import equal_tree, literal_datasets, literal_sources, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def complete_reference(task, profile):
    source, _, utils, datasets, trainer = literal_sources()
    original = literal_datasets(task, datasets)

    class TracedSplit(datasets.Generic_Split):
        def __getitem__(self, index):
            self.access.append(index)
            return super().__getitem__(index)

    train_data, val_data = original["train"], original["val"]
    train_data.__class__ = TracedSplit
    train_data.access = []
    bundle = Bundle(task)
    engine.seed_everything(2026)
    model = source.AttriMIL(dim=bundle.dim, n_classes=len(bundle.classes))
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=.9, weight_decay=1e-5)
    train_loader = utils.get_split_loader(train_data, training=True, weighted=True)
    val_loader = utils.get_split_loader(val_data)
    best_loss, retain, best, stopped, history = 10000., 0, None, False, []
    for epoch in range(200):
        train_data.access = []
        trainer.train_loop(epoch, model, train_loader, optimizer, len(bundle.classes),
                           None, torch.nn.CrossEntropyLoss())
        loss = trainer.validate(epoch, model, val_loader, len(bundle.classes),
                                None, torch.nn.CrossEntropyLoss())
        improved = loss < best_loss
        if improved:
            best_loss, retain = loss, 0
        else:
            retain += 1
        state = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": engine.rng_state(),
        }
        if improved:
            best = copy.deepcopy(state)
        history.append({
            "epoch": epoch + 1, "val_loss": loss, "improved": improved,
            "retain": retain, "train_order": train_data.access,
        })
        # Literal original source condition, with zero-based epoch.
        if profile["native_recipe"]["early_stopping"]:
            if retain > 20 and epoch > 50:
                stopped = True
                break
    return {**state, "history": history, "best": best, "author_stopped": stopped}


@pytest.fixture(scope="module", params=[p["id"] for p in PROFILES])
def full_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    profile = next(p for p in PROFILES if p["id"] == request.param)
    task = make_task(root / "task", zero=profile["native_recipe"]["early_stopping"])
    output = engine.train(task, profile["id"], root / "run")
    reference = complete_reference(task, profile)
    torch.save(reference, root / "independent_complete_original_reference.pt")
    return task, output, profile, reference


def test_complete_native512_author_model_SGD_RNG_best_and_original_sampler_history(full_run):
    _, output, profile, reference = full_run
    state, selected = load(output / "last.pt"), load(output / "best.pt")
    for key in ["model", "optimizer", "rng"]:
        equal_tree(state[key], reference[key])
        equal_tree(selected[key], reference["best"][key])
    history = state["progress"]["history"]
    assert len(history) == len(reference["history"])
    for actual, expected in zip(history, reference["history"]):
        for key in ["epoch", "val_loss", "improved", "retain", "train_order"]:
            assert actual[key] == expected[key], (key, actual["epoch"])
    status = read_json(output / "status.json")
    assert status["completed_epochs"] == len(reference["history"])
    assert status["completed_train_batches"] == 5 * len(reference["history"])
    assert status["maximum_epoch_budget"] == 200 and status["test_evaluated"] is False
    if not profile["native_recipe"]["early_stopping"]:
        assert status["completed_epochs"] == 200
        assert status["termination"] == "complete_author_epoch_budget"
    else:
        assert state["progress"]["author_stopped"] == reference["author_stopped"]
    assert read_json(output / "request.json")["synthetic"] is True
    protocol = read_json(output / "method_protocol.json")
    assert protocol["input_dimension"] == 512 and protocol["attention_dimension"] == 256
    assert any(len(set(h["train_order"])) < 5 for h in history)


@pytest.mark.parametrize("boundary", ["midbatch", "lastbatch_before_selector", "epoch"])
def test_exact_full_recipe_recovery_including_all_queues_weighted_draws_and_SGD(full_run, boundary):
    task, output, profile, _ = full_run
    resumed = output.parent / ("recovered_" + boundary)
    config = ({} if boundary == "epoch" else
              {"model": {"stop_after_train_batches": 2 if boundary == "midbatch" else 5}})
    engine.train(task, profile["id"], resumed, config=config,
                 stop_after=1 if boundary == "epoch" else None)
    assert read_json(resumed / "status.json")["state"] == "interrupted"
    saved = load(resumed / "last.pt")
    assert saved["progress"]["cursor"] == {
        "midbatch": 2, "lastbatch_before_selector": 5, "epoch": 0,
    }[boundary]
    if boundary != "epoch":
        assert saved["progress"]["queues"] is not None
        assert any(q for side in saved["progress"]["queues"] for q in side)
    else:
        assert saved["progress"]["queues"] is None
    engine.train(task, profile["id"], resumed, resume=True)
    equal_tree(load(resumed / "last.pt"), load(output / "last.pt"))
    equal_tree(load(resumed / "best.pt"), load(output / "best.pt"))
    assert read_json(resumed / "history.json") == read_json(output / "history.json")


def test_public_CLI_data_method_launch_resume_and_two_fresh_selected_predictions(full_run):
    task, output, profile, _ = full_run
    public = output.parent / "public_cli_run"
    method = "AttriMIL" if profile["id"] == DEFAULT_PROFILE else profile["id"]
    command = [
        sys.executable, "-m", "milplus", "run", "--data", str(task),
        "--method", method, "--output", str(public), "--device", "cpu",
    ]
    interrupted = subprocess.run(command + ["--stop-after", "1"], capture_output=True, text=True)
    assert interrupted.returncode == 0, interrupted.stdout + interrupted.stderr
    assert read_json(public / "status.json")["state"] == "interrupted"
    complete = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert complete.returncode == 0, complete.stdout + complete.stderr
    equal_tree(load(public / "last.pt"), load(output / "last.pt"))
    equal_tree(load(public / "best.pt"), load(output / "best.pt"))
    (public / "public_cli_interrupted.log").write_text(interrupted.stdout + interrupted.stderr)
    (public / "public_cli_completed.log").write_text(complete.stdout + complete.stderr)
    evaluation = [sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
                  "--subset", "test", "--device", "cpu"]
    first = subprocess.run(evaluation, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    predictions = (public / "test_predictions.csv").read_bytes()
    second = subprocess.run(evaluation, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (public / "test_predictions.csv").read_bytes() == predictions
    frame = pd.read_csv(public / "test_predictions.csv")
    assert len(frame) == frame["patient_id"].nunique() == 2
    assert set(frame["subset"]) == {"test"}
    # Direct entry and the public CLI both reload the actual best model.
    engine.evaluate(output, "test", "cpu")
    assert (output / "test_predictions.csv").read_bytes() == predictions


def test_training_datasets_exclude_TEST_and_coordinate_geometry_or_numerics_changes_are_rejected(
        tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    requested = []
    original = SlideBags.__init__

    def observed(self, bundle, subset):
        requested.append(subset)
        assert subset != "test"
        original(self, bundle, subset)

    monkeypatch.setattr(SlideBags, "__init__", observed)
    output = engine.train(task, DEFAULT_PROFILE, tmp_path / "run", stop_after=1)
    assert requested == ["train", "val"]
    assert not list(output.glob("test_*"))
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if precision != "medium" else "highest")
        with pytest.raises(ValueError, match="changed"):
            engine.train(task, DEFAULT_PROFILE, output, resume=True)
    finally:
        torch.set_float32_matmul_precision(precision)
    cfg_path = task / "task.yaml"
    config = yaml.safe_load(cfg_path.read_text())
    config["encoders"]["synthetic"]["attrimil"] = {"neighbor_step": [512, 256]}
    cfg_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="changed"):
        engine.train(task, DEFAULT_PROFILE, output, resume=True)


@pytest.mark.parametrize("mutation", ["missing_coords", "duplicate_coords", "missing_spatial"])
def test_no_pseudo_grid_or_order_based_neighbor_fallback(tmp_path, mutation):
    import numpy as np
    task = make_task(tmp_path / "task")
    if mutation == "missing_spatial":
        path = task / "task.yaml"
        config = yaml.safe_load(path.read_text())
        config["encoders"]["synthetic"].pop("spatial")
        path.write_text(yaml.safe_dump(config))
    else:
        index = task / "features.csv"
        frame = pd.read_csv(index)
        if mutation == "missing_coords":
            frame = frame.drop(columns="coords_path")
            frame.to_csv(index, index=False)
        else:
            coords = task / frame.iloc[0]["coords_path"]
            points = np.load(coords)
            points[0] = points[1]
            np.save(coords, points)
    with pytest.raises(ValueError):
        engine.train(task, DEFAULT_PROFILE, tmp_path / "rejected", stop_after=1)


def test_all358_prior_profiles_and_164_families_survive_regeneration(tmp_path):
    from milplus.registry import methods, profiles
    from milplus.attrimil_spec import PROFILES as new_profiles
    before = copy.deepcopy(profiles())
    from milplus.carmil_spec import PROFILES as carmil_profiles
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 360
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(before) == 415 and len(before[:394]) == 394
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in new_profiles + carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 358
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert profiles() == before
    assert len(methods()) == 164 and sum(bool(m["profiles"]) for m in methods()) == 164

"""Complete author budgets, exact resume and independent selected-weight reload."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from milplus import engine
from milplus.data import Bundle
from milplus.io import read_json
from milplus.pneamil_driver import evaluate, train
from milplus.pneamil_spec import RECIPE, SOURCE_PROFILE, VAL_PROFILE
from milplus.upstream import source_root
from pneamil_helpers import equal_tree, literal_sources, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def independent_complete_reference(task, population, *, stop_epochs=None):
    """Execute unchanged author Dataset/trainer/evaluator with original defaults.

    Fixed patient partitions and immutable best snapshots are the two explicitly
    named repairs. No training definition, sampler, optimizer or budget changes.
    """
    bundle = Bundle(task)
    model_source, trainer = literal_sources()
    datasets = {}
    for subset in ["train", "val"]:
        rows = bundle.records(subset)
        csv = task / f"literal_{subset}.csv"
        pd.DataFrame([{
            "file": str(task / row["path"]), "label": row["target"],
        } for row in rows]).to_csv(csv, index=False)
        datasets[subset] = trainer.SlideDataset(
            csv, num_patches=1024 if subset == "train" else None)
    train_loader = DataLoader(datasets["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=1, shuffle=False)
    torch.set_num_threads(2)
    engine.seed_everything(2026)
    model = model_source.PNEA_MIL(D_in=bundle.dim, C=len(bundle.classes), seed=2026)
    history, best_loss, bad, best = [], 1e20, 0, None
    for epoch in range(1, 1001):
        training_loss = trainer.train_one_epoch(model, train_loader, torch.device("cpu"))
        selection_loss = trainer.evaluate_loss(
            model, train_loader if population == "train" else val_loader, torch.device("cpu"))
        improved = selection_loss < best_loss - abs(best_loss) * .0005
        if improved:
            best_loss, bad = selection_loss, 0
            best = copy.deepcopy({
                "model": model.state_dict(), "optimizer": model.optim.state_dict(),
                "rng": engine.rng_state(),
            })
        else:
            bad += 1
        history.append({
            "epoch": epoch, "train_loss": training_loss, "selection_loss": selection_loss,
            "improved": improved, "bad_epochs": bad,
        })
        if bad >= 10 or (stop_epochs is not None and epoch >= stop_epochs):
            break
    return {
        "history": history, "best": best, "best_loss": best_loss, "bad": bad,
        "model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(model.optim.state_dict()),
        "rng": engine.rng_state(),
    }


@pytest.fixture(scope="module", params=[SOURCE_PROFILE, VAL_PROFILE])
def complete_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    task = make_task(root / "task", zero=True)
    # A nonseparable synthetic population exercises the complete configured
    # 1000-epoch limit and/or literal patience10 without overriding the recipe.
    run = engine.train(task, request.param, root / "run", seed=2026)
    population = "train" if request.param == SOURCE_PROFILE else "val"
    reference = independent_complete_reference(task, population)
    return task, run, request.param, reference


def test_full_native1024_complete1000_maximum_and_original_patience_equal_unchanged_author_execution(complete_run):
    task, run, method, reference = complete_run
    status, protocol = read_json(run / "status.json"), read_json(run / "method_protocol.json")
    last, best = load(run / "last.pt"), load(run / "best.pt")
    assert status["state"] == "completed"
    assert status["maximum_epoch_budget"] == protocol["recipe"]["max_epochs"] == 1000
    assert protocol["input_dimension"] == 1024
    assert protocol["recipe"] == RECIPE
    assert protocol["recipe"]["batch_size"] == 8
    assert protocol["recipe"]["train_patches"] == 1024
    assert protocol["recipe"]["evidence_axes"] == 64
    assert status["test_evaluated"] is False
    assert last["progress"]["epoch"] == status["completed_epochs"] == len(reference["history"])
    assert last["progress"]["bad"] == reference["bad"]
    assert last["progress"]["best_loss"] == reference["best_loss"]
    assert 1 <= status["completed_epochs"] <= 1000
    assert status["completed_epochs"] == 1000 or reference["bad"] == 10
    assert status["completed_train_batches"] == 2 * status["completed_epochs"]
    assert len(last["progress"]["history"]) == len(reference["history"])
    for actual, expected in zip(last["progress"]["history"], reference["history"]):
        for key, value in expected.items():
            assert actual[key] == value, (method, actual["epoch"], key)
    for key in ["model", "optimizer", "rng"]:
        equal_tree(last[key], reference[key])
        equal_tree(best[key], reference["best"][key])
    assert best["progress"]["epoch"] <= last["progress"]["epoch"]
    # Snapshots represent the selected epoch, rather than a shallow reference
    # mutated by every subsequent optimizer update.
    assert any(not torch.equal(best["model"][name], value)
               for name, value in last["model"].items())
    assert not (run / "test_predictions.csv").exists()
    assert read_json(run / "request.json")["synthetic"] is True


def test_every_training_slide_and_full_validation_population_without_test_computation(complete_run):
    task, run, method, _ = complete_run
    bundle = Bundle(task)
    training = {row["slide_id"] for row in bundle.records("train")}
    validation = {row["slide_id"] for row in bundle.records("val")}
    expected_selector = training if method == SOURCE_PROFILE else validation
    for epoch in read_json(run / "history.json"):
        assert sorted(epoch["train_order"]) == list(range(10))
        assert len(epoch["train_trace"]) == 10
        assert {row["slide_id"] for row in epoch["train_trace"]} == training
        assert all(row["sampled_patches"] == 1024 for row in epoch["train_trace"])
        trace = epoch["selection_trace"]
        assert {row["slide_id"] for row in trace} == expected_selector
        if method == SOURCE_PROFILE:
            assert all(row["sampled_patches"] == 1024 for row in trace)
        else:
            assert all(row["all_original_patches"] for row in trace)
            assert max(row["sampled_patches"] for row in trace) == 1041


def test_selected_snapshot_fresh_process_reload_prediction_bytes_and_patient_metrics_are_identical(complete_run):
    _, run, _, _ = complete_run
    metrics = evaluate(run)
    first_predictions = (run / "test_predictions.csv").read_bytes()
    first_metrics = (run / "test_metrics.json").read_bytes()
    selected_before = (run / "best.pt").read_bytes()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "milplus", "evaluate", "--run", str(run)],
        check=True, capture_output=True, text=True, env=os.environ.copy(),
    )
    assert json.loads(result.stdout) == metrics
    assert (run / "test_predictions.csv").read_bytes() == first_predictions
    assert (run / "test_metrics.json").read_bytes() == first_metrics
    assert (run / "best.pt").read_bytes() == selected_before
    frame = pd.read_csv(run / "test_predictions.csv")
    assert len(frame) == 2 and frame.patient_id.nunique() == 2
    assert set(frame.subset) == {"test"}


@pytest.mark.parametrize("method", [SOURCE_PROFILE, VAL_PROFILE])
@pytest.mark.parametrize("boundary", ["first_minibatch", "last_minibatch_before_selector", "epoch"])
def test_actual_mid_minibatch_and_selector_boundary_recovery_exact_model_optimizer_RNG_and_traces(
        method, boundary, tmp_path):
    task = make_task(tmp_path / "task")
    uninterrupted = train(task, method, tmp_path / "whole", stop_after=3)
    config = {"model": {"stop_after_train_batches": 1 if boundary == "first_minibatch" else 2}}
    if boundary == "epoch":
        interrupted = train(task, method, tmp_path / "resumed", stop_after=1)
    else:
        interrupted = train(task, method, tmp_path / "resumed", config=config)
    saved = load(interrupted / "last.pt")["progress"]
    assert saved["updates"] == (1 if boundary == "first_minibatch" else 2)
    if boundary == "last_minibatch_before_selector":
        assert saved["epoch"] == 0 and saved["cursor"] == 10 and not saved["history"]
    assert read_json(interrupted / "status.json")["state"] == "interrupted"
    train(task, method, interrupted, resume=True, stop_after=3)
    assert read_json(interrupted / "status.json")["state"] == "interrupted"
    # Runtime-only interruption limits are absent from the immutable request.
    equal_tree(load(uninterrupted / "last.pt"), load(interrupted / "last.pt"))
    equal_tree(load(uninterrupted / "best.pt"), load(interrupted / "best.pt"))
    assert (uninterrupted / "history.json").read_bytes() == (interrupted / "history.json").read_bytes()
    with pytest.raises(ValueError, match="completed training"):
        evaluate(interrupted)


def test_resume_rejects_changed_seed_software_or_feature_bytes(tmp_path, monkeypatch):
    import milplus.pneamil_driver as driver
    task = make_task(tmp_path / "task")
    run = train(task, SOURCE_PROFILE, tmp_path / "run", stop_after=1)
    with pytest.raises(ValueError, match="changed actual inputs"):
        train(task, SOURCE_PROFILE, run, resume=True, seed=2027)
    original_identity = driver.software_identity
    monkeypatch.setattr(driver, "software_identity", lambda: {"source_sha256": "changed"})
    with pytest.raises(ValueError, match="changed actual inputs"):
        train(task, SOURCE_PROFILE, run, resume=True)
    monkeypatch.setattr(driver, "software_identity", original_identity)
    path = task / "features/train_000.pt"
    tensor = torch.load(path, weights_only=True)
    tensor[0, 0] += .125
    torch.save(tensor, path)
    with pytest.raises(ValueError, match="changed actual inputs"):
        train(task, SOURCE_PROFILE, run, resume=True)


def test_resume_rejects_an_actual_dirty_author_checkout_and_preserves_saved_evidence(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    run = train(task, VAL_PROFILE, tmp_path / "run", stop_after=1)
    original_checkpoint = (run / "last.pt").read_bytes()
    own_sources = tmp_path / "sources"
    own_sources.mkdir()
    shutil.copytree(source_root("pneamil"), own_sources / "pneamil")
    author = own_sources / "pneamil/PNEA_MIL.py"
    author.write_text(author.read_text() + "\n# Explicit test of dirty source rejection.\n")
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(own_sources))
    with pytest.raises(ValueError, match="tracked local modifications"):
        train(task, VAL_PROFILE, run, resume=True)
    assert (run / "last.pt").read_bytes() == original_checkpoint


def test_live_process_lock_is_never_adopted_as_stale(tmp_path):
    from milplus.pneamil_driver import own_lock
    lock = tmp_path / "RUNNING.lock"
    own_lock(lock)
    before = lock.read_bytes()
    with pytest.raises(ValueError, match="still live"):
        own_lock(lock)
    assert lock.read_bytes() == before


def test_public_method_name_and_dataset_path_reach_complete_author_recipe(tmp_path):
    task = make_task(tmp_path / "task")
    result = subprocess.run([
        sys.executable, "-B", "-m", "milplus", "run",
        "--data", str(task), "--method", "PNEA-MIL",
        "--output", str(tmp_path / "run"), "--stop-after", "1",
    ], check=True, capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0
    request = read_json(tmp_path / "run/request.json")
    assert request["method"] == VAL_PROFILE
    assert request["backend"] == "pneamil_complete_author_recipe"
    assert request["config"]["recipe"] == RECIPE
    assert load(tmp_path / "run/last.pt")["progress"]["updates"] == 2


@pytest.mark.parametrize("change", [{"task_type": "ordinal"}, {"training_unit": "patient"}])
def test_author_recipe_rejects_incompatible_task_or_training_unit_before_creating_run(tmp_path, change):
    task = make_task(tmp_path / "task")
    path = task / "task.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(change)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="individual slide classification"):
        train(task, VAL_PROFILE, tmp_path / "run")
    assert not (tmp_path / "run").exists()

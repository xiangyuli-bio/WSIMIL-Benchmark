from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
"""Full original100epoch budget, native workers4 replay and selected CLI output."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from milplus import engine
from milplus.io import read_json
from milplus.nnmil_spec import DEFAULT_PROFILE, PROFILES
from nnmil_helpers import equal_tree, independent_complete, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module", params=[p["id"] for p in PROFILES])
def full_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    profile = next(p for p in PROFILES if p["id"] == request.param)
    metric = profile["native_recipe"]["metric"]
    task = make_task(root / "task", dimension={"bacc": 256, "kappa": 1536}.get(metric, 768))
    run = engine.train(task, profile["id"], root / "run")
    reference = independent_complete(task, read_json(run / "plan.json"), root / "independent_original")
    return task, run, profile, reference


def test_complete_native_train_body_model_AdamW_bfloat16_sampler_LR_and_RNG_match(full_run):
    _, run, _, reference = full_run
    state, selected = load(run / "last.pt"), load(run / "best.pt")
    assert reference["complete_native_train_body_executed"]
    for key in ("model", "optimizer", "scaler", "early_stopping", "loader", "rng"):
        equal_tree(state[key], reference[key])
    equal_tree(selected["model"], reference["selected_model"])
    history = state["progress"]["history"]
    assert len(history) == len(reference["history"])
    for actual, expected in zip(history, reference["history"]):
        for key in expected:
            if key == "train_trace":
                assert len(actual[key]) == len(expected[key])
                for a, b in zip(actual[key], expected[key]):
                    for item in b:
                        equal_tree(a[item], b[item])
            else:
                equal_tree(actual[key], expected[key])
    assert read_json(run / "status.json")["maximum_epoch_budget"] == 100
    assert read_json(run / "status.json")["completed_epochs"] == len(history)
    assert read_json(run / "status.json")["test_evaluated"] is False
    assert read_json(run / "method_protocol.json")["native_workers"] == 4
    assert read_json(run / "request.json")["synthetic"] is True


@pytest.mark.parametrize("boundary", ["midbatch", "lastbatch_before_VAL", "epoch"])
def test_native_workers4_exact_recovery_without_replaying_optimizer_updates(full_run, boundary):
    task, run, profile, _ = full_run
    recovered = run.parent / ("recovered_" + boundary)
    if boundary == "epoch":
        engine.train(task, profile["id"], recovered, stop_after=1)
    else:
        count = 1 if boundary == "midbatch" else read_json(run / "method_protocol.json")["native_train_batches"]
        engine.train(task, profile["id"], recovered,
                     config={"model": {"stop_after_train_batches": count}})
    assert read_json(recovered / "status.json")["state"] == "interrupted"
    engine.train(task, profile["id"], recovered, resume=True)
    equal_tree(load(recovered / "last.pt"), load(run / "last.pt"))
    equal_tree(load(recovered / "best.pt"), load(run / "best.pt"))
    assert (recovered / "history.json").read_bytes() == (run / "history.json").read_bytes()


def test_public_cli_resume_and_two_independent_selected_weight_evaluations(full_run):
    task, run, profile, _ = full_run
    public = run.parent / "public_cli_run"
    method = "nnMIL" if profile["id"] == DEFAULT_PROFILE else profile["id"]
    command = [sys.executable, "-m", "milplus", "run", "--data", str(task),
               "--method", method, "--output", str(public), "--device", "cpu"]
    for suffix, filename in [
        (["--stop-after", "1"], "interrupted"), (["--resume"], "completed")]:
        result = subprocess.run(command + suffix, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        (public / f"public_cli_{filename}.log").write_text(result.stdout + result.stderr)
    equal_tree(load(public / "last.pt"), load(run / "last.pt"))
    equal_tree(load(public / "best.pt"), load(run / "best.pt"))
    command = [sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
               "--subset", "test", "--device", "cpu"]
    predictions = None
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        current = (public / "test_predictions.csv").read_bytes()
        if predictions is None:
            predictions = current
        else:
            assert predictions == current
    engine.evaluate(run, "test", "cpu")
    assert predictions == (run / "test_predictions.csv").read_bytes()


def test_native_plan_budget_and_model_substitutions_are_rejected(tmp_path):
    task = make_task(tmp_path / "task")
    for config in [
        {"epochs": 2}, {"model": {"hidden_dim": 32}}, {"model": {"batch_size": 2}},
        {"model": {"workers": 0}}, {"model": {"amp": False}},
    ]:
        with pytest.raises(ValueError):
            engine.train(task, DEFAULT_PROFILE, tmp_path / "run", config=config)
    assert not (tmp_path / "run").exists()


def test_all388_parent_profiles_and164_entries_survive_regeneration():
    from milplus.registry import methods, profiles
    before = copy.deepcopy(profiles())
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(before) == 415 and len(before[:394]) == 394
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in PROFILES + FCMIL_PROFILES}]) == 388
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert profiles() == before and len(methods()) == 164
    assert sum(bool(row["profiles"]) for row in methods()) == 164

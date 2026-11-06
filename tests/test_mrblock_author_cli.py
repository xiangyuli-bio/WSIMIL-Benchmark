from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Complete native MR and control trajectories, recovery and selected evaluation."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from milplus import engine
from milplus.io import read_json
from milplus.mrblock_spec import DEFAULT_PROFILE, PROFILES
from mrblock_helpers import equal_tree, independent_complete, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module", params=[p["id"] for p in PROFILES])
def full_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    profile = next(p for p in PROFILES if p["id"] == request.param)
    task = make_task(root / "task", concepts=profile["native_recipe"]["model_type"].endswith("cate"))
    output = engine.train(task, profile["id"], root / "run")
    original_dir = root / "independent_original"
    original_dir.mkdir()
    reference = independent_complete(task, profile, original_dir)
    torch.save(reference, root / "independent_complete_original_reference.pt")
    return task, output, profile, reference


def test_complete_original_200epoch_models_AdamW_per_update_schedule_merge_flags_and_RNG(full_run):
    _, output, profile, reference = full_run
    state, selected = load(output / "last.pt"), load(output / "best.pt")
    for key in ("model", "optimizer", "scheduler", "early_stopping", "mode_state", "rng"):
        equal_tree(state[key], reference[key])
        equal_tree(selected[key], reference["best"][key])
    history = state["progress"]["history"]
    assert len(history) == len(reference["history"])
    if not profile["native_recipe"]["early_stopping"]:
        assert len(history) == 200 and not reference["author_stopped"]
    for actual, expected in zip(history, reference["history"]):
        for key in expected:
            equal_tree(actual[key], expected[key])
    status = read_json(output / "status.json")
    assert status["maximum_epoch_budget"] == 200
    assert status["completed_epochs"] == len(history)
    assert status["completed_train_batches"] == 3 * len(history)
    assert status["test_evaluated"] is False
    assert read_json(output / "request.json")["synthetic"] is True


@pytest.mark.parametrize("boundary", ["midbatch", "lastbatch_before_VAL", "epoch"])
def test_exact_midbatch_preVAL_and_merged_epoch_recovery(full_run, boundary):
    task, output, profile, _ = full_run
    recovered = output.parent / ("recovered_" + boundary)
    config = {"model": {"stop_after_train_batches": 2 if boundary == "midbatch" else 3}}
    engine.train(task, profile["id"], recovered,
                 config=None if boundary == "epoch" else config,
                 stop_after=1 if boundary == "epoch" else None)
    assert read_json(recovered / "status.json")["state"] == "interrupted"
    saved = load(recovered / "last.pt")
    assert saved["progress"]["cursor"] == {"midbatch": 2, "lastbatch_before_VAL": 3, "epoch": 0}[boundary]
    engine.train(task, profile["id"], recovered, resume=True)
    equal_tree(load(recovered / "last.pt"), load(output / "last.pt"))
    equal_tree(load(recovered / "best.pt"), load(output / "best.pt"))
    assert read_json(recovered / "history.json") == read_json(output / "history.json")


def test_public_CLI_resume_and_two_fresh_selected_weight_evaluations(full_run):
    task, output, profile, _ = full_run
    public = output.parent / "public_cli_run"
    method = "MR-Block" if profile["id"] == DEFAULT_PROFILE else profile["id"]
    command = [sys.executable, "-m", "milplus", "run", "--data", str(task), "--method", method,
               "--output", str(public), "--device", "cpu"]
    first = subprocess.run(command + ["--stop-after", "1"], capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    equal_tree(load(public / "last.pt"), load(output / "last.pt"))
    equal_tree(load(public / "best.pt"), load(output / "best.pt"))
    (public / "public_cli_interrupted.log").write_text(first.stdout + first.stderr)
    (public / "public_cli_completed.log").write_text(second.stdout + second.stderr)
    evaluate = [sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
                "--subset", "test", "--device", "cpu"]
    for i in range(2):
        result = subprocess.run(evaluate, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        current = (public / "test_predictions.csv").read_bytes()
        if i == 0:
            predictions = current
        else:
            assert current == predictions
    engine.evaluate(output, "test", "cpu")
    assert predictions == (output / "test_predictions.csv").read_bytes()


def test_reduced_budget_and_architecture_substitution_are_rejected_before_training(tmp_path):
    task = make_task(tmp_path / "task")
    for config in ({"epochs": 2}, {"model": {"rank": 2}}, {"model": {"hidden_dim": 32}}):
        with pytest.raises(ValueError):
            engine.train(task, DEFAULT_PROFILE, tmp_path / "run", config=config)
    assert not (tmp_path / "run").exists()


def test_all368_prior_profiles_and164_families_survive_regeneration():
    from milplus.registry import methods, profiles
    before = copy.deepcopy(profiles())
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(before) == 415 and len(before[:394]) == 394
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 368
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert profiles() == before and len(methods()) == 164
    assert sum(bool(m["profiles"]) for m in methods()) == 164

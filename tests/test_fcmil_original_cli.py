"""Complete declared benchmark budget versus literal repaired native FC manager."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from milplus import engine
from milplus.fcmil_spec import DEFAULT_PROFILE, PROFILES
from milplus.io import read_json
from fcmil_helpers import equal_tree, independent_complete, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module", params=[p["id"] for p in PROFILES])
def full_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    profile = next(p for p in PROFILES if p["id"] == request.param)
    task = make_task(root / "task", dimension=1024)
    run = engine.train(task, profile["id"], root / "run")
    reference = independent_complete(task, profile, root / "independent_original")
    return task, run, profile, reference


def test_complete100epoch_original_manager_model_loss_Adam_plateau_selector_and_RNG_equal(full_run):
    _, run, _, reference = full_run
    last, best = load(run / "last.pt"), load(run / "best.pt")
    assert reference["complete_original_manager_train_body_executed"]
    assert reference["author_training_configuration_recovered"] is False
    for key in ("model", "optimizer", "scheduler", "best_auc", "model_training", "rng"):
        equal_tree(last[key], reference[key])
        equal_tree(best[key], reference["selected"][key])
    equal_tree(best["model"], load(reference["literal_original_selected_weights"]))
    equal_tree(last["progress"]["history"], reference["history"])
    status = read_json(run / "status.json")
    assert status["completed_epochs"] == status["maximum_epoch_budget"] == 100
    assert status["completed_train_batches"] == 300 and status["test_evaluated"] is False
    assert read_json(run / "request.json")["synthetic"]


@pytest.mark.parametrize("boundary", ["midbatch", "lastbatch_before_VAL", "epoch"])
def test_exact_full_budget_resume_and_history(full_run, boundary):
    task, run, profile, _ = full_run
    recovered = run.parent / ("recovered_" + boundary)
    if boundary == "epoch":
        engine.train(task, profile["id"], recovered, stop_after=1)
    else:
        count = 1 if boundary == "midbatch" else 3
        engine.train(task, profile["id"], recovered,
                     config={"model": {"stop_after_train_batches": count}})
    assert read_json(recovered / "status.json")["state"] == "interrupted"
    engine.train(task, profile["id"], recovered, resume=True)
    equal_tree(load(recovered / "last.pt"), load(run / "last.pt"))
    equal_tree(load(recovered / "best.pt"), load(run / "best.pt"))
    assert (recovered / "history.json").read_bytes() == (run / "history.json").read_bytes()


def test_public_cli_resume_and_selected_predictions(full_run):
    task, run, profile, _ = full_run
    public = run.parent / "public_cli_run"
    method = "FC-MIL" if profile["id"] == DEFAULT_PROFILE else profile["id"]
    command = [sys.executable, "-m", "milplus", "run", "--data", str(task),
               "--method", method, "--output", str(public), "--device", "cpu"]
    for suffix, name in [(["--stop-after", "1"], "interrupted"), (["--resume"], "completed")]:
        result = subprocess.run(command + suffix, capture_output=True, text=True)
        (run.parent / f"public_cli_{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
    equal_tree(load(public / "last.pt"), load(run / "last.pt"))
    equal_tree(load(public / "best.pt"), load(run / "best.pt"))
    engine.evaluate(run, "test", "cpu")
    command = [sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
               "--subset", "test", "--device", "cpu"]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (public / "test_predictions.csv").read_bytes() == (run / "test_predictions.csv").read_bytes()


def test_foreign_caller_numeric_context_cannot_change_request_or_complete_recovery(full_run):
    task, run, profile, _ = full_run
    recovered = run.parent / "foreign_numeric_context"
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    engine.train(task, profile["id"], recovered, stop_after=1)
    assert read_json(recovered / "request.json") == read_json(run / "request.json")
    engine.train(task, profile["id"], recovered, resume=True)
    equal_tree(load(recovered / "last.pt"), load(run / "last.pt"))
    equal_tree(load(recovered / "best.pt"), load(run / "best.pt"))


def test_invalid_original_inputs_and_budget_changes_fail_before_creating_run(tmp_path):
    task = make_task(tmp_path / "task", dimension=1024)
    for config in [
        {"epochs": 2}, {"model": {"hidden_dim": 32}}, {"model": {"batch_size": 2}},
        {"model": {"max_instances": 5}}, {"model": {"alpha_cr": 0}},
    ]:
        with pytest.raises(ValueError):
            engine.train(task, DEFAULT_PROFILE, tmp_path / "invalid", config=config)
    assert not (tmp_path / "invalid").exists()
    other = make_task(tmp_path / "wrong_dim", dimension=768)
    with pytest.raises(ValueError, match="1024"):
        engine.train(other, DEFAULT_PROFILE, tmp_path / "invalid_dim")
    assert not (tmp_path / "invalid_dim").exists()

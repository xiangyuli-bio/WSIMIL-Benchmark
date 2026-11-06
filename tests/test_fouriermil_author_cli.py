"""CPU reference complete recipe and public recovery; native AMP has own scope."""

import subprocess
import sys

import pytest
import torch

from milplus import engine
from milplus.fouriermil_spec import PROFILES
from milplus.io import read_json
from fouriermil_helpers import equal_tree, independent_complete, make_task


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def full_cpu(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("fourier_full_cpu_reference")
    profile = PROFILES[1]
    task = make_task(root / "task", dimension=1024, zero=True, train_slides=3)
    run = engine.train(task, profile["id"], root / "run", device="cpu")
    reference = independent_complete(task, profile, root / "independent", device="cpu")
    return task, run, profile, reference


def assert_reference(run, reference):
    last, best = load(run / "last.pt"), load(run / "best.pt")
    for state, expected in [(last, reference), (best, reference["selected"])]:
        for key in ("model", "model_training", "scaler", "gradients", "rng"):
            equal_tree(state[key], expected[key])
        equal_tree({k: state["optimizer"][k] for k in expected["optimizer"]},
                   expected["optimizer"])
    equal_tree(last["progress"]["history"], reference["history"])
    status = read_json(run / "status.json")
    assert status["maximum_epoch_budget"] == reference["maximum_epoch_budget"] == 200
    assert status["completed_epochs"] == len(reference["history"])
    assert status["completed_epochs"] == 200 or status["stopped_early"]
    assert status["completed_train_batches"] == reference["train_batches"]
    assert status["optimizer_steps"] == reference["optimizer_steps"]
    assert status["skipped_steps"] == reference["skipped_steps"]
    assert status["test_evaluated"] is False
    assert reference["actual_original_dataset_and_step_bodies"]
    assert reference["original_Lightning_environment_reproduced"] is False


def test_complete_cpu32_reference_original_dataset_steps_optimizer_history_and_selected_state(full_cpu):
    assert_reference(full_cpu[1], full_cpu[3])


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_complete_cpu32_recovery_mid_accumulation_after_step_and_final_batch(full_cpu, boundary):
    task, run, profile, _ = full_cpu
    recovered = run.parent / f"recover_{boundary}"
    engine.train(task, profile["id"], recovered,
                 config={"model": {"stop_after_train_batches": boundary}})
    assert read_json(recovered / "status.json")["state"] == "interrupted"
    engine.train(task, profile["id"], recovered, resume=True)
    equal_tree(load(run / "last.pt"), load(recovered / "last.pt"))
    equal_tree(load(run / "best.pt"), load(recovered / "best.pt"))
    assert (run / "history.json").read_bytes() == (recovered / "history.json").read_bytes()


def test_complete_cpu32_public_cli_epoch_recovery_and_two_fresh_evaluations(full_cpu):
    task, run, profile, _ = full_cpu
    public = run.parent / "public"
    command = [sys.executable, "-m", "milplus", "run", "--data", str(task),
               "--method", profile["id"], "--output", str(public), "--device", "cpu"]
    for suffix in (["--stop-after", "1"], ["--resume"]):
        result = subprocess.run(command + suffix, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    equal_tree(load(run / "last.pt"), load(public / "last.pt"))
    equal_tree(load(run / "best.pt"), load(public / "best.pt"))
    engine.evaluate(run, "test", "cpu")
    for _ in range(2):
        result = subprocess.run([
            sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
            "--subset", "test", "--device", "cpu"], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (run / "test_predictions.csv").read_bytes() == (public / "test_predictions.csv").read_bytes()

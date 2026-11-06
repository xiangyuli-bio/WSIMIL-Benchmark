"""Actual native CUDA FP16; CPU tests cannot stand in for this scope."""

import subprocess
import sys

import pytest
import torch

from milplus import engine
from milplus.fouriermil_spec import PROFILES
from milplus.io import read_json
from fouriermil_helpers import equal_tree, independent_complete, make_task
from test_fouriermil_author_cli import assert_reference, load


@pytest.fixture(scope="module")
def full_cuda(tmp_path_factory):
    assert torch.cuda.is_available(), "Run original Fourier FP16 scope on the existing GPU worker"
    root = tmp_path_factory.mktemp("fourier_full_native_cuda")
    profile = PROFILES[0]
    task = make_task(root / "task", dimension=1024, zero=True, train_slides=3)
    run = engine.train(task, profile["id"], root / "run", device="cuda")
    reference = independent_complete(task, profile, root / "independent", device="cuda")
    return task, run, profile, reference


def test_complete_native_FP16_original_model_steps_scaler_accumulation_and_literal_VAL_equal(full_cuda):
    assert_reference(full_cuda[1], full_cuda[3])
    request = read_json(full_cuda[1] / "request.json")
    assert request["numerical_identity"]["amp"] is True
    history = read_json(full_cuda[1] / "history.json")
    assert all(r["val"]["forward_logits_dtype"] == "torch.float16" for r in history)
    assert all(r["val"]["literal_loss_dtype"] == "torch.float16" for r in history)


@pytest.mark.parametrize("boundary", [1, 2, 3, 17])
def test_complete_native_FP16_recovery_including_scaled_gradients_and_Lookahead(full_cuda, boundary):
    task, run, profile, _ = full_cuda
    recovered = run.parent / f"recovered_{boundary}"
    engine.train(task, profile["id"], recovered, device="cuda",
                 config={"model": {"stop_after_train_batches": boundary}})
    assert read_json(recovered / "status.json")["state"] == "interrupted"
    engine.train(task, profile["id"], recovered, device="cuda", resume=True)
    equal_tree(load(run / "last.pt"), load(recovered / "last.pt"))
    equal_tree(load(run / "best.pt"), load(recovered / "best.pt"))


def test_complete_native_FP16_public_cli_and_two_fresh_evaluations(full_cuda):
    task, run, profile, _ = full_cuda
    public = run.parent / "public"
    command = [sys.executable, "-m", "milplus", "run", "--data", str(task),
               "--method", profile["id"], "--output", str(public), "--device", "cuda"]
    for suffix in (["--stop-after", "1"], ["--resume"]):
        result = subprocess.run(command + suffix, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    equal_tree(load(run / "last.pt"), load(public / "last.pt"))
    equal_tree(load(run / "best.pt"), load(public / "best.pt"))
    engine.evaluate(run, "test", "cuda")
    for _ in range(2):
        result = subprocess.run([
            sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
            "--subset", "test", "--device", "cuda"], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (run / "test_predictions.csv").read_bytes() == (public / "test_predictions.csv").read_bytes()

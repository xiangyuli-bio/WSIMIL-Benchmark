from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Complete100epoch native CAR trajectories, preselection and fresh evaluation."""

from contextlib import suppress
import copy
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch
from torch.utils.data import DataLoader

from milplus import engine
from milplus.carmil_driver import SlideBags
from milplus.carmil_spec import DEFAULT_PROFILE, PROFILES
from milplus.data import Bundle
from milplus.io import read_json
from carmil_helpers import (
    equal_tree, literal_datasets, literal_sources, make_task, native_args, native_model,
)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def complete_reference(task, profile):
    original = literal_sources()
    datasets = literal_datasets(task, original.datasets)

    class TracedC16(original.datasets.C16Dataset):
        def __getitem__(self, index):
            self.access.append(index)
            return super().__getitem__(index)

    train_data, val_data = datasets["train"], datasets["val"]
    train_data.__class__ = TracedC16
    train_data.access = []
    original.utils.seed_torch(2026)
    generator = torch.Generator().manual_seed(7784414403328510413)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    model = native_model(original, profile, Bundle(task).dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 0)
    criterion = (original.trainer.CustomLoss_L1 if profile["native_recipe"]["reg_dist"] == "L1"
                 else original.trainer.CustomLoss_COS)("ce", .2, .2)
    early = original.utils.EarlyStopping(patience=30, stop_epoch=130, save_best_model_stage=0.)
    original.utils.seed_torch(2026)
    best_auc, best, best_epoch, history = 0., None, None, []

    def snapshot():
        return copy.deepcopy({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "early_stopping": early.state_dict(),
            "loader_generator": generator.get_state(), "rng": engine.rng_state(),
        })

    for epoch in range(100):
        # This ordering is copied from one_fold: select before TRAIN,
        # including untrained epoch0. The final100th update is never selected again.
        stop, accuracy, auc, precision, recall, f1, val_loss = original.trainer.val_loop(
            native_args(profile), model, val_loader, torch.device("cpu"), criterion, early, epoch)
        improved = auc > best_auc
        if improved:
            best_auc, best, best_epoch = auc, snapshot(), epoch
        assert not stop
        train_data.access = []
        original.trainer.train_loop(
            native_args(profile), model, train_loader, optimizer, torch.device("cpu"),
            suppress, criterion, None, scheduler, 0, epoch)
        history.append({
            "epoch_zero_based": epoch, "val_auc": auc, "val_loss": float(val_loss),
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "improved": improved, "train_order": train_data.access,
        })
    assert best is not None, "Literal original strict AUC selector never selected weights"
    return {**snapshot(), "history": history, "best": best, "best_selector_epoch": best_epoch}


@pytest.fixture(scope="module", params=[p["id"] for p in PROFILES])
def full_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    profile = next(p for p in PROFILES if p["id"] == request.param)
    task = make_task(root / "task")
    output = engine.train(task, profile["id"], root / "run")
    reference = complete_reference(task, profile)
    torch.save(reference, root / "independent_complete_original_reference.pt")
    return task, output, profile, reference


def test_all_eight_complete_native1024_original_100epoch_models_Adam_scheduler_RNG_and_best(full_run):
    _, output, _, reference = full_run
    state, selected = load(output / "last.pt"), load(output / "best.pt")
    for key in ["model", "optimizer", "scheduler", "early_stopping", "loader_generator", "rng"]:
        equal_tree(state[key], reference[key])
        equal_tree(selected[key], reference["best"][key])
    history = state["progress"]["history"]
    assert len(history) == len(reference["history"]) == 100
    for actual, expected in zip(history, reference["history"]):
        for key in expected:
            assert actual[key] == expected[key], (key, actual["epoch_zero_based"])
    assert selected["progress"]["epoch"] == reference["best_selector_epoch"] < 100
    assert selected["progress"]["phase"] == "train"
    assert selected["progress"]["cursor"] == 0
    assert selected["progress"]["history"][-1]["train_order"] is None
    status = read_json(output / "status.json")
    assert status["completed_epochs"] == status["maximum_epoch_budget"] == status["completed_selectors"] == 100
    assert status["completed_train_batches"] == 300 and status["test_evaluated"] is False
    assert read_json(output / "request.json")["synthetic"] is True
    assert read_json(output / "method_protocol.json")["no_post_final_training_selector"] is True


@pytest.mark.parametrize("boundary", ["pretrain_selector", "midbatch", "lastbatch_before_scheduler", "epoch"])
def test_exact_full_recipe_recovery_of_private_sampler_pretrain_phase_Adam_and_scheduler(full_run, boundary):
    task, output, profile, _ = full_run
    recovered = output.parent / ("recovered_" + boundary)
    config = {
        "pretrain_selector": {"model": {"stop_after_selectors": 1}},
        "midbatch": {"model": {"stop_after_train_batches": 2}},
        "lastbatch_before_scheduler": {"model": {"stop_after_train_batches": 3}},
        "epoch": {},
    }[boundary]
    engine.train(task, profile["id"], recovered, config=config,
                 stop_after=1 if boundary == "epoch" else None)
    assert read_json(recovered / "status.json")["state"] == "interrupted"
    saved = load(recovered / "last.pt")
    assert saved["progress"]["phase"] == ("validate" if boundary == "epoch" else "train")
    assert saved["progress"]["cursor"] == {
        "pretrain_selector": 0, "midbatch": 2, "lastbatch_before_scheduler": 3, "epoch": 0,
    }[boundary]
    if boundary == "pretrain_selector":
        assert not saved["optimizer"]["state"] and saved["progress"]["updates"] == 0
        assert saved["progress"]["selectors"] == 1
    engine.train(task, profile["id"], recovered, resume=True)
    equal_tree(load(recovered / "last.pt"), load(output / "last.pt"))
    equal_tree(load(recovered / "best.pt"), load(output / "best.pt"))
    assert read_json(recovered / "history.json") == read_json(output / "history.json")


def test_public_CLI_launch_epoch_resume_and_two_fresh_best_weight_evaluations(full_run):
    task, output, profile, _ = full_run
    public = output.parent / "public_cli_run"
    method = "CAR-MIL" if profile["id"] == DEFAULT_PROFILE else profile["id"]
    command = [sys.executable, "-m", "milplus", "run", "--data", str(task), "--method", method,
               "--output", str(public), "--device", "cpu"]
    first = subprocess.run(command + ["--stop-after", "1"], capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert read_json(public / "status.json")["state"] == "interrupted"
    second = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    equal_tree(load(public / "last.pt"), load(output / "last.pt"))
    equal_tree(load(public / "best.pt"), load(output / "best.pt"))
    (public / "public_cli_interrupted.log").write_text(first.stdout + first.stderr)
    (public / "public_cli_completed.log").write_text(second.stdout + second.stderr)
    evaluate = [sys.executable, "-m", "milplus", "evaluate", "--run", str(public),
                "--subset", "test", "--device", "cpu"]
    first_eval = subprocess.run(evaluate, capture_output=True, text=True)
    assert first_eval.returncode == 0, first_eval.stdout + first_eval.stderr
    predicted = (public / "test_predictions.csv").read_bytes()
    second_eval = subprocess.run(evaluate, capture_output=True, text=True)
    assert second_eval.returncode == 0, second_eval.stdout + second_eval.stderr
    assert predicted == (public / "test_predictions.csv").read_bytes()
    engine.evaluate(output, "test", "cpu")
    assert predicted == (output / "test_predictions.csv").read_bytes()
    frame = pd.read_csv(public / "test_predictions.csv")
    assert len(frame) == frame["patient_id"].nunique() == 2
    assert set(frame["subset"]) == {"test"}


def test_original_optional_baseline_partial_checkpoint_load_and_exact_asset_binding(tmp_path):
    task = make_task(tmp_path / "task")
    original = literal_sources()
    engine.seed_everything(42)
    baseline = original.attmil.DAttention(1024, 2, .25, "relu")
    checkpoint = tmp_path / "baseline_synthetic_software_fixture.pt"
    torch.save({"model": baseline.state_dict(), "synthetic": True}, checkpoint)
    output = engine.train(task, DEFAULT_PROFILE, tmp_path / "run", config={
        "model": {"initial_checkpoint": str(checkpoint), "stop_after_selectors": 1}})
    state = load(output / "last.pt")
    original.utils.seed_torch(2026)
    expected = native_model(original, PROFILES[0])
    native_load = expected.load_state_dict(baseline.state_dict(), strict=False)
    equal_tree(state["model"], expected.state_dict())
    protocol = read_json(output / "method_protocol.json")
    assert protocol["warm_start"]["missing_keys"] == native_load.missing_keys
    assert protocol["warm_start"]["unexpected_keys"] == native_load.unexpected_keys
    assert native_load.missing_keys and not native_load.unexpected_keys
    torch.save({"model": baseline.state_dict(), "changed": True}, checkpoint)
    with pytest.raises(ValueError, match="changed"):
        engine.train(task, DEFAULT_PROFILE, output, resume=True,
                     config={"model": {"initial_checkpoint": str(checkpoint)}})


def test_training_never_constructs_TEST_and_changed_numerics_or_features_reject_resume(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    requested = []
    original = SlideBags.__init__

    def observed(self, bundle, subset):
        requested.append(subset)
        assert subset != "test"
        original(self, bundle, subset)

    monkeypatch.setattr(SlideBags, "__init__", observed)
    output = engine.train(task, DEFAULT_PROFILE, tmp_path / "run", stop_after=1)
    assert requested == ["train", "val"] and not list(output.glob("test_*"))
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if precision != "medium" else "highest")
        with pytest.raises(ValueError, match="changed"):
            engine.train(task, DEFAULT_PROFILE, output, resume=True)
    finally:
        torch.set_float32_matmul_precision(precision)
    path = task / "features/train_000.pt"
    value = load(path)
    value[0, 0] += 1
    torch.save(value, path)
    with pytest.raises(ValueError, match="changed"):
        engine.train(task, DEFAULT_PROFILE, output, resume=True)


def test_all360_parent_profiles_and_every164_family_survive_catalog_regeneration():
    from milplus.registry import methods, profiles
    before = copy.deepcopy(profiles())
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(before) == 415 and len(before[:394]) == 394
    assert len([p for p in before[:394] if p["id"] not in {n["id"] for n in PROFILES + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 360
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert profiles() == before
    assert len(methods()) == 164 and sum(bool(method["profiles"]) for method in methods()) == 164

from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Native original MR definitions, mode flags and per-update scheduling."""

import copy
from pathlib import Path

import pytest
import torch

from milplus.data import Bundle
from milplus.mrblock_driver import (collate, concept_asset, make_components, mode_state,
                                   restore_mode_state, train_step)
from milplus.mrblock_source import contract, definitions
from milplus.mrblock_spec import DEFAULT_PROFILE, MODELS, PROFILES
from milplus.registry import inspect_method, profiles, require_runnable, resolve_for_bundle
from mrblock_helpers import args, equal_tree, literal_sources, make_task, native_model


@pytest.mark.parametrize("model_type", MODELS)
def test_all_ten_native_models_outputs_AdamW_schedule_modes_and_RNG_equal_original(tmp_path, model_type):
    original, actual = literal_sources(), definitions()
    profile = next(p for p in PROFILES if p["native_recipe"]["model_type"] == model_type)
    task = make_task(tmp_path / "task", concepts=model_type.endswith("cate"))
    bundle = Bundle(task)
    actual.seed.seed_torch(2026)
    model, optimizer, scheduler, _ = make_components(
        bundle, profile, actual, concept_asset(bundle, profile), torch.device("cpu"))
    original.seed(2026)
    reference = native_model(original, profile, task)
    ref_optimizer = original.optimizer.get_optim(reference, args(task))
    ref_scheduler = original.optimizer.get_scheduler(ref_optimizer, args(task))
    equal_tree(model.state_dict(), reference.state_dict())
    features = torch.randn(17, 512, generator=torch.Generator().manual_seed(51))
    batch = (features, torch.tensor([1]))
    actual.seed.seed_torch(62)
    model.train()
    result = train_step(model, optimizer, scheduler, batch, torch.device("cpu"))
    rng_actual = torch.get_rng_state().clone()
    original.seed(62)
    reference.train()
    logits = reference(features)[0]
    loss = torch.nn.functional.cross_entropy(logits, batch[1])
    loss.backward()
    ref_optimizer.step()
    ref_scheduler.step()
    ref_optimizer.zero_grad()
    assert result["loss"] == loss.item() and result["lr_used"] == 5e-6
    equal_tree(model.state_dict(), reference.state_dict())
    equal_tree(optimizer.state_dict(), ref_optimizer.state_dict())
    equal_tree(scheduler.state_dict(), ref_scheduler.state_dict())
    assert torch.equal(rng_actual, torch.get_rng_state())
    model.eval()
    reference.eval()
    actual.seed.seed_torch(72)
    result_a = model(features)
    rng_actual = torch.get_rng_state().clone()
    original.seed(72)
    result_b = reference(features)
    equal_tree(result_a, result_b)
    assert torch.equal(rng_actual, torch.get_rng_state())


def test_native_merge_flags_restore_exact_selected_model_without_double_merge(tmp_path):
    source = definitions()
    profile = PROFILES[0]
    task = make_task(tmp_path / "task")
    bundle = Bundle(task)
    model, optimizer, scheduler, _ = make_components(
        bundle, profile, source, None, torch.device("cpu"))
    features = torch.randn(17, 512, generator=torch.Generator().manual_seed(21))
    model.train()
    for label in (0, 1, 0):
        train_step(model, optimizer, scheduler, (features, torch.tensor([label])), torch.device("cpu"))
    model.eval()
    state = copy.deepcopy(model.state_dict())
    flags = mode_state(model, source)
    expected = model(features)[0]
    fresh, _, _, _ = make_components(bundle, profile, source, None, torch.device("cpu"))
    fresh.load_state_dict(state)
    fresh.eval()
    assert not torch.equal(fresh(features)[0], expected)
    fresh.load_state_dict(state)
    restore_mode_state(fresh, source, flags)
    fresh.eval()
    assert torch.equal(fresh(features)[0], expected)
    assert flags["merged"] and all(flags["merged"].values())


def test_all_twenty_native_profiles_preserve200epoch_budget_rank64_and_original_controls():
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len(PROFILES) == 20 and len(MODELS) == 10 and len([p for p in profiles() if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(profiles()) == 394
    assert sum(p["comparison_role"] == "author_baseline_control" for p in PROFILES) == 10
    for profile in PROFILES:
        recipe = profile["native_recipe"]
        assert recipe["epochs"] == 200 and recipe["rank"] == 64 and recipe["patch_cap"] is None
        assert recipe["lr"] == 5e-4 and recipe["weight_decay"] == 1e-5
    assert contract()["original_CLI_defaults"]["--early_stopping"] is False
    from types import SimpleNamespace
    context = SimpleNamespace(classes=["a", "b"], task_type="classification", config={})
    for name in ["MR-Block", "MR-AB-MIL", "MR-Block / MR-AB-MIL"]:
        assert inspect_method(name)["id"] == DEFAULT_PROFILE
        assert resolve_for_bundle(name, context)["id"] == DEFAULT_PROFILE
    assert require_runnable("mrblock")["id"] == "mrblock"


def test_native_CATE_concepts_are_required_and_never_replaced_with_generated_real_assets(tmp_path):
    task = make_task(tmp_path / "task")
    profile = next(p for p in PROFILES if p["native_recipe"]["model_type"] == "mr-cate")
    with pytest.raises(ValueError, match="aligned real concepts"):
        concept_asset(Bundle(task), profile)

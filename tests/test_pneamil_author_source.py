from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Native author definitions, loss, optimizer, RNG and full sampling contracts."""

import ast
from types import SimpleNamespace

import numpy as np
import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.pneamil_driver import (
    RecordingRandomSampler, SlideBags, make_model, optimizer_groups, options,
)
from milplus.pneamil_source import original_definitions, source_contract
from milplus.pneamil_spec import PROFILES, RECIPE, SOURCE_PROFILE, VAL_PROFILE
from milplus.registry import inspect_method, profiles, require_runnable, resolve_for_bundle
from pneamil_helpers import equal_tree, literal_sources, make_task


@pytest.mark.parametrize("dimension,classes", [(1024, 2), (1536, 3)])
def test_native64_axes_loss_all_gradients_and_three_Adam_updates_equal_complete_literal_file(dimension, classes):
    torch.set_num_threads(2)
    adapted, _ = original_definitions()
    literal, _ = literal_sources()
    a = adapted(D_in=dimension, C=classes, seed=311)
    b = literal.PNEA_MIL(D_in=dimension, C=classes, seed=311)
    assert a.D == b.D == 64
    assert a.f.weight.shape == (classes * 64, dimension)
    groups = optimizer_groups(a)
    assert groups[0]["names"] == ["W", "f.weight"]
    assert groups[1]["names"] == ["b", "f.bias"]
    assert all(group["lr"] == 1e-4 and group["weight_decay"] == 0 for group in groups)
    assert all(group["betas"] == [.9, .999] and group["eps"] == 1e-8 for group in groups)
    for step in range(3):
        x = torch.randn(8, 1024, dimension, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        targets = torch.arange(8) % classes
        a.optim.zero_grad()
        b.optim.zero_grad()
        equal_tree(a(x, return_evidence=True), b(y, return_evidence=True))
        loss_a, loss_b = a.return_loss(x, targets), b.return_loss(y, targets)
        expected = torch.nn.functional.cross_entropy(a(x), targets) + 5 * a.W.relu().mean()
        assert torch.equal(loss_a, loss_b) and torch.equal(loss_a, expected)
        loss_a.backward()
        loss_b.backward()
        assert torch.equal(x.grad, y.grad)
        for name, parameter in a.named_parameters():
            assert torch.equal(parameter.grad, dict(b.named_parameters())[name].grad), (step, name)
        a.optim.step()
        b.optim.step()
        equal_tree(a.state_dict(), b.state_dict())
        equal_tree(a.optim.state_dict(), b.optim.state_dict())


def test_no_definition_rewrites_and_complete_author_defaults_are_read_from_actual_pin():
    model, trainer = original_definitions()
    literal_model, literal_trainer = literal_sources()
    assert ast.dump(ast.parse(__import__("inspect").getsource(model))) == ast.dump(
        ast.parse(__import__("inspect").getsource(literal_model.PNEA_MIL)))
    for name in ["train_one_epoch", "evaluate_loss"]:
        assert ast.dump(ast.parse(__import__("inspect").getsource(getattr(trainer, name)))) == ast.dump(
            ast.parse(__import__("inspect").getsource(getattr(literal_trainer, name))))
    contract = source_contract()
    assert contract["commit"] == "ef204ff5179ce7ea4bb7d04744f1553bee2f039c"
    assert contract["source_definition_replacements"] == []
    assert contract["author_defaults"]["--epochs"] == 1000
    assert contract["author_defaults"]["--batch_size"] == 8
    assert contract["author_defaults"]["--num_train_patches"] == 1024
    assert "TRAIN" in contract["literal_selector_population"]


def test_exact_RandomSampler_order_patch_replacement_and_global_RNG_equal_original_dataset(tmp_path):
    task = make_task(tmp_path / "task")
    bundle = Bundle(task)
    _, literal = literal_sources()
    rows = bundle.records("train")
    csv = tmp_path / "source_train.csv"
    __import__("pandas").DataFrame([
        {"file": str(task / row["path"]), "label": row["target"]} for row in rows
    ]).to_csv(csv, index=False)
    original = literal.SlideDataset(csv, num_patches=1024)
    adapted = SlideBags(bundle, "train", sample=True)
    torch.manual_seed(3112)
    expected = list(DataLoader(original, batch_size=8, shuffle=True))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(3112)
    sampler = RecordingRandomSampler(adapted)
    actual = list(DataLoader(adapted, batch_size=8, sampler=sampler))
    equal_tree(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert sorted(sampler.order) == list(range(len(rows)))
    assert [len(y) for _, y in actual] == [8, 2]
    assert all(row["sampled_patches"] == 1024 for row in adapted.access)
    assert all(row["unique_indices"] <= row["original_patches"] < 1024 for row in adapted.access)
    validation = SlideBags(bundle, "val")
    assert len(validation[0][0]) == 1041
    assert validation.access[0]["all_original_patches"] is True


def test_optimizer_always_owns_actual_parameters_after_parameter_replacing_device_conversion(monkeypatch):
    # Exercise Module.to's documented parameter replacement possibility on CPU.
    original_to = torch.nn.Module.to

    def replacing_to(module, *args, **kwargs):
        result = original_to(module, *args, **kwargs)
        for child in result.modules():
            for name, parameter in list(child._parameters.items()):
                if parameter is not None:
                    child._parameters[name] = torch.nn.Parameter(parameter.detach().clone())
        return result

    monkeypatch.setattr(torch.nn.Module, "to", replacing_to)
    model, _ = make_model(SimpleNamespace(dim=1024, classes=["a", "b"]), 23, torch.device("cpu"))
    assert {id(p) for p in model.parameters()} == {
        id(p) for group in model.optim.param_groups for p in group["params"]}
    before = model.f.weight.detach().clone()
    model.return_loss(torch.randn(8, 1024, 1024), torch.arange(8) % 2).backward()
    model.optim.step()
    assert not torch.equal(before, model.f.weight)


def test_original_profile_is_preserved_and_display_name_chooses_explicit_repaired_author_recipe():
    assert len(PROFILES) == 2
    from milplus.carmil_spec import PROFILES as carmil_profiles
    from milplus.asmil_spec import PROFILES as new_profiles
    from milplus.attrimil_spec import PROFILES as newest_profiles
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in new_profiles + newest_profiles + carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 354
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in newest_profiles + carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 358
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 360
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(profiles()) == 394
    context = SimpleNamespace(classes=["a", "b"], task_type="classification", config={})
    for name in ["PNEA-MIL", "PNEAMIL", "pnea-mil"]:
        assert inspect_method(name)["id"] == VAL_PROFILE
        assert resolve_for_bundle(name, context)["id"] == VAL_PROFILE
    assert require_runnable("pneamil")["id"] == "pneamil"
    assert resolve_for_bundle("pneamil", context)["id"] == "pneamil"
    for name, population in [(SOURCE_PROFILE, "train"), (VAL_PROFILE, "val")]:
        assert require_runnable(name)["selection_population"] == population
        assert require_runnable(name)["native_recipe"] == RECIPE


@pytest.mark.parametrize("config", [
    {"epochs": 1}, {"lr": 1e-3}, {"batch_size": 1},
    {"model": {"evidence_axes": 8}}, {"model": {"train_patches": 32}},
    {"model": {"stop_after_train_batches": 0}}, {"threads": True},
    {"model": {"stop_after_train_batches": True}},
])
def test_no_reduced_epoch_budget_changed_sampling_optimizer_or_native_axes(config):
    with pytest.raises(ValueError):
        options(config)
    assert options({"epochs": 1000})[0]["recipe"] == RECIPE

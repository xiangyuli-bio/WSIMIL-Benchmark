"""Native full-width original math, Lookahead state, accumulation and precision."""

import copy
from types import SimpleNamespace

import pytest
import torch

from milplus.engine import rng_state, seed_everything
from milplus.fouriermil_driver import (
    components, initialize_runtime, minibatch, options, policy,
)
from milplus.fouriermil_source import contract, definitions
from milplus.fouriermil_spec import PROFILES
from milplus.fouriermil_state import (
    gradients, optimizer_snapshot, restore_gradients, restore_optimizer,
)
from fouriermil_helpers import compat_optimizer, equal_tree, literal_sources, named_optimizer


@pytest.mark.parametrize("classes,count", [(2, 1), (2, 2), (2, 7), (2, 16), (3, 7)])
def test_original_initialization_all_outputs_full_gradients_and_updates(classes, count):
    initialize_runtime()
    torch.set_num_threads(2)
    author = literal_sources()
    seed_everything(2026)
    original = author.Model(n_classes=classes)
    opt = compat_optimizer(author, original, PROFILES[1]["author_recipe"])
    iface = SimpleNamespace(model=original, loss=torch.nn.CrossEntropyLoss(),
                            data=[{"count": 0, "correct": 0} for _ in range(classes)])
    seed_everything(2026)
    model, interface, optimizer, scaler, adapter = components(
        classes, torch.device("cpu"), PROFILES[1]["author_recipe"])
    equal_tree(original.state_dict(), model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == 1848322 + (classes - 2) * 513
    features = torch.randn(1, count, 1024, generator=torch.Generator().manual_seed(321))
    targets = torch.tensor([classes - 1])
    equal_tree(original(data=features), model(data=features))
    for index in range(7):
        if index % 2 == 0:
            opt.zero_grad(set_to_none=False)
        loss = author.Interface.training_step(iface, (features, targets), index)["loss"]
        (loss / 2).backward()
        if (index + 1) % 2 == 0 or index == 6:
            opt.step()
        result = minibatch(model, interface, optimizer, scaler, adapter,
                           (features, targets), index, 7, torch.device("cpu"), False)
        assert result["loss"] == float(loss.detach())
        assert result["normalized_loss"] == float(loss.detach() / 2)
        equal_tree(model.state_dict(), original.state_dict())
        equal_tree(gradients(model), {n: p.grad for n, p in original.named_parameters()})
        saved = optimizer_snapshot(model, optimizer)
        equal_tree({k: saved[k] for k in named_optimizer(original, opt)}, named_optimizer(original, opt))
        equal_tree(interface.data, iface.data)
    assert result["optimizer_boundary"]  # final singleton group still normalizes by2


@pytest.mark.parametrize("boundary", [1, 2, 3, 5, 6, 7, 12])
def test_full_model_optimizer_gradient_and_rng_recovery_across_accumulation_and_slow_boundaries(boundary):
    initialize_runtime()
    torch.set_num_threads(2)
    seed_everything(2026)
    model, iface, opt, scaler, author = components(2, torch.device("cpu"), PROFILES[1]["author_recipe"])
    saved = None
    batches = [
        (torch.randn(1, 2 + i % 5, 1024, generator=torch.Generator().manual_seed(53 + i)),
         torch.tensor([i % 2])) for i in range(14)]
    for i, batch in enumerate(batches):
        minibatch(model, iface, opt, scaler, author, batch, i, len(batches), torch.device("cpu"), False)
        if i + 1 == boundary:
            saved = copy.deepcopy((model.state_dict(), optimizer_snapshot(model, opt),
                                   gradients(model), scaler.state_dict(), rng_state(), iface.data))
            recovered, ri, ro, rs, ra = components(2, torch.device("cpu"), PROFILES[1]["author_recipe"])
            recovered.load_state_dict(saved[0])
            restore_optimizer(recovered, ro, saved[1])
            restore_gradients(recovered, saved[2])
            rs.load_state_dict(saved[3])
            ri.data = saved[5]
        elif i + 1 > boundary:
            minibatch(recovered, ri, ro, rs, ra, batch, i, len(batches), torch.device("cpu"), False)
            equal_tree(model.state_dict(), recovered.state_dict())
            equal_tree(optimizer_snapshot(model, opt), optimizer_snapshot(recovered, ro))
            equal_tree(gradients(model), gradients(recovered))
            equal_tree(iface.data, ri.data)
    assert saved is not None


@pytest.mark.parametrize("mutation", ["name", "shape", "dtype", "cache"])
def test_optimizer_checkpoint_binding_rejects_corruption(mutation):
    torch.set_num_threads(2)
    model, _, opt, _, _ = components(2, torch.device("cpu"), PROFILES[1]["author_recipe"])
    state = optimizer_snapshot(model, opt)
    if mutation == "cache":
        state["original_RAdam_ten_slot_cache"] = []
    else:
        state["parameter_binding"][0][0][mutation] = "wrong"
    with pytest.raises(ValueError):
        restore_optimizer(model, opt, state)


@pytest.mark.parametrize("config", [
    {"epochs": 2}, {"epochs": True}, {"workers": 0}, {"precision": 32},
    {"model": {"dropout": .1}}, {"model": {"hidden_dim": 8}},
    {"model": {"stop_after_train_batches": 0}}, {"threads": False},
])
def test_author_budget_and_model_cannot_be_silently_replaced(config):
    with pytest.raises(ValueError):
        options(config)


def test_native_profile_refuses_cpu_and_cpu_reference_refuses_gpu():
    with pytest.raises(ValueError, match="CUDA"):
        policy(PROFILES[0]["id"], "cpu")
    with pytest.raises(ValueError, match="CPU32"):
        policy(PROFILES[1]["id"], "cuda")
    assert policy(PROFILES[1]["id"], "cpu") == (torch.device("cpu"), False)


def test_fixed_original_recipe_license_and_literal_val_loss():
    observed = contract()
    assert observed["commit"] == "0192fe7f7a4fdbb4161148d0b88b0d67dbaa14f5"
    assert observed["original_configuration"]["General"]["epochs"] == 200
    assert observed["current_framework_is_original_environment"] is False
    assert "MIT" in observed["license"]
    author, actual = literal_sources(), definitions()
    logits = torch.tensor([[2., -1.], [1., 2.]], dtype=torch.float16)
    labels = torch.tensor([[0], [1]])
    expected = author.val_loss(logits, labels)
    equal_tree(actual.val_loss(logits, labels), expected)
    assert expected.dtype == torch.float16 and not expected.requires_grad

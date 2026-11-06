from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Native original CAR backbones, scalar/attention losses and lazy RNG order."""

from contextlib import suppress
from types import SimpleNamespace

import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch
from torch.utils.data import DataLoader, TensorDataset

from milplus import engine
from milplus.carmil_driver import RecordingShuffle, options, train_step
from milplus.carmil_source import arguments, contract, definitions, model_and_loss
from milplus.carmil_spec import DEFAULT_PROFILE, PROFILES
from milplus.registry import inspect_method, profiles, require_runnable, resolve_for_bundle
from carmil_helpers import equal_tree, literal_sources, native_args, native_model


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p["id"])
@pytest.mark.parametrize("dimension", [1024, 512])
def test_all_native_original_models_outputs_gradients_Adam_scheduler_and_RNG(profile, dimension):
    torch.set_num_threads(2)
    source = literal_sources()
    engine.seed_everything(2026)
    expected = native_model(source, profile, dimension)
    reference_init_rng = engine.rng_state()
    engine.seed_everything(2026)
    actual, criterion = model_and_loss(definitions(), profile, dimension, "cpu")
    equal_tree(actual.state_dict(), expected.state_dict())
    equal_tree(engine.rng_state(), reference_init_rng)
    opt_a = torch.optim.Adam(actual.parameters(), lr=2e-4, weight_decay=1e-5)
    opt_b = torch.optim.Adam(expected.parameters(), lr=2e-4, weight_decay=1e-5)
    schedule_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, 100, 0)
    schedule_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, 100, 0)
    original_loss = (source.trainer.CustomLoss_L1 if profile["native_recipe"]["reg_dist"] == "L1"
                     else source.trainer.CustomLoss_COS)("ce", .2, .2)
    # Exercise native full1024D and author-mentioned512D, with long bags.
    x = torch.randn(1, 1025, dimension)
    y = torch.tensor([1])
    before = engine.rng_state()
    actual.train()
    out_a = actual(x, y, torch.nn.CrossEntropyLoss()) if arguments(profile).model == "dsmil" else actual(x)
    after = engine.rng_state()
    engine.restore_rng(before)
    expected.train()
    out_b = expected(x, y, torch.nn.CrossEntropyLoss()) if arguments(profile).model == "dsmil" else expected(x)
    equal_tree(out_a, out_b)
    equal_tree(after, engine.rng_state())
    if arguments(profile).model == "transmil":
        # Preserve original flattened2N output and scalar indexing; no hidden repair.
        assert out_a[1].shape == (2050,)
    x = x[:, :19]  # distinct short-bag native trainer/update comparison
    before = engine.rng_state()
    train_step(actual, criterion, opt_a, arguments(profile), (x, y), torch.device("cpu"))
    after = engine.rng_state()
    engine.restore_rng(before)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    # Native DataLoader has a global base-seed draw; compare the actual model
    # step with the same draw on the implementation side.
    iterator = iter(loader)
    # Re-execute from the untouched pre-update state through original trainer
    # below using a loader without another global draw.
    class AlreadyIterated:
        def __len__(self):
            return 1
        def __iter__(self):
            return iterator
    engine.restore_rng(before)
    source.trainer.train_loop(native_args(profile), expected, AlreadyIterated(), opt_b,
                              torch.device("cpu"), suppress, original_loss, None, schedule_b, 0, 0)
    schedule_a.step()
    equal_tree(actual.state_dict(), expected.state_dict())
    equal_tree(opt_a.state_dict(), opt_b.state_dict())
    equal_tree(after, engine.rng_state())
    equal_tree(schedule_a.state_dict(), schedule_b.state_dict())
    for a, b in zip(actual.parameters(), expected.parameters()):
        equal_tree(a.grad, b.grad)


def test_lazy_recorded_shuffle_matches_original_order_and_private_generator_at_each_actual_batch():
    dataset = TensorDataset(torch.arange(5))
    native_gen = torch.Generator().manual_seed(7784414403328510413)
    actual_gen = torch.Generator().manual_seed(7784414403328510413)
    native = DataLoader(dataset, batch_size=1, shuffle=True, generator=native_gen)
    sampler = RecordingShuffle(dataset, generator=actual_gen)
    actual = DataLoader(dataset, batch_size=1, sampler=sampler, generator=actual_gen)
    original_iterator, iterator = iter(native), iter(actual)
    equal_tree(native_gen.get_state(), actual_gen.get_state())
    order = []
    for expected, got in zip(original_iterator, iterator):
        equal_tree(expected, got)
        equal_tree(native_gen.get_state(), actual_gen.get_state())
        order.append(int(got[0]))
    # zip stops when the first iterator ends; explicitly exhaust the second.
    with pytest.raises(StopIteration):
        next(iterator)
    equal_tree(native_gen.get_state(), actual_gen.get_state())
    assert sampler.order == order


def test_native_cli_contract_no_hidden_recipe_shrink_and_display_vs_legacy_route():
    original = contract()
    assert original["original_CLI_defaults"]["--num_epoch"] == 100
    assert original["original_CLI_defaults"]["--reg_dist"] == "L1"
    assert len(PROFILES) == 8
    assert {(p["native_recipe"]["backbone"], p["native_recipe"]["reg_dist"]) for p in PROFILES} == {
        (backbone, distance) for backbone in ["attmil", "gattmil", "transmil", "dsmil"]
        for distance in ["L1", "COS"]}
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(profiles()) == 394
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in PROFILES + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}]) == 360
    assert require_runnable("carmil")["id"] == "carmil"
    context = SimpleNamespace(classes=["a", "b"], task_type="classification", config={})
    for name in ["CAR-MIL", "CAR_MIL", "CARMIL"]:
        assert inspect_method(name)["id"] == DEFAULT_PROFILE
        assert resolve_for_bundle(name, context)["id"] == DEFAULT_PROFILE
    for bad in [{"epochs": 2}, {"model": {"hidden_dim": 8}},
                {"model": {"max_instances": 10}}, {"model": {"optimizer": "SGD"}}]:
        with pytest.raises(ValueError):
            options(bad)


def test_original_runtime_failures_remain_observable_and_numpy_repair_is_explicit():
    original = literal_sources(repair_numpy=False, repair_CPU_timing=False)
    with pytest.raises(AttributeError, match="Inf"):
        original.utils.EarlyStopping()
    with pytest.raises((AssertionError, RuntimeError), match="CUDA|cuda"):
        original.trainer.train_loop(None, None, None, None, torch.device("cpu"),
                                    suppress, None, None, None, 0, 0)
    repaired = definitions().utils.EarlyStopping(patience=30, stop_epoch=130)
    assert repaired.val_loss_min == float("inf")
    model = torch.nn.Linear(1, 1)
    for epoch in range(100):
        repaired(epoch, 0. if epoch == 0 else 1., model)
    assert not repaired.early_stop

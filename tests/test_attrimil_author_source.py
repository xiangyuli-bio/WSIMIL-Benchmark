from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Native original architecture, constraints, nine-neighbor geometry and queues."""

import copy
import queue
from types import SimpleNamespace

import numpy as np
import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from milplus.attrimil_data import RecordingWeightedSampler, SlideBags, nearest_indices, single_bag
from milplus.attrimil_driver import (
    make_model, new_queues, options, restore_queues, snapshot_queues,
    stop_under_author_rule, train_step,
)
from milplus.attrimil_source import contract, definitions
from milplus.attrimil_spec import DEFAULT_PROFILE, PROFILES
from milplus.data import Bundle
from milplus.engine import restore_rng, rng_state, seed_everything
from milplus.registry import inspect_method, profiles, require_runnable, resolve_for_bundle
from attrimil_helpers import equal_tree, literal_sources, make_task, original_neighbors


@pytest.mark.parametrize("dimension,classes", [(512, 2), (1024, 3)])
def test_native_long_bag_outputs_all_gradients_SGD_and_queue_rotation_equal_repaired_original(
        dimension, classes):
    torch.set_num_threads(2)
    literal = literal_sources()
    seed_everything(3901)
    model, optimizer, author = make_model(SimpleNamespace(dim=dimension, classes=list(range(classes))), "cpu")
    initial_rng = rng_state()
    seed_everything(3901)
    expected = literal[0].AttriMIL(n_classes=classes, dim=dimension)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=2e-4, momentum=.9, weight_decay=1e-5)
    equal_tree(initial_rng, rng_state())
    equal_tree(model.state_dict(), expected.state_dict())
    actual_queues = new_queues(classes)
    expected_queues = new_queues(classes)
    gradients, expected_gradients = [], []
    optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: gradients.append({n: p.grad.clone() if p.grad is not None else None
                                                   for n, p in model.named_parameters()}))
    expected_optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: expected_gradients.append({
            n: p.grad.clone() if p.grad is not None else None for n, p in expected.named_parameters()}))
    any_spatial, any_rank = False, False
    for step in range(3 * classes + 5):
        count = 1025 + step
        x = torch.randn(count, dimension)
        coords = torch.stack([torch.arange(count) % 33, torch.arange(count) // 33], -1) * 256
        nearest = nearest_indices(coords, [256, 256])
        target = torch.tensor([step % classes])
        saved = rng_state()
        equal_tree(model(x), expected(x))
        result = train_step(model, optimizer, author[1], actual_queues,
                            (x, target, coords, nearest), classes, "cpu")
        after = rng_state()
        restore_rng(saved)
        expected.train()
        logits, _, _, attributes, _ = expected(x)
        ce = torch.nn.functional.cross_entropy(logits, target)
        spatial = literal[1].spatial_constraint(attributes, classes, nearest, ks=3)
        rank, _, _ = literal[1].rank_constraint(x, target, expected, attributes, classes, *expected_queues)
        loss = ce + 1.0 * spatial + 5.0 * rank
        loss.backward()
        expected_optimizer.step()
        expected_optimizer.zero_grad()
        equal_tree(after, rng_state())
        equal_tree(model.state_dict(), expected.state_dict())
        equal_tree(optimizer.state_dict(), expected_optimizer.state_dict())
        equal_tree(snapshot_queues(actual_queues), snapshot_queues(expected_queues))
        equal_tree(gradients[-1], expected_gradients[-1])
        assert result["spatial"] == spatial.item() and result["rank"] == rank.item()
        any_spatial |= spatial.item() > 0
        any_rank |= rank.item() > 0
    assert any_spatial and any_rank
    assert any(q.full() for side in actual_queues for q in side)
    # This independent trainer checks all minibatch operations and creates its
    # own epoch-local queues; compare an additional whole epoch at native width.
    batches = []
    for i in range(classes + 2):
        x = torch.randn(31, dimension)
        coords = torch.stack([torch.arange(31) % 7, torch.arange(31) // 7], -1) * 256
        batches.append((x, torch.tensor([i % classes]), coords, nearest_indices(coords, [256, 256])))
    reset = new_queues(classes)
    saved = rng_state()
    for batch in batches:
        train_step(model, optimizer, author[1], reset, batch, classes, "cpu")
    after = rng_state()
    restore_rng(saved)
    literal[4].train_loop(0, expected, batches, expected_optimizer, classes, None,
                          torch.nn.CrossEntropyLoss())
    equal_tree(after, rng_state())
    equal_tree(model.state_dict(), expected.state_dict())
    equal_tree(optimizer.state_dict(), expected_optimizer.state_dict())


def test_actual_original_nine_neighbor_function_matches_lookup_for_holes_edges_and_row_order(tmp_path):
    coords = np.array([[0, 0], [256, 0], [0, 256], [512, 512], [256, 256],
                       [768, 256], [512, 256]], dtype=np.int64)
    expected = original_neighbors(coords, tmp_path / "original_neighbors.h5")
    actual = nearest_indices(torch.from_numpy(coords), [256, 256])
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert actual.shape == (7, 9) and torch.equal(actual[:, 0], torch.arange(7))
    assert actual[0, 1] == 0 and actual[0, 2] == 2 and actual[0, 4] == 1
    with pytest.raises(ValueError, match="distinct"):
        nearest_indices(torch.zeros(3, 2), [256, 256])


def test_original_sampler_weights_replacement_order_loader_RNG_and_recovered_queue_values(tmp_path):
    task = make_task(tmp_path / "task", train_slides=7)
    dataset = SlideBags(Bundle(task), "train")
    weights = definitions()[2].make_weights_for_balanced_classes_split(dataset)
    assert weights.dtype == torch.float64
    assert weights.tolist() == [7 / 4, 7 / 3, 7 / 4, 7 / 3, 7 / 4, 7 / 3, 7 / 4]
    seed_everything(88)
    expected_order = list(WeightedRandomSampler(weights, len(weights)))
    expected_rng = rng_state()
    seed_everything(88)
    sampler = RecordingWeightedSampler(weights, len(weights))
    assert list(sampler) == expected_order
    equal_tree(rng_state(), expected_rng)
    seed_everything(88)
    assert len(set(list(RecordingWeightedSampler(weights, len(weights))))) < len(weights)
    queues = new_queues(2)
    for i in range(4):
        queues[0][0].put(torch.arange(512, dtype=torch.float32)[None] + i)
    state = snapshot_queues(queues)
    recovered = restore_queues(state, 2, "cpu")
    equal_tree(snapshot_queues(recovered), state)
    assert recovered[0][0] is not queues[0][0] and recovered[0][0].maxsize == 4
    assert all(q.empty() for side in new_queues(2) for q in side)


def test_original_source_failures_preserved_and_only_named_repairs_make_the_epoch_run(tmp_path):
    x = torch.randn(17, 512)
    coords = torch.stack([torch.arange(17) % 5, torch.arange(17) // 5], -1) * 256
    nearest = nearest_indices(coords, [256, 256])
    batch = (x, torch.tensor([1]), coords, nearest)
    original = literal_sources(repair_axis=False, repair_logging=False)
    model = original[0].AttriMIL(n_classes=2, dim=512)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=.9, weight_decay=1e-5)
    with pytest.raises(IndexError):
        original[4].train_loop(0, model, [batch], optimizer, 2, None, torch.nn.CrossEntropyLoss())
    axis_only = literal_sources(repair_logging=False)
    with pytest.raises(NameError, match="loss_ins"):
        axis_only[4].train_loop(0, model, [batch], optimizer, 2, None, torch.nn.CrossEntropyLoss())
    assert not optimizer.state
    source = contract()
    assert source["commit"] == "e5736b9b4b5d3d4a64340b6ff5fa7fd470a338a9"
    assert source["function_defaults"]["max_epoch"] == 200
    assert source["function_defaults"]["early_stopping"] is True


@pytest.mark.parametrize("config", [
    {"epochs": 1}, {"lr": .01}, {"model": {"queue_size": 1}},
    {"model": {"spatial_weight": .1}}, {"model": {"rank_weight": .1}},
    {"model": {"attention_dim": 16}}, {"model": {"patch_cap": 32}},
    {"model": {"stop_after_train_batches": True}}, {"threads": True},
])
def test_complete_original_recipe_and_native_widths_cannot_be_reduced(config):
    with pytest.raises(ValueError):
        options(PROFILES[0], config)


def test_exact_author_earlystop_thresholds_and_complete_main_budget():
    main, function = PROFILES
    assert stop_under_author_rule(main, 199, 1000) is False
    assert stop_under_author_rule(function, 50, 21) is False
    assert stop_under_author_rule(function, 51, 20) is False
    assert stop_under_author_rule(function, 51, 21) is True
    assert main["native_recipe"]["epochs"] == function["native_recipe"]["epochs"] == 200


def test_old358_profiles_preserved_and_AttriMIL_display_chooses_complete_author_recipe():
    new_ids = {p["id"] for p in PROFILES}
    from milplus.carmil_spec import PROFILES as carmil_profiles
    car_ids = {p["id"] for p in carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}
    assert len([p for p in profiles() if p["id"] not in car_ids]) == 360
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(profiles()) == 394
    assert len([p for p in profiles() if p["id"] not in new_ids | car_ids]) == 358
    assert require_runnable("attrimil")["id"] == "attrimil"
    context = SimpleNamespace(classes=["normal", "tumor"], task_type="classification", config={})
    assert inspect_method("AttriMIL")["id"] == DEFAULT_PROFILE
    assert resolve_for_bundle("Attri-MIL", context)["id"] == DEFAULT_PROFILE

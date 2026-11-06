from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Original architecture, train-mode teacher, objective and exact update semantics."""

import copy
from types import SimpleNamespace

import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES
import torch

from milplus.asmil_driver import make_models, metric_runtime, options, train_step
from milplus.asmil_source import author_utilities, contract, definitions
from milplus.asmil_spec import DIMENSIONS, PROFILES, DEFAULT_PROFILE
from milplus.engine import restore_rng, rng_state, seed_everything
from milplus.registry import profiles, inspect_method, require_runnable, resolve_for_bundle
from asmil_helpers import equal_tree, literal_source


@pytest.mark.parametrize("dimension,classes,tokens,drop", [
    (384, 2, 1, 0), (512, 3, 3, 1), (768, 2, 2, 1),
    (1024, 3, 4, 2), (1536, 2, 1, 0),
])
@pytest.mark.parametrize("repair", [False, True])
def test_all_native_dimensions_exact_outputs_gradients_teacher_RNG_and_three_author_updates(
        dimension, classes, tokens, drop, repair):
    torch.set_num_threads(2)
    metric_runtime()
    profile = next(p for p in PROFILES if p["native_recipe"]["epochs"] == 50
                   and (p["teacher_update"] == "finite_ema_repaired") == repair)
    cfg, _ = options(profile, {"model": {"n_token": tokens, "n_drop": drop, "temp": .5}})
    recipe = cfg["recipe"]
    literal, trainer = literal_source(repair_teacher=repair)
    bundle = SimpleNamespace(dim=dimension, classes=list(range(classes)))
    seed_everything(3901)
    online, teacher, optimizer, conf, source = make_models(bundle, recipe, "cpu")
    initial_rng = rng_state()
    seed_everything(3901)
    expected_online = literal.ACMIL_MYMHA(conf, n_token=tokens, n_drop=drop)
    expected_teacher = literal.ACMIL_MYMHA_EMA(conf, n_token=tokens)
    expected_optimizer = torch.optim.AdamW(expected_online.parameters(), lr=.001, weight_decay=1e-5)
    equal_tree(rng_state(), initial_rng)
    equal_tree(online.state_dict(), expected_online.state_dict())
    equal_tree(teacher.state_dict(), expected_teacher.state_dict())
    assert online.dimreduction.fc1.weight.shape == (DIMENSIONS[dimension], dimension)
    assert online.sub_attention[0].num_heads == 8
    assert online.sub_attention[0].dropout.p == .1
    assert not torch.equal(online.dimreduction.fc1.weight, teacher.dimreduction.fc1.weight)
    initial_teacher = copy.deepcopy(teacher.state_dict())
    for step in range(3):
        batch = {"input": torch.randn(1, 1025 + step, dimension),
                 "label": torch.tensor([step % classes])}
        saved = rng_state()
        train_step(online, teacher, optimizer, conf, source, batch, "cpu",
                   step, .5, profile["teacher_update"], author_utilities().adjust_learning_rate)
        after = rng_state()
        restore_rng(saved)
        trainer.train_one_epoch(expected_online, expected_teacher, torch.nn.CrossEntropyLoss(),
                                [batch], expected_optimizer, "cpu", step, conf, .5)
        equal_tree(after, rng_state())
        equal_tree(online.state_dict(), expected_online.state_dict())
        equal_tree(teacher.state_dict(), expected_teacher.state_dict())
        equal_tree(optimizer.state_dict(), expected_optimizer.state_dict())
        for (name, p), (other, q) in zip(online.named_parameters(), expected_online.named_parameters()):
            assert name == other
            equal_tree(p.grad, q.grad)
    if repair:
        assert not torch.equal(teacher.q, initial_teacher["q"])
    else:
        equal_tree(teacher.state_dict(), initial_teacher)


def test_literal_teacher_really_stays_independent_on_finite_states_and_schedule_is_per_batch():
    p = PROFILES[0]
    conf, _ = options(p, {})
    source = contract(p)
    assert source["commit"] == "5d062271e1626b27a955c309d10add902311865c"
    assert source["architecture_definition_replacements"] == []
    assert "NaN" not in source["literal_teacher_condition"]  # exact source expression is lowercase
    assert source["literal_teacher_condition"].startswith("torch.isnan")
    a, _, optim, cfg, _ = make_models(SimpleNamespace(dim=1024, classes=["a", "b"]), conf["recipe"], "cpu")
    schedule = author_utilities().adjust_learning_rate
    assert optim.param_groups[0]["lr"] == .001
    assert schedule(optim, 0., cfg) == 1e-4
    halfway = schedule(optim, 25., cfg)
    assert halfway == pytest.approx(5e-5)
    assert schedule(optim, 49.75, cfg) > 0
    assert a.dimreduction.numRes == 0 and a.dimreduction.bn.normalized_shape == (512,)


@pytest.mark.parametrize("config", [
    {"epochs": 1}, {"batch_size": 2}, {"model": {"inner_dim": 8}},
    {"model": {"patch_cap": 32}}, {"model": {"n_token": 0}},
    {"model": {"n_token": True}}, {"model": {"n_drop": 1}},
    {"model": {"temp": 0}}, {"model": {"lr": float("nan")}},
    {"threads": True}, {"model": {"stop_after_train_batches": 0}},
])
def test_no_shrinking_native_budget_dimensions_or_invalid_author_choices(config):
    with pytest.raises(ValueError):
        options(PROFILES[0], config)


def test_declared_metric_defaults_are_actual_micro_F1_and_macro_AUC():
    tm = metric_runtime()
    labels = torch.tensor([0, 0, 0, 1, 2])
    probability = torch.tensor([[.9, .05, .05], [.8, .1, .1], [.7, .2, .1],
                                [.7, .2, .1], [.1, .8, .1]])
    default = tm.F1Score(task="multiclass", num_classes=3)(probability, labels)
    micro = tm.F1Score(task="multiclass", num_classes=3, average="micro")(probability, labels)
    macro = tm.F1Score(task="multiclass", num_classes=3, average="macro")(probability, labels)
    assert torch.equal(default, micro) and not torch.equal(default, macro)
    metric = tm.AUROC(task="multiclass", num_classes=3)
    assert metric.average == "macro"


def test_previous354_profiles_unchanged_new4_are_explicit_and_display_keeps_legacy_technical_ID():
    new_ids = {p["id"] for p in PROFILES}
    from milplus.carmil_spec import PROFILES as carmil_profiles
    car_ids = {p["id"] for p in carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES}
    from milplus.attrimil_spec import PROFILES as newest_profiles
    newest_ids = {p["id"] for p in newest_profiles}
    assert len([p for p in profiles() if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES}]) == 388
    assert len(new_ids) == 4 and len([p for p in profiles() if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(profiles()) == 394
    assert len([p for p in profiles() if p["id"] not in car_ids]) == 360
    assert len([p for p in profiles() if p["id"] not in newest_ids | car_ids]) == 358
    assert len([p for p in profiles() if p["id"] not in new_ids | newest_ids | car_ids]) == 354
    assert require_runnable("asmil")["id"] == "asmil"
    context = SimpleNamespace(classes=["a", "b"], task_type="classification", config={})
    for display in ["ASMIL", "AS-MIL"]:
        assert inspect_method(display)["id"] == DEFAULT_PROFILE
        assert resolve_for_bundle(display, context)["id"] == DEFAULT_PROFILE
    for p in PROFILES:
        assert require_runnable(p["id"])["teacher_update"] == p["teacher_update"]
        assert contract(p)["config"]["train_epoch"] in {50, 100}

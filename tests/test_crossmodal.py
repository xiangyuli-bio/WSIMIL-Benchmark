"""Cross-modal mechanisms, author numerical equivalence, and streamed patient objectives."""
import copy
import random
import sys
import types
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch

from milplus.crossmodal import alignment_loss, micro_indices
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.discrete_survival import discrete_epoch, discrete_nll
from milplus.engine import evaluate, train, units
from milplus.io import read_json
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, require_runnable
from milplus.stochastic import inference_rng
from milplus.upstream import load_crossmodal, pinned_text, source_root, verified_source


def source(method):
    name = 'motcat' if method.startswith('motcat') else 'cmta'
    if not source_root(name).is_dir():
        pytest.skip(f'Fetch pinned author code with milplus setup --method {name}')
    if name == 'motcat':
        pytest.importorskip('ot')
    torch.set_num_threads(1)
    return name


def model_and_bundle(tmp_path, method, **options):
    source(method)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, survival=True, omics=True))
    model = build_bundle_model(method, bundle, {'bins': 3, **options})
    model.prepare(bundle, 2026, 'cpu')
    return model, bundle


def test_alignment_targets_are_detached_but_translation_gradients_remain():
    path, translated_path, molecular, translated_molecular = [torch.randn(1, 8, requires_grad=True) for _ in range(4)]
    loss = alignment_loss(path, translated_path, molecular, translated_molecular)
    expected = (translated_path-path).abs().mean() + (translated_molecular-molecular).abs().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert path.grad is None and molecular.grad is None
    assert translated_path.grad.abs().sum() > 0 and translated_molecular.grad.abs().sum() > 0


@pytest.mark.parametrize('method', ['cmta', 'motcat', 'motcat_micro'])
def test_crossmodal_train_reload_and_exact_resume(tmp_path, method):
    source(method)
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    options = {'bins': 3, 'accumulation_steps': 5}
    if method == 'motcat_micro':
        options['micro_batch_size'] = 16
    config = {'epochs': 2, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=config)
    split = train(root, method, tmp_path/'split', config=config, stop_after=1)
    train(root, method, split, config=config, resume=True)
    left, right = [torch.load(p/'last.pt', weights_only=False) for p in (full, split)]
    assert left['history'] == right['history']
    for key in left['model']:
        torch.testing.assert_close(left['model'][key], right['model'][key], atol=0, rtol=0)
    assert not (full/'test_predictions.csv').exists()
    evaluate(full)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))
    assert len(first) == Bundle(root).rows.query("subset == 'test'").global_patient_id.nunique()
    survival = first[[f'survival_{i}' for i in range(3)]].to_numpy()
    hazards = first[[f'hazards_{i}' for i in range(3)]].to_numpy()
    np.testing.assert_allclose(survival, np.cumprod(1-hazards, axis=1), atol=2e-7)
    np.testing.assert_allclose(first.risk, -survival.sum(1), atol=2e-7)
    protocol = read_json(full/'survival_protocol.json')
    if method == 'cmta':
        assert protocol['alignment_weight'] == 1.
    else:
        assert protocol['unbalanced_regularizer'].startswith('entropy')


@pytest.mark.parametrize('method', ['cmta', 'motcat'])
@pytest.mark.parametrize('fusion', ['concat', 'bilinear'])
def test_single_instance_different_dimension_and_both_modalities(tmp_path, method, fusion):
    model, bundle = model_and_bundle(tmp_path, method, fusion=fusion)
    model.eval()
    x, omics = torch.randn(1, bundle.dim, requires_grad=True), torch.randn(18, requires_grad=True)
    result = model(x, omics)
    model.objective(result, 2, 1).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(omics.grad).all() and omics.grad.abs().sum() > 0
    assert not torch.allclose(result['hazard_logits'], model(x+2, omics)['hazard_logits'])
    assert not torch.allclose(result['hazard_logits'], model(x, omics+2)['hazard_logits'])


@pytest.mark.parametrize('impl', ['pot-uot-l2', 'pot-sinkhorn-l2'])
def test_transport_handles_zero_cost_and_preserves_value_gradients(impl):
    source('motcat')
    transport = load_crossmodal('motcat').OT_Attn_assem(impl=impl)
    for count in (1, 7):
        x, y = torch.zeros(count, 1, 8, requires_grad=True), torch.zeros(6, 1, 8, requires_grad=True)
        flow, distance = transport(x, y)
        assert flow.shape == (1, 1, 6, count) and torch.isfinite(flow).all()
        assert not flow.requires_grad and distance.item() == 0
        values = torch.randn(count, 4, requires_grad=True)
        (flow[0, 0] @ values).square().sum().backward()
        assert values.grad.abs().sum() > 0


def test_unbalanced_solver_matches_legacy_entropy_scaling_not_new_kl_default():
    source('motcat')
    import ot
    torch.manual_seed(4)
    x, y = torch.rand(9, 8), torch.rand(6, 8)
    transport = load_crossmodal('motcat').OT_Attn_assem()
    flow, _ = transport.OT(x, y)
    cost = torch.cdist(x, y).square().double()
    cost /= cost.max()
    a, b = torch.full((9,), 1/9, dtype=torch.float64), torch.full((6,), 1/6, dtype=torch.float64)
    kernel = (-cost/.1).exp()
    u, v = torch.ones_like(a), torch.ones_like(b)
    # Independent fixed-point equations for the legacy entropy kernel exp(-C/reg).
    for _ in range(2000):
        u = (a / (kernel @ v)) ** (.5/.6)
        v = (b / (kernel.T @ u)) ** (.5/.6)
    expected = u[:, None] * kernel * v[None, :]
    torch.testing.assert_close(flow, expected.float(), rtol=2e-5, atol=1e-7)
    changed_default = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost, reg=.1, reg_m=.5)
    assert not torch.allclose(flow, changed_default.float(), rtol=1e-3, atol=1e-4)


def test_micro_partition_covers_each_instance_and_preserves_author_boundary():
    for count, size in [(1, 2), (12, 4), (13, 4), (500, 4096)]:
        chunks = micro_indices(count, size)
        assert len(chunks) == count // size + 1
        assert sorted(i for chunk in chunks for i in chunk) == list(range(count))
        assert all(0 < len(chunk) <= size for chunk in chunks)
    with pytest.raises(ValueError, match='>= 2'):
        micro_indices(1, 1)


def test_micro_mean_loss_streamed_gradient_matches_joint_author_objective(tmp_path):
    model, bundle = model_and_bundle(tmp_path, 'motcat_micro', micro_batch_size=11, accumulation_steps=2)
    reference = copy.deepcopy(model)
    groups = units(bundle, 'train')[:3]  # includes a final partial accumulation window
    ours = torch.optim.SGD(model.parameters(), lr=.001)
    expected = torch.optim.SGD(reference.parameters(), lr=.001)
    config = {'max_instances': 30, 'grad_clip': 1e5}
    with inference_rng('cpu', 14):
        actual_loss = discrete_epoch(model, bundle, groups, ours, 'cpu', config)
    losses = []
    with inference_rng('cpu', 14):
        for window in (groups[:2], groups[2:]):
            expected.zero_grad()
            objectives = []
            for rows in window:
                result = reference.forward_unit(bundle, rows, 'cpu', config, train=True)
                logits = result['micro_hazard_logits']
                objective = discrete_nll(logits, reference.time_label([rows[0]['time']]).expand(len(logits)),
                                         logits.new_full((len(logits),), rows[0]['event']))
                objectives.append(objective)
                losses.append(float(objective.detach()))
            torch.stack(objectives).mean().backward()
            expected.step()
    assert actual_loss == pytest.approx(np.mean(losses), rel=1e-6)
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)


def test_micro_inference_is_deterministic_and_does_not_consume_training_rng(tmp_path):
    model, _ = model_and_bundle(tmp_path, 'motcat_micro', micro_batch_size=5)
    model.eval()
    x, omics = torch.randn(17, 16), torch.randn(18)
    state, numpy_state, tensor_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    first, second = model(x, omics), model(x, omics)
    assert random.getstate() == state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), tensor_state)
    torch.testing.assert_close(first['survival'], second['survival'], rtol=0, atol=0)
    constituent_survival = (-torch.nn.functional.softplus(first['micro_hazard_logits'])).cumsum(1).exp()
    torch.testing.assert_close(first['survival'], constituent_survival.mean(0, keepdim=True))
    expected_loss = discrete_nll(first['micro_hazard_logits'], model.time_label([3]).expand(4), torch.ones(4))
    torch.testing.assert_close(model.objective(first, 3, 1), expected_loss)


@pytest.mark.parametrize('method', ['cmta', 'motcat', 'motcat_micro'])
def test_six_signatures_required_before_training(tmp_path, method):
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, survival=True, omics=True))
    bundle.omics.groups = bundle.omics.groups[:2]
    with pytest.raises(ValueError, match='six'):
        check_compatibility(require_runnable(method), bundle)


def reference_network(source_name):
    """Import independent unmodified author model with CPU/POT API shims only."""
    root, sha = verified_source(source_name)
    package_name = '_test_author_reference_' + source_name
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    pairs = [('util', 'models/cmta/util.py'), ('model', 'models/cmta/network.py')] if source_name == 'cmta' else [
        ('model_utils', 'models/model_utils.py'), ('model', 'models/model_motcat.py')]
    for name, path in pairs:
        module = types.ModuleType(package_name + '.' + name)
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        text = pinned_text(root, sha, path).replace('from models.', 'from ' + package_name + '.')
        exec(compile(text, str(root/path), 'exec'), module.__dict__)  # noqa: S102 -- independently verified author Git blobs
    if source_name == 'motcat':
        import ot
        module.ot = types.SimpleNamespace(unif=ot.unif, unbalanced=types.SimpleNamespace(
            sinkhorn_knopp_unbalanced=partial(ot.unbalanced.sinkhorn_knopp_unbalanced, reg_type='entropy')))
    return module


@pytest.mark.parametrize('method', ['cmta', 'motcat'])
def test_complete_author_forward_and_loss_gradients_match(tmp_path, method, monkeypatch):
    source(method)
    # Compare at the author's original 1024-dimensional input, with identical parameters.
    author = reference_network(method)
    cls = author.CMTA if method == 'cmta' else author.MOTCAT_Surv
    reference = cls(omic_sizes=[3]*6, n_classes=3)
    adapted_cls = load_crossmodal(method).CMTA if method == 'cmta' else load_crossmodal(method).MOTCAT_Surv
    adapted = adapted_cls(omic_sizes=[3]*6, n_classes=3)
    adapted.load_state_dict(reference.state_dict())
    reference.eval()
    adapted.eval()
    inputs = {'x_path': torch.randn(11, 1024), **{f'x_omic{i+1}': torch.randn(3) for i in range(6)}}
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor)
    original, actual = reference(**inputs), adapted(**inputs)
    for old, new in zip(original[:2], actual[1:3]):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    nll = -(torch.log(original[1][:, 0]) + torch.log(original[0][:, 1])).mean()
    ours = discrete_nll(actual[0], torch.tensor([1]), torch.tensor([1]))
    if method == 'cmta':
        p, p_hat, g, g_hat = original[2:]
        nll = nll + torch.nn.functional.l1_loss(p.detach(), p_hat) + torch.nn.functional.l1_loss(g.detach(), g_hat)
        ours = ours + alignment_loss(*actual[3:])
    torch.testing.assert_close(ours, nll, rtol=1e-6, atol=1e-7)
    ours.backward()
    nll.backward()
    for name, parameter in adapted.named_parameters():
        expected = dict(reference.named_parameters())[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=2e-7)

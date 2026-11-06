"""Variational objectives, real APCRF rasterization, context tokens and VarMIL."""
import copy
import math
import numbers
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from milplus.bayesian import BAYES_METHODS, BayesMIL, canvas_shape, load_bayes, validate_canvas
from milplus.context_variance import CAPR_METHODS, ContextPatchMIL, VarianceMIL, load_capr
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.registry import check_compatibility, require_runnable
from milplus.stochastic import inference_rng
from milplus.upstream import load_definitions, source_root


def source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch the pinned '+name+' source')
    torch.set_num_threads(1)


def raw_bayes():
    linear = load_definitions('bayesmil', 'models/linear_vdo.py', ['LinearVDO', 'Conv2dVDO'],
        {'torch': torch, 'nn': nn, 'F': F, 'Parameter': nn.Parameter, 'eps': 1e-8})
    init = load_definitions('bayesmil', 'utils/utils.py', ['initialize_weights'], {'nn': nn}).initialize_weights
    return load_definitions('bayesmil', 'models/model_bmil.py',
        ['Attn_Net', 'Attn_Net_Gated', 'DAttn_Net_Gated', 'GaussianSmoothing',
         'probabilistic_MIL_Bayes_vis', 'probabilistic_MIL_Bayes_enc',
         'probabilistic_MIL_Bayes_spvis', 'get_ard_reg_vdo'],
        {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'numbers': numbers, 'np': np,
         'LinearVDO': linear.LinearVDO, 'Conv2dVDO': linear.Conv2dVDO,
         'initialize_weights': init, 'PATCH_SIZE': 256})


@pytest.mark.parametrize('profile', sorted(BAYES_METHODS))
def test_bayes_author_outputs_losses_and_gradients(profile):
    source('bayesmil')
    reference = raw_bayes()
    cls = {'bayesmil_vis': reference.probabilistic_MIL_Bayes_vis,
           'bayesmil_sdpr': reference.probabilistic_MIL_Bayes_enc,
           'bayesmil_apcrf': reference.probabilistic_MIL_Bayes_spvis}[profile]
    original = cls(dropout=True)
    model = BayesMIL(1024, 2, profile)
    model.network.load_state_dict(original.state_dict(), strict=False)
    x = torch.randn(7, 1024)
    label = torch.tensor([1])
    # First raster row makes corrected/uncorrected stride identical, isolating
    # the unmodified stochastic architecture/objective from the explicit repair.
    coords = torch.stack([torch.arange(len(x))*256, torch.zeros(len(x))], 1).long()
    torch.manual_seed(35)
    if profile == 'bayesmil_vis':
        out = original(x)
    elif profile == 'bayesmil_sdpr':
        out = original(x, slide_label=label)
    else:
        out = original(x, coords.numpy(), 256, 1792, slide_label=label)
    loss = F.cross_entropy(out[0], label)+1e-8*reference.get_ard_reg_vdo(original)
    if profile != 'bayesmil_vis':
        loss = loss+1e-6*out[3].mean()
    torch.manual_seed(35)
    actual = model(x, coords, label, (1792, 256))
    torch.testing.assert_close(out[0], actual['logits'], atol=0, rtol=0)
    torch.testing.assert_close(loss, actual['total_loss'], atol=0, rtol=0)
    loss.backward(); actual['total_loss'].backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), original.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        if name.endswith('log_alp'):
            assert p.grad.abs().sum() > 0


def test_variational_zero_input_has_finite_gradient_without_changing_forward():
    source('bayesmil')
    raw = raw_bayes().LinearVDO(8, 3)
    fixed = load_bayes().LinearVDO(8, 3)
    fixed.load_state_dict(raw.state_dict())
    x = torch.cat([torch.zeros(1, 8), torch.randn(2, 8)]).requires_grad_()
    torch.manual_seed(99)
    expected = raw(x)
    torch.manual_seed(99)
    actual = fixed(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in fixed.parameters())
    # The unmodified author backward has inf * zero at this exact zero row.
    expected.sum().backward()
    assert any(not torch.isfinite(p.grad).all() for p in raw.parameters())


def test_apcrf_two_dimensional_mapping_matches_explicit_grid_and_gradients():
    source('bayesmil')
    model = BayesMIL(16, 2, 'bayesmil_apcrf', hidden=32, attention_dim=8)
    ref = copy.deepcopy(model.network)
    x = torch.randn(5, 16)
    coords = torch.tensor([[0, 0], [256, 0], [0, 256], [256, 256], [768, 512]])
    torch.manual_seed(43)
    h = F.relu(ref.dp_0(ref.linear1(x)))
    params = ref.linear3(ref.dp_a(torch.sigmoid(ref.linear2a(h)))*ref.dp_b(torch.tanh(ref.linear2b(h))))
    mu, logvar = torch.zeros(1, 4, 5), torch.zeros(1, 4, 5)
    for i, (a, b) in enumerate((coords//256).tolist()):
        mu[0, b, a], logvar[0, b, a] = params[i, 0], params[i, 1]
    regularizer = ref.kl_logistic_normal(ref.prior_mu[1].expand_as(mu), mu, ref.prior_logvar[1], logvar).mean()
    smooth = ref.gaus_smoothing(F.pad(mu, (1, 1, 1, 1))[None])
    attention = torch.sigmoid(ref.reparameterize(smooth, logvar)).squeeze(0)
    weights = torch.stack([attention[0, b, a] for a, b in (coords//256).tolist()])[None]
    logits = ref.classifiers(weights@h/weights.sum())
    loss = F.cross_entropy(logits, torch.tensor([1]))+1e-8*model.author.get_ard_reg_vdo(ref)+1e-6*regularizer
    torch.manual_seed(43)
    actual = model(x, coords, torch.tensor([1]), (1024, 768))
    torch.testing.assert_close(actual['logits'], logits, atol=0, rtol=0)
    torch.testing.assert_close(actual['attention'], weights, atol=0, rtol=0)
    torch.testing.assert_close(actual['total_loss'], loss, atol=0, rtol=0)
    actual['total_loss'].backward(); loss.backward()
    for p, q in zip(model.network.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize('profile', sorted(BAYES_METHODS))
def test_mc_average_probability_and_uncertainty_preserve_rng(profile):
    source('bayesmil')
    model = BayesMIL(16, 2, profile, hidden=32, attention_dim=8, mc_samples=5).eval()
    x, c = torch.randn(6, 16), torch.tensor([[i%3*256, i//3*256] for i in range(6)])
    with inference_rng('cpu', model.inference_seed):
        draws = torch.stack([model.sample(x, c, canvas=(768, 512))['probabilities'] for _ in range(5)])
    rng = torch.get_rng_state().clone()
    prediction = model(x, c, canvas=(768, 512))
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(prediction['probabilities'], draws.mean(0), atol=0, rtol=0)
    assert not torch.equal(prediction['probabilities'], draws[-1])
    assert (prediction['mutual_information'] >= -1e-7).all()
    torch.testing.assert_close(prediction['predictive_entropy'],
        prediction['expected_entropy']+prediction['mutual_information'])
    torch.testing.assert_close(prediction['probabilities'], model(x, c, canvas=(768, 512))['probabilities'], atol=0, rtol=0)


def test_sdpr_preserves_author_formula_and_label_conditioning():
    source('bayesmil')
    model = BayesMIL(16, 2, 'bayesmil_sdpr', hidden=32, attention_dim=8)
    x = torch.randn(5, 16)
    torch.manual_seed(75); positive = model(x, label=torch.tensor([1]))
    torch.manual_seed(75); negative = model(x, label=torch.tensor([0]))
    torch.testing.assert_close(positive['logits'], negative['logits'], atol=0, rtol=0)
    assert positive['patch_regularizer'].item() != negative['patch_regularizer'].item()
    # Literal source expression is intentionally not renamed as verified analytic KL.
    mu, logvar = torch.tensor([2.]), torch.tensor([.3])
    actual = model.network.kl_logistic_normal(torch.tensor(-5.), mu, torch.tensor(-1.), logvar)
    torch.testing.assert_close(actual, (-1-logvar)/2+(logvar.square()+(-5-mu).square())/2-.5)


def test_real_canvas_alignment_collision_budget_and_metadata(tmp_path):
    source('bayesmil')
    with pytest.raises(ValueError, match='width_level0'):
        canvas_shape({})
    c = torch.tensor([[0, 0], [256, 0]])
    with pytest.raises(ValueError, match='outside'):
        validate_canvas(c, 200, 512, 256, 100)
    with pytest.raises(ValueError, match='same APCRF'):
        validate_canvas(torch.tensor([[0, 0], [128, 0]]), 512, 512, 256, 100)
    with pytest.raises(ValueError, match='max_grid_cells'):
        validate_canvas(c, 5120, 5120, 256, 1)
    root = create_demo(tmp_path/'data', patients=20)
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['encoders']['toy']['spatial']['origin'] = 'center'
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='top_left'):
        check_compatibility(require_runnable('bayesmil_apcrf'), Bundle(root))


@pytest.mark.parametrize('profile', sorted(CAPR_METHODS))
@pytest.mark.parametrize('dim', [16, 32])
def test_capr_author_logits_objective_gradients_and_context_parameters(profile, dim):
    source('caprmil')
    mode = {'caprmil_mean': 'mean', 'caprmil_attention': 'attn', 'caprmil_gated': 'gated_attn'}[profile]
    model = ContextPatchMIL(dim, 3, profile, hidden=32, heads=4, clusters=3)
    original = load_capr()(dim, 32, num_heads=4, cluster_num=3, aggregator=mode, num_classes=3)
    original.load_state_dict(model.network.state_dict())
    x = torch.randn(9, dim)
    torch.manual_seed(53); expected = original(x[None])
    torch.manual_seed(53); actual = model(x)['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    y = torch.tensor([2]); F.cross_entropy(actual, y).backward(); F.cross_entropy(expected, y).backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), original.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        if name.endswith(('temperature', 'in_project_slice.weight', 'to_q.weight', 'to_out.0.weight')):
            assert p.grad.abs().sum() > 0, name
    model.eval()
    torch.testing.assert_close(model(x)['logits'], model(x[torch.randperm(len(x))])['logits'], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('scheduler', ['cosine', 'linearcosine'])
def test_capr_author_schedule_and_restore(scheduler):
    source('caprmil')
    from milplus.context_variance import capr_scheduler
    model = ContextPatchMIL(16, 2, hidden=32, scheduler=scheduler, warmup_epochs=3)
    model.configure_training({'epochs': 8, 'lr': .001})
    optimizer = model.make_optimizer(.001, .0001)
    p = nn.Parameter(torch.ones(1)); expected_opt = torch.optim.AdamW([p], lr=.001, weight_decay=.0001)
    if scheduler == 'cosine':
        expected = torch.optim.lr_scheduler.CosineAnnealingLR(expected_opt, 8, eta_min=1e-7)
    else:
        expected = capr_scheduler()(expected_opt, 3, 8, warmup_start_lr=1e-5, eta_min=1e-7)
    for _ in range(8):
        assert optimizer.param_groups[0]['lr'] == expected_opt.param_groups[0]['lr']
        optimizer.step(); expected_opt.step(); model.after_validation({}); expected.step()
    snapshot = copy.deepcopy(model.state_dict())
    restored = ContextPatchMIL(16, 2, hidden=32, scheduler=scheduler, warmup_epochs=3)
    restored.configure_training({'epochs': 8, 'lr': .001}); new_opt = restored.make_optimizer(.001, .0001)
    restored.load_state_dict(snapshot); new_opt.load_state_dict(optimizer.state_dict())
    assert restored.scheduler.state_dict() == model.scheduler.state_dict()


def original_varmil():
    class Base(nn.Module):
        def save_hyperparameters(self):
            pass
    return load_definitions('varmil', 'dlup_lightning_mil/models/heads/deepmil.py', ['Attention', 'VarAttention'],
        {'torch': torch, 'F': F, 'nn': nn, 'LightningModule': Base,
         'torchmetrics': SimpleNamespace(AUROC=nn.Identity, F1=nn.Identity, PrecisionRecallCurve=nn.Identity)}).VarAttention


@pytest.mark.parametrize('dim', [16, 32])
def test_varmil_author_weighted_variance_loss_and_gradients(dim):
    source('varmil')
    model = VarianceMIL(dim, 3, hidden=8)
    original = original_varmil()(lr=.001, hidden_features=8, in_features=dim, num_classes=3, weight_decay=0.)
    original.load_state_dict(model.network.state_dict())
    x = torch.randn(8, dim)
    actual = model(x); expected, attention = original(x[None])
    torch.testing.assert_close(actual['logits'], expected, atol=0, rtol=0)
    mean = attention@x
    variance = ((x-mean).square()*attention.T).sum(0, keepdim=True)*len(x)/(len(x)-1)
    torch.testing.assert_close(model.network.compute_weighted_std(attention, x, mean), variance)
    F.cross_entropy(actual['logits'], torch.tensor([2])).backward()
    F.cross_entropy(expected, torch.tensor([2])).backward()
    for p, q in zip(model.network.parameters(), original.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    torch.testing.assert_close(model.network.compute_weighted_std(attention, torch.ones_like(x), torch.ones_like(mean)), torch.zeros_like(mean))
    with pytest.raises(ValueError, match='two instances'):
        model(x[:1])


@pytest.mark.parametrize('method', sorted(BAYES_METHODS | CAPR_METHODS | {'varmil'}))
def test_new_profile_exact_resume_reload_and_metadata_integrity(tmp_path, method):
    source('bayesmil' if method in BAYES_METHODS else 'caprmil' if method in CAPR_METHODS else method)
    root = create_demo(tmp_path/'data', patients=20)
    options = {'hidden': 32}
    if method in BAYES_METHODS:
        options.update(attention_dim=8, mc_samples=3, weighted_sampling=True, plateau_patience=0)
    cfg = {'epochs': 3, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=cfg)
    partial = train(root, method, tmp_path/'partial', config=cfg, stop_after=1)
    train(root, method, partial, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    assert a['history'] == b['history']
    assert a['model'].pop('_extra_state', None) == b['model'].pop('_extra_state', None)
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    assert evaluate(full) == evaluate(partial)
    if method in BAYES_METHODS:
        predictions = pd.read_csv(full/'test_predictions.csv')
        assert {'predictive_entropy', 'expected_entropy', 'mutual_information'} <= set(predictions)
        assert predictions['mutual_information'].min() >= -1e-6
    if method == 'bayesmil_apcrf':
        table = pd.read_csv(root/'slides.csv'); table.loc[0, 'width_level0'] += 256
        table.to_csv(root/'slides.csv', index=False)
        with pytest.raises(ValueError, match='inputs differ'):
            evaluate(full)

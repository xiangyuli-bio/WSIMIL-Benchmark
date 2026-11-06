"""Pinned survival architectures, full objectives, patient boundaries and resume."""
import copy
import math

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from milplus.cluster_survival import DeepAttentionSurvival, deepattn_epoch, load_deepattn, patient_cox_mean
from milplus.continuous_survival import CPUKMeans, SparseContextSurvival, load_scmil, spatial_patient_bags
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import load_definitions, source_root


def source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch pinned '+name+' source')
    torch.set_num_threads(1)


def raw_scmil():
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math,
             'rearrange': rearrange, 'KMeans': CPUKMeans}
    scope['SoftFilterLayer'] = load_definitions('scmil', 'models/SoftFilter.py', ['SoftFilterLayer'], scope).SoftFilterLayer
    scope['AMIL_layer'] = load_definitions('scmil', 'models/AMIL.py', ['AMIL_layer'], scope).AMIL_layer
    backbone = load_definitions('scmil', 'models/SCMIL.py', ['SelfAttention', 'ClusterLocalAttention', 'SCMIL'], scope).SCMIL
    mdn = load_definitions('scmil', 'models/MDN.py',
        ['safe_inverse_softplus', 'inverse_softplus_grad', 'logsumexp', 'MLP', 'SurvivalMDN'], scope)
    return backbone, mdn


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('event', [0, 1])
def test_scmil_author_mixture_censor_likelihood_and_parameter_gradients(dim, event):
    source('scmil')
    model = SparseContextSurvival(dim, hidden=16, components=4, cluster_size=4).eval()
    with torch.no_grad():
        model.network.backbone.softfilter.layers[-2].bias.fill_(2.)
        model.network.layer_mu[0].weight.copy_(torch.eye(4)*.01)
        model.network.layer_mu[0].bias.fill_(2.)
        model.network.layer_sigma[0].weight.copy_(torch.eye(4)*.1)
        model.network.layer_sigma[0].bias.fill_(1.)
        model.prepared.fill_(True)
        model.curve_times.copy_(torch.linspace(0, 15, 64))
    backbone, raw = raw_scmil()
    ref = raw.SurvivalMDN(backbone=backbone(input_size=dim, hidden_size=dim, deep=1,
        cluster_size=4, feature_weight=.8, as_backbone=True, dropout_rate=.25, with_softfilter=True),
        input_size=dim, hidden_size=16, K=4, param_share=['mu', 'sigma']).eval()
    ref.load_state_dict(model.network.state_dict())
    x, coords = torch.randn(19, dim), torch.randn(19, 2)*100
    actual = model([(x, coords)])
    params, _ = ref(x[None], coords=coords[None])
    for key in ['w', 'mu', 'sigma']:
        torch.testing.assert_close(actual['params'][0][key], params[key], atol=0, rtol=0)
    t = torch.tensor([5.])
    survival, pdf = 1-ref.cdf(params, t), ref.log_prob(params, t).exp()
    criterion = load_definitions('scmil', 'utils/loss_funcs.py', ['MDNML'], {'torch': torch}).MDNML()
    expected = criterion({'c': torch.tensor([1-event]), 'pdf': pdf, 'survival_func': survival}).mean()
    loss = model.objective(actual, 5., event)
    torch.testing.assert_close(loss, expected)
    loss.backward(); expected.backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), ref.named_parameters()):
        if name.startswith(('backbone.fc1.', 'backbone.attens.0.atten.')):
            assert p.grad is None and q.grad is None, name
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-4)
    assert model.network.backbone.softfilter.layers[-2].weight.grad.abs().sum() > 0
    assert model.network.backbone.attens[0].to_qkv.weight.grad.abs().sum() > 0
    assert model.network.mlp.fc.weight.grad.abs().sum() > 0


def test_scmil_long_followup_stability_and_original_formula_equivalence():
    source('scmil')
    _, fixed = load_scmil(); _, raw = raw_scmil()
    t = torch.tensor([.1, 1., 10., 70.], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(fixed.safe_inverse_softplus(t), raw.safe_inverse_softplus(t))
    torch.testing.assert_close(fixed.inverse_softplus_grad(t), raw.inverse_softplus_grad(t))
    long = torch.tensor([100., 500., 1000.], requires_grad=True)
    assert not torch.isfinite(raw.safe_inverse_softplus(long)).all()
    value = fixed.safe_inverse_softplus(long)+fixed.inverse_softplus_grad(long)
    assert torch.isfinite(value).all()
    value.sum().backward(); assert torch.isfinite(long.grad).all()


def test_scmil_patient_mixture_boundaries_grid_and_budget(tmp_path):
    source('scmil')
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, survival=True))
    model = SparseContextSurvival(16, hidden=16, components=5, cluster_size=8).eval()
    model.prepare(bundle, 2, 'cpu')
    assert model.curve_times[-1] == bundle.rows[bundle.rows.subset == 'train'].time.max()
    rows = next(rows for rows in units(bundle, 'train') if len(rows) > 1)
    bags = spatial_patient_bags(bundle, rows, 'cpu', 8)
    assert len(bags) == 2 and sum(len(x) for x, _ in bags) == 8
    with pytest.raises(ValueError, match='one patch per slide'):
        spatial_patient_bags(bundle, rows, 'cpu', 1)
    result = model(bags)
    expected = torch.stack([1-model.network.cdf(p, model.curve_times[None]) for p in result['params']]).mean(0)
    torch.testing.assert_close(result['survival'], expected)
    torch.testing.assert_close(result['logits'].flatten(), -torch.trapezoid(expected, model.curve_times, dim=1))
    assert (result['survival'].diff() <= 1e-7).all()
    t = torch.tensor([5.])
    patient_pdf = torch.stack([model.network.log_prob(p, t).exp() for p in result['params']]).mean()
    patient_survival = torch.stack([1-model.network.cdf(p, t) for p in result['params']]).mean()
    torch.testing.assert_close(model.objective(result, 5., 1), -(patient_pdf+1e-6).log())
    torch.testing.assert_close(model.objective(result, 5., 0), -(patient_survival+1e-6).log())
    before = model.curve_times.clone()
    bundle.rows.loc[bundle.rows.subset != 'train', 'time'] *= 1000
    model.prepare(bundle, 2, 'cpu'); torch.testing.assert_close(before, model.curve_times, atol=0, rtol=0)
    assert model.prediction_fields(result, rows[0])['survival_time_63'] == float(before[-1])


def test_scmil_time_units_and_degenerate_normalization(tmp_path):
    source('scmil')
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    task = yaml.safe_load((root/'task.yaml').read_text()); task.pop('time_unit')
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='time_unit'):
        check_compatibility(require_runnable('scmil'), Bundle(root))
    with pytest.raises(ValueError, match='time_divisor'):
        SparseContextSurvival(16, time_unit='days')
    model = SparseContextSurvival(16, hidden=16, components=4, time_unit='days', time_divisor=30)
    task['time_unit'] = 'days'; (root/'task.yaml').write_text(yaml.safe_dump(task))
    bundle = Bundle(root); model.prepare(bundle, 1, 'cpu')
    assert float(model.curve_times[-1]) == pytest.approx(bundle.rows[bundle.rows.subset == 'train'].time.max()/30)
    with torch.no_grad():
        model.network.backbone.softfilter.layers[-2].weight.zero_()
        model.network.backbone.softfilter.layers[-2].bias.fill_(4)
    with pytest.raises(ValueError, match='nonfinite normalized'):
        model.slide_parameters(torch.randn(8, 16), torch.zeros(8, 2))
    # A single selected patch skips clustering in the original model and remains valid.
    assert torch.isfinite(model.slide_parameters(torch.randn(1, 16), torch.zeros(1, 2))['w']).all()


@pytest.mark.parametrize('dim', [16, 4096])
def test_deepattn_author_phenotype_pool_output_and_gradient(dim):
    source('deepattnmisl')
    model = DeepAttentionSurvival(dim, clusters=4).eval()
    ref = load_definitions('deepattnmisl', 'DeepAttnMISL_model.py', ['DeepAttnMIL_Surv'],
        {'torch': torch, 'nn': nn}, replacements=[('nn.Conv2d(4096, 64, 1)', f'nn.Conv2d({dim}, 64, 1)')]).DeepAttnMIL_Surv(4).eval()
    ref.load_state_dict(model.network.state_dict())
    x = torch.randn(9, dim); labels = torch.tensor([0, 2, 2, 2, 0, 0, 3, 3, 0])
    groups = [(x[labels == k] if (labels == k).any() else x.new_zeros(1, dim)).T[None, :, None, :] for k in range(4)]
    expected = ref(groups, torch.tensor([[1., 0., 1., 1.]]))
    actual = model(x, labels)['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward(); expected.sum().backward()
    for p, q in zip(model.network.parameters(), ref.parameters()):
        assert p.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_deepattn_original_cox_patient_denominator_ties_and_extreme_values():
    source('deepattnmisl')
    original = load_definitions('deepattnmisl', 'main.py', ['_neg_partial_log'], {'torch': torch, 'np': np},
        replacements=[('train_R.cuda()', 'train_R.to(prediction.device)'),
                      ('torch.FloatTensor(E).cuda()', 'torch.FloatTensor(E).to(prediction.device)')])._neg_partial_log
    risk = torch.tensor([.1, .5, -.2, .8], requires_grad=True)
    times, events = torch.tensor([2., 2., 5., 7.]), torch.tensor([1., 1., 0., 1.])
    expected = original(risk, times.numpy(), events.numpy())
    actual = patient_cox_mean(risk, times, events)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, risk)[0], torch.autograd.grad(expected, risk)[0])
    extreme = risk.detach().add(1000).requires_grad_()
    stable = patient_cox_mean(extreme, times, events); stable.backward()
    assert torch.isfinite(stable) and torch.isfinite(extreme.grad).all()
    assert patient_cox_mean(risk, times, torch.zeros(4)) == 0


def test_deepattn_mask_overflow_is_repaired_without_empty_cluster_contribution():
    source('deepattnmisl')
    model = load_deepattn(16)(3)
    scores = torch.tensor([[0., 1000., 1.]], requires_grad=True)
    weights = model.masked_softmax(scores, torch.tensor([[1., 0., 1.]]))
    torch.testing.assert_close(weights[:, [0, 2]], scores[:, [0, 2]].softmax(1))
    assert weights[0, 1] == 0
    weights.square().sum().backward(); assert torch.isfinite(scores.grad).all()


def test_advmil_author_discriminator_formula_is_not_standard_fake_bce():
    source('advmil')
    raw = load_definitions('advmil', 'loss/utils.py', ['real_fake_loss'],
                          {'torch': torch, 'nn': nn, 'F': F}).real_fake_loss
    fake = torch.tensor([-2., .5, 2.], requires_grad=True)
    actual = raw(None, fake, which='bce')
    literal = (torch.log(fake.sigmoid()+1e-8)-1).mean()
    standard = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    torch.testing.assert_close(actual, literal, atol=0, rtol=0)
    assert not torch.allclose(torch.autograd.grad(actual, fake)[0], torch.autograd.grad(standard, fake)[0])


def test_deepattn_batch_update_includes_cox_l1_and_tail(tmp_path):
    source('deepattnmisl')
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, survival=True))
    model = DeepAttentionSurvival(16, clusters=3, batch_size=4)
    model.prepare(bundle, 3, 'cpu'); ref = copy.deepcopy(model)
    groups = [r for r in units(bundle, 'train') if r[0]['event'] == 1][:5]
    actual_opt = model.make_optimizer(.0001, .0005); expected_opt = ref.make_optimizer(.0001, .0005)
    cfg = {'max_instances': None, 'grad_clip': 5.}
    torch.manual_seed(17)
    deepattn_epoch(model, bundle, groups, actual_opt, 'cpu', cfg)
    torch.manual_seed(17)
    for batch in [groups[:4], groups[4:]]:
        risks = torch.cat([ref.forward_unit(bundle, rows, 'cpu', cfg, True)['logits'] for rows in batch])
        loss = patient_cox_mean(risks, risks.new_tensor([r[0]['time'] for r in batch]), risks.new_ones(len(batch)))
        loss += ref.l1_weight*sum(p.abs().sum() for p in ref.parameters())
        expected_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 5., error_if_nonfinite=True); expected_opt.step()
    for p, q in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


@pytest.mark.parametrize('method,provided', [('scmil', False), ('deepattnmisl', False), ('deepattnmisl', True)])
def test_survival_methods_exact_resume_and_feature_fingerprints(tmp_path, method, provided):
    source(method)
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    if provided:
        table = pd.read_csv(root/'features.csv')
        bundle = Bundle(root)
        for i, row in table.iterrows():
            n = len(bundle.load(row)[0]); path = root/f'clusters_{i}.npy'
            np.save(path, np.arange(n) % 3)
            table.loc[i, 'cluster_labels_path'] = path.name
        table.to_csv(root/'features.csv', index=False)
        task = yaml.safe_load((root/'task.yaml').read_text())
        task['encoders']['toy']['phenotype_clusters'] = {'scope': 'patient', 'count': 3, 'provenance': 'synthetic shared phenotype IDs'}
        (root/'task.yaml').write_text(yaml.safe_dump(task))
    options = ({'hidden': 16, 'components': 5, 'cluster_size': 8} if method == 'scmil' else
               {'clusters': 3, 'batch_size': 4, 'clustering': 'provided' if provided else 'kmeans', 'plateau_patience': 0})
    config = {'epochs': 3, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=config)
    partial = train(root, method, tmp_path/'partial', config=config, stop_after=1)
    train(root, method, partial, config=config, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    assert a['history'] == b['history']
    if '_extra_state' in a['model']:
        extra_a, extra_b = a['model'].pop('_extra_state'), b['model'].pop('_extra_state')
        assert extra_a.pop('scheduler') == extra_b.pop('scheduler')
        torch.testing.assert_close(extra_a, extra_b, atol=0, rtol=0)
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    assert evaluate(full) == evaluate(partial)
    if method == 'scmil':
        result = pd.read_csv(full/'test_predictions.csv')
        assert {'survival_time_0', 'survival_time_63', 'survival_63', 'survival_loss'} <= set(result)
    if provided:
        np.save(root/'clusters_0.npy', np.arange(len(np.load(root/'clusters_0.npy'))) % 2)
        with pytest.raises(ValueError, match='inputs differ'):
            evaluate(full)

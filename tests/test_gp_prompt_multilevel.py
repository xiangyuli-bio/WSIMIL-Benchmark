"""Author-mechanism checks, stochastic inference isolation and stateful recipe recovery."""
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, rng_state, train, units
from milplus.models import build_model
from milplus.recipes import classification_loss, classification_step, make_optimizers
from milplus.registry import require_runnable
from milplus.stochastic import inference_rng
from milplus.upstream import load_definitions, load_single, source_root

NEW = ['sgpmil', 'mmil', 'mmil_coords', 'ptcmil']


def needs_source(method):
    source = 'mmil' if method == 'mmil_coords' else method
    if not source_root(source).is_dir():
        pytest.skip(f'Prepare author source with milplus setup --method {source}')
    if method == 'sgpmil':
        pytest.importorskip('gpytorch')
    if method == 'ptcmil':
        pytest.importorskip('nystrom_attention')


def model_config(method):
    cfg = {'hidden': 32}
    if method == 'sgpmil':
        cfg.update(hidden1=16, hidden2=8, attention_dim=4, inducing_points=6, mc_samples=4)
    if method == 'ptcmil':
        cfg.update(clusters=3)
    return cfg


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('method', NEW)
def test_new_recipe_train_reload_exact_resume(tmp_path, method):
    needs_source(method)
    root = create_demo(tmp_path/'data', patients=20)
    cfg = {'epochs': 3, 'threads': 1, 'model': model_config(method)}
    full = train(root, method, tmp_path/'full', config=cfg)
    part = train(root, method, tmp_path/'part', config=cfg, stop_after=1)
    train(root, method, part, config=cfg, resume=True)
    left, right = [torch.load(run/'last.pt', weights_only=False) for run in (full, part)]
    assert left['history'] == right['history']
    assert_state_equal(left['model'], right['model'])
    assert_state_equal(left['optimizers'], right['optimizers'])
    assert not (full/'test_predictions.csv').exists()
    evaluate(full)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))


@pytest.mark.parametrize('method', ['sgpmil', 'mmil', 'mmil_coords'])
def test_stochastic_inference_preserves_all_training_rng(tmp_path, method):
    needs_source(method)
    torch.set_num_threads(1)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    model = build_model(method, 16, 2, model_config(method))
    if hasattr(model, 'prepare'):
        model.prepare(bundle, 2, 'cpu')
    x, coords = bundle.load(bundle.records('train')[0])
    model(x, coords=coords)  # initialize any lazy variational parameters on training inputs
    model.eval()
    before = rng_state()
    with torch.no_grad():
        a = model(x, coords=coords)
        b = model(x, coords=coords)
    assert_state_equal(before, rng_state())
    torch.testing.assert_close(a['probabilities'], b['probabilities'], rtol=0, atol=0)


def test_inference_rng_restores_state_even_after_exception():
    before = rng_state()
    with pytest.raises(RuntimeError), inference_rng('cpu', 7):
        random.random()
        np.random.rand()
        torch.randn(3)
        raise RuntimeError('expected')
    assert_state_equal(before, rng_state())


@pytest.mark.parametrize('sampling', ['var', 'cov'])
def test_gp_matches_author_logits_probabilities_and_kl(tmp_path, sampling):
    needs_source('sgpmil')
    torch.set_num_threads(1)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, classes=2))
    cfg = {**model_config('sgpmil'), 'sampling': sampling, 'kl_factor': .7}
    model = build_model('sgpmil', 16, 2, cfg)
    model.prepare(bundle, 0, 'cpu')
    x = torch.randn(29, 16)
    target = torch.tensor([1])
    model(x)  # initialize variational state before copying it
    author = load_single('sgpmil', 'src/gp_models.py').SGPMIL(model.network.config)
    author.load_state_dict(model.network.state_dict())
    torch.manual_seed(15)
    actual = model(x)
    torch.manual_seed(15)
    expected = author(x.unsqueeze(0))
    torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)
    torch.testing.assert_close(actual['probabilities'], expected['y_hat'], rtol=0, atol=0)
    kl = author.sgp.variational_strategy.kl_divergence().sum() * .7 / len(units(bundle, 'train'))
    torch.testing.assert_close(classification_loss('sgpmil', actual, target), F.cross_entropy(expected['logits'], target)+kl)
    # Author averaging is probability mean for prediction and logit mean for the CE objective.
    torch.testing.assert_close(actual['probabilities'], expected['pre_mc_integration'].mean(1))


def test_gp_kl_train_bag_count_and_all_variational_parameters_update(tmp_path):
    needs_source('sgpmil')
    torch.set_num_threads(1)
    torch.manual_seed(9)
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    model = build_model('sgpmil', 16, 2, model_config('sgpmil'))
    model.prepare(bundle, 0, 'cpu')
    assert model.training_units == len(units(bundle, 'train'))
    x = torch.randn(25, 16)
    model(x)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if 'variational' in n}
    optimizers = make_optimizers(model, 'sgpmil', .0006, .00001)
    assert isinstance(optimizers[0], torch.optim.AdamW)
    classification_step(model, 'sgpmil', x, None, torch.tensor([1]), optimizers)
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
            assert not torch.equal(before[name], parameter.detach())
    strategy = model.network.sgp.variational_strategy
    q, p = strategy.variational_distribution, strategy.prior_distribution
    torch.testing.assert_close(strategy.kl_divergence(), torch.distributions.kl_divergence(q, p))


def test_gp_variance_sampling_preserves_author_shared_noise():
    needs_source('sgpmil')
    model = build_model('sgpmil', 8, 2, model_config('sgpmil'))
    distribution = SimpleNamespace(mean=torch.tensor([[1., 2., 3.]]), variance=torch.tensor([[1., 4., 9.]]))
    torch.manual_seed(4)
    samples = model.network.mc_sampling(distribution).squeeze(-1)
    standardized = (samples-distribution.mean)/distribution.variance.sqrt()
    torch.testing.assert_close(standardized[:, 0], standardized[:, 1])
    torch.testing.assert_close(standardized[:, 1], standardized[:, 2])


@pytest.mark.parametrize('grouping', ['random', 'seq', 'embed'])
def test_mmil_matches_author_on_nonsingleton_groups(grouping):
    needs_source('mmil')
    torch.set_num_threads(1)
    scope = {'torch': torch, 'nn': nn, 'F': F, 'random': random, 'np': np, 'rearrange': rearrange, 'KMeans': KMeans}
    helpers = load_definitions('mmil', 'utils/utils.py', ['cat_msg2cluster_group', 'split_array'], scope)
    scope.update({key: getattr(helpers, key) for key in ['cat_msg2cluster_group', 'split_array']})
    scope['grouping'] = load_definitions('mmil', 'utils/core.py', ['grouping'], scope).grouping
    author_module = load_definitions('mmil', 'models/models.py',
        ['Attention', 'AttenLayer', 'NyAttenLayer', 'GroupsAttenLayer', 'GroupsMSGAttenLayer', 'BasicLayer', 'MultipleMILTransformer'], scope)
    adapter = build_model('mmil', 16, 3, {'hidden': 32, 'groups': 4, 'grouping': grouping})
    author = author_module.MultipleMILTransformer(adapter.network.args)
    author.load_state_dict(adapter.network.state_dict())
    x = torch.randn(80, 16)
    with inference_rng('cpu', 3):
        actual = adapter(x)['logits']
    with inference_rng('cpu', 3):
        expected = author(x.unsqueeze(0), mask_ratio=.6)['logits']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mmil_singletons_masking_and_message_gradients():
    needs_source('mmil')
    model = build_model('mmil', 16, 2, {'hidden': 32, 'groups': 4, 'mask_ratio': .5})
    x = torch.randn(4, 16)
    F.cross_entropy(model(x)['logits'], torch.tensor([1])).backward()
    assert model.network.msg_tokens.grad.abs().sum() > 0
    assert model.network.msgcls_token.grad.abs().sum() > 0
    for layer in model.network.layers:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.GroupsMSGAttenLayer.parameters())
    with pytest.raises(ValueError, match='instances'):
        model(x[:2])
    with pytest.raises(ValueError, match='retain'):
        build_model('mmil', 16, 2, {'groups': 4, 'mask_ratio': .99})
    assert 'coords' in require_runnable('mmil_coords')['required_inputs']
    assert require_runnable('MMIL-Transformer')['id'] == 'mmil'


def test_mmil_collapsed_clusters_keep_indices_and_reject_empty_mask():
    needs_source('mmil')
    from sklearn.exceptions import ConvergenceWarning
    model = build_model('mmil_coords', 16, 2, {'hidden': 32, 'groups': 4, 'mask_ratio': .6})
    indices = model.network.grouping.indicer(np.array([0, 2, 2, 4]))
    assert [index.tolist() for index in indices] == [[0], [1, 2], [3]]
    with pytest.warns(ConvergenceWarning), pytest.raises(ValueError, match='retain'):
        model(torch.randn(8, 16), coords=torch.ones(8, 2))


def test_ptcmil_moving_prompts_loss_gradients_and_validation_isolation():
    needs_source('ptcmil')
    torch.set_num_threads(1)
    model = build_model('ptcmil', 16, 3, model_config('ptcmil'))
    optimizer = make_optimizers(model, 'ptcmil', .0002, .00001)
    x, y = torch.randn(24, 16), torch.tensor([1])
    model.start_epoch(0)
    assert not model.context_ready
    out = model(x, label=y)
    torch.testing.assert_close(classification_loss('ptcmil', out, y), F.cross_entropy(out['logits'], y)+out['regularization_loss'])
    assert out['regularization_loss'] > 0
    classification_step(model, 'ptcmil', x, None, y, optimizer)
    assert model.context_ready and model.pending_context is None
    assert model.network.pmt_token.grad.abs().sum() > 0
    assert model.network.ctm.score.weight.grad.abs().sum() > 0
    context = model.prompt_context.clone()
    classification_step(model, 'ptcmil', x*2, None, y, optimizer)
    assert not torch.equal(context, model.prompt_context)
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        eval_logits = model(x)['logits']
        direct = model.network(x, result_dict=None)[0]
    torch.testing.assert_close(eval_logits, direct, rtol=0, atol=0)
    assert_state_equal(state, model.state_dict())
    model.start_epoch(1)
    assert not model.context_ready and model.prompt_context.count_nonzero() == 0


def test_ptcmil_merge_matches_manual_weighted_clusters():
    needs_source('ptcmil')
    module = load_definitions('ptcmil', 'classification/models/tcformer/tcformer_utils.py', ['merge_tokens'], {'torch': torch})
    x = torch.tensor([[1., 2.], [5., 7.], [9., 11.]])
    idx, weight = torch.tensor([0, 0, 2]), torch.tensor([[1.], [3.], [2.]])
    result = module.merge_tokens(x, idx, 3, weight)[0]
    expected = torch.stack([(x[0]+3*x[1])/(4+1e-6), torch.zeros(2), 2*x[2]/(2+1e-6)])
    torch.testing.assert_close(result, expected)


def test_single_author_loader_distinguishes_two_files():
    needs_source('sgpmil')
    model = load_single('sgpmil', 'src/gp_models.py')
    kernel = load_single('sgpmil', 'src/kernels.py')
    assert hasattr(model, 'SGPMIL') and hasattr(kernel, 'CustomRBFKernel')
    assert model is not kernel

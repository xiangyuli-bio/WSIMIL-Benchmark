"""PIBD released trainer and full-network comparisons, train-only fitting and recovery."""
import ast
import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.pibd import PIBDSurvival, patient_sampling_weights, pibd_epoch, sampled_padded_bag
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import load_definitions, pinned_text, verified_source


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, normalization='none', **options):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['omics']['normalization'] = normalization
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    bundle = Bundle(root)
    model = PIBDSurvival(bundle.dim, bundle.omics, **{'hidden': 16, 'num_patches': 32,
        'micro_bag_size': 16, 'sample_num': 3, 'batch_size': 5, **options})
    model.prepare(bundle, 19, 'cpu')
    return model, bundle


def independent_author(model):
    """Read Git definitions independently; only remove CPU-incompatible cuda calls."""
    root, sha = verified_source('pibd')
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'BAG_SIZE': model.micro_bag_size}
    for path in ['utils/loss_func.py', 'models/omics_encoder.py', 'models/club.py',
                 'models/disentangle_transformer.py', 'models/model_PIBD.py']:
        text = pinned_text(root, sha, path).replace('.cuda()', '')
        nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)  # noqa: S102 -- locked Git blobs, independent CPU reference
    net = scope['PIBD'](copy.deepcopy(model.network.args))
    net.load_state_dict(model.network.state_dict())
    return net, scope['NLLSurvLoss'](alpha=.5)


class RecordingAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=5e-4, weight_decay=0.)
        self.records = []

    def step(self, closure=None):
        parameters = self.param_groups[0]['params']
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        super().step(closure)
        self.records.append((gradients, [p.detach().clone() for p in parameters]))


def source_trainer():
    def forward(net, format_name, device, batch):
        x, omics, y, e = batch
        args = {f'x_omic{i+1}': omics[:, index] for i, index in enumerate(net.test_indices)}
        out = net(x_wsi=x, y=y, c=1-e, return_attn=False, **args)
        return out, y, y.float()+1, 1-e, []
    return load_definitions('pibd', 'utils/core_utils.py',
        ['_train_loop_survival', '_calculate_risk', '_update_arrays'],
        {'torch': torch, 'np': np, '_process_data_and_forward': forward,
         'concordance_index_censored': lambda *args, **kwargs: (0.5,)})._train_loop_survival


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('weights', [(0.1, .01, 1., .1), (.2, .02, .7, .3), (0., 0., 0., 1.)])
def test_released_full_trainer_three_adam_updates(tmp_path, batch_size, weights):
    model, bundle = fixture(tmp_path, proxy_nll_weight=weights[0], proxy_kl_weight=weights[1],
                            proxy_contrast_weight=weights[2], disentangle_weight=weights[3])
    reference, nll = independent_author(model)
    reference.test_indices = model.group_indices
    torch.manual_seed(232)
    batches = [(torch.randn(batch_size, 32, bundle.dim), torch.randn(batch_size, 18),
                (torch.arange(batch_size)+i) % 4, (torch.arange(batch_size)+i) % 2) for i in range(3)]
    original_optimizer = RecordingAdam(reference.parameters())
    actual_optimizer = RecordingAdam(model.network.parameters())
    class Loader(list):
        dataset = tuple(range(batch_size*3))
    torch.manual_seed(717)
    source_trainer()(SimpleNamespace(gamma=weights[2], sigma=weights[3]), 0, reference, 'pathways',
                     Loader(batches), original_optimizer, nll, io.StringIO())
    torch.manual_seed(717)
    for x, molecular, y, e in batches:
        actual_optimizer.zero_grad(set_to_none=True)
        outputs = model.network_output(x, molecular, y, e)
        loss, parts = model.batch_objective(outputs, y, e)
        torch.testing.assert_close(loss, sum(v for k, v in parts.items() if k != 'loss'))
        loss.backward()
        actual_optimizer.step()
    for (grads, params), (ref_grads, ref_params) in zip(actual_optimizer.records, original_optimizer.records):
        for a, b in zip(grads, ref_grads):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(params, ref_params):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_all_core_branches_have_nonzero_gradients_and_patient_mi(tmp_path):
    model, bundle = fixture(tmp_path)
    x, omics = torch.randn(4, 32, bundle.dim, requires_grad=True), torch.randn(4, 18, requires_grad=True)
    output = model.network_output(x, omics, torch.arange(4), torch.tensor([1, 0, 1, 0]))
    assert abs(float(output[3].detach())) > 1e-7
    loss, _ = model.batch_objective(output, torch.arange(4), torch.tensor([1, 0, 1, 0]))
    loss.backward()
    for prefix in ['sig_networks', 'PIB_wsi.encoder', 'PIB_wsi.decoder_logits', 'PIB_wsi.proxies',
                   'PIB_omics.encoder', 'PIB_omics.proxies', 'PoE.phi', 'PID', 'CLUB.mimin', 'CLUB.mimin_glob', 'to_logits']:
        values = [p.grad for name, p in model.network.named_parameters() if name.startswith(prefix)]
        assert values and all(v is not None and torch.isfinite(v).all() for v in values)
        assert sum(v.abs().sum() for v in values) > 0, prefix
    assert x.grad.abs().sum() > 0 and omics.grad.abs().sum() > 0


def test_source_eval_matches_and_isolates_rng_and_outcomes(tmp_path):
    model, bundle = fixture(tmp_path)
    reference, _ = independent_author(model)
    reference.eval(); model.eval()
    x, molecular = torch.randn(1, 23, bundle.dim), torch.randn(1, 18)
    for module in (reference.PIB_wsi, reference.PIB_omics, reference.PoE):
        module.seed = 19
    before = torch.get_rng_state()
    actual = model.network_output(x, molecular)[0]
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(41)
    torch.testing.assert_close(actual, model.network_output(x, molecular)[0], rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        kwargs = {f'x_omic{i+1}': molecular[:, index] for i, index in enumerate(model.group_indices)}
        expected = reference(x_wsi=x, y=None, c=None, return_attn=False, **kwargs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='outcome labels'):
        model.network_output(x, molecular, torch.tensor([1]), torch.tensor([1]))


def test_source_poe_literal_scale_not_unannounced_covariance_repair(tmp_path):
    model, _ = fixture(tmp_path)
    poe = model.network.PoE.eval()
    mu, sigma = [torch.randn(3, 16) for _ in range(2)], [torch.rand(3, 16)+.2 for _ in range(2)]
    with torch.random.fork_rng(devices=[]):
        actual = poe(mu, sigma)
        torch.manual_seed(poe.seed)
        noise = torch.normal(torch.zeros(3, 3, 16), torch.ones(3, 3, 16))
        precision = sum(.5*(1/(s+1e-8)) for s in sigma)
        mean = sum(m*.5*(1/(s+1e-8)) for m, s in zip(mu, sigma))/precision
        expected = mean[:, None] + (1/precision)[:, None]*noise
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_flattened_train_minmax_zero_restore_and_heldout_isolation(tmp_path):
    model, bundle = fixture(tmp_path, normalization='train_global_minmax_preserve_zero')
    ids = sorted({r['global_patient_id'] for r in bundle.records('train')})
    values = bundle.omics.values.loc[ids].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(values.reshape(-1, 1))
    samples = torch.tensor([[0., -90., 90.]*6])
    expected = scaler.transform(samples.numpy().reshape(-1, 1)).reshape(1, -1)
    expected[samples.numpy() == 0] = 0
    torch.testing.assert_close(model.normalize_omics(samples), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    old = copy.deepcopy(model.state_dict())
    heldout = sorted(set(bundle.rows.global_patient_id)-set(ids))
    bundle.omics.values.loc[heldout] = 1e9
    bundle.rows.loc[bundle.rows.subset != 'train', 'time'] = 1e10
    model.prepare(bundle, 19, 'cpu')
    for key in ['omics_multiplier', 'omics_offset', 'bin_edges']:
        torch.testing.assert_close(model.state_dict()[key], old[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='only by PIBD'):
        check_compatibility(require_runnable('mcat'), bundle)


def test_constant_global_minmax_matches_sklearn(tmp_path):
    model, bundle = fixture(tmp_path, normalization='train_global_minmax_preserve_zero')
    bundle.omics.values.loc[:, :] = 5.
    model.prepare(bundle, 19, 'cpu')
    values = torch.full((1, 18), 5.)
    torch.testing.assert_close(model.normalize_omics(values), torch.full_like(values, -1.))


def test_sampling_padding_and_occupied_class_weights():
    x = torch.arange(60.).reshape(20, 3)
    np.random.seed(41)
    expected = x[np.sort(np.random.choice(20, 12, replace=False))]
    np.random.seed(41)
    torch.testing.assert_close(sampled_padded_bag(x, 12), expected)
    short = sampled_padded_bag(x[:3], 9)
    torch.testing.assert_close(short[:3], x[:3])
    assert short[3:].eq(0).all()
    weights = patient_sampling_weights(torch.tensor([0, 0, 2, 2]), torch.tensor([1, 1, 0, 1]))
    torch.testing.assert_close(weights, torch.tensor([2., 2., 4., 4.], dtype=torch.double))


def test_epoch_consumes_patient_batches_and_tail_and_complete_sampling(tmp_path):
    model, bundle = fixture(tmp_path, weighted_sampling=False)
    batches = []
    hook = model.network.register_forward_pre_hook(lambda net, args, kw: batches.append(kw['x_wsi'].detach().clone()), with_kwargs=True)
    records = pibd_epoch(model, bundle, units(bundle, 'train'), torch.optim.Adam(model.parameters(), lr=5e-4),
                         'cpu', {'grad_clip': None})
    hook.remove()
    assert [r['batch_patients'] for r in records] == [5, 5, 2]
    assert [len(b) for b in batches] == [5, 5, 2]
    assert all(b.shape[1:] == (32, bundle.dim) for b in batches)


@pytest.mark.parametrize('options,match', [({'bins': 3}, 'four-bin'), ({'hidden': 15}, 'divisible'),
    ({'num_patches': 31}, 'multiple'), ({'ratio_omics': .01}, 'zero instances'),
    ({'proxy_kl_weight': float('nan')}, 'nonnegative'), ({'weighted_sampling': 1}, 'boolean')])
def test_invalid_model_contracts(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        fixture(tmp_path, **options)


def test_missing_inputs_and_generic_sampler_rejected(tmp_path):
    model, bundle = fixture(tmp_path)
    with pytest.raises(ValueError, match='two-stage'):
        model.configure_training({'max_instances': 100, 'grad_clip': None})
    with pytest.raises(ValueError, match='observed training targets'):
        model.network_output(torch.randn(1, 32, bundle.dim), torch.randn(1, 18))
    bundle.omics = None
    with pytest.raises(ValueError, match='omics'):
        check_compatibility(require_runnable('pibd'), bundle)


@pytest.mark.parametrize('weighted', [True, False])
def test_exact_resume_reload_defaults_and_no_test_selection(tmp_path, weighted):
    _, bundle = fixture(tmp_path, normalization='train_global_minmax_preserve_zero')
    config = {'epochs': 3, 'threads': 1, 'model': {'hidden': 16, 'batch_size': 5,
        'num_patches': 32, 'micro_bag_size': 16, 'sample_num': 3, 'weighted_sampling': weighted}}
    full = train(bundle.root, 'PIBD', tmp_path/'full', seed=772, config=config)
    partial = train(bundle.root, 'PIBD', tmp_path/'resume', seed=772, config=config, stop_after=1)
    assert not (partial/'test_predictions.csv').exists()
    train(bundle.root, 'PIBD', partial, seed=772, config=config, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['history'] == b['history']
    assert evaluate(full) == evaluate(partial)
    first = (full/'test_predictions.csv').read_bytes()
    evaluate(full)
    assert first == (full/'test_predictions.csv').read_bytes()
    import json
    req = json.loads((full/'request.json').read_text())
    assert req['config']['lr'] == 5e-4 and req['config']['weight_decay'] == 0
    assert req['config']['grad_clip'] is None and req['config']['patience'] == 3

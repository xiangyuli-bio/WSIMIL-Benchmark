"""Original complete ProtoSurv graphs/recipes, clinical identities and recovery."""
import ast
import contextlib
import copy
import io
import os
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json
from milplus.protosurv import COMMIT, ProtoSurv, source_c_index
from milplus.protosurv_data import ProtoSurvAssets
from milplus.protosurv_demo import create_protosurv_demo
from milplus.upstream import load_definitions, pinned_text, verified_source

FAST = {'batch_size': 3, 'hidden': 8, 'prototype_dimension': 24, 'prototype_hidden': 16,
        'prototypes_per_class': 2, 'prototype_iterations': 1, 'fusion_layers': 1}


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def data(path, **kwargs):
    return Bundle(create_protosurv_demo(path, patients=20, **kwargs))


def independent_module(file, namespace):
    root, sha = verified_source('protosurv')
    assert sha == COMMIT
    tree = ast.parse(pinned_text(root, sha, file))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
                                                  node.module and node.module.startswith('models.'))]
    scope = dict(namespace)
    exec(compile(tree, 'independent_original_'+file, 'exec'), scope)  # noqa: S102 -- pinned original Git objects
    return SimpleNamespace(**scope)


def independent_network(dimension, options=None):
    utils = independent_module('models/utils/model_utils.py', {})
    proto = independent_module('models/utils/proto_utils_v1.py', vars(utils))
    factory = independent_module('models/model_protosurv_v1.py',
        {**vars(utils), 'ProtoGNN': proto.ProtoGNN, 'ProtoFusion': proto.ProtoFusion})
    options = options or {}
    kwargs = {'input_dim': dimension, 'nr_types': 5,
        'hidden_dim': options.get('hidden', 128), 'dim_proto': options.get('prototype_dimension', 768),
        'proto_fc_hidden_dim': options.get('prototype_hidden', 512),
        'num_proto': options.get('prototypes_per_class', 8),
        'proto_iters': options.get('prototype_iterations', 2),
        'fusion_layer_to_p': options.get('fusion_layers', 4)}
    with contextlib.redirect_stdout(io.StringIO()):
        model = factory.LINKX_PROTO_oldv(**kwargs)
    losses = independent_module('utils/utils_loss.py', {})
    return model, losses


def native_ci_reference():
    functions = ['naive_concordance_index', '_preprocess_scoring_data',
                 '_naive_concordance_summary_statistics', '_concordance_ratio']
    return load_definitions('lifelines_metric', 'lifelines/utils/concordance.py', functions,
                             {'np': np}).naive_concordance_index


@pytest.mark.parametrize('dimension', [1024, 1536])
def test_native_full_original_outputs_three_losses_and_all_gradients(tmp_path, dimension):
    bundle = data(tmp_path/'data', dimension=dimension)
    seed_everything(66)
    actual = ProtoSurv(bundle, 'protosurv_source')
    seed_everything(66)
    reference, losses = independent_network(dimension)
    exact(actual.network.state_dict(), reference.state_dict())
    actual.eval();reference.eval()
    a = [bundle.protosurv.load(row) for row in bundle.records('train')[:2]]
    b = [graph.clone() for graph in a]
    for graph in [*a, *b]:
        graph.x = graph.x.detach().clone().requires_grad_()
    before = rng_state()
    outputs = actual.forward_graphs(a, 'cpu')
    objective = actual.objective(a, outputs, 'cpu')
    after = rng_state();restore_rng(before)
    expected = reference(Batch.from_data_list(b))
    with contextlib.redirect_stdout(io.StringIO()):
        cox = losses.CoxPHLoss(alpha=0.)
    times = torch.tensor([graph.event_time for graph in b])
    censorship = torch.tensor([graph.c for graph in b])
    ref_loss = cox(S=expected[1], c=censorship, event_time=times)
    ref_loss += .01*losses.get_compatibility_loss_batch(b, expected[4], 5)
    ref_loss += .1*losses.get_orthogonal_regularization_loss(expected[4])
    exact(outputs[1], expected[1]);exact(outputs[4], expected[4]);exact(objective, ref_loss)
    exact(after, rng_state())
    objective.backward();ref_loss.backward()
    for x, y in zip(a, b):
        exact(x.x.grad, y.x.grad)
    for (name, p), (ref_name, q) in zip(actual.network.named_parameters(), reference.named_parameters()):
        assert name == ref_name and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            exact(p.grad, q.grad)
    assert actual.network.ProtoGNN.proto_attention.rand_mul.grad.abs().max() > 0
    assert actual.network.proto_fusion_to_p.layers[0].to_q.weight.grad.abs().max() > 0


@pytest.mark.parametrize('custom_sample', [False, True])
def test_actual_complete_original_train_validate_adam_scheduler_rng(tmp_path, custom_sample):
    bundle = data(tmp_path/'data')
    seed_everything(13)
    actual = ProtoSurv(bundle, 'protosurv_source', custom_sample=custom_sample, **FAST)
    config = {'epochs': 3, 'patience': 15, 'max_instances': None, 'grad_clip': None,
              'selection_metric': 'protosurv_val_c_index'}
    actual.configure_training(config)
    optimizer = actual.make_optimizer(5e-5, 1e-5)
    seed_everything(13)
    ref_net, losses = independent_network(bundle.dim, FAST)
    exact(actual.network.state_dict(), ref_net.state_dict())

    class OriginalSingleDeviceDataParallel(nn.Module):
        def __init__(self, network):
            super().__init__();self.network = network

        def forward(self, graphs):
            return self.network(Batch.from_data_list(graphs))

    reference = OriginalSingleDeviceDataParallel(ref_net)
    ref_optimizer = torch.optim.Adam(reference.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ref_optimizer, T_max=5, eta_min=1e-6)
    original_aug = load_definitions('protosurv', 'datasets/wsi_dataset.py', ['data_transform'],
                                    {'torch': torch, 'np': np, 'random': random}).data_transform
    original_sampler = load_definitions('protosurv', 'utils/utils.py', ['Sampler_custom'],
        {'torch': torch, 'np': np, 'random': random, 'copy': copy, 'Sampler': Sampler}).Sampler_custom

    class OriginalRows(Dataset):
        def __init__(self, subset):
            self.subset, self.rows = subset, bundle.records(subset)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            graph = bundle.protosurv.load(self.rows[i])
            # An explicit valid synthetic latent graph with the original shape.
            # The network never reads it; source augmentation does consume RNG.
            graph.edge_latent = graph.edge_index.T.clone()
            return original_aug(graph) if self.subset == 'train' else graph

    reference_ci = native_ci_reference()
    scope = {'torch': torch, 'np': np, 'device': torch.device('cpu'),
        'GradScaler': torch.cuda.amp.GradScaler, 'autocast': torch.cuda.amp.autocast,
        'concordance_index': reference_ci, 'os': os,
        'get_compatibility_loss_batch': losses.get_compatibility_loss_batch,
        'get_orthogonal_regularization_loss': losses.get_orthogonal_regularization_loss}
    original = load_definitions('protosurv', 'utils/core_funcs.py',
        ['train_loop_survival', 'validate_survival', 'EarlyStopping_cindex'], scope,
        replacements=[('np.Inf', 'np.inf')])
    callback = original.EarlyStopping_cindex(warmup=0, patience=15, stop_epoch=20)
    with contextlib.redirect_stdout(io.StringIO()):
        criterion = losses.CoxPHLoss(alpha=0.)
    args = SimpleNamespace(fp16_precision=False, model_type='protosurv_v1', comp_alpha=.01,
                           ortho_beta=.1, nr_types=5)
    for epoch in range(2):
        before = rng_state()
        actual.epoch(bundle, optimizer);actual.finish_epoch(bundle, 'cpu', config)
        predictions = actual.predict_subset(bundle, 'val', 'cpu', config)
        actual.after_validation(actual.validation_metrics(predictions))
        after = rng_state();restore_rng(before)
        if custom_sample:
            with contextlib.redirect_stdout(io.StringIO()):
                sampler = original_sampler([1-r['event'] for r in bundle.records('train')], 3)
            loader = DataLoader(OriginalRows('train'), batch_sampler=sampler, num_workers=0, collate_fn=list)
        else:
            loader = DataLoader(OriginalRows('train'), batch_size=3, shuffle=True,
                                drop_last=True, num_workers=0, collate_fn=list)
        with contextlib.redirect_stdout(io.StringIO()):
            original.train_loop_survival(epoch+1, reference, loader, ref_optimizer, 1, args,
                loss_fn=criterion, gc=2, bar=False)
        collected = []
        hook = reference.register_forward_hook(lambda _module, _inputs, out, collected=collected: collected.append(out[1].detach().clone()))
        with contextlib.redirect_stdout(io.StringIO()):
            original.validate_survival(epoch+1, reference,
                DataLoader(OriginalRows('val'), batch_size=1, shuffle=False, num_workers=0, collate_fn=list),
                1, early_stopping=callback, loss_fn=criterion, results_dir=str(tmp_path), cur=0, bar=False)
        hook.remove();scheduler.step()
        np.testing.assert_array_equal(predictions.risk.to_numpy(), torch.cat(collected).flatten().numpy().astype('float64'))
        exact(actual.network.state_dict(), ref_net.state_dict())
        exact(optimizer.state_dict(), ref_optimizer.state_dict());exact(actual.scheduler.state_dict(), scheduler.state_dict())
        exact(after, rng_state())
        assert actual.early_best == callback.best_cindex_num and actual.early_wait == callback.counter


def test_source_c_index_matches_pinned_lifelines_including_all_ties():
    reference = native_ci_reference()
    rng = np.random.default_rng(39)
    cases = [([1, 1], [1, 0], [2., 1.]), ([1, 1], [1, 0], [1., 2.]),
             ([1, 1], [1, 0], [1., 1.]), ([1, 1], [1, 1], [2., 1.])]
    cases += [(rng.integers(1, 9, 20), rng.integers(0, 2, 20), rng.integers(0, 5, 20)) for _ in range(20)]
    for time, event, risk in cases:
        try:
            expected = 1-reference(np.array(time), np.array(risk), np.array(event))
        except ZeroDivisionError:
            expected = None
        assert source_c_index(time, event, risk) == expected


def test_source_global_pad_reseed_and_named_local_generator(tmp_path):
    bundle = data(tmp_path/'data')
    source = ProtoSurv(bundle, 'protosurv_source', **FAST).eval()
    repaired = ProtoSurv(bundle, 'protosurv_repaired', **FAST).eval()
    repaired.network.load_state_dict(source.network.state_dict())
    graphs = [bundle.protosurv.load(bundle.records('val')[0])]
    before = rng_state();a = source.forward_graphs(graphs, 'cpu')[1]
    after = rng_state();assert not torch.equal(before['torch'], after['torch'])
    restore_rng(before);b = repaired.forward_graphs(graphs, 'cpu')[1]
    assert torch.equal(before['torch'], rng_state()['torch'])
    exact(a, b)


@pytest.mark.parametrize('profile', ['protosurv_source', 'protosurv_repaired'])
def test_complete_training_selected_checkpoint_resume_and_predictions(tmp_path, profile):
    bundle = data(tmp_path/'data')
    config = {'epochs': 3, 'threads': 1, 'model': FAST}
    full, resumed = tmp_path/'full', tmp_path/'resume'
    train(bundle.root, profile, full, config=config, seed=19)
    train(bundle.root, profile, resumed, config=config, seed=19, stop_after=1)
    train(bundle.root, profile, resumed, config=config, seed=19, resume=True)
    for name in ['model', 'optimizers', 'history', 'rng', 'best', 'bad']:
        a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, resumed]]
        exact(a[name], b[name])
    assert not (full/'test_predictions.csv').exists()
    assert evaluate(full) == evaluate(resumed)
    sha = digest_file(full/'test_predictions.csv');evaluate(full)
    assert sha == digest_file(full/'test_predictions.csv')
    protocol = read_json(full/'method_protocol.json')
    assert protocol['completed_epochs'] == 3 and protocol['optimizer_steps'] > 0
    assert protocol['settings']['network']['dim_proto'] == 24


def test_missing_highest_tissue_type_original_vs_repaired_objective(tmp_path):
    bundle = data(tmp_path/'data')
    source = ProtoSurv(bundle, 'protosurv_source', **FAST)
    repaired = ProtoSurv(bundle, 'protosurv_repaired', **FAST)
    graphs = [bundle.protosurv.load(r) for r in bundle.records('train')[:2]]
    for graph in graphs:
        graph.patch_classify_type[graph.patch_classify_type == 4] = 0
    result = source.forward_graphs(graphs, 'cpu')
    with pytest.raises(ValueError, match='highest tissue category'):
        source.objective(graphs, result, 'cpu')
    # Native evaluation never calls compatibility and must still work.
    source.eval();assert torch.isfinite(source.forward_graphs(graphs, 'cpu')[1]).all()
    value = repaired.objective(graphs, repaired.forward_graphs(graphs, 'cpu'), 'cpu')
    value.backward();assert torch.isfinite(value)


def test_unknown_fixed_prototypes_require_explicit_compatibility_condition(tmp_path):
    bundle = data(tmp_path/'data', fixed_prototypes=True)
    with pytest.raises(ValueError, match='comp_alpha=0'):
        ProtoSurv(bundle, pre_proto=True, **FAST)
    model = ProtoSurv(bundle, pre_proto=True, comp_alpha=0., **FAST)
    graphs = [bundle.protosurv.load(r) for r in bundle.records('train')[:2]]
    assert all((g.patch_classify_type == -1).all() for g in graphs)
    value = model.objective(graphs, model.forward_graphs(graphs, 'cpu'), 'cpu')
    value.backward();assert torch.isfinite(value)
    assert 'ProtoGNN.pre_prototypes' in model.network.state_dict()


@pytest.mark.parametrize('kind', ['graph', 'primary', 'coordinates', 'prototype'])
def test_input_mutations_cannot_reuse_prepared_assets(tmp_path, kind):
    bundle = data(tmp_path/'data', fixed_prototypes=kind == 'prototype')
    row = bundle.records('train')[0]
    assets = bundle.protosurv
    if kind == 'prototype':
        p = assets.path.parent/assets.meta['prototype']['path']
        x = np.load(p);x[0, 0] += 1;np.save(p, x)
        with pytest.raises(ValueError, match='prototype contents changed'):
            assets.prototype()
        return
    path = assets.path.parent/assets.meta['slides'][row['slide_id']]['path'] if kind == 'graph' else bundle.root/row['path']
    with h5py.File(path, 'r+') as f:
        key = 'coords' if kind == 'coordinates' else 'features' if kind == 'primary' else 'node_types'
        f[key][0] = f[key][0]+1
    with pytest.raises(ValueError, match='changed'):
        assets.load(row)


def test_prototype_population_rechecked_when_loading_manifest(tmp_path):
    bundle = data(tmp_path/'data', fixed_prototypes=True)
    path = bundle.protosurv.path
    meta = read_json(path);meta['prototype']['fit_subset'] = 'test'
    path.write_text(__import__('json').dumps(meta))
    with pytest.raises(ValueError, match='never VAL/TEST'):
        ProtoSurvAssets(bundle)


def test_same_dimension_wrong_FM_prototypes_refused(tmp_path):
    bundle = data(tmp_path/'data', fixed_prototypes=True)
    path = bundle.protosurv.path
    meta = read_json(path);meta['prototype']['feature_space']['weights'] = 'another_FM_checkpoint_same_width'
    path.write_text(__import__('json').dumps(meta))
    with pytest.raises(ValueError, match='not merely its dimension'):
        ProtoSurvAssets(bundle)


def test_repaired_Cox_tied_risk_sets_match_independent_formula(tmp_path):
    bundle = data(tmp_path/'data')
    model = ProtoSurv(bundle, comp_alpha=0., ortho_beta=0., **FAST)
    risk = torch.tensor([[.1], [.3], [.2]], requires_grad=True)
    other = risk.detach().clone().requires_grad_()
    graphs = [bundle.protosurv.load(r) for r in bundle.records('train')[:3]]
    for graph, time, event in zip(graphs, [4., 4., 7.], [1., 0., 1.]):
        graph.event_time, graph.c = time, 1-event
    result = (None, risk, None, None, torch.empty(3, 10, bundle.dim))
    actual = model.objective(graphs, result, 'cpu')
    expected = (torch.logsumexp(other.flatten(), 0)-other[0, 0])/2
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.backward();expected.backward()
    torch.testing.assert_close(risk.grad, other.grad, rtol=1e-6, atol=1e-7)


def test_censored_skips_do_not_invent_updates_and_undefined_ci_fails(tmp_path):
    bundle = data(tmp_path/'data')
    model = ProtoSurv(bundle, **FAST)
    with pytest.raises(ValueError, match='undefined'):
        model.after_validation({'protosurv_val_c_index': None})
    with pytest.raises(ValueError, match='No valid|produced no'):
        fake = copy.copy(bundle)
        fake.rows = bundle.rows.copy();fake.rows.loc[fake.rows.subset == 'train', 'event'] = 0
        model.epoch(fake, model.make_optimizer(5e-5, 1e-5))


def test_patient_representative_not_silently_selected(tmp_path):
    bundle = data(tmp_path/'data')
    fake = copy.copy(bundle);fake.rows = pd.concat([bundle.rows, bundle.rows.iloc[:1]])
    with pytest.raises(ValueError, match='one declared WSI'):
        ProtoSurv(fake, **FAST)


@pytest.mark.parametrize('field,value', [('early_wait', -1), ('completed_epochs', True),
    ('early_best', float('nan')), ('training_done', 1), ('contract', 'wrong')])
def test_invalid_checkpoint_extra_state_refused(tmp_path, field, value):
    model = ProtoSurv(data(tmp_path/'data'), **FAST)
    before = copy.deepcopy(model.get_extra_state())
    state = copy.deepcopy(before);state[field] = value
    with pytest.raises(ValueError):
        model.set_extra_state(state)
    exact(before, model.get_extra_state())


@pytest.mark.parametrize('profile', ['protosurv_source', 'protosurv_repaired'])
def test_tail_update_policy_against_independent_analytic_Adam(tmp_path, profile):
    """Isolate accumulation policy; complete real networks are checked above."""
    root = create_protosurv_demo(tmp_path/'data', patients=30)
    slides, split = pd.read_csv(root/'slides.csv'), pd.read_csv(root/'split.csv')
    slides.loc[slides.patient_id.isin(split.loc[split.subset == 'train', 'patient_id']), 'event'] = 1
    slides.to_csv(root/'slides.csv', index=False)
    model = ProtoSurv(Bundle(root), profile, custom_sample=False, **{**FAST, 'batch_size': 6})
    parameter = nn.Parameter(torch.tensor(.1))
    model.network.register_parameter('_analytic_tail_probe', parameter)
    targets, calls = [1., 2., -3.], []
    model.forward_graphs = lambda graphs, device: None

    def objective(graphs, result, device):
        target = targets[len(calls)];calls.append(target)
        return (parameter-target).square()

    model.objective = objective
    optimizer = model.make_optimizer(.01, 0.)
    model.epoch(model.bundle, optimizer)
    assert calls == targets and model.optimizer_steps == 2
    expected = nn.Parameter(torch.tensor(.1))
    reference = torch.optim.Adam([expected], lr=.01)
    ((expected-1).square()/2+(expected-2).square()/2).backward()
    reference.step();reference.zero_grad()
    scale = .5 if profile == 'protosurv_source' else 1.
    (scale*(expected+3).square()).backward();reference.step()
    exact(parameter, expected)
    exact(optimizer.state[parameter], reference.state[expected])


def test_corrupt_scheduler_refused_before_extra_state_changes(tmp_path):
    model = ProtoSurv(data(tmp_path/'data'), **FAST)
    model.make_optimizer(5e-5, 1e-5)
    before = copy.deepcopy(model.get_extra_state())
    state = copy.deepcopy(before);state['scheduler']['T_max'] = 100
    with pytest.raises(ValueError, match='scheduler checkpoint'):
        model.set_extra_state(state)
    exact(before, model.get_extra_state())


def test_real_cli_demo_check_train_evaluate(tmp_path, capsys):
    from milplus.cli import main
    root, output = tmp_path/'data', tmp_path/'run'
    assert main(['demo', '--protosurv', '--patients', '20', '--output', str(root)]) == 0
    assert main(['check', '--data', str(root), '--method', 'ProtoSurv', '--deep']) == 0
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'epochs': 2, 'threads': 1, 'model': FAST}))
    assert main(['run', '--data', str(root), '--method', 'ProtoSurv', '--config', str(config), '--output', str(output)]) == 0
    assert not (output/'test_predictions.csv').exists()
    assert main(['evaluate', '--run', str(output)]) == 0
    assert (output/'test_predictions.csv').is_file()
    assert read_json(output/'request.json')['method'] == 'protosurv_repaired'
    assert read_json(output/'request.json')['synthetic'] is True
    capsys.readouterr()

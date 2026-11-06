"""Pinned author recipe, independent WKV sums, heterogeneous targets and recovery."""
import ast
import copy
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json
from milplus.multitask import multitask_metrics
from milplus.pathrwkv import PathRWKVMIL, corrected_loss
from milplus.pathrwkv_backend import COMMIT, Backend, recurrence
from milplus.pathrwkv_data import PathInputs, TrainingSlides
from milplus.pathrwkv_demo import create_pathrwkv_demo
from milplus.pathrwkv_source import FactorizedSlidePE
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):exact(x, y)
    else:
        assert a == b


def data(path, task='multitask', dimension=16):
    return Bundle(create_pathrwkv_demo(path, task_type=task, dimension=dimension))


def model_for(bundle, profile='pathrwkv_repaired', **options):
    return PathRWKVMIL(bundle, profile=profile, **{
        'hidden': 64, 'grids': 64, 'max_tiles': 7,
        'inference_chunk': 5, 'precision': '32', 'backend': 'torch', **options})


def cfg(**options):
    return {'epochs': 3, 'patience': 10, 'lr': 1e-4, 'weight_decay': .01,
            'max_instances': None, 'grad_clip': None,
            'selection_metric': 'pathrwkv_val_loss', **options}


def independent_definitions(relative, names, scope):
    root, sha = verified_source('pathrwkv')
    assert sha == COMMIT
    tree = ast.parse(pinned_text(root, sha, relative))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    context = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_author_'+relative, 'exec'), context)  # noqa: S102 -- pinned Git objects
    return SimpleNamespace(**context)


def independent_source(args, hidden=64, grids=64, chunk=5):
    """Actual author definitions, without the product import/sampling/trainer layer."""
    pos = independent_definitions('DownStream/model/pe.py',
        ['get_1d_sincos_pos_embed_from_grid', 'get_2d_sincos_pos_embed_from_grid',
         'get_2d_sincos_pos_embed', 'SlidePE'], {'torch': torch, 'nn': nn})
    # WKV tensor reference has an independently tested closed-form oracle below.
    kernel = Backend('torch')
    source = independent_definitions('DownStream/model/pathrwkv.py',
        ['TimeMix', 'ChannelMix', 'Block', 'PathRWKVv6'],
        {'torch': torch, 'nn': nn, 'HEAD_SIZE': 64, 'CHUNK_LEN_INFERENCE': chunk,
         'SlidePE': pos.SlidePE,
         'cuda_wkv_parallel': lambda *x: kernel(*x),
         'cuda_wkv_state': lambda *x: kernel(*x)})

    class LightningShim(nn.Module):
        def log(self, *args, **kwargs):pass

    pipe = independent_definitions('DownStream/utils/pipeline.py', ['Head', 'WSIPipeline'],
        {'torch': torch, 'nn': nn, 'optim': optim, 'math': math, 'LambdaLR': LambdaLR,
         'Path': Path, 'LightningModule': LightningShim, 'PathRWKV': source.PathRWKVv6})
    cox = independent_definitions('DownStream/utils/utils.py', ['CoxPHLoss'], {'torch': torch, 'nn': nn})
    result = pipe.WSIPipeline.__new__(pipe.WSIPipeline)
    nn.Module.__init__(result)
    result.args = copy.deepcopy(args)
    result.backbone = source.PathRWKVv6(result.args, embed_dim=hidden, n_layers=2, slide_ngrids=grids)
    result.input_projector = nn.Sequential(nn.Linear(args.input_dim, hidden), nn.ReLU(), nn.Dropout(.25))
    result.head = pipe.Head(result.args, hidden)
    result.init_weights()
    losses = {'cls': nn.CrossEntropyLoss, 'reg': nn.SmoothL1Loss, 'surv': cox.CoxPHLoss}
    result.loss_functions = nn.ModuleDict({name: losses[spec['type']]() for name, spec in args.tasks.items()})
    return result


def independent_dataset(bundle, tasks, subset, max_tiles):
    """Actual source __getitem__/randperm/padding, with only the storage bridge changed."""
    module = independent_definitions('DownStream/utils/dataset.py', ['SlideDataset'],
        {'torch': torch, 'Dataset': Dataset, 'math': math})
    rows = []
    for row in bundle.records(subset):
        item = {'data_path': bundle.root/row['path']}
        for spec in tasks.specs:
            value = bundle.multitask.values[row['slide_id']][spec['name']]
            if value is not None and spec['type'] == 'classification':value = spec['classes'][value]
            if value is not None and spec['type'] == 'survival':value = '|'.join(map(str, value))
            item[spec['name']] = value
        rows.append(item)

    class Frame:
        height = len(rows)
        def row(self, index, named):return rows[index]

    result = module.SlideDataset.__new__(module.SlideDataset)
    result.df, result.mode = Frame(), 'Train' if subset == 'train' else 'Val'
    result.args = SimpleNamespace(tasks=tasks.source_tasks, max_tiles=max_tiles)

    def storage(self, path):
        with h5py.File(path) as handle:
            x = torch.from_numpy(handle['features'][:])
            # This fixture explicitly declares top-left level0 xy, pitch256.
            c = torch.from_numpy((handle['coords'][:][:, ::-1]//256*224).copy()).float()
        return x, c

    result._load_tiles = MethodType(storage, result)
    return result


def closed_form(r, k, v, w, u, initial):
    """Causal weighted sums over historical tokens, with no recursive state scan."""
    decay = (-w.exp()).exp()
    values = []
    for t in range(r.shape[1]):
        current = (initial*decay[:, :t].prod(1)[:, :, None, :]*r[:, t, :, None, :]).sum(-1)
        for s in range(t):
            current = current+v[:, s]*(r[:, t]*k[:, s]*decay[:, s+1:t].prod(1)).sum(-1, keepdim=True)
        values.append(current+v[:, t]*(r[:, t]*k[:, t]*u).sum(-1, keepdim=True))
    state = initial*decay.prod(1)[:, :, None, :]
    for s in range(r.shape[1]):
        state = state+v[:, s, :, :, None]*k[:, s, :, None, :]*decay[:, s+1:].prod(1)[:, :, None, :]
    return torch.stack(values, 1), state


@pytest.mark.parametrize('shape', [(1, 1, 1, 4), (2, 7, 2, 4), (1, 9, 3, 64)])
def test_wkv_independent_causal_sums_state_and_all_six_gradients(shape):
    seed_everything(71)
    b, t, h, n = shape
    values = [torch.randn(*shape, dtype=torch.double, requires_grad=True) for _ in range(4)]
    u = torch.randn(h, n, dtype=torch.double, requires_grad=True)
    state = torch.randn(b, h, n, n, dtype=torch.double, requires_grad=True)
    actual = recurrence(*values, u, state)
    expected = closed_form(*values, u, state)
    for x, y in zip(actual, expected):torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)
    weights = [torch.randn_like(x) for x in actual]
    grad = torch.autograd.grad(sum((x*w).sum() for x, w in zip(actual, weights)), [*values, u, state])
    other = torch.autograd.grad(sum((x*w).sum() for x, w in zip(expected, weights)), [*values, u, state])
    for x, y in zip(grad, other):torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)
    split = max(1, t//2)
    left, carry = recurrence(*[x[:, :split] for x in values], u, state)
    if split < t:
        right, carry = recurrence(*[x[:, split:] for x in values], u, carry)
        left = torch.cat([left, right], 1)
    exact((left, carry), actual)


def test_explicit_native_backend_rejects_cpu_and_no_fallback():
    args = [torch.ones(1, 3, 64) for _ in range(4)]+[torch.ones(1, 64)]
    with pytest.raises(ValueError, match='CPU; no fallback'):Backend('native')(*args)
    with pytest.raises(ValueError, match='no auto fallback'):Backend('auto')
    with pytest.raises(ValueError, match='64-channel'):Backend('torch')(*[x[..., :63] for x in args])


@pytest.mark.parametrize('hidden', [64, 768])
@pytest.mark.parametrize('prefix', [0, 3])
def test_factorized_position_exact_source_table_and_yx_order(hidden, prefix):
    original = independent_definitions('DownStream/model/pe.py',
        ['get_1d_sincos_pos_embed_from_grid', 'get_2d_sincos_pos_embed_from_grid',
         'get_2d_sincos_pos_embed', 'SlidePE'], {'torch': torch, 'nn': nn})
    reference, actual = original.SlidePE(hidden, 8, prefix), FactorizedSlidePE(hidden, 8, prefix)
    coords = torch.tensor([[[0., 0.], [224., 448.], [7*224., 7*224.]],
                           [[224., 0.], [0., 224.], [7*224., 0.]]])
    x = torch.randn(2, 3, hidden)
    tokens = torch.randn(1, prefix, hidden) if prefix else None
    exact(actual.coords_to_pos(coords), reference.coords_to_pos(coords))
    exact(actual(x, coords, tokens), reference(x, coords, tokens))
    native = FactorizedSlidePE(hidden, 1200)
    assert native.one_dim.shape == (1200, hidden//2)
    assert native.one_dim.numel()*4 < 2*1024**2
    with pytest.raises(ValueError, match='outside'):actual(x, coords+8*224)


@pytest.mark.parametrize('coordinate_order', ['xy', 'yx'])
@pytest.mark.parametrize('origin', ['top_left', 'center'])
def test_explicit_physical_coordinates_scale_orientation_and_bounds(tmp_path, coordinate_order, origin):
    bundle = data(tmp_path/'data')
    bundle.encoder['spatial'].update(coordinate_order=coordinate_order, origin=origin)
    inputs = PathInputs(bundle, 64)
    raw = np.array([[3., 5.], [1., 7.]])*256+(128 if origin == 'center' else 0)
    value = inputs.coordinates(raw)
    expected = np.array([[3., 5.], [1., 7.]])
    if coordinate_order == 'xy':expected = expected[:, ::-1]
    exact(value, torch.from_numpy((expected*224).copy()).float())
    with pytest.raises(ValueError, match='precedes|precede'):inputs.coordinates(raw-20000)
    with pytest.raises(ValueError, match='exceed'):inputs.coordinates(raw+64*256)
    with pytest.raises(ValueError, match='real numbers'):inputs.coordinates(raw.astype(complex))


@pytest.mark.parametrize('max_tiles', [7, 45])
def test_sampling_exact_source_item_padding_and_rng(tmp_path, max_tiles):
    bundle = data(tmp_path/'data')
    model = model_for(bundle, max_tiles=max_tiles)
    actual = TrainingSlides(model.inputs, model.tasks, bundle.records('train'), max_tiles)
    reference = independent_dataset(bundle, model.tasks, 'train', max_tiles)
    for i in [0, 1, len(reference)-1]:
        seed_everything(123+i)
        before = rng_state()
        item = actual[i]
        after = rng_state()
        restore_rng(before)
        source = reference[i]
        exact({k: item[k] for k in source}, source)
        exact(rng_state(), after)


@pytest.mark.parametrize('dimension,hidden', [(16, 64), (1024, 768), (1536, 768)])
def test_actual_source_initialization_forward_joint_loss_all_parameter_gradients(tmp_path, dimension, hidden):
    bundle = data(tmp_path/'data', dimension=dimension)
    seed_everything(47)
    actual = model_for(bundle, 'pathrwkv_source', hidden=hidden)
    init_rng = rng_state()
    seed_everything(47)
    reference = independent_source(actual.network.args, hidden)
    exact(rng_state(), init_rng)
    exact(actual.network.state_dict(), reference.state_dict())
    # The source pipeline's global Xavier init overwrites TimeMix's zero W_o.
    assert actual.network.backbone.blocks[0].time_mix.W_o.weight.abs().sum() > 0
    x = torch.randn(4, 7, dimension, requires_grad=True)
    c = torch.arange(7).view(1, 7, 1).expand(4, -1, 2).float()*224
    other = x.detach().clone().requires_grad_()
    targets, observed = actual.tasks.tensors(bundle.records('train')[:4])
    before = rng_state()
    out = actual.network(x, c)
    after = rng_state()
    restore_rng(before)
    expected = reference(other, c)
    exact(out, expected);exact(rng_state(), after)
    loss = actual.objective(out, targets, observed)
    ref_loss = reference.compute_loss(expected, targets)
    exact(loss, ref_loss)
    loss.backward();ref_loss.backward()
    exact(x.grad, other.grad)
    for (name, p), (ref_name, q) in zip(actual.network.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        exact(p.grad, q.grad)
        if p.grad is not None:assert torch.isfinite(p.grad).all()
    assert all(head.weight.grad is not None for head in actual.network.head.task_heads.values())


def test_source_cox_broadcast_counterexample_and_repaired_breslow_ties(tmp_path):
    model = model_for(data(tmp_path/'data'))
    source = model.network.loss_functions['OS']
    risk = torch.tensor([[.3], [-.4], [1.2], [.8]], dtype=torch.double, requires_grad=True)
    target = torch.tensor([[2., 1.], [2., 0.], [4., 1.], [5., 0.]], dtype=torch.double)
    order = target[:, 0].argsort(descending=True)
    nll = risk[order].logcumsumexp(0)-risk[order]
    # Actual [B,1]*[B] broadcasting, retained only in the explicitly named source route.
    expected_source = (nll*target[order, 1]).sum()/target[:, 1].sum()
    torch.testing.assert_close(source(risk, target), expected_source, rtol=1e-14, atol=1e-14)
    repaired = corrected_loss({'OS': risk}, {'OS': target}, {'OS': torch.ones(4, dtype=torch.bool)}, model.tasks.specs[2:])
    flat = risk.float().flatten()
    manual = sum(torch.logsumexp(flat[target[:, 0] >= target[i, 0]], 0)-flat[i]
                 for i in range(4) if target[i, 1])/int(target[:, 1].sum())
    torch.testing.assert_close(repaired, manual, rtol=1e-6, atol=1e-6)
    assert abs(float(repaired-source(risk, target))) > .1
    grad = torch.autograd.grad(repaired, risk)[0]
    expected_grad = torch.autograd.grad(manual, risk)[0]
    torch.testing.assert_close(grad, expected_grad)
    zeros = corrected_loss({'OS': risk}, {'OS': target*target.new_tensor([1., 0.])},
        {'OS': torch.ones(4, dtype=torch.bool)}, model.tasks.specs[2:])
    assert torch.autograd.grad(zeros, risk, allow_unused=True)[0] is None


def test_explicit_missingness_preserves_valid_minus_one_regression(tmp_path):
    root = create_pathrwkv_demo(tmp_path/'data')
    frame = pd.read_csv(root/'targets.csv', dtype=str, keep_default_na=False)
    bundle = Bundle(root)
    patient = bundle.records('train')[0]['global_patient_id']
    ids = bundle.rows[bundle.rows.global_patient_id == patient].slide_id
    frame.loc[frame.slide_id.isin(ids), 'age'] = '-1'
    frame.to_csv(root/'targets.csv', index=False)
    bundle = Bundle(root)
    model = model_for(bundle)
    rows = [r for r in bundle.records('train') if r['global_patient_id'] == patient]
    targets, masks = model.tasks.tensors(rows)
    assert masks['age'].all() and targets['age'].eq(-1).all()
    prediction = {'age': torch.ones(len(rows), 1, requires_grad=True)}
    loss = corrected_loss(prediction, {'age': targets['age']}, {'age': masks['age']}, [model.tasks.specs[1]])
    loss.backward()
    assert prediction['age'].grad.abs().sum() > 0
    source_loss = model.network.source_loss({s['name']: torch.zeros(len(rows), 2 if s['type'] == 'classification' else 1)
        for s in model.tasks.specs}, targets)
    assert torch.isfinite(source_loss)


@pytest.mark.parametrize('problem', ['duplicate_id', 'unknown_class', 'partial_survival', 'patient_conflict', 'no_coverage', 'nonfinite', 'all_missing'])
def test_heterogeneous_target_contract_rejects_ambiguous_or_leaking_data(tmp_path, problem):
    root = create_pathrwkv_demo(tmp_path/'data')
    frame = pd.read_csv(root/'targets.csv', dtype=str, keep_default_na=False)
    if problem == 'duplicate_id':frame.loc[1, 'slide_id'] = frame.loc[0, 'slide_id']
    elif problem == 'unknown_class':frame.loc[0, 'disease'] = 'unmapped'
    elif problem == 'partial_survival':frame.loc[0, ['OS_time', 'OS_event']] = ['2', '']
    elif problem == 'patient_conflict':frame.loc[2, 'age'] = '9999'
    elif problem == 'no_coverage':frame = frame.iloc[:-1]
    elif problem == 'nonfinite':frame.loc[0, 'age'] = 'nan'
    else:frame.loc[0, ['disease', 'age', 'OS_time', 'OS_event']] = ''
    frame.to_csv(root/'targets.csv', index=False)
    with pytest.raises(ValueError):Bundle(root)


@pytest.mark.parametrize('profile', ['pathrwkv_source', 'pathrwkv_repaired'])
@pytest.mark.parametrize('length', [1, 5, 6, 13])
def test_stateful_inference_source_reset_vs_repaired_full_sequence(tmp_path, profile, length):
    bundle = data(tmp_path/'data')
    model = model_for(bundle, profile)
    model.eval()
    x = torch.randn(length, bundle.dim)
    c = torch.arange(length).view(-1, 1).expand(-1, 2).float()*224
    with torch.no_grad():
        actual = model(x, c)['outputs']
        if profile == 'pathrwkv_source':expected = model.network(x[None], c[None])
        else:
            projected = model.network.input_projector(x[None])
            expected = model.network.head(model.network.backbone.forward_parallel(projected, c[None]))
        for name in actual:torch.testing.assert_close(actual[name], expected[name], atol=5e-6, rtol=5e-6)
    assert all(not m._forward_hooks for block in model.network.backbone.blocks
               for m in [block.time_mix.time_shift, block.channel_mix.time_shift])


@pytest.mark.parametrize('hidden,max_tiles', [(64, 7), (768, 45)])
def test_independent_author_complete_multi_epoch_recipe_optimizer_scheduler_and_rng(tmp_path, hidden, max_tiles):
    bundle = data(tmp_path/'data')
    seed_everything(31)
    actual = model_for(bundle, 'pathrwkv_source', hidden=hidden, max_tiles=max_tiles)
    actual.configure_training(cfg())
    optimizer = actual.make_optimizer(1e-4, .01)
    reference = independent_source(actual.network.args, hidden)
    reference.load_state_dict(actual.network.state_dict())
    recipe = reference.configure_optimizers()
    ref_optim, scheduler = recipe['optimizer'], recipe['lr_scheduler']['scheduler']
    train_set = independent_dataset(bundle, actual.tasks, 'train', max_tiles)
    val_set = independent_dataset(bundle, actual.tasks, 'val', max_tiles)
    for epoch in range(2):
        before = rng_state()
        losses = actual.epoch(bundle, optimizer)
        actual.finish_epoch(bundle, 'cpu', cfg())
        frame = actual.predict_subset(bundle, 'val', 'cpu', cfg())
        actual.after_validation(actual.validation_metrics(frame))
        after = rng_state()
        restore_rng(before)
        reference.train()
        expected = []
        for i, batch in enumerate(DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True, num_workers=0)):
            ref_optim.zero_grad(set_to_none=True)
            loss = reference.training_step(batch, i)
            loss.backward();ref_optim.step()
            expected.append(float(loss.detach()))
        scheduler.step()
        reference.eval()
        val_losses = []
        with torch.no_grad():
            for batch in DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0):
                output = reference(batch['image_features'], batch['coords_yx'])
                val_losses.append(float(reference.compute_loss(output, batch['task_labels'])))
        exact([l['loss'] for l in losses], expected)
        exact(actual.network.state_dict(), reference.state_dict())
        exact(optimizer.state_dict(), ref_optim.state_dict())
        exact(actual.scheduler.state_dict(), scheduler.state_dict())
        exact(frame._pathrwkv_loss.tolist(), val_losses)
        exact(after, rng_state())
        assert actual.optimizer_steps == (epoch+1)*(len(train_set)//4)


def test_early_stop_delta_separate_from_strict_best_checkpoint(tmp_path):
    model = model_for(data(tmp_path/'data'))
    model.configure_training(cfg(patience=2))
    model.after_validation({'pathrwkv_val_loss': 1.})
    assert model.early_best == 1. and model.early_wait == 0
    tiny = 1.-model.min_delta/2
    model.after_validation({'pathrwkv_val_loss': tiny})
    assert model.selection_objective({'pathrwkv_val_loss': tiny}, 1, 0.) > -1.
    assert model.early_best == 1. and model.early_wait == 1
    model.after_validation({'pathrwkv_val_loss': tiny})
    assert model.training_done


def test_epoch_boundary_resume_model_optimizer_scheduler_rng_and_predictions(tmp_path):
    bundle = data(tmp_path/'data')
    seed_everything(42)
    model = model_for(bundle)
    model.configure_training(cfg());optimizer = model.make_optimizer(1e-4, .01)
    model.epoch(bundle, optimizer);model.finish_epoch(bundle, 'cpu', cfg())
    prediction = model.predict_subset(bundle, 'val', 'cpu', cfg())
    model.after_validation(model.validation_metrics(prediction))
    saved = copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
    uninterrupted = model.epoch(bundle, optimizer)
    model.finish_epoch(bundle, 'cpu', cfg())
    expected = model.predict_subset(bundle, 'val', 'cpu', cfg())
    model.after_validation(model.validation_metrics(expected))
    final = copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
    resumed = model_for(bundle)
    resumed.configure_training(cfg());other = resumed.make_optimizer(1e-4, .01)
    resumed.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    exact(resumed.epoch(bundle, other), uninterrupted)
    resumed.finish_epoch(bundle, 'cpu', cfg())
    actual = resumed.predict_subset(bundle, 'val', 'cpu', cfg())
    resumed.after_validation(resumed.validation_metrics(actual))
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    exact(resumed.state_dict(), final['model']);exact(other.state_dict(), final['optimizer']);exact(rng_state(), final['rng'])
    prediction_only = model_for(bundle)
    prediction_only.load_state_dict(saved['model'])
    pd.testing.assert_frame_equal(prediction_only.predict_subset(bundle, 'val', 'cpu', cfg()), prediction, check_exact=True)
    with pytest.raises(ValueError, match='budget changed'):
        changed = model_for(bundle);changed.configure_training(cfg(epochs=5));changed.load_state_dict(saved['model'])


def test_multitask_metrics_known_patient_rows_and_undefined_endpoint(tmp_path):
    bundle = data(tmp_path/'data')
    model = model_for(bundle)
    prediction = model.predict_subset(bundle, 'val', 'cpu', cfg())
    actual = multitask_metrics(bundle, prediction)
    for i, spec in enumerate(bundle.multitask.specs):
        known = prediction[prediction[f't{i}_observed']]
        assert actual['tasks'][spec['name']]['n'] == known.global_patient_id.nunique()
    prediction['t2_observed'] = False
    assert multitask_metrics(bundle, prediction)['tasks']['OS'] == {'n': 0, 'unit': 'patient', 'defined': False}


@pytest.mark.parametrize('task', ['classification', 'regression', 'survival'])
def test_standard_task_routes_are_real_full_updates_and_reload(tmp_path, task):
    bundle = data(tmp_path/'data', task)
    model = model_for(bundle)
    model.configure_training(cfg());optimizer = model.make_optimizer(1e-4, .01)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    losses = model.epoch(bundle, optimizer);model.finish_epoch(bundle, 'cpu', cfg())
    prediction = model.predict_subset(bundle, 'val', 'cpu', cfg())
    model.after_validation(model.validation_metrics(prediction))
    assert losses and any(not torch.equal(v, before[k]) for k, v in model.named_parameters())
    replica = model_for(bundle);replica.load_state_dict(copy.deepcopy(model.state_dict()))
    pd.testing.assert_frame_equal(replica.predict_subset(bundle, 'val', 'cpu', cfg()), prediction, check_exact=True)


@pytest.mark.parametrize('format', ['npy', 'pt', 'h5'])
def test_real_storage_formats_indexed_sampling_and_chunk_alignment(tmp_path, format):
    bundle = data(tmp_path/'data')
    frame = pd.read_csv(bundle.root/'features.csv')
    original = {}
    for index, row in frame.iterrows():
        with h5py.File(bundle.root/row.path) as handle:
            x, c = handle['features'][:], handle['coords'][:]
        original[row.slide_id] = (x, c)
        if format == 'h5':continue
        p = (bundle.root/row.path).with_suffix('.'+format)
        cp = p.with_name(p.stem+'_coords.npy')
        np.save(cp, c)
        if format == 'npy':np.save(p, x)
        else:torch.save({'features': torch.from_numpy(x)}, p)
        frame.loc[index, 'path'] = str(p.relative_to(bundle.root))
        frame.loc[index, 'coords_path'] = str(cp.relative_to(bundle.root))
    frame.to_csv(bundle.root/'features.csv', index=False)
    bundle = Bundle(bundle.root)
    inputs = PathInputs(bundle, 64)
    row = bundle.records('train')[0]
    x, coords = original[row['slide_id']]
    expected_coords = torch.from_numpy((coords[:, ::-1]//256*224).copy()).float()
    pieces = list(inputs.chunks(row, 7))
    exact(torch.cat([a for a, _ in pieces]), torch.from_numpy(x))
    exact(torch.cat([b for _, b in pieces]), expected_coords)
    seed_everything(51)
    before = rng_state();ids = torch.randperm(len(x))[:7];after = rng_state()
    restore_rng(before)
    actual = inputs.sampled(row, 7)
    exact(actual, (torch.from_numpy(x)[ids], expected_coords[ids]));exact(rng_state(), after)


@pytest.mark.parametrize('problem', ['external_link', 'soft_link', 'virtual_features', 'nonfinite'])
def test_storage_rejects_non_self_contained_or_nonfinite_feature_data(tmp_path, problem):
    bundle = data(tmp_path/'data')
    row = bundle.records('train')[0]
    path = bundle.root/row['path']
    with h5py.File(path, 'r+') as handle:
        x = handle['features'][:]
        del handle['features']
        if problem == 'external_link':handle['features'] = h5py.ExternalLink('other.h5', 'features')
        elif problem == 'soft_link':
            handle['other'] = x;handle['features'] = h5py.SoftLink('other')
        elif problem == 'virtual_features':
            layout = h5py.VirtualLayout(shape=x.shape, dtype=x.dtype)
            layout[:] = h5py.VirtualSource('other.h5', 'features', shape=x.shape)
            handle.create_virtual_dataset('features', layout)
        else:x[0, 0] = np.nan;handle['features'] = x
    with pytest.raises(ValueError):PathInputs(bundle, 64).full(row)


@pytest.mark.parametrize('profile', ['pathrwkv_source', 'pathrwkv_repaired'])
def test_unified_engine_complete_resume_selection_and_checkpoint_reload(tmp_path, profile):
    bundle = data(tmp_path/'data')
    config = {'epochs': 3, 'threads': 1, 'model': {
        'backend': 'torch', 'hidden': 64, 'precision': '32', 'max_tiles': 7, 'inference_chunk': 5}}
    whole = train(bundle.root, profile, tmp_path/'whole', config=config)
    partial = train(bundle.root, profile, tmp_path/'resumed', config=config, stop_after=1)
    assert read_json(partial/'status.json')['state'] == 'interrupted'
    assert not (whole/'test_predictions.csv').exists() and not (partial/'test_predictions.csv').exists()
    train(bundle.root, profile, partial, config=config, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [whole, partial]]
    exact(a['model'], b['model']);exact(a['optimizers'], b['optimizers']);exact(a['rng'], b['rng']);exact(a['history'], b['history'])
    values = [h['validation']['pathrwkv_val_loss'] for h in a['history']]
    selected = values.index(min(values))
    assert torch.load(whole/'best.pt', weights_only=False)['epoch'] == selected
    metrics = evaluate(whole)
    checksum = digest_file(whole/'test_predictions.csv')
    assert metrics == evaluate(whole) == evaluate(partial)
    assert checksum == digest_file(whole/'test_predictions.csv') == digest_file(partial/'test_predictions.csv')
    request = read_json(whole/'request.json')
    assert request['config']['lr'] == 1e-4 and request['config']['weight_decay'] == .01
    assert request['config']['grad_clip'] is None and request['config']['selection_metric'] == 'pathrwkv_val_loss'
    assert a['model']['_extra_state']['optimizer_steps'] == 3*(len(bundle.records('train'))//4)


def test_heldout_targets_never_change_training_or_validation_selection(tmp_path):
    first = data(tmp_path/'first')
    second = data(tmp_path/'second')
    frame = pd.read_csv(second.root/'targets.csv', dtype=str, keep_default_na=False)
    test_ids = {r['slide_id'] for r in second.records('test')}
    mask = frame.slide_id.isin(test_ids)
    frame.loc[mask, 'disease'] = frame.loc[mask, 'disease'].map({'class_0': 'class_1', 'class_1': 'class_0'})
    frame.loc[mask, 'age'] = '999'
    frame.loc[mask, 'OS_time'] = '999';frame.loc[mask, 'OS_event'] = '1'
    frame.to_csv(second.root/'targets.csv', index=False)
    config = {'epochs': 2, 'threads': 1, 'model': {
        'backend': 'torch', 'hidden': 64, 'precision': '32', 'max_tiles': 7, 'inference_chunk': 5}}
    one, two = [train(b.root, 'PathRWKV', tmp_path/name, config=config)
                for b, name in [(first, 'one'), (second, 'two')]]
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [one, two]]
    exact({k: v for k, v in a['model'].items() if torch.is_tensor(v)},
          {k: v for k, v in b['model'].items() if torch.is_tensor(v)})
    exact(a['history'], b['history']);exact(a['optimizers'], b['optimizers'])
    assert a['best'] == b['best']


def test_multitask_matrix_blocks_incompatible_methods_and_default_native_is_explicit(tmp_path):
    from milplus.registry import check_compatibility, inspect_method
    bundle = data(tmp_path/'data')
    assert inspect_method('PathRWKV')['id'] == 'pathrwkv_repaired'
    with pytest.raises(ValueError, match='does not support task multitask'):
        check_compatibility(inspect_method('ABMIL'), bundle)
    native = PathRWKVMIL(bundle, hidden=64, max_tiles=7)
    assert native.backend.name == 'native'
    with pytest.raises(ValueError, match='Native WKV6 cannot execute on CPU'):
        native.network(torch.ones(4, 7, bundle.dim), torch.zeros(4, 7, 2))



"""Native two-stage raw-image CNN/RNN training, input contracts and exact continuation."""
import ast
import copy
import gc
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import models, transforms

from milplus.campanella import CampanellaMIL, FlatTiles, campanella_epoch, state_sha256
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file, write_json
from milplus.metrics import classification_metrics, paired_bootstrap
from milplus.raw_region_demo import create_raw_region_demo
from milplus.raw_regions import RESNET34_V1_SHA256, grid_sha256
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)
    yield
    gc.collect()


def weight_path():
    p = Path(os.environ.get('MILPLUS_RESNET34_WEIGHTS', '/data/milplus_model_weights/resnet34-b627a593.pth'))
    assert p.is_file() and digest_file(p) == RESNET34_V1_SHA256, 'Download official ResNet34V1 weights for native source verification'
    return p


def bundle(path, pretrained=True, **options):
    return Bundle(create_raw_region_demo(path, initial_weights=weight_path() if pretrained else None, **options))


def fixture(path, **options):
    b = bundle(path/'data')
    cfg = {'cnn_epochs': 1, 'rnn_epochs': 2, 'cnn_batch_size': 4, 'rnn_batch_size': 4, 'test_every': 1, **options}
    return CampanellaMIL(b, **cfg), b, cfg


def equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def author(filename, names, **extras):
    root, sha = verified_source('campanella_rnn')
    code = subprocess.check_output(['git', 'show', sha+':'+filename], cwd=root, text=True)
    code = code.replace('.cuda()', ".to('cpu')")
    tree = ast.parse(code)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = dict(torch=torch, nn=nn, F=F, np=np, random=random, data=data, sys=sys, openslide=openslide,
              Image=Image, models=models, args=types.SimpleNamespace(nepochs=2, batch_size=4), **extras)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), ns)  # noqa: S102 -- independent pinned source definitions, only CUDA->CPU device patch
    return ns


def native_library(b, rows, path, top_cache=None):
    lib = {'slides': [str(b.root/r['path']) for r in rows], 'targets': [r['target'] for r in rows],
           'grid': [list(map(tuple, b.raw_regions.grid(r) if top_cache is None else
                            b.raw_regions.grid(r)[top_cache[r['slide_id']]['indices']])) for r in rows],
           'level': b.raw_regions.level, 'mult': b.raw_regions.mult}
    torch.save(lib, path)
    return path


@pytest.mark.parametrize('setting,value', [('coordinate_unit', 'grid_index'), ('coordinate_order', 'yx'), ('level', -1),
    ('mult', 0), ('mult', float('nan')), ('mult', True), ('mult', .0001), ('mult', 10000),
    ('provenance', ''), ('kind', 'frozen_features')])
def test_raw_schema_rejects_ambiguous_or_unsafe_coordinates(tmp_path, setting, value):
    b = bundle(tmp_path/'data', pretrained=False)
    c = b.config
    c['encoders']['raw_resnet34']['raw_regions'][setting] = value
    (b.root/'task.yaml').write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        Bundle(b.root)


@pytest.mark.parametrize('bad', [np.ones((3, 2), dtype=float), np.ones((3, 2), dtype=bool),
    np.ones((3, 1), dtype=np.int64), np.empty((0, 2), dtype=np.int64),
    np.array([[0, 0], [0, 0]]), np.array([[-1, 0], [0, 1]]), np.array([[896, 0], [0, 1]])])
def test_real_grid_shapes_values_duplicates_and_canvas_bounds(tmp_path, bad):
    b = bundle(tmp_path/'data', pretrained=False)
    np.save(b.root/b.rows.iloc[0].coords_path, bad)
    with pytest.raises(ValueError):
        b.check(deep=True)


@pytest.mark.parametrize('kind', ['external_link', 'soft_to_external', 'virtual', 'external_storage'])
def test_coordinate_hdf5_cannot_hide_unregistered_storage(tmp_path, kind):
    b = bundle(tmp_path/'data', pretrained=False)
    outside, dest = b.root/'outside.h5', b.root/'grid.h5'
    with h5py.File(outside, 'w') as f:
        f['coords'] = np.array([[0, 0], [224, 0]], dtype=np.int64)
    with h5py.File(dest, 'w') as f:
        if kind == 'external_link':
            f['coords'] = h5py.ExternalLink(str(outside), 'coords')
        elif kind == 'soft_to_external':
            f['hidden'] = h5py.ExternalLink(str(outside), 'coords')
            f['coords'] = h5py.SoftLink('/hidden')
        elif kind == 'virtual':
            layout = h5py.VirtualLayout((2, 2), dtype='i8')
            layout[:] = h5py.VirtualSource(str(outside), 'coords', shape=(2, 2))
            f.create_virtual_dataset('coords', layout)
        else:
            f.create_dataset('coords', (2, 2), dtype='i8', external=[(str(b.root/'external.bin'), 0, 32)])
    row = b.records('train')[0]
    row['coords_path'] = str(dest)
    with pytest.raises(ValueError, match='embedded|external|virtual'):
        b.raw_regions.grid(row)


def test_embedded_hdf5_real_coords_and_bounded_immutable_lru_cache(tmp_path):
    b = bundle(tmp_path/'data', pretrained=False)
    reader = b.raw_regions
    reader.max_grid_cache_bytes = 192
    rows = b.records('train')
    a = reader.grid(rows[0])
    with pytest.raises(ValueError):
        a[0, 0] = 5
    reader.grid(rows[1])
    assert len(reader._grids) == 1 and reader._grid_cache_bytes == 192
    with h5py.File(b.root/'embedded.h5', 'w') as f:
        f['coords'] = a
        f['alias'] = h5py.SoftLink('/coords')
    row = {**rows[0], 'coords_path': 'embedded.h5'}
    np.testing.assert_array_equal(reader.grid(row), a)
    assert grid_sha256(a) == grid_sha256(a.copy())
    assert grid_sha256(a) != grid_sha256(a[::-1])
    reader.max_rows = 1
    reader._grids.clear()
    with pytest.raises(ValueError, match='bounded'):
        reader.grid(row)


@pytest.mark.parametrize('mult', [1., 1.003])
def test_real_streamed_openslide_regions_equal_both_author_datasets(tmp_path, mult):
    b = bundle(tmp_path/'data', pretrained=False, mult=mult)
    rows = b.records('train')
    lib = native_library(b, rows, tmp_path/'source.pt')
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3, [.1]*3)])
    mil = author('MIL_train.py', ['MILdataset'])['MILdataset'](str(lib), normalize)
    recurrent = author('RNN_train.py', ['rnndata'])['rnndata'](str(lib), 10, False, normalize)
    mil.setmode(1)
    for index in [0, 5, 11]:
        equal(b.raw_regions.read(rows[0], index, 'cnn'), mil[index])
    sequence, label = recurrent[0]
    for index in range(10):
        equal(b.raw_regions.read(rows[0], index, 'rnn'), sequence[index])
    assert label == rows[0]['target']
    if mult != 1:
        assert b.raw_regions.size('cnn') == 225 and b.raw_regions.size('rnn') == 224
        assert not torch.equal(b.raw_regions.read(rows[0], 0, 'cnn'), b.raw_regions.read(rows[0], 0, 'rnn'))
    info = b.check(deep=True)
    assert all(len(r['raw_regions']['grid_sha256']) == 64 for r in info['features'])


def test_raw_streaming_contract_weights_budget_and_no_frozen_fallback(tmp_path, monkeypatch):
    b = bundle(tmp_path/'data')
    assert b.raw_regions.weights == weight_path()
    with pytest.raises(ValueError, match='not frozen'):
        b.load(b.records('train')[0])
    with pytest.raises(ValueError, match='contract'):
        check_compatibility(require_runnable('abmil'), b)
    f = Bundle(create_demo(tmp_path/'features', patients=20))
    with pytest.raises(ValueError, match='contract'):
        check_compatibility(require_runnable('campanella_rnn'), f)
    monkeypatch.setattr(models, 'resnet34', lambda **kw: pytest.fail('Budget failure must precede backbone allocation'))
    with pytest.raises(ValueError, match='max_dense_bytes'):
        CampanellaMIL(b, max_dense_bytes=1024)
    b.raw_regions.max_grid_cache_bytes = 1
    b.raw_regions._grids.clear()
    b.raw_regions._grid_cache_bytes = 0
    with pytest.raises(ValueError, match='before allocation'):
        b.raw_regions.grid(b.records('train')[0])


@pytest.mark.parametrize('options', [{'cnn_epochs': 1, 'test_every': 10}, {'steps': 13}, {'positive_weight': 1},
    {'shuffle_sequence': 1}, {'rnn_lr': float('nan')}, {'ndims': 0}])
def test_two_stage_configuration_guards(tmp_path, options):
    b = bundle(tmp_path/'data', pretrained=False)
    with pytest.raises(ValueError):
        CampanellaMIL(b, **options)


@pytest.mark.parametrize('training', [True, False])
def test_native_full_resnet34_logits_ce_all_gradients_and_bn(tmp_path, training):
    seed_everything(6)
    m, _, _ = fixture(tmp_path)
    reference = models.resnet34(weights=None)
    reference.fc = nn.Linear(512, 2)
    reference.load_state_dict(m.cnn.state_dict())
    m.cnn.train(training)
    reference.train(training)
    x = (torch.rand(2, 3, 224, 224)*10-5).requires_grad_()
    xx = x.detach().clone().requires_grad_()
    target = torch.tensor([0, 1])
    actual, expected = m.cnn(x), reference(xx)
    equal(actual, expected)
    F.cross_entropy(actual, target).backward()
    F.cross_entropy(expected, target).backward()
    equal(x.grad, xx.grad)
    for (name, p), (refname, q) in zip(m.cnn.named_parameters(), reference.named_parameters()):
        assert name == refname and p.grad is not None
        equal(p.grad, q.grad)
    assert m.cnn.conv1.weight.grad.abs().sum() > 0
    equal(m.cnn.state_dict(), reference.state_dict())


def test_native_custom_recurrence_every_state_final_loss_and_bptt(tmp_path):
    m, _, _ = fixture(tmp_path)
    ref = author('RNN_train.py', ['rnn_single'])['rnn_single'](128)
    ref.load_state_dict(m.rnn.state_dict())
    m.rnn.requires_grad_(True)
    x = torch.randn(3, 10, 512, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    state, expected = m.rnn.init_hidden(3), ref.init_hidden(3)
    for step in range(10):
        logits, state = m.rnn(x[:, step], state)
        target_logits, expected = ref(xx[:, step], expected)
        equal(state, expected)
        equal(logits, target_logits)
    loss, reference_loss = F.cross_entropy(logits, torch.tensor([0, 1, 0])), F.cross_entropy(target_logits, torch.tensor([0, 1, 0]))
    loss.backward()
    reference_loss.backward()
    equal(x.grad, xx.grad)
    assert x.grad[:, 0].abs().sum() > 0
    for p, q in zip(m.rnn.parameters(), ref.parameters()):
        equal(p.grad, q.grad)


@pytest.mark.parametrize('weight,batch', [(.5, 4), (.7, 3)])
def test_complete_author_hard_mil_epochs_adam_selection_bn_and_rng(tmp_path, weight, batch):
    seed_everything(6)
    m, b, _ = fixture(tmp_path, cnn_epochs=2, positive_weight=weight, cnn_batch_size=batch)
    optimizers = m.make_optimizers(1e-4, 1e-4)
    reference = models.resnet34(weights=None)
    reference.fc = nn.Linear(512, 2)
    reference.load_state_dict(m.cnn.state_dict())
    native_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4, weight_decay=1e-4)
    src = author('MIL_train.py', ['MILdataset', 'inference', 'train', 'group_argtopk'])
    src['args'].batch_size = batch
    rows = b.records('train')
    lib = native_library(b, rows, tmp_path/'source.pt')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3, [.1]*3)])
    ds = src['MILdataset'](str(lib), transform)
    loader = data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    stream = FlatTiles(m, rows)
    assert stream.pairs is None and len(stream) == 48 and stream.pair(12) == (1, 0)
    for epoch in range(2):
        m.start_epoch(epoch)
        before = rng_state()
        loss = campanella_epoch(m, b, units(b, 'train'), optimizers, torch.device('cpu'), {})[0]['loss']
        after = rng_state()
        restore_rng(before)
        ds.setmode(1)
        probs = src['inference'](epoch, loader, reference)
        top = src['group_argtopk'](np.asarray(ds.slideIDX), probs, 1)
        ds.maketraindata(top)
        ds.shuffletraindata()
        ds.setmode(2)
        native_loss = src['train'](epoch, loader, reference, m.criterion('cpu'), native_optimizer)
        assert loss == native_loss
        equal(m.cnn.state_dict(), reference.state_dict())
        equal(optimizers[0].state_dict(), native_optimizer.state_dict())
        for p, q in zip(m.cnn.parameters(), reference.parameters()):
            equal(p.grad, q.grad)
        equal(after, rng_state())
        selected = [(rows[s]['slide_id'], tuple(b.raw_regions.grid(rows[s])[i])) for s, i in
                    [(next(j for j, r in enumerate(rows) if r['slide_id'] == name), i) for name, i in m.selected_training_rows]]
        assert selected == [(rows[s]['slide_id'], coord) for s, coord, _ in ds.t_data]
    write_json(tmp_path/'campanella_author_cnn.json', {'state': 'passed', 'epochs': 2, 'optimizer_updates': m.cnn_updates,
        'positive_weight': weight, 'batch_size': batch, 'scope': 'Native full ResNet34, real OpenSlide ROI, original inference/top1/train, Adam/BN/grads/RNG bitwise equal, including weighted-CE short final batch'})


@pytest.mark.parametrize('batch,warmup', [(4, 1), (3, 10)])
def test_complete_author_recurrent_epochs_sgd_frozen_encoder_bn_and_rng(tmp_path, batch, warmup):
    seed_everything(6)
    m, b, _ = fixture(tmp_path, cnn_epochs=warmup, rnn_batch_size=batch)
    opts = m.make_optimizers(1e-4, 1e-4)
    for epoch in range(warmup):
        m.start_epoch(epoch)
        campanella_epoch(m, b, units(b, 'train'), opts, torch.device('cpu'), {})
    m.after_validation({'campanella_balanced_accuracy': 1.})
    m.start_epoch(warmup)
    saved = tmp_path/'cnn.pt'
    torch.save({'state_dict': m.cnn_best_state}, saved)
    src = author('RNN_train.py', ['ResNetEncoder', 'rnn_single', 'rnndata', 'train_single', 'test_single', 'errors'])
    embedder = src['ResNetEncoder'](str(saved)).eval().requires_grad_(False)
    recurrent = src['rnn_single'](128)
    recurrent.load_state_dict(m.rnn.state_dict())
    optimizer = torch.optim.SGD(recurrent.parameters(), .1, momentum=.9, dampening=0, weight_decay=1e-4, nesterov=True)
    rows = b.records('train')
    lib = native_library(b, rows, tmp_path/'sequence.pt', m.top_cache)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3, [.1]*3)])
    ds = src['rnndata'](str(lib), 10, False, transform)
    loader = data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    fixed = copy.deepcopy(m.cnn.state_dict())
    for epoch in range(2):
        m.train()
        m.start_epoch(warmup+epoch)
        assert not m.cnn.training and all(not p.requires_grad for p in m.cnn.parameters())
        before = rng_state()
        actual = campanella_epoch(m, b, units(b, 'train'), opts, torch.device('cpu'), {})[0]
        after = rng_state()
        restore_rng(before)
        loss, fpr, fnr = src['train_single'](epoch, embedder, recurrent, loader, m.criterion('cpu'), optimizer)
        assert (actual['loss'], actual['fpr'], actual['fnr']) == (loss, fpr, fnr)
        equal(m.rnn.state_dict(), recurrent.state_dict())
        equal(opts[1].state_dict(), optimizer.state_dict())
        equal(m.cnn.state_dict(), fixed)
        equal(after, rng_state())
        assert all(p.grad is None for p in m.cnn.parameters())
        for p, q in zip(m.rnn.parameters(), recurrent.parameters()):
            equal(p.grad, q.grad)
    write_json(tmp_path/'campanella_author_rnn.json', {'state': 'passed', 'epochs': 2, 'optimizer_updates': m.rnn_updates,
        'batch_size': batch, 'scope': 'Native10-step128-hidden recurrence after full trainable ImageNet ResNet34 hardMIL stage; complete train_single SGD/Nesterov/grads/BN/RNG bitwise equal, including short final batch'})


def test_best_cnn_transition_latest_ties_and_source_validation_intervals(tmp_path):
    m, b, _ = fixture(tmp_path, cnn_epochs=2, test_every=2)
    m.make_optimizers(1e-4, 1e-4)
    m.start_epoch(0)
    assert not m.should_validate(0) and m.should_validate(1)
    m.epoch = 1
    m.after_validation({'campanella_balanced_accuracy': .7})
    best = copy.deepcopy(m.cnn_best_state)
    with torch.no_grad():
        m.cnn.conv1.weight.add_(1)
    m.after_validation({'campanella_balanced_accuracy': .5})
    equal(m.cnn_best_state, best)
    assert not m.selection_improved(.9, .7)
    m.start_epoch(2)
    equal(m.cnn.state_dict(), best)
    assert m.should_validate(2) and m.selection_improved(.7, .7)
    assert m.cnn_best_sha256 == state_sha256(best) and len(m.top_cache) == len(b.records('train'))
    assert all(r['selector_sha256'] == m.cnn_best_sha256 for r in m.top_cache.values())


def test_test_label_independent_top_selection_and_exact_source_tie_decisions(tmp_path):
    m, b, _ = fixture(tmp_path)
    m.make_optimizers(1e-4, 1e-4)
    m.after_validation({'campanella_balanced_accuracy': .5})
    m.start_epoch(1)
    rows = b.records('test')
    m.prepare_top(rows, torch.device('cpu'))
    original = copy.deepcopy(m.top_cache)
    m.top_cache = {}
    m.prepare_top([{**r, 'target': 1-r['target']} for r in rows], torch.device('cpu'))
    for row in rows:
        equal(original[row['slide_id']], m.top_cache[row['slide_id']])
    with torch.no_grad():
        m.rnn.fc3.weight.zero_()
        m.rnn.fc3.bias.zero_()
    frame = m.predict_subset(b, 'test', torch.device('cpu'), {})
    assert (frame.p_1 == .5).all() and (frame.prediction == 1).all()
    assert (frame.campanella_selection_prediction == 0).all()
    assert m.validation_metrics(frame)['campanella_balanced_accuracy'] == .5
    positive = frame.copy()
    positive['target'] = 1
    assert classification_metrics(positive, 2)['accuracy'] == 1
    assert classification_metrics(positive.drop(columns='decision_rule'), 2)['accuracy'] == 0


@pytest.mark.parametrize('interruption', [1, 2])
def test_full_two_phase_exact_resume_before_and_after_transition(tmp_path, interruption):
    b = bundle(tmp_path/'data')
    config = {'threads': 1, 'model': {'cnn_epochs': 1, 'rnn_epochs': 2, 'test_every': 1,
                                    'cnn_batch_size': 4, 'rnn_batch_size': 4}}
    full = train(b.root, 'campanella_rnn', tmp_path/'full', seed=6, config=config)
    partial = train(b.root, 'campanella_rnn', tmp_path/'resumed', seed=6, config=config, stop_after=interruption)
    assert json.loads((partial/'status.json').read_text())['state'] == 'interrupted'
    if interruption == 1:
        assert not (partial/'best.pt').exists()
    train(b.root, 'campanella_rnn', partial, seed=6, config=config, resume=True)
    a, c = [torch.load(p/'last.pt', map_location='cpu', weights_only=False) for p in [full, partial]]
    for key in ['model', 'optimizers', 'rng', 'best', 'bad', 'history']:
        equal(a[key], c[key])
    assert a['model']['_extra_state']['cnn_updates'] == 1
    assert a['model']['_extra_state']['rnn_updates'] == 2
    assert a['model']['rnn.fc1.weight'].shape == (128, 512)
    assert json.loads((full/'method_protocol.json').read_text())['steps'] == 10
    assert evaluate(full) == evaluate(partial)
    assert (full/'test_predictions.csv').read_bytes() == (partial/'test_predictions.csv').read_bytes()
    before = digest_file(partial/'test_predictions.csv')
    evaluate(partial)
    assert before == digest_file(partial/'test_predictions.csv')


def test_scheduled_validation_no_extra_cnn_queries_or_premature_best(tmp_path):
    b = bundle(tmp_path/'data')
    run = train(b.root, 'campanella_rnn', tmp_path/'run', seed=6, stop_after=1,
                config={'threads': 1, 'model': {'cnn_epochs': 2, 'rnn_epochs': 1, 'test_every': 2,
                                              'cnn_batch_size': 4, 'rnn_batch_size': 4}})
    record = torch.load(run/'last.pt', weights_only=False)
    assert record['history'][0]['validation'] == {'validation_performed': False}
    assert record['model']['_extra_state']['cnn_best_state'] is None and not (run/'best.pt').exists()
    assert record['model']['_extra_state']['top_cache'] == {}


def test_decision_rule_survives_patient_fusion_and_paired_cluster_bootstrap():
    left = pd.DataFrame([{'slide_id': f'{i:02d}', 'global_patient_id': str(i//2), 'target': int(i//2 < 6),
                          'subset': 'test', 'p_0': .5, 'p_1': .5, 'decision_rule': 'binary_positive_at_half'} for i in range(16)])
    right = left.drop(columns='decision_rule')
    assert classification_metrics(left, 2, 'patient')['accuracy'] == .75
    assert classification_metrics(right, 2, 'patient')['accuracy'] == .25
    result = paired_bootstrap(left, right, 2, metric='accuracy', unit='patient', repetitions=30, seed=6)
    assert result['difference_left_minus_right'] == .5
    rng = np.random.default_rng(6)
    draws = [2*np.mean(rng.choice(np.arange(8), 8, replace=True) < 6)-1 for _ in range(30)]
    np.testing.assert_allclose(result['ci95'], np.quantile(draws, [.025, .975]))
    bad = left.copy()
    bad.loc[0, 'decision_rule'] = 'argmax'
    with pytest.raises(ValueError, match='decision rule'):
        classification_metrics(bad, 2)


def test_cli_full_native_raw_roi_demo_run_and_evaluate(tmp_path):
    root = tmp_path/'data'
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'threads': 1, 'model': {'cnn_epochs': 1, 'rnn_epochs': 1,
                        'test_every': 1, 'cnn_batch_size': 4, 'rnn_batch_size': 4}}))
    base = [sys.executable, '-m', 'milplus']
    subprocess.run(base+['demo', '--raw-regions', '--patients', '8', '--backbone-weights', str(weight_path()), '--output', str(root)], check=True)
    subprocess.run(base+['check', '--data', str(root), '--method', 'campanella_rnn', '--deep'], check=True)
    subprocess.run(base+['run', '--data', str(root), '--method', 'campanella_rnn', '--config', str(config), '--seed', '6', '--output', str(tmp_path/'run')], check=True)
    subprocess.run(base+['evaluate', '--run', str(tmp_path/'run')], check=True)
    assert len(pd.read_csv(tmp_path/'run/test_predictions.csv')) == 2


@pytest.mark.parametrize('component', ['cnn', 'snapshot'])
def test_corrupted_frozen_selector_or_best_snapshot_cannot_load(tmp_path, component):
    m, _, _ = fixture(tmp_path)
    m.make_optimizers(1e-4, 1e-4)
    m.epoch = 0
    m.after_validation({'campanella_balanced_accuracy': .5})
    m.start_epoch(1)
    state = copy.deepcopy(m.state_dict())
    if component == 'cnn':
        state['cnn.conv1.weight'][0, 0, 0, 0] += 1
    else:
        state['_extra_state']['cnn_best_state']['conv1.weight'][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match='snapshot'):
        m.load_state_dict(state)


def test_partial_edge_region_preserves_native_rgba_to_rgb_pixels(tmp_path):
    b = bundle(tmp_path/'data', pretrained=False)
    row = b.records('train')[0]
    coords = np.array(b.raw_regions.grid(row))
    coords[0] = [895, 671]
    np.save(b.root/row['coords_path'], coords)
    b.raw_regions.describe(row)
    actual = b.raw_regions.read(row, 0, 'cnn')
    with openslide.OpenSlide(str(b.root/row['path'])) as slide:
        original = slide.read_region((895, 671), 0, (224, 224)).convert('RGB')
    expected = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3, [.1]*3)])(original)
    equal(actual, expected)
    assert (actual[:, 1:, 1:] == -5).all()


def test_input_inventory_binds_real_coordinates_order_and_wsi_content(tmp_path):
    b = bundle(tmp_path/'data', pretrained=False)
    first = b.check(deep=True)
    row = b.records('train')[0]
    coords = np.array(b.raw_regions.grid(row))
    np.save(b.root/row['coords_path'], coords[::-1])
    c = Bundle(b.root)
    assert b.fingerprint != c.fingerprint
    assert first['feature_inventory_sha256'] != c.check(deep=True)['feature_inventory_sha256']
    previous = c.raw_regions.inventory(row, True)
    (b.root/row['path']).write_bytes((b.root/b.records('train')[1]['path']).read_bytes())
    assert previous['sha256'] != c.raw_regions.inventory(row, True)['sha256']
    with pytest.raises(ValueError, match='hash mismatch'):
        c.raw_regions.inventory({**row, 'sha256': previous['sha256']}, True)


def test_raw_roi_index_rejects_mixed_frozen_cluster_labels(tmp_path):
    b = bundle(tmp_path/'data', pretrained=False)
    frame = pd.read_csv(b.root/'images.csv')
    frame['cluster_labels_path'] = frame.coords_path
    frame.to_csv(b.root/'images.csv', index=False)
    with pytest.raises(ValueError, match='frozen-feature cluster'):
        Bundle(b.root)

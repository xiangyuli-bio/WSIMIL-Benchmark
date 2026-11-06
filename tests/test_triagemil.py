"""Original TRIAGE-MIL preprocessing, full network/epoch and unified-data verification."""
import ast
import copy
import json
import os
import subprocess
import sys
import types
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json, write_json
from milplus.triagemil import TRIAGEMIL
from milplus.triagemil_data import original_arrays, prepare
from milplus.triagemil_demo import create_triagemil_demo
from milplus.triagemil_source import metric, preprocessing
from milplus.upstream import pinned_text, verified_source

FAST = {'hidden': 16, 'heads': 2, 'hypergraph_layers': 2}


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def data(path, **kwargs):
    return Bundle(create_triagemil_demo(path, patients=20, **kwargs))


def exact(x, y):
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            exact(x.indices(), y.indices());exact(x.values(), y.values());assert x.shape == y.shape
        else:
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    elif isinstance(x, np.ndarray):
        np.testing.assert_array_equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for k in x:exact(x[k], y[k])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for a, b in zip(x, y):exact(a, b)
    else:
        assert x == y


def independent(filename, scope=None, names=None):
    root, sha = verified_source('triagemil')
    code = pinned_text(root, sha, filename)
    tree = ast.parse(code)
    if names is not None:
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert set(names) == {n.name for n in tree.body}
    else:
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and
                                                n.module in {'mass_io_utils', 'mass_core'})]
    name = 'independent_triage_'+filename.replace('/', '_').replace('.', '_')
    module = types.ModuleType(name);module.__dict__.update(scope or {});sys.modules[name] = module
    exec(compile(tree, str(root/filename), 'exec'), module.__dict__)  # noqa: S102 -- independent verified original Git objects
    return module


def original_trainer():
    return independent('src/train_triage_mil.py', {'torch': torch, 'nn': nn,
        'Optional': Optional, 'np': np, 'pd': pd, 'concordance_index': metric()},
        ['cox_ph_loss', 'c_index', 'HypergraphGatedMILSurvival', 'collate_bags', 'train_one_epoch', 'evaluate'])


def original_builders():
    core = independent('src/mass_core.py')
    selector = independent('src/mass_selector.py',
                           {k: v for k, v in vars(core).items() if not k.startswith('__')})
    hypergraph = independent('src/precompute_hypergraphs.py')
    return core, selector, hypergraph


def test_complete_mass_and_hypergraph_equal_independent_original(tmp_path):
    b = data(tmp_path/'data', nodes=160)
    core, selector, graph = original_builders()
    for row in b.records('train')[:3]:
        x, xy = b.load(row, require_coords=True)
        indices, labels, proxy, selection = selector.select_mass_tiles(x.numpy(), xy.numpy(), 4096,
                                                                     core.MASSConfig(), verbose=False)
        H, hierarchy = graph.build_semantic_hierarchical_hypergraph(x.numpy()[indices], xy.numpy()[indices], labels)
        actual = b.triagemil.load(row, rebuild=True)
        exact(actual['features'], x[indices.astype(np.int64)])
        exact(actual['coords'], xy[indices.astype(np.int64)])
        exact(actual['mass_labels'].numpy(), labels);exact(actual['risk_scores'].numpy(), proxy)
        exact(actual['H'], H.coalesce())
        entry = b.triagemil.meta['slides'][row['slide_id']]
        with h5py.File(b.triagemil.path.parent/entry['path'], 'r') as stream:
            exact(json.loads(stream.attrs['metadata_json']), {'selection': selection, 'hierarchy': hierarchy})
        assert len(indices) <= 160 and len(set(indices.tolist())) == len(indices)


@pytest.mark.parametrize('dimension', [16, 1024, 1536])
def test_full_native_network_cox_input_and_all_parameter_gradients(tmp_path, dimension):
    b = data(tmp_path/'data', dimension=dimension, nodes=32)
    seed_everything(35);adapter = TRIAGEMIL(b)
    original = original_trainer()
    seed_everything(35);network = original.HypergraphGatedMILSurvival(dimension)
    exact(adapter.network.state_dict(), network.state_dict())
    samples = [b.triagemil.load(b.records('train')[i]) for i in range(2)]
    features = [s['features'].requires_grad_() for s in samples]
    before = rng_state()
    actual = torch.stack([adapter(x, s['H']) for x, s in zip(features, samples)])
    times, events = torch.tensor([2., 4.]), torch.tensor([1, 0])
    loss = adapter.author.cox_ph_loss(actual, times, events);loss.backward()
    grads = {n: p.grad.clone() if p.grad is not None else None for n, p in adapter.network.named_parameters()}
    input_grads = [x.grad.clone() for x in features]
    after = rng_state()
    for x in features:x.grad = None
    restore_rng(before)
    expected = torch.stack([network(x, s['H']) for x, s in zip(features, samples)])
    reference_loss = original.cox_ph_loss(expected, times, events);reference_loss.backward()
    exact(actual, expected);exact(loss, reference_loss);exact(after, rng_state())
    for name, p in network.named_parameters():exact(grads[name], p.grad)
    for a, x in zip(input_grads, features):exact(a, x.grad)


@pytest.mark.parametrize('accumulation', [1, 2])
def test_actual_original_epochs_adamw_val_and_rng_equal(tmp_path, accumulation):
    b = data(tmp_path/'data')
    seed_everything(35);adapter = TRIAGEMIL(b, accumulation_steps=accumulation, **FAST)
    opt = adapter.make_optimizer(2e-4, 1e-4)
    source = original_trainer()
    seed_everything(35);network = source.HypergraphGatedMILSurvival(b.dim, hidden=16, heads=2, num_hyper_layers=2)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4, weight_decay=1e-4)
    exact(adapter.network.state_dict(), network.state_dict())
    state = rng_state()
    train_samples = [b.triagemil.load(b.triagemil.by_patient[g]) for g in b.triagemil.order['train']]
    val_samples = [b.triagemil.load(b.triagemil.by_patient[g]) for g in b.triagemil.order['val']]
    for epoch in range(3):
        restore_rng(state)
        actual_loss = adapter.epoch(b, opt);adapter.finish_epoch(b, 'cpu', {})
        frame = adapter.predict_subset(b, 'val', 'cpu', {})
        actual_rng = rng_state()
        restore_rng(state)
        loader = DataLoader(train_samples, batch_size=4, shuffle=True, num_workers=0, collate_fn=source.collate_bags)
        source_loss = source.train_one_epoch(network, loader, optimizer, torch.device('cpu'),
            {'gradient_accumulation_steps': accumulation})
        val_loader = DataLoader(val_samples, batch_size=1, shuffle=False, num_workers=0, collate_fn=source.collate_bags)
        ci, native = source.evaluate(network, val_loader, torch.device('cpu'))
        exact(actual_loss[0]['loss'], source_loss)
        exact(adapter.network.state_dict(), network.state_dict());exact(opt.state_dict(), optimizer.state_dict())
        exact(frame.risk.to_numpy(), native.risk.to_numpy())
        exact(frame.native_time.to_numpy(), native.time.to_numpy())
        exact(frame.attrs['triagemil_native_c_index'], ci);exact(actual_rng, rng_state())
        state = actual_rng


def test_all_censored_batches_skip_original_optimizer_step(tmp_path):
    b = data(tmp_path/'data')
    for row in b.triagemil.by_patient.values():
        if row['subset'] == 'train':row['event'] = 0
    m = TRIAGEMIL(b, **FAST);opt = m.make_optimizer(2e-4, 1e-4)
    before = copy.deepcopy(m.network.state_dict())
    loss = m.epoch(b, opt)
    assert loss == [{'loss': 0., 'actual_optimizer_updates': 0}] and m.optimizer_steps == 0
    exact(before, m.network.state_dict());assert not opt.state


def test_source_ties_not_replaced_by_breslow_and_no_event_zero(tmp_path):
    source = original_trainer()
    risk = torch.tensor([.1, .7, -.2], requires_grad=True)
    times, events = torch.tensor([2., 2., 1.]), torch.tensor([1, 1, 1])
    value = source.cox_ph_loss(risk, times, events)
    order = torch.argsort(times, descending=True)
    expected = -(risk[order]-torch.logcumsumexp(risk[order], 0)).mean()
    exact(value, expected)
    zero = source.cox_ph_loss(risk, times, torch.zeros(3, dtype=torch.long));exact(zero, risk.sum()*0)


def test_exact_resume_selected_first_tie_and_repeated_prediction(tmp_path):
    b = data(tmp_path/'data');cfg = {'epochs': 4, 'threads': 1, 'model': FAST}
    train(b.root, 'TRIAGE-MIL', tmp_path/'whole', seed=35, config=cfg)
    train(b.root, 'TRIAGE-MIL', tmp_path/'resume', seed=35, config=cfg, stop_after=2)
    train(b.root, 'TRIAGE-MIL', tmp_path/'resume', seed=35, config=cfg, resume=True)
    a, c = [torch.load(tmp_path/p/'last.pt', weights_only=False) for p in ['whole', 'resume']]
    for k in ['model', 'optimizers', 'rng', 'best', 'bad', 'history']:exact(a[k], c[k])
    best = torch.load(tmp_path/'whole/best.pt', weights_only=False)
    values = [h['validation']['triagemil_selection'] for h in a['history']]
    assert best['epoch'] == values.index(max(values))
    evaluate(tmp_path/'whole');first = digest_file(tmp_path/'whole/test_predictions.csv')
    evaluate(tmp_path/'whole');assert first == digest_file(tmp_path/'whole/test_predictions.csv')


def test_undefined_native_ci_is_not_a_fabricated_score_or_selected_checkpoint(tmp_path):
    b = data(tmp_path/'data');m = TRIAGEMIL(b, **FAST)
    frame = pd.DataFrame({'subset': ['val']});frame.attrs['triagemil_native_c_index'] = None
    result = m.validation_metrics(frame)
    assert result == {'triagemil_native_c_index': None, 'triagemil_selection': -1., 'triagemil_selection_defined': False}
    assert not m.selection_improved(-1., -float('inf'))
    m.completed_epochs = 1;m.after_validation(result)
    assert m.best_ci is None and m.bad_epochs == 1 and not m.training_done


def test_early_stop_minimum_epoch_and_first_tie_rule(tmp_path):
    b = data(tmp_path/'data');m = TRIAGEMIL(b, early_min_epochs=3, **FAST)
    m.patience = 1
    for epoch in range(1, 4):
        m.completed_epochs = epoch;m.after_validation({'triagemil_native_c_index': .7})
        assert m.training_done == (epoch == 3)
    assert m.bad_epochs == 2 and m.best_ci == .7
    assert m.selection_improved(.71, .7) and not m.selection_improved(.7, .7)


def test_fm_coordinates_and_hypergraph_drift_rejected(tmp_path):
    b = data(tmp_path/'data');row = b.records('train')[0]
    e = b.triagemil.meta['slides'][row['slide_id']];path = b.triagemil.path.parent/e['path']
    with h5py.File(path, 'r+') as stream:
        a = stream['indices'][:];stream['indices'][:] = a[::-1]
    with pytest.raises(ValueError, match='contents changed'):b.triagemil.load(row)
    # Even a newly declared file hash is insufficient to claim source-equivalent
    # preprocessing: the optional source rebuild detects a reordered selection.
    meta = read_json(b.triagemil.path);meta['slides'][row['slide_id']]['sha256'] = digest_file(path)
    write_json(b.triagemil.path, meta);changed = Bundle(b.root)
    with pytest.raises(ValueError, match='actual original source builder'):
        changed.triagemil.load(changed.records('train')[0], rebuild=True)


def test_wrong_fm_and_patient_split_schema_rejected(tmp_path):
    b = data(tmp_path/'data');schema = read_json(b.root/'triagemil_schema.json')
    schema['patient_order']['train'][0] = schema['patient_order']['test'][0]
    with pytest.raises(ValueError, match='leakage'):prepare(b, schema, tmp_path/'bad')
    task = yaml.safe_load((b.root/'task.yaml').read_text());task['encoders']['toy']['weights'] = 'another-same-width-model'
    (b.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='FM identity'):Bundle(b.root)


def test_original_fallback_quality_and_real_budget_on_constant_features():
    source = preprocessing();x = torch.ones(25, 16);coords = torch.stack([torch.arange(25), torch.zeros(25)], 1)
    arrays, meta = original_arrays(x, coords, source[0].MASSConfig(min_quality_threshold=.99), 4096,
        {'k_intra': 10, 'k_inter': 5, 'tiles_per_super': 12, 'random_state': 35}, source)
    assert len(arrays['indices']) == 1 and meta['selection']['clean_tiles'] == 1
    assert arrays['H_shape'].tolist() == [1, 1]


def test_malformed_checkpoint_and_hypergraph_bypass_rejected(tmp_path):
    b = data(tmp_path/'data');m = TRIAGEMIL(b, **FAST)
    state = m.get_extra_state()
    for key, value in [('bad_epochs', -1), ('best_ci', float('nan')), ('training_done', True), ('contract', 'wrong')]:
        bad = copy.deepcopy(state);bad[key] = value
        with pytest.raises(ValueError):m.set_extra_state(bad)
    sample = b.triagemil.load(b.records('train')[0])
    with pytest.raises(ValueError, match='sparse hypergraph'):m(sample['features'], sample['H'].to_dense())
    with pytest.raises(ValueError, match='sparse hypergraph'):m(sample['features'], None)


def test_real_cli_check_training_and_selected_reload(tmp_path):
    b = data(tmp_path/'data');config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'epochs': 2, 'threads': 1, 'model': FAST}))
    run = tmp_path/'run'
    for args in [['check', '--data', str(b.root), '--method', 'TRIAGE-MIL', '--deep'],
                 ['run', '--data', str(b.root), '--method', 'TRIAGE-MIL', '--output', str(run), '--config', str(config)],
                 ['evaluate', '--run', str(run)]]:
        result = subprocess.run([sys.executable, '-m', 'milplus', *args], check=False,
            env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, result.stdout
    assert read_json(run/'request.json')['method'] == 'triagemil_source'
    assert read_json(run/'test_metrics.json')['unit'] == 'patient'

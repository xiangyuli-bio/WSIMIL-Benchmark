"""Published PMIL networks, Shapley/pseudo bags, full main loops and exact continuation."""
import ast
import copy
import glob
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.pmil_shapley import (
    PROFILES,
    PMILShapley,
    pmil_epoch,
    shapley_scores,
    source_metrics,
    tensor_tree_sha,
)
from milplus.registry import require_runnable
from milplus.upstream import verified_source

SMALL = {'hidden': 32, 'attention_dim': 16, 'rounds': 2, 'round_epochs': 3, 'max_pseudo': 2,
         'pseudo_step': 1, 'search_rate': 2, 'sample_rate': 2, 'k_sample': 4}


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def data(path, dimension=16, classes=2, compact=False, instances=None):
    root = create_demo(path, patients=30 if classes == 3 else 20, dimension=dimension, classes=classes)
    if compact:
        slides, index, split = [pd.read_csv(root/name) for name in ['slides.csv', 'features.csv', 'split.csv']]
        joined = slides.merge(split, on='patient_id')
        chosen = joined.groupby(['subset', 'label'], sort=False).head(1)
        slides[slides.slide_id.isin(chosen.slide_id)].to_csv(root/'slides.csv', index=False)
        index[index.slide_id.isin(chosen.slide_id)].to_csv(root/'features.csv', index=False)
        split[split.patient_id.isin(chosen.patient_id)].to_csv(root/'split.csv', index=False)
    if instances is not None:
        rng = np.random.default_rng(44)
        for row in pd.read_csv(root/'features.csv').itertuples():
            with h5py.File(root/row.path, 'r+') as f:
                del f['features'];del f['coords']
                f['features'] = rng.normal(size=(instances, dimension)).astype(np.float32)
                f['coords'] = np.column_stack([np.arange(instances)*256, np.zeros(instances)]).astype(np.int64)
        slides = pd.read_csv(root/'slides.csv');slides['width_level0'] = instances*256
        slides.to_csv(root/'slides.csv', index=False)
    return Bundle(root)


def equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def text(path):
    root, sha = verified_source('pmil_shapley')
    return subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)


def definitions(path, names, namespace, replacements=()):
    code = text(path)
    for old, new in replacements:
        assert old in code
        code = code.replace(old, new)
    tree = ast.parse(code)
    chosen = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    ns = dict(namespace)
    exec(compile(ast.Module(body=chosen, type_ignores=[]), path, 'exec'), ns)  # noqa: S102 -- independent verified Git source
    return ns


def namespace():
    return {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'random': random, 'os': os, 'tqdm': tqdm,
        'Dataset': Dataset, 'DataLoader': DataLoader, 'accuracy_score': accuracy_score, 'f1_score': f1_score,
        'roc_curve': roc_curve, 'auc': auc, 'confusion_matrix': confusion_matrix, 'label_binarize': label_binarize}


def original_models(dim=1024, hidden=512, attention=128):
    # Tensor-width adaptation is explicit; native dimensions execute the original class bodies.
    code = text('models/MIL_models.py')
    if dim != 1024 or hidden != 512 or attention != 128:
        code = code.replace('nn.Linear(1024, 512)', f'nn.Linear({dim}, {hidden})')
        code = code.replace('self.L = 512', f'self.L = {hidden}').replace('self.D = 128', f'self.D = {attention}')
        code = code.replace('[1024, 512, 256]', f'[{dim}, {hidden}, {attention}]').replace('[1024, 512, 384]', f'[{dim}, {hidden}, {attention}]')
    ns = namespace()
    exec(compile(code, 'original_models.py', 'exec'), ns)  # noqa: S102 -- original Git blob, only declared test widths
    return ns


def original_functions(ns, repaired=False):
    ns = definitions('utils.py', ['set_seed', 'EarlyStopping', 'calculate_metrics', 'shapley_value'], ns,
                     [('np.Inf', 'np.inf')])
    replacements = []
    if repaired:
        replacements += [('\n                    bag_feat = feat[:, idx, :]', '\n                    optimizer.zero_grad()\n                    bag_feat = feat[:, idx, :]')]
    return definitions('main.py', ['MIL_train_epoch', 'MIL_pred', 'split_slide_data'], ns, replacements)


def dataset(b, subset):
    return [{'slide_id': r['slide_id'], 'feat': b.load(r)[0], 'target': r['target']} for r in b.records(subset)]


@pytest.mark.parametrize('head', ['abmil', 'clam_sb', 'clam_mb'])
@pytest.mark.parametrize('classes', [2, 3])
def test_native_network_initialization_outputs_losses_gradients(tmp_path, head, classes):
    b = data(tmp_path/'data', 1024, classes, compact=True)
    seed_everything(7)
    m = PMILShapley(b, f'pmil_shapley_{head}_source')
    actual_rng = rng_state()
    seed_everything(7)
    ns = original_models()
    args = {'n_classes': classes}
    if head != 'abmil':
        args.update(k_sample=8, instance_loss_fn=nn.CrossEntropyLoss())
    ref = ns[{'abmil': 'ABMIL', 'clam_sb': 'CLAM_SB', 'clam_mb': 'CLAM_MB'}[head]](**args)
    equal(actual_rng, rng_state()); equal(m.network.state_dict(), ref.state_dict())
    x, label = torch.randn(32, 1024), torch.tensor([classes-1])
    state = rng_state()
    got = m(x, label=label)
    after = rng_state(); restore_rng(state)
    result = ref(x[None]) if head == 'abmil' else ref(x[None], label, instance_eval=True)
    expected_loss = F.cross_entropy(result[0], label)
    if head != 'abmil':
        expected_loss = .7*expected_loss+.3*result[2]['instance_loss']
    equal(after, rng_state()); equal(got['logits'], result[0]); equal(got['total_loss'], expected_loss)
    got['total_loss'].backward(); expected_loss.backward()
    for p, q in zip(m.network.parameters(), ref.parameters()):
        equal(p.grad, q.grad)


@pytest.mark.parametrize('head', ['abmil', 'clam_sb', 'clam_mb'])
def test_shapley_value_and_random_stream_match_original(tmp_path, head):
    b = data(tmp_path/'data', classes=3)
    m = PMILShapley(b, f'pmil_shapley_{head}_source', **SMALL)
    ns = original_functions(namespace())
    x = torch.randn(1, 36, 16)
    selected, label = np.array([1, 5, 9, 13, 17, 21]), torch.tensor([2])
    state = rng_state()
    expected = ns['shapley_value'](selected, x, label, m.network, 'cpu', 'ABMIL' if head == 'abmil' else 'CLAM', subset_num=3)
    after = rng_state(); restore_rng(state)
    got = shapley_scores(m.network, head, selected, x, 2, subsets=3)
    equal(expected, got); equal(after, rng_state())


@pytest.mark.parametrize('head', ['abmil', 'clam_sb', 'clam_mb'])
@pytest.mark.parametrize('previous', [False, True])
def test_complete_partition_matches_native_source_and_rng(tmp_path, head, previous):
    b = data(tmp_path/'data', compact=True)
    m = PMILShapley(b, f'pmil_shapley_{head}_source', **{**SMALL, 'max_pseudo': 3})
    m.train_lengths = {r['slide_id']: len(b.load(r)[0]) for r in b.records('train')}
    if previous:
        m.partitions = {k: [torch.arange(n)[i::2] for i in range(2)] for k, n in m.train_lengths.items()}
    previous_ref = None if m.partitions is None else {k: [p.numpy().copy() for p in v] for k, v in m.partitions.items()}
    m.split_index = 2
    snapshot = copy.deepcopy(m.network.state_dict())
    target = tmp_path/'selector.pt'; torch.save(snapshot, target)
    ns = original_functions(namespace())
    state = rng_state()
    loader = DataLoader(dataset(b, 'train'), batch_size=1, shuffle=True, num_workers=0)
    expected = ns['split_slide_data'](3, m.network, str(target), loader, 'cpu',
        'ABMIL' if head == 'abmil' else 'CLAM', previous_ref, 'shap', 2, 2)
    after = rng_state(); restore_rng(state)
    m.divide(snapshot)
    equal(after, rng_state())
    for slide, parts in expected.items():
        for p, q in zip(parts, m.partitions[slide]):
            np.testing.assert_array_equal(p, q.numpy())
    assert m.events[-1]['partition_sha256'] == tensor_tree_sha(m.partitions)


def capture_steps(optimizer, parameters):
    original = optimizer.step
    records = []
    def step(*args, **kwargs):
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        result = original(*args, **kwargs)
        records.append({'grads': gradients, 'weights': [p.detach().clone() for p in parameters],
                        'optimizer': copy.deepcopy(optimizer.state_dict())})
        return result
    optimizer.step = step
    return records


@pytest.mark.parametrize('head', ['abmil', 'clam_sb', 'clam_mb'])
@pytest.mark.parametrize('mode', ['source', 'repaired'])
def test_complete_author_training_loops_and_all_adam_updates(tmp_path, head, mode):
    b = data(tmp_path/'data', compact=True)
    m = PMILShapley(b, f'pmil_shapley_{head}_{mode}', **SMALL)
    m.configure_training({'epochs': 6, 'lr': .001, 'weight_decay': 1e-5, 'selection_metric': 'pmil_selection', 'grad_clip': None, 'max_instances': None})
    m.prepare(b, 4, 'cpu')
    m.partitions = {r['slide_id']: [torch.arange(len(b.load(r)[0]))[i::2] for i in range(2)] for r in b.records('train')}
    opt = m.make_optimizer(.001, 1e-5)
    ref = copy.deepcopy(m.network); refopt = torch.optim.Adam(ref.parameters(), lr=.001, weight_decay=1e-5)
    actual_steps = capture_steps(opt, list(m.network.parameters()))
    expected_steps = capture_steps(refopt, list(ref.parameters()))
    ns = original_functions(namespace(), repaired=mode == 'repaired')
    for epoch in range(2):
        m.train(); ref.train()
        state = rng_state()
        pmil_epoch(m, b, units(b, 'train'), opt, 'cpu', {})
        after = rng_state(); restore_rng(state)
        loader = DataLoader(dataset(b, 'train'), batch_size=1, shuffle=True, num_workers=0)
        ns['MIL_train_epoch'](0, epoch, ref, refopt, loader, nn.CrossEntropyLoss(), 'cpu', 2,
            'ABMIL' if head == 'abmil' else 'CLAM', {k: [p.numpy() for p in v] for k, v in m.partitions.items()})
        equal(after, rng_state())
        equal(actual_steps, expected_steps)
        equal(m.network.state_dict(), ref.state_dict()); equal(opt.state_dict(), refopt.state_dict())
    assert len(actual_steps) == 4*len(units(b, 'train'))
    write_json(tmp_path/'pmil_author_training.json', {'state': 'passed', 'profile': m.profile, 'epochs': 2,
        'optimizer_updates': len(actual_steps), 'scope': 'Complete fixed source or explicitly gradient-reset source loop, all step gradients/weights/Adam moments and RNG bitwise equal; synthetic CPU'})


@pytest.mark.parametrize('profile', list(PROFILES))
@pytest.mark.parametrize('boundary', [2, 3])
def test_all_routes_exact_resume_round_boundaries_best_reload(tmp_path, profile, boundary):
    b = data(tmp_path/'data', compact=True)
    cfg = {'threads': 1, 'model': SMALL}
    full = train(b.root, profile, tmp_path/'full', seed=8, config=cfg)
    part = train(b.root, profile, tmp_path/'part', seed=8, config=cfg, stop_after=boundary)
    assert read_json(part/'status.json')['state'] == 'interrupted'
    train(b.root, profile, part, seed=8, config=cfg, resume=True)
    a = torch.load(full/'last.pt', weights_only=False)
    c = torch.load(part/'last.pt', weights_only=False)
    equal(a['model'], c['model']); equal(a['optimizers'], c['optimizers']); equal(a['rng'], c['rng']); equal(a['history'], c['history'])
    extra = a['model']['_extra_state']
    assert extra['training_done'] and len(extra['completed_rounds']) == 2 and extra['reset_count'] == 2
    assert extra['events'][-1]['estimator'] == 'shap' and extra['events'][-1]['round'] == 1
    best = torch.load(full/'best.pt', weights_only=False)
    assert best['model']['_extra_state']['round_id'] == 1
    final_round = torch.load(full/'rounds/round_1.pt', weights_only=False)
    equal({k.removeprefix('network.'): v for k, v in best['model'].items() if k.startswith('network.')}, final_round['network'])
    result = evaluate(full); digest = digest_file(full/'test_predictions.csv')
    equal(result, evaluate(full)); assert digest == digest_file(full/'test_predictions.csv')
    equal(result, evaluate(part)); assert digest == digest_file(part/'test_predictions.csv')
    assert not read_json(full/'status.json')['test_evaluated']


@pytest.mark.parametrize('classes', [2, 3])
def test_source_micro_auc_and_selection_metrics(tmp_path, classes):
    logits = torch.tensor([[3., 1., 0.], [1., 3., 4.], [0., 2., 1.], [1., 3., 0.], [3., 2., 1.], [1., 0., 3.]])[:, :classes]
    target = torch.arange(6) % classes
    ns = original_functions(namespace())
    acc, f1, score = ns['calculate_metrics'](logits, target, classes)
    got = source_metrics(logits, target, classes)
    assert got == {'acc': acc, 'f1': f1, 'auc': score}


def test_progressive_plateau_early_stop_and_pending_resume(tmp_path):
    b = data(tmp_path/'data', compact=True)
    opts = {**SMALL, 'round_epochs': 10, 'max_pseudo': 3, 'search_rate': 1, 'lr_patience': 1,
            'patience': 2, 'warmup_epoch': 0}
    m = PMILShapley(b, **opts)
    m.configure_training({'epochs': 20, 'lr': .001, 'weight_decay': 1e-5, 'selection_metric': 'pmil_selection', 'grad_clip': None, 'max_instances': None})
    m.prepare(b, 6, 'cpu'); opt = m.make_optimizer(.001, 1e-5)
    for epoch in range(6):
        m.start_epoch(epoch)
        pmil_epoch(m, b, units(b, 'train'), opt, 'cpu', {})
        m.after_validation({'pmil_loss': 1.+epoch, 'pmil_selection': 1./(1+epoch)})
        if epoch in [2, 3]:
            clone = PMILShapley(b, **opts)
            clone.load_state_dict(m.state_dict())
            assert clone.pending == 'divide'
            assert clone.partitions is None or clone.events[-1]['pseudo_count'] < clone.schedule[clone.split_index]
    assert m.pending == 'round' and m.current_lr == 1e-4
    # Native code also redivides at the maximum pseudo count on lr_patience stalls.
    assert [e['pseudo_count'] for e in m.events] == [2, 3, 3]
    assert m.round_epoch == 5 and m.completed_rounds[0]['epochs'] == 6


def test_constant_shapley_is_explicit_and_repaired(tmp_path):
    b = data(tmp_path/'data')
    m = PMILShapley(b, **SMALL)
    for p in m.network.parameters():
        nn.init.zeros_(p)
    x = torch.ones(1, 20, 16)
    with pytest.raises(ValueError, match='zero score range'):
        shapley_scores(m.network, m.head, [0, 1, 2], x, 0)
    equal(shapley_scores(m.network, m.head, [0, 1, 2], x, 0, repaired=True), torch.zeros(3))


@pytest.mark.parametrize('change', ['duplicate', 'snapshot', 'profile', 'schema', 'stage'])
def test_corrupt_checkpoint_is_rejected(tmp_path, change):
    b = data(tmp_path/'data', compact=True)
    m = PMILShapley(b, **SMALL)
    m.configure_training({'epochs': 6, 'lr': .001, 'weight_decay': 1e-5, 'selection_metric': 'pmil_selection', 'grad_clip': None, 'max_instances': None})
    m.prepare(b, 1, 'cpu');m.make_optimizer(.001, 1e-5)
    m.start_epoch(0);m.after_validation({'pmil_loss': 1., 'pmil_selection': .5})
    m.split_index = 1;m.divide(m.best_network)
    state = m.state_dict(); extra = state['_extra_state']
    if change == 'duplicate':
        first = next(iter(extra['partitions'].values())); first[0][0] = first[1][0]
    elif change == 'snapshot':
        next(iter(extra['best_network'].values())).flatten()[0] += 1
    elif change == 'profile':
        extra['profile'] = 'pmil_shapley_clam_sb_repaired'
    elif change == 'schema':
        extra['unexpected'] = True
    else:
        extra['split_index'] = 999
    with pytest.raises(ValueError):
        PMILShapley(b, **SMALL).load_state_dict(state)


@pytest.mark.parametrize('options', [{'rounds': 1}, {'max_pseudo': 1}, {'sample_rate': 0}, {'search_rate': True},
    {'round_epochs': 0}, {'min_lr': float('nan')}, {'metric': 'test_auc'}, {'hidden': 0}, {'warmup_epoch': -1}])
def test_invalid_method_contracts(tmp_path, options):
    with pytest.raises(ValueError):
        PMILShapley(data(tmp_path/'data'), **{**SMALL, **options})


def test_full_bag_budget_and_single_slide_requirements(tmp_path):
    b = data(tmp_path/'data', compact=True)
    m = PMILShapley(b, **{**SMALL, 'max_pseudo': 100})
    with pytest.raises(ValueError, match='too small'):
        m.prepare(b, 2, 'cpu')
    m = PMILShapley(b, **{**SMALL, 'max_partition_instances': 1})
    with pytest.raises(ValueError, match='partition budget'):
        m.prepare(b, 2, 'cpu')
    m = PMILShapley(b, **SMALL)
    for cfg in [{'epochs': 1}, {'max_instances': 10}, {'grad_clip': 1}, {'selection_metric': 'test_auc'}, {'patience': 1}]:
        with pytest.raises(ValueError):
            m.configure_training({'epochs': 6, 'lr': .001, 'weight_decay': 1e-5, 'max_instances': None,
                'grad_clip': None, 'selection_metric': 'pmil_selection', **cfg})
    b.config['training_unit'] = 'patient'
    with pytest.raises(ValueError, match='individual slide'):
        PMILShapley(b, **SMALL)


def test_alias_and_labels_never_enter_inference_logits(tmp_path):
    assert require_runnable('PMIL Shapley pseudo-bag')['id'] == 'pmil_shapley_abmil_repaired'
    b = data(tmp_path/'data', compact=True)
    m = PMILShapley(b, 'pmil_shapley_clam_sb_repaired', **SMALL).eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('No inference instance-label evaluation')
    m.network.inst_eval = forbidden
    x = torch.randn(12, 16)
    equal(m(x)['logits'], m(x)['logits'])
    first = m.predict_subset(b, 'test', 'cpu', {})
    mask = b.rows.subset == 'test'
    b.rows.loc[mask, 'target'] = 1-b.rows.loc[mask, 'target']
    second = m.predict_subset(b, 'test', 'cpu', {})
    np.testing.assert_array_equal(first[['p_0', 'p_1']], second[['p_0', 'p_1']])
    assert not np.array_equal(first.target.to_numpy(), second.target.to_numpy())


def test_named_gradient_repair_changes_the_second_pseudo_update(tmp_path):
    b = data(tmp_path/'data', compact=True)
    models, histories = [], []
    for mode in ['source', 'repaired']:
        seed_everything(9)
        m = PMILShapley(b, 'pmil_shapley_abmil_'+mode, **SMALL)
        m.configure_training({'epochs': 6, 'lr': .001, 'weight_decay': 1e-5, 'max_instances': None, 'grad_clip': None, 'selection_metric': 'pmil_selection'})
        m.prepare(b, 9, 'cpu');opt = m.make_optimizer(.001, 1e-5)
        m.partitions = {r['slide_id']: [torch.arange(len(b.load(r)[0]))[i::2] for i in range(2)] for r in b.records('train')}
        histories.append(capture_steps(opt, list(m.parameters())))
        pmil_epoch(m, b, units(b, 'train'), opt, 'cpu', {})
        models.append(m)
    equal(histories[0][0], histories[1][0])
    assert any(not torch.equal(a, b) for a, b in zip(histories[0][1]['weights'], histories[1][1]['weights']))


def test_repaired_multibranch_uses_the_target_branch(tmp_path, monkeypatch):
    import milplus.pmil_shapley as module
    b = data(tmp_path/'data', classes=3, compact=True)
    row = next(r for r in b.records('train') if r['target'] == 2)
    m = PMILShapley(b, 'pmil_shapley_clam_mb_repaired', **SMALL)
    m.train_lengths = {row['slide_id']: len(b.load(row)[0])};m.split_index = 1
    monkeypatch.setattr('milplus.engine.units', lambda bundle, subset: [[row]])
    seen = []
    def scores(network, head, selected, *args, **kwargs):
        seen.append(selected.copy());return torch.arange(len(selected), dtype=torch.float32)
    monkeypatch.setattr(module, 'shapley_scores', scores)
    x = b.load(row)[0]
    expected = np.argsort(-m.network(x[None])[1][2])[:4]
    m.divide(copy.deepcopy(m.network.state_dict()))
    np.testing.assert_array_equal(seen[0], expected)


@pytest.mark.parametrize('head', ['abmil', 'clam_sb', 'clam_mb'])
@pytest.mark.parametrize('mode', ['source', 'repaired'])
@pytest.mark.parametrize('progressive', [False, True])
def test_original_complete_native_main_multi_round_program(tmp_path, head, mode, progressive, monkeypatch):
    """Execute the actual author's if-main body, CPU/test-sealing changes declared."""
    b = data(tmp_path/'data', 1024, compact=True, instances=48 if progressive else None)
    source_features = tmp_path/'native_features';source_features.mkdir()
    splits = []
    for subset in ['train', 'val']:
        rows = b.records(subset)
        splits.append({r['slide_id']: r['target'] for r in rows})
        for row in rows:
            torch.save(b.load(row)[0], source_features/(row['slide_id']+'.pt'))
    name = {'abmil': 'ABMIL', 'clam_sb': 'CLAM_SB', 'clam_mb': 'CLAM_MB'}[head]
    maximum, epochs, patience, lr_patience, warmup = (3, 8, 2, 1, 0) if progressive else (2, 2, 20, 7, 20)
    args = SimpleNamespace(experiment_name='native', MIL_model=name, metric2save='f1_auc', device_ids=0,
        seed=23, fold=0, num_classes=2, split_data=True, search_rate=2, sample_rate=2, max_pseudo_num=maximum,
        pseudo_step=1, metrics='shap', rounds=2, epochs=epochs, patience=patience, lr_patience=lr_patience,
        max_lr=.001, min_lr=.0001, csv_dir=str(tmp_path), feat_dir=str(source_features),
        ckpt_dir=str(tmp_path/'native_checkpoints'), test=False, logger_dir=str(tmp_path/'native_logs'))
    ns = original_functions(original_models(), repaired=mode == 'repaired')
    ns.update({'glob': glob, 'get_args': lambda: args, 'return_splits': lambda **kwargs: splits,
        'SummaryWriter': lambda *args, **kwargs: None, 'draw_metrics': lambda *args, **kwargs: None})
    ns.update(definitions('main.py', ['MILDataset'], ns))
    if progressive:
        stopper = ns['EarlyStopping']
        ns['EarlyStopping'] = lambda **kwargs: stopper(warmup_epoch=warmup, **kwargs)
        original_prediction = ns['MIL_pred']
        calls = {}
        def forced_observation(round_id, *a, **kwargs):
            result = original_prediction(round_id, *a, **kwargs)
            epoch = calls.get(round_id, 0);calls[round_id] = epoch+1
            return (1.+epoch, result[1], 1./(1+epoch), 1./(1+epoch), result[-1])
        ns['MIL_pred'] = forced_observation
    if mode == 'repaired':
        ns.update(definitions('utils.py', ['shapley_value'], ns, [
            ('score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))',
             'span = torch.max(score)-torch.min(score)\n        score = torch.zeros_like(score) if span == 0 else (score-torch.min(score))/span')]))
        ns.update(definitions('main.py', ['split_slide_data'], ns, [
            ('attn[:, idx] = bag_attn[0]', "attn[:, idx] = bag_attn[int(target) if model_suffix == 'CLAM_MB' else 0]"),
            ('score = attn[0]', "score = attn[int(target) if model_suffix == 'CLAM_MB' and prev_split is None else 0]")]))
    code = text('main.py').replace("torch.device('cuda:{}'.format(args.device_ids))", "torch.device('cpu')")
    tree = ast.parse(code)
    block = next(n for n in tree.body if isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))
    exec(compile(ast.Module(body=block.body, type_ignores=[]), 'native_full_main.py', 'exec'), ns)  # noqa: S102 -- actual author program, CPU and sealed TEST only
    original_rng = rng_state()
    native_state = {k: v.detach().clone() for k, v in ns['model'].state_dict().items()}
    native_partition = {k: [torch.tensor(np.asarray(p).copy(), dtype=torch.long) for p in parts] for k, parts in ns['split'].items()}
    profile = f'pmil_shapley_{head}_{mode}'
    if progressive:
        actual_observe = PMILShapley.after_validation
        def observe(self, metrics):
            actual_observe(self, {**metrics, 'pmil_loss': 1.+self.round_epoch, 'pmil_selection': 1./(1+self.round_epoch)})
        monkeypatch.setattr(PMILShapley, 'after_validation', observe)
    run = train(b.root, profile, tmp_path/'run', seed=23, config={'threads': 1, 'model': {
        'rounds': 2, 'round_epochs': epochs, 'max_pseudo': maximum, 'pseudo_step': 1, 'search_rate': 2, 'sample_rate': 2,
        'patience': patience, 'lr_patience': lr_patience, 'warmup_epoch': warmup}})
    last = torch.load(run/'last.pt', weights_only=False)
    best = torch.load(run/'best.pt', weights_only=False)
    adapted = {k.removeprefix('network.'): v for k, v in best['model'].items() if k.startswith('network.')}
    equal(native_state, adapted)
    equal(original_rng, last['rng'])
    equal(native_partition, last['model']['_extra_state']['partitions'])
    if progressive:
        extra = last['model']['_extra_state']
        assert [v['pseudo_count'] for v in extra['events']] == [2, 3, 3, 3, 3]
        assert last['epoch']+1 == 10 < 2*epochs
        assert extra['training_done'] and extra['current_lr'] == .0001
    for index in [0, 1]:
        native = torch.load(Path(args.ckpt_dir)/'native'/f'{name}_model_{index}.pth', weights_only=True)
        exported = torch.load(run/'rounds'/f'round_{index}.pt', weights_only=False)
        equal(native, exported['network'])
    write_json(tmp_path/'pmil_native_main.json', {'state': 'passed', 'profile': profile,
        'rounds': 2, 'epochs_per_round': epochs, 'hidden': 512, 'dimension': 1024,
        'forced_plateau_observations': progressive,
        'updates': last['model']['_extra_state']['optimizer_steps'],
        'scope': 'Original main program vs complete adapter, native dimensions, both round-selected checkpoints/partitions/final RNG bitwise equal. Native TEST disabled and logging omitted; repaired routes name their gradient/ranking/zero-range changes. CPU synthetic.'})


@pytest.mark.parametrize('profile', list(PROFILES))
@pytest.mark.parametrize('boundary', [3, 4, 6, 9])
def test_full_progressive_pending_transitions_resume_exactly(tmp_path, profile, boundary, monkeypatch):
    b = data(tmp_path/'data', compact=True, instances=48)
    options = {**SMALL, 'round_epochs': 8, 'max_pseudo': 3, 'search_rate': 1,
               'lr_patience': 1, 'patience': 2, 'warmup_epoch': 0}
    observe = PMILShapley.after_validation
    def forced(self, metrics):
        observe(self, {**metrics, 'pmil_loss': 1.+self.round_epoch, 'pmil_selection': 1./(1+self.round_epoch)})
    monkeypatch.setattr(PMILShapley, 'after_validation', forced)
    cfg = {'threads': 1, 'model': options}
    full = train(b.root, profile, tmp_path/'full', config=cfg, seed=13)
    partial = train(b.root, profile, tmp_path/'part', config=cfg, seed=13, stop_after=boundary)
    before = torch.load(partial/'last.pt', weights_only=False)
    assert before['model']['_extra_state']['pending'] == ('round' if boundary == 6 else 'divide')
    assert read_json(partial/'status.json')['state'] == 'interrupted'
    train(b.root, profile, partial, config=cfg, seed=13, resume=True)
    a = torch.load(full/'last.pt', weights_only=False)
    c = torch.load(partial/'last.pt', weights_only=False)
    equal(a['model'], c['model']);equal(a['optimizers'], c['optimizers']);equal(a['rng'], c['rng']);equal(a['history'], c['history'])
    assert a['epoch'] == 9 and a['model']['_extra_state']['training_done']
    equal(evaluate(full), evaluate(partial))
    assert digest_file(full/'test_predictions.csv') == digest_file(partial/'test_predictions.csv')

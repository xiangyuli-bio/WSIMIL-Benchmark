"""Original GAT-Mamba mathematical, full trainer and physical-input evidence."""
import ast
import copy
import math
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from einops import rearrange, repeat
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from milplus.data import Bundle
from milplus.engine import evaluate, rng_state, seed_everything, train
from milplus.gatmamba import GATMamba, PatientGraphs, native_c_index, patients
from milplus.gatmamba_data import category_table, prepare
from milplus.gatmamba_demo import create_gatmamba_demo
from milplus.io import digest_file, read_json
from milplus.upstream import pinned_text, verified_source

FAST = {'uni_hidden': 8, 'positional_embedding_size': 4, 'batch_size': 4}


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def data(path, **kwargs):
    return Bundle(create_gatmamba_demo(path, patients=20, **kwargs))


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def independent(source, filename, scope, names=None, replacements=()):
    root, sha = verified_source(source)
    text = pinned_text(root, sha, filename)
    for old, new in replacements:
        assert text.count(old) == 1
        text = text.replace(old, new)
    tree = ast.parse(text)
    if names is not None:
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in tree.body} == set(names)
    else:
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'mamba_ssm')]
    ns = dict(scope)
    exec(compile(tree, 'independent_original_'+filename, 'exec'), ns)  # noqa: S102 -- independent fixed Git blobs
    return SimpleNamespace(**ns)


def independent_factory(dimension, edge=False):
    scope = {'torch': torch, 'nn': nn, 'F': F, 'Tensor': Tensor, 'math': math,
             'Optional': Optional, 'rearrange': rearrange, 'repeat': repeat}
    scan = independent('gatmamba_ssm', 'mamba_ssm/ops/selective_scan_interface.py', scope, ['selective_scan_ref'])
    scope.update(selective_scan_fn=scan.selective_scan_ref, causal_conv1d_fn=None,
                 causal_conv1d_update=None, selective_state_update=None)
    mamba = independent('gatmamba_ssm', 'mamba_ssm/modules/mamba_simple.py', scope, ['Mamba'])
    patches = [('self.num_uni_features = 1024', f'self.num_uni_features = {dimension}'),
               ('feature_start_index+1024', f'feature_start_index+{dimension}')]
    if edge:
        patches.append(('heads = self.num_heads_gnn, dropout = gnn_dropout)',
                        'heads = self.num_heads_gnn, dropout = gnn_dropout, edge_dim=16)'))
    source = independent('gatmamba', 'modeling/model.py', {'Mamba': mamba.Mamba}, replacements=patches)
    return source.GATMamba


def original_ci():
    return independent('lifelines_metric', 'lifelines/utils/concordance.py', {'np': np},
        ['naive_concordance_index', '_preprocess_scoring_data', '_naive_concordance_summary_statistics',
         '_concordance_ratio']).naive_concordance_index


def test_actual_original_graph_builder_all_arrays_and_patient_union(tmp_path):
    bundle = data(tmp_path/'data', dimension=1024)
    functions = independent('gatmamba', 'modeling/create_dataset_uni.py',
        {'torch': torch, 'np': np, 'pd': pd, 'Data': Data, 'to_undirected': to_undirected,
         'kneighbors_graph': kneighbors_graph, 'euclidean_distances': euclidean_distances},
        ['build_graph_as_data_object', 'get_edge_index', 'subtype_subtype_edge'])
    groups = patients(bundle, 'train')
    rows = next(g for g in groups if len(g) == 2)
    sizes = []
    for row in rows:
        x, coords = bundle.load(row, True)
        order = np.lexsort((coords[:, 0].numpy(), coords[:, 1].numpy()))
        points = coords[order].numpy()
        category = bundle.root/'categories'/(row['slide_id']+'.h5')
        with h5py.File(category) as stream:
            types = stream['node_types'][:][order]
        frame = pd.DataFrame(x[order].numpy().astype(np.float64))
        frame.insert(0, 'sid', row['slide_id'])
        frame.insert(1, 'tile_name', [f'demo-tile-r{int(y/512)}-c{int(v/512)}-x{int(v)}-y{int(y)}-w512-h512'
                                      for v, y in points])
        frame.insert(2, 'days', row['time']);frame.insert(3, 'event', row['event'])
        frame.insert(4, 'predicted_class', types)
        expected = functions.build_graph_as_data_object(frame, 8, row['patient_id'])
        actual = bundle.gatmamba.load(row)
        exact(actual.x, expected.x);exact(actual.edge_index, expected.edge_index)
        torch.testing.assert_close(actual.edge_attr, expected.edge_attr, rtol=1e-6, atol=1e-6)
        sizes.append(len(actual.x))
    graph = bundle.gatmamba.patient(rows)
    edges = graph.edge_index
    assert not ((edges[0] < sizes[0]) != (edges[1] < sizes[0])).any()
    assert len(graph.x) == sum(sizes) and graph.source_slides == ';'.join(r['slide_id'] for r in sorted(rows, key=lambda r:r['slide_id']))
    table = category_table()
    assert np.array_equal(table, table.T) and set(table.flatten()) == set(range(21))


@pytest.mark.parametrize('dimension', [1024, 1536])
def test_native_whole_source_network_cox_l1_all_gradients(tmp_path, dimension):
    bundle = data(tmp_path/'data', dimension=dimension)
    seed_everything(61);actual = GATMamba(bundle)
    seed_everything(61);expected = independent_factory(dimension)()
    exact(actual.network.state_dict(), expected.state_dict())
    actual.eval();expected.eval()
    graphs = [bundle.gatmamba.patient(g) for g in patients(bundle, 'train')[:2]]
    a = Batch.from_data_list(graphs);b = a.clone()
    a.x.requires_grad_();b.x.requires_grad_()
    y, embedding = actual(a);z, reference_embedding = expected(b)
    exact(y, z);exact(embedding, reference_embedding)
    original = independent('gatmamba', 'modeling/main_gat_mamba.py', {'torch': torch, 'np': np}, ['TrainModel']).TrainModel
    loss = actual.objective(a, y.squeeze(1))
    reference = original.CoxLoss(None, b.days, b.event, z.squeeze(1), 'cpu')+1e-5*original.regularize_weights(None, expected)
    exact(loss, reference);loss.backward();reference.backward();exact(a.x.grad, b.x.grad)
    assert len(list(actual.network.parameters())) > 20
    for (name,p),(ref_name,q) in zip(actual.network.named_parameters(), expected.named_parameters()):
        assert name == ref_name and (p.grad is None) == (q.grad is None)
        if p.grad is not None:exact(p.grad, q.grad)


@pytest.mark.parametrize('profile', ['gatmamba_source', 'gatmamba_edge_conditioned'])
def test_real_edge_channels_source_inactive_and_named_correction(tmp_path, profile):
    bundle = data(tmp_path/'data');seed_everything(92)
    actual = GATMamba(bundle, profile, **FAST).eval()
    graph = Batch.from_data_list([bundle.gatmamba.patient(g) for g in patients(bundle,'train')[:3]])
    risk, _ = actual(graph)
    cox = actual.author.CoxLoss(None, graph.days, graph.event, risk.squeeze(1), 'cpu')
    parameters = [actual.network.edge_embedding.weight, actual.network.edge_linear_transform.weight]
    gradients = torch.autograd.grad(cox, parameters, allow_unused=True, retain_graph=True)
    changed = graph.clone();changed.edge_attr[:, 0] = (changed.edge_attr[:, 0]+9)%21
    changed.edge_attr[:, 1:] += 1000
    second, _ = actual(changed)
    if profile.endswith('source'):
        assert all(g is None for g in gradients);exact(risk, second)
    else:
        assert all(g is not None and g.abs().max() > 0 for g in gradients)
        assert not torch.equal(risk, second)


def test_complete_actual_original_trainer_validation_scheduler_rng_and_selected_weights(tmp_path):
    bundle = data(tmp_path/'data')
    seed_everything(10);actual = GATMamba(bundle, **FAST)
    actual.configure_training({'epochs': 2, 'patience': 5, 'max_instances': None,
                              'grad_clip': None, 'selection_metric': 'gatmamba_val_loss'})
    optimizer = actual.make_optimizer(5e-5, 1e-4)
    actual_states = []
    for _ in range(2):
        actual.epoch(bundle, optimizer);actual.finish_epoch(bundle, 'cpu', {})
        frame = actual.predict_subset(bundle, 'val', 'cpu', {})
        metrics = actual.validation_metrics(frame);actual.after_validation(metrics)
        actual_states.append({'model': copy.deepcopy(actual.network.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()), 'scheduler': copy.deepcopy(actual.scheduler.state_dict()),
            'rng': rng_state(), 'validation_loss': metrics['gatmamba_val_loss'],
            'validation_risk': frame.risk.to_numpy(), 'validation_time': frame.native_time.to_numpy(),
            'validation_ci': metrics['gatmamba_native_c_index']})
    references, captured_models, captured_optimizers = [], [], []
    factory = independent_factory(bundle.dim)

    def build(*args):
        model = factory(*args);captured_models.append(model);return model

    class Adam(torch.optim.Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs);captured_optimizers.append(self)

    class Plateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def step(self, metric, epoch=None):
            super().step(metric, epoch)
            references.append({'model': copy.deepcopy(captured_models[0].state_dict()),
                'optimizer': copy.deepcopy(captured_optimizers[0].state_dict()),
                'scheduler': copy.deepcopy(self.state_dict()), 'rng': rng_state(), 'validation_loss': metric})

    class PlotRecorder:
        def __getattr__(self, name):
            # Exclude only visualization side effects; all native training,
            # validation, checkpoint writes and scheduler code actually execute.
            return lambda *a, **k:None

    (tmp_path/'models').mkdir()
    scope = {'torch': torch, 'np': np, 'os': os, 'DataLoader': DataLoader, 'GATMamba': build,
        'optim': SimpleNamespace(Adam=Adam, lr_scheduler=SimpleNamespace(ReduceLROnPlateau=Plateau)),
        'MODEL_DIR': str(tmp_path/'models'), 'FIGURE_DIR': str(tmp_path/'figures'),
        'plt': PlotRecorder(), 'concordance_index': original_ci()}
    original = independent('gatmamba', 'modeling/main_gat_mamba.py', scope, ['TrainModel'],
        replacements=[('valid_loss_min = np.Inf','valid_loss_min = np.inf')]).TrainModel
    seed_everything(10)
    runner = original(1, 1, 4, 5e-5, 1e-4, 1e-5, 5, 8, 4, .1, .3, 1, 1, 0)
    runner.start_training(PatientGraphs(bundle.gatmamba, patients(bundle, 'train')),
                          PatientGraphs(bundle.gatmamba, patients(bundle, 'val')))
    assert len(references) == len(actual_states) == 2
    for expected, reference in zip(actual_states, references):
        for key in ['model','optimizer','scheduler','rng','validation_loss']:
            exact(expected[key], reference[key])
    selected = torch.load(tmp_path/'models/model_fold0.pth', weights_only=False)
    losses = [s['validation_loss'] for s in actual_states]
    index = max(i for i,value in enumerate(losses) if value == min(losses))
    exact(selected, actual_states[index]['model'])


@pytest.mark.parametrize('profile', ['gatmamba_source', 'gatmamba_edge_conditioned'])
def test_full_engine_exact_resume_selection_and_repeated_predictions(tmp_path, profile):
    bundle = data(tmp_path/'data')
    config = {'epochs':3,'threads':1,'model':FAST}
    complete = train(bundle.root, profile, tmp_path/'complete', config=config)
    stopped = train(bundle.root, profile, tmp_path/'resumed', config=config, stop_after=1)
    train(bundle.root, profile, stopped, config=config, resume=True)
    a = torch.load(complete/'last.pt', weights_only=False)
    b = torch.load(stopped/'last.pt', weights_only=False)
    for key in ['model','optimizers','rng','history','epoch','best','bad']:exact(a[key], b[key])
    values = [row['validation']['gatmamba_val_loss'] for row in a['history']]
    best = torch.load(complete/'best.pt', weights_only=False)
    assert best['epoch'] == max(i for i,value in enumerate(values) if value == min(values))
    assert not (complete/'test_predictions.csv').exists()
    metrics = evaluate(complete);sha = digest_file(complete/'test_predictions.csv')
    assert metrics == evaluate(complete) and sha == digest_file(complete/'test_predictions.csv')
    assert metrics == evaluate(stopped)
    assert digest_file(complete/'test_predictions.csv') == digest_file(stopped/'test_predictions.csv')


def test_original_last_tie_early_stop_and_inclusive_default_budget(tmp_path):
    bundle = data(tmp_path/'data');model = GATMamba(bundle, **FAST)
    model.make_optimizer(5e-5, 1e-4)
    assert model.maximum_epochs == 201 and model.patience == 5
    for i in range(2):
        model.completed_epochs += 1;model.after_validation({'gatmamba_val_loss':1.})
        assert model.early_wait == 0 and not model.training_done
    assert model.selection_improved(-1.,-1.)
    for _ in range(5):
        model.completed_epochs += 1;model.after_validation({'gatmamba_val_loss':2.})
        assert not model.training_done
    model.completed_epochs += 1;model.after_validation({'gatmamba_val_loss':2.})
    assert model.training_done and model.early_wait == 6


@pytest.mark.parametrize('asset', ['feature','coords','graph'])
def test_reject_changed_physical_and_feature_assets(tmp_path, asset):
    bundle = data(tmp_path/'data');row = bundle.records('train')[0]
    if asset in {'feature','coords'}:
        path = bundle.root/row['path']
        with h5py.File(path,'a') as stream:
            key = 'features' if asset == 'feature' else 'coords'
            stream[key][0,0] += 1
    else:
        entry = bundle.gatmamba.meta['slides'][row['slide_id']]
        with h5py.File(bundle.gatmamba.path.parent/entry['path'],'a') as stream:
            stream['edge_attr'][0,1] += 2
    with pytest.raises(ValueError, match='changed'):
        bundle.gatmamba.load(row)


def test_missing_provenance_wrong_FM_and_coordinate_order_rejected(tmp_path):
    bundle = data(tmp_path/'data')
    schema = copy.deepcopy(bundle.gatmamba.meta['schema'])
    schema['classifier']['models'][0]['weights_sha256'] = 'placeholder'
    with pytest.raises(ValueError,match='SHA256'):
        prepare(bundle,bundle.root/'categories.csv',schema,tmp_path/'bad')
    task = yaml.safe_load((bundle.root/'task.yaml').read_text())
    task['encoders']['toy']['weights'] = 'different_fm_same_dimension'
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='FM space'):
        Bundle(bundle.root)
    task['encoders']['toy']['weights'] = 'synthetic'
    task['encoders']['toy']['spatial']['coordinate_order'] = 'yx'
    del task['encoders']['toy']['gatmamba']
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='xy coordinates'):
        prepare(Bundle(bundle.root),bundle.root/'categories.csv',bundle.gatmamba.meta['schema'],tmp_path/'wrong_order')


@pytest.mark.parametrize('mode', ['test','train_with_test_patient','train_duplicate'])
def test_classifier_cannot_fit_on_validation_or_test_patients(tmp_path, mode):
    bundle = data(tmp_path/'data');schema = copy.deepcopy(bundle.gatmamba.meta['schema'])
    classifier = schema['classifier']
    if mode == 'test':
        classifier['fit_subset'] = 'test'
    else:
        classifier['fit_subset'] = 'train'
        patient = bundle.records('test' if mode == 'train_with_test_patient' else 'train')[0]['global_patient_id']
        classifier['training_patients'] = [patient] if mode == 'train_with_test_patient' else [patient,patient]
    with pytest.raises(ValueError,match='TRAIN|VAL/TEST'):
        prepare(bundle,bundle.root/'categories.csv',schema,tmp_path/'bad_fitted_classifier')


def test_actual_efficient_lifelines_metric_matches_independent_quadratic_oracle():
    oracle = original_ci();rng = np.random.default_rng(51)
    for n in [2,7,30,100]:
        for _ in range(8):
            times = rng.integers(1,10,n);events = rng.integers(0,2,n);risk = rng.integers(-3,4,n)
            try:expected = oracle(times,-risk,events)
            except ZeroDivisionError:expected = None
            assert native_c_index(times,events,risk) == expected


def test_explicit_small_graph_and_singleton_BN_correction(tmp_path):
    bundle = data(tmp_path/'data',nodes=1)
    with pytest.raises(ValueError,match='fewer than10'):
        GATMamba(bundle,**FAST)
    model = GATMamba(bundle,'gatmamba_edge_conditioned',**FAST).train()
    rows = next(g for g in patients(bundle,'train') if len(g)==1)
    graph = Batch.from_data_list([bundle.gatmamba.patient(rows)])
    loss = model.objective(graph, model(graph)[0]);loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize('change', ['counters','contract','scheduler'])
def test_checkpoint_contract_validation(tmp_path, change):
    bundle = data(tmp_path/'data');model = GATMamba(bundle,**FAST)
    model.make_optimizer(5e-5,1e-4);state=copy.deepcopy(model.get_extra_state())
    if change=='counters':state['completed_epochs']=-1
    elif change=='contract':state['contract']='wrong'
    else:state['scheduler']['patience']=100
    with pytest.raises(ValueError):model.set_extra_state(state)


def test_true_cli_demo_check_train_and_evaluate(tmp_path):
    root = tmp_path/'cli_data';run = tmp_path/'cli_run'
    config = tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'epochs':2,'threads':1,'model':FAST}))
    base = [sys.executable,'-m','milplus']
    for args in [['demo','--gatmamba','--patients','20','--output',str(root)],
                 ['check','--data',str(root),'--method','GAT-Mamba','--deep'],
                 ['run','--data',str(root),'--method','GAT-Mamba','--config',str(config),'--output',str(run)],
                 ['evaluate','--run',str(run)]]:
        result = subprocess.run(base+args,capture_output=True,text=True,check=False)
        assert result.returncode==0,result.stdout+result.stderr
    assert read_json(run/'status.json')['state']=='completed'


def test_native_cuda_dependency_absence_is_not_a_fallback(tmp_path, monkeypatch):
    bundle = data(tmp_path/'data')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no fallback'):
        GATMamba(bundle,backend='cuda',**FAST)

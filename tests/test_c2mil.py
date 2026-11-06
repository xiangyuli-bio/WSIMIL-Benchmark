"""Independent original-source C²MIL mechanisms, recipe and real-input contracts."""
import ast
import contextlib
import copy
import io
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from milplus.c2mil import C2MIL, native_c_index, source_batch_boundaries
from milplus.c2mil_data import prepare, spatial_edges
from milplus.c2mil_demo import create_c2mil_demo
from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json, write_json
from milplus.upstream import pinned_text, verified_source

FAST = {'hidden': 8, 'batch_size': 8}


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def fixture(path, **kwargs):
    return Bundle(create_c2mil_demo(path, patients=30, **kwargs))


def exact(x, y):
    if isinstance(x, torch.Tensor):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
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


def independent(filename, scope=None, names=None, replacements=()):
    root, sha = verified_source('c2mil')
    text = pinned_text(root, sha, filename)
    for a, b in replacements:
        assert a in text
        text = text.replace(a, b)
    tree = ast.parse(text)
    if names is not None:
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        assert set(names) == {n.name for n in tree.body}
    else:
        blocked = {'softKmeans', 'graph_transformer_pytorch_v', 'Graphtransformer', 'VGAE',
            'bias_removal', 'rotary_embedding_torch', 'torch_scatter', 'timm.models.layers'}
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module in blocked)
            and not (isinstance(n, ast.Import) and any(a.name == 'matplotlib.pyplot' for a in n.names))]
    ns = {'__name__': 'independent_c2mil_'+filename.replace('.', '_'), **(scope or {})}
    exec(compile(tree, 'independent_original_'+filename, 'exec'), ns)  # noqa: S102 -- independent verified author Git objects
    return SimpleNamespace(**ns)


def original_network():
    soft = independent('softKmeans.py')
    bias = independent('bias_removal.py', {'SoftCluster': soft.SoftCluster,
                                         'SoftCluster_adaptiveK': soft.SoftCluster_adaptiveK})
    vgae = independent('VGAE.py')
    attention = independent('graph_transformer_pytorch_v.py')
    graph = independent('Graphtransformer.py', {'GraphTransformer': attention.GraphTransformer})
    model = independent('model_causal_double.py', {'DeepVGAE': vgae.DeepVGAE,
        'DeepVGAE_mse': vgae.DeepVGAE_mse, 'DeepGraphTransformer_mse': graph.DeepGraphTransformer_mse,
        'BiasRemoval': bias.BiasRemoval})
    return model.GraphCausal2_Surv


def original_losses():
    return independent('utils.py', {'torch': torch, 'np': np, 'F': F, 'device': torch.device('cpu')},
        names=['CoxSurvLoss', 'causal_contrastive_loss', 'adjust_learning_rate'],
        replacements=[('train_R = train_R.cuda()', 'train_R = train_R.to(hazards.device)')])


def test_reference_pool_matches_literal_original_sampler_and_padding(tmp_path):
    b = fixture(tmp_path/'data')
    before = rng_state()
    torch.manual_seed(98)
    expected, thumbs = [], []
    for gid in b.c2mil.order['train']:
        graph, thumbnail, _ = b.c2mil.load(b.c2mil.by_patient[gid])
        selected = graph.x[torch.randperm(len(graph.x))[:1500]]
        if len(selected) < 1500:
            selected = torch.cat([selected, selected.mean(0, keepdim=True).repeat(1500-len(selected), 1)])
        expected.append(selected);thumbs.append(thumbnail[0])
    after = torch.get_rng_state()
    restore_rng(before)
    actual_thumbs, actual = b.c2mil.reference_population()
    exact(actual_thumbs, torch.stack(thumbs));exact(actual, torch.stack(expected))
    exact(after.numpy(), np.load(b.c2mil.path.parent/'sampling_rng_after.npy'))
    assert not set(b.c2mil.order['train']) & set(b.c2mil.order['val']+b.c2mil.order['test'])


def test_physical_graph_matches_original_builder(tmp_path):
    from itertools import chain

    import nmslib
    from torch_geometric.data import Data
    graph = independent('data_preprocess/to_Graph.py', {'np': np, 'torch': torch,
        'nmslib': nmslib, 'geomData': Data, 'chain': chain}, names=['Hnsw', 'pt2graph'])
    source_hnsw = graph.pt2graph.__globals__['Hnsw']
    class SingleThread(source_hnsw):
        def fit(self, x):
            self.index_params = {'M': 16, 'post': 0, 'efConstruction': 400, 'indexThreadQty': 1}
            return super().fit(x)
    graph.pt2graph.__globals__['Hnsw'] = SingleThread
    rng = np.random.default_rng(71)
    coords = rng.uniform(0, 10000, (19, 2))
    features = rng.normal(size=(19, 16)).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        expected = graph.pt2graph({'coords': coords, 'features': features})
    exact(expected.edge_index.numpy(), spatial_edges(torch.from_numpy(coords)))


@pytest.mark.parametrize('dimension', [16, 1024, 1536])
def test_full_native_network_five_losses_and_all_gradients_match(tmp_path, dimension):
    b = fixture(tmp_path/'data', dimension=dimension)
    seed_everything(8)
    adapter = C2MIL(b, hidden=128, batch_size=8)
    thumbs, patches = b.c2mil.reference_population()
    seed_everything(8)
    original = original_network()(thumbs, patches, K=0, num_features=dimension, hidden_dim=128)
    exact(original.state_dict(), adapter.network.state_dict())
    graphs = [b.c2mil.load(b.c2mil.by_patient[gid])[:2] for gid in b.c2mil.order['train'][:2]]
    for g, _ in graphs:g.x.requires_grad_()
    random_state = rng_state()
    actual = [adapter(g, thumb) for g, thumb in graphs]
    actual_loss, actual_terms = adapter.joint_objective(actual, [1.0000000001, 1.0000000002], [1, 0])
    actual_loss.backward()
    actual_rng = rng_state()
    actual_grads = {n: None if p.grad is None else p.grad.clone() for n, p in adapter.network.named_parameters()}
    input_grads = [g.x.grad.clone() for g, _ in graphs]
    for g, _ in graphs:g.x.grad = None
    restore_rng(random_state)
    objective = original_losses()
    with contextlib.redirect_stdout(io.StringIO()):
        expected = [original(g, thumb) for g, thumb in graphs]
        h, hc, hr = [torch.stack([r[i] for r in expected]) for i in [1, 3, 4]]
        terms = [objective.CoxSurvLoss(torch.cat([r[i].squeeze(0) for r in expected]),
            [1.0000000001, 1.0000000002], [1, 0]) for i in [0, 2]]
        terms.extend([objective.causal_contrastive_loss(h, hc, hr),
            sum(r[5]**2 for r in expected)/8,
            F.mse_loss(torch.cat([r[7] for r in expected]), torch.cat([r[6] for r in expected]))])
        expected_loss = terms[0]+terms[1]+.1*terms[2]+.1*terms[3]+.1*terms[4]
        expected_loss.backward()
    exact(actual, expected);exact(actual_loss, expected_loss)
    exact(list(actual_terms.values())[1:], terms)
    exact(actual_rng, rng_state())
    for name, p in original.named_parameters():exact(actual_grads[name], p.grad)
    for i, (g, _) in enumerate(graphs):exact(input_grads[i], g.x.grad)


def test_three_actual_author_training_epochs_adam_schedule_validation_and_rng(tmp_path):
    b = fixture(tmp_path/'data')
    seed_everything(8);adapter = C2MIL(b, **FAST)
    opt = adapter.make_optimizer(1e-4, 5e-4)
    initial_rng = rng_state()
    thumb_all, patch_all = b.c2mil.reference_population()
    seed_everything(8)
    original = original_network()(thumb_all, patch_all, K=0, num_features=b.dim, hidden_dim=8)
    source_opt = torch.optim.Adam(original.parameters(), lr=1e-4, weight_decay=5e-4)
    source_order = copy.deepcopy(b.c2mil.order['train'])
    assets = {gid: b.c2mil.load(row) for gid, row in b.c2mil.by_patient.items()}
    cli = {gid: [row['event'], row['time']] for gid, row in b.c2mil.by_patient.items()}
    graphs = {gid: item[0] for gid, item in assets.items()}
    thumbs = {gid: item[1] for gid, item in assets.items()}
    # Inject only the data path bridge. The entire original epoch, batching,
    # all losses, zero_grad/backward/Adam and validation loops execute unchanged.
    def thumb_load(path, **kwargs):
        return thumbs[str(path).rsplit('/', 1)[-1][:-3]]
    proxy = SimpleNamespace(**{n: getattr(torch, n) for n in dir(torch)})
    proxy.load = thumb_load
    source_loss = original_losses()
    def calculate_ci(times, risks, events):
        ids = list(risks)
        return native_c_index([times[g] for g in ids], [events[g] for g in ids],
                              [float(risks[g][0]) for g in ids])
    trainer = independent('train.py', {'torch': proxy, 'np': np, 'random': random, 'F': F,
        'tqdm': lambda x: x, 'device': torch.device('cpu'), 'calculate_ci': calculate_ci,
        'CoxSurvLoss': source_loss.CoxSurvLoss, 'causal_contrastive_loss': source_loss.causal_contrastive_loss},
        names=['train_one_epoch', 'prediction'])
    args = SimpleNamespace(batch_size=8, fea_type='synthetic', lossver='')
    state = initial_rng
    # Use epochs0/1/2 and independently assert the source epoch40/80 schedule.
    for epoch in range(3):
        restore_rng(state)
        losses = adapter.epoch(b, opt);adapter.finish_epoch(b, 'cpu', {})
        prediction = adapter.predict_subset(b, 'val', 'cpu', {})
        actual_rng = rng_state()
        restore_rng(state)
        source_loss.adjust_learning_rate(source_opt, 1e-4, epoch, 40, .5)
        with contextlib.redirect_stdout(io.StringIO()):
            source_value, _ = trainer.train_one_epoch(original, source_order, graphs, cli, source_opt, args)
            val_loss, val_ci, val_risk = trainer.prediction(original, b.c2mil.order['val'], graphs, cli, args)
        exact(float(np.mean([v['loss'] for v in losses])), source_value)
        exact(original.state_dict(), adapter.network.state_dict())
        exact(opt.state_dict(), source_opt.state_dict())
        exact(adapter.training_order, source_order);exact(actual_rng, rng_state())
        exact(prediction.risk.to_numpy(), np.array([val_risk[g][0] for g in b.c2mil.order['val']]))
        exact(prediction.attrs['c2mil_native_full_cox'], val_loss)
        exact(adapter.validation_metrics(prediction)['c2mil_val_c_index'], val_ci)
        exact(adapter.get_extra_state()['centers'], original.biasremoval.cluster.saved_cluster_centers)
        exact(adapter.get_extra_state()['biases'], original.biasremoval.biases)
        state = actual_rng
    for epoch, expected in [(39, 1e-4), (40, 5e-5), (79, 5e-5), (80, 2.5e-5)]:
        source_loss.adjust_learning_rate(source_opt, 1e-4, epoch, 40, .5)
        exact(source_opt.param_groups[0]['lr'], expected)


@pytest.mark.parametrize('n,expected', [(16, [16]), (31, [31]), (32, [16, 32]), (33, [16, 33]), (47, [16, 47])])
def test_source_final_full_batch_merges_remainder(n, expected):
    assert source_batch_boundaries(n, 16) == expected


def test_actual_engine_resume_selected_cache_and_repeat_prediction(tmp_path):
    b = fixture(tmp_path/'data')
    cfg = {'epochs': 4, 'threads': 1, 'model': FAST}
    uninterrupted, resumed = tmp_path/'all', tmp_path/'resumed'
    train(b.root, 'C2MIL', uninterrupted, seed=8, config=cfg)
    train(b.root, 'C2MIL', resumed, seed=8, config=cfg, stop_after=2)
    assert not (resumed/'best.pt').exists()  # first two epochs cannot select
    train(b.root, 'C2MIL', resumed, seed=8, config=cfg, resume=True)
    a = torch.load(uninterrupted/'last.pt', weights_only=False)
    c = torch.load(resumed/'last.pt', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'best', 'bad', 'history']:exact(a[key], c[key])
    selected = torch.load(uninterrupted/'best.pt', weights_only=False)
    assert selected['epoch'] >= 2
    assert selected['model']['_extra_state']['centers'] is not None
    evaluate(uninterrupted);before = digest_file(uninterrupted/'test_predictions.csv')
    evaluate(uninterrupted);assert before == digest_file(uninterrupted/'test_predictions.csv')
    assert read_json(uninterrupted/'status.json')['state'] == 'completed'


def test_reference_rng_and_thumbnail_or_primary_drift_rejected(tmp_path):
    b = fixture(tmp_path/'data')
    gid = b.c2mil.order['train'][0];row = b.c2mil.by_patient[gid]
    entry = b.c2mil.meta['slides'][row['slide_id']]
    path = b.c2mil.path.parent/entry['path']
    with h5py.File(path, 'r+') as stream:
        x = stream['reference_indices'][:];stream['reference_indices'][:] = x[::-1]
    with pytest.raises(ValueError, match='contents changed'):b.c2mil.load(row)
    meta = read_json(b.c2mil.path);meta['slides'][row['slide_id']]['sha256'] = digest_file(path)
    write_json(b.c2mil.path, meta)
    with pytest.raises(ValueError, match='RNG stream'):Bundle(b.root)


def test_provenance_wrong_fm_patch_mean_and_patient_leakage_rejected(tmp_path):
    b = fixture(tmp_path/'data')
    schema = read_json(b.root/'c2mil_schema.json')
    schema['thumbnail']['kind'] = 'mean_patch_features'
    with pytest.raises(ValueError, match='actual thumbnail'):
        prepare(b, b.root/'thumbnails.csv', schema, tmp_path/'bad')
    schema = read_json(b.root/'c2mil_schema.json')
    schema['patient_order']['train'][0] = schema['patient_order']['test'][0]
    with pytest.raises(ValueError, match='leakage'):
        prepare(b, b.root/'thumbnails.csv', schema, tmp_path/'leaked')
    task = yaml.safe_load((b.root/'task.yaml').read_text())
    task['encoders']['toy']['weights'] = 'another-same-width-FM'
    (b.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='encoder identity'):Bundle(b.root)


def test_dynamic_cache_and_persistent_order_are_required(tmp_path):
    b = fixture(tmp_path/'data');m = C2MIL(b, **FAST);opt = m.make_optimizer(1e-4, 5e-4)
    m.epoch(b, opt);m.finish_epoch(b, 'cpu', {})
    good = copy.deepcopy(m.state_dict())
    for key, value in [('centers', None), ('biases', None), ('best_k', 7), ('training_order', ['wrong'])]:
        state = copy.deepcopy(good);state['_extra_state'][key] = value
        fresh = C2MIL(b, **FAST)
        with pytest.raises(ValueError):fresh.load_state_dict(state)
    raw = {k: v for k, v in good.items() if k != '_extra_state'}
    with pytest.raises(RuntimeError, match='_extra_state'):C2MIL(b, **FAST).load_state_dict(raw)


def test_epoch_three_last_tie_and_original_batch_domain(tmp_path):
    b = fixture(tmp_path/'data');m = C2MIL(b, **FAST)
    assert m.minimum_selection_epoch == 2 and m.handles_early_stopping
    assert m.selection_improved(.6, .6) and not m.selection_improved(.5, .6)
    with pytest.raises(ValueError, match='full TRAIN batch'):C2MIL(b, batch_size=100)
    with pytest.raises(ValueError, match='epoch3'):
        m.configure_training({'max_instances': None, 'grad_clip': None,
                              'selection_metric': 'c2mil_val_c_index', 'epochs': 2})


def test_supplied_original_fold_rng_is_frozen_and_does_not_change_training_rng(tmp_path):
    b = fixture(tmp_path/'data')
    schema = read_json(b.root/'c2mil_schema.json')
    generator = torch.Generator().manual_seed(98)
    torch.randperm(113, generator=generator)  # actual prior-fold sampling draw
    rng_file = tmp_path/'prior_fold_state.npy'
    np.save(rng_file, generator.get_state().numpy())
    before = rng_state()
    manifest = prepare(b, b.root/'thumbnails.csv', schema, tmp_path/'next_fold', rng_file)
    exact(before, rng_state())
    meta = read_json(manifest)
    assert meta['sampling']['mode'] == 'provided_original_fold_rng'
    gid = schema['patient_order']['train'][0];row = b.c2mil.by_patient[gid]
    n = b.c2mil.meta['slides'][row['slide_id']]['nodes']
    with h5py.File(manifest.parent/meta['slides'][row['slide_id']]['path'], 'r') as stream:
        exact(np.asarray(stream['reference_indices']), torch.randperm(n, generator=generator).numpy()[:1500])


def test_actual_unified_cli_check_train_selected_reload_and_predict(tmp_path):
    b = fixture(tmp_path/'data')
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'threads': 1, 'epochs': 3, 'model': FAST}))
    output = tmp_path/'run'
    env = {**os.environ}
    for args in [['check', '--data', str(b.root), '--method', 'C2MIL', '--deep'],
                 ['run', '--data', str(b.root), '--method', 'C2MIL', '--output', str(output), '--config', str(config)],
                 ['evaluate', '--run', str(output)]]:
        result = subprocess.run([sys.executable, '-m', 'milplus', *args], env=env, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, result.stdout
    assert read_json(output/'request.json')['method'] == 'c2mil_source'
    assert read_json(output/'test_metrics.json')['unit'] == 'patient'

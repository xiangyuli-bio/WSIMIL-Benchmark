"""Actual author CoD kernels/trainer, physical contracts and complete resume."""
import ast
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from milplus.codmil import CoDMIL
from milplus.codmil_data import prepare
from milplus.codmil_demo import create_codmil_demo
from milplus.codmil_network import COMMIT, CoDReference
from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, digest_object, read_json, write_json
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def threads():
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


def definitions(path, names, scope, replacements=()):
    root, sha = verified_source('codmil'); assert sha == COMMIT
    code = pinned_text(root, sha, path)
    for old, new in replacements:
        assert old in code
        code = code.replace(old, new)
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_original_CoD_'+path, 'exec'), scope)  # noqa: S102 -- reviewed fixed Git definitions
    return scope


class MapTorch:
    def __init__(self, maps):
        self.maps = maps

    def __getattr__(self, name):
        return getattr(torch, name)

    def load(self, path):
        assert str(path).startswith('map_10x_20x_files/')
        return self.maps[str(path).removeprefix('map_10x_20x_files/').removesuffix('.pt')]


def original_model(state, maps):
    from typing import Optional
    helpers = definitions('models/model_utils.py', ['MultiheadAttention'],
        {'torch': torch, 'Module': nn.Module, 'Optional': Optional, 'Parameter': nn.Parameter,
         '_LinearWithBias': nn.Linear, 'xavier_uniform_': nn.init.xavier_uniform_,
         'xavier_normal_': nn.init.xavier_normal_, 'constant_': nn.init.constant_})
    module = definitions('models/model_CoT.py', ['attention_block', 'CoT'],
        {'torch': MapTorch(maps), 'nn': nn, 'F': F, 'MultiheadAttention': helpers['MultiheadAttention']})
    model = module['CoT'](SimpleNamespace(hidden_size=192), num_classes=2)
    model.load_state_dict(state)
    return model


def test_native_original_all_outputs_inputs_and_parameter_gradients():
    model = CoDReference()
    mapping = torch.arange(96*4).reshape(96, 4)
    original = original_model(model.network.state_dict(), {'synthetic': mapping})
    x = [torch.randn(n, 1024, requires_grad=True) for n in [96, 384, 9]]
    y = [v.detach().clone().requires_grad_() for v in x]
    result = model(*x, mapping, torch.tensor([1]))
    expected = original(y[0], None, y[1], None, None, torch.tensor([1]), y[2], torch.tensor([ord(c) for c in 'synthetic']))
    for key, value in zip(['logits', 'probabilities', 'prediction', 'patch_prediction', 'attention', 'loss'], expected):
        exact(result[key], value)
    result['loss'].backward(); expected[-1].backward()
    for a, b in zip(x, y):
        exact(a.grad, b.grad)
    for (name, a), (other_name, b) in zip(model.network.named_parameters(), original.named_parameters()):
        assert name == other_name
        exact(a.grad, b.grad)
    assert all(p.grad is None for p in model.network.self_attention.parameters())
    assert all(p.grad is None for p in model.network.classifier.parameters())


@pytest.mark.parametrize('dimension,text_dimension,classes', [(1024, 512, 3), (1536, 512, 3), (1536, 1024, 5)])
def test_complete_full_dimension_cway_loss_gradients_label_independence_and_singleton(dimension, text_dimension, classes):
    model = CoDReference(dimension, text_dimension, classes)
    coarse, fine, text = [torch.randn(n, d, requires_grad=True) for n, d in
                          [(32, dimension), (128, dimension), (2*classes+4, text_dimension)]]
    mapping = torch.arange(128).reshape(32, 4)
    expected = model(coarse, fine, text, mapping)['logits'].detach()
    for label in range(classes):
        value = model(coarse, fine, text, mapping, torch.tensor([label]))
        exact(value['logits'], expected)
    # Independently reconstruct all source branches, including background
    # class competition. This is not just an output-shape/forward check.
    network = model.network
    weighted, _ = network.attention_block(coarse)
    projected = network.fc_text(text)
    scores = projected[:classes] @ weighted.T
    low_logits = scores.sum(1)[None]
    attention = scores[low_logits.argmax(1).item()][None]
    mask = torch.ones(32, dtype=torch.bool)
    mask[attention.topk(3, 1).indices.flatten()] = False
    background = (weighted[mask] @ torch.cat([projected[:classes], projected[2*classes:-1]]).T).sum(0)[None]
    ids = mapping[attention.topk(16, 1).indices.flatten()].flatten()
    high, _ = network.attention_block(fine[ids])
    logits = low_logits+(high @ projected[classes:2*classes].T).sum(0)[None]
    loss = F.cross_entropy(logits, torch.tensor([classes-1]))+F.cross_entropy(background, torch.tensor([classes-1]))
    actual = model(coarse, fine, text, mapping, torch.tensor([classes-1]))
    exact(actual['logits'], logits); exact(actual['loss'], loss)
    a = torch.autograd.grad(actual['loss'], [coarse, fine, text], retain_graph=True)
    b = torch.autograd.grad(loss, [coarse, fine, text])
    for x, y in zip(a, b):
        exact(x, y)
        assert x.norm() > 0
    assert torch.isfinite(model(coarse[:1], fine, text, mapping[:1], torch.tensor([0]))['loss'])


@pytest.mark.parametrize('weighted', [False, True])
def test_actual_author_complete_train_loop_optimizer_order_and_rng(tmp_path, weighted):
    bundle = Bundle(create_codmil_demo(tmp_path/'data', patients=20, classes=2, dimension=1024))
    seed_everything(186)
    model = CoDMIL(bundle, weighted_sampling=weighted)
    model.configure_training({'max_instances': None, 'grad_clip': None, 'selection_metric': 'codmil_val_loss'})
    optimizer = model.make_optimizer(.0001, 1e-5)
    model.start_epoch(0)
    rows = bundle.records('train')
    maps = {r['slide_id']: bundle.codmil.load(r)[2] for r in rows}
    original = original_model(model.core.network.state_dict(), maps)
    original_optimizer = torch.optim.Adam(original.parameters(), lr=.0001, weight_decay=1e-5)
    utils = definitions('utils/utils.py', ['collate_tranformer', 'calculate_error'], {'torch': torch})
    logger = definitions('utils/core_utils.py', ['Accuracy_Logger'], {'np': np})['Accuracy_Logger']
    # Only device compatibility is changed in the literal full source loop.
    loop = definitions('utils/core_utils.py', ['train_loop'],
        {'torch': torch, 'Accuracy_Logger': logger, 'calculate_error': utils['calculate_error']},
        [('.cuda()', ".to('cpu')")])['train_loop']
    seen = []

    class SourceData(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]; seen.append(row['slide_id'])
            low, high, _mapping = bundle.codmil.load(row)
            return low, low, high, high, row['target'], torch.zeros(len(low)), torch.tensor([ord(c) for c in row['slide_id']])

    dataset = SourceData()
    counts = np.bincount([r['target'] for r in rows], minlength=2)
    sampler = (WeightedRandomSampler(torch.tensor([len(rows)/counts[r['target']] for r in rows], dtype=torch.float64), len(rows))
               if weighted else RandomSampler(dataset))
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=utils['collate_tranformer'], num_workers=0)
    text_path = tmp_path/'actual_source_text.pt'; torch.save(model.text, text_path)
    before = rng_state()
    actual_losses = model.epoch(bundle, optimizer, 'cpu')
    after = rng_state()
    restore_rng(before)
    observed = []
    hook = original.register_forward_hook(lambda _m, _args, result: observed.append(float(result[-1].detach())))
    loop(SimpleNamespace(text_prompt_feature_path=str(text_path)), 0, original, loader, original_optimizer, 2)
    hook.remove()
    exact(model.core.network.state_dict(), original.state_dict())
    exact(optimizer.state_dict(), original_optimizer.state_dict())
    exact([x['loss'] for x in actual_losses], observed)
    exact(model.train_orders[0], seen); exact(rng_state(), after)
    assert model.optimizer_updates == len(rows)


def test_complete_train_resume_reload_and_full_val_objective(tmp_path):
    data = create_codmil_demo(tmp_path/'data', patients=30, classes=3, dimension=32, text_dimension=16)
    cfg = {'epochs': 3, 'threads': 1}
    complete = train(data, 'CoD-MIL', tmp_path/'complete', seed=89, config=cfg)
    partial = train(data, 'CoD-MIL', tmp_path/'partial', seed=89, config=cfg, stop_after=1)
    train(data, 'CoD-MIL', partial, seed=89, config=cfg, resume=True)
    a = torch.load(complete/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(partial/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'history', 'rng']:
        exact(a[key], b[key])
    assert not (complete/'test_predictions.csv').exists()
    best = torch.load(complete/'best.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == min(a['history'], key=lambda r: r['validation']['codmil_val_loss'])['epoch']
    metrics = evaluate(complete); sha = digest_file(complete/'test_predictions.csv')
    assert metrics['unit'] == 'slide' and 'codmil_val_loss' in metrics
    exact(evaluate(complete), metrics); assert digest_file(complete/'test_predictions.csv') == sha
    exact(evaluate(partial), metrics); assert digest_file(partial/'test_predictions.csv') == sha
    frame = pd.read_csv(complete/'test_predictions.csv')
    assert np.isclose(metrics['codmil_val_loss'], frame.codmil_total_loss.mean())


def test_physical_schema_and_stable_map_row_order_are_enforced(tmp_path):
    bundle = Bundle(create_codmil_demo(tmp_path/'data', classes=3, dimension=32))
    for row in bundle.rows.to_dict('records'):
        low, fine, mapping = bundle.codmil.load(row)
        assert mapping.shape == (len(low), 4) and len(fine) == 4*len(low)
    record = bundle.codmil.manifest['records'][0]
    path = record['map']['path']
    mapping = np.load(path, allow_pickle=False)
    mapping[0, 0] = mapping[-1, 0]
    np.save(path, mapping)
    with pytest.raises(ValueError, match='asset bytes changed'):
        bundle.codmil.load(bundle.rows.to_dict('records')[0])


@pytest.mark.parametrize('mutation', ['geometry', 'duplicate', 'fine_id', 'class_order', 'matched_space', 'normal_role'])
def test_preparation_rejects_wrong_geometry_mapping_and_text_contracts(tmp_path, mutation):
    data = create_codmil_demo(tmp_path/'data', classes=3, dimension=32)
    task = data/'task.yaml'; cfg = yaml.safe_load(task.read_text()); del cfg['encoders']['toy']['codmil']
    task.write_text(yaml.safe_dump(cfg)); bundle = Bundle(data); folder = data/'codmil_assets'
    if mutation in {'geometry', 'duplicate', 'fine_id'}:
        row = pd.read_csv(folder/'index.csv').iloc[0]
        path = data/row.map_path; mapping = np.load(path, allow_pickle=False)
        mapping[0, 0] = mapping[-1, 0] if mutation == 'geometry' else mapping[0, 1] if mutation == 'duplicate' else 10**8
        np.save(path, mapping)
    else:
        meta = read_json(folder/'text.json')
        if mutation == 'class_order':
            meta['prompts'][0]['class'] = bundle.classes[-1]
        elif mutation == 'matched_space':
            meta['embedding_space'] = 'different_unaligned_model'
        else:
            meta['prompts'][-2]['role'] = 'high'
        write_json(folder/'text.json', meta)
    with pytest.raises(ValueError, match='physical region|duplicate|fine row IDs|text rows|matched VLM|normal'):
        prepare(bundle, folder/'index.csv', folder/'schema.json', folder/'text.json', folder/'new_manifest.json')


def test_checkpoint_rejects_text_or_input_contract_and_bad_train_order(tmp_path):
    bundle = Bundle(create_codmil_demo(tmp_path/'data', dimension=32))
    model = CoDMIL(bundle)
    state = copy.deepcopy(model.state_dict()); state['text'][0, 0] += 1
    with pytest.raises(ValueError, match='text differs'):
        model.load_state_dict(state)
    for mutation in ['contract', 'order', 'steps']:
        state = copy.deepcopy(model.state_dict())
        if mutation == 'contract':
            state['_extra_state']['contract'] = 'different'
        elif mutation == 'order':
            state['_extra_state']['completed_epochs'] = 1
            state['_extra_state']['epoch_index'] = 0
            state['_extra_state']['train_orders'] = [['test_patient']]
        else:
            state['_extra_state']['optimizer_updates'] = 1
        with pytest.raises(ValueError, match='contract changed|TRAIN|update'):
            model.load_state_dict(state)


def test_exact_original_stop_policy_first_ties_and_minimum_epoch(tmp_path):
    bundle = Bundle(create_codmil_demo(tmp_path/'data', dimension=32))
    model = CoDMIL(bundle)
    cls = definitions('utils/core_utils.py', ['EarlyStopping'], {'np': np, 'torch': torch}, [('np.Inf', 'np.inf')])['EarlyStopping']
    reference = cls(patience=20, stop_epoch=50)
    dummy = nn.Linear(1, 1); best, bad = -np.inf, 0
    for epoch in range(60):
        loss = 2. if epoch == 0 else 1.
        reference(epoch, loss, dummy, ckpt_name=str(tmp_path/'source_stop.pt'))
        objective = model.selection_objective({'codmil_val_loss': loss}, epoch, None)
        improved = model.selection_improved(objective, best)
        best, bad = (objective, 0) if improved else (best, bad+1)
        stop = bad >= 20 and epoch+1 > model.stop_after_epoch
        assert stop == reference.early_stop
        if stop:
            assert epoch == 51
            break


def test_cli_fixture_setup_alias_and_input_compatibility(tmp_path):
    from milplus.cli import dispatch, parser
    from milplus.registry import check_compatibility, resolve_for_bundle
    result = dispatch(parser().parse_args(['demo', '--codmil', '--classes', '3', '--patients', '30', '--output', str(tmp_path/'cli')]))
    bundle = Bundle(result['created']); spec = resolve_for_bundle('CoD-MIL', bundle)
    assert spec['id'] == 'codmil_source_repaired' and spec['external_source'] == 'codmil'
    check_compatibility(spec, bundle)
    for extra in [['--ins'], ['--survival']]:
        with pytest.raises(ValueError, match='separate physical'):
            dispatch(parser().parse_args(['demo', '--codmil', '--output', str(tmp_path/'bad'), *extra]))


def test_coarse_coordinate_index_and_embedded_source_metadata_mismatch_rejected(tmp_path):
    data = create_codmil_demo(tmp_path/'data', dimension=32)
    features = pd.read_csv(data/'features.csv')
    coords = data/'changed_coarse_coords.npy'; np.save(coords, np.zeros((1, 2)))
    features['coords_path'] = str(coords)
    features.to_csv(data/'features.csv', index=False)
    with pytest.raises(ValueError, match='coarse coordinate path'):
        Bundle(data)
    features.drop(columns=['coords_path']).to_csv(data/'features.csv', index=False)
    path = data/'codmil_assets/manifest.json'; manifest = read_json(path)
    manifest['text_metadata']['text_provenance'] = 'different from actual source metadata file'
    manifest['contract'] = digest_object({k: v for k, v in manifest.items() if k != 'contract'})
    write_json(path, manifest)
    with pytest.raises(ValueError, match='embedded schema/text metadata'):
        Bundle(data)


def test_inconsistent_checkpoint_epoch_and_boolean_update_rejected(tmp_path):
    model = CoDMIL(Bundle(create_codmil_demo(tmp_path/'data', dimension=32)))
    for key, value in [('epoch_index', 4), ('optimizer_updates', False), ('completed_epochs', -1)]:
        state = copy.deepcopy(model.state_dict()); state['_extra_state'][key] = value
        with pytest.raises(ValueError, match='epoch/update state'):
            model.load_state_dict(state)


def test_native_defaults_recipe_and_memory_limits(tmp_path):
    bundle = Bundle(create_codmil_demo(tmp_path/'data', dimension=32))
    model = CoDMIL(bundle, max_dense_bytes=1)
    with pytest.raises(ValueError, match='max_dense_bytes'):
        model.prepare(bundle, 1, 'cpu')
    cfg = {'max_instances': None, 'grad_clip': None, 'selection_metric': 'codmil_val_loss'}
    for update in [{'max_instances': 16}, {'grad_clip': 1}, {'selection_metric': 'macro_auc'}]:
        with pytest.raises(ValueError):
            model.configure_training({**cfg, **update})
    assert model.stop_after_epoch == 51 and model.core.network.attention_block.D == 192
    assert model.protocol()['text_condition'] == 'matched_vlm'

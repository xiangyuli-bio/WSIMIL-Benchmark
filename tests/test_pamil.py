"""PAMIL complete recipes against independent pinned author ASTs and input attacks."""
import ast
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from milplus.data import Bundle
from milplus.engine import evaluate, predict, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file
from milplus.pamil import PAMIL, PROFILES, pamil_epoch, target_vector
from milplus.pamil_data import prepare_annotations
from milplus.pamil_demo import create_pamil_demo
from milplus.registry import check_compatibility, require_runnable, resolve_for_bundle
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, profile='pamil_prototype_repaired', **options):
    root = create_pamil_demo(tmp_path/'data', multilabel=profile.endswith('multilabel'), dimension=4)
    bundle = Bundle(root)
    opts = {'hidden': 16, 'attention_dim': 8, 'prototypes': 6,
            'warmup_epochs': 1, 'push_epochs': [2], 'last_epochs': 1, **options}
    model = PAMIL(bundle, profile, **opts)
    return model, bundle, opts


class Logger:
    def __init__(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass


def reference(variant):
    """Independently import fixed Git source, never the adapter loader or its functions."""
    root, sha = verified_source('pamil_prototype')
    branch = 'PAMIL_multi_label' if variant == 'multilabel' else 'PAMIL_multi_class'
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'KMeans': KMeans,
             'Accuracy_Logger': Logger, 'calculate_error': lambda *args: 0.,
             'print': lambda *args, **kwargs: None, 'device': torch.device('cpu')}
    def text(path):
        return subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)
    def definitions(code, names, path):
        nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)  # noqa: S102 -- independent locked Git definitions
    path = branch+'/models/model_protoattention.py'
    code = text(path)
    code = code.replace('gate = True, size_arg', 'input_dim=4, hidden_dim=16, attention_dim=8, gate = True, size_arg')
    code = code.replace('size = self.size_dict[size_arg]', 'size = [input_dim, hidden_dim, attention_dim]')
    code = code.replace('if proto_path:', 'if True:').replace('torch.from_numpy(np.load(proto_path))', 'torch.zeros(n_protos, input_dim)')
    code = code.replace('.cuda()', '.to(device)').replace('loss_inst = torch.tensor(0.)', 'loss_inst = h.new_zeros(())')
    code = code.replace('np.where(inst_label == proto_class[i])', 'np.where(np.asarray(inst_label[0]) == int(proto_class[i]))')
    code = code.replace('proto_score = proto_score / torch.max(proto_score)',
        'proto_score = proto_score / torch.where(proto_score.max() == 0, proto_score.new_tensor(1.), proto_score.max())')
    if variant != 'source':
        code = code.replace('torch.log(probs1)', 'F.logsigmoid(logits1)')
        code = code.replace('self.inst_eval(A, h, instance_classifier)', 'self.inst_eval(A[c:c+1], h, instance_classifier)')
        code = code.replace('self.inst_eval_out(A, h, instance_classifier)', 'self.inst_eval_out(A[c:c+1], h, instance_classifier)')
    definitions(code, ['Attn_Net_Gated', 'KL_Divergence_Loss', 'PAMIL', 'push'], path)
    path = branch+'/utils/train_utils.py'
    definitions(text(path), ['_freeze_layer', '_unfreeze_layer', 'warm_only', 'joint', 'last_only'], path)
    path = branch+'/utils/core_utils.py'
    code = text(path)
    if variant == 'repaired':
        code = code.replace('loss_fn(Y_prob, label_bn.squeeze().float())', 'loss_fn(logits.squeeze(0), label_bn.squeeze().float())')
        code = code.replace('inst_pred=args.inst_pred, label=label)', 'inst_pred=args.inst_pred, label=label_bn.unsqueeze(0))')
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'train_loop_pamil')
    loop_index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.For))
    loop = node.body[loop_index]
    last = next(i for i, n in enumerate(loop.body) if ast.unparse(n) == 'optimizer.zero_grad()')
    loop.body = loop.body[:last+1]  # Only strip post-update annotation reports, never numeric training statements.
    node.body = node.body[:loop_index+1]+ast.parse('return train_loss/len(loader)').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, 'exec'), scope)  # noqa: S102
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_pamil')
    idx = next(i for i, n in enumerate(node.body) if isinstance(n, ast.With))
    block = node.body[idx]
    loop = next(n for n in block.body if isinstance(n, ast.For))
    end = next(i for i, n in enumerate(loop.body) if ast.unparse(n) == 'val_loss += loss.item()')
    loop.body = loop.body[:end+1]
    node.body = node.body[:idx+1]+ast.parse('return val_loss/len(loader)').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, 'exec'), scope)  # noqa: S102
    return scope


class SourceDataset:
    def __init__(self, bundle, subset='train'):
        self.bundle, self.rows = bundle, bundle.records(subset)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        x, coords = self.bundle.load(row)
        label = torch.tensor(row['targets']) if self.bundle.task_type == 'multilabel' else torch.tensor(row['target'])
        return x, label, coords, self.bundle.pamil.load(row) if row['subset'] == 'train' else [], row['slide_id']


def reference_loader(bundle, weighted=False, reporting=False, subset='train'):
    data = SourceDataset(bundle, subset)
    if subset != 'train':
        sampler = SequentialSampler(data)
    elif weighted:
        keys = [tuple(r['targets']) if bundle.task_type == 'multilabel' else r['target'] for r in data.rows]
        sampler = WeightedRandomSampler(torch.tensor([len(keys)/keys.count(k) for k in keys], dtype=torch.double), len(keys))
    else:
        sampler = RandomSampler(data)
    def collate(batch):
        x, target, coords, labels, slide = batch[0]
        return x, target.unsqueeze(0), [coords], [[] if reporting else labels], [slide]
    return DataLoader(data, batch_size=1, sampler=sampler, collate_fn=collate)


def assert_nested(a, b, atol=0):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=atol)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: assert_nested(a[key], b[key], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested(x, y, atol)
    else:
        assert a == b


@pytest.mark.parametrize('profile,inst_pred', [(p, inst) for p in PROFILES for inst in ([False] if p.endswith('source') else [False, True])])
@pytest.mark.parametrize('weighted', [False, True])
def test_full_author_epochs_push_and_optimizer(tmp_path, profile, inst_pred, weighted):
    seed_everything(37)
    model, bundle, _ = fixture(tmp_path, profile, dropout=True, weighted_sampling=weighted, inst_pred=inst_pred)
    model.prepare(bundle, 37, 'cpu')
    source = reference(model.variant)
    native = source['PAMIL'](n_protos=6, n_classes=3, proto_pred=True, dropout=True)
    native.load_state_dict(copy.deepcopy(model.network.state_dict()))
    native_optimizer = torch.optim.Adam(native.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = model.make_optimizer(1e-4, 1e-5)
    source['warm_only'](native)
    args = SimpleNamespace(inst_pred=inst_pred, w_er=.5, w_clst=-.5, w_inst=.5, w_proto_clst=-.2, attention_er=False)
    cfg = {'max_instances': None, 'grad_clip': None}
    groups = units(bundle, 'train')
    def trace_updates(opt):
        trace, original = [], opt.step
        def step(*args, **kwargs):
            gradients = [None if p.grad is None else p.grad.detach().clone() for g in opt.param_groups for p in g['params']]
            result = original(*args, **kwargs)
            trace.append({'gradients': gradients,
                          'parameters': [p.detach().clone() for g in opt.param_groups for p in g['params']],
                          'optimizer': copy.deepcopy(opt.state_dict())})
            return result
        opt.step = step
        return trace
    actual_trace, expected_trace = trace_updates(optimizer), trace_updates(native_optimizer)
    for epoch in range(3):
        start = rng_state()
        model.train(); model.start_epoch(epoch)
        actual = pamil_epoch(model, bundle, groups, optimizer, 'cpu', cfg)
        after = rng_state()
        restore_rng(start)
        if epoch == 1:
            source['joint'](native)
        if epoch == 2:
            native.eval()
            source['push'](reference_loader(bundle, weighted), native, epoch)
            source['last_only'](native)
            torch.testing.assert_close(model.network.proto, native.proto, atol=0, rtol=0)
        native.train()
        expected = source['train_loop_pamil'](epoch, native, reference_loader(bundle, weighted, reporting=True),
            native_optimizer, 3, args, loss_fn=F.binary_cross_entropy if model.variant == 'multilabel' else nn.CrossEntropyLoss())
        assert np.mean([r['loss'] for r in actual]) == pytest.approx(expected, abs=3e-7)
        for key, value in native.state_dict().items():
            torch.testing.assert_close(model.network.state_dict()[key], value, atol=3e-6, rtol=3e-5)
        # A projected prototype remains frozen. Its source replacement is outside Adam;
        # the adapter preserves ownership instead of copying that invalid pointer.
        for a, b in zip(optimizer.param_groups[0]['params'][1:], native_optimizer.param_groups[0]['params'][1:]):
            if a in optimizer.state and b in native_optimizer.state:
                assert_nested(optimizer.state[a], native_optimizer.state[b], atol=3e-7)
        assert torch.equal(after['torch'], rng_state()['torch'])
        # Prototype value changes outside Adam in both routes. The source's old
        # unowned Parameter is excluded from post-push tensor trace comparison.
        for a, b in zip(actual_trace, expected_trace):
            assert_nested(a['gradients'][1:], b['gradients'][1:], atol=3e-6)
            assert_nested(a['parameters'][1:], b['parameters'][1:], atol=3e-6)
            assert_nested(a['optimizer'], b['optimizer'], atol=3e-7)
        assert len(actual_trace) == len(expected_trace) == len(groups)
        actual_trace.clear(); expected_trace.clear()
        restore_rng(after)
        model.finish_epoch(bundle, 'cpu', cfg)
        before_validation = rng_state()
        frame = predict(model, bundle, profile, 'val', 'cpu', cfg)
        after_validation = rng_state()
        restore_rng(before_validation)
        expected_validation = source['validate_pamil'](0, epoch, native,
            reference_loader(bundle, reporting=True, subset='val'), 3, args,
            loss_fn=F.binary_cross_entropy if model.variant == 'multilabel' else nn.CrossEntropyLoss())
        assert frame.pamil_loss.mean() == pytest.approx(expected_validation, abs=3e-7)
        assert torch.equal(after_validation['torch'], rng_state()['torch'])
        restore_rng(after_validation)
    assert len(model.pushes) == 1 and model.mode == 'last'
    assert all(w['slide_id'] in {r['slide_id'] for r in bundle.records('train')} for w in model.pushes[0]['winners'])


@pytest.mark.parametrize('profile', list(PROFILES))
def test_two_level_clustering_matches_independent_source_block(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile)
    seed_everything(71)
    start = rng_state()
    model.prepare(bundle, 71, 'cpu')
    after = rng_state()
    restore_rng(start)
    root, sha = verified_source('pamil_prototype')
    branch = 'PAMIL_multi_label' if model.variant == 'multilabel' else 'PAMIL_multi_class'
    code = subprocess.check_output(['git', 'show', sha+':'+branch+'/clustering.py'], cwd=root, text=True)
    start_text = 'cluster_centers = []'
    begin = code.rfind('\n', 0, code.index(start_text))+1
    block = code[begin:code.index('            os.makedirs', code.index(start_text))]
    import textwrap
    block = textwrap.dedent(block)
    if model.variant != 'source':
        block = block.replace('min(len(cors), args.num_clusters)', 'min(len(data), args.num_clusters)')
    scope = {'np': np, 'torch': torch, 'KMeans': lambda **kw: KMeans(n_init=10, **kw),
             'args': SimpleNamespace(num_clusters=8, num_protos=6), 'print': lambda *args: None,
             'train_loader': reference_loader(bundle)}
    exec(block, scope)  # noqa: S102 -- isolated original clustering block with explicit local-count repair
    np.testing.assert_array_equal(model.network.proto.detach().numpy(), scope['protos'])
    assert torch.equal(after['torch'], start['torch'])
    assert np.array_equal(after['numpy'][1], start['numpy'][1])


@pytest.mark.parametrize('profile', list(PROFILES))
@pytest.mark.parametrize('boundary', [1, 2, 3])
def test_full_train_reload_and_exact_resume_across_push(tmp_path, profile, boundary):
    data = create_pamil_demo(tmp_path/'data', multilabel=profile.endswith('multilabel'), dimension=4)
    cfg = {'epochs': 5, 'threads': 1, 'model': {'hidden': 16, 'attention_dim': 8, 'prototypes': 6,
        'warmup_epochs': 1, 'push_epochs': [2], 'last_epochs': 1, 'dropout': True}}
    full = train(data, profile, tmp_path/'full', config=cfg)
    part = train(data, profile, tmp_path/'part', config=cfg, stop_after=boundary)
    train(data, profile, part, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, part]]
    assert_nested(a['model'], b['model']); assert_nested(a['optimizers'], b['optimizers'])
    assert a['history'] == b['history']
    first = evaluate(full); frame = (full/'test_predictions.csv').read_bytes()
    assert first == evaluate(full) and frame == (full/'test_predictions.csv').read_bytes()
    protocol = json.loads((full/'method_protocol.json').read_text())
    assert protocol['pushes'][0]['epoch'] == 2 and protocol['optimizer_steps'] > 0
    assert protocol['gpu_validation'] == protocol['paper_reproduction'] == 'not_run'
    manifest = json.loads((full/'test_instances/manifest.json').read_text())
    for item in manifest['records']:
        assert digest_file(item['path']) == item['sha256']
        with h5py.File(item['path'], 'r') as h:
            row = next(r for r in Bundle(data).records('test') if r['slide_id'] == item['slide_id'])
            with h5py.File(data/row['path'], 'r') as features:
                np.testing.assert_array_equal(h['coords'][:], features['coords'][:])
            assert h['prototype_similarity'].shape[1] == 6
            np.testing.assert_allclose(h['attention'][:].sum(0), np.ones(3), atol=1e-6)


@pytest.mark.parametrize('field,value', [('warmup_epochs', 0), ('push_epochs', []), ('push_epochs', [1]),
    ('push_epochs', [2, 2]), ('push_epochs', [True]), ('push_epochs', [2, 3]), ('last_epochs', 0),
    ('prototypes', 1), ('local_clusters', False), ('w_er', 0), ('w_clst', .1), ('w_proto_clst', 0),
    ('proto_weight', 1), ('w_inst', float('nan')), ('inst_pred', 1), ('weighted_sampling', 0)])
def test_invalid_method_contract(tmp_path, field, value):
    with pytest.raises(ValueError): fixture(tmp_path, **{field: value})


@pytest.mark.parametrize('kind', ['heldout', 'missing', 'duplicate', 'float', 'out_of_range', 'short', 'object'])
def test_annotations_are_exact_training_only_safe_integer_rows(tmp_path, kind):
    _, bundle, _ = fixture(tmp_path)
    frame = pd.read_csv(bundle.root/'annotations.csv')
    if kind == 'heldout': frame.loc[0, 'slide_id'] = bundle.records('test')[0]['slide_id']
    if kind == 'missing': frame = frame.iloc[1:]
    if kind == 'duplicate': frame = pd.concat([frame, frame.iloc[:1]])
    if kind in {'float', 'out_of_range', 'short', 'object'}:
        path = bundle.root/frame.iloc[0]['path']
        data = np.load(path)
        if kind == 'float': data = data.astype(float)
        if kind == 'out_of_range': data[0] = 3
        if kind == 'short': data = data[:-1]
        if kind == 'object': data = data.astype(object)
        np.save(path, data)
    frame.to_csv(bundle.root/'bad.csv', index=False)
    with pytest.raises(ValueError): prepare_annotations(bundle, bundle.root/'bad.csv', bundle.root/'bad.json', 'provided labels')


@pytest.mark.parametrize('mutation', ['annotation', 'features', 'coordinates', 'fm', 'split', 'classes', 'provenance', 'weights'])
def test_frozen_annotations_reject_mutations(tmp_path, mutation):
    _, bundle, _ = fixture(tmp_path)
    manifest_path = bundle.pamil.path
    manifest = json.loads(manifest_path.read_text())
    row = bundle.records('train')[0]
    if mutation == 'annotation':
        path = Path(manifest['records'][row['slide_id']]['path'])
        labels = np.load(path); labels[0] = -1; np.save(path, labels)
    elif mutation in {'features', 'coordinates'}:
        with h5py.File(bundle.root/row['path'], 'r+') as h:
            key = 'features' if mutation == 'features' else 'coords'
            h[key][0, 0] += 1
    elif mutation == 'weights':
        task = yaml.safe_load((bundle.root/'task.yaml').read_text())
        task['encoders']['toy']['weights'] = 'different-model-version'
        (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    else:
        manifest[mutation] = '' if mutation == 'provenance' else ['different'] if mutation == 'classes' else 'other'
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): Bundle(bundle.root).check(deep=True)


@pytest.mark.parametrize('failure', ['exhausted', 'missing'])
def test_push_is_atomic_and_never_loops_or_zeroes_unmatched_prototypes(tmp_path, failure):
    model, bundle, _ = fixture(tmp_path)
    model.prepare(bundle, 1, 'cpu')
    with torch.no_grad():
        model.network.attention_net.attention_c.weight.fill_(0)
        model.network.attention_net.attention_c.weight[0].fill_(1)
    manifest = bundle.pamil.manifest
    for row in bundle.records('train'):
        item = manifest['records'][row['slide_id']]
        path = Path(item['path']); labels = np.full(item['instances'], -1, dtype=np.int64)
        if failure == 'exhausted': labels[0] = 0
        np.save(path, labels); item['sha256'] = digest_file(path)
    original = model.network.proto.detach().clone(); identity = id(model.network.proto)
    with pytest.raises(ValueError, match='exhausted|without eligible'):
        model.push()
    assert id(model.network.proto) == identity and torch.equal(model.network.proto, original)
    assert not model.pushes


@pytest.mark.parametrize('profile', list(PROFILES))
def test_all_query_scores_independent_of_query_labels(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile)
    model.prepare(bundle, 7, 'cpu'); model.eval()
    cfg = {'max_instances': None}
    row = bundle.records('test')[0]
    a = model.forward_unit(bundle, [row], 'cpu', cfg)
    changed = {**row, 'target': (row.get('target', 0)+1) % 3, 'targets': (1, 0, 1)}
    b = model.forward_unit(bundle, [changed], 'cpu', cfg)
    for key in ['logits', 'probabilities', 'proto_A', 'A_raw']:
        assert torch.equal(a[key], b[key])
    assert model.export_instances(a, row, tmp_path/'a')['sha256'] == model.export_instances(b, changed, tmp_path/'b')['sha256']
    with pytest.raises(ValueError, match='restricted'):
        bundle.pamil.load(row)


@pytest.mark.parametrize('profile', ['pamil_prototype_repaired', 'pamil_prototype_multilabel'])
def test_class_specific_instance_heads_and_complete_auxiliary_losses(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile, inst_pred=True)
    model.prepare(bundle, 7, 'cpu'); model.start_epoch(0)
    row = next(r for r in bundle.records('train') if r.get('target') == 0 or r.get('targets') == (1, 0, 0))
    x, _ = bundle.load(row)
    target = target_vector(row, 3, model.variant, 'cpu')
    result = model(x, label=target)
    loss, terms = model.loss(result, target); loss.backward()
    assert 'instance_prediction' in terms
    assert ('prototype_cluster' in terms) == (model.variant != 'multilabel')
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()) for head in model.network.instance_classifiers)
    assert model.network.proto.grad is None


def test_stage_freezes_and_parameter_ownership(tmp_path):
    for i, profile in enumerate(PROFILES):
        model, bundle, _ = fixture(tmp_path/str(i), profile)
        model.prepare(bundle, 4, 'cpu')
        optimizer = model.make_optimizer(.001, 0)
        identity = id(model.network.proto)
        for epoch, mode in [(0, 'warm'), (1, 'joint'), (2, 'last'), (3, 'joint')]:
            model.start_epoch(epoch)
            assert model.mode == mode
            assert all(p.requires_grad == (mode != 'last' and not (epoch == 3 and model.variant == 'source')) for p in model.network.fc.parameters())
            model.finish_epoch(bundle, 'cpu', {})
        assert id(model.network.proto) == identity
        assert any(model.network.proto is p for p in optimizer.param_groups[0]['params'])


def test_cli_binding_routing_and_missing_annotation_rejection(tmp_path):
    from milplus.cli import main
    from milplus.execution import matrix
    _, bundle, _ = fixture(tmp_path)
    assert main(['prepare-pamil', '--data', str(bundle.root), '--annotations', str(bundle.root/'annotations.csv'),
                 '--output', str(tmp_path/'new.json'), '--provenance', 'synthetic supplied rows']) == 0
    assert resolve_for_bundle('pamil_prototype', bundle)['id'] == 'pamil_prototype_repaired'
    assert resolve_for_bundle('pamil_prototype_source', bundle)['id'] == 'pamil_prototype_source'
    entries = matrix(bundle.root, ['all'], [bundle.fm], [1], tmp_path/'matrix.json')
    assert len(entries) == 164
    assert next(e for e in entries if e['family'] == 'pamil_prototype')['state'] == 'ready'
    task = yaml.safe_load((bundle.root/'task.yaml').read_text())
    del task['encoders']['toy']['pamil']
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='TRAIN instance'):
        check_compatibility(require_runnable('pamil_prototype_repaired'), Bundle(bundle.root))


@pytest.mark.parametrize('config', [{'epochs': 2}, {'epochs': 3, 'max_instances': 4},
                                  {'epochs': 3, 'grad_clip': 5}, {'epochs': 3, 'selection_metric': 'macro_auc'}])
def test_incomplete_recipe_budget_refused(tmp_path, config):
    _, bundle, opts = fixture(tmp_path)
    with pytest.raises(ValueError):
        train(bundle.root, 'pamil_prototype_repaired', tmp_path/'run', config={'model': opts, **config})


@pytest.mark.parametrize('validation_selection', [False, True])
def test_source_last_checkpoint_or_explicit_native_validation_selection(tmp_path, validation_selection):
    model, bundle, opts = fixture(tmp_path, validation_selection=validation_selection)
    cfg = {'epochs': 3, 'threads': 1, 'patience': 1, 'model': opts}
    out = train(bundle.root, model.profile, tmp_path/'run', config=cfg)
    last = torch.load(out/'last.pt', weights_only=False)
    best = torch.load(out/'best.pt', weights_only=False)
    assert last['epoch'] == 2  # Source stopping flag is ignored; all stages execute.
    losses = [h['validation']['pamil_loss'] for h in last['history']]
    expected = max(i for i, loss in enumerate(losses) if loss == min(losses)) if validation_selection else 2
    assert best['epoch'] == expected


def test_zero_prototype_similarity_has_finite_loss_and_gradient(tmp_path):
    model, bundle, _ = fixture(tmp_path)
    model.prepare(bundle, 1, 'cpu'); model.start_epoch(0)
    with torch.no_grad():
        model.network.attention_net.attention_a[0].weight.zero_()
        model.network.attention_net.attention_a[0].bias.zero_()
    row = bundle.records('train')[0]
    target = target_vector(row, model.classes, model.variant, 'cpu')
    result = model(bundle.load(row)[0], label=target)
    loss, _ = model.loss(result, target); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(result['proto_A'], torch.zeros_like(result['proto_A']))


def test_preparation_rejects_shared_label_files_and_overwrites(tmp_path):
    _, bundle, _ = fixture(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_annotations(bundle, bundle.root/'annotations.csv', bundle.pamil.path, 'synthetic existing')
    frame = pd.read_csv(bundle.root/'annotations.csv'); frame.loc[1, 'path'] = frame.loc[0, 'path']
    frame.to_csv(bundle.root/'duplicate.csv', index=False)
    with pytest.raises(ValueError, match='physical annotation'):
        prepare_annotations(bundle, bundle.root/'duplicate.csv', bundle.root/'new.json', 'synthetic supplied')


@pytest.mark.parametrize('profile', list(PROFILES))
@pytest.mark.parametrize('dropout', [False, True])
def test_author_constructor_initialization_and_rng(tmp_path, profile, dropout):
    seed_everything(111)
    model, _, _ = fixture(tmp_path, profile, dropout=dropout)
    after = rng_state()
    source = reference(model.variant)
    seed_everything(111)
    network = source['PAMIL'](n_protos=6, n_classes=3, proto_pred=True, dropout=dropout)
    assert_nested(model.network.state_dict(), network.state_dict())
    assert torch.equal(after['torch'], rng_state()['torch'])

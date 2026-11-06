"""Independent author networks/whole loops, temporal states and paper equations."""
import ast
import copy
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, predict, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file
from milplus.loss_attention import PROFILES, LossAttention, loss_attention_epoch
from milplus.registry import inspect_method
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, profile, fixed_instances=None, **options):
    root = create_demo(tmp_path/'data', patients=30, classes=3, dimension=4)
    if fixed_instances is not None:
        for row in Bundle(root).rows.to_dict('records'):
            with h5py.File(root/row['path'], 'r+') as h:
                for key in ['features', 'coords']:
                    data = h[key][:fixed_instances]
                    del h[key]; h[key] = data
    bundle = Bundle(root)
    opts = {'widths': [24, 16, 8]} if profile.endswith('classical') else {'hidden': 16}
    opts.update(options)
    return LossAttention(bundle, profile, **opts), bundle, opts


def definitions(scope, filename, names, replacements=()):
    root, sha = verified_source('loss_attention')
    code = subprocess.check_output(['git', 'show', sha+':'+filename], cwd=root, text=True)
    for old, new in replacements:
        assert old in code
        code = code.replace(old, new)
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)  # noqa: S102 -- independent pinned author Git AST


def reference():
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'defaultdict': defaultdict,
             'Iterable': Iterable, 'deepcopy': copy.deepcopy, 'chain': chain, 'required': object(),
             'Variable': torch.autograd.Variable, 'args': SimpleNamespace(cuda=False),
             'print': lambda *args, **kwargs: None}
    definitions(scope, 'MIL/weight_loss.py',
        ['log_sum_exp', 'class_select', 'cross_entropy_with_weights', 'CrossEntropyLoss'],
        [('b= torch.unsqueeze(torch.squeeze(b),dim=1)', 'b = b.unsqueeze(1)')])
    definitions(scope, 'MIL/model.py', ['AttentionLayer'], [('torch.squeeze(alpha)', 'alpha.reshape(-1)')])
    scope['MILAttention'] = scope['AttentionLayer']
    definitions(scope, 'Image classification/resnet.py', ['AttentionLayer'])
    scope['ImageAttention'] = scope['AttentionLayer']
    definitions(scope, 'Image classification/main.py', ['rampup', 'rampdown', 'step_rampup'], [('np.float(', 'float(')])
    definitions(scope, 'Image classification/optimizer.py', ['Optimizer'])
    adam_scope = dict(scope)
    definitions(adam_scope, 'Image classification/robust_adam.py', ['Adam'])
    scope['Adam'] = adam_scope['Adam']
    return scope


def native_mil_network(scope, model):
    """Original CNN forward with just the declared frozen-feature encoder boundary."""
    root, sha = verified_source('loss_attention')
    text = subprocess.check_output(['git', 'show', sha+':MIL/model.py'], cwd=root, text=True)
    text = text.replace('self.L = 500', 'self.L = hidden')
    text = text.replace('def __init__(self):', 'def __init__(self, input_dim, hidden, classes, dropout):')
    text = text.replace('50 * 4 * 4, self.L', 'input_dim, self.L')
    text = text.replace('nn.Dropout()', 'nn.Dropout(dropout)')
    text = text.replace('self.L*self.K, 4', 'self.L*self.K, classes')
    text = text.replace('H.view(-1, 50 * 4 * 4)', 'H.reshape(-1, self.input_dim)')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'CNN')
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    for assignment in init.body:
        if isinstance(assignment, ast.Assign) and ast.unparse(assignment.targets[0]) == 'self.feature_extractor_part1':
            assignment.value = ast.parse('nn.Identity()', mode='eval').body
    init.body.extend(ast.parse('self.input_dim = input_dim').body)
    scope['AttentionLayer'] = scope['MILAttention']
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'MIL/model.py', 'exec'), scope)  # noqa: S102
    net = scope['CNN'](model.dim, model.widths[0], model.classes, model.dropout)
    net.feature_extractor_part2.load_state_dict(model.project.state_dict())
    net.att_layer.load_state_dict(model.attention.state_dict())
    net.linear.load_state_dict(model.classifier.state_dict())
    return net


class SourceData:
    def __init__(self, bundle, subset='train'):
        self.bundle, self.rows = bundle, bundle.records(subset)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return self.bundle.load(row)[0], torch.tensor(row['target']), index


def native_loader(bundle, mil=False, subset='train'):
    data = SourceData(bundle, subset)
    if mil:
        def collate(batch):
            x, label, _ = batch[0]
            return x.unsqueeze(0), [label.unsqueeze(0), []]
        return DataLoader(data, batch_size=1, shuffle=subset == 'train', collate_fn=collate)
    return DataLoader(data, batch_size=1, shuffle=subset == 'train')


def assert_nested(a, b, atol=0):
    if torch.is_tensor(a): torch.testing.assert_close(a, b, rtol=0, atol=atol)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: assert_nested(a[key], b[key], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested(x, y, atol)
    else: assert a == b


def capture_steps(opt):
    result, original = [], opt.step
    def step(*args, **kwargs):
        gradients = [None if p.grad is None else p.grad.detach().clone() for g in opt.param_groups for p in g['params']]
        answer = original(*args, **kwargs)
        result.append({'gradients': gradients,
                       'parameters': [p.detach().clone() for g in opt.param_groups for p in g['params']],
                       'moments': [copy.deepcopy(opt.state[p]) for g in opt.param_groups for p in g['params']]})
        return answer
    opt.step = step
    return result


def maximum_difference(a, b):
    if torch.is_tensor(a): return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a, dict): return max((maximum_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)): return max((maximum_difference(x, y) for x, y in zip(a, b)), default=0.)
    return 0.


@pytest.mark.parametrize('dropout', [0., .5])
@pytest.mark.parametrize('count', [1, 16, None])
def test_complete_original_mil_epochs_and_each_adam_update(tmp_path, dropout, count):
    seed_everything(43)
    model, bundle, _ = fixture(tmp_path, 'loss_attention_mil_source', fixed_instances=count, dropout=dropout)
    scope = reference(); native = native_mil_network(scope, model)
    definitions(scope, 'MIL/main.py', ['train'])
    actual_opt = model.make_optimizer(.0005, .0001)
    expected_opt = torch.optim.Adam(native.parameters(), lr=.0005, weight_decay=.0001)
    actual_trace, expected_trace = capture_steps(actual_opt), capture_steps(expected_opt)
    cfg = {'max_instances': None}
    differences = []
    for epoch in range(2):
        start = rng_state()
        model.train(); model.start_epoch(epoch)
        loss_attention_epoch(model, bundle, units(bundle, 'train'), actual_opt, 'cpu', cfg)
        after = rng_state()
        restore_rng(start)
        scope['train'](epoch, native_loader(bundle, mil=True), native, nn.CrossEntropyLoss(),
                       scope['CrossEntropyLoss']('mean'), expected_opt)
        assert_nested(actual_trace, expected_trace, atol=8e-7)
        differences.append(maximum_difference(actual_trace, expected_trace))
        assert torch.equal(after['torch'], rng_state()['torch'])
        actual_trace.clear(); expected_trace.clear()
        restore_rng(after)
    (tmp_path/'author_mil_comparison.json').write_text(json.dumps({'epochs': 2, 'optimizer_updates': 2*len(bundle.records('train')),
        'dropout': dropout, 'instances': count, 'maximum_numeric_differences': differences,
        'scope': 'Complete fixed author MIL epoch, declared frozen-feature boundary and singleton shape repairs; CPU synthetic only'}))


@pytest.mark.parametrize('dropout', [0., .5])
def test_complete_original_image_temporal_loop_and_ema(tmp_path, dropout):
    seed_everything(19)
    model, bundle, _ = fixture(tmp_path, 'loss_attention_temporal', fixed_instances=16, dropout=dropout)
    scope = reference()
    class Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = copy.deepcopy(model.project)
            self.unused = copy.deepcopy(model.attention.linear)
            self.classifier = copy.deepcopy(model.classifier)
            self.image_attention = scope['ImageAttention']()
        def forward(self, x):
            h = self.project(x[0]).T.reshape(1, model.widths[0], 4, 4)
            context, instance, attention = self.image_attention(h, self.classifier.weight, self.classifier.bias)
            return self.classifier(F.avg_pool2d(context, 4).flatten(1)), instance, attention
    native = Native()
    opt = model.make_optimizer(.003, 0)
    model.prepare(bundle, 19, 'cpu')
    start = rng_state()
    actual, cfg = [], {'max_instances': None}
    for epoch in range(3):
        model.train(); model.start_epoch(epoch)
        trace = capture_steps(opt)
        loss_attention_epoch(model, bundle, units(bundle, 'train'), opt, 'cpu', cfg)
        model.finish_epoch(bundle, 'cpu', cfg)
        predictions = predict(model, bundle, model.profile, 'val', 'cpu', cfg)
        actual.append({'steps': copy.deepcopy(trace), 'bank': torch.stack(list(model.bank.values())),
                       'rng': rng_state()['torch'].clone(), 'error': predictions.loss_attention_error.mean()})
        # Restore step to avoid wrapping an older epoch's trace in the next one.
        opt.step = type(opt).step.__get__(opt)
    restore_rng(start)
    observers, legacy_optimizers, differences = [], [], []
    def adam_factory(*args, **kwargs):
        optimizer = scope['LegacyAdam'](*args, **kwargs)
        trace = capture_steps(optimizer); legacy_optimizers.append((optimizer, trace))
        return optimizer
    def observer(epoch, optimizer, Z, Z_now, z):
        native.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, target, index in native_loader(bundle, subset='val'):
                logits, _, _ = native(inputs)
                correct += int(logits.argmax(1) == target)
                total += 1
        assert_nested(actual[epoch]['steps'], legacy_optimizers[-1][1], atol=8e-7)
        # Source 4x4 AvgPool2d and WSI variable-bag mean use different reduction
        # layouts; compare finite float32 numerical agreement, not bitwise equality.
        torch.testing.assert_close(actual[epoch]['bank'], Z, rtol=0, atol=2e-8)
        torch.testing.assert_close(z, Z/(1-.6**(epoch+1)), rtol=0, atol=0)
        assert actual[epoch]['error'] == pytest.approx((total-correct)/total, abs=1e-15)
        assert torch.equal(actual[epoch]['rng'], rng_state()['torch'])
        differences.append({'epoch': epoch, 'max_step_abs': maximum_difference(actual[epoch]['steps'], legacy_optimizers[-1][1]),
                            'max_ema_abs': maximum_difference(actual[epoch]['bank'], Z)})
        observers.append(epoch)
    scope.update(net=native, train_datasets=SourceData(bundle), trainloader=native_loader(bundle), start_epoch=0,
        use_cuda=False, criterion=nn.CrossEntropyLoss(), weight_criterion=scope['CrossEntropyLoss']('mean'),
        Adam=adam_factory, LegacyAdam=scope['Adam'], observer=observer, epochs=3)
    definitions(scope, 'Image classification/main.py', ['train'], [
        ('.cuda()', '.to("cpu")'), ('start_epoch+200', 'start_epoch+epochs'),
        ('test(epoch)', 'observer(epoch, optimizer, Z, Z_now, z)')])
    scope['train']()
    assert observers == [0, 1, 2] and len(legacy_optimizers) == model.optimizer_resets == 3
    assert all(next(iter(opt.state.values()))['step'] == len(bundle.records('train')) for opt, _ in legacy_optimizers)
    (tmp_path/'author_temporal_comparison.json').write_text(json.dumps({'epochs': 3,
        'optimizer_updates': 3*len(bundle.records('train')), 'dropout': dropout, 'differences': differences,
        'scope': 'Complete fixed author image-temporal loop on feature-boundary inputs, legacy optimizer, source EMA and schedules; variable-bag mean vs native AvgPool2d float32 rounding is bounded; CPU synthetic only'}))


@pytest.mark.parametrize('count', [1, 3, 29])
def test_classical_equations_and_full_gradient(tmp_path, count):
    model, _, _ = fixture(tmp_path, 'loss_attention_classical', dropout=0)
    x = torch.randn(count, 4, requires_grad=True); target = torch.tensor([2])
    output = model(x); actual, _ = model.objective(output, target)
    h = model.project(x)
    instance = h@model.classifier.weight.T+model.classifier.bias
    alpha = instance.logsumexp(-1).softmax(0)
    logits = F.linear((alpha[:, None]*h).sum(0, keepdim=True), model.classifier.weight, model.classifier.bias)
    expected = F.cross_entropy(logits, target)+.1*(alpha*F.cross_entropy(instance, target.expand(count), reduction='none')).sum()
    params = [x]+list(model.project.parameters())+list(model.classifier.parameters())
    a = torch.autograd.grad(actual, params, retain_graph=True)
    b = torch.autograd.grad(expected, params)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    for first, second in zip(a, b): torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize('profile', sorted(PROFILES))
@pytest.mark.parametrize('stop_after', [1, 2])
def test_full_train_exact_resume_reload_and_instance_export(tmp_path, profile, stop_after):
    _, bundle, options = fixture(tmp_path, profile)
    cfg = {'epochs': 4, 'threads': 1, 'model': options}
    full = train(bundle.root, profile, tmp_path/'full', config=cfg)
    part = train(bundle.root, profile, tmp_path/'part', config=cfg, stop_after=stop_after)
    train(bundle.root, profile, part, config=cfg, resume=True)
    a, b = [torch.load(path/'last.pt', weights_only=False) for path in [full, part]]
    assert_nested(a['model'], b['model']); assert_nested(a['optimizers'], b['optimizers'])
    assert a['history'] == b['history']
    first = evaluate(full); csv = (full/'test_predictions.csv').read_bytes()
    assert first == evaluate(full) and csv == (full/'test_predictions.csv').read_bytes()
    protocol = json.loads((full/'method_protocol.json').read_text())
    assert protocol['optimizer_steps'] == 4*len(bundle.records('train'))
    assert protocol['optimizer_resets'] == (4 if profile.endswith('temporal') else 0)
    manifest = json.loads((full/'test_instances/manifest.json').read_text())
    for item in manifest['records']:
        assert digest_file(item['path']) == item['sha256']
        row = next(r for r in bundle.records('test') if r['slide_id'] == item['slide_id'])
        with h5py.File(item['path']) as h, h5py.File(bundle.root/row['path']) as original:
            np.testing.assert_array_equal(h['coords'][:], original['coords'][:])
            assert h['attention'].shape == (len(original['features']),)
            assert h['instance_logits'].shape == (len(original['features']), 3)
            assert 'target' not in h


@pytest.mark.parametrize('epoch', [0, 1, 79, 80, 81, 149, 150, 175, 199])
def test_native_temporal_schedule_and_reset(tmp_path, epoch):
    model, bundle, _ = fixture(tmp_path, 'loss_attention_temporal')
    model.prepare(bundle, 1, 'cpu'); opt = model.make_optimizer(.003, 0)
    opt.state[next(iter(model.parameters()))]['sentinel'] = 7
    model.start_epoch(epoch)
    source = reference(); up = source['rampup'](epoch); down = source['rampdown'](epoch)
    expected = {'lr': .003*up*down, 'betas': (down*.9+(1-down)*.5, .99 if epoch <= 80 else .999)}
    assert not opt.state and model.optimizer_resets == 1
    assert opt.param_groups[0]['lr'] == expected['lr'] and opt.param_groups[0]['betas'] == expected['betas']
    assert model.last_schedule['consistency_weight'] == (0 if epoch == 0 else 400*up)


def test_instance_length_reduction_and_shared_head_gradients(tmp_path):
    model, _, _ = fixture(tmp_path, 'loss_attention_mil_source', dropout=0)
    target = torch.tensor([1]); x = torch.randn(7, 4)
    a = model(x); b = model(x.repeat(2, 1))
    _, terms_a = model.objective(a, target); _, terms_b = model.objective(b, target)
    torch.testing.assert_close(a['logits'], b['logits'])
    torch.testing.assert_close(terms_a['weighted_instance_loss'], 2*terms_b['weighted_instance_loss'])
    loss, _ = model.objective(a, target); loss.backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.attention.linear.parameters())
    # Attention must carry gradient into W,b; detaching it changes classifier gradients.
    detached = F.cross_entropy(F.linear((a['attention'].detach()[:, None]*model.project(x)).sum(0, keepdim=True),
        model.classifier.weight, model.classifier.bias), target)
    g1 = torch.autograd.grad(detached, model.classifier.weight, retain_graph=True)[0]
    fresh = model(x)
    g2 = torch.autograd.grad(F.cross_entropy(fresh['logits'], target), model.classifier.weight)[0]
    assert not torch.allclose(g1, g2, atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_query_labels_do_not_change_scores_or_attention_banks(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile)
    model.prepare(bundle, 1, 'cpu'); model.eval()
    before = copy.deepcopy(model.get_extra_state())
    row = bundle.records('test')[0]; x, _ = bundle.load(row)
    a = model(x, label=torch.tensor([0])); b = model(x, label=torch.tensor([2]))
    for key in a: assert torch.equal(a[key], b[key])
    assert_nested(before, model.get_extra_state())
    assert model.export_instances(a, row, tmp_path/'first')['sha256'] == model.export_instances(b, row, tmp_path/'second')['sha256']


@pytest.mark.parametrize('kind', ['heldout', 'duplicate', 'short', 'incomplete', 'memory'])
def test_temporal_training_input_and_bank_contract(tmp_path, kind):
    model, bundle, _ = fixture(tmp_path, 'loss_attention_temporal', max_bank_bytes=1 if kind == 'memory' else 512*1024**2)
    if kind == 'memory':
        with pytest.raises(ValueError, match='before allocation'): model.prepare(bundle, 1, 'cpu')
        assert not model.bank
        return
    model.prepare(bundle, 1, 'cpu'); model.make_optimizer(.003, 0); model.start_epoch(0)
    row = bundle.records('test' if kind == 'heldout' else 'train')[0]
    x, _ = bundle.load(row)
    if kind == 'short': x = x[:-1]
    result = model(x); target = torch.tensor([row['target']])
    if kind == 'duplicate': model.objective(result, target, row, update_bank=True)
    with pytest.raises(ValueError, match='TRAIN|aligned|incomplete'):
        if kind == 'incomplete': model.finish_epoch(bundle, 'cpu', {})
        else: model.objective(result, target, row, update_bank=True)


@pytest.mark.parametrize('option,value', [('dropout', 1), ('dropout', True), ('hidden', 0), ('instance_weight', 0),
                                        ('instance_weight', float('inf')), ('max_bank_bytes', False)])
def test_invalid_model_configuration(tmp_path, option, value):
    with pytest.raises(ValueError): fixture(tmp_path, 'loss_attention_mil_source', **{option: value})


@pytest.mark.parametrize('options', [{'epochs': 1}, {'max_instances': 10}, {'grad_clip': 1},
                                   {'weight_decay': .01}, {'selection_metric': 'macro_auc'}])
def test_temporal_incomplete_recipe_rejected(tmp_path, options):
    _, bundle, cfg = fixture(tmp_path, 'loss_attention_temporal')
    with pytest.raises(ValueError):
        train(bundle.root, 'loss_attention_temporal', tmp_path/'run', config={'epochs': 3, 'model': cfg, **options})


def test_validation_only_selection_tie_policy_and_alias(tmp_path):
    for profile in sorted(PROFILES):
        model, _, _ = fixture(tmp_path/profile, profile)
        model.after_validation({'loss_attention_error': .2, 'loss_attention_bag_loss': .8})
        assert model.selection_improved(0, 0)
        model.after_validation({'loss_attention_error': .2, 'loss_attention_bag_loss': .7})
        assert model.selection_improved(0, 0) == profile.endswith('classical')
        model.after_validation({'loss_attention_error': .3, 'loss_attention_bag_loss': .1})
        assert not model.selection_improved(0, 0)
    assert inspect_method('Loss-based Attention')['id'] == 'loss_attention_classical'
    assert inspect_method('loss_attention')['id'] == 'loss_attention_classical'


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_nonfinite_gradient_is_rejected_before_any_update(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile)
    model.prepare(bundle, 1, 'cpu'); opt = model.make_optimizer(.003 if model.temporal else .0005, 0)
    model.start_epoch(0)
    before = [p.detach().clone() for p in model.parameters()]
    bank = copy.deepcopy(model.bank)
    hook = next(model.project.parameters()).register_hook(lambda g: torch.full_like(g, float('nan')))
    try:
        with pytest.raises(ValueError, match='gradient is nonfinite'):
            loss_attention_epoch(model, bundle, units(bundle, 'train'), opt, 'cpu', {'max_instances': None})
    finally:
        hook.remove()
    assert model.optimizer_steps == 0 and not opt.state
    for a, b in zip(before, model.parameters()): assert torch.equal(a, b)
    assert_nested(bank, model.bank)


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_finite_gradient_square_overflow_is_not_training_success(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile)
    model.prepare(bundle, 1, 'cpu'); opt = model.make_optimizer(.003 if model.temporal else .0005, 0)
    model.start_epoch(0)
    hook = next(model.project.parameters()).register_hook(lambda g: torch.full_like(g, 1e30))
    try:
        with pytest.raises(ValueError, match='nonfinite'):
            loss_attention_epoch(model, bundle, units(bundle, 'train'), opt, 'cpu', {'max_instances': None})
    finally:
        hook.remove()
    assert model.optimizer_steps == 0


def test_source_nan_optimizer_sanitization_is_not_silently_used(tmp_path):
    model, bundle, _ = fixture(tmp_path, 'loss_attention_temporal')
    model.prepare(bundle, 1, 'cpu'); opt = model.make_optimizer(.003, 0); model.start_epoch(0)
    p = next(model.project.parameters())
    opt.state[p].update(step=1, exp_avg=torch.full_like(p, float('nan')), exp_avg_sq=torch.zeros_like(p))
    before = p.detach().clone()
    with pytest.raises(ValueError, match='nonfinite optimizer state'):
        loss_attention_epoch(model, bundle, units(bundle, 'train'), opt, 'cpu', {'max_instances': None})
    assert torch.equal(p, before) and model.optimizer_steps == 0

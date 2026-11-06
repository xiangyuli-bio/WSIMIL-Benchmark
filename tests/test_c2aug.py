"""Independent source/formula references, real complete updates and recovery."""
import ast
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from nystrom_attention import NystromAttention
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from milplus.c2aug import COMMIT, PROFILE, C2AugMIL, PaperAugmentation, bag_infonce, group_kl
from milplus.c2aug_data import TrainClassBank
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def data(path, classes=2, dimension=32):
    return Bundle(create_demo(path, patients=10*classes, classes=classes, dimension=dimension))


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


def independent_source(dimension, classes):
    root, sha = verified_source('c2aug')
    assert sha == COMMIT
    code = pinned_text(root, sha, 'models/transmil.py').replace(
        'self.cls_token.expand(B, -1, -1).cuda()', 'self.cls_token.expand(B, -1, -1).to(h.device)')
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name in {'TransLayer', 'PPEG', 'TransMIL'}]
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'NystromAttention': NystromAttention}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_original_c2aug_transmil.py', 'exec'), scope)  # noqa: S102 -- actual pinned author model, independent import
    return scope['TransMIL'](dimension, classes)


@pytest.mark.parametrize('dimension', [16, 1024, 1536])
@pytest.mark.parametrize('classes', [2, 3])
def test_actual_native_source_network_outputs_loss_and_all_gradients(tmp_path, dimension, classes):
    bundle = data(tmp_path/'data', classes, dimension)
    seed_everything(7)
    model = C2AugMIL(bundle, max_views=2)
    ref = independent_source(dimension, classes)
    ref.load_state_dict(model.student.mil.state_dict())
    x = torch.randn(23, dimension, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    before = rng_state()
    actual = model(x)
    after = rng_state()
    restore_rng(before)
    expected = ref(data=other.unsqueeze(0))
    exact(actual['logits'], expected['logits'])
    exact(actual['probabilities'], expected['Y_prob'])
    exact(actual['embedding'], expected['feature'])
    exact(rng_state(), after)
    target = torch.tensor([classes-1])
    F.cross_entropy(actual['logits'], target).backward()
    F.cross_entropy(expected['logits'], target).backward()
    exact(x.grad, other.grad)
    for (n, p), (m, q) in zip(model.student.mil.named_parameters(), ref.named_parameters()):
        assert n == m
        exact(p.grad, q.grad)
    assert model.student.mil.cls_token.shape == (1, 1, 512)
    assert model.student.mil.layer1.attn.num_landmarks == 256


@pytest.mark.parametrize('format', ['h5', 'npy', 'pt'])
def test_indexed_reservoir_exact_materialized_order_no_heldout_or_current_slide(tmp_path, format):
    bundle = data(tmp_path/'data')
    if format != 'h5':
        frame = pd.read_csv(bundle.root/'features.csv')
        for i, row in frame.iterrows():
            x, _ = bundle.load(bundle.rows[bundle.rows.slide_id == row.slide_id].to_dict('records')[0])
            path = (bundle.root/row.path).with_suffix('.'+format)
            if format == 'npy':np.save(path, x.numpy())
            else:torch.save({'features': x}, path)
            frame.loc[i, 'path'] = str(path.relative_to(bundle.root))
        frame.to_csv(bundle.root/'features.csv', index=False)
        bundle = Bundle(bundle.root)
    bank = TrainClassBank(bundle)
    row = bank.rows[0]
    candidates = [r for r in bundle.records('train') if r['target'] == row['target'] and r['slide_id'] != row['slide_id']]
    reference = torch.cat([bundle.load(r)[0] for r in candidates])
    seed_everything(9)
    before = rng_state()
    indices = random.sample(range(len(reference)), 53)
    after = rng_state()
    restore_rng(before)
    exact(bank.sample(row['target'], 53, row['slide_id']), reference[indices])
    exact(rng_state(), after)
    assert set(bank.by_slide) == {r['slide_id'] for r in bundle.records('train')}
    assert bank.max_count == max(len(bundle.load(r)[0]) for r in bundle.records('train'))
    with pytest.raises(ValueError, match='TRAIN'):
        bank.sample(row['target'], 1, bundle.records('test')[0]['slide_id'])
    with pytest.raises(ValueError, match='exceeds'):
        bank.sample(row['target'], len(reference)+1, row['slide_id'])
    with pytest.raises(ValueError, match='OTHER'):
        bank.validate_views(100000)


def test_paper_original_query_fusion_independent_equation_and_gradients():
    seed_everything(22)
    module = PaperAugmentation(16).double()
    ref = copy.deepcopy(module)
    x = torch.randn(5, 16, dtype=torch.double, requires_grad=True)
    views = torch.randn(5, 3, 16, dtype=torch.double, requires_grad=True)
    a, b = x.detach().clone().requires_grad_(), views.detach().clone().requires_grad_()
    mask = torch.tensor([[False, True, False], [False, False, True], [True, False, True], [False, False, False], [True, True, False]])
    actual = module.fuse(x, views, mask)
    rows = []
    for i in range(len(a)):
        q = ref.view_q(ref.reduce(a[i]))
        keys, values = ref.view_k(ref.reduce(b[i])), ref.view_v(ref.reduce(b[i]))
        weights = (keys@q/(256**.5)).masked_fill(mask[i], -9999).softmax(0)
        rows.append(ref.recover(weights@values))
    expected = torch.stack(rows)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(x.grad, a.grad, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(views.grad, b.grad, rtol=1e-11, atol=1e-12)
    for (name, p), (_, q) in zip(module.named_parameters(), ref.named_parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad, q.grad, rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match='retain'):
        module.fuse(x, views, torch.ones_like(mask))


@pytest.mark.parametrize('length', [1, 2, 7, 17])
@pytest.mark.parametrize('ratio', [2, 16])
def test_paper_compression_short_padding_attention_and_gradients(length, ratio):
    seed_everything(8)
    module = PaperAugmentation(16).double();ref = copy.deepcopy(module)
    x = torch.randn(length, 16, dtype=torch.double, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    pad = torch.arange((-length) % ratio) % length
    actual = module.compress(x, ratio, pad)
    reduced = ref.reduce(other)
    padded = torch.cat([reduced, reduced[pad]])
    folds = padded.split(ratio)
    expected = torch.cat([ref.recover((ref.compression_q@ref.compression_k(fold).T/16).softmax(-1)@ref.compression_v(fold)) for fold in folds])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-11, atol=1e-12)
    for (name, p), (_, q) in zip(module.named_parameters(), ref.named_parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad, q.grad, rtol=1e-11, atol=1e-12)


def test_paper_all_groups_forward_kl_and_teacher_stop_gradient():
    seed_everything(77)
    x = torch.randn(8, 16, dtype=torch.double, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    teacher = torch.randn(8, 16, dtype=torch.double, requires_grad=True)
    center = torch.randn(1, 16, dtype=torch.double)*.01
    actual = group_kl(x, teacher, center, .1, .04)
    expected = torch.tensor(0., dtype=torch.double)
    for i in range(8):
        ps = F.softmax(ref[i]/.1, dim=-1)
        log_t = ((teacher[i].detach()-center[0])/.04).log_softmax(-1)
        expected = expected+(ps*(F.log_softmax(ref[i]/.1, dim=-1)-log_t)).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-12)
    actual.backward();expected.backward()
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-12, atol=1e-12)
    assert (x.grad.abs().sum(1) > 0).all() and teacher.grad is None


@pytest.mark.parametrize('negatives', [0, 1, 7])
def test_paper_bag_infonce_logsumexp_and_gradient(negatives):
    torch.manual_seed(9)
    x = F.normalize(torch.randn(1, 16, dtype=torch.double), dim=-1).requires_grad_()
    y = F.normalize(torch.randn(1+negatives, 16, dtype=torch.double), dim=-1).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    actual = bag_infonce(x, y[:1], y[1:], .07)
    logits = ref@y.detach().T/.07
    expected = logits.logsumexp(-1)[0]-logits[0, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    actual.backward();expected.backward()
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-12, atol=1e-12)
    assert y.grad is None
    if negatives == 0:assert actual.item() == 0 and not x.grad.any()


def reference_augmentation(branch, x, row, corpus, max_count, views):
    """Independent materialized TRAIN-bank trainer; never calls adapter helpers."""
    module = branch.augment
    candidates = torch.cat([tensor for slide, target, tensor in corpus
                            if target == row['target'] and slide != row['slide_id']])
    def draw(count):
        return candidates[random.sample(range(len(candidates)), count)].to(x.device)
    events = dict.fromkeys(['multiview', 'expansion', 'compression'], 0)
    if random.random() > .5:
        if random.random() > .5:
            n = random.randint(1, views)
            others = draw(n*len(x)).view(n, len(x), -1).transpose(0, 1)
            query = module.view_q(module.reduce(x)).unsqueeze(1)
            reduced = module.reduce(others)
            dots = (query*module.view_k(reduced)).sum(-1)/np.sqrt(reduced.shape[-1])
            ranks = torch.rand_like(dots).argsort(-1).argsort(-1)
            removed = torch.randint(n, (len(x), 1), device=x.device)
            weights = dots.masked_fill(ranks < removed, -9999).softmax(-1)
            x = module.recover((weights[..., None]*module.view_v(reduced)).sum(1))
            events['multiview'] += 1
        elif len(x) < max_count:
            extra = random.randint(1, max_count-len(x))
            x = torch.cat([x, draw(extra)])
            events['expansion'] += 1
    if random.random() > .5:
        reduced = module.reduce(x)
        ratio = random.randint(2, module.compression_max)
        missing = (-len(reduced)) % ratio
        pad = torch.randint(len(reduced), (missing,), device=x.device)
        folded = torch.cat([reduced, reduced[pad]]).view(-1, ratio, reduced.shape[-1])
        query = module.compression_q.expand(len(folded), -1).unsqueeze(1)
        weights = (query@module.compression_k(folded).transpose(-1, -2)/np.sqrt(folded.shape[-1])).softmax(-1)
        x = module.recover((weights@module.compression_v(folded)).squeeze(1))
        events['compression'] += 1
    return x, events


def reference_encode(branch, x):
    result = branch.mil(data=x.unsqueeze(0))
    tokens = result['contra_feature']
    membership = (F.normalize(tokens, dim=-1)@F.normalize(branch.prototypes, dim=-1).T).argmax(1)
    representations = []
    for i in range(len(branch.prototypes)):
        prototype = branch.prototypes[i]
        instances = tokens[membership == i]
        instances = prototype[None] if len(instances) == 0 else instances
        weights = (branch.group_q(prototype)[None]@branch.group_k(instances).T/(512**.5)).softmax(-1)
        representations.append(weights@branch.group_v(instances))
    bag = F.normalize(branch.bag_projection(result['feature']), dim=-1)
    groups = F.normalize(branch.group_projection(torch.cat(representations)), dim=-1)
    return result['logits'], bag, groups


def snapshot(model, optimizer):
    return {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
            'gradients': {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()},
            'rng': rng_state()}


@pytest.mark.parametrize('classes,dimension', [(2, 1024), (3, 1536)])
def test_complete_paper_recipe_every_actual_update_matches_independent_trainer(tmp_path, classes, dimension):
    bundle = data(tmp_path/'data', classes, dimension)
    seed_everything(2026)
    model = C2AugMIL(bundle, max_views=2, queue_size=7)
    ref = copy.deepcopy(model)
    for branch in [ref.student, ref.teacher]:
        native = independent_source(dimension, classes)
        native.load_state_dict(branch.mil.state_dict())
        branch.mil = native
    ref.teacher.requires_grad_(False)
    optimizer = model.make_optimizer(1e-4, 0.)
    other_optimizer = torch.optim.Adam(ref.student.parameters(), lr=1e-4)
    corpus = [(row['slide_id'], row['target'], bundle.load(row)[0]) for row in bundle.records('train')]
    max_count = max(len(x) for _, _, x in corpus)
    original = {n: p.detach().clone() for n, p in model.student.named_parameters()}
    checks = 0
    for epoch in range(2):
        model.train();ref.train()
        before = rng_state();trace = []
        rows = DataLoader(bundle.records('train'), batch_size=1, shuffle=True, generator=ref.generator, collate_fn=lambda batch: batch[0])
        for row in rows:
            other_optimizer.zero_grad(set_to_none=True)
            x, _ = bundle.load(row)
            sx, se = reference_augmentation(ref.student, x, row, corpus, max_count, 2)
            logits, zs, gs = reference_encode(ref.student, sx)
            with torch.no_grad():
                tx, te = reference_augmentation(ref.teacher, x, row, corpus, max_count, 2)
                _, zt, gt = reference_encode(ref.teacher, tx)
            index = [i for i, slide in enumerate(ref.queue_ids) if slide is not None and slide != row['slide_id']]
            negatives = ref.queue[index].detach().clone()
            ce = F.cross_entropy(logits, torch.tensor([row['target']]))
            bag_scores = zs@torch.cat([zt.detach(), negatives]).T/.07
            lb = F.cross_entropy(bag_scores, torch.tensor([0]))
            ps = (gs/.1).log_softmax(-1)
            pt = ((gt.detach()-ref.center)/.04).log_softmax(-1)
            lg = (ps.exp()*(ps-pt)).sum()
            loss = ce+lb+lg
            loss.backward();other_optimizer.step()
            with torch.no_grad():
                for old, new in zip(ref.teacher.parameters(), ref.student.parameters()):
                    old.mul_(.9999).add_(new, alpha=1-.9999)
                ref.center.mul_(.9).add_(gt.mean(0, keepdim=True), alpha=1-.9)
                ref.queue[ref.queue_pointer].copy_(zt[0])
            ref.queue_ids[ref.queue_pointer] = row['slide_id']
            ref.queue_pointer = (ref.queue_pointer+1) % 7
            ref.queue_count = min(ref.queue_count+1, 7)
            ref.optimizer_steps += 1
            for event in ref.augmentation_counts:ref.augmentation_counts[event] += se[event]+te[event]
            trace.append(snapshot(ref, other_optimizer))
        restore_rng(before)
        commit = model.commit_update
        def check_update(result, slide, _commit=commit, _trace=trace):
            nonlocal checks
            _commit(result, slide)
            exact(snapshot(model, optimizer), _trace.pop(0))
            checks += 1
        model.commit_update = check_update
        losses = model.epoch(bundle, optimizer)
        model.commit_update = commit
        assert not trace and all(np.isfinite(row['loss']) for row in losses)
        assert any(row['bag'] > 0 for row in losses) and all(row['group'] > 0 for row in losses)
        model.finish_epoch(bundle, 'cpu', {});ref.completed_epochs += 1
        exact(model.generator.get_state(), ref.generator.get_state())
    assert checks == 2*len(corpus)
    assert all(v > 0 for v in model.augmentation_counts.values())
    assert any(not torch.equal(original[n], p) for n, p in model.student.named_parameters() if n.startswith('augment.view_'))
    assert not torch.equal(original['prototypes'], model.student.prototypes)
    for prefix in ['mil.', 'group_projection.', 'bag_projection.', 'augment.compression_']:
        assert any(not torch.equal(original[n], p) for n, p in model.student.named_parameters() if n.startswith(prefix))
    assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters())
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.student.parameters()}


@pytest.mark.parametrize('boundary', [1, 2])
def test_real_engine_exact_resume_with_fifo_center_and_selection(tmp_path, boundary):
    bundle = data(tmp_path/'data')
    cfg = {'epochs': 3, 'threads': 1, 'model': {'max_views': 2, 'queue_size': 5}}
    full = train(bundle.root, PROFILE, tmp_path/'full', config=cfg)
    part = train(bundle.root, PROFILE, tmp_path/'part', config=cfg, stop_after=boundary)
    train(bundle.root, PROFILE, part, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, part]]
    for key in ['model', 'optimizers', 'history', 'rng', 'best', 'bad']:exact(a[key], b[key])
    values = [h['validation']['c2aug_val_macro_accuracy'] for h in a['history']]
    assert torch.load(full/'best.pt', weights_only=False)['epoch'] == values.index(max(values))
    assert a['model']['_extra_state']['optimizer_steps'] == 3*len(bundle.records('train'))
    assert not (full/'test_predictions.csv').exists()
    assert evaluate(full) == evaluate(part)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(part/'test_predictions.csv'))
    checksum = digest_file(full/'test_predictions.csv')
    evaluate(full)
    assert checksum == digest_file(full/'test_predictions.csv')
    protocol = read_json(full/'method_protocol.json')
    assert protocol['completed_epochs'] == 3
    assert all(v > 0 for v in protocol['augmentation_counts'].values())


def test_inference_uses_original_bag_and_never_reads_labels_or_changes_training_state(tmp_path):
    bundle = data(tmp_path/'data')
    model = C2AugMIL(bundle, max_views=2)
    model.eval()
    before = copy.deepcopy(model.state_dict());rng = rng_state()
    row = bundle.records('test')[0];x, _ = bundle.load(row)
    def forbidden(*args, **kwargs):raise AssertionError('Inference read a TRAIN reservoir')
    model.bank.sample = forbidden
    actual = model(x)
    expected = model.student.mil(data=x.unsqueeze(0))
    exact(actual['logits'], expected['logits']);exact(actual['probabilities'], expected['Y_prob'])
    exact(model.state_dict(), before);exact(rng_state(), rng)


def test_fifo_wraparound_repeated_bag_exclusion_and_checkpoint_validation(tmp_path):
    bundle = data(tmp_path/'data');model = C2AugMIL(bundle, max_views=2, queue_size=3)
    rows = bundle.records('train')
    for i in range(7):
        row = rows[i % 2]
        result = {'teacher_groups': F.normalize(torch.randn(8, 128), dim=-1),
                  'teacher_bag': F.normalize(torch.randn(1, 128), dim=-1),
                  'trace_s': dict.fromkeys(model.augmentation_counts, 0), 'trace_t': dict.fromkeys(model.augmentation_counts, 0)}
        model.commit_update(result, row['slide_id'])
    assert model.queue_count == 3 and model.queue_pointer == 1
    assert len(model.negatives(rows[0]['slide_id'])) == 1
    original = copy.deepcopy(model.get_extra_state())
    for key, value in [('queue_ids', [rows[0]['slide_id'], 'test-slide', rows[1]['slide_id']]),
                       ('queue_count', 2), ('queue_pointer', 0), ('optimizer_steps', -1), ('contract', 'changed')]:
        state = copy.deepcopy(original);state[key] = value
        with pytest.raises(ValueError):model.set_extra_state(state)
    model.set_extra_state(original)


@pytest.mark.parametrize('option,value', [('alpha', 0), ('beta', 0), ('temperature', float('inf')),
    ('teacher_temperature', -1), ('momentum', 1), ('queue_size', True), ('compression_max', 1), ('max_views', 0)])
def test_invalid_objective_and_recipe_are_not_silently_repaired(tmp_path, option, value):
    bundle = data(tmp_path/'data')
    with pytest.raises(ValueError):C2AugMIL(bundle, **{option: value})


def test_default_supplement_recipe_and_unsupported_task_contract(tmp_path):
    bundle = data(tmp_path/'data')
    task = yaml.safe_load((bundle.root/'task.yaml').read_text());task['training_unit'] = 'patient'
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='individual'):
        C2AugMIL(Bundle(bundle.root))
    cfg = yaml.safe_load(Path('configs/c2aug.yaml').read_text())
    assert cfg['epochs'] == 200 and cfg['model']['max_views'] == 8 and cfg['model']['queue_size'] == 256
    assert cfg['model']['prototypes'] == 8 and cfg['model']['reduced'] == 256 and cfg['lr'] == 1e-4


def test_empty_group_alignment_uses_own_prototype_and_remains_trainable(tmp_path):
    bundle = data(tmp_path/'data')
    model = C2AugMIL(bundle, max_views=2)
    branch = model.student.double();ref = copy.deepcopy(branch)
    x = torch.randn(1, 512, dtype=torch.double, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual = branch.align(x)
    membership = (F.normalize(other, dim=-1)@F.normalize(ref.prototypes, dim=-1).T).argmax(1).item()
    groups = []
    for i in range(8):
        q = ref.prototypes[i]
        kv = other if i == membership else q[None]
        scores = ref.group_q(q)[None]@ref.group_k(kv).T/(512**.5)
        groups.append(scores.softmax(-1)@ref.group_v(kv))
    expected = F.normalize(ref.group_projection(torch.cat(groups)), dim=-1)
    exact(actual, expected)
    actual.sum().backward();expected.sum().backward();exact(x.grad, other.grad)
    exact(branch.prototypes.grad, ref.prototypes.grad)
    assert (branch.prototypes.grad[[i for i in range(8) if i != membership]].abs().sum(1) > 0).all()


def test_expansion_feasibility_is_checked_before_any_random_training_failure(tmp_path):
    bundle = data(tmp_path/'data')
    bank = TrainClassBank(bundle)
    bank.max_count = 10000000
    with pytest.raises(ValueError, match='expansion'):
        bank.validate_views(1)

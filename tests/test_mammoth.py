"""Expert routing, complete author objectives and reproducible classification training."""
import ast
import copy
import math
import subprocess
import types
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.mammoth import HEADS, BinarySmoothTop1, MammothMIL, author_core, mammoth_epoch
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import verified_source

SMALL = {'hidden': 32, 'attention_dim': 16, 'experts': 3, 'slots': 3, 'heads': 4,
         'slot_dim': 16, 'aggregation_heads': 4, 'ilra_landmarks': 4, 'minimum_epochs': 1}


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def data(path, **kwargs):
    return Bundle(create_demo(path, patients=30 if kwargs.get('classes') == 3 else 20, **kwargs))


def equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def reference_file(source, path, scope):
    """Fresh complete author definitions, without the adapter's loader/patch pipeline."""
    root, sha = verified_source(source)
    text = subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and not (isinstance(n, ast.ClassDef) and any(isinstance(b, ast.Name) and b.id in
                      {'PreTrainedModel', 'PretrainedConfig'} for b in n.bases))]
    ns = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)  # noqa: S102 -- independent verified author definitions
    return ns


def reference_head(kind, dim, classes, model):
    from nystrom_attention import NystromAttention
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'Optional': Optional, 'Union': Union,
             'ABC': ABC, 'abstractmethod': abstractmethod, 'NystromAttention': NystromAttention,
             'Mammoth': author_core().Mammoth}
    scope.update(reference_file('mammoth', 'millab/src/models/mil_template.py', scope))
    scope.update(reference_file('mammoth', 'millab/src/models/layers.py', scope))
    if kind == 'transformer':
        scope.update(reference_file('mammoth', 'src/mammoth/components.py', {**scope, 'math': math}))
    scope = reference_file('mammoth', f'millab/src/models/{kind}.py', scope)
    name = {'abmil': 'ABMIL', 'clam': 'CLAMSB', 'dsmil': 'DSMIL', 'transmil': 'TransMIL',
            'transformer': 'Transformer', 'ilra': 'ILRA'}[kind]
    cfg = {'in_dim': dim, 'embed_dim': 32, 'num_classes': classes, 'moe_args': model.moe_args.copy()}
    if kind in {'abmil', 'dsmil'}:
        cfg.update(dropout=.25, attn_dim=16, num_fc_layers=1)
    elif kind == 'clam':
        cfg.update(dropout=.25, attention_dim=16, k_sample=8, subtyping=False, bag_weight=.7,
                   instance_loss_fn='ce', n_fc_layers=1)
    elif kind in {'transformer', 'transmil'}:
        cfg.update(num_fc_layers=1, dropout=.25, num_attention_layers=2, num_heads=4)
        if kind == 'transformer':
            cfg['encoder_mlp_dim'] = 128
    else:
        cfg.update(num_attention_layers=2, num_heads=4, topk=4, ln=True, mode='classification')
    result = scope[name](**cfg)
    result.load_state_dict(model.network.state_dict())
    return result


@pytest.mark.parametrize('profile', list(HEADS))
def test_registered_full_objective_gradients_and_repeat_inference(tmp_path, profile):
    b = data(tmp_path/'data', classes=3)
    m = MammothMIL(b, profile, **SMALL)
    x = torch.randn(27, b.dim)
    out = m(x, label=torch.tensor([2]))
    out['total_loss'].backward()
    for name, p in m.network.patch_embed.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert m.network.patch_embed.slot_embeds.grad.abs().sum() > 0
    assert m.network.patch_embed.wq.weight.grad.abs().sum() > 0
    if profile == 'mammoth_clam':
        assert m.network.instance_classifiers[2].weight.grad.abs().sum() > 0
        assert m.network.instance_classifiers[0].weight.grad is None
    m.eval()
    first = m(x, return_routing=True)
    equal(first, m(x, return_routing=True))
    assert first['logits'].shape == (1, 3)
    assert first['dispatch_weights'].shape == (1, 27, 3, 4, 3)
    torch.testing.assert_close(first['dispatch_weights'].sum(1), torch.ones(1, 3, 4, 3))
    clone = MammothMIL(b, profile, **SMALL)
    clone.load_state_dict(m.state_dict()); clone.eval()
    equal(m(x), clone(x))
    check_compatibility(require_runnable(profile), b)


@pytest.mark.parametrize('profile', [p for p in HEADS if HEADS[p] not in {'mean', 'max'}])
def test_three_complete_author_head_adamw_updates(tmp_path, profile):
    """Native objectives, gradients, parameters and optimizer moments agree after each update."""
    b = data(tmp_path/'data', classes=3)
    m = MammothMIL(b, profile, **{**SMALL, 'instance_loss': 'ce'})
    ref = reference_head(m.kind, b.dim, 3, m)
    actual = torch.optim.AdamW(m.network.parameters(), lr=1e-4, weight_decay=1e-5)
    expected = torch.optim.AdamW(ref.parameters(), lr=1e-4, weight_decay=1e-5)
    for label in [0, 2, 1]:
        x = torch.randn(29, b.dim)
        state = rng_state()
        actual.zero_grad(set_to_none=True)
        out = m(x, label=torch.tensor([label])); out['total_loss'].backward()
        after = rng_state()
        restore_rng(state)
        expected.zero_grad(set_to_none=True)
        y = m.input_dropout(x)[None]
        source, _ = ref(y, label=torch.tensor([label]), loss_fn=nn.CrossEntropyLoss())
        equal(source['loss'], out['total_loss']); equal(source['logits'], out['logits'])
        source['loss'].backward()
        for (name, p), (ref_name, q) in zip(m.network.named_parameters(), ref.named_parameters()):
            assert name == ref_name
            equal(p.grad, q.grad)
        equal(after, rng_state())
        actual.step(); expected.step()
        equal(m.network.state_dict(), ref.state_dict())
        equal(actual.state_dict(), expected.state_dict())
    write_json(tmp_path/'mammoth_author_training.json', {'state': 'passed', 'profile': profile,
        'optimizer_updates': 3, 'comparison': 'native author logits, complete loss, all gradients, AdamW weights/moments and RNG bitwise equal',
        'precision': 'fp32', 'scope': 'CPU synthetic 16-dimensional input,32hidden; no natural WSI or paper-score reproduction'})


@pytest.mark.parametrize('keep_slots', [True, False])
def test_explicit_route_factorization_and_recombination(tmp_path, keep_slots):
    m = MammothMIL(data(tmp_path/'data'), **{**SMALL, 'keep_slots': keep_slots})
    core = m.network.patch_embed.eval()
    x = torch.randn(2, 13, 16, requires_grad=True)
    out, dispatch = core(x, return_weights=True)
    q = core.norm(F.linear(x, core.wq.weight, core.wq.bias)).reshape(2, 13, 4, 4)
    slot = core.slot_norm(core.slot_embeds)
    scores = torch.einsum('bnhd,ehsd->bnehs', q, slot)
    route = scores.softmax(1)
    torch.testing.assert_close(dispatch, route)
    pooled = torch.einsum('bnhd,bnehs->bsehd', q, route).reshape(6, 3, 4, 4)
    layer = core.expert_heads.lora_layers
    # Independent per-head/per-expert matmul, no factorized layer forward call.
    expert = []
    for e in range(3):
        head = []
        for h in range(4):
            first = layer.shared_weight[h]
            bias = layer.shared_bias[h]
            head.append((pooled[:, e, h]@first+bias)@layer.expert_weights[e, h]+layer.bias[e, h])
        expert.append(torch.stack(head, 1))
    z = torch.stack(expert, 1)
    flat = z.reshape(6, 3, -1)
    z = F.normalize(flat, dim=-1)*math.sqrt(32)*core.expert_heads.norm.gamma[None]
    z = z.relu().reshape(2, 3, 3, 4, 8).permute(0, 2, 1, 3, 4).reshape(2, 9, 32)
    if not keep_slots:
        combine = scores.permute(0, 1, 3, 2, 4).reshape(2, 13, 4, 9).softmax(-1)
        z = torch.einsum('bphd,bnhp->bnhd', z.reshape(2, 9, 4, 8), combine).reshape(2, 13, 32)
    torch.testing.assert_close(out, z, rtol=1e-5, atol=1e-6)
    g1 = torch.autograd.grad(out.square().sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(z.square().sum(), x)[0]
    torch.testing.assert_close(g1, g2, rtol=3e-4, atol=2e-5)


@pytest.mark.parametrize('values', [[[1., -1.], [.3, .4], [10., 0.], [-5., 5.]], [[2., 2.]], [[100., -100.], [0., 0.]]])
def test_binary_svm_matches_pinned_original_and_gradients(values):
    ns = {'torch': torch, 'math': math, 'ag': torch.autograd, 'nn': nn, 'np': np}
    utils = reference_file('smooth_topk', 'topk/utils.py', ns)
    logsum = reference_file('smooth_topk', 'topk/polynomial/sp.py', {**ns, 'divide_and_conquer': None,
        'Multiplication': None, 'd_logS_d_expX': None})['log_sum_exp']
    functional = reference_file('smooth_topk', 'topk/functional.py', {**utils, 'log_sum_exp': logsum})
    oracle = reference_file('smooth_topk', 'topk/svm.py', {**ns, 'F': types.SimpleNamespace(**functional),
                           'detect_large': utils['detect_large']})['SmoothTop1SVM'](2)
    x = torch.tensor(values, requires_grad=True)
    y = torch.arange(len(x)) % 2
    a, b = BinarySmoothTop1()(x, y), oracle(x, y)
    torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-7)
    ga = torch.autograd.grad(a, x, retain_graph=True)[0]
    gb = torch.autograd.grad(b, x)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('kind', ['mean', 'max'])
def test_pool_semantics_not_classwise_max(tmp_path, kind):
    m = MammothMIL(data(tmp_path/'data', classes=3), 'mammoth_'+kind, **SMALL).eval()
    x = torch.randn(15, 16)
    z = m.network.patch_embed(x[None])
    instance = m.network.classifier(z)[0]
    expected = (m.network.classifier(z.mean(1)) if kind == 'mean' else
                instance[instance.flatten().argmax()//3][None])
    equal(m(x)['logits'], expected)


@pytest.mark.parametrize('profile', list(HEADS))
def test_end_to_end_resume_best_reload_and_repeat_csv(tmp_path, profile):
    b = data(tmp_path/'data')
    cfg = {'epochs': 3, 'threads': 1, 'model': SMALL}
    full = train(b.root, profile, tmp_path/'full', seed=51, config=cfg)
    part = train(b.root, profile, tmp_path/'part', seed=51, config=cfg, stop_after=1)
    assert read_json(part/'status.json')['state'] == 'interrupted'
    train(b.root, profile, part, seed=51, config=cfg, resume=True)
    a = torch.load(full/'last.pt', weights_only=False)
    c = torch.load(part/'last.pt', weights_only=False)
    equal(a['model'], c['model']); equal(a['optimizers'], c['optimizers']); equal(a['rng'], c['rng'])
    equal(a['history'], c['history'])
    assert a['model']['_extra_state']['optimizer_steps'] == 3*len(units(b, 'train'))
    assert a['optimizers'][0]['param_groups'][0]['lr'] == 0
    assert a['model']['_extra_state']['scheduler']['last_epoch'] == 3
    first = evaluate(full); sha = digest_file(full/'test_predictions.csv')
    equal(first, evaluate(full)); assert sha == digest_file(full/'test_predictions.csv')
    equal(first, evaluate(part))
    assert sha == digest_file(part/'test_predictions.csv')
    assert read_json(full/'method_protocol.json')['optimizer_steps'] == 3*len(units(b, 'train'))
    assert read_json(full/'request.json')['synthetic']


def test_sampler_full_epoch_and_exact_adamw_cosine_reference(tmp_path):
    b = data(tmp_path/'data')
    m = MammothMIL(b, **SMALL)
    ref = copy.deepcopy(m)
    cfg = {'epochs': 2, 'grad_clip': None, 'max_instances': None}
    m.configure_training(cfg); ref.configure_training(cfg)
    opt = m.make_optimizer(1e-4, 1e-5); refopt = ref.make_optimizer(1e-4, 1e-5)
    groups = units(b, 'train')
    from milplus.engine import load_unit
    for epoch in range(2):
        m.start_epoch(epoch); ref.start_epoch(epoch)
        state = rng_state()
        losses = mammoth_epoch(m, b, groups, opt, 'cpu', cfg)
        final = rng_state()
        restore_rng(state)
        labels = torch.tensor([r[0]['target'] for r in groups])
        weights = torch.tensor([1/float((labels == i).sum()) for i in labels], dtype=torch.double)
        indices = list(torch.utils.data.WeightedRandomSampler(weights, len(groups), replacement=True))
        for index, loss in zip(indices, losses):
            x, coords = load_unit(b, groups[index], 'cpu', None, True)
            target = torch.tensor([groups[index][0]['target']])
            refopt.zero_grad(set_to_none=True)
            value = ref(x, coords, target)['total_loss']
            assert value.item() == loss['loss']
            value.backward(); refopt.step(); ref.optimizer_steps += 1
        equal(final, rng_state())
        m.finish_epoch(b, 'cpu', cfg); ref.finish_epoch(b, 'cpu', cfg)
        equal(m.state_dict(), ref.state_dict()); equal(opt.state_dict(), refopt.state_dict())


@pytest.mark.parametrize('options', [
    {'hidden': 16}, {'heads': 3}, {'slot_dim': 32}, {'experts': 0}, {'slots': True}, {'auto_rank': 1},
    {'expert_dropout': 1.}, {'feature_dropout': float('nan')}, {'rank': 0}, {'max_dense_bytes': 1},
    {'instance_loss': 'anything'}, {'bag_weight': 1.}, {'precision': 'half'}, {'minimum_epochs': 0}, {'share_weights': False},
    {'hidden': 32, 'slot_dim': 16, 'experts': 10000}])
def test_invalid_native_and_budget_configs_fail(tmp_path, options):
    with pytest.raises(ValueError):
        MammothMIL(data(tmp_path/'data'), **{**SMALL, **options})


def test_native_published_dimension_and_explicit_ilra_repair(tmp_path):
    b = data(tmp_path/'data', dimension=1024)
    for name in HEADS:
        seed_everything(13)
        m = MammothMIL(b, name).eval()
        result = m(torch.randn(5, 1024), return_routing=True)
        assert result['logits'].shape == (1, 2)
        assert result['dispatch_weights'].shape == (1, 5, 30, 16, 10)
        assert m.network.patch_embed.lora_rank > 0
        assert m.hidden == (256 if name.endswith('ilra_repaired') else 512)
        assert m.slot_dim == (128 if name.endswith('ilra_repaired') else 256)
        assert m.network.patch_embed.keep_slots


def test_inference_never_requires_clam_labels_and_uses_no_targets(tmp_path):
    b = data(tmp_path/'data')
    m = MammothMIL(b, 'mammoth_clam', **SMALL).eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('Evaluation must not execute instance label losses')
    m.network.forward_instance_heads = forbidden
    assert m(torch.randn(10, 16))['logits'].shape == (1, 2)


def test_invalid_inputs_and_training_contracts(tmp_path):
    b = data(tmp_path/'data')
    m = MammothMIL(b, **SMALL)
    for x in [torch.zeros(0, 16), torch.zeros(2, 17), torch.full((2, 16), float('nan')), torch.zeros(1, 2, 16)]:
        with pytest.raises(ValueError):
            m(x)
    for label in [torch.tensor([2]), torch.tensor([1.]), torch.tensor([0, 1])]:
        with pytest.raises(ValueError):
            m(torch.ones(4, 16), label=label)
    for overrides in [{'grad_clip': 5}, {'max_instances': 10}]:
        with pytest.raises(ValueError):
            m.configure_training({'epochs': 2, 'grad_clip': None, 'max_instances': None, **overrides})
    with pytest.raises(ValueError):
        m.max_dense_bytes = 1
        m(torch.ones(4, 16))


def test_rejects_patient_training_survival_amp_and_missing_class(tmp_path):
    b = data(tmp_path/'data')
    m = MammothMIL(b, **SMALL)
    m.configure_training({'epochs': 2, 'grad_clip': None, 'max_instances': None})
    opt = m.make_optimizer(1e-4, 1e-5)
    groups = [g for g in units(b, 'train') if g[0]['target'] == 0]
    with pytest.raises(ValueError, match='every registered class'):
        mammoth_epoch(m, b, groups, opt, 'cpu', {})
    b.config['training_unit'] = 'patient'
    with pytest.raises(ValueError, match='single-slide'):
        MammothMIL(b, **SMALL)
    b.config['training_unit'] = 'slide'
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match='CUDA amp'):
            MammothMIL(b, **{**SMALL, 'precision': 'amp'})
    surv = data(tmp_path/'surv', survival=True)
    with pytest.raises(ValueError, match='classification'):
        MammothMIL(surv, **SMALL)


def test_defaults_alias_and_native_scheduler(tmp_path):
    b = data(tmp_path/'data')
    assert require_runnable('MAMMOTH')['id'] == 'mammoth_abmil'
    run = train(b.root, 'MAMMOTH', tmp_path/'run', config={'epochs': 1, 'threads': 1, 'model': SMALL})
    cfg = read_json(run/'request.json')['config']
    assert {k: cfg[k] for k in ['lr', 'weight_decay', 'patience', 'grad_clip', 'selection_metric']} == {
        'lr': 1e-4, 'weight_decay': 1e-5, 'patience': 5, 'grad_clip': None, 'selection_metric': 'log_loss'}
    m = MammothMIL(b, **{**SMALL, 'minimum_epochs': 10})
    assert m.stop_after_epoch == 9


def test_source_config_mismatch_is_explicit(tmp_path):
    root, _ = verified_source('mammoth')
    cfg = yaml.safe_load((root/'millab/src/model_configs/ilra/base_mammoth.yaml').read_text())
    assert cfg['embed_dim'] == cfg['moe_args']['slot_dim'] == 256
    b = data(tmp_path/'data', dimension=1024)
    with pytest.raises(ValueError, match='widths must differ'):
        MammothMIL(b, 'mammoth_ilra_repaired', slot_dim=256)


def test_native_nonshared_broadcast_error_is_not_hidden(tmp_path):
    m = MammothMIL(data(tmp_path/'data'), **SMALL)
    args = {**m.moe_args, 'share_lora_weights': False}
    native = author_core().Mammoth(**args)
    with pytest.raises(AssertionError, match='4 dimensional'):
        native(torch.randn(2, 13, 16))


def test_grad_scaler_overflow_skips_real_updates_and_saves_state(tmp_path):
    """CPU scaler control-flow probe; this does not claim CUDA/half-precision validation."""
    b = data(tmp_path/'data')
    m = MammothMIL(b, **SMALL)
    m.configure_training({'epochs': 1, 'grad_clip': None, 'max_instances': None})
    opt = m.make_optimizer(1e-4, 1e-5)
    m.scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    weight = next(m.parameters())
    before = weight.detach().clone()
    def overflow(*args, **kwargs):
        return {'total_loss': weight.flatten()[0]*1e38}
    m.forward = overflow
    # Two classes present; each sampled gradient overflows after scale multiplication.
    groups = [next(g for g in units(b, 'train') if g[0]['target'] == i) for i in [0, 1]]
    result = mammoth_epoch(m, b, groups, opt, 'cpu', {})
    assert all(r['amp_skipped_update'] == 1 for r in result)
    assert m.optimizer_steps == 0 and m.skipped_updates == 2
    equal(before, weight)
    assert not opt.state
    assert m.scaler.get_scale() == 16384.
    clone = MammothMIL(b, **SMALL)
    clone.scaler = torch.amp.GradScaler('cpu')
    clone.load_state_dict(m.state_dict())
    assert clone.skipped_updates == 2 and clone.scaler.get_scale() == 16384.

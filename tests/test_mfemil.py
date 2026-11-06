"""MFE-MIL original graphs, loss mixing, Adam updates and complete resumable task routes."""
import ast
import builtins
import copy
import json
import math
import subprocess
import sys
from functools import partial
from types import SimpleNamespace
from typing import Final

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from einops import rearrange, reduce
from sklearn.metrics import f1_score
from torch import einsum, nn
from torch.nn import functional as F
from torch.utils import checkpoint

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.mfemil import HEADS, MFEMIL, mfemil_epoch, timm_block
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import pinned_text, verified_source

ROUTES = [(p, s) for s in [False, True] for p in HEADS if s or p != 'mfemil_transmil']


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def raw_definitions(source, path, namespace, patches=()):
    root, sha = verified_source(source)
    code = pinned_text(root, sha, path)
    for old, new in patches:
        assert old in code
        code = code.replace(old, new)
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)  # noqa: S102 -- independent author definitions
    return scope


def original_model(decoder_dim=32, decoder_depth=1, decoder_heads=4):
    # Original timm classes are independently read; no adapter factory is called.
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'Final': Final,
             'partial': partial, 'to_2tuple': lambda x: (x, x), 'use_fused_attn': lambda: True,
             'ceil': math.ceil, 'checkpoint': checkpoint, 'rearrange': rearrange, 'reduce': reduce,
             'einsum': einsum}
    mlp = raw_definitions('timm_mfemil', 'timm/layers/mlp.py', scope)['Mlp']
    # Only zero DropPath is constructed by MFE. Class names for unrelated timm
    # definitions are resolved at definition time, without importing torchvision.
    root, sha = verified_source('timm_mfemil')
    nodes = [n for n in ast.parse(pinned_text(root, sha, 'timm/models/vision_transformer.py')).body
             if isinstance(n, ast.ClassDef) and n.name in {'Attention', 'LayerScale', 'Block'}]
    ts = {**scope, 'Mlp': mlp, 'DropPath': nn.Identity}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'timm_reference', 'exec'), ts)  # noqa: S102 -- pinned original block
    scope['Block'] = ts['Block']
    modules = {}
    for path in ['models/Mean_Max_MIL.py', 'models/ABMIL.py', 'models/CLAM.py', 'models/TransMIL.py']:
        patches = [('self.instance_loss_fn = instance_loss_fn.cuda()', 'self.instance_loss_fn = instance_loss_fn')] if path.endswith('CLAM.py') else []
        modules[path[:-3].replace('/', '.')] = SimpleNamespace(**raw_definitions('mfemil', path, scope, patches))
    def imports(name, *args, **kwargs):
        return modules[name] if name in modules else builtins.__import__(name, *args, **kwargs)
    scope['__builtins__'] = {**vars(builtins), '__import__': imports}
    raw = raw_definitions('mfemil', 'mfe_mil.py', scope,
        [('decoder_embed_dim=512', f'decoder_embed_dim={decoder_dim}'),
         ('decoder_depth=4', f'decoder_depth={decoder_depth}'),
         ('decoder_num_heads=16', f'decoder_num_heads={decoder_heads}')])
    raw['class_trainer'] = raw_definitions('mfemil', 'train_mfe.py',
        {**scope, 'f1_score': f1_score, 'calculate_error': lambda a, b: 1-a.eq(b).float().mean().item()})['train_loop']
    raw['survival_trainer'] = raw_definitions('mfemil', 'train_survival.py',
        {**scope, 'device': torch.device('cpu'), 'concordance_index_censored': lambda *args: (.5,)})['train_loop']
    raw['survival_loss'] = raw_definitions('mfemil', 'utils/survival_utils.py', scope)['NLLSurvLoss']
    return raw


def fixture(path, profile='mfemil_abmil', survival=False, **options):
    dimension = options.pop('dimension', 16)
    bundle = Bundle(create_demo(path, patients=20, dimension=dimension, survival=survival))
    model = MFEMIL(bundle, profile=profile,
                   **{'decoder_dim': 32, 'decoder_depth': 1, 'decoder_heads': 4, **options})
    model.prepare(bundle, 2026, 'cpu')
    return model, bundle


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('profile,survival', ROUTES)
def test_full_original_joint_output_losses_and_all_parameter_gradients(tmp_path, profile, survival):
    model, bundle = fixture(tmp_path/'data', profile, survival)
    raw = original_model()
    args = SimpleNamespace(task='synthetic', no_existing=True, stop_grad=False, in_dim=16, n_classes=4 if survival else 2, dropout=.25)
    source = raw['MFE_MIL'](args, feat_dim=16, shared_layers=2, n_classes=args.n_classes,
                             mil_head_type=HEADS[profile], survival=survival)
    source.load_state_dict(model.network.state_dict())
    x = torch.randn(17, 16)
    state = rng_state(); actual = model(x, joint=True)
    restore_rng(state); mse, (first, second, _) = source(x, mode='joint')
    equal_state(actual['mfe_loss'], mse)
    equal_state(actual['hazards' if survival else 'logits'], first)
    if survival:
        row = units(bundle, 'train')[0][0]
        task_a = model.objective(actual, row['time'], row['event'])
        task_b = raw['survival_loss'](alpha=0.)(first, second, model.time_label([row['time']]), torch.tensor([1-row['event']]))
    else:
        task_a = F.cross_entropy(actual['logits'], torch.tensor([1]))
        task_b = F.cross_entropy(first, torch.tensor([1]))
    loss_a, loss_b = .7*task_a+.3*actual['mfe_loss'], .7*task_b+.3*mse
    equal_state(loss_a, loss_b); loss_a.backward(); loss_b.backward()
    for (name, p), (other, q) in zip(model.network.named_parameters(), source.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            equal_state(p.grad, q.grad)
        else:
            assert name == 'feat_scale' or name.startswith(('norm.', 'mil_head.instance_classifiers.')), name
    assert model.network.decoder.mask_token.grad.abs().sum() > 0
    assert model.network.shared_layers[0][0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('length', [1, 7, 17, 64, 100])
def test_original_window_mask_exact_counts_padding_and_rng(tmp_path, length):
    model, _ = fixture(tmp_path/'data')
    source = original_model()['MFE_MIL']
    side = math.ceil(math.sqrt(length)); n = side*side
    x = torch.randn(n, 16)
    state = rng_state(); masked_a, mask_a = model.network.mask_features_window(x, .75, (side, side))
    restore_rng(state); masked_b, mask_b = source.mask_features_window(model.network, x, .75, (side, side))
    equal_state(masked_a, masked_b); equal_state(mask_a, mask_b)
    assert int(mask_a.sum()) == int(.75*n)
    assert not masked_a[mask_a].count_nonzero()
    equal_state(masked_a[~mask_a], x[~mask_a])


@pytest.mark.parametrize('length', [1, 7, 19])
def test_timm_sdpa_matches_independent_manual_decoder_block(length):
    cls = timm_block('sdpa')
    block = cls(32, 4, qkv_bias=True).double()
    other = copy.deepcopy(block)
    other.attn.fused_attn = False
    x = torch.randn(1, length, 32, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = block(x), other(y)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-11, atol=2e-11)
    for p, q in zip(block.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize('profile,survival', ROUTES)
def test_two_epochs_match_original_full_training_loop_and_adam(tmp_path, profile, survival):
    model, bundle = fixture(tmp_path/'data', profile, survival, accumulation_steps=2 if survival else 1)
    raw = original_model()
    source = copy.deepcopy(model.network)
    opt, ref_opt = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (model, source)]
    groups = [g for g in units(bundle, 'train') if len(g) == 1][:4]
    loader = []
    for rows in groups:
        row = rows[0]; x, coords = bundle.load(row)
        loader.append((x, coords, None, model.time_label([row['time']]), row['time'], torch.tensor([1-row['event']], dtype=torch.float32)) if survival else (x, torch.tensor([row['target']])))
    for epoch in range(2):
        state = rng_state()
        if survival:
            raw['survival_trainer'](epoch, source, loader, ref_opt, raw['survival_loss'](alpha=0.), .3, 2)
        else:
            raw['class_trainer'](SimpleNamespace(), epoch, source, loader, ref_opt,
                                   torch.amp.GradScaler('cuda', enabled=False), .3, 2)
        restore_rng(state)
        mfemil_epoch(model, bundle, groups, opt, 'cpu', {})
        equal_state(model.network.state_dict(), source.state_dict())
        for p, q in zip(model.network.parameters(), source.parameters()):
            equal_state(opt.state[p], ref_opt.state[q])
    assert model.optimizer_steps == (4 if survival else 8)


@pytest.mark.parametrize('profile,survival', ROUTES)
def test_all_task_routes_exact_resume_reload_and_inference_has_no_decoder(tmp_path, profile, survival):
    data = create_demo(tmp_path/'data', patients=20, survival=survival)
    config = {'epochs': 2, 'threads': 1, 'model': {'decoder_dim': 32, 'decoder_depth': 1, 'decoder_heads': 4}}
    full = train(data, profile, tmp_path/'full', config=config)
    resumed = train(data, profile, tmp_path/'resume', config=config, stop_after=1)
    train(data, profile, resumed, config=config, resume=True)
    for name in ['last', 'best']:
        a, b = [torch.load(p/(name+'.pt'), weights_only=False) for p in (full, resumed)]
        equal_state(a['model'], b['model']); equal_state(a['optimizers'], b['optimizers'])
        assert a['history'] == b['history']
    assert evaluate(full) == evaluate(resumed)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(resumed/'test_predictions.csv'))
    model, bundle = fixture(tmp_path/'other', profile, survival)
    def forbidden(*args):
        raise AssertionError('inference must not run the decoder')
    model.network.decoder.register_forward_pre_hook(forbidden)
    model.eval(); model.forward_unit(bundle, units(bundle, 'test')[0], 'cpu', {}, train=False)
    request = json.loads((full/'request.json').read_text())['config']
    assert request['lr'] == (2e-4 if survival else 1e-4)
    assert request['weight_decay'] == (1e-5 if survival else 0.)
    assert request['grad_clip'] is None


def test_native_1024_input_four_512_decoder_blocks(tmp_path):
    model, _ = fixture(tmp_path/'data', 'mfemil_mean', dimension=1024,
                        decoder_dim=512, decoder_depth=4, decoder_heads=16)
    raw = original_model(512, 4, 16)
    args = SimpleNamespace(task='synthetic', no_existing=True, stop_grad=False, in_dim=1024, n_classes=2, dropout=.25)
    source = raw['MFE_MIL'](args, feat_dim=1024, shared_layers=2, n_classes=2, mil_head_type='mean')
    source.load_state_dict(model.network.state_dict())
    x = torch.randn(7, 1024)
    state = rng_state(); a = model(x, joint=True)
    restore_rng(state); mse, (logits, _, _) = source(x, mode='joint')
    equal_state(a['mfe_loss'], mse); equal_state(a['logits'], logits)
    (.7*a['logits'].square().mean()+.3*a['mfe_loss']).backward()
    (.7*logits.square().mean()+.3*mse).backward()
    for p, q in zip(model.network.parameters(), source.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            equal_state(p.grad, q.grad)


@pytest.mark.parametrize('profile', HEADS)
def test_singleton_feature_dimension_not_collapsed(tmp_path, profile):
    model, _ = fixture(tmp_path/'data', profile, profile == 'mfemil_transmil')
    result = model(torch.randn(1, 16), joint=True)
    assert result['logits'].shape == (1, 1 if model.survival else 2)
    assert result['mfe_loss'] == 0


def test_survival_tail_window_uses_actual_count_and_fits_training_only(tmp_path):
    model, bundle = fixture(tmp_path/'data', survival=True, accumulation_steps=4)
    expected = np.quantile(bundle.rows[(bundle.rows.subset == 'train') & (bundle.rows.event == 1)].drop_duplicates('global_patient_id').time, [.25, .5, .75])
    np.testing.assert_array_equal(model.bin_edges, expected)
    groups = units(bundle, 'train')[:3]
    reference = copy.deepcopy(model)
    opt, other = [m.make_optimizer(1e-4, 0.) for m in (model, reference)]
    state = rng_state(); mfemil_epoch(model, bundle, groups, opt, 'cpu', {})
    restore_rng(state); other.zero_grad(set_to_none=True)
    for rows in groups:
        output = reference.forward_unit(bundle, rows, 'cpu', {}, train=True)
        loss = .7*reference.objective(output, rows[0]['time'], rows[0]['event'])+.3*output['mfe_loss']
        (loss/3).backward()
    other.step(); reference.optimizer_steps += 1
    equal_state(model.state_dict(), reference.state_dict())


@pytest.mark.parametrize('options', [{'alpha': 0.}, {'alpha': 1.}, {'mask_ratio': 0.},
                                   {'mask_ratio': float('nan')}, {'shared_layers': 0}, {'sub_batch': 0},
                                   {'decoder_dim': 30}, {'decoder_heads': 3}, {'decoder_depth': 0},
                                   {'attention_backend': 'auto'}, {'precision': 'auto'},
                                   {'weighted_sampling': True}, {'accumulation_steps': 2}])
def test_invalid_configuration_fails(tmp_path, options):
    with pytest.raises(ValueError):
        fixture(tmp_path/'data', **options)


def test_fail_on_nan_and_backend_change_and_truncation(tmp_path):
    model, _ = fixture(tmp_path/'data')
    with pytest.raises(ValueError, match='finite nonempty'):
        model(torch.full((7, 16), float('nan')))
    with pytest.raises(ValueError, match='max_instances'):
        model.configure_training({'max_instances': 2})
    state = model.get_extra_state(); state['attention_backend'] = 'other'
    with pytest.raises(ValueError, match='numerical backend'):
        model.set_extra_state(state)


def test_survival_sorts_each_slide_frame_and_keeps_duplicate_positions_separate(tmp_path, monkeypatch):
    model, bundle = fixture(tmp_path/'data', survival=True)
    values = {'a': (torch.tensor([[10.]*16, [11.]*16, [12.]*16]), torch.tensor([[2e6, 0.], [0., 1.], [0., 0.]])),
              'b': (torch.tensor([[20.]*16, [21.]*16]), torch.tensor([[1., 0.], [0., 0.]]))}
    monkeypatch.setattr(bundle, 'load', lambda row, require_coords=False: values[row['slide_id']])
    monkeypatch.setattr(model, 'forward', lambda x, joint=False: x)
    actual = model.forward_unit(bundle, [{'slide_id': 'b'}, {'slide_id': 'a'}], 'cpu', {})
    torch.testing.assert_close(actual[:, 0], torch.tensor([12., 10., 11., 21., 20.]), rtol=0, atol=0)


def test_survival_requires_real_coordinates_before_matrix_or_training(tmp_path):
    _, bundle = fixture(tmp_path/'data', survival=True)
    for path in bundle.rows.path:
        with h5py.File(bundle.root/path, 'a') as handle:
            del handle['coords']
    with pytest.raises(ValueError, match='real coordinates'):
        check_compatibility(require_runnable('MFE-MIL'), bundle)


def test_nonfinite_reconstruction_fails_instead_of_source_zero_replacement(tmp_path, monkeypatch):
    model, _ = fixture(tmp_path/'data')
    monkeypatch.setattr(model.network.decoder, 'forward', lambda features, mask, pos_embed: features*float('nan'))
    with pytest.raises(ValueError, match='Nonfinite MFE reconstruction'):
        model(torch.randn(17, 16), joint=True)


def test_cli_family_name_runs_checks_and_evaluates(tmp_path):
    data = create_demo(tmp_path/'data', patients=20)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'epochs': 1, 'threads': 1, 'model': {'decoder_dim': 32, 'decoder_depth': 1, 'decoder_heads': 4}}))
    for args in [['check', '--data', str(data), '--method', 'MFE-MIL', '--deep'],
                 ['run', '--data', str(data), '--method', 'MFE-MIL', '--config', str(config), '--output', str(tmp_path/'run')],
                 ['evaluate', '--run', str(tmp_path/'run')]]:
        result = subprocess.run([sys.executable, '-m', 'milplus', *args], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['n'] > 0
    assert json.loads((tmp_path/'run/request.json').read_text())['method'] == 'mfemil_abmil'

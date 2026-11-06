"""Author networks, active mechanisms, phase transitions and continuation evidence."""
import copy
import math

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.dynamic_hypergraph import DynamicHypergraphMIL, load_dyhg
from milplus.engine import evaluate, train
from milplus.frequency_causal import FrequencyCausalMIL, load_mfc
from milplus.recipes import classification_loss, classification_step
from milplus.registry import check_compatibility, require_runnable
from milplus.slide_prompt import SlidePromptMIL, load_sdfp_backbone
from milplus.stochastic import inference_rng
from milplus.upstream import load_definitions, source_root


def source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch pinned '+name+' source')
    torch.set_num_threads(1)


@pytest.mark.parametrize('dim', [16, 32])
def test_dyhg_full_author_logits_ce_gradients(dim):
    source('dyhg')
    model = DynamicHypergraphMIL(dim, 3, hidden=32, temperature=.8)
    ref = load_dyhg(singleton_fix=False).DyHG(in_dim=dim, emb_dim=32, n_classes=3,
        hyper_num=4, tau=.8, dropout=.25, num_layers=1, device='cpu')
    ref.load_state_dict(model.network.state_dict())
    x, y = torch.randn(23, dim), torch.tensor([2])
    torch.manual_seed(56); a = model(x)['logits']
    torch.manual_seed(56); b = ref(x[None])
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    F.cross_entropy(a, y).backward(); F.cross_entropy(b, y).backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), ref.named_parameters()):
        if name.startswith('fc2.'):
            assert p.grad is None and q.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert model.network.construct_hyper[0].weight.grad.abs().sum() > 0


def test_dyhg_soft_incidence_messages_singleton_and_inference_rng():
    source('dyhg')
    model = DynamicHypergraphMIL(16, 3, hidden=32, temperature=.7, inference_seed=14).eval()
    x = torch.randn(19, 16)
    state = torch.get_rng_state().clone(); numpy_state = np.random.get_state()
    first = model(x)['logits']; second = model(x)['logits']
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    net = model.network
    with inference_rng('cpu', 14):
        z = net._fc1(x[None])
        assignments = F.gumbel_softmax(net.construct_hyper(z), tau=.7, hard=False)
        edge = F.leaky_relu(assignments.transpose(1, 2) @ z)
        node = F.normalize(F.leaky_relu(assignments @ edge), dim=1)
        h = net.norm((z+node)/2).squeeze(0)
        attention, h = net.attention_net(h)
        expected = net.classifiers(attention.T.softmax(1) @ h)
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    assert torch.isfinite(model(x[:1])['logits']).all()
    raw = load_dyhg(singleton_fix=False).DyHG(in_dim=16, emb_dim=32, n_classes=3, device='cpu')
    with pytest.raises((IndexError, RuntimeError)):
        raw(x[:1][None])
    assert not torch.equal(net(x[None]), net(x[None]))  # preserve stochastic author eval


def test_mfc_native_abmil_full_author_network_and_gradient():
    source('mfcmil')
    model = FrequencyCausalMIL(16, 2, hidden=512)
    module, *_ = load_mfc('abmil', flexible=False)
    ref = module.Attention({'model': {'input_dim': 16, 'embed_dim': 512}})
    ref.load_state_dict(model.network.state_dict())
    x, target = torch.randn(17, 16), torch.tensor([1])
    left, right = model(x)['logits'], ref(x[None])['logits']
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    F.cross_entropy(left, target).backward(); F.cross_entropy(right, target).backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), ref.named_parameters()):
        if name.startswith(('fuse.', 'causal_ll.')):
            assert p.grad is None and q.grad is None
        else:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('kind', ['abmil', 'transmil'])
def test_mfc_multiclass_widths_both_memories_frequency_and_downsample_gradients(dim, kind):
    source('mfcmil')
    if kind == 'transmil':
        pytest.importorskip('nystrom_attention')
    model = FrequencyCausalMIL(dim, 3, backbone=kind, hidden=32)
    module, *_ = load_mfc(kind)
    ref = (module.Attention if kind == 'abmil' else module.TransMIL)(
        {'model': {'input_dim': dim, 'embed_dim': 32, 'num_classes': 3}})
    ref.load_state_dict(model.network.state_dict())
    x, y = torch.randn(35, dim), torch.tensor([2])
    torch.manual_seed(39); actual = model(x)['logits']
    torch.manual_seed(39); expected = ref(x[None])['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    F.cross_entropy(actual, y).backward(); F.cross_entropy(expected, y).backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), ref.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    active = ['FFP.in_proj.weight', 'FFP.out_proj.weight', 'memory.memory', 'memory_ll.memory',
              'down.down_sampling_1.weight', 'down.down_sampling_2.weight', 'down.down_sampling_3.weight',
              'causal.q.weight', 'causal.k.weight', 'causal.v.weight']
    params = dict(model.network.named_parameters())
    for name in active:
        assert params[name].grad is not None and params[name].grad.abs().sum() > 0, name
    initial = {name: params[name].detach().clone() for name in active}
    opt = torch.optim.Adam(model.parameters(), lr=.001); opt.step()
    assert all(not torch.equal(initial[name], params[name]) for name in active)
    model.eval(); ref.eval(); ref.load_state_dict(model.network.state_dict())
    torch.testing.assert_close(model(x)['logits'], ref(x[None])['logits'], atol=0, rtol=0)
    with pytest.raises(ValueError, match='at least 16'):
        model(x[:15])


def test_mfc_hilbert_real_identity_memory_write_and_registration():
    source('mfcmil')
    _, memory, scale, frequency = load_mfc()
    ff = frequency.FeatureFreProcessing(16)
    x = torch.randn(1, 19, 16, requires_grad=True)
    projected = ff.in_proj(x)
    # The last source definition is active; taking real analytic signal restores
    # a real input. Do not claim phase-only filtering absent from source forward.
    torch.testing.assert_close(ff.torch_hilbert(projected), projected, atol=4e-7, rtol=1e-5)
    torch.testing.assert_close(ff(x), ff.out_proj(projected)+x, atol=5e-7, rtol=1e-5)
    bank = memory.InstanceMemory(memory_size=4, d_model=16)
    before = bank.memory.detach().clone()
    values = x.reshape(-1, 16)
    slots = bank.memory+bank.write_weight(values).softmax(-1).T @ values
    expected = (bank.read_weight(values).softmax(-1) @ slots)[None]
    torch.testing.assert_close(bank(x), expected, atol=0, rtol=0)
    assert torch.equal(before, bank.memory)  # functional write, not persistent mutation
    bank(x).sum().backward()
    assert 'memory' in dict(bank.named_parameters()) and bank.memory.grad.abs().sum() > 0
    # Device/dtype conversion after Parameter creation can lose registration.
    assert not isinstance(nn.Parameter(torch.ones(4)).to(torch.float64), nn.Parameter)
    for n in [16, 17, 31, 32, 49]:
        hi, lo = scale.DownSample(16)(torch.randn(1, n, 16))
        assert hi.shape == (1, n, 16) and lo.shape == (1, n//16, 16)


def test_mfc_schedule_matches_author_lambda_and_optimizer():
    source('mfcmil')
    model = FrequencyCausalMIL(16, 2, hidden=16, warmup_epochs=1)
    model.configure_training({'epochs': 4}); model.steps_per_epoch = 3
    opt = model.make_optimizer(.001, .0001)
    ref = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=.001, betas=(.95, .99),
                           weight_decay=.0001, amsgrad=True)
    original = load_definitions('mfcmil', 'utils/optim/lr_scheduler.py',
        ['get_linear_schedule_with_warmup'], {'torch': torch, 'math': math})
    sched = original.get_linear_schedule_with_warmup(ref, 3, 12)
    for _ in range(12):
        assert opt.param_groups[0]['lr'] == pytest.approx(ref.param_groups[0]['lr'])
        opt.step(); model.after_step(); ref.step(); sched.step()
    assert opt.param_groups[0]['amsgrad'] and opt.param_groups[0]['betas'] == (.95, .99)
    assert opt.param_groups[0]['lr'] == pytest.approx(.000001)


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('kind', ['abmil', 'gated', 'dsmil', 'transmil'])
def test_sdfp_full_author_models_two_stages_loss_and_gradients(dim, kind):
    source('sdfp')
    if kind == 'transmil':
        pytest.importorskip('nystrom_attention')
    model = SlidePromptMIL(dim, 3, backbone=kind, hidden=None if kind == 'dsmil' else 32,
                           num_prompts=3, warmup_epochs=1)
    module = load_sdfp_backbone(kind, prune_unused=False)
    if kind in {'abmil', 'gated'}:
        ref = (module.Attention if kind == 'abmil' else module.GatedAttention)([dim, 32], 3)
    elif kind == 'dsmil':
        ref = module.MILNet(module.FCLayer(dim, 3), module.BClassifier(dim, 3))
    else:
        ref = module.TransMIL([dim, 32], 3)
    missing = ref.load_state_dict(model.network.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(name.startswith('feature_extractor_part') for name in missing.missing_keys)
    prompt_cls = load_definitions('sdfp', 'model/s_dfp.py', ['SDFP'], {'torch': torch, 'nn': nn}).SDFP
    prompt = prompt_cls(dim, 3); prompt.load_state_dict(model.prompt.state_dict())
    x, y = torch.randn(29, dim), torch.tensor([2])
    for epoch in [0, 1]:
        model.zero_grad(); ref.zero_grad(); prompt.zero_grad(); model.start_epoch(epoch)
        torch.manual_seed(56); actual = model(x, label=y)
        torch.manual_seed(56); h = prompt(x, slide_aware=epoch >= 1)
        if kind == 'dsmil':
            ins, logits, *_ = ref(h)
            expected_loss = .5*(F.cross_entropy(ins, y.expand(len(ins)))+F.cross_entropy(logits, y))
        else:
            logits = ref(data=h[None])['logits'] if kind == 'transmil' else ref(h)[0]
            expected_loss = F.cross_entropy(logits, y)
        loss = classification_loss('sdfp_'+kind, actual, y)
        torch.testing.assert_close(actual['logits'], logits, atol=0, rtol=0)
        torch.testing.assert_close(loss, expected_loss, atol=0, rtol=0)
        loss.backward(); expected_loss.backward()
        for name, p in model.network.named_parameters():
            q = dict(ref.named_parameters())[name]
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        for (name, p), (_, q) in zip(model.prompt.named_parameters(), prompt.named_parameters()):
            if epoch == 0 and name.startswith('modulation.'):
                assert p.grad is None and q.grad is None
            else:
                assert p.grad is not None and p.grad.abs().sum() > 0
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        assert all(p.grad is None for n, p in ref.named_parameters() if n.startswith('feature_extractor_part'))


@pytest.mark.parametrize('mode', ['multiply', 'add', 'prepend'])
@pytest.mark.parametrize('stage', [False, True])
def test_sdfp_prompt_equations_and_no_dropout(mode, stage):
    source('sdfp')
    model = SlidePromptMIL(16, 3, hidden=16, num_prompts=3, aggregation=mode)
    prompt = model.prompt
    x = torch.randn(11, 16, requires_grad=True)
    delta = F.linear(F.relu(F.linear(x.mean(0, keepdim=True), prompt.modulation[0].weight,
                                     prompt.modulation[0].bias)), prompt.modulation[2].weight,
                     prompt.modulation[2].bias) if stage else 0
    tokens = prompt.prompts+delta
    expected = (x*tokens.mean(0) if mode == 'multiply' else x+tokens.mean(0) if mode == 'add'
                else torch.cat([x, tokens]))
    actual = prompt(x, slide_aware=stage)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), x, retain_graph=True)[0])
    assert torch.equal(actual, prompt(x, slide_aware=stage))
    prompt.eval(); assert torch.equal(actual, prompt(x, slide_aware=stage))


def test_sdfp_joint_update_stage_boundary_and_author_cosine():
    source('sdfp')
    model = SlidePromptMIL(16, 3, hidden=32, warmup_epochs=1, prompt_lr=.001,
                           prompt_weight_decay=.003)
    model.configure_training({'epochs': 3}); model.steps_per_epoch = 3
    opt = model.make_optimizer(.1, .002)
    reference = load_definitions('sdfp', 'utils/runner_dfp.py', ['_warmup_lr', 'cosine_lr'], {'np': np})
    dummy = torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=.1)
    scheduler = reference.cosine_lr(dummy, .001, 9//5, 9)
    initial = copy.deepcopy(model.prompt.modulation.state_dict())
    for step in range(9):
        model.start_epoch(step//3)
        classification_step(model, 'sdfp_abmil', torch.randn(17, 16), None, torch.tensor([step % 3]), [opt])
        scheduler(step)
        assert all(g['lr'] == pytest.approx(dummy.param_groups[0]['lr']) for g in opt.param_groups)
        assert [g['weight_decay'] for g in opt.param_groups] == [.003, .002]
        if step < 3:
            assert all(torch.equal(initial[k], v) for k, v in model.prompt.modulation.state_dict().items())
    assert any(not torch.equal(initial[k], v) for k, v in model.prompt.modulation.state_dict().items())
    assert model.minimum_selection_epoch == 1
    assert all(v['step'] == 9 for p, v in opt.state.items() if p is model.prompt.prompts)
    assert all(v['step'] == 6 for p, v in opt.state.items() if any(p is q for q in model.prompt.modulation.parameters()))
    with pytest.raises(ValueError, match='both shared-prompt'):
        model.configure_training({'epochs': 1})


def test_sdfp_author_gated_rgb_call_is_invalid_for_feature_input():
    source('sdfp')
    raw = load_definitions('sdfp', 'model/ABMIL/model_abmil.py', ['GatedAttention'],
                           {'torch': torch, 'nn': nn, 'F': F})
    reference = raw.GatedAttention('tiny', 3)
    with pytest.raises(RuntimeError, match='Expected 3D.*4D'):
        reference(torch.randn(7, 192))
    model = SlidePromptMIL(192, 3, backbone='gated')
    assert torch.isfinite(model(torch.randn(7, 192))['logits']).all()


@pytest.mark.parametrize('method', ['dyhg', 'mfcmil_abmil', 'mfcmil_transmil',
                                   'sdfp_abmil', 'sdfp_gated', 'sdfp_dsmil', 'sdfp_transmil'])
def test_prompt_frequency_hypergraph_exact_resume_and_validation(tmp_path, method):
    source('dyhg' if method == 'dyhg' else method.split('_')[0])
    if 'transmil' in method:
        pytest.importorskip('nystrom_attention')
    root = create_demo(tmp_path/'data', patients=20)
    opts = {'hidden': 32}
    if method == 'sdfp_dsmil':
        opts = {}
    if method != 'dyhg':
        opts['warmup_epochs'] = 1
    cfg = {'epochs': 3, 'threads': 1, 'model': opts}
    full = train(root, method, tmp_path/'full', config=cfg)
    partial = train(root, method, tmp_path/'partial', config=cfg, stop_after=1)
    if method.startswith('sdfp_'):
        assert not (partial/'best.pt').exists()
        assert torch.load(partial/'last.pt', weights_only=False)['bad'] == 0
    train(root, method, partial, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    assert a['history'] == b['history']
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    torch.testing.assert_close(a['optimizers'], b['optimizers'], atol=0, rtol=0)
    assert evaluate(full) == evaluate(partial)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(partial/'test_predictions.csv'))
    if method.startswith(('sdfp_', 'mfcmil_')):
        bundle = Bundle(root); bundle.config['training_unit'] = 'patient'
        with pytest.raises(ValueError, match='individual slide'):
            check_compatibility(require_runnable(method), bundle)
    feat = root/pd.read_csv(root/'features.csv').iloc[0].path
    with h5py.File(feat, 'r+') as f:
        f['features'][:] = f['features'][:]+.1
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(full)

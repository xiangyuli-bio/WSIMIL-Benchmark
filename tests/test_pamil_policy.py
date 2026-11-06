"""Actual original PAMIL controls, complete objective/PPO and exact continuation."""
import ast
import copy
import math
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, train
from milplus.io import read_json
from milplus.pamil_policy import PAMILPolicyMIL
from milplus.pamil_policy_demo import create_pamil_policy_demo
from milplus.pamil_policy_network import COMMIT, PAPER, SOURCE, PAMILPolicyReference, paper_selection
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


def original(relative, names, extra=None):
    root, sha = verified_source('pamil_policy'); assert sha == COMMIT
    code = pinned_text(root, sha, relative)
    if relative == 'models/DPSF.py':
        code = code.replace('.cuda()', '')  # Independent CPU-only original control.
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'random': random,
             'math': math, 'rearrange': rearrange, **(extra or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_original_PAMIL_'+relative, 'exec'), scope)  # noqa: S102 -- actual fixed author source
    return SimpleNamespace(**{n: scope[n] for n in names})


def source_modules(reference):
    helper = original('utilmodule/utils.py', ['cat_msg2cluster_group', 'cosine_scheduler'])
    base = original('models/basemodel.py', ['Attention', 'AttenLayer', 'GroupsAttenLayer',
        'basedBasicLayer', 'BasedMILTransformer'], {'cat_msg2cluster_group': helper.cat_msg2cluster_group})
    fusion = original('models/SFFR.py', ['Cross_Attention', 'FusionHistoryFeatures'])
    head = original('models/classmodel.py', ['Attention', 'AttenLayer', 'GroupsMSGAttenLayer',
        'HighBasicLayer', 'ClassMultiMILTransformer'])
    policy = original('models/DPSF.py', ['Memory', 'Cat_Attention', 'ActorCritic', 'PPO'])
    env = original('utilmodule/environment.py', ['expand_data'])
    grouping = original('utilmodule/core.py', ['interpolate_probs', 'grouping'])
    args = reference.base.args
    b, f, h = base.BasedMILTransformer(args), fusion.FusionHistoryFeatures(512, 512), head.ClassMultiMILTransformer(args)
    for a, c in [(b, reference.base), (f, reference.fusion), (h, reference.head)]:
        a.load_state_dict(c.state_dict())
    p = policy.PPO(512, 512, 512, False, action_std=.5, lr=1e-5, gamma=.1, K_epochs=3, action_size=reference.bag_size)
    for a, c in [(p.policy, reference.policy), (p.policy_old, reference.policy_old)]:
        a.load_state_dict({k: v for k, v in c.state_dict().items() if k != 'action_var'})
    return SimpleNamespace(base=b, fusion=f, head=h, ppo=p, memory=policy.Memory(), env=env, grouping=grouping)


def independent_source_rollout(module, x, size):
    features = module.base.fc1(x)+module.base.absolute_pos_embed[0]
    ids = torch.arange(len(x)).reshape(1, -1, 1)
    ledger, pool, steps = module.env.expand_data(ids, features[None], action_size=size, total_steps=1)
    grouping = module.grouping.grouping(size)
    predictors, selected = [], []
    for step in range(steps):
        scores, memory = grouping.rlselectindex_grouping(module.ppo, module.memory, ledger, restart=step == 0)
        group, ledger, pool, memory = grouping.action_make_subbags(module.ppo, memory, scores, ledger, pool,
            action_size=size, restart=step == 0, delete_begin=True)
        selected.append(memory.coords_actions[-1].reshape(-1).tolist())
        _, state, memory, predictor = module.base(module.fusion, group, memory)
        predictors.append(predictor)
        module.ppo.select_action(state, memory, restart_batch=step == 0)
    value, memory = module.head(memory)
    return value, memory, predictors, selected


def reconstructed_loss(value, memory, predictors, target, stl=.075, sia=.25):
    wsl = -F.log_softmax(value['logits'], 1)[0, target]
    local = torch.stack([-F.log_softmax(v['logits'], 1)[0, target] for v in memory.results_dict]).mean()
    contrast = value['logits'].sum()*0
    for step in range(1, len(predictors)):
        for a, b in [(step, step-1), (step-1, step)]:
            prediction = predictors[a]
            token = memory.merge_msg_states[b].view(1, 512).detach()
            contrast = contrast-.5*((prediction/prediction.norm(dim=1, keepdim=True))*
                (token/token.norm(dim=1, keepdim=True))).sum()/len(predictors)
    return wsl+stl*local+sia*contrast


def test_literal_native_source_full_rollout_all_gradients_and_selection_exact():
    model = PAMILPolicyReference(1024, 2, profile=SOURCE)
    control = source_modules(model)
    x = torch.randn(1536, 1024, requires_grad=True); y = x.detach().clone().requires_grad_()
    state = rng_state()
    value = model(x, torch.tensor([1]), sia_weight=.25, stl_weight=.075)
    actual_rng = rng_state(); restore_rng(state)
    expected, memory, predictors, selected = independent_source_rollout(control, y, 512)
    loss = reconstructed_loss(expected, memory, predictors, 1)
    exact(value['logits'], expected['logits']); exact(value['probabilities'], expected['Y_prob'])
    exact(value['selected_row_ids'], selected); exact(actual_rng, rng_state())
    torch.testing.assert_close(value['loss'], loss, rtol=1e-6, atol=1e-7)
    value['loss'].backward(); loss.backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-7)
    for a, b in [(model.base, control.base), (model.fusion, control.fusion), (model.head, control.head)]:
        for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
            assert name == other
            if p.grad is None or q.grad is None:
                assert p.grad is None and q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
    assert len(selected) == 3 and any(len(set(ids)) < 512 for ids in selected)


def test_actual_train_reward_to_independent_original_full_ppo_and_main_gradients():
    model = PAMILPolicyReference(1024, 2, profile=PAPER)
    control = source_modules(model)
    captured = []
    update = model.ppo.update

    def capture(memory):
        captured.append(memory)
        update(memory)

    model.ppo.update = capture
    x = torch.randn(1537, 1024, requires_grad=True)
    value = model(x, torch.tensor([1]), sia_weight=.25, stl_weight=.075, update_policy=True)
    memory = captured[0]
    cls = memory.cls_states[-1].detach().numpy().reshape(-1)
    tokens = torch.cat([x.reshape(1, 512) for x in memory.merge_msg_states]).detach().numpy()
    expected_reward = float(int(value['prediction'][0]) == 1)-tokens@cls/(np.linalg.norm(tokens, axis=1)*np.linalg.norm(cls))
    np.testing.assert_allclose(value['rewards'].numpy(), expected_reward, rtol=1e-5, atol=1e-6)
    control.ppo.update(memory)
    for a, b in [(model.policy, control.ppo.policy), (model.policy_old, control.ppo.policy_old)]:
        for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
            assert name == other
            exact(p, q)
    exact(model.ppo.optimizer.state_dict(), control.ppo.optimizer.state_dict())
    value['loss'].backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.fusion.parameters())
    assert model.base.predictor[0].weight.grad.abs().sum() > 0
    assert value['steps'] == 4 and [len(ids) for ids in value['selected_row_ids']] == [512, 512, 512, 1]
    ids = [i for group in value['selected_row_ids'] for i in group]
    assert sorted(ids) == list(range(1537))


@pytest.mark.parametrize('scheme', ['gmss', 'ghss'])
def test_native512_group_scalar_policy_against_independent_normal_ppo(scheme):
    model = PAMILPolicyReference(1536, 3, profile=PAPER, scheme=scheme,
        input_projection='learned_full_dim_extension')
    policy = copy.deepcopy(model.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5, betas=(.9, .999))
    captured = []
    update = model.ppo.update

    def capture(memory):
        captured.append(memory)
        update(memory)

    model.ppo.update = capture
    x = torch.randn(1537, 1536, requires_grad=True)
    result = model(x, torch.tensor([2]), threshold=.4, update_policy=True)
    memory = captured[0]
    rewards, discounted = [], 0.
    for value in reversed(memory.rewards):
        discounted = value+.1*discounted
        rewards.insert(0, discounted)
    rewards = torch.cat(rewards)
    rewards = ((rewards-rewards.mean())/(rewards.std()+1e-5)).reshape(-1, 1)
    states = torch.stack(memory.merge_msg_states).detach().reshape(4, 1, 512)
    actions = torch.stack(memory.actions[1:]).detach().reshape(4, 1)
    old = torch.stack(memory.logprobs[1:]).detach().reshape(4, 1)
    for _ in range(3):
        output, _ = policy.gru(states, states.new_zeros((1, 1, 512)))
        output = output.reshape(4, 512)
        distribution = torch.distributions.Normal(policy.actor(output), policy.action_var)
        logprob, entropy, values = distribution.log_prob(actions), distribution.entropy(), policy.critic(output)
        ratio = (logprob-old).exp()
        advantage = rewards-values.detach()
        objective = (-torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
            +.5*F.mse_loss(values, rewards)-.01*entropy).mean()
        optimizer.zero_grad(); objective.backward(); optimizer.step()
    for (name, p), (other, q) in zip(model.policy.named_parameters(), policy.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    result['loss'].backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert sorted(i for group in result['selected_row_ids'] for i in group) == list(range(1537))
    assert model.policy.actor[0].weight.shape == (1, 512)


@pytest.mark.parametrize('scheme', ['liis', 'gmss', 'ghss'])
def test_full1536_input_multiclass_complete_masking_and_label_independent_inference(scheme):
    model = PAMILPolicyReference(1536, 3, profile=PAPER, scheme=scheme,
        input_projection='learned_full_dim_extension', bag_size=16).eval()
    x = torch.randn(53, 1536)
    rng = rng_state()
    with torch.no_grad():
        expected = model(x, threshold=.4)
        for label in [0, 1, 2]:
            restore_rng(rng)
            result = model(x, torch.tensor([label]), threshold=.4)
            exact(result['probabilities'], expected['probabilities'])
            exact(result['selected_row_ids'], expected['selected_row_ids'])
    assert sorted(i for group in expected['selected_row_ids'] for i in group) == list(range(53))
    assert model.base.fc1.weight.shape == (512, 1536) and model.head.fc2.weight.shape == (3, 512)


def test_actual_id_liis_interpolation_and_literal_greedy_selection_oracles():
    ids = torch.tensor([0, 2, 4, 6, 8, 10, 12])
    previous = torch.tensor([11, 1, 7])
    action = torch.tensor([[.8, .1, .5]])
    features = torch.randn(13, 512); state = torch.randn(1, 512)
    expected = np.argsort(-np.interp(ids.numpy(), [1, 7, 11], [.1, .5, .8]), kind='stable')[:3]
    exact(paper_selection(ids, features, previous, action, state, 3, 'liis', .9), torch.tensor(expected))
    rank = F.cosine_similarity(features[ids], state).argsort(descending=True, stable=True)
    exact(paper_selection(ids, features, previous, torch.ones(1, 1), state, 3, 'gmss', .9), rank[:3])
    chosen = paper_selection(ids, features, previous, torch.ones(1, 1), state, 3, 'ghss', .9)
    assert chosen.shape == (3,) and len(chosen.unique()) == 3 and chosen[:2].tolist() == rank[:2].tolist()


@pytest.mark.parametrize('profile,scheme', [(PAPER, 'liis'), (SOURCE, 'liis'), (PAPER, 'gmss'), (PAPER, 'ghss')])
def test_single_row_final_groups_scalar_actions_and_finite_actual_policy_update(profile, scheme):
    model = PAMILPolicyReference(512, 2, profile=profile, scheme=scheme, bag_size=16)
    value = model(torch.randn(1, 512), torch.tensor([0]), update_policy=True)
    assert value['steps'] == 1 and torch.isfinite(value['loss'])
    assert all(torch.isfinite(x).all() for x in model.policy.state_dict().values())
    assert all(int(x['step']) == 3 for x in model.ppo.optimizer.state.values())


def test_paper_fused_history_formula_and_released_raw_route_are_distinct():
    paper = PAMILPolicyReference(1024, 2, profile=PAPER, bag_size=16).eval()
    source = PAMILPolicyReference(1024, 2, profile=SOURCE, bag_size=16).eval()
    source.load_state_dict(paper.state_dict())
    raw = [torch.randn(1, 1, 512) for _ in range(3)]
    previous = torch.randn(1, 1, 512)
    memory = SimpleNamespace(msg_states=[raw[0], raw[1]], merge_msg_states=[previous, raw[1].clone()])
    x = torch.cat([previous, raw[1]], dim=1)
    module = paper.fusion.merge_catmsg_selfatten
    q, k, v = [t.reshape(1, 2, 8, 64).transpose(1, 2) for t in module.to_qkv(x).chunk(3, -1)]
    expected = module.to_out(((q@k.transpose(-1, -2)/8).softmax(-1)@v).transpose(1, 2).reshape(1, 2, 512))[:, -1:]
    paper.fusion.SFFR((None, [raw[1][None]], 1), memory)
    torch.testing.assert_close(memory.merge_msg_states[-1], expected, rtol=1e-6, atol=1e-7)
    literal = SimpleNamespace(msg_states=[raw[0], raw[1]], merge_msg_states=[previous, raw[1].clone()])
    source.fusion.SFFR((None, [raw[1][None]], 1), literal)
    assert not torch.equal(literal.merge_msg_states[-1], expected)


@pytest.mark.parametrize('profile', [PAPER, SOURCE])
def test_complete_three_epoch_both_optimizers_bitwise_resume_and_selected_reload(tmp_path, profile):
    data = create_pamil_policy_demo(tmp_path/'data')
    cfg = {'epochs': 3, 'threads': 1, 'model': {'bag_size': 16}}
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    train(data, profile, full, seed=9, config=cfg)
    train(data, profile, resumed, seed=9, config=cfg, stop_after=1)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(data, profile, resumed, seed=9, config=cfg, resume=True)
    a, b = [torch.load(path/'last.pt', map_location='cpu', weights_only=False) for path in [full, resumed]]
    for key in ['model', 'optimizers', 'history', 'best', 'bad', 'rng']:
        exact(a[key], b[key])
    extra = a['model']['_extra_state']
    count = len(Bundle(data).records('train'))
    assert extra['completed_epochs'] == 3 and extra['main_updates'] == 3*count and extra['policy_updates'] == 9*count
    assert len(a['optimizers']) == 2
    assert all(v['step'] == 9*count for v in a['optimizers'][1]['state'].values())
    assert not (full/'test_predictions.csv').exists()
    evaluate(full); before = (full/'test_predictions.csv').read_bytes()
    evaluate(full); assert (full/'test_predictions.csv').read_bytes() == before
    protocol = read_json(full/'method_protocol.json')
    assert protocol['native_hidden'] == 512 and protocol['policy_hidden'] == 512


def test_reference_cosine_schedule_and_disjoint_main_policy_optimizer_ownership(tmp_path):
    bundle = Bundle(create_pamil_policy_demo(tmp_path/'data')); model = PAMILPolicyMIL(bundle)
    cfg = {'epochs': 300, 'max_instances': None, 'grad_clip': None, 'selection_metric': 'macro_auc'}
    model.configure_training(cfg); optimizers = model.make_optimizers(1e-4, 1e-5)
    helper = original('utilmodule/utils.py', ['cosine_scheduler'])
    for epoch in [0, 100, 299]:
        model.start_epoch(epoch); model.main_updates = epoch*len(bundle.records('train'))
        sia, stl, threshold = model.coefficients()
        assert sia == helper.cosine_scheduler(.5, 0., 300, 1)[epoch]
        assert stl == helper.cosine_scheduler(0., .1, 300, 1)[epoch]
        assert threshold == .9*(epoch*len(bundle.records('train'))/(300*len(bundle.records('train'))-1))
    model.start_epoch(0); model.main_updates = 1
    # Multiplying a normalized iteration and dividing its scaled numerator
    # differ by one rounding unit. Check the independent formula to near
    # machine precision, while keeping exact checks for the endpoints below.
    assert math.isclose(model.coefficients()[2], .9/(300*len(bundle.records('train'))-1),
                        rel_tol=2e-16, abs_tol=0)
    model.main_updates = 300*len(bundle.records('train'))-1
    assert model.coefficients()[2] == .9
    coarse = PAMILPolicyMIL(bundle, threshold_schedule='epoch_reference')
    coarse.configure_training(cfg); coarse.start_epoch(100)
    assert coarse.coefficients()[2] == .9*(100/299)
    ids = [{id(p) for group in optimizer.param_groups for p in group['params']} for optimizer in optimizers]
    assert not ids[0] & ids[1]
    assert ids[1] == {id(p) for p in model.core.policy.parameters()}
    assert not ({id(p) for p in model.core.policy_old.parameters()} & (ids[0] | ids[1]))
    assert isinstance(optimizers[0], torch.optim.Adamax) and isinstance(optimizers[1], torch.optim.Adam)


def test_checkpoint_rejects_wrong_policy_sync_horizon_and_train_identity(tmp_path):
    bundle = Bundle(create_pamil_policy_demo(tmp_path/'data')); model = PAMILPolicyMIL(bundle, bag_size=16)
    model.configure_training({'epochs': 3, 'max_instances': None, 'grad_clip': None, 'selection_metric': 'macro_auc'})
    state = model.state_dict()
    name = 'core.policy_old.actor.0.bias'
    corrupted = {**state, name: state[name]+1}
    with pytest.raises(ValueError, match='synchronized'):
        model.load_state_dict(corrupted)
    extra = copy.deepcopy(state['_extra_state']); extra['total_epochs'] = 4
    with pytest.raises(ValueError, match='horizon'):
        model.load_state_dict({**state, '_extra_state': extra})
    extra = copy.deepcopy(state['_extra_state']); extra.update(epoch_index=0, completed_epochs=1,
        main_updates=len(bundle.records('train')), policy_updates=3*len(bundle.records('train')),
        train_orders=[[bundle.records('test')[0]['slide_id']]*len(bundle.records('train'))])
    with pytest.raises(ValueError, match='TRAIN'):
        model.load_state_dict({**state, '_extra_state': extra})


def test_missing_row_order_wrong_input_modes_and_memory_never_fallback(tmp_path):
    data = create_pamil_policy_demo(tmp_path/'data'); bundle = Bundle(data)
    model = PAMILPolicyMIL(bundle, max_dense_bytes=1)
    with pytest.raises(ValueError, match='max_dense_bytes'):
        model.prepare(bundle, 9, 'cpu')
    for change in [{'max_instances': 5}, {'grad_clip': 5}, {'selection_metric': 'accuracy'}]:
        with pytest.raises(ValueError):
            model.configure_training({'epochs': 3, 'max_instances': None, 'grad_clip': None,
                'selection_metric': 'macro_auc', **change})
    path = data/'task.yaml'; cfg = yaml.safe_load(path.read_text()); cfg['encoders']['toy'].pop('row_order_provenance')
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='row_order_provenance'):
        PAMILPolicyMIL(Bundle(data))
    from milplus.registry import check_compatibility, inspect_method
    cfg['encoders']['toy']['row_order_provenance'] = '   '
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='row_order_provenance'):
        check_compatibility(inspect_method('pamil_policy'), Bundle(data))


def test_changed_actual_feature_bytes_refused_for_evaluation(tmp_path):
    data = create_pamil_policy_demo(tmp_path/'data'); run = tmp_path/'run'
    train(data, PAPER, run, config={'epochs': 1, 'threads': 1, 'model': {'bag_size': 16}})
    bundle = Bundle(data); path = data/bundle.records('test')[0]['path']
    with h5py.File(path, 'r+') as f:
        f['features'][0] += .1
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(run)


def test_cli_alias_and_pamil_prototype_family_not_shadowed(tmp_path):
    from milplus.cli import dispatch, parser
    from milplus.registry import inspect_method

    cfg = parser().parse_args(['demo', '--pamil-policy', '--patients', '20', '--output', str(tmp_path/'cli')])
    bundle = Bundle(dispatch(cfg)['created'])
    assert bundle.dim == 1024 and bundle.config['synthetic'] and bundle.encoder['row_order_provenance']
    for name in ['PAMIL dynamic policy', 'pamil_policy']:
        assert inspect_method(name)['id'] == PAPER
    assert inspect_method('pamil_prototype_repaired')['family'] != 'pamil_policy'
    for other in ['--focus', '--ins', '--raw-images', '--survival']:
        with pytest.raises(ValueError, match='mixed input'):
            dispatch(parser().parse_args(['demo', '--pamil-policy', other, '--output', str(tmp_path/'bad')]))

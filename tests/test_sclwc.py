"""Original source, independent published objectives and complete recipe gates."""
import ast
import copy
import random

import numpy as np
import pytest
import torch
import yaml
from scipy.special import logsumexp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file, read_json
from milplus.sclwc import PAPER_PROFILE, SCLWC, SOURCE_PROFILE, SlideDataset, sclwc_epoch
from milplus.sclwc_demo import create_sclwc_demo
from milplus.sclwc_network import COMMIT, SCLPaperReference
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
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def data(path, dimension=32, classes=3):
    return Bundle(create_sclwc_demo(path, patients=10*classes, dimension=dimension, classes=classes))


def definitions(relative, names, scope):
    root, sha = verified_source('sclwc')
    assert sha == COMMIT
    tree = ast.parse(pinned_text(root, sha, relative))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    if relative == 'utils/loss.py' and 'loss_bg' in names:
        # The original unused binary default allocates CUDA at definition time.
        # Supply explicit targets and change only that default, never the body.
        bg = next(n for n in nodes if n.name == 'loss_bg')
        assert len(bg.args.defaults) == 1 and '.cuda()' in ast.unparse(bg.args.defaults[0])
        bg.args.defaults = [ast.copy_location(ast.Constant(None), bg.args.defaults[0])]
    namespace = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_original_'+relative, 'exec'), namespace)  # noqa: S102 -- independent fixed Git definitions
    return namespace


def original_model(dimension=768, classes=2, k=8):
    initialize = definitions('utils/utils.py', ['initialize_weights'], {'nn': nn})['initialize_weights']
    memory = definitions('utils/memory.py', ['Memory'], {'torch': torch, 'nn': nn, 'random': random})['Memory']
    contrast = definitions('utils/loss.py', ['SupConLoss'], {'torch': torch, 'nn': nn, 'F': F})['SupConLoss']
    source = definitions('models/public_models/SCL.py', ['Att_Head', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'SCL'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'initialize_weights': initialize,
         'Memory': memory, 'SupConLoss': contrast})['SCL'](n_classes=classes, k_sample=k)
    if dimension != 768:
        source.attention_net[0] = nn.Linear(dimension, 512)
    return source


@pytest.mark.parametrize('dimension', [768, 1024, 1536])
def test_native_cda_original_outputs_instances_input_and_parameter_gradients(dimension):
    seed_everything(51)
    model = SCLPaperReference(dimension, classes=3)
    original = original_model(dimension, classes=3)
    original.attention_net[0].load_state_dict(model.projector[0].state_dict())
    original.attention_net[3].load_state_dict(model.attention.state_dict())
    original.classifiers.load_state_dict(model.bag_classifiers.state_dict())
    original.instance_classifiers.load_state_dict(model.instance_classifiers.state_dict())
    x = torch.randn(96, dimension, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    before = rng_state()
    actual = model(x, label=1, slide_id='train_a', epoch=0)
    after = rng_state()
    restore_rng(before)
    expected = original(z, label=torch.tensor([1]), instance_eval=True)
    exact(actual['logits'], expected['bag_logits'])
    exact(actual['attention_raw'], expected['attention_raw'])
    exact(actual['pooled'], expected['M'])
    exact(actual['terms']['instance'], expected['inst_loss'])
    exact(rng_state(), after)
    actual_cda = .8*actual['terms']['mil']+.2*actual['terms']['instance']
    expected_cda = .8*F.cross_entropy(expected['bag_logits'], torch.tensor([1]))+.2*expected['inst_loss']
    actual_cda.backward()
    expected_cda.backward()
    exact(x.grad, z.grad)
    for ours, theirs in [(model.projector, original.attention_net[:3]),
            (model.attention, original.attention_net[3]), (model.bag_classifiers, original.classifiers),
            (model.instance_classifiers, original.instance_classifiers)]:
        for p, q in zip(ours.parameters(), theirs.parameters()):
            exact(p.grad, q.grad)
    assert model.projector[0].out_features == 512


def test_pnm_source_loss_helpers_and_gate_derivatives():
    model = SCLPaperReference(32, classes=3, dropout=False)
    result = model(torch.randn(24, 32), label=2, slide_id='train')
    source = definitions('utils/loss.py', ['loss_fg', 'loss_bg'], {'torch': torch, 'F': F, 'np': np})
    assert callable(source['loss_fg']) and callable(source['loss_bg'])
    h, gate = result['features'], result['pnm_gate']
    pos, neg = (gate*h).sum(0)/len(h), ((1-gate)*h).sum(0)/len(h)
    torch.testing.assert_close(pos+neg, h.mean(0), rtol=1e-6, atol=1e-6)
    p = model.pnm_classifier(pos).log_softmax(-1)
    n = model.pnm_classifier(neg).log_softmax(-1)
    expected = -p[2]-n[0]
    actual = result['terms']['pnm_positive']+result['terms']['pnm_negative']
    torch.testing.assert_close(actual, expected)
    author_loss = source['loss_fg'](model.pnm_classifier(pos)[None], F.one_hot(torch.tensor([2]), 3).float())
    author_loss += source['loss_bg'](model.pnm_classifier(neg)[None], F.one_hot(torch.tensor([0]), 3).float())
    torch.testing.assert_close(actual, author_loss, rtol=0, atol=1e-7)
    derivative = torch.autograd.grad(actual, model.pnm_gate.fc1.weight, retain_graph=True)[0]
    independent = torch.autograd.grad(expected, model.pnm_gate.fc1.weight)[0]
    torch.testing.assert_close(derivative, independent)
    assert derivative.norm() > 0


def test_equation7_numpy_live_gradient_and_current_slide_exclusion():
    model = SCLPaperReference(32, classes=3, warmup_epochs=0, dropout=False)
    xs = [torch.randn(24, 32) for _ in range(5)]
    for i, y in enumerate([0, 1, 2, 1, 0]):
        model.enqueue(model(xs[i], label=y, slide_id=str(i)))
    result = model(xs[1], label=1, slide_id='1')
    records = [r for bank in model.banks.values() for r in bank if r['slide_id'] != '1']
    anchors = result['features'][result['selected_top']].detach().numpy().copy()
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    keys = np.concatenate([r['features'].numpy() for r in records])
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    logits = anchors@keys.T/.07
    target = np.repeat([r['label'] == 1 for r in records], 8)
    expected = -(logits-logsumexp(logits, axis=1, keepdims=True))[:, target].mean(1).sum()
    np.testing.assert_allclose(result['terms']['wscl'].detach(), expected, rtol=2e-6, atol=2e-5)
    gradient = torch.autograd.grad(result['terms']['wscl'], model.projector[0].weight)[0]
    assert result['contrastive_info']['active'] and gradient.norm() > 0
    assert len(model.banks['positive_1']) == 2 and len(model.banks['positive_2']) == 1
    assert result['contrastive_info']['positive_bags'] == 1
    assert all(not r['features'].requires_grad for bank in model.banks.values() for r in bank)


def test_real_bank_warmup_coldstart_fifo_and_inference_immutability():
    model = SCLPaperReference(16, classes=3, positive_capacity=16, negative_capacity=16,
                             hard_negative_capacity=16, dropout=False)
    x = torch.randn(24, 16)
    cold = model(x, label=1, slide_id='a', epoch=0)
    assert cold['terms']['wscl'] == 0 and not cold['contrastive_info']['active']
    assert not any(model.banks.values())
    for slide, y in [('a', 1), ('b', 1), ('c', 1), ('d', 0)]:
        model.enqueue(model(x, label=y, slide_id=slide, epoch=0))
    assert [r['slide_id'] for r in model.banks['positive_1']] == ['b', 'c']
    assert not model(x, label=1, slide_id='e', epoch=4)['contrastive_info']['active']
    assert model(x, label=1, slide_id='e', epoch=5)['contrastive_info']['active']
    before = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        model(x)
    exact(model.state_dict(), before)


@pytest.mark.parametrize('profile,classes', [(PAPER_PROFILE, 2), (PAPER_PROFILE, 3), (SOURCE_PROFILE, 2)])
def test_complete_engine_exact_epoch_resume_and_selected_checkpoint_reload(tmp_path, profile, classes):
    bundle = data(tmp_path/'data', classes=classes)
    cfg = {'epochs': 7, 'threads': 1}
    full = train(bundle.root, profile, tmp_path/'full', seed=77, config=cfg)
    partial = train(bundle.root, profile, tmp_path/'partial', seed=77, config=cfg, stop_after=3)
    assert read_json(partial/'status.json')['state'] == 'interrupted'
    train(bundle.root, profile, partial, seed=77, config=cfg, resume=True)
    a = torch.load(full/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(partial/'last.pt', map_location='cpu', weights_only=False)
    exact(a['model'], b['model'])
    exact(a['optimizers'], b['optimizers'])
    exact(a['history'], b['history'])
    exact(a['rng'], b['rng'])
    assert not (full/'test_predictions.csv').exists()
    expected = evaluate(full)
    sha = digest_file(full/'test_predictions.csv')
    assert evaluate(full) == expected and digest_file(full/'test_predictions.csv') == sha
    assert evaluate(partial) == expected and digest_file(partial/'test_predictions.csv') == sha
    protocol = read_json(full/'method_protocol.json')
    assert protocol['optimizer_updates'] == 7*len(bundle.records('train'))
    if profile == PAPER_PROFILE:
        assert any(r['train']['wscl_active'] > 0 for r in a['history'][5:])
        assert all(r['train']['pnm_active'] == 1 for r in a['history'])
    else:
        assert all(r['train']['wscl'] == 0 and r['train']['pnm_active'] == 0 for r in a['history'])


def test_literal_actual_author_epoch_all_updates_weights_queues_and_rng(tmp_path):
    bundle = data(tmp_path/'data', classes=2, dimension=768)
    model = SCLWC(bundle, profile=SOURCE_PROFILE)
    cfg = {'max_instances': None, 'grad_clip': None, 'selection_metric': 'macro_auc', 'epochs': 7, 'patience': 20}
    model.configure_training(cfg)
    optimizer = model.make_optimizer(.0003, .00001)
    original = original_model(768, 2, 5)
    original.load_state_dict(model.network.state_dict())
    other_optimizer = torch.optim.Adam(original.parameters(), lr=.0003, weight_decay=1e-5)
    # train.py constructs cosine before the first epoch, adding initial_lr.
    torch.optim.lr_scheduler.CosineAnnealingLR(other_optimizer, T_max=50, eta_min=1e-6)
    rows = bundle.records('train')
    counts = {c: sum(r['target'] == c for r in rows) for c in range(2)}
    weights = [len(rows)/counts[r['target']] for r in rows]
    reference_loader = DataLoader(SlideDataset(bundle, rows), batch_size=None,
        sampler=WeightedRandomSampler(weights, len(weights)), num_workers=0)
    original_loop = definitions('training_methods/SCL.py', ['train_loop'], {'torch': torch, 'tqdm': tqdm})['train_loop']
    model.start_epoch(6)
    before = rng_state()
    trace = sclwc_epoch(model, bundle, units(bundle, 'train'), optimizer, 'cpu', cfg)
    after = rng_state()
    restore_rng(before)
    original_loop(6, original, reference_loader, other_optimizer, None, None)
    exact(model.network.state_dict(), original.state_dict())
    exact(optimizer.state_dict(), other_optimizer.state_dict())
    exact(rng_state(), after)
    assert len(trace) == len(rows) and all(r['forward_epoch'] == 0 and r['wscl'] == 0 for r in trace)
    assert all(p.grad is None for p in model.network.att_head.parameters())


def test_checkpoint_rejects_nontrain_memory_wrong_label_and_recipe(tmp_path):
    bundle = data(tmp_path/'data')
    model = SCLWC(bundle)
    row = bundle.records('train')[0]
    x, _ = bundle.load(row)
    model.network.enqueue(model.network(x, label=row['target'], slide_id=row['slide_id']))
    state = copy.deepcopy(model.state_dict())
    bank = next(v for v in state['network._extra_state']['banks'].values() if v)
    bank[0]['slide_id'] = bundle.records('test')[0]['slide_id']
    with pytest.raises(ValueError, match='held-out'):
        model.load_state_dict(state)
    state = copy.deepcopy(model.state_dict())
    bank = next(v for v in state['network._extra_state']['banks'].values() if v)
    bank[0]['label'] = 99
    with pytest.raises(ValueError, match='mislabelled'):
        model.load_state_dict(state)
    altered = SCLWC(bundle, temperature=.1)
    with pytest.raises(ValueError, match='contract'):
        altered.load_state_dict(model.state_dict())


def test_sampler_exact_inverse_frequency_replacement_and_no_heldout(tmp_path):
    bundle = data(tmp_path/'data')
    model = SCLWC(bundle)
    cfg = {'max_instances': None, 'grad_clip': None, 'selection_metric': 'macro_auc', 'epochs': 7, 'patience': 20}
    model.configure_training(cfg)
    optimizer = model.make_optimizer(.001, .00001)
    seen = []
    original = model.network.enqueue

    def traced(result):
        seen.append(result['slide_id'])
        original(result)

    model.network.enqueue = traced
    rows = bundle.records('train')
    counts = {c: sum(r['target'] == c for r in rows) for c in range(3)}
    weights = [len(rows)/counts[r['target']] for r in rows]
    before = rng_state()
    expected = [b['slide_id'] for b in DataLoader(SlideDataset(bundle, rows), batch_size=None,
        sampler=WeightedRandomSampler(weights, len(weights)), num_workers=0)]
    restore_rng(before)
    model.start_epoch(0)
    sclwc_epoch(model, bundle, units(bundle, 'train'), optimizer, 'cpu', cfg)
    assert seen == expected and len(seen) == len(rows)
    assert set(seen) <= set(model.allowed_train)


def test_scheduler_earlystop_first_tie_and_original_minimum_epoch(tmp_path):
    model = SCLWC(data(tmp_path/'data'))
    cfg = {'max_instances': None, 'grad_clip': None, 'selection_metric': 'macro_auc', 'epochs': 100, 'patience': 20}
    model.configure_training(cfg)
    optimizer = model.make_optimizer(.001, .00001)
    reference = torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=.001, weight_decay=1e-5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(reference, T_max=50, eta_min=1e-6)
    for epoch in range(32):
        model.start_epoch(epoch)
        optimizer.step()
        reference.step()
        model.after_validation({'macro_auc': .5})
        if epoch <= 30:
            cosine.step()
        exact(model.scheduler.state_dict(), cosine.state_dict())
        assert model.training_done == (epoch == 31)
    assert model.bad == 31 and model.best == .5 and model.completed_epochs == 32


@pytest.mark.parametrize('issue', ['missing_normal', 'patient_eval', 'patient_train', 'nonbinary_control'])
def test_invalid_input_protocol_is_rejected(tmp_path, issue):
    bundle = data(tmp_path/'data')
    path = bundle.root/'task.yaml'
    cfg = yaml.safe_load(path.read_text())
    profile = PAPER_PROFILE
    if issue == 'missing_normal':
        del cfg['sclwc']
    elif issue == 'patient_eval':
        cfg['evaluation_unit'] = 'patient'
    elif issue == 'patient_train':
        cfg['training_unit'] = 'patient'
    else:
        profile = SOURCE_PROFILE
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        SCLWC(Bundle(bundle.root), profile=profile)


def test_short_bag_wrong_full_dimension_and_inactive_recipe_rejected(tmp_path):
    model = SCLPaperReference(16)
    for x in [torch.randn(7, 16), torch.randn(15, 16), torch.randn(24, 15), torch.full((24, 16), float('nan'))]:
        with pytest.raises(ValueError):
            model(x)
    wrapper = SCLWC(data(tmp_path/'data'))
    base = {'max_instances': None, 'grad_clip': None, 'selection_metric': 'macro_auc', 'epochs': 6, 'patience': 20}
    for change in [{'max_instances': 8}, {'grad_clip': 5}, {'selection_metric': 'accuracy'}, {'epochs': 5}]:
        with pytest.raises(ValueError):
            wrapper.configure_training({**base, **change})


def test_cli_data_and_method_alias_dispatch(tmp_path):
    from milplus.cli import dispatch, parser
    from milplus.registry import require_runnable
    # Exercise the same parser/dispatch used by the installed command.
    args = parser().parse_args(['demo', '--sclwc', '--patients', '30', '--classes', '3', '--output', str(tmp_path/'data')])
    dispatch(args)
    bundle = Bundle(tmp_path/'data')
    assert bundle.dim == 768 and bundle.config['evaluation_unit'] == 'slide'
    assert require_runnable('SCL-WC')['id'] == PAPER_PROFILE
    with pytest.raises(ValueError, match='separate'):
        dispatch(parser().parse_args(['demo', '--sclwc', '--deepgraphsurv', '--output', str(tmp_path/'bad')]))


def test_explicit_normal_class_can_be_nonzero_in_multiclass_paper_route(tmp_path):
    bundle = data(tmp_path/'data')
    path = bundle.root/'task.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['sclwc']['negative_class'] = 'class_2'
    path.write_text(yaml.safe_dump(cfg))
    model = SCLWC(Bundle(bundle.root))
    assert model.negative_class == 2 and set(model.network.banks) == {'positive_0', 'positive_1', 'negative', 'hard_negative'}
    row = next(row for row in model.rows if row['target'] == 2)
    x, _ = model.bundle.load(row)
    result = model.network(x, label=2, slide_id=row['slide_id'])
    expected = F.cross_entropy(model.network.pnm_classifier(result['pnm_negative'])[None], torch.tensor([2]))
    torch.testing.assert_close(result['terms']['pnm_negative'], expected)
    model.network.enqueue(result)
    assert len(model.network.banks['negative']) == len(model.network.banks['hard_negative']) == 1
    assert not model.network.banks['positive_0'] and not model.network.banks['positive_1']


def test_endpoint_reference_requires_provenance_and_runs_complete_recipe_without_normal_claim(tmp_path):
    bundle = data(tmp_path/'data', classes=2)
    path = bundle.root/'task.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['sclwc'] = {'negative_class': 'class_1', 'negative_class_role': 'reference_endpoint_class'}
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='background_target_provenance'):
        SCLWC(Bundle(bundle.root))
    cfg['sclwc']['background_target_provenance'] = 'Synthetic endpoint reference, not clinically normal tissue; explicit PNM background target condition'
    path.write_text(yaml.safe_dump(cfg))
    run = train(bundle.root, PAPER_PROFILE, tmp_path/'run', config={'epochs': 6, 'threads': 1})
    protocol = read_json(run/'method_protocol.json')
    assert protocol['settings']['negative_class_role'] == 'reference_endpoint_class'
    history = read_json(run/'history.json')
    assert history[-1]['train']['pnm_active'] == 1 and history[-1]['train']['wscl_active'] > 0
    assert protocol['complete_paper_objectives']
    assert not (run/'test_predictions.csv').exists()
    assert evaluate(run)['unit'] == 'slide'

"""Independent author kernels, full mechanisms, real four-cluster updates and resume."""
import copy
import math

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from milplus.amiml import AMIML
from milplus.amiml_data import assign, cluster_bags, feature_bag, patient_bags, train_kmeans
from milplus.amiml_demo import create_amiml_demo
from milplus.amiml_network import PAPER, SOURCE, author_class, forward_bags, make_network
from milplus.data import Bundle
from milplus.engine import train, evaluate, seed_everything
from milplus.io import read_json, digest_file
from milplus.upstream import pinned_text, verified_source


def same(a, b):
    if torch.is_tensor(a): assert torch.equal(a, b)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else: assert a == b


def independent_original(dimension, classes, R=100):
    root, sha = verified_source('amiml')
    code = pinned_text(root, sha, 'AMIML/model.py')
    code = code.replace('super(HE2RNA, self).__init__()', 'super().__init__()')
    code = code.replace('in_channels=256, out_channels=128', 'in_channels=input_dim, out_channels=128')
    code = code.replace('        return y', '        return x')
    scope = {'__name__': '_amiml_independent_author'}
    exec(compile(code, str(root/'AMIML/model.py'), 'exec'), scope)
    result = scope['AMIML'](input_dim=dimension, R=R, d=8)
    result.value = nn.Linear(8, classes)
    return result


def test_unmodified_source_fails_and_constructor_only_port_still_breaks_binary_CE():
    root, sha = verified_source('amiml'); scope = {'__name__': '_amiml_unmodified'}
    exec(compile(pinned_text(root, sha, 'AMIML/model.py'), str(root/'AMIML/model.py'), 'exec'), scope)
    with pytest.raises(NameError, match='HE2RNA'): scope['AMIML']()
    source = author_class()().eval()
    with torch.no_grad(): out = source(torch.randn(2, 204, 256))
    assert out.shape == (2, 8)
    with pytest.raises(RuntimeError, match='weight tensor'): F.cross_entropy(out, torch.tensor([0, 1]), weight=torch.ones(2))


@pytest.mark.parametrize('dimension,classes', [(256, 2), (1024, 3), (1536, 2)])
@pytest.mark.parametrize('training', [True, False])
def test_full_native_R100_all_source_outputs_input_and_parameter_gradients(dimension, classes, training):
    torch.set_num_threads(1); seed_everything(73)
    port = make_network(dimension, classes).double().train(training)
    ref = independent_original(dimension, classes).double().train(training)
    ref.load_state_dict(port.state_dict())
    a = torch.randn(2, 207, dimension, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    torch.manual_seed(109); actual = forward_bags(port, list(a.unbind()))
    torch.manual_seed(109); expected = ref(b)
    torch.testing.assert_close(actual['logits'], expected, rtol=2e-13, atol=2e-13)
    assert actual['selected_ids'].shape == (2, 200) and actual['attention'].shape == (2, 200, 200)
    actual['logits'].square().sum().backward(); expected.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=2e-10, atol=2e-12)
    for (name, p), (other, q) in zip(port.named_parameters(), ref.named_parameters()):
        assert name == other
        if p.grad is None or q.grad is None:
            assert p.grad is q.grad is None and name.startswith(('bn4.', 'bn5.'))
        else: torch.testing.assert_close(p.grad, q.grad, rtol=3e-10, atol=3e-12)
    assert port.conv1.weight.shape == (128, dimension, 1) and port.conv4.weight.shape == (8, 32, 1)
    assert port.conv6.weight.shape == (1, 8, 1) and port.value.weight.shape == (classes, 8)


def test_ragged_BN_sees_only_real_patch_cells_and_no_cross_bag_ranking():
    torch.set_num_threads(1); seed_everything(51)
    model = make_network(16, 2, R=11).train()
    bags = [torch.randn(17, 16), torch.randn(23, 16)]
    inputs = model.conv1(torch.cat(bags).T[None]).detach()
    expected = .1*inputs.mean((0, 2))
    result = forward_bags(model, bags)
    torch.testing.assert_close(model.bn1.running_mean, expected)
    assert result['selected_ids'][0].max() < 17 and result['selected_ids'][1].max() < 23
    assert result['logits'].shape == (2, 2)
    result['logits'].sum().backward()
    assert model.conv6.weight.grad.abs().sum() > 0


def test_four_cluster_seed_streams_match_an_independently_scheduled_full_trainer(tmp_path):
    from milplus.engine import rng_state, restore_rng
    from torch.utils.data import DataLoader, RandomSampler
    b = Bundle(create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12))
    seed_everything(59); model = AMIML(b, R=11, batch_size=4)
    for net in model.networks[1:]: same(model.networks[0].state_dict(), net.state_dict())
    model.prepare(b, 59, 'cpu'); optimizers = model.make_optimizers(.001, 0)
    ref = copy.deepcopy(model.networks[2]); optimizer = torch.optim.Adam(ref.parameters(), lr=.001)
    branch_start = copy.deepcopy(model.branch_rng[2])
    torch.manual_seed(902); outer = rng_state()
    model.epoch(b, optimizers, 'cpu'); same(outer, rng_state())
    restore_rng(branch_start)
    groups = patient_bags(b, 'train'); loader = DataLoader(groups, batch_size=4,
        sampler=RandomSampler(groups), num_workers=0, collate_fn=list, drop_last=False)
    order = []
    for batch in loader:
        bags, _ = cluster_bags(b, batch, model.centers, 11, 2)
        target = torch.tensor([rows[0]['target'] for rows in batch])
        optimizer.zero_grad(set_to_none=True)
        output = forward_bags(ref, bags)
        loss = F.cross_entropy(output['logits'], target, weight=model.class_weights)
        loss.backward(); optimizer.step()
        order.extend(rows[0]['global_patient_id'] for rows in batch)
    same(ref.state_dict(), model.networks[2].state_dict())
    same(optimizer.state_dict(), optimizers[2].state_dict())
    same(rng_state(), model.branch_rng[2]); restore_rng(outer)
    assert order == model.train_orders[0][2]
    assert all(order == other for other in model.train_orders[0])


@pytest.mark.parametrize('profile', [PAPER, SOURCE])
def test_original_equal_loss_early_stopper_and_actual_plateau_decay_and_last_tie_selection(tmp_path, profile):
    import types
    from milplus.upstream import load_definitions
    b = Bundle(create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12))
    model = AMIML(b, profile, R=11, batch_size=4)
    model.prepare(b, 2026, 'cpu'); model.make_optimizers(.001, 0)
    source = load_definitions('amiml', 'AMIML/Earlystopping.py', ['EarlyStopping'],
        {'np':np, 'torch':types.SimpleNamespace(save=lambda *args:None)}, [('np.Inf','np.inf')]).EarlyStopping(30)
    frames = [model._predict_cluster('val',c,n,'cpu') for c,n in enumerate(model.networks)]
    source_lr = torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=.001)
    source_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(source_lr,mode='min',factor=.5,
        patience=10 if profile==PAPER else 5)
    for epoch,loss in enumerate([1.,.9,.9]+[1.]*30):
        source(loss, None); source_schedule.step(loss)
        model.completed_epochs = epoch+1
        current = []
        for frame in frames:
            f = frame.copy(); f.attrs['amiml_mean_batch_loss'] = loss; current.append(f)
        combined = current[0].copy(); combined.attrs['amiml_all_cluster_frames'] = current
        model.validation_metrics(combined)
        assert model.bad == [source.counter]*4 and model.stopped == [source.early_stop]*4
        assert all(s.get_last_lr()==source_schedule.get_last_lr() for s in model.schedulers)
    assert model.training_done and all(m['epoch']==32 for m in model.best_meta)
    assert source_schedule.get_last_lr()[0] < .001


@pytest.mark.parametrize('profile', [PAPER, SOURCE])
def test_actual_weighted_objective_matches_source_choice_and_full_Adam_update(tmp_path, profile):
    b = Bundle(create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12))
    seed_everything(31); model = AMIML(b, profile, R=11, batch_size=4)
    model.prepare(b, 31, 'cpu'); optimizers = model.make_optimizers(.001, 0)
    groups = [g for g in patient_bags(b, 'train') if len(g)==1][:4]
    assert len(groups)==4
    bags, _ = cluster_bags(b, groups, model.centers, 11, 0)
    network = model.networks[0]
    ref = independent_original(16, 2, R=11); ref.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(ref.parameters(), lr=.001)
    target = torch.tensor([rows[0]['target'] for rows in groups])
    torch.manual_seed(24); actual = forward_bags(network, bags); loss = model.objective(actual, target)
    torch.manual_seed(24); logits = ref(torch.stack(bags))
    value = logits.softmax(-1) if profile == SOURCE else logits
    expected = F.cross_entropy(value, target, weight=model.class_weights)
    assert torch.equal(loss, expected)
    loss.backward(); optimizers[0].step(); expected.backward(); optimizer.step()
    for p, q in zip(network.parameters(), ref.parameters()): torch.testing.assert_close(p, q, rtol=1e-7, atol=1e-8)
    same(optimizers[0].state_dict(), optimizer.state_dict())


def test_train_only_centers_and_patient_weights_unchanged_by_heldout_features_and_labels(tmp_path):
    root = create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12)
    b = Bundle(root); a, meta = train_kmeans(b, 2026, 10**8)
    weight = AMIML(b, R=11).class_weights.clone()
    for row in b.records('test')+b.records('val'):
        with h5py.File(root/row['path'], 'r+') as f:
            f['features'][:] = f['features'][:]+17
    labels = __import__('pandas').read_csv(root/'slides.csv')
    held = {r['slide_id'] for r in b.records('test')+b.records('val')}
    mask=labels.slide_id.isin(held)
    labels.loc[mask,'label']=labels.loc[mask,'label'].map({b.classes[0]:b.classes[1],b.classes[1]:b.classes[0]})
    labels.to_csv(root/'slides.csv', index=False)
    after = Bundle(root); c, other = train_kmeans(after, 2026, 10**8)
    assert torch.equal(a, c) and meta == other
    assert torch.equal(weight, AMIML(after, R=11).class_weights)
    assert meta['patients'] == len(patient_bags(b, 'train'))


def test_short_real_cluster_fails_before_any_optimization_and_budget_does_not_subsample(tmp_path):
    b = Bundle(create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12))
    model = AMIML(b, R=100)
    with pytest.raises(ValueError, match='EVERY morphology cluster'): model.prepare(b, 2026, 'cpu')
    assert model.optimizer_steps == [0]*4
    with pytest.raises(ValueError, match='no silent subsampling'): train_kmeans(b, 2026, 100)


def test_true_multislide_patient_bags_cluster_ids_and_no_square_or_order_proxy(tmp_path):
    b = Bundle(create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12))
    groups = [g for g in patient_bags(b, 'train') if len(g)==2]; assert groups
    x = feature_bag(b, groups[0]); assert x.shape == (96, 16)
    centers, _ = train_kmeans(b, 2026, 10**8)
    labels = assign(x, centers); ids = []
    for c in range(4):
        bags, idx = cluster_bags(b, [groups[0]], centers, 11, c)
        assert torch.equal(bags[0], x[idx[0]]) and len(idx[0]) == 24
        ids += idx[0].tolist()
    assert sorted(ids) == list(range(len(x)))
    permutation = torch.randperm(len(x))
    assert torch.equal(assign(x[permutation], centers), labels[permutation])


@pytest.mark.parametrize('profile', [PAPER, SOURCE])
def test_eight_epoch_all_four_optimizer_BN_scheduler_RNG_exact_resume_and_selected_reload(tmp_path, profile):
    data = create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12)
    cfg = {'epochs':8, 'threads':1, 'model':{'R':11, 'batch_size':4}}
    direct, resumed = tmp_path/'direct', tmp_path/'resumed'
    train(data, profile, direct, seed=55, config=cfg)
    train(data, profile, resumed, seed=55, config=cfg, stop_after=6)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(data, profile, resumed, seed=55, config=cfg, resume=True)
    a = torch.load(direct/'last.pt', weights_only=False); b = torch.load(resumed/'last.pt', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'history', 'best', 'bad']: same(a[key], b[key])
    state = a['model']['_extra_state']
    assert state['completed_epochs'] == 8 and len(a['optimizers']) == 4
    expected=8*math.ceil(len(patient_bags(Bundle(data),'train'))/4)
    assert state['optimizer_steps'] == [expected]*4
    assert all(m['epoch'] >= 5 for m in state['best_meta'])
    assert all(s['patience'] == (10 if profile == PAPER else 5) for s in state['schedulers'])
    assert not (direct/'test_predictions.csv').exists()
    result = evaluate(direct); sha = digest_file(direct/'test_predictions.csv')
    assert evaluate(direct) == result and digest_file(direct/'test_predictions.csv') == sha
    assert evaluate(resumed) == result and digest_file(resumed/'test_predictions.csv') == sha
    assert evaluate(direct, 'val')['amiml_selected_cluster'] == state['selected_cluster']


def test_binary_source_refuses_Cway_and_all_routes_refuse_slide_units_and_invented_sampling(tmp_path):
    b = Bundle(create_amiml_demo(tmp_path/'data', patients=30, classes=3, dimension=16, per_cluster=12))
    with pytest.raises(ValueError, match='binary'): AMIML(b, SOURCE, R=11)
    model = AMIML(b, PAPER, R=11)
    with pytest.raises(ValueError, match='full bags'): model.configure_training({'epochs':6, 'patience':30,
        'max_instances':10, 'grad_clip':None, 'weight_decay':0, 'selection_metric':'amiml_validation_generation'})
    b.config['training_unit'] = 'slide'
    with pytest.raises(ValueError, match='patient-bag'): AMIML(b, R=11)

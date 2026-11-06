"""Author-step comparisons, region geometry, masked outcomes and exact resume."""
import copy
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.advmil import AdversarialSurvival, adversarial_epoch, corrected_bce
from milplus.data import Bundle
from milplus.demo import create_demo, create_region_demo
from milplus.engine import evaluate, train, units
from milplus.region_data import patient_regions
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import load_definitions


def small(profile='advmil_esat_source', **kwargs):
    torch.set_num_threads(1)
    return AdversarialSurvival(8, 'months', profile=profile, hidden=16, discriminator_dim=8,
                               batch_size=4, selection_warmup=0, test_samples=4, **kwargs)


def author_handler(model):
    collector = load_definitions('advmil', 'utils/func.py', ['collect_tensor'], {'torch': torch}).collect_tensor
    logs = []
    namespace = {'torch': torch, 'wandb': SimpleNamespace(log=logs.append), 'collect_tensor': collector,
                 'fake_generator_loss': model.author.fake_generator_loss}
    handler = load_definitions('advmil', 'model/model_handler.py', ['MyHandler'], namespace).MyHandler
    h = object.__new__(handler)
    h.netG, h.netD = copy.deepcopy(model.generator), copy.deepcopy(model.discriminator)
    h.optimizerG = torch.optim.Adam(model.author.add_weight_decay(h.netG, 5e-4), lr=8e-5)
    h.optimizerD = torch.optim.Adam(h.netD.parameters(), lr=8e-5)
    h.task, h.bcb = 'cont_gansurv', 'patch' if 'esat' in model.profile else 'abmil'
    h.real_fake_loss = model.d_loss
    h.supervised_loss, h.coef_ganloss = model.reconstruction, model.gan_weight
    h.loss_l1 = model.author.loss_reg_l1(model.l1_weight)
    return h, logs


@pytest.mark.parametrize('profile', ['advmil_esat', 'advmil_esat_source', 'advmil_abmil', 'advmil_abmil_source'])
@pytest.mark.parametrize('mask', [[True, True, True], [False, True, False], [False, False, False]])
def test_full_alternating_adam_updates_match_author_handler(profile, mask):
    torch.manual_seed(71)
    model = small(profile)
    opts = model.make_optimizers(8e-5, 5e-4)
    ref, logs = author_handler(model)
    xs = [torch.randn(1, n, 8) for n in [16, 32, 48]]
    ys = [torch.tensor([[t, e]]) for t, e in [(0.2, 1.), (0.8, 0.), (0.6, 1.)]]
    source_x = [(x, None) for x in xs]
    for step in range(3):
        torch.manual_seed(120 + step)
        ref._update_disc(step, source_x, ys, mode='wolabel', label_visible_mask=mask)
        ref._update_gen(step, source_x, ys, mode='wolabel', label_visible_mask=mask)
        torch.manual_seed(120 + step)
        d = model.discriminator_step(xs, ys, mask, opts[1])
        g = model.generator_step(xs, ys, mask, opts[0])
        assert d == pytest.approx(logs[-2]['train_batch/netD/Loss_D'], abs=1e-7)
        assert g['loss'] == pytest.approx(logs[-1]['train_batch/netG/Loss_G_total'], abs=1e-7)
        for network, reference in [(model.generator, ref.netG), (model.discriminator, ref.netD)]:
            for (name, p), (other, q) in zip(network.named_parameters(), reference.named_parameters()):
                assert name == other
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_phase_ownership_and_active_adversarial_gradients():
    model = small('advmil_esat')
    optimizers = model.make_optimizers(8e-5, 5e-4)
    x, y = [torch.randn(1, 32, 8)], [torch.tensor([[.5, 1.]])]
    before = copy.deepcopy(model.generator.state_dict())
    model.discriminator_step(x, y, [True], optimizers[1])
    for k, v in before.items():
        torch.testing.assert_close(v, model.generator.state_dict()[k], rtol=0, atol=0)
    before = copy.deepcopy(model.discriminator.state_dict())
    result = model.generator_step(x, y, [False], optimizers[0])
    assert result['reconstruction'] == 0.
    for k, v in before.items():
        torch.testing.assert_close(v, model.discriminator.state_dict()[k], rtol=0, atol=0)
    # Remove L1 to directly prove the GAN route reaches the generator.
    model.l1_weight = 0.
    model.generator_step(x, y, [False], optimizers[0])
    assert model.generator.MLPs[-1][0].weight.grad.norm() > 0


def test_source_and_corrected_bce_are_explicit_and_finite():
    model = small()
    fake = torch.tensor([-2., 0., 2.], requires_grad=True)
    real = torch.tensor([1., -1.])
    source = model.d_loss(real, fake)
    expected = (fake.sigmoid()+1e-8).log().mean()-1-(real.sigmoid()+1e-8).log().mean()
    torch.testing.assert_close(source, expected)
    repaired = corrected_bce(real, fake)
    assert not torch.isclose(source, repaired)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    expected += torch.nn.functional.binary_cross_entropy_with_logits(real, torch.ones_like(real))
    torch.testing.assert_close(repaired, expected)
    extreme = torch.tensor([-1000., 1000.], requires_grad=True)
    corrected_bce(None, extreme).backward()
    assert torch.isfinite(extreme.grad).all()


def test_region_order_and_multislide_boundaries(tmp_path):
    b = Bundle(create_region_demo(tmp_path/'data', dimension=8))
    rows = next(g for g in units(b, 'train') if len(g) > 1)
    expected = patient_regions(b, rows, 'cpu')
    for row in rows:
        with h5py.File(b.root/row['path'], 'r+') as f:
            p = np.random.default_rng(3).permutation(len(f['features']))
            f['features'][:] = f['features'][:][p]
            f['coords'][:] = f['coords'][:][p]
        ids = b.root/row['region_ids_path']
        np.save(ids, np.load(ids)[p])
    torch.testing.assert_close(patient_regions(b, rows, 'cpu'), expected, rtol=0, atol=0)
    assert patient_regions(b, rows, 'cpu', 31).shape == (1,16,8)
    assert patient_regions(b, rows, 'cpu', 32, train=True).shape == (1,32,8)
    with pytest.raises(ValueError, match='>=16'):
        patient_regions(b, rows, 'cpu', 15)


@pytest.mark.parametrize('failure', ['missing_patch', 'duplicate_coord', 'stride', 'ids_shape', 'hash'])
def test_invalid_regions_rejected(tmp_path, failure):
    b = Bundle(create_region_demo(tmp_path/'data', dimension=8))
    row = b.records('train')[0]
    ids = b.root/row['region_ids_path']
    if failure == 'missing_patch':
        v = np.load(ids); v[0] = 9; np.save(ids, v)
    if failure == 'ids_shape':
        np.save(ids, np.load(ids)[:-1])
    if failure in {'duplicate_coord', 'stride'}:
        with h5py.File(b.root/row['path'], 'r+') as f:
            v = f['coords'][:]
            if failure == 'duplicate_coord':
                v[0] = v[1]
            else:
                v *= 2
            f['coords'][:] = v
    if failure == 'hash':
        row['region_ids_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        b.regions.inventory(row, deep=True)


def test_masking_never_uses_hidden_or_heldout_outcomes(tmp_path):
    b = Bundle(create_region_demo(tmp_path/'data', dimension=8, semi=True))
    a = small('advmil_abmil')
    a.prepare(b, 43, 'cpu')
    scale = a.time_scale.clone()
    state = copy.deepcopy(a.state_dict())
    opts = a.make_optimizers(8e-5, 5e-4)
    torch.manual_seed(17)
    first = adversarial_epoch(a, b, units(b,'train'), opts, 'cpu', {'max_instances':32})
    hidden = ~b.rows.global_patient_id.isin(a.plan['visible'])
    b.rows.loc[hidden, 'time'] = 1000000.
    b.rows.loc[hidden, 'event'] = 1-b.rows.loc[hidden, 'event']
    other = small('advmil_abmil'); other.load_state_dict(state); other.prepare(b, 43, 'cpu')
    torch.testing.assert_close(other.time_scale, scale)
    other_opts = other.make_optimizers(8e-5, 5e-4)
    torch.manual_seed(17)
    second = adversarial_epoch(other,b,units(b,'train'),other_opts,'cpu',{'max_instances':32})
    assert first == second
    for k, v in a.state_dict().items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, other.state_dict()[k], rtol=0, atol=0)
    active0 = {r[0]['global_patient_id'] for r in a.active_groups(units(b,'train'))}
    a.start_epoch(1)
    active1 = {r[0]['global_patient_id'] for r in a.active_groups(units(b,'train'))}
    assert active0 & active1 == set(a.plan['visible'])
    assert active0 | active1 == set(b.rows[b.rows.subset=='train'].global_patient_id)


def test_author_semi_supervised_sampling_contract(tmp_path):
    b=Bundle(create_region_demo(tmp_path/'data',dimension=8,semi=True))
    model=small(); model.prepare(b,42,'cpu')
    groups=units(b,'train')
    visible=set(model.plan['visible'])
    hidden=sorted(set(b.rows[b.rows.subset=='train'].global_patient_id)-visible)
    reference=load_definitions('advmil','utils/func.py',['get_kfold_pids'],
        {'np':np,'KFold':__import__('sklearn.model_selection',fromlist=['KFold']).KFold}).get_kfold_pids
    expected=reference(hidden,2,keep_pids=sorted(visible),random_state=42)
    for epoch in range(4):
        model.start_epoch(epoch)
        for mode in ['UD+LD','UD','LD']:
            model.plan['mode']=mode
            actual={r[0]['global_patient_id'] for r in model.active_groups(groups)}
            assert actual==(set(expected[epoch%2]) if mode=='UD+LD' else set(hidden) if mode=='UD' else visible)


@pytest.mark.parametrize('mode', ['supervised','UD+LD','UD','LD'])
def test_exact_resume_and_stochastic_inference(tmp_path, mode):
    path = create_region_demo(tmp_path/'data', dimension=8, semi=mode!='supervised')
    if mode!='supervised':
        task = yaml.safe_load((path/'task.yaml').read_text()); task['adversarial_training']['mode']=mode
        (path/'task.yaml').write_text(yaml.safe_dump(task))
    cfg={'epochs':3,'threads':1,'max_instances':32,'model':{'hidden':16,'discriminator_dim':8,
         'batch_size':5,'selection_warmup':0,'test_samples':4}}
    full=train(path,'advmil_esat',tmp_path/'full',config=cfg)
    part=train(path,'advmil_esat',tmp_path/'part',config=cfg,stop_after=1)
    train(path,'advmil_esat',part,config=cfg,resume=True)
    a,b=[torch.load(p/'last.pt',weights_only=False) for p in [full,part]]
    assert a['history']==b['history']
    assert len(a['optimizers'])==2
    for k in a['model']:
        if isinstance(a['model'][k], torch.Tensor):
            torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
        else:
            assert a['model'][k]==b['model'][k]
    assert not (full/'test_predictions.csv').exists()
    assert evaluate(full)==evaluate(part)
    frame=pd.read_csv(full/'test_predictions.csv')
    evaluate(full); pd.testing.assert_frame_equal(frame,pd.read_csv(full/'test_predictions.csv'))
    assert np.allclose(frame.risk,-frame.predicted_time)
    assert (frame.predicted_time_std>0).any()
    model=small(); model.load_state_dict(a['model']); model.eval()
    bundle=Bundle(path); state=torch.get_rng_state()
    model.forward_unit(bundle,units(bundle,'test')[0],'cpu',{'max_instances':32})
    assert torch.equal(state,torch.get_rng_state())


def test_tail_batch_and_censoring_and_input_rejection(tmp_path):
    b=Bundle(create_region_demo(tmp_path/'data',dimension=8))
    model=small(); model.prepare(b,0,'cpu')
    opts=model.make_optimizers(8e-5,5e-4)
    groups=units(b,'train')[:5]
    losses=adversarial_epoch(model,b,groups,opts,'cpu',{'max_instances':32})
    assert [v['patients'] for v in losses]==[4,1]
    pred=torch.tensor([[.2],[.8]],requires_grad=True)
    model.reconstruction(pred,torch.tensor([[.5],[.5]]),torch.zeros(2,1)).backward()
    torch.testing.assert_close(pred.grad,torch.tensor([[-.5],[0.]]))
    plain=Bundle(create_demo(tmp_path/'plain',patients=20,survival=True))
    with pytest.raises(ValueError,match='regions'):
        check_compatibility(require_runnable('advmil'),plain)
    b.config['adversarial_training']={'mode':'UD','visible_training_patients':[b.records('test')[0]['global_patient_id']],
        'unlabeled_folds':2,'label_provenance':'fixture'}
    with pytest.raises(ValueError,match='training-patient'):
        model.prepare(b,0,'cpu')
    with pytest.raises(ValueError,match='only implemented'):
        check_compatibility(require_runnable('abmil_cox'),b)


def test_region_change_invalidates_resume(tmp_path):
    data=create_region_demo(tmp_path/'data',dimension=8)
    cfg={'epochs':2,'threads':1,'model':{'hidden':16,'discriminator_dim':8,'selection_warmup':0}}
    run=train(data,'advmil_abmil_source',tmp_path/'run',config=cfg,stop_after=1)
    row=Bundle(data).records('train')[0]; ids=data/row['region_ids_path']
    np.save(ids,np.load(ids)+1)
    with pytest.raises(ValueError,match='Resume refused'):
        train(data,'advmil_abmil_source',run,config=cfg,resume=True)

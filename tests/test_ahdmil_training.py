import copy

import numpy as np
import pytest
import torch

from milplus.ahdmil import AHDMIL
from milplus.data import Bundle
from milplus.engine import evaluate, rng_state, restore_rng, seed_everything, train
from milplus.hdmil import snapshot, tensor_state_hash
from milplus.hdmil_demo import create_hdmil_demo
from milplus.io import read_json


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def config():
    return dict(epochs=5,threads=1,model=dict(dmin_epochs=2,joint_epochs=3,dmin_selection_start=1))


@pytest.mark.parametrize('boundary',[1,2,3])
def test_complete_pipeline_resume_across_pretrain_transfer_and_alternating_updates(tmp_path,boundary):
    data=create_hdmil_demo(tmp_path/'data',patients=20,dimension=1024)
    full,split=tmp_path/'full',tmp_path/'split'
    train(data,'AHDMIL',full,seed=2026,config=config())
    train(data,'AHDMIL',split,seed=2026,config=config(),stop_after=boundary)
    assert read_json(split/'status.json')['state']=='interrupted'
    train(data,'AHDMIL',split,seed=2026,config=config(),resume=True)
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(split/'last.pt',weights_only=False)
    for k in ['model','optimizers','history','rng','best','bad','epoch']:equal(a[k],b[k])
    extra=a['model']['_extra_state'];n=len(Bundle(data).records('train'))
    assert extra['updates']==dict(pretrain=2*n,lipn=3*n,dmin=3*n) and extra['ema_calls']==4
    assert extra['selected_dmin_hash']==tensor_state_hash(extra['selected_dmin'])
    assert extra['training_done'] and not extra['transfer_pending']
    for run in [full,split]:evaluate(run)
    prediction=(full/'test_predictions.csv').read_bytes();equal(prediction,(split/'test_predictions.csv').read_bytes())
    evaluate(full);equal(prediction,(full/'test_predictions.csv').read_bytes())


def test_automatic_pretraining_matches_complete_hdmil_pretraining_state(tmp_path):
    data=create_hdmil_demo(tmp_path/'data',patients=20,dimension=1024)
    train(data,'AHDMIL',tmp_path/'a',seed=2026,config=config(),stop_after=2)
    train(data,'HDMIL',tmp_path/'h',seed=2026,
        config=dict(epochs=5,threads=1,model=dict(dmin_epochs=2,lipn_epochs=3,dmin_selection_start=1)),stop_after=2)
    a=torch.load(tmp_path/'a/last.pt',weights_only=False);h=torch.load(tmp_path/'h/last.pt',weights_only=False)
    equal({k[9:]:v for k,v in a['model'].items() if k.startswith('pretrain.')},
          {k[5:]:v for k,v in h['model'].items() if k.startswith('dmin.')})
    equal(a['optimizers'][0],h['optimizers'][0]);equal(a['rng'],h['rng'])
    equal(a['model']['_extra_state']['selected_dmin'],h['model']['_extra_state']['best_dmin'])


def test_checkpoint_rejects_missing_ema_updates_pretrained_weights_and_selection(tmp_path):
    data=create_hdmil_demo(tmp_path/'data',patients=20)
    train(data,'AHDMIL',tmp_path/'run',config=config(),seed=2026)
    state=torch.load(tmp_path/'run/last.pt',weights_only=False)['model']
    for mutate in ['ema','updates','pretrained','history','generation','branch']:
        m=AHDMIL(Bundle(data),dmin_epochs=2,joint_epochs=3,dmin_selection_start=1)
        bad=copy.deepcopy(state);extra=bad['_extra_state']
        if mutate=='ema':extra['ema_calls']-=1
        elif mutate=='updates':extra['updates']['lipn']-=1
        elif mutate=='pretrained':next(v for v in extra['selected_dmin'].values() if v.is_floating_point()).add_(1)
        elif mutate=='history':extra['selection_history'][0]['value']+=100
        elif mutate=='generation':extra['generation_updates']-=1
        else:extra['train_orders'][-1]['manners'][0]='attn_train'
        with pytest.raises(ValueError):m.load_state_dict(bad)


@pytest.mark.parametrize('task,epoch,classes',[('brca',5,2),('rcc',55,3)])
def test_no_eligible_best_reinitializes_native_models_and_adams_with_exact_rng(tmp_path,task,epoch,classes):
    data=create_hdmil_demo(tmp_path/'data',patients=10*classes,classes=classes)
    bundle=Bundle(data);profile='ahdmil_'+task+'_source_repaired';m=AHDMIL(bundle,profile=profile)
    m.make_optimizers(m.native['lr'],1e-5)
    m.selected_dmin=snapshot(m.pretrain);m.selected_dmin_hash=tensor_state_hash(m.selected_dmin)
    m.stage='joint';m.apply_phase();m.stage_epoch=epoch;m.epoch_index=100+epoch
    m.stage_bad['joint']=4;m.generation_updates=25
    # Populate real optimizer state before reset, then independently reconstruct
    # the same source constructor/load sequence from the captured RNG.
    for optimizer,module in zip(m.optimizers[1:],[m.lipn,m.dmin]):
        optimizer.zero_grad();sum(p.sum() for p in module.parameters()).backward();optimizer.step()
    from milplus.ahdmil_source import build_joint
    rng=rng_state();a,b=build_joint(profile,bundle.dim);a.load_state_dict(m.selected_dmin);after=rng_state()
    restore_rng(rng);m.after_validation(dict(ahdmil_source_ce=1.,ahdmil_retention=.1))
    equal(m.dmin.state_dict(),a.state_dict());equal(m.lipn.state_dict(),b.state_dict());equal(rng_state(),after)
    assert m.optimizers[1].state=={} and m.optimizers[2].state=={} and m.generation_updates==0
    assert m.lower_ratio==(.25 if task=='rcc' else .3)
    assert m.joint_generation==1 and m.ema_calls==0 and m.stage_bad['joint']==0
    assert not m.training_done and m.stage_best['joint'] is None


def test_eligible_strict_ties_stop_at_five_and_invalid_low_retention_never_saves(tmp_path):
    m=AHDMIL(Bundle(create_hdmil_demo(tmp_path/'data',patients=20)))
    m.stage='joint';m.stage_epoch=1;m.epoch_index=0
    m.after_validation(dict(ahdmil_source_ce=2.,ahdmil_retention=.2))
    assert m.stage_best['joint'] is None and not m.selection_improved(-2.,-float('inf'))
    m.stage_epoch=2;m.epoch_index=1;m.after_validation(dict(ahdmil_source_ce=3.,ahdmil_retention=.3))
    assert m.stage_best['joint']==3.
    for e in range(3,8):
        m.stage_epoch=e;m.epoch_index=e-1;m.after_validation(dict(ahdmil_source_ce=3.,ahdmil_retention=.3))
    assert m.training_done and m.stage_bad['joint']==5


def test_incomplete_stage_budget_and_wrong_task_refused(tmp_path):
    b=Bundle(create_hdmil_demo(tmp_path/'data',patients=20))
    with pytest.raises(ValueError,match='full stages'):AHDMIL(b,dmin_epochs=2)
    with pytest.raises(ValueError,match='class count'):AHDMIL(b,profile='ahdmil_rcc_source_repaired')

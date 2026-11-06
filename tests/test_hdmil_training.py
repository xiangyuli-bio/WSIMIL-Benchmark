"""Actual both-stage optimization, teacher transfer, restart and public execution."""
import copy

import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.hdmil import HDMIL, tensor_state_hash
from milplus.hdmil_demo import create_hdmil_demo
from milplus.io import read_json


def equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def config():
    return dict(epochs=4, threads=1, model=dict(dmin_epochs=2, lipn_epochs=2, dmin_selection_start=1))


@pytest.mark.parametrize('boundary', [1,2,3])
def test_both_complete_stages_exact_resume_model_adam_history_rng_and_selected_reload(tmp_path,boundary):
    data = create_hdmil_demo(tmp_path/'data', patients=20, dimension=1024)
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    train(data,'HDMIL',full,seed=2026,config=config())
    train(data,'HDMIL',resumed,seed=2026,config=config(),stop_after=boundary)
    assert read_json(resumed/'status.json')['state']=='interrupted'
    train(data,'HDMIL',resumed,seed=2026,config=config(),resume=True)
    a=torch.load(full/'last.pt',weights_only=False); b=torch.load(resumed/'last.pt',weights_only=False)
    for k in ['model','optimizers','history','rng','best','bad','epoch']:equal(a[k],b[k])
    state=a['model']['_extra_state']; n=len(Bundle(data).records('train'))
    assert state['updates']==dict(dmin=2*n,lipn=2*n)
    assert [r['stage'] for r in state['selection_history']]==['dmin','dmin','lipn','lipn']
    assert state['best_dmin_hash']==tensor_state_hash({k[len('teacher.'):]:v for k,v in a['model'].items() if k.startswith('teacher.')})
    assert state['training_done'] and not state['transfer_pending']
    assert len(a['optimizers'])==2 and all(o['state'] for o in a['optimizers'])
    for run in [full,resumed]:
        evaluate(run)
    first=(full/'test_predictions.csv').read_bytes(); equal(first,(resumed/'test_predictions.csv').read_bytes())
    evaluate(full); equal(first,(full/'test_predictions.csv').read_bytes())


def test_checkpoint_rejects_omitted_stage_or_changed_teacher(tmp_path):
    data=create_hdmil_demo(tmp_path/'data',patients=20)
    train(data,'HDMIL',tmp_path/'run',config=config(),seed=2026)
    state=torch.load(tmp_path/'run/last.pt',weights_only=False)['model']
    for mutate in ['updates','teacher','history']:
        m=HDMIL(Bundle(data),dmin_epochs=2,lipn_epochs=2,dmin_selection_start=1)
        bad=copy.deepcopy(state)
        if mutate=='updates':bad['_extra_state']['updates']['lipn']-=1
        elif mutate=='teacher':bad[next(k for k in bad if k.startswith('teacher.') and bad[k].is_floating_point())].add_(1)
        else:bad['_extra_state']['selection_history'][0]['value']+=100
        with pytest.raises(ValueError):m.load_state_dict(bad)


def test_original_stage_selection_has_one_based50_boundary_strict_ties_and_two_patiences(tmp_path):
    b=Bundle(create_hdmil_demo(tmp_path/'data',patients=20)); m=HDMIL(b)
    for e in range(1,50):
        m.stage_epoch=e; m.epoch_index=e-1; m.after_validation(dict(hdmil_source_ce=.1))
    assert m.best_dmin is None
    m.stage_epoch=50; m.epoch_index=49; m.after_validation(dict(hdmil_source_ce=.2))
    assert m.best_dmin_epoch==50
    for e in range(51,71):
        m.stage_epoch=e; m.epoch_index=e-1; m.after_validation(dict(hdmil_source_ce=.2))
    assert m.transfer_pending and not m.training_done and m.stage_bad['dmin']==20
    assert m.best_dmin_epoch==50


def test_incomplete_topk_and_missing_second_stage_refused(tmp_path):
    b=Bundle(create_hdmil_demo(tmp_path/'data',patients=20))
    with pytest.raises(ValueError,match='both complete'):
        HDMIL(b,dmin_epochs=2)
    with pytest.raises(ValueError,match='class count'):
        HDMIL(b,profile='hdmil_rcc_source')

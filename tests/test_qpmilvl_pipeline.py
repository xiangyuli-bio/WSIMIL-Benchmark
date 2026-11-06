import json
import os
import pytest
import torch

from milplus.qpmilvl_demo import create_demo
from milplus.qpmilvl_driver import train,evaluate,PROFILES
from milplus.qpmilvl_data import QPMILInputs
from milplus.icmil_coupled_training import load
from milplus.data import Bundle
from test_qpmilvl_source import same,WEIGHTS,SHA,DEVICE


@pytest.mark.parametrize('profile',list(PROFILES))
def test_full_four_task_protocol_all_gradients_frequency_optimizer_generators_rng_resume_and_reload(tmp_path,profile,monkeypatch):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);cfg={'epochs':2,'threads':2}
    native=QPMILInputs.features;observed=[]
    def guarded(self,row):
        assert row['subset']!='test';observed.append((row['subset'],row['target']));return native(self,row)
    monkeypatch.setattr(QPMILInputs,'features',guarded)
    whole=train(data,tmp_path/'whole',config=cfg,device=DEVICE,method=profile);baseline=load(whole/'last.pt')
    assert baseline['next_task']==4 and len(baseline['history'])==8 and baseline['updates']==8
    assert all(len(r['train_order'])==16 and r['dropped_native_incomplete_tail_slides']==0 for r in baseline['history'])
    for point in [1,2,3]:
        run=train(data,tmp_path/('resumed'+str(point)),config=cfg,device=DEVICE,method=profile,stop_after=point)
        assert json.loads((run/'status.json').read_text())['state']=='interrupted'
        train(data,run,config=cfg,device=DEVICE,method=profile,resume=True);actual=load(run/'last.pt')
        for key in ['pools','gradients','next_task','next_epoch','optimizer','scheduler','early','frequency','val_frequency','history','updates','generators','rng']:same(baseline[key],actual[key])
        for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']))
    monkeypatch.setattr(QPMILInputs,'features',native)
    result=evaluate(whole,device=DEVICE);previous=(whole/'test_predictions.csv').read_bytes()
    assert result==evaluate(whole,device=DEVICE) and previous==(whole/'test_predictions.csv').read_bytes()
    request=json.loads((whole/'request.json').read_text())
    assert request['recipe']['pooling']==PROFILES[profile]['pooling']
    assert request['recipe']['pool_size']==20 and request['recipe']['prompt_length']==24 and request['recipe']['match_size']==5
    assert not request['test_used_for_training_or_selection'] and not request['recipe']['load_best_ckpt']
    assert len(result['matrix'])==4 and [len(row) for row in result['matrix']]==[1,2,3,4]
    assert observed


def test_first_epoch_does_not_decode_future_or_TEST_and_rejects_partial_evaluation(tmp_path,monkeypatch):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);native=QPMILInputs.features;observed=[]
    def guarded(self,row):
        assert row['subset']!='test' and row['target']<2;observed.append(row['slide_id']);return native(self,row)
    monkeypatch.setattr(QPMILInputs,'features',guarded)
    run=train(data,tmp_path/'run',config={'epochs':2},device=DEVICE,stop_after=1)
    assert observed and json.loads((run/'status.json').read_text())['state']=='interrupted'
    with pytest.raises(ValueError,match='Complete every'):evaluate(run,device=DEVICE)


def test_literal_incomplete16bag_tail_is_disclosed_and_never_silently_flushed(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA,train_per_class=9)
    run=train(data,tmp_path/'run',config={'epochs':1},device=DEVICE);state=load(run/'last.pt')
    assert state['updates']==4 and len(state['history'])==4
    assert all(len(r['train_order'])==18 and r['dropped_native_incomplete_tail_slides']==2 for r in state['history'])

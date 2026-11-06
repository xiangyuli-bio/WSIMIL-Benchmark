import copy
import json
from pathlib import Path

import h5py
import pytest
import torch
import yaml

from milplus import comel_source as source
from milplus.comel_demo import create_demo
from milplus.comel_driver import train,evaluate
from milplus.comel_data import CoMELInputs
from milplus.data import Bundle
from milplus.icmil_coupled_training import load


def same(a,b):
    import numpy as np
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('profile',sorted(source.PROFILES))
def test_every_task_full_state_interruption_midtask_boundary_and_later_lora(tmp_path,profile,monkeypatch):
    data=create_demo(tmp_path/'data',tasks=3);cfg={'epochs':2,'threads':2};seen=[]
    original=CoMELInputs.features
    def audit(self,row):
        seen.append((row['subset'],row['target']//2));assert row['subset']!='test'
        return original(self,row)
    monkeypatch.setattr(CoMELInputs,'features',audit)
    whole=train(data,profile,tmp_path/'whole',config=cfg,seed=83)
    assert seen and {t for subset,t in seen if subset=='train'}=={0,1,2}
    baseline=load(whole/'last.pt');assert baseline['updates']==12 and len(baseline['history'])==6
    assert baseline['next_task']==3 and baseline['architecture']['heads']==3
    assert all(len(d['ranks'])==4 for d in baseline['architecture']['modules'].values())
    # Each interruption uses the original complete requested budgets. Resume
    # must consume saved RNG and accumulated non-optimized gradients exactly.
    for stop in [1,2,3]:
        out=train(data,profile,tmp_path/('resume'+str(stop)),config=cfg,seed=83,stop_after=stop)
        assert json.loads((out/'status.json').read_text())['state']=='interrupted'
        train(data,profile,out,config=cfg,seed=83,resume=True)
        actual=load(out/'last.pt')
        for key in ['architecture','model','gradients','owner','history','validation_matrix','thresholds','updates','rng','training_modes']:
            same(baseline[key],actual[key])
        for left,right in zip(baseline['snapshots'],actual['snapshots']):
            same(load(left['path']),load(right['path']))
    monkeypatch.setattr(CoMELInputs,'features',original)
    first=evaluate(whole,'test');second=evaluate(whole,'test');assert first==second
    assert len(first['matrix'])==3 and [len(row) for row in first['matrix']]==[1,2,3]
    assert first['threshold_selection']=='VAL_only' and not first['test_selection']


def test_future_tasks_not_loaded_early_and_no_train_patch_ground_truth(tmp_path,monkeypatch):
    data=create_demo(tmp_path/'data');original=CoMELInputs.features;seen=[]
    def guarded(self,row):
        assert row['target']<2 and row['subset']=='train';seen.append(row['slide_id']);return original(self,row)
    monkeypatch.setattr(CoMELInputs,'features',guarded)
    out=train(data,'comel_owlora_source_patient_reference',tmp_path/'run',config={'epochs':2},stop_after=1)
    assert len(seen)==2 and json.loads((out/'status.json').read_text())['state']=='interrupted'
    with pytest.raises(ValueError,match='Complete every'):evaluate(out)


def test_feature_manifest_tamper_rejected(tmp_path):
    data=create_demo(tmp_path/'data');bundle=Bundle(data);row=bundle.records('test')[0]
    with h5py.File(row['path'],'r+') as f:f['features'][0,0]+=1
    with pytest.raises(ValueError,match='feature bytes changed'):Bundle(data)


def test_task_partition_change_rejected(tmp_path):
    data=create_demo(tmp_path/'data');p=data/'comel_manifest.json';d=json.loads(p.read_text());d['tasks'].reverse();p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='integrity'):Bundle(data)


def test_model_budget_or_architecture_cannot_silently_shrink(tmp_path):
    data=create_demo(tmp_path/'data')
    with pytest.raises(ValueError,match='native512D'):train(data,'comel_owlora_source_patient_reference',tmp_path/'run',config={'hidden':8})
    with pytest.raises(ValueError,match='positive'):train(data,'comel_owlora_source_patient_reference',tmp_path/'bad',config={'epochs':0})


def test_same_config_required_for_resume_and_source_has_no_test_selection(tmp_path):
    data=create_demo(tmp_path/'data');out=train(data,'comel_owlora_source_patient_reference',tmp_path/'run',config={'epochs':2},stop_after=1)
    with pytest.raises(ValueError,match='resume changed'):train(data,'comel_owlora_source_patient_reference',out,config={'epochs':3},resume=True)
    request=json.loads((out/'request.json').read_text());assert not request['test_used_for_training_or_selection']
    assert not request['training_instance_labels_used']

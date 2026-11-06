"""Actual full text patient training, exact resume and compact-state integrity."""
import copy
import json

import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.engine import evaluate,train
from milplus.io import digest_file,read_json
from milplus.vlsa import Patients,VLSAMIL
from milplus.vlsa_demo import create_vlsa_demo
from milplus.vlsa_source import weights_path

WEIGHTS=weights_path()


def test_complete_patient_recipe_resume_selection_reload_and_adam_integrity(tmp_path):
    data=create_vlsa_demo(tmp_path/'inputs',WEIGHTS)
    cfg={'epochs':2,'threads':2,'model':{'update_patients':5}}
    uninterrupted=train(data,'VLSA',tmp_path/'whole',seed=2067,config=cfg)
    resumed=train(data,'VLSA',tmp_path/'resumed',seed=2067,config=cfg,stop_after=1)
    assert read_json(resumed/'status.json')['state']=='interrupted'
    train(data,'VLSA',resumed,seed=2067,config=cfg,resume=True)
    a=torch.load(uninterrupted/'last.pt',weights_only=False)
    b=torch.load(resumed/'last.pt',weights_only=False)
    best=torch.load(resumed/'best.pt',weights_only=False)
    assert a['epoch']==b['epoch']==best['epoch']==1
    for name,value in a['model'].items():
        if torch.is_tensor(value):assert torch.equal(value,b['model'][name]),name
        else:assert value==b['model'][name],name
    assert a['history']==b['history']
    protocol=read_json(resumed/'survival_protocol.json')
    assert protocol['completed_epochs']==2 and protocol['optimizer_updates']==6
    assert protocol['num_queries']==10 and protocol['update_patients']==5
    extra=b['model']['_extra_state']
    assert extra['window_sizes']==[[5,5,2],[5,5,2]] and extra['optimizer_updates']==6
    assert all(set(order)==set(Bundle(data).vlsa.train_patients) for order in extra['train_orders'])
    assert not any(name.startswith('core.prompt_encoder.') and name!='core.prompt_encoder.attn_mask'
                   for name in b['model'])
    assert b['model']['core.prompt_encoder.attn_mask'].shape==(128,128)
    assert b['model']['core.prompt_learner.context_embeds'].shape==(4,768)
    metrics=evaluate(resumed);pred_sha=digest_file(resumed/'test_predictions.csv')
    assert evaluate(resumed)==metrics and digest_file(resumed/'test_predictions.csv')==pred_sha
    assert 'vlsa_source_mean_time_c_index' in metrics
    m=VLSAMIL(Bundle(data),update_patients=5);m.load_state_dict(b['model'])
    optimizer=m.make_optimizer(2e-4,1e-5);optimizer.load_state_dict(b['optimizers'][0]);m.validate_optimizer_state()
    optimizer.param_groups[0]['weight_decay']=1e-5
    with pytest.raises(ValueError,match='parameter grouping'):m.validate_optimizer_state()
    with pytest.raises(ValueError,match='Move VLSA'):m.to('cpu')
    for corruption in ['fixed','missing','nonfinite','order','window']:
        state=copy.deepcopy(b['model'])
        if corruption=='fixed':state['time_coordinates'][1]+=1
        elif corruption=='missing':state.pop('core.logit_scale')
        elif corruption=='nonfinite':state['core.logit_scale'].fill_(float('nan'))
        elif corruption=='order':state['_extra_state']['train_orders'][0][0]='TEST-patient'
        else:state['_extra_state']['window_sizes'][0]=[1]*12
        with pytest.raises(ValueError):m.load_state_dict(state)


def test_native_manifest_alignment_patient_completeness_and_postload_mutation(tmp_path):
    data=create_vlsa_demo(tmp_path/'inputs',WEIGHTS);b=Bundle(data)
    patient=next(rows for rows in Patients(b,'train').rows if len(rows)>1)
    assert b.vlsa.patient(patient).shape[2]==512
    with pytest.raises(ValueError,match='all patient slides'):b.vlsa.patient(patient[:1])
    with pytest.raises(ValueError,match='stable slide-ID order'):b.vlsa.patient(patient[::-1])
    manifest=data/'vlsa_inputs.json';original=manifest.read_text()
    manifest.write_text(original+' ')
    with pytest.raises(ValueError,match='manifest bytes changed'):b.vlsa.patient(patient)
    manifest.write_text(original)
    task=yaml.safe_load((data/'task.yaml').read_text());task['encoders']['toy']['model']='other_FM'
    (data/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='no arbitrary FM projection'):Bundle(data)

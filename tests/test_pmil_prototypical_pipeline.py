"""Complete raw PMIL lifecycle on explicitly synthetic five-patch software bags."""
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.icmil_coupled_training import load
from milplus.io import read_json,write_json
from milplus.pmil_prototypical_demo import create_demo
from milplus.pmil_prototypical_driver import PROFILE
from milplus.pmil_prototypical_data import PMILRGB
from test_pmil_prototypical_components import equal

WEIGHTS=Path(os.environ.get('MILPLUS_PMIL_WEIGHTS', Path.home()/'.cache/milplus/pmil/resnet34-333f7ec4.pth'))
CFG={'epochs':2,'threads':2,'model':{'encoder_epochs':2}}


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    base=tmp_path_factory.mktemp('pmil_complete_reference')
    data=create_demo(base/'data',WEIGHTS)
    run=train(data,PROFILE,base/'continuous',seed=2026,config=CFG)
    return base,data,run


def test_full_raw_encoder_clustering_dual_loss_pipeline_and_two_fresh_reloads(reference):
    base,data,run=reference
    status=read_json(run/'status.json');request=read_json(run/'request.json')
    assert status['state']=='completed' and status['test_evaluated'] is False
    assert request['synthetic'] is True and request['test_labels_used_before_explicit_evaluate'] is False
    max_state=load(run/'max_mil/last.pt');bag_state=load(run/'pmil/last.pt')
    assert max_state['epoch']==2 and max_state['updates']==2
    assert bag_state['epoch']==2 and bag_state['updates']==4
    assert not max_state['validation_backprop'] and bag_state['source_train_eval_mode']
    prototypes=load(run/'prototype_discovery/prototypes.pt')['result']
    assert prototypes['all_patch_candidates']==10 and prototypes['features'].shape[1]==512
    assert len(prototypes['local'])==2
    assert all(row['slide_id'] in {'synthetic_0','synthetic_1'} for row in prototypes['origins'])
    selected=read_json(run/'selected.json')
    assert Path(selected['encoder']).exists() and Path(selected['bag']).exists()
    first=evaluate(run);before=(run/'test_predictions.csv').read_bytes()
    second=evaluate(run);assert (run/'test_predictions.csv').read_bytes()==before
    equal(first,second)
    # Actual downstream inputs come from the task-trained original encoder.
    index=load(run/'features/index.pt')['rows']
    assert {r['subset'] for r in index}=={'train','val'}
    assert all(np.load(r['path']).shape==(5,512) for r in index)


@pytest.mark.parametrize('stop',[1,3])
def test_genuine_each_stage_interruption_restores_all_weights_BN_moments_RNG_and_history(reference,stop):
    base,data,continuous=reference
    run=train(data,PROFILE,base/('interrupted_'+str(stop)),seed=2026,config=CFG,stop_after=stop)
    assert read_json(run/'status.json')['state']=='interrupted'
    assert read_json(run/'status.json')['phase']==('max_mil' if stop==1 else 'pmil')
    for phase in ['max_mil','pmil']:
        if (run/phase/'last.checkpoint.json').exists():
            # Preserve the genuine pre-resume generation separately as evidence.
            value=load(run/phase/'last.pt')
            torch.save(value,run/phase/'interrupted_preserved.pt')
    train(data,PROFILE,run,seed=2026,config=CFG,resume=True)
    assert read_json(run/'status.json')['state']=='completed'
    for phase in ['max_mil','pmil']:
        a,b=load(continuous/phase/'last.pt'),load(run/phase/'last.pt')
        for key in ['model','optimizer','scheduler','history','rng','epoch','updates','best']:
            equal(a[key],b[key])
    a=load(continuous/'prototype_discovery/prototypes.pt')['result']
    b=load(run/'prototype_discovery/prototypes.pt')['result']
    equal(a,b)
    evaluate(run)
    assert (run/'test_predictions.csv').read_bytes()==(continuous/'test_predictions.csv').read_bytes()


def test_raw_patches_cannot_enter_frozen_MIL_and_invalid_recipe_cannot_change_native_model(reference,tmp_path):
    _,data,_=reference;bundle=Bundle(data)
    with pytest.raises(ValueError,match='full max-MIL'):
        bundle.load(bundle.records('train')[0])
    with pytest.raises(ValueError,match='cannot replace frozen'):
        train(data,'ABMIL',tmp_path/'frozen')
    for cfg in [{'epochs':1,'model':{'hidden_dim':128}}, {'lr':.001}, {'model':{'encoder_epochs':0}}]:
        with pytest.raises(ValueError,match='preserves full|positive integer'):
            train(data,PROFILE,tmp_path/'invalid',config=cfg)


def test_no_TEST_pixels_enter_any_training_or_prototype_stage(tmp_path,monkeypatch):
    data=create_demo(tmp_path/'data',WEIGHTS);original=PMILRGB.open
    def checked(self,row):
        assert row['subset']!='test','Held-out TEST pixels entered the training pipeline'
        return original(self,row)
    monkeypatch.setattr(PMILRGB,'open',checked)
    run=train(data,PROFILE,tmp_path/'run',seed=2026,config={'epochs':1,'model':{'encoder_epochs':1}})
    assert read_json(run/'status.json')['state']=='completed'


def test_actual_pixel_ID_or_preprocessing_changes_are_rejected(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS)
    bundle=Bundle(data);row=bundle.records('train')[0]
    with h5py.File(row['path'],'r+') as f:f['rgb'][0,0,0,0]^=1
    with pytest.raises(ValueError,match='actual RGB bytes'):
        bundle.check(deep=True)
    data2=create_demo(tmp_path/'data2',WEIGHTS)
    task=yaml.safe_load((data2/'task.yaml').read_text())
    manifest=data2/task['encoders']['raw_rgb']['pmil_prototypical']['manifest']
    metadata=read_json(manifest);metadata['schema']['normalization']='ImageNet_mean_std'
    write_json(manifest,metadata)
    with pytest.raises(ValueError,match='original normalization'):
        Bundle(data2)

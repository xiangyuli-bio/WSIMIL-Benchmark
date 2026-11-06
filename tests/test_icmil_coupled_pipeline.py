from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from milplus.data import Bundle
from milplus.icmil_coupled_demo import create_demo
from milplus.icmil_coupled_driver import train,evaluate,PROFILES
from milplus.icmil_coupled_stages import CachedBags
from milplus.icmil_coupled_training import save,load
from milplus.io import read_json,digest_file

WEIGHTS='/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/development/icmil_coupled_v92/legacy14_assets_r1/resnet50-11ad3fa6.pth'


def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def test_feature_order_map_preserves_actual_source_shuffle_and_RNG():
    for n in [1,5,17,257,1000]:
        x=np.arange(n*1024,dtype=np.float32).reshape(n,1024)
        np.random.seed(32);np.random.shuffle(x);after=np.random.get_state()
        order=np.arange(n,dtype=np.int64);np.random.seed(32);np.random.shuffle(order)
        assert np.array_equal(x,np.arange(n*1024,dtype=np.float32).reshape(n,1024)[order])
        exact(after,np.random.get_state())


def test_checkpoint_commit_cannot_pair_new_payload_with_old_checksum(tmp_path,monkeypatch):
    import milplus.icmil_coupled_training as training
    path=tmp_path/'last.pt';save({'model':torch.tensor([1.])},path);old=read_json(path.with_suffix('.checkpoint.json'))
    original=training.write_json
    def fail_pointer(target,value):
        if Path(target)==path.with_suffix('.checkpoint.json'):raise RuntimeError('simulated interruption before atomic checkpoint commit')
        return original(target,value)
    monkeypatch.setattr(training,'write_json',fail_pointer)
    with pytest.raises(RuntimeError,match='simulated interruption'):save({'model':torch.tensor([2.])},path)
    assert read_json(path.with_suffix('.checkpoint.json'))==old
    assert torch.equal(load(path)['model'],torch.tensor([1.]))
    assert len(list(tmp_path.glob('last.*.pt')))==2


@pytest.mark.parametrize('profile',list(PROFILES))
def test_complete_native_graph_extract_train_distill_reextract_retrain_and_fresh_reload(tmp_path,profile):
    data=create_demo(tmp_path/'data',WEIGHTS);config={'epochs':1,'threads':2,'model':{'iterations':1,'distill_epochs':1}}
    run=train(data,profile,tmp_path/'run',config=config)
    status=read_json(run/'status.json');assert status['state']=='completed' and status['MIL_phases']==2 and status['encoder_distillation_phases']==1
    assert status['test_evaluated'] is False and not (run/'evaluation_features_test').exists()
    for index in [0,1]:
        receipt=read_json(run/('features_'+str(index))/'index.json')
        assert receipt['all_registered_patches_used'] is True and {r['subset'] for r in receipt['rows']}=={'train','val'}
        assert all(r['shape']==[5,1024] for r in receipt['rows'])
    first=load(run/'encoder_0.pt')['encoder'];last=load(run/'encoder_1.pt')['encoder']
    assert not torch.equal(first['conv1.weight'],last['conv1.weight'])
    state=load(run/'distill_0/last.pt')
    assert state['history'][0]['patches']==10 and state['updates']==1
    assert torch.equal(state['model']['teacher_encoder.conv1.weight'],first['conv1.weight'])
    assert torch.equal(state['model']['teacher_encoder.bn1.running_mean'],first['bn1.running_mean'])
    start=load(run/'bag_1/initial_model.pt');previous=load(run/'bag_0/selected.pt')['model']
    if 'abmil' in profile:
        assert start['initialization']=='fresh_source_constructor'
        assert any(not torch.equal(start['model'][k],previous[k]) for k in previous)
    else:
        assert start['initialization']=='previous_selected_MIL_checkpoint'
        exact(start['model'],previous)
    a=evaluate(run);sha=digest_file(run/'test_predictions.csv');b=evaluate(run)
    assert a==b and digest_file(run/'test_predictions.csv')==sha


@pytest.mark.parametrize('stop_after',[1,3])
@pytest.mark.parametrize('method',list(PROFILES))
def test_both_MIL_and_encoder_interruption_restore_every_weight_moment_order_and_RNG(tmp_path,stop_after,method):
    data=create_demo(tmp_path/'data',WEIGHTS)
    config={'epochs':2,'threads':2,'model':{'iterations':1,'distill_epochs':2}}
    continuous=train(data,method,tmp_path/'continuous',config=config,seed=449)
    resumed=train(data,method,tmp_path/'resumed',config=config,seed=449,stop_after=stop_after)
    assert read_json(resumed/'status.json')['state']=='interrupted'
    train(data,method,resumed,config=config,seed=449,resume=True)
    for stage in ['bag_0','distill_0','bag_1']:exact(load(continuous/stage/'last.pt'),load(resumed/stage/'last.pt'))
    a=evaluate(continuous);b=evaluate(resumed);assert a==b
    pd.testing.assert_frame_equal(pd.read_csv(continuous/'test_predictions.csv'),pd.read_csv(resumed/'test_predictions.csv'),check_exact=True)

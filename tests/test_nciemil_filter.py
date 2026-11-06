"""Actual native Swin gradients/recovery plus independent multi-chunk max oracle."""
import copy
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from milplus.nciemil_filter import (FilterInputs,SwinFilter,WEIGHTS_SHA256,
    replay_max_instance_backward,torch_rng,train_filter)


ASSET=Path(os.environ.get('MILPLUS_NCIEMIL_SWIN_WEIGHTS',
    '/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/model_assets/nciemil_swin_torchvision0201/swin_t-704ceda3.pth'))


def gradients_equal(a,b):
    assert dict(a.named_parameters()).keys()==dict(b.named_parameters()).keys()
    for name,p in a.named_parameters():
        q=dict(b.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)


def test_full_pretrained_swin_train_mode_replay_equals_literal_max_loss_every_gradient_and_rng():
    torch.set_num_threads(2);torch.manual_seed(8901)
    model=SwinFilter(ASSET,WEIGHTS_SHA256).train();reference=copy.deepcopy(model).train()
    assert len(model.encoder.features)==8 and model.projection.in_features==768
    assert sum(p.numel() for p in model.parameters())==27_520_123
    images=torch.rand(2,3,256,256)
    torch.manual_seed(8902)
    _,scores=reference(images);loss=F.binary_cross_entropy_with_logits(scores.max(),torch.tensor(1.))
    loss.backward();after=torch_rng()
    torch.manual_seed(8902)
    trace=replay_max_instance_backward(model,lambda a,b:images[a:b],2,1,'cpu')
    assert trace['winner']==int(scores.argmax()) and trace['loss']==float(loss)
    assert trace['patches_scanned']==2 and trace['replayed_chunk_count']==2
    gradients_equal(model,reference)
    assert torch.equal(torch_rng()['cpu'],after['cpu'])
    assert model.encoder.features[0][0].weight.grad.abs().sum()>0
    assert model.projection.weight.grad.abs().sum()>0


class StochasticChunkModel(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(3,11),nn.GELU(),nn.Dropout(.35),nn.Linear(11,1))
    def forward(self,x):
        features=x.mean((2,3));return features,self.network(features).squeeze(-1)


def test_all257_patches_and_original128_plus128_plus1_stochastic_chunks_match_retained_graph():
    torch.set_num_threads(2);torch.manual_seed(8903)
    model=StochasticChunkModel().train();reference=copy.deepcopy(model)
    values=torch.linspace(-1,3,257*3).reshape(257,3,1,1)
    read=lambda start,end:values[start:end].expand(-1,-1,3,3).contiguous()
    torch.manual_seed(8904)
    scores=torch.cat([reference(read(i,min(i+128,257)))[1] for i in range(0,257,128)])
    loss=F.binary_cross_entropy_with_logits(scores.max(),torch.tensor(0.))/3;loss.backward();after=torch_rng()
    torch.manual_seed(8904)
    trace=replay_max_instance_backward(model,read,257,0,'cpu',normalization=3)
    assert trace['winner']==int(scores.argmax()) and trace['patches_scanned']==257
    assert trace['replayed_chunk_count'] in {1,128}
    gradients_equal(model,reference)
    assert torch.equal(torch_rng()['cpu'],after['cpu'])


def test_equal_maxima_keep_first_original_patch_and_reject_batchnorm_replay():
    model=StochasticChunkModel().train()
    for p in model.parameters():nn.init.zeros_(p)
    trace=replay_max_instance_backward(model,lambda a,b:torch.ones(b-a,3,3,3),257,1,'cpu')
    assert trace['winner']==0 and trace['replayed_chunk_start']==0
    model.network.insert(0,nn.BatchNorm1d(3))
    with pytest.raises(ValueError,match='BatchNorm'):
        replay_max_instance_backward(model,lambda a,b:torch.ones(b-a,3,3,3),257,1,'cpu')


def raw_fixture(root):
    root.mkdir();rng=np.random.default_rng(8905);slides=[];splits=[];index=[]
    for subset in ['train','val','test']:
        for label in ['negative','positive']:
            slide=subset+'_'+label;path=root/(slide+'.h5')
            with h5py.File(path,'w') as f:
                f.create_dataset('pixels',data=rng.integers(0,256,(2,256,256,3),dtype=np.uint8))
                f.create_dataset('patch_ids',data=np.array([slide+'_p0',slide+'_p1'],dtype=h5py.string_dtype()))
            slides.append({'slide_id':slide,'patient_id':slide,'label':label})
            splits.append({'patient_id':slide,'subset':subset});index.append({'slide_id':slide,'path':path.name})
    pd.DataFrame(slides).to_csv(root/'slides.csv',index=False)
    pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    pd.DataFrame(index).to_csv(root/'images.csv',index=False)
    config={'task_id':'nciemil_filter_native_test','schema_version':1,'task_type':'classification','classes':['negative','positive'],
        'defaults':{'fm':'rgb','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'rgb':{'dimension':3,'index':'images.csv','nciemil_raw':{
            'kind':'nciemil_rgb256_uint8_hdf5','schema_version':1,'image_provenance':'synthetic_RGB256',
            'patch_provenance':'two_unique_test_patches','weak_targets':{'negative':0,'positive':1}}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config));return root


def test_raw_identity_requires_explicit_weak_mapping_and_detects_rgb_changes(tmp_path):
    path=raw_fixture(tmp_path/'data');inputs=FilterInputs(path);inputs.check(deep=True)
    assert len(inputs.rows)==6 and len(inputs.bundle.records('train'))==2
    item=inputs.records['train_negative']
    with h5py.File(item['path'],'r+') as f:f['pixels'][0,0,0,0]=255-f['pixels'][0,0,0,0]
    with pytest.raises(ValueError,match='RGB bytes'):inputs.check(deep=True)
    config=yaml.safe_load((path/'task.yaml').read_text())
    del config['encoders']['rgb']['nciemil_raw']['weak_targets']
    (path/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='mapping'):FilterInputs(path)


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:tree_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):tree_equal(x,y)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a==b


def test_native_swin_filter_train_only_interruption_restores_all_weights_adamw_moments_rng_and_winners(tmp_path):
    torch.set_num_threads(2);data=raw_fixture(tmp_path/'data')
    kwargs={'epochs':2,'lr':2e-4,'weight_decay':1e-5,'bag_batch_size':2,'seed':8906}
    train_filter(data,ASSET,WEIGHTS_SHA256,tmp_path/'continuous',**kwargs)
    train_filter(data,ASSET,WEIGHTS_SHA256,tmp_path/'resumed',stop_after=1,**kwargs)
    train_filter(data,ASSET,WEIGHTS_SHA256,tmp_path/'resumed',resume=True,**kwargs)
    a=torch.load(tmp_path/'continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'resumed/last.pt',map_location='cpu',weights_only=False)
    assert a['completed_epochs']==2 and a['updates']==2
    assert len(a['optimizer']['state'])==len(list(SwinFilter(ASSET,WEIGHTS_SHA256).parameters()))
    for epoch in a['history']:
        assert set(epoch['order'])=={'train_negative','train_positive'}
        assert all(t['patches_scanned']==2 for t in epoch['traces'])
    tree_equal(a,b)
    with pytest.raises(ValueError,match='recipe'):
        train_filter(data,ASSET,WEIGHTS_SHA256,tmp_path/'resumed',resume=True,**{**kwargs,'lr':1e-4})

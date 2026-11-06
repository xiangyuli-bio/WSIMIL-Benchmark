"""Original complete DT-MIL, physical data, actual optimizer recipe and recovery."""
import copy
import math
import os.path as osp
import pickle
import re
import types

import h5py
import numpy as np
import pytest
import torch
import torch.utils.data as data_utils
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from milplus.data import Bundle
from milplus.dtmil import DTMIL
from milplus.dtmil_data import augmentation, encode_map
from milplus.dtmil_demo import create_dtmil_demo
from milplus.dtmil_network import REPAIRED, SOURCE, author_modules, network
from milplus.engine import evaluate, restore_rng, rng_state, train
from milplus.io import read_json
from milplus.registry import check_compatibility, inspect_method
from milplus.upstream import load_definitions


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def exact(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def original_valid_ratio():
    module=load_definitions('dtmil','models/deformable_transformer.py',['DeformableTransformer'],{'torch':torch,'nn':nn})
    return module.DeformableTransformer.get_valid_ratio


def bilinear(value,shapes,loc,weight):
    b,_,heads,d=value.shape;_,q,_,_,points,_=loc.shape
    result=value.new_zeros((b,q,heads,d));offset=0
    bb=torch.arange(b)[:,None,None,None];hh=torch.arange(heads)[None,None,:,None]
    for level,(height,width) in enumerate(shapes.tolist()):
        x=loc[:,:,:,level,:,0]*width-.5;y=loc[:,:,:,level,:,1]*height-.5
        x0,y0=x.floor(),y.floor();sample=value.new_zeros((b,q,heads,points,d))
        for ix,iy,w in [(x0,y0,(x0+1-x)*(y0+1-y)),(x0+1,y0,(x-x0)*(y0+1-y)),
                        (x0,y0+1,(x0+1-x)*(y-y0)),(x0+1,y0+1,(x-x0)*(y-y0))]:
            valid=(ix>=0)&(ix<width)&(iy>=0)&(iy<height)
            indices=offset+iy.long().clamp(0,height-1)*width+ix.long().clamp(0,width-1)
            sample=sample+value[bb,indices,hh,:]*(w*valid)[...,None]
        result=result+(sample*weight[:,:,:,level,:,None]).sum(3);offset+=height*width
    return result.reshape(b,q,heads*d)


def original_dataset(root,bundle,train_mode,map_size):
    aug,transform=augmentation()
    mod=load_definitions('dtmil','datasets/wsi_feat_dataset.py',
        ['parse_patch_fname','WSIFeatDataset'],{'torch':torch,'np':np,'re':re,'pickle':pickle,
        'osp':osp,'Tuple':tuple,'data_utils':data_utils,'tensor_aug':aug})
    root.mkdir();lines=[]
    for i,row in enumerate(bundle.records('train')):
        path=root/(str(i)+'.pkl')
        with h5py.File(bundle.root/row['path'],'r') as f:
            features,coords=f['features'][:],f['coords'][:]
            views=f['train_views'][:] if 'train_views' in f else None
        records=[]
        for j,(value,coord) in enumerate(zip(features,coords)):
            item={'feat_name':f's-tile-r{coord[1]}-c{coord[0]}-256x256.npy','val':value}
            if views is not None:item['tr']=views[j]
            records.append(item)
        with path.open('wb') as f:pickle.dump(records,f)
        lines.append(f'{row["slide_id"]},{path},{row["target"]}')
    ann=root/'index.txt';ann.write_text('\n'.join(lines)+'\n')
    return mod.WSIFeatDataset(str(ann),transforms=transform.TrFeatureMapAug() if train_mode else transform.ValFeatureMapAug(),
                             num_class=2,is_training=train_mode,feat_map_size=map_size)


def test_exact_author_deformable_operator_vs_independent_bilinear_all_gradients():
    torch.manual_seed(79);shapes=torch.tensor([[3,4],[2,3]])
    inputs=[torch.randn(2,18,2,3,dtype=torch.float64),torch.rand(2,5,2,2,3,2,dtype=torch.float64)*1.4-.2,
            torch.randn(2,5,2,2,3,dtype=torch.float64)]
    a=[x.clone().requires_grad_() for x in inputs];b=[x.clone().requires_grad_() for x in inputs]
    op=author_modules()[0]['ms_deform_attn_core_pytorch']
    x=op(a[0],shapes,a[1],a[2]);y=bilinear(b[0],shapes,b[1],b[2]);torch.testing.assert_close(x,y,rtol=2e-13,atol=2e-13)
    probe=torch.randn_like(x)
    for u,v in zip(torch.autograd.grad((x*probe).sum(),a),torch.autograd.grad((y*probe).sum(),b)):
        torch.testing.assert_close(u,v,rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('dimension,classes',[(1280,2),(1024,3),(1536,3)])
def test_complete_native60map_256hidden_6encoder6decoder_100query_source_outputs_all_gradients(dimension,classes):
    torch.manual_seed(81);model,criterion=network(dimension,classes)
    original=copy.deepcopy(model)
    original.transformer.get_valid_ratio=types.MethodType(original_valid_ratio(),original.transformer)
    x=torch.randn(1,dimension,60,60);mask=torch.zeros(1,60,60,dtype=torch.bool)
    a=x.clone().requires_grad_();b=x.clone().requires_grad_();state=rng_state()
    expected=original(types.SimpleNamespace(tensors=a,mask=mask));loss1=criterion(expected,torch.tensor([classes-1]));loss1.backward()
    restore_rng(state)
    actual=model(types.SimpleNamespace(tensors=b,mask=mask));loss2=criterion(actual,torch.tensor([classes-1]));loss2.backward()
    exact(expected,actual);exact(a.grad,b.grad)
    assert model.query_embed.weight.shape==(100,256)
    assert len(model.transformer.encoder.layers)==len(model.transformer.decoder.layers)==6
    inactive=[]
    for (name,p),(name2,q) in zip(model.named_parameters(),original.named_parameters()):
        assert name==name2
        if p.grad is None:assert q.grad is None;inactive.append(name)
        else:exact(p.grad,q.grad);assert torch.isfinite(p.grad).all()
    assert set(inactive)=={'transformer.reference_points.weight','transformer.reference_points.bias'}


def test_actual_source_crop_and_cached_train_views_match_while_test_does_not_use_views(tmp_path):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=4,cached_views=True,grid_shape=(10,10)))
    source=DTMIL(b,profile=SOURCE,map_size=8,batch_size=4)
    ds=original_dataset(tmp_path/'original',b,True,8)
    np.random.seed(11);a,mask,_=ds[0]
    np.random.seed(11);x,y,meta=source.assets.row(b.records('train')[0],True)
    exact(a,x);exact(mask,y);assert meta['cropped_shape']==[1,8]  # Source stale10 crop on the already8row canvas.
    fixed=DTMIL(b,map_size=8,batch_size=4)
    _,_,corrected=fixed.assets.row(b.records('train')[0],False)
    assert corrected['cropped_shape']==[8,8] and corrected['retained_rows']==64
    row=b.records('test')[0]
    a,ma,one=source.assets.row(row,False);np.random.seed(999);b2,mb,two=source.assets.row(row,False)
    exact(a,b2);exact(ma,mb);assert one['views_sha256']==two['views_sha256']


def test_center_padding_zero_ratio_failure_and_explicit_finite_grid_port(tmp_path):
    model,_=network(32,2);mask=torch.ones(1,8,8,dtype=torch.bool);mask[:,2:5,2:6]=False
    old=original_valid_ratio()(model.transformer,mask)
    assert torch.equal(old,torch.zeros_like(old))
    fixed=model.transformer.get_valid_ratio(mask);exact(fixed,torch.ones_like(fixed))
    cls=type(model.transformer.encoder);shapes=torch.tensor([[8,8]])
    assert not torch.isfinite(cls.get_reference_points(shapes,old[:,None],'cpu')).all()
    assert torch.isfinite(cls.get_reference_points(shapes,fixed[:,None],'cpu')).all()
    x=torch.randn(1,32,8,8,requires_grad=True)
    loss=model(types.SimpleNamespace(tensors=x,mask=mask)).sum();loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_actual_original_dataset_minibatch_transforms_adamw_lr_groups_clip_and_step_match(tmp_path):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=32,cached_views=True))
    m=DTMIL(b,profile=SOURCE,map_size=8,batch_size=4,lr_drop=1)
    cfg={'epochs':3,'patience':3,'selection_metric':'log_loss','grad_clip':.1,'max_instances':None}
    m.configure_training(cfg);opt=m.make_optimizer(2e-4,1e-4);m.start_epoch(0)
    manual=copy.deepcopy(m.network);params=list(manual.named_parameters())
    o=torch.optim.AdamW([
        {'params':[p for n,p in params if 'backbone.0' not in n and not any(s in n for s in ['reference_points','sampling_offsets'])],'lr':2e-4,'group_name':'main'},
        {'params':[p for n,p in params if 'backbone.0' in n],'lr':2e-5,'group_name':'backbone'},
        {'params':[p for n,p in params if any(s in n for s in ['reference_points','sampling_offsets'])],'lr':2e-4*.1,'group_name':'projection'}],lr=2e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(o,step_size=1,gamma=.1)
    ds=original_dataset(tmp_path/'original',b,True,8);state=rng_state()
    m.train();m.epoch(b,opt,'cpu');m.finish_epoch(b,'cpu',cfg);after=rng_state()
    restore_rng(state);manual.train()
    loader=DataLoader(ds,batch_size=4,sampler=RandomSampler(ds),num_workers=0,collate_fn=list)
    for rows in loader:
        maps,masks,targets=zip(*rows)
        sample=types.SimpleNamespace(tensors=torch.stack(maps),mask=torch.stack(masks))
        o.zero_grad(set_to_none=True);out=manual(sample)
        F.cross_entropy(out,torch.stack([t['label'] for t in targets])).backward()
        torch.nn.utils.clip_grad_norm_(manual.parameters(),.1,error_if_nonfinite=True);o.step()
    scheduler.step();exact(m.network.state_dict(),manual.state_dict());exact(opt.state_dict(),o.state_dict());exact(after,rng_state())
    assert m.optimizer_steps==math.ceil(len(b.records('train'))/4)
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([2e-5,2e-6,2e-6],rel=1e-15,abs=0)


@pytest.mark.parametrize('profile',[SOURCE,REPAIRED])
def test_three_epoch_complete_model_optimizer_bn_scheduler_order_all_rng_exact_resume(tmp_path,profile):
    data=create_dtmil_demo(tmp_path/'data',dimension=32,cached_views=True)
    cfg={'epochs':3,'threads':1,'model':{'map_size':8,'batch_size':4,'lr_drop':2}}
    full,resume=tmp_path/'full',tmp_path/'resume'
    train(data,profile,full,seed=17,config=cfg)
    train(data,profile,resume,seed=17,config=cfg,stop_after=1)
    assert read_json(resume/'status.json')['state']=='interrupted'
    train(data,profile,resume,seed=17,config=cfg,resume=True)
    a,b=[torch.load(p/'last.pt',map_location='cpu',weights_only=False) for p in [full,resume]]
    for key in ['model','optimizers','history','rng','best','bad']:exact(a[key],b[key])
    state=a['model']['_extra_state'];assert state['completed_epochs']==3 and state['scheduler']['last_epoch']==3
    assert not (full/'test_predictions.csv').exists()
    evaluate(full);prediction=(full/'test_predictions.csv').read_bytes();evaluate(full)
    assert (full/'test_predictions.csv').read_bytes()==prediction


@pytest.mark.parametrize('change',[{'max_instances':3},{'grad_clip':None},{'patience':1},{'selection_metric':'accuracy'}])
def test_incomplete_reference_recipe_rejected(tmp_path,change):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=8));m=DTMIL(b,map_size=8,batch_size=4)
    with pytest.raises(ValueError):m.configure_training({'epochs':3,'patience':3,'selection_metric':'log_loss','max_instances':None,'grad_clip':.1,**change})


def test_physical_input_and_memory_guards_never_fabricate_grid_or_reduce_model(tmp_path):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=8));m=DTMIL(b,map_size=8,batch_size=4)
    with pytest.raises(ValueError,match='before allocation'):DTMIL(b,max_dense_bytes=1)
    x=torch.ones(2,8)
    for coords in [torch.tensor([[0.,0.],[100.,0.]]),torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([[-1.,0.],[255.,0.]])]:
        with pytest.raises(ValueError):encode_map(x,coords,256,8,True)
    with pytest.raises(ValueError):m.forward(x,coords=None)
    cfg=yaml.safe_load((b.root/'task.yaml').read_text());cfg['encoders']['toy'].pop('spatial')
    (b.root/'task.yaml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):check_compatibility(inspect_method('DT-MIL'),Bundle(b.root))


def test_source_binary_contract_and_alias_do_not_shadow_other_mil(tmp_path):
    b=Bundle(create_dtmil_demo(tmp_path/'data',patients=30,classes=3,dimension=8))
    with pytest.raises(ValueError,match='binary'):DTMIL(b,profile=SOURCE,map_size=8,batch_size=4)
    assert inspect_method('DT-MIL')['id']==REPAIRED
    assert inspect_method('DTFD-MIL')['id']=='dtfd'
    assert inspect_method('TransMIL')['id']=='transmil'


def test_checkpoint_rejects_lr_horizon_counts_train_identity_and_changed_features(tmp_path):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=8));m=DTMIL(b,map_size=8,batch_size=4)
    m.configure_training({'epochs':3,'patience':3,'grad_clip':.1,'selection_metric':'log_loss','max_instances':None})
    state=m.state_dict()
    for change in [{'total_epochs':4},{'completed_epochs':1},{'optimizer_steps':1},{'gradient_clip':float('nan')}]:
        e=copy.deepcopy(state['_extra_state']);e.update(change)
        with pytest.raises(ValueError):m.load_state_dict({**state,'_extra_state':e})
    path=b.root/b.records('train')[0]['path']
    with h5py.File(path,'r+') as f:f['features'][0]+=1
    changed=DTMIL(Bundle(b.root),map_size=8,batch_size=4)
    with pytest.raises(ValueError,match='contract differs'):changed.load_state_dict(state)

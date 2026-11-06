"""SETMIL: independent author model, actual WSIFeatDataset and whole training loop."""
import ast
import json
import math
import os.path as osp
import pickle
import re
import subprocess
import types
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from functools import wraps
from typing import Optional

import h5py
import numpy as np
import pytest
import torch
import torch.utils.data as data_utils
import yaml
from sklearn import metrics
from torch import Tensor, nn
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train, units
from milplus.setmil import SETMIL, encode_map, physical_indices, setmil_epoch
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def fixture(tmp_path,profile='setmil_repaired',**opts):
    root=create_demo(tmp_path/'data',patients=30,classes=3,dimension=8);b=Bundle(root)
    cfg={'map_size':12,'token_dim':8,'embed_dim':8,'depth':2,'heads':4,'batch_size':4,**opts}
    return SETMIL(b,profile,**cfg),b,cfg


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def source(path):
    root,sha=verified_source('setmil')
    return subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)


def definitions(path,names,scope,replacements=()):
    code=source(path)
    for old,new in replacements:
        assert old in code;code=code.replace(old,new)
    tree=ast.parse(code);nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent verified source
    return scope


def independent_network(model):
    class Config(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    def full(path,extra):
        tree=ast.parse(source(path));tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.Try,ast.If))]
        scope={'torch':torch,'nn':nn,'np':np,'math':math,'edict':Config,'RPEIndexFunction':None,
               'trunc_normal_':nn.init.trunc_normal_,**extra}
        exec(compile(tree,path,'exec'),scope)  # noqa: S102 -- source classes independent of runtime loader
        return scope
    irpe=full('trans/models/t2t_module/irpe.py',{})
    block=full('trans/models/t2t_module/transformer_block.py',{'build_rpe':irpe['build_rpe'],'get_rpe_config':irpe['get_rpe_config']})
    token=full('trans/models/t2t_module/token_transformer.py',{'build_rpe':irpe['build_rpe'],'get_rpe_config':irpe['get_rpe_config'],'Mlp':block['Mlp']})
    module=full('trans/models/irpe_t2t.py',{'Token_transformer':token['Token_transformer'],'Block':block['Block'],'get_sinusoid_encoding':block['get_sinusoid_encoding']})
    net=module['T2T_ViT_iRPE'](img_size=model.map_size,in_chans=model.dim,num_classes=len(model.bundle.classes),
        token_dim=model.token_dim,embed_dim=model.embed_dim,depth=model.depth,num_heads=model.heads,
        drop_rate=model.dropout,attn_drop_rate=model.attention_dropout,channel_reduce_rate=model.reduction,
        irpe={'rpe':1,'method':'euc','mode':'bias','shared_head':False,'rpe_on':'q'},
        token_t={'rpe':0,'drop':model.dropout,'attn_drop':model.attention_dropout,'method':'euc','mode':'bias','shared_head':False,'rpe_on':'q'})
    net.load_state_dict(model.network.state_dict());return net,irpe


def author_data(tmp_path,model,views=False,coords=None,profile='source',subset='train'):
    scope={'np':np,'torch':torch,'data_utils':data_utils,'pickle':pickle,'re':re,'osp':osp,'Tuple':tuple}
    aug=definitions('trans/util/numpy_img_aug.py',['check_size','center_crop','random_crop','horizontal_flip','vertical_flip',
        'scale_augmentation','cutout','pad_image_and_mask'],scope)
    scope['tensor_aug']=types.SimpleNamespace(**aug)
    replacements=[]
    if profile=='repaired':
        replacements.append(('tensor_aug.center_crop(feat_map, (h, self.feat_map_size))','tensor_aug.center_crop(feat_map, (feat_map.shape[0], self.feat_map_size))'))
    definitions('trans/datasets/wsi_feat_dataset.py',['parse_patch_fname_origin','WSIFeatDataset','TrFeatureMapAug','ValFeatureMapAug'],scope,replacements)
    lines=[];rows=model.bundle.records(subset)
    for idx,row in enumerate(rows):
        x,c=model.bundle.load(row,require_coords=True)
        if coords is not None:c=coords
        records=[]
        for j,(v,(cx,cy)) in enumerate(zip(x.numpy(),c.numpy())):
            obj={'feat_name':f'tile-r{int(cy)}-c{int(cx)}-256x256','val':v}
            if views:
                obj['tr']=np.stack([v+10.,v-10.])
            records.append(obj)
        path=tmp_path/f'{idx}.pkl';path.write_bytes(pickle.dumps(records));lines.append(f'{row["slide_id"]},{path},{row["target"]}')
    ann=tmp_path/'ann.txt';ann.write_text('\n'.join(lines))
    transform=scope['TrFeatureMapAug']() if subset=='train' else scope['ValFeatureMapAug']()
    ds=scope['WSIFeatDataset'](ann_file=str(ann),transforms=transform,num_class=3,feat_map_size=model.map_size)
    return ds,scope


@pytest.mark.parametrize('training',[False,True])
def test_native_model_outputs_losses_all_gradients(tmp_path,training):
    root=create_demo(tmp_path/'data',patients=30,classes=3,dimension=1280)
    m=SETMIL(Bundle(root));ref,_=independent_network(m);m.train(training);ref.train(training)
    assert sum(p.numel() for p in m.network.parameters())==29075587
    sample=types.SimpleNamespace(tensors=torch.randn(1,1280,48,48));target=torch.tensor([2]);state=rng_state()
    a=m.batch_forward(sample,labels=target);a['total_loss'].backward()
    restore_rng(state);out=ref(sample);loss=nn.functional.cross_entropy(out,target);loss.backward()
    equal(a['logits'],out);equal(a['total_loss'],loss)
    for p,q in zip(m.network.parameters(),ref.parameters()):
        if p.grad is not None:equal(p.grad,q.grad)
    assert m.network.pos_embed.shape==(1,257,192) and m.network.blocks[0].attn.rpe_q.lookup_table_bias.shape==(16,8)
    assert all(k in m.network.state_dict() for k in ['aspp.p1.attention.attn.qkv.weight','aspp.p2.attention.attn.qkv.weight','aspp.p3.attention.attn.qkv.weight'])


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('views',[False,True])
def test_actual_author_dataset_ppe_all_augmentations_and_rng(tmp_path,training,views):
    m,b,_=fixture(tmp_path);ds,_=author_data(tmp_path,m,views=views,subset='train' if training else 'val')
    row=b.records('train' if training else 'val')[0];x,c=b.load(row)
    v=np.stack([x.numpy()+10,x.numpy()-10],axis=1) if views else None
    for seed in range(20):
        seed_everything(seed);ref,mask,_=ds[0];state=rng_state()
        seed_everything(seed);actual,actualmask,_=encode_map(x,c,256,12,False,training,v)
        equal(actual,ref);equal(actualmask,mask);equal(rng_state(),state)


def test_real_lattice_preserves_holes_and_axis_order(tmp_path):
    coords=torch.tensor([[4096.,1024.],[4608.,1280.],[4096.,1280.]],dtype=torch.float64)
    x=torch.tensor([[1.],[2.],[3.]])
    result,mask,meta=encode_map(x,coords,256,12,True)
    assert result[0,5,4]==1 and result[0,6,6]==2 and result[0,6,4]==3 and result[0,5,5]==0
    assert result.sum()==6 and mask.sum()==12*12-2*3
    assert meta['source_canvas_shape']==[2,3] and meta['retained_before_augmentation']==3
    equal(physical_indices(coords,256),np.array([[0,0],[1,2],[1,0]]))


def test_actual_author_both_dimension_crop_bug_and_explicit_repair(tmp_path):
    m,b,_=fixture(tmp_path,map_size=48)
    coords=torch.tensor([[0.,0.],[69*256.,59*256.],[35*256.,30*256.],[34*256.,54*256.]])
    row=b.records('val')[0]
    with h5py.File(b.root/row['path'],'w') as h:
        h['features']=np.arange(4*8,dtype=np.float32).reshape(4,8);h['coords']=coords.numpy()
    for profile,repaired,shape in [('source',False,[6,48]),('repaired',True,[48,48])]:
        d=tmp_path/profile;d.mkdir();ds,_=author_data(d,m,coords=None,profile=profile,subset='val')
        x,c=b.load(row);a,mask,meta=encode_map(x,c,256,48,repaired);ref,refmask,_=ds[0]
        equal(a,ref);equal(mask,refmask);assert meta['cropped_shape']==shape
    assert encode_map(x,c,256,48,True)[2]['retained_before_augmentation']!=encode_map(x,c,256,48,False)[2]['retained_before_augmentation']


def author_epoch(scope):
    import datetime
    import time
    from collections.abc import Sequence

    from torch import distributed as dist
    utils={'torch':torch,'np':np,'defaultdict':defaultdict,'deque':deque,'datetime':datetime,'time':time,'dist':dist}
    definitions('trans/util/misc.py',['SmoothedValue','MetricLogger','is_dist_avail_and_initialized','get_world_size','reduce_dict'],utils)
    class DummyStream:
        def wait_stream(self,other):pass
    definitions('trans/util/misc.py',['NestedTensor','collate_fn'],scope,[('self.tensors.record_stream(*args, **kwargs)','pass'),('self.mask.record_stream(*args, **kwargs)','pass')])
    pref={'torch':torch,'nullcontext':nullcontext,'DummyStream':DummyStream}
    definitions('trans/datasets/data_prefetcher.py',['to_cuda','data_prefetcher'],pref,[('torch.cuda.Stream()','DummyStream()'),('torch.cuda.current_stream()','DummyStream()'),('torch.cuda.stream(self.stream)','nullcontext()')])
    gather={'Callable':Callable,'Sequence':Sequence,'wraps':wraps,'defaultdict':defaultdict,'torch':torch,'np':np,'dist':dist}
    definitions('trans/util/gpu_gather.py',['reinit__is_reduced','GpuGather'],gather)
    env={'torch':torch,'nn':nn,'np':np,'Iterable':Iterable,'utils':types.SimpleNamespace(**utils),'metrics':metrics,
         'GpuGather':gather['GpuGather'],'data_prefetcher':pref['data_prefetcher']}
    definitions('trans/engine.py',['train_one_epoch'],env,[('label.cuda()','label.to(device)')])
    return env['train_one_epoch']


@pytest.mark.parametrize('profile',['setmil_source','setmil_repaired'])
def test_full_author_engine_epochs_adamw_clipping_scheduler_and_rng(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile);ref,_=independent_network(m)
    ds,scope=author_data(tmp_path,m,profile='repaired' if profile.endswith('repaired') else 'source')
    scope.update(Optional=Optional,Tensor=Tensor);epoch=author_epoch(scope)
    opt=m.make_optimizer(.0002,.05);refopt=torch.optim.AdamW(ref.parameters(),lr=.0002,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.StepLR(refopt,100)
    # Exercise the native milestone without running one hundred redundant synthetic epochs.
    m.scheduler.last_epoch=scheduler.last_epoch=98;m.scheduler._step_count=scheduler._step_count=99
    count=0
    for idx in [98,99,100]:
        m.start_epoch(idx);m.train();state=rng_state()
        actual=setmil_epoch(m,b,units(b,'train'),opt,'cpu',{'grad_clip':.01});m.finish_epoch(b,'cpu',{})
        after=rng_state();restore_rng(state)
        loader=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0,collate_fn=scope['collate_fn'])
        logger=types.SimpleNamespace(info=lambda *a:None)
        epoch(logger,ref,nn.CrossEntropyLoss(),loader,refopt,torch.device('cpu'),idx,.01)
        scheduler.step();equal(after,rng_state());equal(m.network.state_dict(),ref.state_dict());equal(opt.state_dict(),refopt.state_dict());equal(m.scheduler.state_dict(),scheduler.state_dict())
        for p,q in zip(m.network.parameters(),ref.parameters()):
            if p.grad is not None:equal(p.grad,q.grad)
        count+=len(actual)
    assert m.optimizer_steps==count and opt.param_groups[0]['lr']==.00002
    (tmp_path/'setmil_author_training.json').write_text(json.dumps({'profile':profile,'state':'passed','optimizer_updates':count,'epoch_indices':[98,99,100],
        'max_parameter_abs':0.,'scope':'Full original engine.train_one_epoch, actual WSIFeatDataset with source or explicitly corrected crop, original AdamW/clip/StepLR and prefetch consumption; CPU stream compatibility only. Bitwise model/gradient/AdamW/scheduler/RNG.'}))


@pytest.mark.parametrize('profile',['setmil_source','setmil_repaired'])
def test_exact_resume_best_reload_evaluate_and_crop_artifacts(tmp_path,profile):
    _m,b,opts=fixture(tmp_path,profile,lr_drop=2)
    cfg={'epochs':3,'threads':1,'model':opts}
    run=train(b.root,profile,tmp_path/'full',config=cfg)
    interrupted=train(b.root,profile,tmp_path/'resumed',config=cfg,stop_after=1)
    train(b.root,profile,interrupted,config=cfg,resume=True)
    a=torch.load(run/'last.pt',weights_only=False);other=torch.load(interrupted/'last.pt',weights_only=False)
    for key in ['model','optimizers','rng','best','bad','history']:equal(a[key],other[key])
    assert a['model']['_extra_state']['scheduler_state']['last_epoch']==3
    assert a['optimizers'][0]['param_groups'][0]['lr']==.00002
    first=evaluate(run);csv=(run/'test_predictions.csv').read_bytes();assert evaluate(run)==first and csv==(run/'test_predictions.csv').read_bytes()
    import pandas as pd
    frame=pd.read_csv(run/'test_predictions.csv');assert (frame.ppe_total_patches>=frame.ppe_retained_before_augmentation).all()
    assert 'setmil_bag_auc' in first and not json.loads((run/'status.json').read_text())['test_evaluated']
    protocol=json.loads((run/'method_protocol.json').read_text());assert protocol['optimizer_steps']==3*math.ceil(len(b.records('train'))/4)
    assert a['epoch']==2 and all(int(v['step'])==protocol['optimizer_steps'] for v in a['optimizers'][0]['state'].values())


@pytest.mark.parametrize('profile',['setmil_source','setmil_repaired'])
def test_query_labels_are_not_inputs_and_ppe_is_permutation_invariant(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile);m.eval();row=b.records('test')[0];x,c=b.load(row)
    state=rng_state();a=m(x,c,label=torch.tensor([0]));restore_rng(state);z=m(x,c,label=torch.tensor([2]));equal(a['logits'],z['logits'])
    equal(m(x.flip(0),c.flip(0))['logits'],a['logits'])
    shifted=c.double()+torch.tensor([2**31,2**32],dtype=torch.float64)
    equal(m(x,shifted)['logits'],a['logits']);equal(m(x,c)['logits'],a['logits'])


def test_cached_views_are_hashed_and_source_vs_repaired_eval_is_explicit(tmp_path):
    m,b,opts=fixture(tmp_path)
    p=b.root/'task.yaml';cfg=yaml.safe_load(p.read_text());cfg['encoders']['toy']['setmil_cached_views']={'dataset':'views','provenance':'synthetic two-view source fixture'};p.write_text(yaml.safe_dump(cfg))
    for row in b.rows.to_dict('records'):
        with h5py.File(b.root/row['path'],'a') as h:
            h['views']=np.stack([h['features'][:]+10,h['features'][:]-10],1)
    b=Bundle(b.root);m=SETMIL(b,**opts);s=SETMIL(b,'setmil_source',**opts);row=b.records('test')[0]
    assert not m.map_row(row,False)[2]['cached_views_used'] and m.map_row(row,True)[2]['cached_views_used']
    assert s.map_row(row,False)[2]['cached_views_used']
    before=b.check(deep=True)['feature_inventory_sha256']
    with h5py.File(b.root/row['path'],'a') as h:h['views'][0,0,0]+=1
    assert b.check(deep=True)['feature_inventory_sha256']!=before


@pytest.mark.parametrize('coords',[torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([[0.,0.],[128.,256.]]),torch.tensor([[0.,0.],[float('nan'),256.]])])
def test_invalid_or_false_lattice_rejected(coords):
    with pytest.raises(ValueError):encode_map(torch.ones(2,3),coords,256,12,True)


def test_dense_canvas_guard_before_allocation(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('allocation before budget check')
    monkeypatch.setattr(np,'zeros',forbidden)
    with pytest.raises(ValueError,match='physical canvas'):
        encode_map(torch.ones(2,8),torch.tensor([[0.,0.],[256e8,256e8]],dtype=torch.float64),256,12,True,max_bytes=1024**2)


def test_model_guard_before_construction(tmp_path,monkeypatch):
    _,b,_=fixture(tmp_path)
    import milplus.setmil as module
    monkeypatch.setattr(module,'load_network',lambda:(_ for _ in ()).throw(AssertionError('constructor called')))
    with pytest.raises(ValueError,match='before allocation'):SETMIL(b,embed_dim=1000000,heads=1)


@pytest.mark.parametrize('options',[{'map_size':0},{'map_size':47},{'reduction':4},{'heads':7},{'token_dim':True},{'dropout':float('nan')},{'attention_dropout':1.},{'batch_size':0},{'lr_drop':0}])
def test_constructor_guards(tmp_path,options):
    _,b,_=fixture(tmp_path)
    with pytest.raises(ValueError):SETMIL(b,**options)


@pytest.mark.parametrize('changes',[{'max_instances':10},{'selection_metric':'macro_auc'},{'grad_clip':None},{'grad_clip':0}])
def test_training_contract_guards(tmp_path,changes):
    m,_,_=fixture(tmp_path)
    cfg={'max_instances':None,'selection_metric':'setmil_bag_auc','grad_clip':.01,**changes}
    with pytest.raises(ValueError):m.configure_training(cfg)


def test_validation_selection_is_slide_not_patient_auc(tmp_path):
    m,_,_=fixture(tmp_path)
    import pandas as pd
    f=pd.DataFrame({'target':[0,1,2,0,1,2],'p_0':[.6,.1,.2,.4,.3,.5],'p_1':[.2,.6,.2,.3,.4,.1],'p_2':[.2,.3,.6,.3,.3,.4]})
    ref=np.mean([metrics.roc_auc_score(f.target==c,f['p_'+str(c)]) for c in range(3)])
    assert m.validation_metrics(f)=={'setmil_bag_auc':ref}
    with pytest.raises(ValueError):m.validation_metrics(f.iloc[[0,1,3,4]])


@pytest.mark.parametrize('invalid',[torch.empty(2,0),torch.ones(2,3,dtype=torch.long),torch.full((2,3),float('inf'))])
def test_invalid_features_are_rejected(invalid):
    with pytest.raises(ValueError):encode_map(invalid,torch.tensor([[0.,0.],[256.,0.]]),256,12,True)


@pytest.mark.parametrize('labels',[torch.tensor([1.]),torch.tensor([-1]),torch.tensor([3]),torch.tensor([0,1])])
def test_invalid_labels_do_not_consume_dropout_rng(tmp_path,labels):
    m,_,_=fixture(tmp_path);sample=types.SimpleNamespace(tensors=torch.zeros(1,8,12,12));state=rng_state()
    with pytest.raises(ValueError,match='labels'):m.batch_forward(sample,labels=labels)
    equal(state,rng_state())


def test_padding_mask_is_ignored_as_author_and_irpe_is_trainable(tmp_path):
    m,_,_=fixture(tmp_path);m.eval();sample=types.SimpleNamespace(tensors=torch.randn(2,8,12,12),mask=torch.ones(2,12,12))
    a=m.batch_forward(sample,labels=torch.tensor([0,2]));a['total_loss'].backward()
    for block in m.network.blocks:assert block.attn.rpe_q.lookup_table_bias.grad.abs().sum()>0
    sample.mask.zero_();equal(m.batch_forward(sample)['logits'],a['logits'])


def test_physical_metadata_and_single_slide_contract(tmp_path):
    _,b,opts=fixture(tmp_path);b.encoder['spatial']['patch_stride']=0
    with pytest.raises(ValueError,match='positive'):SETMIL(b,**opts)
    del b.encoder['spatial'];
    with pytest.raises(ValueError,match='Spatial'):SETMIL(b,**opts)
    b.config['training_unit']='patient'
    with pytest.raises(ValueError,match='individual-slide'):SETMIL(b,**opts)


def test_binary_selection_matches_author_auc(tmp_path):
    root=create_demo(tmp_path/'data',patients=20,classes=2,dimension=8);m=SETMIL(Bundle(root),map_size=12,token_dim=8,embed_dim=8,depth=1,heads=4)
    import pandas as pd
    frame=pd.DataFrame({'target':[0,0,1,1],'p_0':[.7,.8,.6,.1],'p_1':[.3,.2,.4,.9]})
    assert m.validation_metrics(frame)=={'setmil_bag_auc':metrics.roc_auc_score(frame.target,frame.p_1)}


def test_declared_cached_views_missing_or_misaligned_refused(tmp_path):
    _,b,opts=fixture(tmp_path);b.encoder['setmil_cached_views']={'dataset':'missing','provenance':'synthetic'}
    m=SETMIL(b,**opts);row=b.records('train')[0]
    with pytest.raises(ValueError,match='missing'):m.map_row(row,True)
    with h5py.File(b.root/row['path'],'a') as h:h['missing']=np.ones((2,8))
    with pytest.raises(ValueError,match='aligned'):m.map_row(row,True)


def test_evaluation_rejects_feature_or_view_tampering(tmp_path):
    _,b,opts=fixture(tmp_path)
    run=train(b.root,'SETMIL',tmp_path/'run',config={'epochs':1,'threads':1,'model':opts})
    checkpoint=torch.load(run/'last.pt',weights_only=False)
    group=checkpoint['optimizers'][0]['param_groups'][0]
    assert group['weight_decay']==.05 and group['initial_lr']==.0002
    row=b.records('test')[0]
    with h5py.File(b.root/row['path'],'a') as h:h['features'][0,0]+=1
    with pytest.raises(ValueError,match='inputs differ'):evaluate(run)

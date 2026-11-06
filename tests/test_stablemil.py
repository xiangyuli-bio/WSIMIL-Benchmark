"""StableMIL: original physical preparation, complete networks and both original task loops."""
import ast
import collections.abc
import json
import math
import subprocess
from functools import partial
from itertools import repeat
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.metrics import harrell_c_index
from milplus.smmile_data import load_spatial_inputs
from milplus.stable_demo import create_stable_demo
from milplus.stable_geometry import GeometryCache, build_geometry
from milplus.stablemil import PROFILES, StableMIL, stable_epoch
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def fixture(tmp_path,profile='stablemil_classification_repaired',**opts):
    survival='survival' in profile
    root=create_stable_demo(tmp_path/'data',patients=30,dimension=8,classes=3,survival=survival)
    b=Bundle(root);options={'hidden':32,'depth':2,'heads':4,**opts}
    return StableMIL(b,profile,**options),b,options


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def definitions(path,names,scope,source='stablemil',patches=()):
    root,sha=verified_source(source);code=subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)
    for old,new in patches:
        assert old in code;code=code.replace(old,new)
    nodes=[n for n in ast.parse(code).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent verified source objects
    return scope


def native_geometry(coords,reference_size=512,ratio=2,aggregate_num=256,k_neighbors=8,radius=6*2**.5):
    s={'torch':torch,'np':np,'ConvexHull':ConvexHull,'NearestNeighbors':NearestNeighbors}
    definitions('models/token_fuse.py',['create_bounding_box','assign_labels_to_regions','fuse_labels'],s)
    definitions('models/find_Region.py',['minimum_bounding_rectangle','distance_to_line_vectorized','assign_coords_to_regions_vectorized','find_region_ag'],s)
    definitions('classification/preprocess/preprocess.py',['fuse_token','region_sort','create_tokens'],s)
    label,order,fused=s['fuse_token'](np.floor(coords.detach().cpu().numpy()/reference_size).astype(np.int64),ratio)
    region,region_order=s['region_sort'](fused,aggregate_num)
    index,out=s['create_tokens'](fused[region_order],k_neighbors,radius)
    return {'fuse_labels':label,'fuse_sorted_idx':order,'fuse_coords':fused,'region_indices':region,
            'region_sorted_index':region_order,'attention_mask_1':index,'attention_mask_2':out}


def native_network(m,repair=False):
    s={'torch':torch,'nn':nn,'np':np,'F':F,'math':math,'autocast':autocast,'partial':partial,
       'Optional':Optional,'Union':Union,'Type':type,'Tuple':tuple,'collections':collections,'repeat':repeat}
    definitions('timm/layers/helpers.py',['_ntuple'],s,source='timm_stablemil');s['to_2tuple']=s['_ntuple'](2)
    definitions('timm/layers/drop.py',['drop_path','DropPath'],s,source='timm_stablemil')
    definitions('timm/layers/mlp.py',['Mlp'],s,source='timm_stablemil')
    definitions('timm/layers/layer_scale.py',['LayerScale'],s,source='timm_stablemil')
    definitions('models/d2d_rope.py',['compute_axial_cis','apply_rotary_emb','rotary_','project_','random_project'],s)
    patches=[('torch.zeros([B, self.num_heads , pile_k.shape[2],pile_k.shape[3] - self.k_neighbors - 1])','pile_k.new_zeros([B,self.num_heads,pile_k.shape[2],pile_k.shape[3] - self.k_neighbors - 1])'),
             ('torch.zeros(pile_k.shape[:-1])','pile_k.new_zeros(pile_k.shape[:-1])')]
    if repair:patches.append(('pile_k.shape[3] - self.k_neighbors - 1','pile_k.shape[3] - k_neighbors - 1'))
    definitions('models/attn_module.py',['flex_doucment_Attention','pile_attention'],s,patches=patches)
    definitions('models/model.py',['initialize_weights','attn_block','sequentialV2','stableMIL'],s)
    net=s['stableMIL'](dim=m.dim,hidden_dim=m.hidden,depth=m.depth,num_heads=m.heads,
        aggregate_num=m.aggregate_num,k_neighbors=m.k_neighbors,ratio=m.ratio,learnable_mapping=m.learnable_mapping,
        n_classes=m.bins if m.survival else len(m.bundle.classes),task='survival' if m.survival else 'subtype',
        drop_rate=m.dropout,drop_path_rate=m.drop_path)
    net.load_state_dict(m.network.state_dict())
    return net,s


def native_forward(net,x,coords,geometry):
    args=[torch.from_numpy(np.asarray(geometry[k]).copy()) for k in ['fuse_coords','fuse_labels','fuse_sorted_idx','region_indices','region_sorted_index','attention_mask_1','attention_mask_2']]
    args[0]=args[0].unsqueeze(0)
    return net(x.unsqueeze(0),*args)


@pytest.mark.parametrize('task',['classification','survival'])
@pytest.mark.parametrize('training',[False,True])
def test_native_1024_512_complete_outputs_and_all_gradients(tmp_path,task,training):
    data=create_stable_demo(tmp_path/'data',patients=30,dimension=1024,classes=3,survival=task=='survival')
    b=Bundle(data);m=StableMIL(b,'stablemil_'+task+'_source');ref,scope=native_network(m)
    m.train(training);ref.train(training);m.prepare(b,0,'cpu')
    row=b.records('train')[0];x,c=load_spatial_inputs(b,row);x.requires_grad_(True);other=x.detach().clone().requires_grad_(True)
    geometry=native_geometry(c);state=rng_state();a=m(x,c)
    loss=m.objective(a,row['time'],row['event']) if m.survival else F.cross_entropy(a['logits'],torch.tensor([row['target']]))
    loss.backward();after=rng_state();restore_rng(state);out=native_forward(ref,other,c,geometry)
    if m.survival:
        definitions('survival/utils/pro_utils.py',['nll_loss','NLLSurvLoss'],scope)
        # Independently derive TRAIN-only quantiles, not labels from the adapter.
        times=b.rows.query("subset == 'train' and event == 1").time.to_numpy()
        edges=pd.qcut(times,q=4,retbins=True,labels=False)[1][1:-1]
        y=torch.tensor([np.searchsorted(edges,row['time'],side='right')])
        expected=scope['NLLSurvLoss'](alpha=.15)(out[0],out[1],y,torch.tensor([1-row['event']]))
        equal(out[0],a['hazards']);equal(out[1],a['survival'])
    else:expected=F.cross_entropy(out[0],torch.tensor([row['target']]));equal(out[0],a['logits'])
    expected.backward();equal(loss,expected);equal(x.grad,other.grad);equal(after,rng_state())
    for p,q in zip(m.network.parameters(),ref.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:equal(p.grad,q.grad)
    assert m.network.maping.out_features==512 and len(m.network.blocks)==2
    assert m.network.ag_attn.query.grad.abs().sum()>0


@pytest.mark.parametrize('shape',[(10,10),(16,8),(9,13)])
def test_full_author_physical_preprocessing_arrays_and_identity(shape):
    coords=torch.tensor([(x*512,y*512) for x in range(shape[0]) for y in range(shape[1])],dtype=torch.float64)
    arrays,meta=build_geometry(coords,512);reference=native_geometry(coords)
    for key,value in reference.items():equal(arrays[key],value)
    assert meta['raw_instances']==len(coords) and meta['retained_instances']<=len(coords)
    assert len(meta['geometry_sha256'])==64
    expected=sorted(i for label in np.unique(arrays['fuse_labels']) for i in arrays['fuse_sorted_idx'][arrays['fuse_labels'][arrays['fuse_sorted_idx']]==label][:4])
    assert sorted(arrays['retained_rows'].tolist())==expected


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_three_complete_epochs_match_author_loop_and_all_adam_rng_state(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile);ref,scope=native_network(m,repair=m.repaired);m.prepare(b,0,'cpu')
    rows=b.records('train');geometries={r['slide_id']:native_geometry(load_spatial_inputs(b,r)[1]) for r in rows}
    opt=m.make_optimizer(.0001,0);ropt=torch.optim.Adam(ref.parameters(),lr=.0001,weight_decay=0.)
    if m.survival:
        scope.update(concordance_index_censored=lambda event,time,risk,**kwargs:(harrell_c_index(time,event.astype(int),risk),))
        definitions('survival/utils/pro_utils.py',['nll_loss','NLLSurvLoss','collate_MIL_survival'],scope)
        patches=[]
        if m.repaired:patches=[('loss = loss / gc + loss_reg','loss = loss / min(gc,length-(batch_idx//gc)*gc) + loss_reg')]
        definitions('survival/utils/pro_core_utils.py',['train_loop_survival'],scope,patches=patches)
        times=b.rows.query("subset=='train' and event==1").time.to_numpy();edges=pd.qcut(times,q=4,retbins=True,labels=False)[1][1:-1]
        data=[]
        for r in rows:
            x,c=load_spatial_inputs(b,r);data.append((x,c.numpy(),int(np.searchsorted(edges,r['time'],side='right')),r['time'],1-r['event']))
        class BoundSurvival(nn.Module):
            def __init__(self):super().__init__();self.network=ref
            def forward(self,x,coords):return native_forward(self.network,x[0],coords[0],native_geometry(coords[0]))
        original=BoundSurvival();fn=scope['train_loop_survival'];collate=scope['collate_MIL_survival']
    else:
        scope['calculate_error']=lambda predicted,label:1.-float((predicted.flatten()==label.flatten()).float().mean())
        definitions('classification/utils/core_utils.py',['Accuracy_Logger'],scope)
        definitions('classification/utils/utils.py',['collate_MIL'],scope)
        patches=[('region_sorted_index','region_sort_index')]
        if m.repaired:
            patches += [('    model.train()','    model.train()\n    optimizer.zero_grad()'),('loss= loss/gacc_step','loss= loss/min(gacc_step,len(loader)-(batch_idx//gacc_step)*gacc_step)'),
                        ('if (batch_idx + 1)%gacc_step == 0:','if (batch_idx + 1)%gacc_step == 0 or batch_idx+1==len(loader):')]
        definitions('classification/utils/core_utils.py',['train_loop'],scope,patches=patches)
        data=[]
        for r in rows:
            x,c=load_spatial_inputs(b,r);g=geometries[r['slide_id']]
            data.append((x,torch.tensor([r['target']]),*[torch.from_numpy(np.asarray(g[k]).copy()) for k in ['fuse_coords','fuse_labels','fuse_sorted_idx','region_indices','region_sorted_index','attention_mask_1','attention_mask_2']]))
        original=ref;fn=scope['train_loop'];collate=scope['collate_MIL']
    for epoch in range(3):
        m.train();m.start_epoch(epoch);state=rng_state();losses=stable_epoch(m,b,units(b,'train'),opt,'cpu',{'max_instances':None})
        after=rng_state();restore_rng(state)
        loader=DataLoader(data,batch_size=1,sampler=RandomSampler(data),num_workers=0,collate_fn=collate)
        if m.survival:fn(epoch,original,loader,ropt,4,loss_fn=scope['NLLSurvLoss'](alpha=.15),gc=4)
        else:fn(epoch,original,loader,ropt,3,loss_fn=nn.CrossEntropyLoss(),gacc_step=4)
        equal(m.network.state_dict(),ref.state_dict());equal(opt.state_dict(),ropt.state_dict());equal(after,rng_state())
        for p,q in zip(m.network.parameters(),ref.parameters()):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:equal(p.grad,q.grad)
        assert all(math.isfinite(v['loss']) for v in losses)
    (tmp_path/'stablemil_author_training.json').write_text(json.dumps({'profile':profile,'epochs':3,'optimizer_updates':m.optimizer_steps,
        'pending_samples':m.pending_count,'max_parameter_abs':0.,'state':'passed',
        'scope':'Complete original classification/survival train loop, original Adam and NLL/CE. Named variable/preprocessing wiring and repaired tail branches explicit; C-index logging shim only; all weights/gradients/Adam/RNG bitwise equal.'}))


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_exact_resume_pending_gradients_last_best_and_repeated_prediction(tmp_path,profile):
    _m,b,opts=fixture(tmp_path,profile);cfg={'epochs':3,'threads':1,'model':opts}
    run=train(b.root,profile,tmp_path/'full',config=cfg)
    other=train(b.root,profile,tmp_path/'resumed',config=cfg,stop_after=1)
    train(b.root,profile,other,config=cfg,resume=True)
    a=torch.load(run/'last.pt',weights_only=False);z=torch.load(other/'last.pt',weights_only=False)
    for key in ['model','optimizers','rng','best','bad','history']:equal(a[key],z[key])
    first=evaluate(run);csv=(run/'test_predictions.csv').read_bytes();assert evaluate(run)==first and csv==(run/'test_predictions.csv').read_bytes()
    assert not json.loads((run/'status.json').read_text())['test_evaluated']
    p=json.loads((run/'method_protocol.json').read_text());assert p['optimizer_steps']>0
    if profile=='stablemil_classification_source':assert a['model']['_extra_state']['pending_count']==3 and a['model']['_extra_state']['gradients']
    else:assert a['model']['_extra_state']['pending_count']==0
    frame=pd.read_csv(run/'test_predictions.csv');assert (frame.stable_retained_instances<=frame.stable_raw_instances).all()
    assert all(len(v)==64 for v in frame.stable_geometry_sha256)


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_forward_has_no_label_access_and_eval_preserves_rng(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile);m.prepare(b,0,'cpu');m.eval();row=b.records('test')[0];x,c=load_spatial_inputs(b,row)
    before=rng_state();first=m(x,c,label=torch.tensor([0]));second=m(x,c,label=torch.tensor([2]));equal(first,second);equal(before,rng_state())


@pytest.mark.parametrize('kind',['one','line','four'])
def test_source_degenerate_inputs_fail_and_named_repair_trains(tmp_path,kind):
    m,b,opts=fixture(tmp_path)
    coords={'one':torch.tensor([[0.,0.]]),'line':torch.tensor([[i*512.,0.] for i in range(20)]),'four':torch.tensor([[0.,0.],[512.,0.],[0.,512.],[512.,512.]])}[kind]
    x=torch.randn(len(coords),8);m.train();out=m(x,coords);F.cross_entropy(out['logits'],torch.tensor([1])).backward()
    assert torch.isfinite(out['logits']).all() and m.network.maping.weight.grad.abs().sum()>0
    original=StableMIL(b,'stablemil_classification_source',**opts)
    with pytest.raises(ValueError,match='source'):original(x,coords)


def test_cached_geometry_mutation_and_coordinate_changes_are_detected():
    coords=torch.tensor([(i*512.,j*512.) for i in range(10) for j in range(10)])
    cache=GeometryCache();a,m=cache.get(coords,reference_size=512)
    a['attention_mask_1'][0,0]=1
    with pytest.raises(ValueError,match='cache changed'):cache.get(coords,reference_size=512)
    other,meta=GeometryCache().get(coords+512,reference_size=512)
    assert meta['geometry_sha256']!=m['geometry_sha256'] and np.isfinite(other['fuse_coords']).all()


@pytest.mark.parametrize('alpha',[0.,.15,.6])
def test_author_survival_loss_extremes_and_all_gradients(alpha):
    s={'torch':torch};definitions('survival/utils/pro_utils.py',['nll_loss','NLLSurvLoss'],s)
    logits=torch.tensor([[-60.,1.,-2.,3.],[60.,-2.,.3,-4.]],requires_grad=True)
    h=logits.sigmoid();surv=(1-h).cumprod(1);y=torch.tensor([2,1]);c=torch.tensor([0.,1.])
    actual=s['NLLSurvLoss'](alpha=alpha)(h,surv,y,c)
    previous=torch.cat([torch.ones(2,1),surv],1)[torch.arange(2),y]
    next_s=torch.cat([torch.ones(2,1),surv],1)[torch.arange(2),y+1]
    event=-(previous.clamp(min=1e-7).log()+h[torch.arange(2),y].clamp(min=1e-7).log())
    expected=((1-c)*event-(1-alpha)*c*next_s.clamp(min=1e-7).log()).mean()
    equal(actual,expected);ga=torch.autograd.grad(actual,logits,retain_graph=True)[0];gb=torch.autograd.grad(expected,logits)[0];equal(ga,gb)


def test_train_only_time_bins_and_checkpoint_rejects_input_changes(tmp_path):
    m,b,opts=fixture(tmp_path,'stablemil_survival_repaired');m.prepare(b,0,'cpu');expected=m.bin_edges.clone()
    b.rows.loc[b.rows.subset!='train','time']=1e9
    m.prepare(b,0,'cpu');equal(m.bin_edges,expected)
    # Bundle reconstructed from immutable source tables for a genuine engine run.
    run=train(b.root,'StableMIL',tmp_path/'run',config={'epochs':1,'threads':1,'model':opts})
    row=b.records('test')[0]
    with h5py.File(b.root/row['path'],'a') as h:h['coords'][0,0]+=1
    with pytest.raises(ValueError,match='inputs differ'):evaluate(run)


@pytest.mark.parametrize('opts',[{'hidden':30},{'heads':0},{'depth':0},{'ratio':0},{'accumulation_steps':0},{'dropout':1.},{'drop_path':-1},{'reference_size':0},{'geometry_radius':float('nan')}])
def test_constructor_contract_guards(tmp_path,opts):
    _,b,base=fixture(tmp_path)
    with pytest.raises(ValueError):StableMIL(b,**{**base,**opts})


def test_budget_refusal_before_any_model_or_dense_geometry_allocation(tmp_path,monkeypatch):
    _,b,_=fixture(tmp_path)
    import milplus.stablemil as module
    monkeypatch.setattr(module,'author_model',lambda *a:(_ for _ in ()).throw(AssertionError('constructed before budget check')))
    with pytest.raises(ValueError,match='before allocation'):StableMIL(b,hidden=1024**2)
    with pytest.raises(ValueError,match='before allocation'):build_geometry(torch.tensor([[0.,0.],[1e12,1e12]],dtype=torch.float64),1,max_bytes=1024**2)


def test_training_does_not_silently_change_sampling_or_clipping(tmp_path):
    m,_,_=fixture(tmp_path)
    for change in [{'max_instances':12},{'grad_clip':1.},{'selection_metric':'macro_auc'}]:
        cfg={'max_instances':None,'grad_clip':None,'selection_metric':'stablemil_loss',**change}
        with pytest.raises(ValueError):m.configure_training(cfg)
    assert m.selection_improved(-1.,-1.) and m.stop_after_epoch==51


def test_persisted_geometry_is_input_bound_reusable_and_preserves_rng(tmp_path,monkeypatch):
    coords=torch.tensor([(i*512.,j*512.) for i in range(10) for j in range(10)],dtype=torch.float64)
    cache=GeometryCache(max_bytes=1);cache.directory=tmp_path/'geometry'
    state=rng_state();a,meta=cache.get(coords,reference_size=512);equal(state,rng_state())
    import milplus.stable_geometry as module
    monkeypatch.setattr(module,'build_geometry',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('recomputed frozen geometry')))
    b,n=cache.get(coords,reference_size=512);equal(a,b);equal(meta,n)
    path=next(cache.directory.glob('*/arrays.h5'))
    with h5py.File(path,'a') as h:h['fuse_coords'][0,0]+=1
    with pytest.raises(ValueError,match='identity/hash'):cache.get(coords,reference_size=512)


def test_shared_geometry_cache_atomic_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    coords=torch.tensor([(i*512.,j*512.) for i in range(10) for j in range(10)])
    def prepare(_):
        c=GeometryCache();c.directory=tmp_path/'shared';return c.get(coords,reference_size=512)[1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first,second=list(pool.map(prepare,range(2)))
    equal(first,second)
    assert len(list((tmp_path/'shared').glob('*/record.json')))==1
    assert not list((tmp_path/'shared').glob('.*'))


def test_native_small_mask_failure_independently_confirmed(tmp_path):
    m,_,_=fixture(tmp_path);_ref,scope=native_network(m)
    block=scope['pile_attention'](dim=32,num_heads=4,k_neighbors=8)
    coords=torch.tensor([[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]])
    x=torch.randn(1,4,32);seman=torch.randn(1,2,32);sc=coords[:,:2].unsqueeze(1).repeat(1,4,1,1)
    mask=torch.arange(4).repeat(4,1)
    with pytest.raises(RuntimeError,match='negative dimension'):block(x,seman,coords,sc,(mask,mask))


def test_feature_fusion_preserves_native_truncation_and_documents_dropped_rows():
    coords=torch.tensor([(i*256.,j*256.) for i in range(10) for j in range(10)])
    a,m=build_geometry(coords,512,repaired=True)
    assert len(a['retained_rows'])<len(coords) and m['retained_instances']==len(a['retained_rows'])
    assert m['raw_instances']==100 and m['repairs']


def test_representative_slide_contract_rejects_implicit_survival_drop(tmp_path):
    _m,b,opts=fixture(tmp_path,'stablemil_survival_repaired')
    b.rows=pd.concat([b.rows,b.rows.iloc[[0]]])
    with pytest.raises(ValueError,match='representative-slide'):StableMIL(b,'stablemil_survival_repaired',**opts)


def test_true_gradient_carry_and_fixed_tail_are_distinct(tmp_path):
    for mode in ['source','repaired']:
        sub=tmp_path/mode;sub.mkdir();m,b,_=fixture(sub,'stablemil_classification_'+mode)
        opt=m.make_optimizer(.0001,0);m.train();m.start_epoch(0)
        stable_epoch(m,b,units(b,'train'),opt,'cpu',{'max_instances':None})
        if mode=='source':assert m.optimizer_steps==6 and m.pending_count==3 and m.get_extra_state()['gradients']
        else:assert m.optimizer_steps==7 and m.pending_count==0 and not m.get_extra_state()['gradients']

"""GTP: complete author networks, full epochs, min-cut mathematics and GraphCAM."""
import ast
import json
import math
import os
import subprocess
import sys
import types
import warnings
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import dense_mincut_pool

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, predict, restore_rng, rng_state, seed_everything, train, units
from milplus.gtp import GTP, gtp_epoch, lattice_edges, repaired_pool
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def single_thread():torch.set_num_threads(1)


def fixture(tmp_path, profile='gtp_repaired', native_size=False, **options):
    root=create_demo(tmp_path/'data',patients=30,classes=3,dimension=8);b=Bundle(root)
    opts={} if native_size else {'hidden':16,'clusters':8,'depth':2,'heads':4}
    opts.update({'batch_size':4,**options})
    return GTP(b,profile,**opts),b,opts


def equal(a,b,tol=0):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=tol)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k],tol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y,tol)
    else:assert a==b


def native(model):
    """Independently load original source modules; only CPU allocation changes."""
    root,sha=verified_source('gtp');parent='_gtp_independent_'+sha
    p=types.ModuleType(parent);p.__path__=[str(root/'models')];sys.modules[parent]=p
    for filename in ['layers','ViT']:
        code=subprocess.check_output(['git','show',sha+':models/'+filename+'.py'],cwd=root,text=True)
        m=types.ModuleType(parent+'.'+filename);m.__file__=str(root/'models'/(filename+'.py'));m.__package__=parent
        m.__dict__['warnings']=warnings;sys.modules[m.__name__]=m
        exec(compile(code,m.__file__,'exec'),m.__dict__)  # noqa: S102 -- independent pinned Git blobs
    def definitions(path,names,scope,replacements=()):
        code=subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)
        for old,new in replacements:
            assert old in code;code=code.replace(old,new)
        nodes=[n for n in ast.parse(code).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)  # noqa: S102
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'os':os,'Linear':nn.Linear,'VisionTransformer':sys.modules[parent+'.ViT'].VisionTransformer,
           'dense_mincut_pool':dense_mincut_pool}
    definitions('models/gcn.py',['GCNBlock'],scope,[('.cuda()','')])
    definitions('models/GraphTransformer.py',['Classifier'],scope,[('one_hot.cuda()','one_hot.to(X.device)')])
    net=scope['Classifier'](len(model.bundle.classes),model.dim)
    net.load_state_dict(model.network.state_dict())
    captured={};net.transformer.head.register_forward_hook(lambda module,args,out:captured.update(logits=out))
    definitions('utils/metrics.py',['ConfusionMatrix'],scope)
    definitions('helper.py',['collate','preparefeatureLabel','Trainer'],scope,[('.cuda()','')])
    return net,captured,scope,root,sha


@pytest.mark.parametrize('profile',['gtp_source','gtp_repaired'])
@pytest.mark.parametrize('batch_size',[1,3])
def test_full_original_native_network_outputs_losses_gradients(tmp_path,profile,batch_size):
    seed_everything(81);m,b,_=fixture(tmp_path,profile,native_size=True)
    author,captured,_,_,_=native(m)
    rows=b.records('train')[:batch_size];x,adj,mask=m.inputs(rows,'cpu');y=torch.tensor([r['target'] for r in rows])
    result=m.batch_forward(x,y,adj,mask);result['total_loss'].backward()
    _,_,loss=author(x,y,adj,mask);loss.backward()
    equal(result['logits'],captured['logits']);equal(result['total_loss'],loss)
    for (name,p),(other,q) in zip(m.network.named_parameters(),author.named_parameters()):
        assert name==other and p.grad is not None
        equal(p.grad,q.grad)
    assert m.network.pool1.weight.grad.abs().sum()>0


@pytest.mark.parametrize('profile',['gtp_source','gtp_repaired'])
def test_execute_entire_original_training_inner_loop(tmp_path,profile):
    seed_everything(71);m,b,_=fixture(tmp_path,profile,native_size=True)
    author,_,scope,root,sha=native(m)
    cfg={'epochs':3,'max_instances':None,'grad_clip':None,'selection_metric':'gtp_bag_accuracy'}
    m.configure_training(cfg);opt=m.make_optimizer(.001,.0005)
    refopt=torch.optim.Adam(author.parameters(),lr=.001,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(refopt,milestones=[20,100],gamma=.1)
    text=subprocess.check_output(['git','show',sha+':main.py'],cwd=root,text=True)
    epoch_node=next(n for n in ast.parse(text).body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
    train_node=next(n for n in epoch_node.body if isinstance(n,ast.If) and ast.unparse(n.test)=='train')
    batch_node=next(n for n in train_node.body if isinstance(n,ast.For))
    code=compile(ast.Module(body=[batch_node],type_ignores=[]),'author/main.py:full_train_batch_loop','exec')
    scope.update(model=author,optimizer=refopt,scheduler=scheduler,trainer=scope['Trainer'](3),
                 args=SimpleNamespace(n_features=8,log_interval_local=10000),total_train_num=len(b.records('train')))
    def collate(rows):
        bags=[b.load(row) for row in rows]
        return {'image':[x for x,_ in bags],'label':[r['target'] for r in rows],
                'adj_s':[m.inputs([r],'cpu')[1][0] for r in rows]}
    loader=torch.utils.data.DataLoader(b.records('train'),batch_size=4,shuffle=True,drop_last=True,collate_fn=collate)
    for epoch in [0,20,100]:
        m.train();author.train();m.start_epoch(epoch);state=rng_state()
        losses=gtp_epoch(m,b,units(b,'train'),opt,torch.device('cpu'),cfg);after=rng_state();restore_rng(state)
        scope.update(epoch=epoch,dataloader_train=loader,train_loss=0.,total=0.)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');exec(code,scope)  # noqa: S102 -- complete original inner training loop
        equal(m.network.state_dict(),author.state_dict());equal(opt.state_dict(),refopt.state_dict())
        equal(m.scheduler.state_dict(),scheduler.state_dict());equal(torch.get_rng_state(),after['torch'])
        assert np.mean([r['loss'] for r in losses])==pytest.approx(float(scope['train_loss'])/len(loader),abs=1e-6)
    (tmp_path/'gtp_author_training.json').write_text(json.dumps({'state':'passed','profile':profile,'epochs':[0,20,100],
        'optimizer_updates':m.optimizer_steps,'max_parameter_abs':0.,'native_hidden':64,'native_clusters':100,
        'scope':'Original main.py complete inner loop plus helper Trainer, CPU allocation only, all model/Adam/BN/scheduler/RNG states bitwise equal'}))


@pytest.mark.parametrize('lengths',[(3,7),(1,1),(7,7)])
def test_mincut_outputs_and_all_gradients_against_independent_math(lengths):
    torch.manual_seed(9);n=max(lengths);batch=len(lengths);k=3
    x=torch.randn(batch,n,4,dtype=torch.float64,requires_grad=True)
    s=torch.randn(batch,n,k,dtype=torch.float64,requires_grad=True)
    adj=torch.zeros(batch,n,n,dtype=torch.float64);mask=torch.zeros(batch,n,dtype=torch.float64)
    for i,length in enumerate(lengths):
        mask[i,:length]=1
        for j in range(length-1):adj[i,j,j+1]=adj[i,j+1,j]=1
    output=repaired_pool(x,adj,s,mask)
    a=s.softmax(-1)*mask.unsqueeze(-1);features=x*mask.unsqueeze(-1)
    pooled=a.transpose(1,2)@features;raw=a.transpose(1,2)@adj@a
    regularizers=[];orth=[]
    for i in range(batch):
        denominator=torch.trace(a[i].T@torch.diag(adj[i].sum(1))@a[i])
        regularizers.append(-torch.trace(raw[i])/denominator if denominator>0 else x.new_zeros(()))
        gram=a[i].T@a[i];orth.append(torch.linalg.vector_norm(gram/torch.linalg.vector_norm(gram)-torch.eye(k,dtype=x.dtype)/math.sqrt(k)))
    coarse=raw-torch.diag_embed(raw.diagonal(dim1=1,dim2=2))
    degree=coarse.sum(2).sqrt().unsqueeze(1)+1e-15
    expected=(pooled,coarse/degree/degree.transpose(1,2),torch.stack(regularizers).mean(),torch.stack(orth).mean())
    for actual,ref in zip(output,expected):torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
    first=torch.autograd.grad(output[0].square().sum()+output[2]+output[3],(x,s),retain_graph=True)
    second=torch.autograd.grad(expected[0].square().sum()+expected[2]+expected[3],(x,s))
    for p,q in zip(first,second):torch.testing.assert_close(p,q,rtol=1e-10,atol=1e-10)


def test_singleton_noedge_source_rejects_repair_trains_without_nan(tmp_path):
    m,b,_=fixture(tmp_path);x=torch.randn(1,8);coords=torch.tensor([[0.,0.]])
    before=m.network.conv1.bn_layer.num_batches_tracked.clone()
    result=m(x,coords,torch.tensor([1]));result['total_loss'].backward()
    assert torch.isfinite(result['total_loss']) and result['mincut']==0
    equal(m.network.conv1.bn_layer.num_batches_tracked,before)
    assert m.network.conv1.bn_layer.training
    original=GTP(b,'gtp_source',hidden=16,clusters=8,heads=4,depth=2)
    with pytest.raises(ValueError,match='requires graph edges'):original(x,coords)


@pytest.mark.parametrize('profile',['gtp_source','gtp_repaired'])
def test_exact_resume_repeated_predictions_and_graphcam_exports(tmp_path,profile):
    _,b,opts=fixture(tmp_path,profile)
    cfg={'epochs':3,'threads':1,'model':opts}
    first=train(b.root,profile,tmp_path/'direct',seed=12,config=cfg)
    interrupted=train(b.root,profile,tmp_path/'resume',seed=12,config=cfg,stop_after=1)
    train(b.root,profile,interrupted,seed=12,config=cfg,resume=True)
    a=torch.load(first/'last.pt',weights_only=False);c=torch.load(interrupted/'last.pt',weights_only=False)
    for key in ['model','optimizers','history','rng']:equal(a[key],c[key])
    one=evaluate(first);data=(first/'test_predictions.csv').read_bytes()
    assert one==evaluate(first) and data==(first/'test_predictions.csv').read_bytes()
    manifest=json.loads((first/'test_instances/manifest.json').read_text())
    for item in manifest['records']:
        with h5py.File(item['path']) as h:
            assert h['graphcam_raw'].shape==(item['instances'],3)
            assert np.isfinite(h['graphcam_display'][:]).all()


def test_original_graphcam_lrp_attention_and_soft_mapping(tmp_path,monkeypatch):
    m,b,_=fixture(tmp_path,native_size=True);m.eval()
    author,_,_,_,_=native(m);author.eval()
    row=b.records('test')[0];x,adj,mask=m.inputs([row],'cpu')
    work=tmp_path/'author';work.mkdir();monkeypatch.chdir(work)
    before=rng_state();maps=m.graphcam(row,'cpu')
    author(x,torch.zeros(1,dtype=torch.long),adj,mask,graphcam_flag=True)
    assignments=torch.load(work/'graphcam/s_matrix_ori.pt',weights_only=False).softmax(1)
    probability=torch.load(work/'graphcam/prob.pt',weights_only=False)[0].detach().clamp(.4,1)
    for c in range(3):
        cam=torch.load(work/f'graphcam/cam_{c}.pt',weights_only=False)
        raw=(assignments@cam.T).squeeze(1)
        display=((raw-raw.min())/(raw.max()-raw.min())*probability[c]).clamp(0,1)
        torch.testing.assert_close(maps['graphcam_raw'][:,c],raw,rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(maps['graphcam_display'][:,c],display,rtol=1e-5,atol=1e-6)
    equal(torch.get_rng_state(),before['torch'])


@pytest.mark.parametrize('profile',['gtp_source','gtp_repaired'])
def test_query_truth_never_changes_logits_or_graphcam(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile);m.eval()
    one=predict(m,b,profile,'test','cpu',{'max_instances':None})
    row=b.records('test')[0];maps=m.graphcam(row,'cpu')
    b.rows['target']=(b.rows.target+1)%3
    two=predict(m,b,profile,'test','cpu',{'max_instances':None})
    pd.testing.assert_frame_equal(one[['p_0','p_1','p_2']],two[['p_0','p_1','p_2']],check_exact=True)
    equal(maps,m.graphcam(row,'cpu'))


@pytest.mark.parametrize('stride',[1.,256.])
def test_exact_author_chebyshev_topology_and_holes(stride):
    coords=torch.tensor([[0,0],[1,1],[2,1],[0,3],[4,4]],dtype=torch.float64)*stride
    edges=lattice_edges(coords,stride)
    expected={(i,j) for i in range(len(coords)) for j in range(len(coords)) if i!=j and bool((abs(coords[i]-coords[j])<=stride).all())}
    assert set(map(tuple,edges.T.tolist()))==expected


@pytest.mark.parametrize('option,value',[('hidden',0),('hidden',15),('clusters',0),('depth',True),('heads',0),('batch_size',0),('max_dense_bytes',0),('drop_last',1)])
def test_invalid_constructor_options(tmp_path,option,value):
    with pytest.raises(ValueError):fixture(tmp_path,**{option:value})


@pytest.mark.parametrize('kind',['duplicate','fractional','nan','none'])
def test_reject_invalid_physical_coordinates(kind):
    points={'duplicate':torch.zeros(2,2),'fractional':torch.tensor([[0.,0.],[.1,0.]]),
            'nan':torch.tensor([[0.,0.],[float('nan'),0.]]),'none':None}[kind]
    with pytest.raises(ValueError):lattice_edges(points,1.)


def test_dense_budget_fails_before_allocation(tmp_path,monkeypatch):
    m,b,_=fixture(tmp_path)
    m.max_dense_bytes=1
    monkeypatch.setattr('milplus.gtp.lattice_edges',lambda *a:(_ for _ in ()).throw(AssertionError('graph allocated')))
    with pytest.raises(ValueError,match='before allocation'):m.inputs(b.records('train')[:2],'cpu')


@pytest.mark.parametrize('epoch',[0,19,20,99,100,101])
def test_source_scheduler_epoch_boundaries(tmp_path,epoch):
    m,_,_=fixture(tmp_path);m.make_optimizer(.001,.0005)
    with warnings.catch_warnings():warnings.simplefilter('ignore');m.scheduler.step(epoch)
    assert m.optimizer.param_groups[0]['lr']==pytest.approx(.001*(.1**int(epoch>=20))*(.1**int(epoch>=100)),rel=1e-12)


@pytest.mark.parametrize('case',['nan','negative_edge','mask_shape','empty_mask','nonprefix','padded_edge','wrong_target'])
def test_batch_validation_rejects_invalid_alignment(tmp_path,case):
    m,b,_=fixture(tmp_path);rows=b.records('train')[:2];x,adj,mask=m.inputs(rows,'cpu');y=torch.tensor([0,1])
    if case=='nan':x[0,0,0]=float('nan')
    elif case=='negative_edge':adj[0,0,1]=-1
    elif case=='mask_shape':mask=mask[:,:1]
    elif case=='empty_mask':mask[0]=0
    elif case=='nonprefix':mask[0,0]=0
    elif case=='padded_edge':mask[0,-1]=0;adj[0,0,-1]=1
    elif case=='wrong_target':y[0]=3
    with pytest.raises(ValueError):m.batch_forward(x,y,adj,mask)


@pytest.mark.parametrize('option,value',[('hidden',2**30),('clusters',2**30)])
def test_construction_budget_before_large_parameter_allocation(tmp_path,option,value):
    with pytest.raises(ValueError,match='before allocation'):fixture(tmp_path,**{option:value})


@pytest.mark.parametrize('task',['survival','patient','ordinal'])
def test_reject_incompatible_task(tmp_path,task):
    root=create_demo(tmp_path/'data',patients=20,dimension=8,survival=task=='survival')
    if task!='survival':
        import yaml
        p=root/'task.yaml';config=yaml.safe_load(p.read_text())
        config['training_unit' if task=='patient' else 'task_type']='patient' if task=='patient' else 'ordinal'
        p.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):GTP(Bundle(root))


def test_graphcam_preserves_existing_parameter_gradients(tmp_path):
    m,b,_=fixture(tmp_path);m.eval()
    for p in m.parameters():p.grad=torch.ones_like(p)*.2
    values=[p.grad.clone() for p in m.parameters()]
    m.graphcam(b.records('test')[0],'cpu')
    for p,old in zip(m.parameters(),values):equal(p.grad,old)


def test_nonfinite_gradient_is_rejected_without_optimizer_step(tmp_path):
    m,b,_=fixture(tmp_path);m.configure_training({'epochs':3,'max_instances':None,'grad_clip':None,'selection_metric':'gtp_bag_accuracy'})
    opt=m.make_optimizer(.001,.0005);m.start_epoch(0)
    m.network.pool1.weight.register_hook(lambda value:torch.full_like(value,float('nan')))
    with pytest.raises(ValueError,match='gradient'):gtp_epoch(m,b,units(b,'train'),opt,torch.device('cpu'),{})
    assert m.optimizer_steps==0 and not opt.state


def test_repaired_constant_singleton_graphcam_is_zero(tmp_path):
    m,b,_=fixture(tmp_path);row=b.records('test')[0]
    with h5py.File(b.root/row['path'],'r+') as h:
        for key in ['features','coords']:
            value=h[key][:1];del h[key];h[key]=value
    m.eval();maps=m.graphcam(row,'cpu')
    assert maps['graphcam_raw'].shape==(1,3) and (maps['graphcam_display']==0).all()


@pytest.mark.parametrize('profile',['gtp_source','gtp_repaired'])
def test_constant_source_visualization_marked_undefined_without_losing_predictions(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile);m.eval();row=b.records('test')[0]
    with h5py.File(b.root/row['path'],'r+') as h:h['features'][:]=0
    result=m.forward_unit(b,[row],'cpu',{})
    assert torch.isfinite(result['logits']).all()
    maps=m.graphcam(row,'cpu')
    expected_defined=maps['graphcam_raw'].amax(0)>maps['graphcam_raw'].amin(0)
    assert torch.equal(maps['graphcam_display_defined'],expected_defined)
    assert (maps['graphcam_display'][:,~expected_defined]==0).all()
    assert torch.isfinite(maps['graphcam_display']).all()
    assert (maps['graphcam_raw'].amax(0)-maps['graphcam_raw'].amin(0)).max()<1e-6

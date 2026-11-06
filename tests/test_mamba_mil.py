"""MambaMIL: independent pinned reference kernels, original trainers, and resume evidence."""
import ast
import copy
import json
import math
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, load_unit, restore_rng, rng_state, train, units
from milplus.mamba_mil import (
    PROFILES,
    MambaMIL,
    backend_functions,
    mamba_epoch,
    sampled_groups,
    selective_scan_torch,
)
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def reference(hidden=16):
    """Read independent original ASTs; use the author's vectorized reference, not PULS scan."""
    root,sha=verified_source('mambamil')
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'Optional':Optional,'Tensor':Tensor,
           'rearrange':rearrange,'repeat':repeat,'device':torch.device('cpu'),
           'calculate_error':lambda a,b:1-a.eq(b).float().mean().item(),
           'concordance_index_censored':lambda *a,**k:(.5,)}
    def read(path,names):
        code=pinned_text(root,sha,path)
        if path=='models/MambaMIL.py':code=code.replace('512',str(hidden))
        defs=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in defs}==set(names)
        exec(compile(ast.Module(body=defs,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent pinned author equations and training
    read('mamba/mamba_ssm/ops/selective_scan_interface.py',['selective_scan_ref','mamba_inner_ref'])
    scope['selective_scan_fn']=scope['selective_scan_ref']
    def convolution(x,weight,bias,activation):
        assert activation=='silu'
        # Explicit left padding, in contrast to the adapter's right-cropped convolution.
        return F.silu(F.conv1d(F.pad(x,(weight.shape[-1]-1,0)),weight[:,None],bias,groups=x.shape[1]))
    scope['causal_conv1d_fn']=convolution
    scope['mamba_inner_fn']=scope['mamba_inner_ref']
    def no_out(xz,cw,cb,xw,dw,A,*args,**kwargs):
        channels=xz.shape[1]//2
        result=scope['mamba_inner_ref'](xz,cw,cb,xw,dw,torch.eye(channels,device=xz.device,dtype=xz.dtype),None,A,*args,**kwargs)
        return result.transpose(1,2)
    scope['mamba_inner_fn_no_out_proj']=no_out
    for name,path in [('SRMamba','srmamba'),('BiMamba','bimamba'),('Mamba','mamba_simple')]:
        read('mamba/mamba_ssm/modules/'+path+'.py',[name]+(['TransposeTokenReEmbedding'] if name!='Mamba' else []))
    read('models/MambaMIL.py',['initialize_weights','MambaMIL'])
    read('utils/survival_utils.py',['nll_loss','NLLSurvLoss'])
    read('utils/core_utils.py',['Accuracy_Logger','train_loop'])
    read('utils/survival_core_utils.py',['train_loop_survival'])
    return scope


def assert_state(a,b,exact=False):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0 if exact else 3e-4,atol=0 if exact else 2e-6)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_state(a[k],b[k],exact)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state(x,y,exact)
    else:assert a==b


def fixture(path,profile='mambamil_srmamba',survival=False,dimension=16,**kwargs):
    bundle=Bundle(create_demo(path,patients=20,dimension=dimension,survival=survival))
    model=MambaMIL(bundle,profile=profile,**{'hidden':16,'scan_chunk':7,**kwargs})
    model.prepare(bundle,2026,'cpu')
    return model,bundle


@pytest.mark.parametrize('chunk',[1,5,100])
@pytest.mark.parametrize('length',[1,13])
def test_scan_values_and_all_gradients_match_author_vectorized_reference(chunk,length):
    torch.manual_seed(87)
    args=[torch.randn(2,6,length),torch.randn(2,6,length),-torch.rand(6,4),torch.randn(2,4,length),
          torch.randn(2,4,length),torch.randn(6),torch.randn(2,6,length),torch.randn(6)]
    left=[v.clone().requires_grad_() for v in args];right=[v.clone().requires_grad_() for v in args]
    actual=selective_scan_torch(*left,chunk_size=chunk)
    expected=reference()['selective_scan_ref'](*right,delta_softplus=True)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
    weight=torch.randn_like(actual);(actual*weight).sum().backward();(expected*weight).sum().backward()
    for x,y in zip(left,right):torch.testing.assert_close(x.grad,y.grad,rtol=3e-5,atol=3e-6)


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('length',[1,7,10])
def test_full_branch_output_and_gradient_matches_author(tmp_path,profile,length):
    model,_bundle=fixture(tmp_path/'data',profile)
    expected=reference()['MambaMIL'](16,2,0.,'gelu',layer=2,rate=5,type=PROFILES[profile])
    expected.load_state_dict(model.network.state_dict())
    x=torch.randn(length,16,requires_grad=True);y=x.detach().clone().requires_grad_()
    a=model(x)['logits'];b=expected(y)[0]
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-4,atol=3e-6)
    for (name,p),(other,q) in zip(model.network.named_parameters(),expected.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=4e-6)
    if length>1:
        assert not torch.allclose(model(x.flip(0))['logits'],a,atol=1e-7) # preserve order sensitivity


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('survival',[False,True])
def test_three_epochs_match_original_training_loop(tmp_path,profile,survival):
    model,bundle=fixture(tmp_path/'data',profile,survival,accumulation_steps=2 if survival else 1,weighted_sampling=False,scan_chunk=256)
    ref=reference()
    # Isolate the original trainer with identical numerical kernels. Independent kernels
    # are compared separately: sub-ulp softmax-bias gradients can amplify under Adam.
    ref['mamba_inner_fn_no_out_proj'],ref['mamba_inner_fn'],_=backend_functions('torch',256)
    network=ref['MambaMIL'](16,4 if survival else 2,.25 if survival else 0.,'gelu',survival=survival,layer=2,rate=5,type=PROFILES[profile])
    network.load_state_dict(model.network.state_dict())
    opt=model.make_optimizer(2e-4,1e-5);other=torch.optim.Adam(network.parameters(),lr=2e-4,weight_decay=1e-5)
    groups=units(bundle,'train')[:4];loader=[]
    for rows in groups:
        x,_=load_unit(bundle,sorted(rows,key=lambda r:r['slide_id']),'cpu')
        if survival:loader.append((x,torch.zeros(1),model.time_label([rows[0]['time']]),rows[0]['time'],torch.tensor([1-rows[0]['event']],dtype=torch.float32)))
        else:loader.append((x,torch.tensor([rows[0]['target']])))
    for epoch in range(3):
        before=rng_state()
        if survival:ref['train_loop_survival'](epoch,network,loader,other,4,loss_fn=ref['NLLSurvLoss'](alpha=0.),gc=2)
        else:ref['train_loop'](epoch,network,loader,other,2,loss_fn=nn.CrossEntropyLoss())
        restore_rng(before);mamba_epoch(model,bundle,groups,opt,'cpu',{})
        assert_state(model.network.state_dict(),network.state_dict(),exact=True);assert_state(opt.state_dict(),other.state_dict(),exact=True)
    assert model.optimizer_steps==3*(2 if survival else 4)


@pytest.mark.parametrize('profile',PROFILES)
def test_native_width_and_features_all_parameters_receive_gradients(tmp_path,profile):
    model,_=fixture(tmp_path/'data',profile,dimension=1024,hidden=512,scan_chunk=64)
    other=reference(512)['MambaMIL'](1024,2,0.,'gelu',layer=2,rate=5,type=PROFILES[profile]);other.load_state_dict(model.network.state_dict())
    x=torch.randn(6,1024)
    values=[]
    for net in (model.network,other):
        y=net(x)[0];values.append(y)
        F.cross_entropy(y,torch.tensor([1])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    torch.testing.assert_close(*values,rtol=3e-5,atol=3e-6)
    for p,q in zip(model.network.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=8e-6)


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('survival',[False,True])
def test_exact_resume_reload_and_evaluation(tmp_path,profile,survival):
    data=create_demo(tmp_path/'data',patients=20,survival=survival)
    cfg={'epochs':3,'threads':1,'model':{'hidden':16,'scan_chunk':16,'accumulation_steps':5 if survival else 1}}
    train(data,profile,tmp_path/'full',config=cfg)
    train(data,profile,tmp_path/'resume',config=cfg,stop_after=1)
    train(data,profile,tmp_path/'resume',config=cfg,resume=True)
    a,b=[torch.load(tmp_path/p/'last.pt',weights_only=False) for p in ('full','resume')]
    for key in ('model','optimizers','best','history'):assert_state(a[key],b[key],exact=True)
    assert evaluate(tmp_path/'full')==evaluate(tmp_path/'resume')
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'full/test_predictions.csv'),pd.read_csv(tmp_path/'resume/test_predictions.csv'))
    content=(tmp_path/'full/test_predictions.csv').read_bytes();evaluate(tmp_path/'full');assert content==(tmp_path/'full/test_predictions.csv').read_bytes()
    protocol=json.loads((tmp_path/'full'/('survival_protocol.json' if survival else 'method_protocol.json')).read_text())
    assert protocol['optimizer_steps']>0 and protocol['backend']['name']=='torch'
    request=json.loads((tmp_path/'full/request.json').read_text())
    assert request['config']['selection_metric']==('c_index' if survival else 'log_loss')


def test_survival_short_accumulation_window_uses_actual_count(tmp_path):
    model,bundle=fixture(tmp_path/'data',survival=True,dropout=0.,accumulation_steps=4,weighted_sampling=False)
    other=copy.deepcopy(model);opt=model.make_optimizer(2e-4,1e-5)
    groups=units(bundle,'train')[:6];windows=[]
    def verify_window(optimizer,args,kwargs):
        start,size=(0,4) if not windows else (4,2)
        other.network.load_state_dict(model.network.state_dict());other.zero_grad()
        objectives=[other.objective(other.forward_unit(bundle,r,'cpu',{}),r[0]['time'],r[0]['event']) for r in groups[start:start+size]]
        torch.stack(objectives).mean().backward()
        for p,q in zip(model.network.parameters(),other.network.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-6)
        windows.append(size)
    opt.register_step_pre_hook(verify_window)
    mamba_epoch(model,bundle,groups,opt,'cpu',{})
    assert windows==[4,2] and model.optimizer_steps==2



@pytest.mark.parametrize('survival',[False,True])
def test_inverse_frequency_sampler_and_patient_groups(tmp_path,survival):
    model,bundle=fixture(tmp_path/'data',survival=survival,weighted_sampling=True)
    groups=units(bundle,'train')
    labels=[(int(model.time_label([r[0]['time']]).item()),1-r[0]['event']) if survival else r[0]['target'] for r in groups]
    weights=[len(labels)/labels.count(x) for x in labels]
    before=rng_state();actual=sampled_groups(model,groups);restore_rng(before)
    expected=[groups[i] for i in torch.utils.data.WeightedRandomSampler(weights,len(groups))]
    assert actual==expected


def test_training_only_event_quantiles_and_stable_patient_slide_order(tmp_path):
    model,bundle=fixture(tmp_path/'data',survival=True,dropout=0.)
    before=model.bin_edges.clone()
    bundle.rows.loc[bundle.rows.subset!='train','time']=1e10
    model.prepare(bundle,1,'cpu');torch.testing.assert_close(before,model.bin_edges,rtol=0,atol=0)
    rows=units(bundle,'train')[0]
    a=model.forward_unit(bundle,rows,'cpu',{});b=model.forward_unit(bundle,list(reversed(rows)),'cpu',{})
    assert_state(a,b,exact=True)


def test_initialization_matches_actual_author_dt_bias_reset(tmp_path):
    model,_=fixture(tmp_path/'data')
    for name,p in model.named_parameters():
        if 'dt_proj' in name and name.endswith('bias'):assert torch.count_nonzero(p)==0
    assert all(layer[1].use_fast_path for layer in model.network.layers)
    opt=model.make_optimizer(2e-4,1e-5)
    assert all(group['weight_decay']==1e-5 for group in opt.param_groups)


def test_sr_is_reordering_not_bimamba_or_unidirectional(tmp_path):
    models={p:fixture(tmp_path/p,p)[0] for p in PROFILES}
    sr=models['mambamil_srmamba'];bi=models['mambamil_bimamba']
    bi.network.load_state_dict(sr.network.state_dict())
    x=torch.randn(13,16)
    assert not torch.allclose(sr(x)['logits'],bi(x)['logits'])
    initial=sr(x)['logits'];sr.network.rate=1
    assert not torch.allclose(initial,sr(x)['logits'])


def test_checkpoint_backend_identity_is_enforced(tmp_path):
    model,_=fixture(tmp_path/'data');state=copy.deepcopy(model.state_dict())
    state['_extra_state']['backend']['chunk_size']=999
    with pytest.raises(ValueError,match='backend identity'):model.load_state_dict(state)


@pytest.mark.parametrize('options', [{'hidden':0},{'layers':True},{'rate':0},{'scan_chunk':0},{'dropout':1.},{'backend':'automatic'},{'accumulation_steps':2},{'weighted_sampling':1}])
def test_invalid_options_fail(tmp_path,options):
    with pytest.raises(ValueError):fixture(tmp_path/'data',**options)


def test_full_sequence_required_and_cuda_never_silently_falls_back(tmp_path,monkeypatch):
    model,_=fixture(tmp_path/'data')
    with pytest.raises(ValueError,match='full registered'):model.configure_training({'max_instances':12})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no automatic fallback'):fixture(tmp_path/'cuda',backend='cuda')
    for x in (torch.empty(0,16),torch.randn(3,15),torch.full((3,16),float('nan'))):
        with pytest.raises(ValueError,match='finite nonempty'):model(x)


@pytest.mark.parametrize('alpha',[0.,.2,.8])
@pytest.mark.parametrize('event',[0,1])
def test_survival_likelihood_event_censor_and_alpha(tmp_path,alpha,event):
    model,_=fixture(tmp_path/'data',survival=True,alpha=alpha)
    time=float(model.bin_edges[1])+1e-5;index=int(model.time_label([time]).item())
    logits=torch.tensor([[-.5,.2,-.7,1.3]],requires_grad=True)
    hazards=logits.sigmoid();survival=(1-hazards).cumprod(1)
    actual=model.objective({'hazards':hazards,'survival':survival},time,event)
    expected=-(torch.log1p(-hazards[0,:index]).sum()+hazards[0,index].log()) if event else -(1-alpha)*torch.log1p(-hazards[0,:index+1]).sum()
    torch.testing.assert_close(actual,expected)
    a=torch.autograd.grad(actual,logits,retain_graph=True)[0];b=torch.autograd.grad(expected,logits)[0]
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('profile',PROFILES)
def test_multiclass_training_has_three_outputs(tmp_path,profile):
    bundle=Bundle(create_demo(tmp_path/'data',patients=30,classes=3))
    model=MambaMIL(bundle,profile,hidden=16,scan_chunk=16)
    opt=model.make_optimizer(2e-4,1e-5);groups=units(bundle,'train')[:3]
    result=mamba_epoch(model,bundle,groups,opt,'cpu',{})
    assert len(result)==3 and model.optimizer_steps==3
    assert model.forward_unit(bundle,groups[0],'cpu',{})['logits'].shape==(1,3)


def test_missing_cuda_dependencies_report_explicit_failure(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    def missing(name):raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata,'version',missing)
    with pytest.raises(RuntimeError,match='Missing pinned CUDA dependencies'):backend_functions('cuda',64)


def test_long_scan_retains_gradient_across_all_chunk_boundaries():
    torch.manual_seed(4)
    args=[torch.randn(1,4,257),torch.full((1,4,257),-2.),torch.full((4,3),-.001),
          torch.ones(1,3,257),torch.ones(1,3,257)]
    left=[v.clone().requires_grad_() for v in args];right=[v.clone().requires_grad_() for v in args]
    a=selective_scan_torch(*left,chunk_size=16);b=reference()['selective_scan_ref'](*right,delta_softplus=True)
    torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
    a[:,:,-1].sum().backward();b[:,:,-1].sum().backward()
    assert left[0].grad[:,:,0].abs().min()>0.01
    for x,y in zip(left,right):torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=2e-5)

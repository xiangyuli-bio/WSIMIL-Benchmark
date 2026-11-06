"""Original-network/gradient checks, independently read from the locked Git object."""
import ast
import random
import subprocess

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.engine import restore_rng, rng_state
from milplus.smmile_data import NICGrid, author_preprocessing
from milplus.smmile_kernels import author_network, predict_instances, weak_supervision_loss
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def reference(variant, dimension=4, size='small', dropout=True):
    root,sha=verified_source('smmile')
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'randint':random.randint,
           'random':random.random,'sample':random.sample,'device':torch.device('cpu')}
    branch='multi' if variant=='multilabel' else 'single'
    for filename in ['utils/utils.py','models/model_smmile.py']:
        relative=branch+'/'+filename
        code=subprocess.check_output(['git','show',sha+':'+relative],cwd=root,text=True)
        # Independent CPU portability change only; do not import adapter or repair its math.
        code=code.replace('.cuda()', '.to(device)')
        names={'initialize_weights'} if filename.startswith('utils') else {'Attn_Net','Attn_Net_Gated','IAMIL','SMMILe','SMMILe_SINGLE'}
        nodes=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),relative,'exec'),scope)  # noqa: S102 -- independent verified Git AST
    cls=scope['SMMILe_SINGLE'] if variant=='binary' else scope['SMMILe']
    return cls(fea_dim=dimension,n_classes=1 if variant=='binary' else 3,size_arg=size,dropout=dropout,n_refs=3)


def fixture(variant, dimension=4):
    torch.manual_seed(41);random.seed(22)
    coords=np.array([[256*x,256*y] for x in range(7) for y in range(6) if (x,y)!=(2,3)])
    coords=coords[np.random.default_rng(11).permutation(len(coords))]
    x=torch.randn(len(coords),dimension)
    layout=NICGrid(coords,256)
    sp=np.where(np.indices(layout.shape)[0]<3,1,2).astype(np.int64)
    sp[layout.mask==0]=0
    adj=author_preprocessing().generate_adjacency_matrix(sp.T)
    label=torch.tensor([1]) if variant!='multilabel' else torch.tensor([0,2])
    target=torch.tensor([1.]) if variant=='binary' else torch.tensor([0.,1.,0.]) if variant=='multiclass' else torch.tensor([1.,0.,1.])
    return x,layout,sp,adj,label,target


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
@pytest.mark.parametrize('size',['small','big'])
def test_original_full_network_training_updates_and_refinement(variant,size):
    x,layout,sp,adj,label,target=fixture(variant)
    actual=author_network(variant,dimension=4,classes=2 if variant=='binary' else 3,size=size)
    original=reference(variant,size=size)
    original.load_state_dict(actual.state_dict())
    optim_a=torch.optim.Adam(actual.parameters(),lr=2e-5,weight_decay=1e-5)
    optim_b=torch.optim.Adam(original.parameters(),lr=2e-5,weight_decay=1e-5)
    for epoch in range(3):
        opts={'label':label,'instance_eval':epoch>0,'inst_rate':.2,'superpixels':True,
              'group_numbers':3,'drop_with_score':True,'drop_times':1,'return_features':True}
        if variant!='multilabel':opts['sp_smooth']=True
        state=rng_state()
        a=actual(layout.scatter(x),layout.mask,sp,adj,**opts)
        restore_rng(state)
        b=original(layout.scatter(x),layout.mask,sp,adj,**opts)
        equal(a,b)
        loss_a=weak_supervision_loss(a,target,variant)
        if variant=='binary':
            loss_b=F.binary_cross_entropy(b[1][0],target)
            for p in b[1][1:]:loss_b+=F.binary_cross_entropy(p,target)/len(b[1][1:])
        else:
            loss_b=F.binary_cross_entropy(b[1][0],target)/len(b[1])
            for p in b[1][1:]:loss_b=loss_b+F.binary_cross_entropy(p,target)/len(b[1])
        for value in b[4].values():
            if (isinstance(value,(float,int)) or isinstance(value,torch.Tensor) and value.ndim==0) and value>0:
                loss_b=loss_b+value
        equal(loss_a,loss_b)
        optim_a.zero_grad();optim_b.zero_grad();loss_a.backward();loss_b.backward()
        for (name,a_p),(refname,b_p) in zip(actual.named_parameters(),original.named_parameters()):
            assert name==refname
            if a_p.grad is None:assert b_p.grad is None
            else:equal(a_p.grad,b_p.grad)
        optim_a.step();optim_b.step()
        equal(actual.state_dict(),original.state_dict());equal(optim_a.state_dict(),optim_b.state_dict())
    assert actual.ref_net[0].weight.grad is not None
    assert len(a[1])>=4


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
def test_label_free_predictions_match_original_and_input_order(variant):
    x,layout,sp,adj,_label,_target=fixture(variant)
    model=author_network(variant,dimension=4,classes=2 if variant=='binary' else 3).eval()
    result=predict_instances(model,x,layout)
    for query in [torch.tensor([0]),torch.tensor([1]),torch.tensor([0,2]) if variant=='multilabel' else torch.tensor([1])]:
        original=model(layout.scatter(x),layout.mask,sp,adj,label=query,instance_eval=True,inst_rate=.2)
        equal(result['bag_scores'],original[1][0])
        equal(result['instance_attention_product'],layout.restore_rows(original[0]))
        equal(result['refined_scores'][-1],layout.restore_rows(original[3]))
    torch.testing.assert_close(result['attention'].sum(0),torch.ones_like(result['bag_scores']))
    assert result['refined_scores'][-1].shape==(len(x),2 if variant=='binary' else 4)
    assert len(result['refined_scores'])==3
    assert predict_instances(model,x,layout,refinement=False)['refined_scores']==[]
    model.train()
    with pytest.raises(ValueError,match='eval mode'):predict_instances(model,x,layout)


@pytest.mark.parametrize('dimension',[512,1024,1536])
def test_native_spatial_conv_and_all_ref_heads_gradient(dimension):
    x,layout,sp,adj,label,target=fixture('multiclass',dimension)
    model=author_network('multiclass',dimension=dimension,classes=3,dropout=False)
    result=model(layout.scatter(x),layout.mask,sp,adj,label=label,instance_eval=True,inst_rate=.2,
                 superpixels=True,group_numbers=3,sp_smooth=True,drop_with_score=True,mrf=True,tau=.1)
    loss=weak_supervision_loss(result,target,'multiclass');loss.backward()
    assert torch.isfinite(loss)
    assert model.conv_nic[0].weight.grad.norm()>0
    assert result[4]['instance_loss']>0 and torch.isfinite(result[4]['mrf_loss'])
    for head in model.ref_net:assert head.weight.grad is not None


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
def test_mrf_without_stochastic_edge_drop_matches_literal_source_value_and_gradient(variant):
    model=author_network(variant,dimension=4,classes=2 if variant=='binary' else 3)
    original=reference(variant)
    ids=np.array([1,1,1,2,2,2])
    adjacency=np.array([[0,0,0],[0,0,1],[0,1,0]])
    scores=torch.rand(6,4,requires_grad=True)
    b=scores.detach().clone().requires_grad_()
    args=[ids,np.unique(ids),adjacency]
    if variant=='multilabel':args.append(torch.tensor([0,2]))
    loss=model.markov_field_energy(scores,*args,drop=False)
    ref=original.markov_field_energy(b,*args,drop=False)
    equal(loss,ref);loss.backward();ref.backward();equal(scores.grad,b.grad)


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
def test_repaired_mrf_handles_constant_and_isolated_superpixels(variant):
    model=author_network(variant,dimension=4,classes=2 if variant=='binary' else 3)
    scores=torch.tensor([[.2,.8],[.8,.2]],requires_grad=True)
    ids=np.array([1,1]);adj=np.zeros((2,2),np.uint8)
    args=[ids,np.unique(ids),adj]
    if variant=='multilabel':args.append(torch.tensor([0]))
    # Each squared deviation is .09; constant finite losses are kept, no neighbors -> zero pairwise.
    loss=model.markov_field_energy(scores,*args)
    torch.testing.assert_close(loss,torch.tensor(.8*.09));loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    zero=torch.ones(2,2,requires_grad=True)
    value=model.markov_field_energy(zero,*args)
    assert value==0 and torch.isfinite(value)


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
def test_consistency_negative_bags_or_empty_multilabel(variant):
    x,layout,sp,adj,label,target=fixture(variant)
    model=author_network(variant,dimension=4,classes=2 if variant=='binary' else 3)
    if variant=='multiclass':
        assert not hasattr(model,'consistency_penalty')
        return
    label=torch.tensor([0]) if variant=='binary' else torch.tensor([],dtype=torch.long)
    target=torch.tensor([0.]) if variant=='binary' else torch.zeros(3)
    result=model(layout.scatter(x),layout.mask,sp,adj,label=label,consistency=True,
                 instance_eval=True,superpixels=True,group_numbers=3,drop_with_score=True,inst_rate=.2,mrf=True)
    assert len(result[1])==1 and result[4]['consist_loss']>0
    loss=weak_supervision_loss(result,target,variant);loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
def test_parameter_state_reloaded_without_class_pickle(tmp_path,variant):
    x,layout,*_=fixture(variant)
    kwargs={'variant':variant,'dimension':4,'classes':2 if variant=='binary' else 3}
    model=author_network(**kwargs).eval()
    expected=predict_instances(model,x,layout)
    torch.save(model.state_dict(),tmp_path/'state.pt')
    clone=author_network(**kwargs).eval()
    clone.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
    equal(expected,predict_instances(clone,x,layout))


@pytest.mark.parametrize('kwargs',[{'variant':'other'},{'size':'other'},{'dimension':0},
    {'classes':1},{'variant':'binary','classes':3},{'refs':0},{'dropout':1},{'dropout_rate':float('nan')}])
def test_invalid_source_constructor_options(kwargs):
    with pytest.raises(ValueError):author_network(**kwargs)


def test_objective_rejects_label_corruption_and_nonfinite_losses():
    p=torch.tensor([.2,.7,.3],requires_grad=True)
    result=(None,[p],None,None,{'instance_loss':torch.tensor(float('nan'))})
    with pytest.raises(ValueError,match='auxiliary'):weak_supervision_loss(result,torch.tensor([0.,1.,0.]),'multiclass')
    result=(None,[p],None,None,{})
    for target in [torch.tensor([.1,.9,0.]),torch.tensor([1.,1.,0.]),torch.ones(2)]:
        with pytest.raises(ValueError):weak_supervision_loss(result,target,'multiclass')

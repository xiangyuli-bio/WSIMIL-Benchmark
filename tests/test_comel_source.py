"""Full-width source mechanisms, including the confidence-gated BPPL branch."""
import copy

import pytest
import torch
from torch.nn import functional as F

from milplus import comel_source as c
from milplus.engine import rng_state,restore_rng


def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def manual_step(owner,x,y,epoch):
    model=owner.model;model.train();owner.optimizer.zero_grad()
    logits,a,bag,inst=model(x,return_attn=True,no_norm=False,return_inst=True)
    a=a.permute(1,0);prob=(a-a.min(0,keepdim=True).values)/(a.max(0,keepdim=True).values+1e-10)
    prob=torch.cat([1-prob,prob],1)
    confidence,pred=logits.detach().softmax(1).max(1);loss=F.cross_entropy(logits,y);patch=loss.new_zeros(())
    if y.item()%2==1 and pred.item()==y.item():
        features=inst.reshape(-1,inst.shape[-1]);reverse=((1-a)/(1-a).sum(0)).T@features
        unit=features/features.norm(dim=1,keepdim=True)
        prototypes=torch.cat([reverse/reverse.norm(dim=1,keepdim=True),bag/bag.norm(dim=1,keepdim=True)],0)
        conf,pseudo=(unit@prototypes.T/owner.T).softmax(1).max(1)
        keep=(confidence>owner.tau1)*(conf>owner.tau2)
        if keep.any():patch=keep.sum()/len(features)*F.nll_loss(torch.log(prob[keep]+1e-10),pseudo[keep])
    loss=loss+patch
    penalty=loss.new_zeros(())
    if owner.current_task:
        for module in model.update_modules.values():
            layer=module.lora_layers[-1];d,u=layer.down.weight,layer.up.weight;rank=d.shape[0]
            penalty=penalty+((d@d.T-torch.eye(rank)).square().sum()+(u.T@u-torch.eye(rank)).square().sum())/rank**2
    loss=loss+penalty;loss.backward()
    if owner.current_task:
        for module in model.update_modules.values():
            layer=module.lora_layers[-1];gd=layer.down.weight.grad;gu=layer.up.weight.grad
            gd_projection=None;gu_projection=None
            for previous in module.lora_layers[:-1]:
                d,u=previous.down.weight,previous.up.weight
                a=gd@(d.T@d);b=(u@u.T)@gu
                gd_projection=a if gd_projection is None else gd_projection+a
                gu_projection=b if gu_projection is None else gu_projection+b
            layer.down.weight.grad=gd-gd_projection;layer.up.weight.grad=gu-gu_projection
    owner.optimizer.step();owner.scheduler.step()
    tau2=.5*(1+owner.tau*(confidence*(pred==y).float()))
    return float(loss),float(patch),tau2


@pytest.mark.parametrize('task',[0,1,2])
@pytest.mark.parametrize('profile',sorted(c.PROFILES))
def test_complete_native_objective_optimizer_gradient_projection_oracle(task,profile):
    torch.set_num_threads(2);torch.manual_seed(83);a=c.model(1024)
    for current in range(task):
        owner=c.context(a,current,100,'cpu');c.transition(owner,profile);c.add_head(a)
    with torch.no_grad():
        for head in a.classifier:head.weight.zero_();head.bias.fill_(-8)
        a.classifier[-1].bias[1]=8
    b=c.restore_model(1024,c.architecture(a),copy.deepcopy(a.state_dict()),'cpu')
    oa=c.context(a,task,100,'cpu');ob=c.context(b,task,100,'cpu')
    c.optimizer(oa,[None]);c.optimizer(ob,[None]);x=torch.randn(1,17,1024);y=torch.tensor([2*task+1])
    state=rng_state();expected,patch,tau2=manual_step(oa,x,y,0)
    assert patch>0,'The positive, correct, high-confidence BPPL branch must actually execute'
    restore_rng(state);actual=c.epoch(ob,[(x,y,torch.full((1,17,1),-1))],0)
    assert actual['loss']==expected;assert torch.equal(ob.tau2,tau2)
    exact(a.state_dict(),b.state_dict());exact(c.gradients(a),c.gradients(b))
    exact(oa.optimizer.state_dict(),ob.optimizer.state_dict());exact(oa.scheduler.state_dict(),ob.scheduler.state_dict())
    assert len(a.agent_blocks)==2 and a.mlp_dim==512 and a.rank==16
    if task:
        assert any(p.grad is not None for name,p in b.named_parameters() if name.endswith('patch_to_emb.0.weight'))


@pytest.mark.parametrize('count',[17,257,1025])
def test_native_full_regions_gradients_and_dynamic_restore(count):
    torch.set_num_threads(2);torch.manual_seed(19);a=c.model(1024);x=torch.randn(1,count,1024)
    logits,attention,bag,inst=a(x,return_attn=True,return_inst=True)
    assert logits.shape==(1,2) and attention.shape==(1,count) and inst.shape==(1,count,512)
    F.cross_entropy(logits,torch.tensor([1])).backward();c.transition(c.context(a,0,100,'cpu'),'comel_owlora_source_patient_reference');c.add_head(a)
    state=copy.deepcopy(a.state_dict());grads=c.gradients(a);layout=c.architecture(a)
    b=c.restore_model(1024,layout,state,'cpu');c.restore_gradients(b,grads)
    exact(state,b.state_dict());exact(grads,c.gradients(b));assert layout==c.architecture(b)
    a.eval();b.eval()
    assert torch.equal(a(x),b(x));assert a(x).shape==(1,4)


def test_missing_source_svd_assignment_and_explicit_repair_are_different():
    torch.set_num_threads(2);torch.manual_seed(3);a=c.model(1024);b=copy.deepcopy(a);state=rng_state()
    c.transition(c.context(a,0,100,'cpu'),'comel_owlora_source_patient_reference')
    after=rng_state();restore_rng(state);c.transition(c.context(b,0,100,'cpu'),'comel_owlora_svd_basis_repaired')
    assert torch.equal(after['torch'],rng_state()['torch'])
    for name,module in a.update_modules.items():
        repaired=b.update_modules[name];basis=repaired.lora_layers[0];rank=basis.rank
        assert torch.count_nonzero(module.lora_layers[0].up.weight)==0
        assert torch.count_nonzero(basis.up.weight)>0
        torch.testing.assert_close(basis.down.weight@basis.down.weight.T,torch.eye(rank),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(basis.up.weight.T@basis.up.weight,torch.eye(rank),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close((basis.up.weight*basis.scales)@basis.down.weight,repaired.weight,rtol=0,atol=0)
        assert module.lora_layers[-1].rank==(48 if 'qkv' in name else 16)
        assert len(module.lora_layers)==2 and module.num_expanded==2


def test_native_inactive_base_and_previous_lora_grads_preserved_between_epochs():
    torch.set_num_threads(2);torch.manual_seed(13);m=c.model(1024);c.transition(c.context(m,0,100,'cpu'),'comel_owlora_source_patient_reference');c.add_head(m)
    owner=c.context(m,1,100,'cpu');loader=[(torch.randn(1,17,1024),torch.tensor([2]),torch.full((1,17,1),-1))];c.optimizer(owner,loader)
    active={id(p) for group in owner.optimizer.param_groups for p in group['params']}
    base=m.patch_to_emb[0].weight;assert base.requires_grad and id(base) not in active
    c.epoch(owner,loader,0);first=base.grad.clone();c.epoch(owner,loader,1)
    assert base.grad is not None and not torch.equal(base.grad,first)
    saved=c.gradients(m);layout=c.architecture(m);n=c.restore_model(1024,layout,m.state_dict(),'cpu');c.restore_gradients(n,saved)
    exact(saved,c.gradients(n));other=c.context(n,1,100,'cpu');c.optimizer(other,loader);c.restore_owner(other,c.owner_state(owner))
    exact(owner.optimizer.state_dict(),other.optimizer.state_dict());exact(owner.scheduler.state_dict(),other.scheduler.state_dict())

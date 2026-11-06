"""Full dimensions/source operators, both stages and100x500posterior, not WSI scores."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import yaml

from milplus.data import Bundle
from milplus.engine import seed_everything
from milplus.mexd import MExD,source_and_uncertainty_metrics
from milplus.mexd_demo import create_demo
from milplus.mexd_source import components,complete_radam
from milplus.registry import inspect_method


@pytest.fixture
def bundle(tmp_path):return Bundle(create_demo(tmp_path/'task','tcga_nsclc',384,17))


def make(bundle,cohort='tcga_nsclc',moe=1,diffusion=1,seed=73):
    torch.set_num_threads(2)
    profile='mexd_class_port_reference'if cohort=='class_port'else'mexd_'+cohort+'_paper_reference'
    model=MExD(bundle,profile,moe_epochs=moe,diffusion_epochs=diffusion,sampling_seed=seed)
    model.configure_training({'epochs':moe+diffusion,'max_instances':None,'grad_clip':1.})
    return model


def exact(a,b):
    if torch.is_tensor(a):assert torch.is_tensor(b)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray)and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b)and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert type(a)==type(b)and a==b


@pytest.mark.parametrize('cohort,classes,experts',[('camelyon16',2,2),('tcga_nsclc',2,3),('bracs',3,4)])
@pytest.mark.parametrize('dimension',[384,768])
def test_full_three_cohort_and_two_FM_graphs_are_not_thinned(tmp_path,cohort,classes,experts,dimension):
    bundle=Bundle(create_demo(tmp_path/'task',cohort,dimension,17));model=make(bundle,cohort)
    assert len(model.auxiliary.routers)==experts and len(model.auxiliary.classifier)==experts
    assert model.noise.encoder_x[0].weight.shape==(384,512)
    assert model.noise.encoder_x[2].weight.shape==(384,384)
    assert model.noise.encoder_x[4].weight.shape==(512,384)
    assert model.noise.lin1.embed.weight.shape==(501,512)
    for adapter in [model.auxiliary.trans1,model.auxiliary.agg1]:
        for layer in [adapter.layer1,adapter.layer2]:
            assert layer.attn.heads==8 and layer.attn.num_landmarks==256 and layer.attn.pinv_iterations==6
            assert layer.attn.to_qkv.weight.shape==(1536,512)
            assert layer.attn.res_conv.weight.shape==(8,1,33,1)
        assert [adapter.pos_layer.proj.kernel_size,adapter.pos_layer.proj1.kernel_size,adapter.pos_layer.proj2.kernel_size]==[(7,7),(5,5),(3,3)]
    x,_=bundle.load(bundle.records('train')[0]);model.auxiliary.eval();result=model.auxiliary(x)
    assert result['expert_features'].shape==(experts,512)and result['logits'].shape==(1,classes)
    assert sum(result['selected_counts'])==result['observed_sparse_instances']
    assert all(bool((indices<len(x)).all())for indices in result['selected_rows'])


def test_exact_paper_routing_means_confidences_and_unormalized_condition(bundle):
    seed_everything(109);model=make(bundle);model.auxiliary.eval()
    x,_=bundle.load(bundle.records('train')[0]);result=model.auxiliary(x)
    transformed=model.auxiliary.trans1(x)['patch_feature'][0,:len(x)]
    means=[];confidence=[]
    for r,router in enumerate(model.auxiliary.routers):
        score=router(transformed).softmax(-1);eta=0 if r==0 else 1
        assignment=(score.argmax(-1)==eta).nonzero().flatten()
        if len(assignment):
            selected=assignment[torch.argsort(score[assignment,eta],descending=True,stable=True)[:max(1,int((.25 if r==0 else .5)*len(assignment)))]]
            assert torch.equal(selected,result['selected_rows'][r])
            mean=transformed[selected].mean(0)
        else:mean=transformed.new_zeros(512)
        logits=model.auxiliary.classifier[r](mean);p=logits.softmax(-1)
        means.append(mean);confidence.append(p[r])
    expected=sum(a*b for a,b in zip(means,confidence))
    assert torch.equal(result['condition'],expected)


@pytest.mark.parametrize('target',[0,1])
def test_eq9_joint_loss_and_every_active_gradient_matches_literal_equation(bundle,target):
    seed_everything(110);model=make(bundle);aux=model.auxiliary;aux.eval()
    x,_=bundle.load(bundle.records('train')[0]);label=torch.tensor([target])
    actual=aux(x);loss,parts=aux.loss(actual,label)
    expected=F.cross_entropy(actual['logits'],label)
    if actual['selected_counts'][0]:expected+=F.cross_entropy(actual['expert_logits'][0:1],torch.tensor([0]))
    if actual['selected_counts'][target+1]:expected+=F.cross_entropy(actual['expert_logits'][target+1:target+2],torch.tensor([target+1]))
    assert torch.equal(loss,expected)
    parameters=list(aux.parameters())
    a=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(expected,parameters,allow_unused=True)
    for p,q in zip(a,b):
        if p is None:assert q is None
        else:assert torch.equal(p,q)
    assert any(g is not None and g.abs().sum()>0 for g in a)


def test_hard_router_gradient_is_not_replaced_by_soft_or_straight_through(bundle):
    seed_everything(111);model=make(bundle);optim=model.make_optimizers(2e-4,1e-5);model.start_epoch(0)
    before={n:p.detach().clone()for n,p in model.auxiliary.routers.named_parameters()};seen={}
    hook=optim[0].register_step_pre_hook(lambda op,a,k:seen.update({n:p.grad for n,p in model.auxiliary.routers.named_parameters()}))
    x,_=bundle.load(bundle.records('train')[0]);model.train_moe(x,torch.tensor([0]),optim[0]);hook.remove()
    assert seen and all(v is None for v in seen.values())
    for n,p in model.auxiliary.routers.named_parameters():assert torch.equal(p,before[n])
    assert all(p.grad is None for p in model.parameters())


def test_original_RAdam_all_updates_cache_moments_and_recovery_are_preserved():
    cls=components()[3];a=torch.nn.Parameter(torch.randn(3,4));b=torch.nn.Parameter(a.detach().clone())
    x=complete_radam([a],2e-4,1e-5);y=cls([b],lr=2e-4,weight_decay=1e-5)
    for step in range(17):
        lr=2e-4*(1+np.cos(step/17*np.pi))/2
        x.param_groups[0]['lr']=y.param_groups[0]['lr']=lr
        gradient=torch.full_like(a,(step+1)/17)
        a.grad=gradient.clone();b.grad=gradient.clone();x.step();y.step()
        assert torch.equal(a,b)
        state=x.state_dict();cache=state.pop('source_step_cache');exact(state,y.state_dict());exact(cache,y.buffer)
        if step==7:
            replacement=complete_radam([a],2e-4,1e-5);replacement.load_state_dict(x.state_dict());x=replacement


def test_complete_original_denoiser_q_sample_loss_gradients_and_Adam(bundle):
    seed_everything(112);model=make(bundle);optim=model.make_optimizers(2e-4,1e-5)
    model.start_epoch(1);reference=copy.deepcopy(model.noise)
    original=torch.optim.Adam(reference.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    reference_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(original,model.diffusion_epochs,eta_min=0.)
    x,_=bundle.load(bundle.records('train')[0]);label=torch.tensor([0]);cpu_rng=torch.get_rng_state();seen={}
    hook=optim[1].register_step_pre_hook(lambda op,a,k:seen.update({n:None if p.grad is None else p.grad.clone()for n,p in model.noise.named_parameters()}))
    before={n:p.detach().clone()for n,p in model.auxiliary.named_parameters()}
    actual=model.train_diffusion(x,label,optim[1]);hook.remove()
    torch.set_rng_state(cpu_rng);model.auxiliary.eval()
    with torch.no_grad():aux=model.auxiliary(x)
    t=torch.randint(0,500,(1,));t=torch.cat([t,499-t])[:1]
    truth=F.one_hot(label,num_classes=2).float();noise=torch.randn_like(truth)
    value=model.diffusion.q_sample(truth,aux['probabilities'],model.alpha_bar_sqrt,model.one_minus_alpha_bar_sqrt,t,noise=noise)
    expected=(noise-reference(aux['condition'],value,t,aux['probabilities'])).square().mean()
    expected.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
    assert actual['noise_mse']==float(expected.detach())
    for n,p in reference.named_parameters():
        if p.grad is None:assert seen[n]is None
        else:assert torch.equal(p.grad,seen[n]),n
    original.step();original.zero_grad(set_to_none=True)
    exact(model.noise.state_dict(),reference.state_dict());exact(optim[1].state_dict(),original.state_dict())
    for n,p in model.auxiliary.named_parameters():assert not p.requires_grad and torch.equal(p,before[n])


def test_two_stage_epoch_recovery_keeps_both_optimizers_cosines_and_RNG(bundle):
    seed_everything(113);a=make(bundle,moe=1,diffusion=2);oa=a.make_optimizers(2e-4,1e-5)
    a.start_epoch(0);a.epoch(bundle,oa,torch.device('cpu'));a.finish_epoch(bundle,torch.device('cpu'),{})
    saved={'model':copy.deepcopy(a.state_dict()),'optim':[copy.deepcopy(o.state_dict())for o in oa],'rng':torch.get_rng_state()}
    for epoch in [1,2]:a.start_epoch(epoch);a.epoch(bundle,oa,torch.device('cpu'));a.finish_epoch(bundle,torch.device('cpu'),{})
    expected_rng=torch.get_rng_state();b=make(bundle,moe=1,diffusion=2);ob=b.make_optimizers(2e-4,1e-5)
    b.load_state_dict(saved['model'])
    for optimizer,state in zip(ob,saved['optim']):optimizer.load_state_dict(state)
    torch.set_rng_state(saved['rng'])
    for epoch in [1,2]:b.start_epoch(epoch);b.epoch(bundle,ob,torch.device('cpu'));b.finish_epoch(bundle,torch.device('cpu'),{})
    exact(a.state_dict(),b.state_dict())
    for x,y in zip(oa,ob):exact(x.state_dict(),y.state_dict())
    assert torch.equal(expected_rng,torch.get_rng_state())and a.moe_steps==4 and a.diffusion_steps==8


def test_all100_posterior_paths_execute_full500_source_steps_and_restore_RNG(bundle,monkeypatch):
    seed_everything(114);model=make(bundle);model.start_epoch(1);model.eval()
    x,_=bundle.load(bundle.records('test')[0]);before=torch.get_rng_state();calls=[]
    hook=model.noise.register_forward_hook(lambda m,args,value:calls.append((int(args[2].item()),value.shape)))
    actual=model.posterior(x,'slide_untouched');hook.remove()
    assert len(calls)==500 and [i for i,_ in calls]==list(range(499,-1,-1))
    assert all(shape==(100,2)for _,shape in calls)
    assert torch.equal(before,torch.get_rng_state())
    with torch.no_grad():aux=model.auxiliary(x)
    seed=int.from_bytes(hashlib.sha256((str(model.sampling_seed)+'\0slide_untouched\0mexd500x100').encode()).digest()[:8],'big')%(2**63)
    with torch.random.fork_rng():
        torch.manual_seed(seed);prior=aux['probabilities'].expand(100,-1)
        expected=model.diffusion.p_sample_loop(model.noise,aux['condition'],prior,prior,500,model.alphas,model.one_minus_alpha_bar_sqrt,only_last_sample=True)
    assert torch.equal(actual['raw_draws'],expected)
    assert torch.equal(actual['probabilities'],expected.softmax(-1).mean(0,keepdim=True))
    assert torch.equal(actual['source_sigmoid_scores'],expected.softmax(-1).sigmoid().mean(0,keepdim=True))


def test_seeded_prediction_raw_draw_proofs_scores_and_uncertainty_are_replayable(bundle,tmp_path):
    seed_everything(115);model=make(bundle);model.start_epoch(1);model.bind_output(tmp_path/'run')
    a=model.predict_subset(bundle,'test',torch.device('cpu'),{});b=model.predict_subset(bundle,'test',torch.device('cpu'),{})
    assert a.to_csv(index=False)==b.to_csv(index=False)
    assert (a.mexd_posterior_samples==100).all()and(a.mexd_reverse_steps==500).all()
    metrics=source_and_uncertainty_metrics(a)
    assert 0<=metrics['mexd_slide_pavpu']<=1 and 0<=metrics['mexd_source_sigmoid_softmax_macro_ovr_auc']<=1
    assert len(list((tmp_path/'run/posterior_draws').rglob('raw_signals.npy')))==4
    for path in(tmp_path/'run/posterior_draws').rglob('raw_signals.npy'):assert np.load(path,allow_pickle=False).shape==(100,2)


def test_cohort_background_role_is_explicit_and_no_stage_can_be_omitted(tmp_path):
    root=create_demo(tmp_path/'task','camelyon16',384,17);bundle=Bundle(root);m=make(bundle,'camelyon16')
    with pytest.raises(ValueError,match='two|BOTH'):MExD(bundle,'mexd_camelyon16_paper_reference',moe_epochs=0)
    with pytest.raises(ValueError,match='total epochs'):m.configure_training({'epochs':1,'max_instances':None,'grad_clip':1.})
    task=yaml.safe_load((root/'task.yaml').read_text());task.pop('class_roles');(root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='background'):make(Bundle(root),'camelyon16')


def test_native_budgets_and_sampling_architecture_have_no_toy_fallback(bundle):
    spec=inspect_method('MExD');assert spec['native_recipe']['moe_epochs']==100 and spec['native_recipe']['diffusion_epochs']==200
    assert spec['native_recipe']['diffusion_steps']==500 and spec['native_recipe']['posterior_samples']==100
    m=make(bundle)
    with pytest.raises(ValueError,match='every registered'):m.configure_training({'epochs':2,'max_instances':16,'grad_clip':1.})
    with pytest.raises(ValueError,match='clip_norm1'):m.configure_training({'epochs':2,'max_instances':None,'grad_clip':None})
    with pytest.raises(TypeError):MExD(bundle,spec['id'],hidden=32)
    with pytest.raises(TypeError):MExD(bundle,spec['id'],diffusion_steps=10)


@pytest.mark.parametrize('classes,dimension,background',[(5,1536,False),(7,2560,False),(5,1536,True)])
def test_arbitrary_class_and_FM_port_retains_paper_expert_definition(tmp_path,classes,dimension,background):
    bundle=Bundle(create_demo(tmp_path/'class_task','class_port',dimension,17,classes,background))
    model=make(bundle,'class_port')
    assert model.auxiliary.experts==(classes if background else classes+1)
    assert model.auxiliary.trans1._fc1[0].weight.shape==(512,dimension)
    assert model.auxiliary.agg1._fc2.out_features==classes
    x,_=bundle.load(bundle.records('train')[0]);model.auxiliary.eval();out=model.auxiliary(x)
    assert out['expert_logits'].shape==(model.auxiliary.experts,model.auxiliary.experts)
    assert out['logits'].shape==(1,classes)and model.noise.lin4.out_features==classes
    loss,_=model.auxiliary.loss(out,torch.tensor([classes-1]));loss.backward()
    assert model.auxiliary.trans1.layer1.attn.to_qkv.weight.grad is not None


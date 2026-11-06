"""Native full graph/source gradients and independent mathematical selection invariants."""
import copy

import pytest
import torch
from torch.nn import functional as F

from milplus.nciemil_source import NcIEAggregator,author_classes,discriminative_selection,max_instance_bce


@pytest.mark.parametrize('dimension,classes,k',[(768,4,128),(768,2,512),(1024,4,128)])
def test_full_native_source_initialization_logits_every_gradient_and_RNG(dimension,classes,k):
    torch.set_num_threads(2)
    torch.manual_seed(8808);wrapped=NcIEAggregator(dimension,classes,k)
    torch.manual_seed(8808);raw=author_classes().NcIEMIL(in_dim=dimension,in_chans=2*k,latent_dim=1024,
        n_classes=classes,num_heads=4,ratio=32,attn_drop=.2,proj_drop=.2,conv_drop=.2)
    for name,value in raw.state_dict().items():torch.testing.assert_close(wrapped.network.state_dict()[name],value,rtol=0,atol=0)
    g=torch.Generator().manual_seed(2026);x=torch.randn(4,2*k,dimension,generator=g);labels=torch.arange(4)%classes
    a=x.clone().requires_grad_();b=x.clone().requires_grad_();before=torch.get_rng_state()
    expected=raw(a);F.cross_entropy(expected,labels).backward();after=torch.get_rng_state()
    torch.set_rng_state(before);actual=wrapped(b);F.cross_entropy(actual,labels).backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(b.grad,a.grad,rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),after,rtol=0,atol=0)
    for name,p in raw.named_parameters():
        q=dict(wrapped.network.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(q.grad,p.grad,rtol=0,atol=0)
    assert wrapped.network.latent_dim==1024 and wrapped.network.num_heads==4
    assert wrapped.network.c_attn.fc[0].out_channels==32


@pytest.mark.parametrize('k,classes',[(128,4),(512,2)])
def test_named_KV_repair_retains_one_independent_key_and_value_per_original_instance(k,classes):
    torch.set_num_threads(2)
    net=NcIEAggregator(768,classes,k,variant='paper_independent_kv_repaired').eval()
    widths=[];hook=net.network._kv.register_forward_hook(lambda m,args,value:widths.append((args[0].shape,value.shape)))
    x=torch.randn(4,2*k,768)
    output=net(x);hook.remove();assert output.shape==(4,classes)
    assert widths==[(torch.Size([4,2*k,1024]),torch.Size([4,2*k,2048]))]
    F.cross_entropy(output,torch.arange(4)%classes).backward()
    assert net.network._kv.weight.grad.shape==(2048,1024) and torch.isfinite(net.network._kv.weight.grad).all()
    assert net.protocol()['full_method_admission'] is False


def test_selection_retains_all_highK_lowK_rows_in_descending_rank_order_and_drops_only_middle():
    features=torch.arange(281*768,dtype=torch.float32).reshape(281,768).requires_grad_()
    scores=torch.linspace(0,1,281).requires_grad_()
    selected,ids=discriminative_selection(features,scores,128)
    expected=torch.cat([torch.arange(280,152,-1),torch.arange(127,-1,-1)])
    torch.testing.assert_close(ids,expected,rtol=0,atol=0)
    torch.testing.assert_close(selected,features[expected],rtol=0,atol=0)
    selected.sum().backward()
    assert torch.count_nonzero(features.grad[:,0])==256 and scores.grad is None


def test_selection_ties_are_explicitly_stable_without_duplicate_or_padded_instances():
    x=torch.randn(281,768);values=torch.full((281,),.5)
    _,ids=discriminative_selection(x,values,128)
    assert ids.tolist()==list(range(128))+list(range(153,281))
    assert len(ids.unique())==256


@pytest.mark.parametrize('kind',['short','misaligned','nonfinite','not_probability'])
def test_selection_rejects_missing_or_invalid_filter_evidence(kind):
    x=torch.randn(281,768);scores=torch.rand(281)
    if kind=='short':x=x[:200];scores=scores[:200]
    if kind=='misaligned':scores=scores[:-1]
    if kind=='nonfinite':scores[0]=float('nan')
    if kind=='not_probability':scores[0]=1.1
    with pytest.raises(ValueError,match='actual finite positive probabilities'):discriminative_selection(x,scores,128)


def test_no_singleton_batchnorm_fallback_or_arbitrary_instance_padding():
    net=NcIEAggregator()
    with pytest.raises(ValueError,match='at least2'):net(torch.randn(1,256,768))
    with pytest.raises(ValueError,match='complete ordered'):net(torch.randn(4,255,768))


def test_paper_binary_max_objective_matches_probability_equation_and_only_winner_gradients():
    bags=[torch.tensor([-.8,.3,1.7,-1.2],requires_grad=True),torch.tensor([.2,-1.3],requires_grad=True)]
    labels=torch.tensor([1.,0.]);out=max_instance_bce(bags,labels)
    p=torch.stack([torch.sigmoid(x).max() for x in bags])
    reference=-(labels*torch.log(p)+(1-labels)*torch.log1p(-p)).mean()
    torch.testing.assert_close(out['loss'],reference,rtol=1e-6,atol=1e-7)
    assert out['winning_original_indices'].tolist()==[2,0]
    out['loss'].backward()
    assert (bags[0].grad!=0).nonzero().flatten().tolist()==[2]
    assert (bags[1].grad!=0).nonzero().flatten().tolist()==[0]


def test_paper_max_objective_has_explicit_first_patch_ties_and_finite_extreme_loss():
    bag=torch.tensor([3.,3.,1.],requires_grad=True)
    value=max_instance_bce([bag],torch.tensor([0.]));value['loss'].backward()
    assert value['winning_original_indices'].tolist()==[0]
    assert bag.grad[0]>0 and bag.grad[1:].eq(0).all()
    result=max_instance_bce([torch.tensor([-1000.,-1001.]),torch.tensor([999.,1000.])],torch.tensor([1.,0.]))
    assert torch.isfinite(result['loss']) and result['loss'].item()==1000


@pytest.mark.parametrize('bad',['soft_labels','missing_labels','empty_bag','nonfinite'])
def test_binary_weak_objective_rejects_unproven_or_incomplete_targets(bad):
    bags=[torch.zeros(257),torch.ones(1025)];labels=torch.tensor([0.,1.])
    if bad=='soft_labels':labels[0]=.2
    if bad=='missing_labels':labels=labels[:1]
    if bad=='empty_bag':bags[0]=bags[0][:0]
    if bad=='nonfinite':bags[0][0]=float('nan')
    with pytest.raises(ValueError):max_instance_bce(bags,labels)

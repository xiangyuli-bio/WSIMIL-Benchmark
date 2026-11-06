"""Actual full pretrained CLIP mechanisms, not smaller synthetic text networks."""
import pytest
import torch
from torch.nn import functional as F

from milplus.slip_source import ASSETS, SLIPReference, text_backbone, tokenize_function


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2066)
        yield
    torch.set_num_threads(previous)


def model(architecture='clip_vit_b16', classes=3):
    return SLIPReference(architecture,['{}','histology showing {}'],
        [[f'synthetic class {i}'] for i in range(classes)],
        [['synthetic gland','synthetic acinus'],['synthetic stroma'],['synthetic lymphocytes']])


@pytest.mark.parametrize('architecture',list(ASSETS))
def test_native_actual_pretrained_text_and_prompt_only_gradients(architecture):
    m=model(architecture);base=text_backbone(architecture)
    assert len(base.transformer.resblocks)==12
    assert base.transformer.resblocks[0].attn.num_heads==8
    assert base.token_embedding.weight.shape==(49408,512)
    assert base.positional_embedding.shape==(77,512)
    assert base.text_projection.shape==(512,ASSETS[architecture][2])
    assert all(not p.requires_grad for p in base.parameters())
    assert [(n,tuple(p.shape)) for n,p in m.named_parameters() if p.requires_grad]==[('prompt_learner.ctx',(3,1,512))]
    x=torch.randn(31,m.dimension,requires_grad=True)
    loss,pred=m.loss(x,2);loss.backward()
    assert pred.shape==(1,3) and torch.isfinite(loss)
    assert x.grad is None, 'Original SLIP attention and class representations are detached'
    grad=m.prompt_learner.ctx.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize('architecture',list(ASSETS))
@pytest.mark.parametrize('classes',[2,3])
def test_original_source_objective_and_gradients_equal_independent_formula(architecture,classes):
    m=model(architecture,classes);x=torch.randn(19,m.dimension)
    text=F.normalize(m.text_encoder(m.prompt_learner(),m.tokenized_prompts),dim=1).T
    with torch.no_grad():
        features=F.normalize(x,dim=1)
        patch_tissue=(features@m.tissue_weights/m.temperature).softmax(1)
        tissue_class=(m.tissue_weights.T@text/m.temperature).softmax(0)
        attention=patch_tissue@tissue_class
        representations=F.normalize(features.T@attention,dim=0)
    corr=representations.T@text
    expected=F.cross_entropy(corr.flatten().unsqueeze(0)/m.temperature,
                             torch.tensor([classes*1+1]))
    expected_gradient,=torch.autograd.grad(expected,m.prompt_learner.ctx)
    actual,pred=m.loss(x,1)
    actual_gradient,=torch.autograd.grad(actual,m.prompt_learner.ctx)
    assert torch.allclose(actual,expected,rtol=1e-5,atol=1e-6)
    assert torch.allclose(actual_gradient,expected_gradient,rtol=2e-4,atol=2e-5)
    assert torch.allclose(pred[0],corr.diag()/m.temperature,rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize('architecture',list(ASSETS))
def test_singleton_bag_port_and_actual_source_sgd_update(architecture):
    m=model(architecture);optimizer=torch.optim.SGD(m.prompt_learner.parameters(),lr=.002)
    x=torch.randn(1,m.dimension)
    before=m.prompt_learner.ctx.detach().clone()
    loss,pred=m.loss(x,0);loss.backward();gradient=m.prompt_learner.ctx.grad.detach().clone()
    optimizer.step();optimizer.zero_grad()
    # Compare independent float64 arithmetic with the source float32 SGD
    # multiply-add; bitwise equality would incorrectly require the same rounding.
    expected=(before.double()-.002*gradient.double()).float()
    torch.testing.assert_close(m.prompt_learner.ctx.detach(),expected,rtol=3e-7,atol=2e-9)
    assert pred.shape==(1,3)
    # Evaluation recomputes the freshly updated class text; an old cache would
    # retain the exact pre-update logits and invalidate subsequent evaluation.
    with torch.no_grad():assert not torch.equal(m(x)[0],pred[0])


def test_native_tokenizer_prompt_buffers_and_eval_path_are_device_portable():
    m=model();tokens=tokenize_function()(['synthetic class 0'])
    assert tokens.shape==(1,77) and tokens[0,0]==49406 and tokens.max()==49407
    assert 'tokenized_prompts' in dict(m.prompt_learner.named_buffers())
    x=torch.randn(11,512)
    with torch.no_grad():
        a=m(x);b=m(x,eval=True)
    assert all(torch.equal(left,right) for left,right in zip(a,b))


@pytest.mark.parametrize('shape',[(0,512),(2,16),(2,3,512)])
def test_reject_incompatible_native_feature_inputs(shape):
    m=model()
    with pytest.raises(ValueError,match='native CLIP feature space'):m(torch.empty(shape))


def test_reject_nonfinite_features_and_out_of_range_labels():
    m=model()
    with pytest.raises(ValueError):m(torch.full((2,512),float('nan')))
    with pytest.raises(ValueError):m.loss(torch.randn(2,512),3)


@pytest.mark.parametrize('kwargs',[{'context_size':0},{'temperature':0},{'context_gain':float('nan')}])
def test_reject_invalid_prompt_configuration(kwargs):
    with pytest.raises(ValueError):SLIPReference('clip_vit_b16',['{}'],[['a'],['b']],[['t']],**kwargs)

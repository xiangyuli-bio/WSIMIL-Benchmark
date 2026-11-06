"""Original DisPro likelihood vs independent equation/log-domain gradients."""
import inspect

import pytest
import torch

from milplus.discrete_survival import discrete_nll
from milplus.dispro_objectives import check_hazards,predict_multimodal,predict_unimodal,training_target
from milplus.dispro_source import survival_loss


@pytest.mark.parametrize('event',[0,1])
@pytest.mark.parametrize('interval',[0,1,2,3])
def test_actual_author_likelihood_all_four_bins_matches_primary_equation_and_stable_gradient(event,interval):
    a=torch.tensor([[-1.1,.2,.8,-.4]],dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_();c=a.detach().clone().requires_grad_()
    hazards=a.sigmoid();survival=(1-hazards).cumprod(1)
    censor,y=training_target(event,interval,'cpu')
    author=survival_loss()(hazards=hazards,S=survival,Y=y,c=censor)
    h=b.sigmoid()
    equation=-(h[0,interval].log()+(1-h[0,:interval]).log().sum()) if event else -(1-h[0,:interval+1]).log().sum()
    stable=discrete_nll(c,torch.tensor([interval]),torch.tensor([event]))
    torch.testing.assert_close(author,equation,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(author,stable,rtol=1e-14,atol=1e-14)
    author.backward();equation.backward();stable.backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(a.grad,c.grad,rtol=1e-13,atol=1e-14)
    check_hazards(hazards,survival)


@pytest.mark.parametrize('event,interval',[(2,1),(-1,0),(0,-1),(1,4),(1,1.5),(float('nan'),1),([0,1],0)])
def test_invalid_event_or_unfitted_interval_is_rejected_before_forward(event,interval):
    with pytest.raises(ValueError,match='genuine TRAIN-fitted'):training_target(event,interval,'cpu')


def test_outcomes_are_not_accepted_by_public_inference_signatures():
    assert list(inspect.signature(predict_unimodal).parameters)==['model','features']
    assert list(inspect.signature(predict_multimodal).parameters)==['model','inputs']
    with pytest.raises(TypeError):predict_unimodal(None,None,event=1)
    with pytest.raises(TypeError):predict_multimodal(None,{},interval=2)


@pytest.mark.parametrize('kind',['wrong_bins','nonfinite','wrong_cumulative_product'])
def test_invalid_auxiliary_hazards_cannot_silently_enter_joint_distillation(kind):
    h=torch.tensor([[.1,.3,.5,.2]]);s=(1-h).cumprod(1)
    if kind=='wrong_bins':h=h[:,:3];s=s[:,:3]
    elif kind=='nonfinite':h[0,0]=float('nan')
    else:s[0,2]=.9
    with pytest.raises(ValueError,match='DisPro'):check_hazards(h,s)

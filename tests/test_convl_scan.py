"""All four survival derivatives, including zeros; original CPU route unchanged."""
import pytest
import torch

from milplus.convl_scan import FourBinCumprod, survival_cumprod


@pytest.mark.parametrize('values',[[.2,.4,.6,.8],[0.,.4,.6,.8],[.2,0.,.6,.8],
    [.2,.4,0.,.8],[.2,.4,.6,0.],[0.,0.,.6,.8],[.2,0.,.6,0.],[0.,0.,0.,0.]])
def test_complete_closed_form_preserves_native_forward_and_each_bin_derivative(values):
    original=torch.tensor([values],dtype=torch.float64,requires_grad=True)
    port=original.detach().clone().requires_grad_();weight=torch.tensor([[.7,-.2,.5,1.3]],dtype=torch.float64)
    expected=torch.cumprod(original,1);actual=FourBinCumprod.apply(port)
    assert torch.equal(expected,actual)
    (expected*weight).sum().backward();(actual*weight).sum().backward()
    torch.testing.assert_close(port.grad,original.grad,rtol=1e-14,atol=1e-14)
    assert torch.isfinite(port.grad).all()


def test_cpu_source_route_keeps_exact_original_autograd():
    x=torch.tensor([[.3,.2,.4,.8]],requires_grad=True);y=x.detach().clone().requires_grad_()
    a=survival_cumprod(x);b=torch.cumprod(y,1)
    a.square().sum().backward();b.square().sum().backward()
    assert torch.equal(a,b) and torch.equal(x.grad,y.grad)


def test_full_four_bin_analytic_jacobian_passes_double_gradcheck():
    x=torch.tensor([[.2,.4,.6,.8],[0.,.7,.4,.2]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(FourBinCumprod.apply,(x,),eps=1e-6,atol=1e-6,rtol=1e-5)

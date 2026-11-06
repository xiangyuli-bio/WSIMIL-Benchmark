import pytest
import torch

from milplus.icmil_coupled_pool import SingleAdaptiveMax


@pytest.mark.parametrize('kind',['random','ties','zero','negative'])
def test_exact_native_K1_values_tie_selection_and_every_input_gradient(kind):
    torch.manual_seed(4)
    x=torch.randn(7,17,dtype=torch.float64)
    if kind=='ties':x[:,3]=5;x[:,8]=5
    elif kind=='zero':x.zero_()
    elif kind=='negative':x=-torch.arange(119,dtype=torch.float64).reshape(7,17)
    a=x.clone().requires_grad_();b=x.clone().requires_grad_();weight=torch.randn(7,1,dtype=torch.float64)
    expected=torch.nn.functional.adaptive_max_pool1d(a,1);actual=SingleAdaptiveMax.apply(b)
    assert torch.equal(actual,expected)
    expected.backward(weight);actual.backward(weight)
    assert torch.equal(a.grad,b.grad)


def test_K1_gradient_and_double_gradient_match_the_original_derivative():
    x=torch.randn(3,7,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(SingleAdaptiveMax.apply,(x,))
    assert torch.autograd.gradgradcheck(SingleAdaptiveMax.apply,(x,))

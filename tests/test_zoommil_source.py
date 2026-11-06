"""Actual original full graph references; not a completed trainer admission."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.upstream import load_definitions
from milplus.zoommil_source import build_model


@pytest.mark.parametrize('dimension', [1024, 1536])
def test_zoommil_actual_threelevel_source_logits_loss_and_all_gradients(dimension):
    torch.set_num_threads(1)
    topk = load_definitions('zoommil', 'zoommil/utils/perturbedtopk.py',
        ['PerturbedTopK', 'PerturbedTopKFunction'], {'torch': torch, 'nn': nn, 'F': F})
    source = load_definitions('zoommil', 'zoommil/models/zoommil.py',
        ['GatedAttention', 'ZoomMIL'], {'torch': torch, 'nn': nn, 'F': F, 'PerturbedTopK': topk.PerturbedTopK})
    original = source.ZoomMIL(dimension, dropout=.25, k_sample=12, k_sigma=.002, n_cls=3)
    # Use the device the actual source chooses; no CUDA monkeypatch or smaller
    # reference network. Native1024 plus a distinct full1536 FM input.
    device = original.device
    original.to(device)
    adapted = build_model(dimension, 3).to(device)
    adapted.load_state_dict(original.state_dict(), strict=True)
    # NativeBRIGHT hierarchy1.25x->2.5x->10x, full16/64/1024 rows.
    # Coarse16>K12 ensures both selectors perform actual subset selection.
    x = [torch.randn(1, n, dimension, device=device, requires_grad=True) for n in [16, 64, 1024]]
    y = [value.detach().clone().requires_grad_() for value in x]
    torch.manual_seed(96)
    expected = original(x)
    expected_rng = torch.get_rng_state()
    expected_cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
    torch.manual_seed(96)
    actual = adapted(y)
    assert torch.equal(expected_rng, torch.get_rng_state())
    if device.type == 'cuda':
        assert all(torch.equal(a, b) for a, b in zip(expected_cuda_rng, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    a = F.cross_entropy(expected[0], torch.tensor([1], device=device))
    b = F.cross_entropy(actual[0], torch.tensor([1], device=device))
    assert torch.equal(a, b)
    a.backward(); b.backward()
    assert all(torch.equal(i.grad, j.grad) for i, j in zip(x, y))
    parameters = dict(adapted.named_parameters())
    for name, value in original.named_parameters():
        assert value.grad is not None
        assert torch.equal(value.grad, parameters[name].grad), name
    # Both zoom-selection attentions really receive upstream classification
    # gradient through the original MonteCarlo top-K backward.
    assert original.aux_ga_low_mag.attention_c.weight.grad.abs().sum() > 0
    assert original.aux_ga_mid_mag.attention_c.weight.grad.abs().sum() > 0
    original.eval(); adapted.eval()
    torch.manual_seed(101); before = torch.get_rng_state()
    expected = original(x); actual = adapted(y)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(before, torch.get_rng_state())  # hardTopK deterministic

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus.mgtrans_source import build_model
from milplus.upstream import source_root


@pytest.fixture(autouse=True)
def runtime():
    if not source_root('mgtrans').is_dir():
        pytest.skip('Fetch the pinned MG-Trans source')
    torch.set_num_threads(1)


def native_inputs(dimension):
    coords = torch.tensor([[x*2048, y*2048] for y in range(16) for x in range(32)])
    fine = torch.stack([point+torch.tensor(offset) for point in coords
                        for offset in [(0, 0), (1024, 0), (0, 1024), (1024, 1024)]])
    mask = torch.zeros(512, 2048, dtype=torch.bool)
    for i in range(512):
        mask[i, 4*i:4*i+4] = True
    return torch.randn(512, dimension, requires_grad=True), coords, torch.randn(2048, dimension, requires_grad=True), fine, mask


@pytest.mark.parametrize('dimension', [1024, 1536])
def test_full_native_multiscale_graph_and_semantic_objective_backpropagate(dimension):
    model = build_model(dimension, 3)
    x, coords, y, fine, mask = native_inputs(dimension)
    observed = []
    def graph_input(module, args):
        graph = args[0]
        assert graph.x.requires_grad is False  # Deliberate released detach, retained explicitly.
        delta = graph.centroid[graph.edge_index[0]]-graph.centroid[graph.edge_index[1]]
        assert (delta.square().sum(1).sqrt() <= 2048).all()
        assert graph.x.shape[1] == 192
        observed.append(len(graph.x))
    hook = model.transformer.encoder.layer2.gcn.register_forward_pre_hook(graph_input)
    target = torch.tensor([1])
    output = model(x, coords, y, fine, mask, target)
    hook.remove()
    assert output[0].shape == (1, 3) and observed == [512, 2048]
    delta = torch.norm(model.head_high(output[8])-model.head_low(output[7]), p=1, dim=0).unsqueeze(0)
    expected = F.cross_entropy(output[0], target)-.0001*F.cross_entropy(delta, target)
    torch.testing.assert_close(output[4], expected, atol=0, rtol=0)
    output[4].backward()
    assert x.grad.isfinite().all() and y.grad.isfinite().all()
    assert x.grad.abs().sum() > 0 and y.grad.abs().sum() > 0
    assert model.head_low.weight.grad.abs().sum() > 0 and model.head_high.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.transformer.encoder.layer2.gcn.parameters())


def test_native_inference_has_no_label_dependency_and_graph_replay_is_exact():
    model = build_model(1024, 3).eval()
    x, coords, y, fine, mask = native_inputs(1024)
    with torch.no_grad():
        a = model(x, coords, y, fine, mask, torch.tensor([0]))
        b = model(x, coords, y, fine, mask, torch.tensor([2]))
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
    assert float(a[4]) != float(b[4])


def test_repeated_short_bag_graph_rng_and_all_parameter_gradients_replay_exactly():
    a = build_model(1024, 3)
    b = copy.deepcopy(a)
    x, coords, y, fine, mask = native_inputs(1024)
    x, coords, y, fine, mask = x[:16], coords[:16], y[:64], fine[:64], mask[:16, :64]
    state, numpy_state = torch.get_rng_state(), np.random.get_state()
    left = a(x, coords, y, fine, mask, torch.tensor([1]))
    torch.set_rng_state(state); np.random.set_state(numpy_state)
    right = b(x, coords, y, fine, mask, torch.tensor([1]))
    torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)
    left[4].backward(); right[4].backward()
    for p, q in zip(a.parameters(), b.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

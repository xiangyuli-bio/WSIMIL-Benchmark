"""Sparse KL product: independent scatter oracle and real CUDA regression sizes."""
import pytest
import torch

from milplus.probsa import graph_product, sparse_pool_class


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires an actual CUDA sparse backend'))])
@pytest.mark.parametrize('n', [3, 128, 2691, 16848])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sparse_product_outputs_gradients_and_rng_equal_independent_scatter(device, n, dtype):
    torch.manual_seed(390)
    batch = torch.arange(4).repeat_interleave(n)
    row = torch.arange(n).repeat(4)
    diagonal = torch.stack([batch, row, row])
    neighbor = torch.stack([batch, row, (row + 1) % n])
    indices = torch.cat([diagonal, neighbor], 1).to(device)
    weights = torch.cat([torch.full((4*n,), .25), torch.full((4*n,), .75)]).to(device, dtype)
    # Graph weights are constants in ProbSA. Exercise their derivative only
    # on small cases: Torch's sparse-value backward may materialize B*N*N.
    weights.requires_grad_(n <= 128)
    adjacency = torch.sparse_coo_tensor(indices, weights, (4, n, n)).coalesce()
    values = torch.randn(4, n, 1, device=device, dtype=dtype, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_(n <= 128)
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    output = graph_product(adjacency, values)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if device == 'cuda':
        torch.cuda.synchronize()
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    b, r, c = indices
    source = reference_weights[:, None] * reference_values[b, c]
    expected = torch.zeros(4*n, 1, device=device, dtype=dtype).index_add(0, b*n+r, source).reshape(4, n, 1)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
    differentiated = (values, weights) if n <= 128 else (values,)
    reference_differentiated = (reference_values, reference_weights) if n <= 128 else (reference_values,)
    actual_grads = torch.autograd.grad(output.square().sum(), differentiated)
    expected_grads = torch.autograd.grad(expected.square().sum(), reference_differentiated)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=3e-6, atol=2e-6)
    assert output.shape == (4, n, 1) and adjacency._nnz() == 8*n


def test_dense_and_cpu_paths_preserve_existing_bmm_exactly():
    torch.manual_seed(9)
    a = torch.randn(2, 7, 7, dtype=torch.float64)
    v = torch.randn(2, 7, 1, dtype=torch.float64)
    assert torch.equal(graph_product(a, v), torch.bmm(a, v))
    assert torch.equal(graph_product(a.to_sparse(), v), torch.bmm(a.to_sparse(), v))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires actual CUDA sparse KL')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_ragged_sparse_kl_loss_and_both_gradients_match_scatter(dtype):
    lengths = [2691, 2687, 2690, 2688]
    n = max(lengths)
    triples = []
    weights = []
    for batch, length in enumerate(lengths):
        for row in range(length):
            triples.extend([(batch, row, row), (batch, row, (row+1) % length)])
            weights.extend([.25, .75])
    indices = torch.tensor(triples, device='cuda').T
    values = torch.tensor(weights, device='cuda', dtype=dtype)
    adj = torch.sparse_coo_tensor(indices, values, (4, n, n)).coalesce()
    mask = torch.arange(n, device='cuda')[None] < torch.tensor(lengths, device='cuda')[:, None]
    pool = sparse_pool_class(torch.nn.Module)()
    pool.valid_mask = mask
    torch.manual_seed(113)
    mu = torch.randn(4, n, 1, device='cuda', dtype=dtype, requires_grad=True)
    logv = torch.randn_like(mu, requires_grad=True)
    actual = pool._kl_div(mu, logv, adj)
    terms = []
    for batch, length in enumerate(lengths):
        m, v = mu[batch, :length, 0], logv[batch, :length, 0]
        product = .25*m + .75*m.roll(-1)
        terms.append((m.square().sum()-(m*product).sum()+.75*v.exp().sum()-.5*v.sum())/length)
    expected = torch.stack(terms).mean()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual_grads = torch.autograd.grad(actual, (mu, logv), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (mu, logv))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-6)
        assert not a[~mask].any()

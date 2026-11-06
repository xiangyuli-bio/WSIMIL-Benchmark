"""Independent geometry/equation tests; CPU oracles are not native-GPU claims."""
import math

import numpy as np
import pytest
import torch

from milplus.longmil_source import (PaperLongMIL, aligned_bias, chunked_attention,
    dense_oracle, grid_positions, local_plan, public_definitions, spatial_pool2)


def equation(q, k, v, positions, local):
    # Separate formulation, full attention only on deliberately small test bags.
    scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
    scores = scores / math.sqrt(q.shape[-1])
    distances = torch.linalg.vector_norm(positions[:, None].float() - positions[None, :].float(), dim=2)
    if local:
        scores = scores.masked_fill(distances[None, None] > 10, -torch.inf)
    else:
        slopes = 2. ** (-8 * torch.arange(1, q.shape[2] + 1).float() / q.shape[2])
        scores = scores - slopes[None, :, None, None] * distances[None, None]
    return (scores.softmax(-1) @ v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('chunk', [1, 4, 32])
def test_sparse_chunks_match_dense_equation_and_every_gradient(local, chunk):
    torch.manual_seed(104)
    positions = torch.tensor([[0, 0], [1, 1], [6, 8], [0, 10], [0, 11],
                              [31, 1], [32, 1], [90, 20], [91, 21]])
    actual = [torch.randn(1, 9, 1, 24, requires_grad=True) for _ in range(3)]
    expected = [v.detach().clone().requires_grad_() for v in actual]
    output = chunked_attention(*actual, positions, local=local, chunk=chunk, operator=dense_oracle)
    reference = equation(*expected, positions, local)
    torch.testing.assert_close(output, reference, rtol=2e-6, atol=2e-6)
    weight = torch.randn_like(output)
    (output * weight).sum().backward()
    (reference * weight).sum().backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-6, atol=3e-6)


def test_actual_two_dimensional_neighbours_include_boundary_and_distant_sequence_rows():
    positions = torch.tensor([[0, 0], [0, 50], [6, 8], [0, 11], [0, 1]])
    plan = local_plan(positions, chunk=1)
    assert plan[0][2].tolist() == [0, 2, 4]
    assert plan[1][2].tolist() == [1]
    assert not torch.equal(plan[0][2], torch.arange(3))


def test_physical_grid_uses_declared_step_and_origin():
    assert torch.equal(grid_positions(torch.tensor([[12, 24], [460, 472]]), 448, (12, 24)),
                       torch.tensor([[0, 0], [1, 1]]))


@pytest.mark.parametrize('coords,step', [([[0., 0.], [1., 2.]], 448),
    ([[0., 0.], [0., 0.]], 448), ([[-448., 0.]], 448), ([[0., float('nan')]], 448),
    ([[0., 0.]], 0)])
def test_invalid_or_invented_grid_rejected(coords, step):
    with pytest.raises(ValueError):
        grid_positions(torch.tensor(coords), step)


def test_real_spatial_pool_averages_only_observed_tissue_and_preserves_gradients():
    positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [3, 1], [40, 40]])
    values = torch.arange(21, dtype=torch.float32).reshape(1, 7, 3).requires_grad_()
    pooled, centres, groups = spatial_pool2(values, positions)
    assert groups == [[0, 1, 2, 3], [4, 5], [6]]
    torch.testing.assert_close(pooled[0], torch.stack([values[0, g].mean(0) for g in groups]), rtol=0, atol=0)
    assert torch.equal(centres, torch.tensor([[.5, .5], [2.5, .5], [40., 40.]]))
    pooled.sum().backward()
    for group in groups:
        assert torch.equal(values.grad[0, group], torch.full((len(group), 3), 1 / len(group)))


def test_pool_preserves_geometry_under_input_permutation_and_far_extrapolation():
    positions = torch.tensor([[602, 500], [603, 501], [1000, 1000], [1, 0]])
    values = torch.randn(1, 4, 384)
    order = torch.tensor([2, 1, 3, 0])
    a, x, _ = spatial_pool2(values, positions)
    b, y, _ = spatial_pool2(values[:, order], positions[order])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(x, y)


@pytest.mark.parametrize('size', [1, 8, 9, 17])
def test_xformers_bias_actual_key_stride_is_aligned_without_extra_keys(size):
    value = torch.randn(1, 4, 3, size)
    output = aligned_bias(value)
    assert output.shape == value.shape and output.stride(-2) % 8 == 0
    assert torch.equal(value, output)


def test_full_paper_network_retains_two_local_one_global_384_1head_and_real_pool():
    torch.manual_seed(2026)
    model = PaperLongMIL(1024, 3, chunk=4, operator=dense_oracle)
    positions = torch.cartesian_prod(torch.arange(5), torch.arange(5))
    features = torch.randn(25, 1024)
    result = model(features, positions)
    assert result['logits'].shape == (1, 3)
    assert model.last_geometry['input_patches'] == 25
    assert model.last_geometry['pooled_cells'] == 9
    assert [model.local1.attn.local, model.local2.attn.local, model.global_block.attn.local] == [True, True, False]
    assert model.local1.attn.qkv.weight.shape == (1152, 384)
    assert model.local2.mlp[0].weight.shape == (1536, 384)
    F = torch.nn.functional
    F.cross_entropy(result['logits'], torch.tensor([2])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_paper_permutation_equivariance_keeps_actual_spatial_relationships():
    torch.manual_seed(104)
    model = PaperLongMIL(32, 2, chunk=5, operator=dense_oracle).eval()
    positions = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    features = torch.randn(16, 32)
    order = torch.randperm(16)
    a = model(features, positions)['logits']
    b = model(features[order], positions[order])['logits']
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_public_v1_preserves_original_cross_head_minimum_mask_and_full_width():
    # Small synthetic table is only a positional operator test, never a full
    # runnable method or a substitute for the actual8.1GB native asset.
    table = -(torch.arange(16)[:, None] - torch.arange(16)[None, :]).abs().to(torch.int8)
    source = public_definitions(lambda: table, dense_oracle)
    model = source.LongMIL(2, input_size=32)
    coords = torch.stack([torch.zeros(11), torch.arange(11) * 896], 1)
    value = torch.cat([torch.randn(11, 32), coords], 1)
    _, local, global_bias = model.positional_embedding(value, use_alibi=True)
    raw = table[:11, :11, None] * source.get_slopes(4).to(torch.float16)[None, None]
    expected = raw.permute(2, 0, 1).unsqueeze(0).float()
    torch.testing.assert_close(global_bias, expected, rtol=0, atol=0)
    expected_local = expected.clone()
    expected_local[expected_local == expected_local.min()] = -torch.inf
    assert torch.equal(local, expected_local)
    assert all(block.attn.num_heads == 4 for block in [model.layer1, model.layer2, model.layer3])
    assert model.layer1.mlp.fc1.weight.shape == (1536, 384)


def test_original_attention_kernel_error_is_rejected_not_silently_identity():
    def unavailable(*args, **kwargs):
        raise NotImplementedError('test native kernel unavailable')
    source = public_definitions(lambda: torch.zeros(2, 2, dtype=torch.int8), unavailable)
    layer = source.Attention(384)
    with pytest.raises(RuntimeError, match='no silent identity fallback'):
        layer(torch.randn(1, 2, 384), alibi=torch.zeros(1, 4, 2, 2))

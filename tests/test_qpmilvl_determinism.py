"""Duplicate-index accumulation must be exact at original CPU pool dimensions."""
import pytest
import torch

from milplus import qpmilvl_source as source
from milplus.qpmilvl_driver import train


@pytest.mark.parametrize('shape',[(20,512),(20,24,768)])
def test_original_pool_duplicate_index_gradients_repeat_bitwise_without_reducing_threads(shape):
    previous=(torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(2026)
    values=torch.randn(shape,generator=generator)
    indices=torch.tensor([6,11,18,3,17]).repeat(16,1)
    upstream=torch.randn(tuple(indices.shape)+shape[1:],generator=generator)
    try:
        torch.use_deterministic_algorithms(False,warn_only=True)
        with source.reproducible_execution():
            actual=[]
            for _ in range(20):
                parameter=values.clone().requires_grad_(True)
                parameter[indices].backward(upstream)
                actual.append(parameter.grad.clone())
            assert torch.get_num_threads()==2
            assert all(torch.equal(actual[0],value) for value in actual[1:])
            runtime=source.runtime_identity()
            assert runtime['precision']=='original_float32'
            assert runtime['numerical_execution']['deterministic_algorithms']
            assert not runtime['numerical_execution']['warn_only']
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])


def test_numerical_execution_scope_restores_both_flags_after_an_exception():
    previous=(torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(False,warn_only=True)
        with pytest.raises(RuntimeError,match='intentional control'):
            with source.reproducible_execution():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError('intentional control')
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])


def test_public_train_invalid_input_restores_callers_execution_and_keeps_output_absent(tmp_path):
    previous=(torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    output=tmp_path/'not_created'
    try:
        torch.use_deterministic_algorithms(False,warn_only=True)
        with pytest.raises(FileNotFoundError):
            train(tmp_path/'missing_task_bundle',output,config={'epochs':2,'threads':2})
        assert not output.exists()
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])

"""Checkpoint math-branch binding, without manufacturing pretrained assets."""
from types import SimpleNamespace

import pytest
import torch

from milplus.feather_runtime import effective_backend_identity, restore, save
from milplus.feather_source import architecture_classes, checked_source, original_configuration
from milplus.feather_training import new_progress, reference_recipe, setup


def test_actual_backend_identity_records_threads_isa_precision_and_determinism():
    value = effective_backend_identity()
    assert value['intraop_threads'] == torch.get_num_threads()
    assert value['interop_threads'] == torch.get_num_interop_threads()
    assert value['cpu_capability'] == torch.backends.cpu.get_cpu_capability()
    assert value['float32_matmul_precision'] == torch.get_float32_matmul_precision()
    assert value['deterministic_algorithms'] == torch.are_deterministic_algorithms_enabled()
    assert {'cuda_matmul_allow_tf32', 'cuda_allow_fp16_reduced_precision_reduction',
            'cuda_allow_bf16_reduced_precision_reduction'} <= value.keys()


def test_complete_checkpoint_rejects_backend_drift_before_loading_any_weights(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    model = architecture_classes().ABMIL(**original_configuration('conch_v15', 2))
    model._milplus_feather_evidence = {'scope': 'unpretrained_full_original_architecture_IO_component'}
    inputs = SimpleNamespace(contract='component_only_no_pretrained_asset', verify=checked_source)
    recipe = reference_recipe('finetune_no_val10')
    opt, sched, scaler, generator = setup(model, device='cpu', seed=23, recipe=recipe)
    request = {'scope': 'backend_state_IO_component_not_full_FEATHER'}
    path = tmp_path / 'checkpoint.pt'
    state = new_progress()
    save(path, model, opt, sched, scaler, generator, state, request, inputs)
    payload = torch.load(path, weights_only=False)
    assert payload['schema_version'] == 2
    assert payload['effective_backend'] == effective_backend_identity()
    assert restore(path, model, opt, sched, scaler, generator, request, inputs) == state
    old = torch.backends.cuda.matmul.allow_tf32
    before = model.patch_embed[0].weight.detach().clone()
    try:
        torch.backends.cuda.matmul.allow_tf32 = not old
        with pytest.raises(ValueError, match='recovery changed'):
            restore(path, model, opt, sched, scaler, generator, request, inputs)
        assert torch.equal(model.patch_embed[0].weight, before)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old

"""Full native model and original recipe; no reduced text or feature width."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from milplus.io import digest_file, read_json
from milplus.libramil_source import (COMMIT, build_original, encode_original_prompts,
                                   original_model, original_prompts)
from milplus.libramil_training import SourceSelection, original_training, source_optimizer
from milplus.upstream import verified_source

ASSETS = Path('/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/libramil_rcc_r1')


def actual_text():
    receipt = read_json(ASSETS/'receipt.json')
    assert digest_file(ASSETS/'embeddings.pt') == receipt['embeddings_sha256']
    values = torch.load(ASSETS/'embeddings.pt', weights_only=True)
    return values['structural'], values['bags']


def test_real_full_conch_prompt_encoding_matches_actual_precomputed_frozen_assets():
    torch.set_num_threads(2)
    weights = Path('/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth')
    a, b, paths = encode_original_prompts(weights)
    expected_a, expected_b = actual_text()
    assert torch.equal(a, expected_a) and torch.equal(b, expected_b)
    assert a.shape == (46, 512) and b.shape == (3, 512)
    receipt = read_json(ASSETS/'receipt.json')
    assert receipt['original_text_sha256'] == {str(p): digest_file(p) for p in paths}
    assert digest_file(weights) == receipt['checkpoint_sha256']


@pytest.mark.parametrize('prototypes,parameters', [(4, 1869315), (6, 1870339)])
def test_full512_46text3classes257patches8heads_original_SOT_and_gradients(prototypes, parameters):
    torch.set_num_threads(1); torch.manual_seed(42)
    root, revision = verified_source('libramil'); assert revision == COMMIT
    a, b = actual_text(); model = build_original(a, b, prototypes)
    assert model.num_heads == 8 and model.head_dim == 64
    assert model.proto_vis.shape == (prototypes, 512)
    assert model.prompt_inst.shape == (46, 512) and model.prompt_bag.shape == (3, 512)
    assert sum(p.numel() for p in model.parameters()) == parameters
    x = torch.randn(1, 257, 512, requires_grad=True)
    result = model(x, torch.tensor([1])); result['loss'].backward()
    assert result['logits'].shape == (1, 3)
    assert result['transport_plan'].shape == (1, 46, prototypes)
    assert result['patch_attention'].shape == (1, 257)
    assert torch.isfinite(result['transport_plan']).all()
    torch.testing.assert_close(result['patch_attention'].sum(1), torch.ones(1))
    assert x.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name in {'temp_struct', 'temp_vis'}:
            assert parameter.grad is None  # Allocated but unused by source.
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.ot_iter == 20 and model.ot_epsilon == .05


def test_original_complete_ordered_Rcc_language_assets():
    structs, bags, _ = original_prompts()
    assert len(structs) == 46 and len(bags) == 3
    assert all('high resolution' in x for x in bags)
    assert ['chromophobe' in bags[0], 'clear cell' in bags[1], 'papillary' in bags[2]] == [True]*3


def equal_state(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_state(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_state(x, y)
    else:
        assert a == b


def test_twoepoch_full_original_train_function_matches_weights_AdamW_cosine_and_RNG():
    torch.set_num_threads(1); torch.manual_seed(2026)
    a, b = actual_text(); first = build_original(a, b)
    second = copy.deepcopy(first)
    # Native512 full257patches, all three classes; class labels are synthetic.
    x, y = torch.randn(6, 257, 512), torch.arange(6).remainder(3)
    dataset = TensorDataset(x, y)
    opt_a, sched_a = source_optimizer(first, 6, epochs=20)
    opt_b, sched_b = source_optimizer(second, 6, epochs=20)
    start = torch.get_rng_state()
    source = original_training()
    for epoch in range(2):
        source.train(first, DataLoader(dataset, batch_size=1, shuffle=True),
                     torch.device('cpu'), epoch, opt_a, sched_a, 3)
    end = torch.get_rng_state()
    torch.set_rng_state(start)
    for _ in range(2):
        second.train()
        for features, labels in DataLoader(dataset, batch_size=1, shuffle=True):
            result = second(features, labels)
            opt_b.zero_grad(); result['loss'].backward(); opt_b.step(); sched_b.step()
    assert torch.equal(end, torch.get_rng_state())
    equal_state(first.state_dict(), second.state_dict())
    equal_state(opt_a.state_dict(), opt_b.state_dict())
    equal_state(sched_a.state_dict(), sched_b.state_dict())
    assert sched_a.last_epoch == 12
    assert opt_a.param_groups[0]['weight_decay'] == .01
    assert opt_a.param_groups[0]['lr'] == pytest.approx(.0001*12/100)


def test_original_stale_AUC_selection_branch_is_explicitly_preserved_and_restored():
    selection = SourceSelection()
    # Source best_auc remains0 on a strict accuracy increase. Hence a lower
    # AUC on the following accuracy tie can replace the preceding checkpoint.
    assert selection.observe(0, .6, .9)
    assert selection.best_auc == 0
    assert selection.observe(1, .6, .7)
    assert selection.selected_epoch == 1
    assert not selection.observe(2, .6, .65)
    restored = SourceSelection(); restored.load_state_dict(selection.state_dict())
    assert restored.state_dict() == selection.state_dict()
    bad = selection.state_dict(); bad['best_auc'] = .9
    with pytest.raises(ValueError, match='inconsistent'):
        restored.load_state_dict(bad)


@pytest.mark.parametrize('dimension', [128, 768])
def test_native_alignment_does_not_silently_project_other_FM_widths(dimension):
    with pytest.raises(ValueError, match='original512D'):
        build_original(torch.randn(46, dimension), torch.randn(3, dimension))

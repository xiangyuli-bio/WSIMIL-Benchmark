"""Actual full original ConVL mechanism checks; trainer admission is separate."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from milplus.convl_source import (COMMIT, FORK_ROOT, NAMES, author_classes,
                                 build_original, checked_source, original_concepts)
from milplus.focus_network import CONCH_COMMIT
from milplus.upstream import load_definitions, load_single, pinned_text, verified_source

WEIGHTS = Path('/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth')


@pytest.fixture(scope='module')
def definitions():
    torch.set_num_threads(1)
    return author_classes(WEIGHTS)


def test_exact_original_source_and_full_text_architecture(definitions):
    root, commit = checked_source()
    assert commit == COMMIT and all(hasattr(definitions, name) for name in NAMES)
    original, sha = verified_source('conch_focus')
    assert sha == CONCH_COMMIT
    for name in ['transformer.py', 'model_configs/conch_ViT-B-16.json', 'custom_tokenizer.py']:
        assert pinned_text(root, commit, FORK_ROOT+name) == pinned_text(original, sha, 'conch/open_clip_custom/'+name)


def test_original_alignment_is_one_direction_sigmoid_kl_not_symmetric_hazard_kl(definitions):
    a = torch.tensor([[.2, -.8, 1.3, 2.1]], requires_grad=True)
    b = torch.tensor([[-1.1, .5, .9, -.2]], requires_grad=True)
    expected = F.kl_div((a.sigmoid()+1e-8).log(), b.sigmoid(), reduction='batchmean')
    actual = definitions.KL_Divergence_Loss()(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, definitions.KL_Divergence_Loss()(b, a))
    actual.backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_source_named_sinkhorn_is_literal_softmax_euclidean_control(definitions):
    a, b = torch.randn(1, 4), torch.randn(1, 4)
    expected = torch.dist(a.softmax(1), b.softmax(1), p=2)
    torch.testing.assert_close(definitions.SinkhornLoss()(a, b), expected, rtol=0, atol=0)


def test_full_native_cross_attention_and_source_temperature_equations(definitions):
    torch.manual_seed(33)
    model = definitions.CrossAttentionAdapter(L=256, n_concept=11, n_classes=4, temperature=.1)
    assert model.mha.num_heads == 4
    concepts, patches = torch.randn(11, 256), torch.randn(17, 256)
    context, raw_attention = model.mha(concepts[None], patches[None], patches[None])
    expected_concepts = model.layer_norm(concepts[None]+context)
    expected_attention = F.softmax(torch.log(raw_attention.squeeze(0).T+1e-8)/.1, dim=1)
    actual_concepts, actual_attention, logits = model(concepts, patches)
    torch.testing.assert_close(actual_concepts, expected_concepts, rtol=0, atol=0)
    torch.testing.assert_close(actual_attention, expected_attention, rtol=0, atol=0)
    torch.testing.assert_close(logits, model.attention_c(expected_attention), rtol=0, atol=0)


def test_original_clustering_both_max_axes_and_negative_weights(definitions):
    attention = torch.tensor([[.8, .1, .1], [.2, .6, .2]])
    a, b = definitions.ConVL.compute_clustering_loss(None, attention, torch.zeros(2, 256))
    torch.testing.assert_close(a, attention.max(1).values.mean(), rtol=0, atol=0)
    torch.testing.assert_close(b, attention.max(0).values.mean(), rtol=0, atol=0)
    assert float(-.5*a-.3*b) < 0


@pytest.mark.parametrize('mode', ['mean', 'sum', 'max', 'attention_weighted', 'topk_mean'])
def test_all_original_concept_aggregation_modes(definitions, mode):
    attention = torch.rand(17, 11)
    expected = {'mean': attention.mean(0), 'sum': attention.sum(0)/17,
        'max': attention.max(0).values, 'attention_weighted': attention.softmax(1).mean(0),
        'topk_mean': attention.topk(5, dim=0).values.mean(0)}[mode]
    actual = definitions.ConVL.aggregate_concepts(SimpleNamespace(concept_agg_mode=mode), attention)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all17_original_cohorts_keep_full_concepts_and_native_token_bounds(definitions):
    import json
    root, _ = checked_source()
    original = json.loads((root/'models/text_prompt/concepts_patch.json').read_text())
    tokenizer = load_single('convl', FORK_ROOT+'custom_tokenizer.py')
    for key, record in original.items():
        path, mapped = original_concepts('COADREAD' if key == 'CO&RE' else key)
        assert mapped == key
        strings = definitions.ConVL._load_concepts_from_json(None, str(path), mapped)
        assert len(strings) == len(record['phenotypes'])
        prompts = [' '.join(['X']*16)+' '+name.replace('_', ' ') for name in strings]
        tokens = tokenizer.tokenize(tokenizer.get_tokenizer(), prompts)
        assert tokens.shape == (len(strings), 128)
        assert (tokens.argmax(-1) < 127).all(), key
    assert len(original) == 17


def test_complete_actual_pretrained12layer_dynamic_text_CGA_and_three_losses_backward(definitions):
    torch.manual_seed(2024)
    model = build_original(WEIGHTS, 'BLCA').train()
    assert len(model.text_encoder.transformer.resblocks) == 12
    assert model.prompt_learner.ctx.shape == (16, 768)
    assert model.prompt_learner.token_prefix.shape == (11, 1, 768)
    assert model.prompt_learner.token_suffix.shape == (11, 110, 768)
    assert model.text_encoder.text_projection.shape == (768, 512)
    assert model.fc_concept[0].weight.shape == (256, 512)
    assert model.fc_patch[0].weight.shape == (256, 768)
    assert model.attention_net.mha.num_heads == 4
    outputs = model(data=torch.randn(17, 512))
    assert outputs['hazards'].shape == outputs['S'].shape == (1, 4)
    assert outputs['concept_A'].shape == (17, 11)
    assert all(torch.isfinite(value).all() for value in outputs.values() if torch.is_tensor(value))
    expected_auxiliary = .1*outputs['dist_loss_val']-.5*outputs['loss_clst']-.3*outputs['loss_concept_clst']
    torch.testing.assert_close(outputs['concept_loss'], expected_auxiliary, rtol=0, atol=0)
    original = load_definitions('convl', 'utils/utils.py', ['nll_loss', 'NLLSurvLoss'], {'torch': torch})
    nll = original.NLLSurvLoss(alpha=0.)(outputs['hazards'], outputs['S'], torch.tensor([2]), torch.tensor([0]))
    loss = nll+outputs['concept_loss']; assert torch.isfinite(loss)
    loss.backward()
    names = ['prompt_learner.ctx', 'prompt_learner.meta_net.fc1.weight', 'prompt_learner.meta_net.fc2.weight',
        'fc_concept.0.weight', 'classifier.weight', 'concept_pred_classifier.weight',
        'attention_net.mha.in_proj_weight', 'attention_net.attention_c.weight']
    parameters = dict(model.named_parameters())
    for name in names:
        assert parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all(), name
        assert parameters[name].grad.abs().sum() > 0, name
    for index in [0, 11]:
        parameter = model.text_encoder.transformer.resblocks[index].attn.in_proj_weight
        assert parameter.requires_grad and parameter.grad is not None and parameter.grad.abs().sum() > 0
    # The public forward does not use fc_patch, token embedding or CLS; keep
    # that distinction instead of claiming all allocated tensors are trained.
    assert model.fc_patch[0].weight.grad is None
    assert model.clip_model.text.token_embedding.weight.grad is None
    assert model.clip_model.text.cls_emb.grad is None

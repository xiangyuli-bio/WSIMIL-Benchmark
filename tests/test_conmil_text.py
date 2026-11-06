"""Actual pretrained12-layer native CONCH causal/CLS encoding, not substitutes."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus.conmil_demo import default_text
from milplus.conmil_text import load_text, native_encoder, validate_prompts


def test_actual_pretrained_full_text_tower_matches_native_causal_cls_path():
    torch.set_num_threads(2)
    assets = load_text(default_text())
    weights = '/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth'
    model, tokenizer = native_encoder(weights)
    assert len(model.text.transformer.resblocks) == 12 and model.text.token_embedding.weight.shape == (32007, 768)
    assert model.text.text_projection.shape == (768, 512)
    prompts = assets.metadata['prompts']
    with torch.inference_mode():
        for role in ['positive', 'negative']:
            tokens = tokenizer.tokenize(tokenizer.get_tokenizer(), prompts[role])
            raw, _ = model.text(tokens[:, :-1])
            expected = F.normalize(raw, dim=-1)
            actual = model.encode_text(tokens, normalize=True)
            assert torch.equal(actual, expected)
            np.testing.assert_array_equal(actual.numpy(), getattr(assets, role).numpy())


def test_positive_only_source_may_omit_negative_queries():
    import copy
    spec = copy.deepcopy(load_text(default_text()).metadata['prompts']); spec['negative'] = []
    assert validate_prompts(spec)['negative'] == []
    spec['positive'] = []
    with pytest.raises(ValueError, match='at least one positive'):
        validate_prompts(spec)


def test_query_semantics_and_explicit_boolean_marker():
    import copy
    spec = copy.deepcopy(load_text(default_text()).metadata['prompts'])
    spec['synthetic'] = 'false'
    with pytest.raises(ValueError, match='boolean'):
        validate_prompts(spec)
    spec['synthetic'] = True; spec['task_semantics'] = ''
    with pytest.raises(ValueError, match='task semantics'):
        validate_prompts(spec)

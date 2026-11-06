import copy
import json
import types

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.nn import functional as F

from milplus.concept_cate import ConceptCATE, load_cate
from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, inspect_method
from milplus.vila import load_clip_text, load_vila
from milplus.vision_language_demo import create_vision_demo


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def data(tmp_path, method='cate', classes=2):
    kind = 'concept_anchors' if method == 'cate' else 'learned_dual_scale_prompts'
    return Bundle(create_vision_demo(tmp_path/'data', kind, patients=10*classes, classes=classes))


def options(method):
    return {'pim_weight': 1., 'sim_weight': .1} if method == 'cate' else {'hidden': 8, 'prototypes': 3}


@pytest.mark.parametrize('label', [0, 1])
@pytest.mark.parametrize('dimension', [16, 512])
def test_cate_native_full_output_loss_and_gradient(tmp_path, label, dimension):
    bundle = Bundle(create_vision_demo(tmp_path/'data', 'concept_anchors', dimension=dimension))
    model = build_bundle_model('cate', bundle, options('cate'))
    author = load_cate(bundle.vision_language.anchors, bundle.dim, stable=False).DAttention(input_dim=2*bundle.dim)
    author.load_state_dict(model.network.state_dict())
    x = torch.randn(19, bundle.dim, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    y = torch.tensor([label])
    torch.manual_seed(89); result = model(x, label=y)
    torch.manual_seed(89); logits, _, _, losses = author(xx, y)
    torch.testing.assert_close(result['logits'], logits, atol=0, rtol=0)
    loss = F.cross_entropy(result['logits'], y) + result['regularization_loss']
    ref = F.cross_entropy(logits, y) + (losses['infonce_loss'] + .1*losses['kl_loss'])
    torch.testing.assert_close(loss, ref, atol=0, rtol=0)
    loss.backward(); ref.backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=0, rtol=0)
    for p, q in zip(model.network.parameters(), author.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert model.network.cate.encoder_IB[0].weight.grad.abs().sum() > 0
    assert model.network.cate.interv_linear[0].weight.grad.abs().sum() > 0
    assert all(value.grad is None for value in model.network.cate.feats)


def test_cate_detach_concepts_negative_groups_and_inference_rng(tmp_path):
    bundle = data(tmp_path)
    model = build_bundle_model('cate', bundle, options('cate'))
    x = torch.randn(15, bundle.dim, requires_grad=True)
    _, interference, _ = model.network.cate.ib(x, torch.tensor([1]), {})
    interference.square().sum().backward()
    assert x.grad is None and model.network.cate.encoder_IB[0].weight.grad is None
    assert model.network.cate.interv_linear[0].weight.grad is not None
    model.eval()
    rng = torch.get_rng_state().clone()
    a = model(x)['logits']; b = model(x, label=torch.tensor([0]))['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='at least 10'):
        model(x[:9])
    with pytest.raises(TypeError):
        ConceptCATE(bundle.dim, 2, bundle.vision_language)
    with pytest.raises(ValueError, match='positive'):
        ConceptCATE(bundle.dim, 2, bundle.vision_language, pim_weight=0, sim_weight=1)


@pytest.mark.parametrize('classes', [2, 3])
def test_vila_source_full_output_loss_gradient(tmp_path, classes):
    bundle = data(tmp_path, 'vilamil_source', classes)
    model = build_bundle_model('vilamil_source', bundle, options('vilamil_source'))
    text = bundle.vision_language
    clip = load_clip_text(text, bundle.dim)
    definitions = load_vila(clip, text.tokens, text.prompts, structured=False)
    config = types.SimpleNamespace(input_size=bundle.dim, hidden_size=8, prototype_number=3, text_prompt=text.prompts)
    author = definitions.ViLa_MIL_Model(config, classes)
    author.load_state_dict(model.network.state_dict())
    low = torch.randn(11, bundle.dim, requires_grad=True)
    high = torch.randn(17, bundle.dim, requires_grad=True)
    low_ref, high_ref = [v.detach().clone().requires_grad_() for v in (low, high)]
    target = torch.tensor([classes-1])
    actual = model(low, high, target)
    prob, _, ref_loss = author(low_ref, None, high_ref, None, target)
    torch.testing.assert_close(actual['probabilities'], prob, atol=0, rtol=0)
    torch.testing.assert_close(actual['ce_loss'], ref_loss, atol=0, rtol=0)
    actual['ce_loss'].backward(); ref_loss.backward()
    torch.testing.assert_close(low.grad, low_ref.grad, atol=0, rtol=0)
    torch.testing.assert_close(high.grad, high_ref.grad, atol=0, rtol=0)
    for p, q in zip(model.network.parameters(), author.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert model.network.prompt_learner.ctx.grad.abs().sum() > 0
    assert model.network.text_encoder.text_projection.grad.abs().sum() > 0


def test_vila_corrected_eot_freeze_and_single_prototype(tmp_path):
    bundle = data(tmp_path, 'vilamil')
    model = build_bundle_model('vilamil', bundle, {'hidden': 8, 'prototypes': 1})
    text = model.network.text_encoder
    prompt = model.network.prompt_learner()
    ids = model.network.prompt_learner.tokenized_prompts
    actual = text(prompt, ids)
    x = text.ln_final(text.transformer((prompt+text.positional_embedding).permute(1,0,2)).permute(1,0,2))
    expected = x[torch.arange(len(x)), ids.argmax(-1)+16] @ text.text_projection
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    wrong = x[torch.arange(len(x)), ids.argmax(-1)] @ text.text_projection
    assert not torch.allclose(actual, wrong)
    result = model(torch.randn(1,bundle.dim), torch.randn(9,bundle.dim), torch.tensor([0]))
    result['ce_loss'].backward()
    assert model.network.prompt_learner.ctx.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in text.parameters())
    assert model.network.cross_attention_1.in_proj_weight.grad.abs().sum() > 0
    assert model.network.cross_attention_2.in_proj_weight.grad.abs().sum() > 0


@pytest.mark.parametrize('method', ['cate', 'vilamil', 'vilamil_source'])
def test_all_text_profiles_train_reload_resume_and_input_hash(tmp_path, method):
    bundle = data(tmp_path, method)
    cfg = {'epochs': 2, 'threads': 1, 'model': options(method)}
    full = train(bundle.root, method, tmp_path/'full', config=cfg, seed=8)
    partial = train(bundle.root, method, tmp_path/'partial', config=cfg, seed=8, stop_after=1)
    train(bundle.root, method, partial, config=cfg, seed=8, resume=True)
    a, b = evaluate(full), evaluate(partial)
    assert a == b
    astate, bstate = [torch.load(p/'last.pt', weights_only=False) for p in (full, partial)]
    assert astate['history'] == bstate['history']
    torch.testing.assert_close(astate['model'], bstate['model'], atol=0, rtol=0)
    torch.testing.assert_close(astate['optimizers'], bstate['optimizers'], atol=0, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(partial/'test_predictions.csv'), check_exact=True)
    old = bundle.fingerprint
    meta = json.loads((bundle.root/'text.json').read_text())
    meta['text_provenance'] += ' edited'
    (bundle.root/'text.json').write_text(json.dumps(meta))
    assert Bundle(bundle.root).fingerprint != old
    with pytest.raises(ValueError, match='changed|mismatch|differ'):
        evaluate(full)


@pytest.mark.parametrize('change', ['class_order', 'space', 'synthetic', 'prompt_count'])
def test_cate_contract_rejects_invalid_inputs(tmp_path, change):
    bundle = data(tmp_path)
    path = bundle.root/'text.json'; meta = json.loads(path.read_text())
    if change == 'class_order': meta['classes'].reverse()
    if change == 'space': meta['embedding_space'] = 'unrelated'
    if change == 'synthetic': meta['synthetic'] = False
    if change == 'prompt_count': meta['groups'][0]['prompts'].pop()
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError): Bundle(bundle.root)


@pytest.mark.parametrize('change', ['eot', 'too_long', 'dimension', 'nan', 'missing_scale', 'same_scale'])
def test_vila_contract_rejects_invalid_inputs(tmp_path, change):
    bundle = data(tmp_path, 'vilamil')
    if change in {'eot', 'too_long'}:
        path = bundle.root/'tokens.npy'; t = np.load(path)
        t[0, 20] = 1
        if change == 'too_long': t[0, 40] = 63
        np.save(path, t)
    if change in {'dimension', 'nan'}:
        path = bundle.root/'text_weights.pt'; state = torch.load(path, weights_only=True)
        if change == 'dimension': state['text_projection'] = torch.zeros(16, 17)
        else: state['text_projection'][0, 0] = float('nan')
        torch.save(state, path)
    if change in {'missing_scale', 'same_scale'}:
        path = bundle.root/'high.csv'; frame = pd.read_csv(path)
        if change == 'missing_scale': frame = frame.iloc[:-1]
        else: frame.loc[0, 'path'] = pd.read_csv(bundle.root/'features.csv').iloc[0].path
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        b = Bundle(bundle.root)
        build_bundle_model('vilamil', b, options('vilamil'))


def test_missing_text_is_incompatible_and_high_scale_inventory_changes(tmp_path):
    bundle = data(tmp_path, 'vilamil')
    inventory = bundle.check(deep=True)['feature_inventory_sha256']
    row = bundle.rows.iloc[0]
    path = bundle.root/bundle.dual_scale.rows[row.slide_id]['path']
    np.save(path, np.zeros((2,bundle.dim), dtype=np.float32))
    assert Bundle(bundle.root).check(deep=True)['feature_inventory_sha256'] != inventory
    cfg = copy.deepcopy(bundle.config)
    cfg.pop('vision_language')
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(cfg))
    for mid in ('cate', 'vilamil'):
        with pytest.raises(ValueError): check_compatibility(inspect_method(mid), Bundle(bundle.root))


def test_source_short_prompts_collapse_but_corrected_context_is_class_specific(tmp_path):
    bundle = data(tmp_path, 'vilamil')
    tokens = np.zeros((4,48),dtype=np.int64)
    for i in range(4):
        tokens[i,:5] = [62,i+1,i+3,i+6,63]
    np.save(bundle.root/'tokens.npy', tokens)
    bundle = Bundle(bundle.root)
    legacy = build_bundle_model('vilamil_source', bundle, options('vilamil_source'))
    corrected = build_bundle_model('vilamil', bundle, options('vilamil'))
    corrected.network.load_state_dict(legacy.network.state_dict())
    def encode(model):
        learner = model.network.prompt_learner
        return model.network.text_encoder(learner(), learner.tokenized_prompts)
    a, b = encode(legacy), encode(corrected)
    torch.testing.assert_close(a[0], a[1], atol=0, rtol=0)
    assert not torch.allclose(b[0], b[1])


def test_concept_export_preserves_rows_prompts_and_hashes(tmp_path):
    from milplus.text_preparation import prepare_text
    bundle = data(tmp_path)
    path = tmp_path/'concepts.pt'
    torch.save(bundle.vision_language.anchors, path)
    output = prepare_text('concept_anchors', path, bundle.root/'text.json', tmp_path/'export')
    config = copy.deepcopy(bundle.config)
    config['vision_language']['metadata'] = str(output)
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(config))
    exported = Bundle(bundle.root)
    for a,b in zip(exported.vision_language.anchors,bundle.vision_language.anchors):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    target = json.loads(output.read_text())['groups'][0]['path']
    np.save(target,np.zeros((3,bundle.dim),dtype=np.float32))
    with pytest.raises(ValueError,match='hash mismatch'):
        Bundle(bundle.root)
    with pytest.raises(FileExistsError):
        prepare_text('concept_anchors',path,bundle.root/'text.json',tmp_path/'export')


def test_pinned_clip_tokenizer_and_context_limit():
    from milplus.text_preparation import clip_tokenize
    tokens=clip_tokenize(['a photo of a cat','low_scale tissue'])
    assert tokens[0,:7].tolist()==[49406,320,1125,539,320,2368,49407]
    assert (tokens[:,0]==49406).all()
    with pytest.raises(ValueError,match='exceeds CLIP context'):
        clip_tokenize(['tissue '*100])


def test_native_vila_rejects_unrelated_image_fm_even_if_dimension_matches(tmp_path):
    bundle = data(tmp_path, 'vilamil')
    meta = bundle.vision_language.metadata
    meta.update(synthetic=False, model='OpenAI-CLIP-RN50', tokenizer='openai_clip_bpe')
    (bundle.root/'text.json').write_text(json.dumps(meta))
    task=bundle.config
    task['synthetic']=False
    task['encoders']['toy']['model']='unrelated_image_FM'
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='image model/weights'):
        Bundle(bundle.root)

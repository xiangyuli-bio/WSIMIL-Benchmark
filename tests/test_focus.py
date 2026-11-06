"""Full pretrained original FOCUS controls, actual assets, selection and resume."""
import ast
import copy
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.focus import FOCUSMIL
from milplus.focus_data import prepare, validate_metadata
from milplus.focus_demo import create_focus_demo
from milplus.focus_network import COMMIT, WEIGHTS_SHA, FOCUSReference, native_text_model
from milplus.io import digest_file, digest_object, read_json, write_json
from milplus.upstream import load_single, pinned_text, verified_source


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def weights():
    return Path(os.environ.get('MILPLUS_FOCUS_WEIGHTS', Path.home()/'.cache/milplus/assets/focus/conch.pth')).resolve()


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def prompts(classes=2):
    return [f'At {scale} magnification, tissue of tumor subtype {i} has distinctive diagnostic morphology.'
            for scale in ['low', 'high'] for i in range(classes)]


def definitions(relative, names, scope):
    root, sha = verified_source('focus')
    assert sha == COMMIT
    tree = ast.parse(pinned_text(root, sha, relative))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_FOCUS_'+relative, 'exec'), scope)  # noqa: S102 -- fixed original source oracle
    return scope


def original_model(state, texts, dimension):
    from types import SimpleNamespace

    tokenizer = load_single('conch_focus', 'conch/open_clip_custom/custom_tokenizer.py')

    def factory(config, checkpoint):
        assert config == 'conch_ViT-B-16' and checkpoint == 'ckpts/conch.pth'
        return native_text_model(weights()), None

    module = definitions('models/model_FOCUS.py', ['TextEncoder', 'PromptLearner', 'FOCUS'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'get_tokenizer': tokenizer.get_tokenizer,
         'tokenize': tokenizer.tokenize, 'create_model_from_pretrained': factory})
    result = module['FOCUS'](SimpleNamespace(input_size=dimension, window_size=8,
        sim_threshold=.8, max_context_length=8192, text_prompt=texts), num_classes=len(texts)//2)
    result.load_state_dict(state)
    return result


def test_actual_full_pretrained_weights_prompt_buffers_and_trainability():
    model = FOCUSReference(weights(), prompts())
    source = torch.load(weights(), map_location='cpu', weights_only=True, mmap=True)
    for name, value in model.network.text_encoder.state_dict().items():
        exact(value, source['text.'+name])
    text = model.network.text_encoder
    assert len(text.transformer.resblocks) == 12
    assert text.transformer.resblocks[0].attn.num_heads == 12
    assert text.positional_embedding.shape == (128, 768) and text.text_projection.shape == (768, 512)
    assert sum(p.numel() for p in text.parameters()) == 85547520
    assert all(p.requires_grad for p in text.parameters())
    learned = model.network.prompt_learner
    assert learned.ctx.shape == (16, 768) and learned.token_suffix.shape == (4, 111, 768)
    tokenizer = load_single('conch_focus', 'conch/open_clip_custom/custom_tokenizer.py')
    tokens = tokenizer.tokenize(tokenizer.get_tokenizer(), prompts())
    assert tokens.shape == (4, 128)
    embedded = F.embedding(tokens, source['text.token_embedding.weight'])
    exact(learned.token_prefix, embedded[:, :1]); exact(learned.token_suffix, embedded[:, 17:])
    assert not learned.token_prefix.requires_grad and not learned.token_suffix.requires_grad
    assert all('visual' not in name for name, _ in model.named_parameters())


@pytest.mark.parametrize('dimension,classes', [(512, 2), (1536, 3)])
def test_independent_original_full_forward_all_parameter_and_feature_gradients(dimension, classes):
    texts = prompts(classes)
    model = FOCUSReference(weights(), texts, dimension)
    original = original_model(model.network.state_dict(), texts, dimension)
    x = torch.randn(33, dimension, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    rng = rng_state()
    value = model(x, torch.tensor([classes-1]))
    restore_rng(rng)
    expected = original(None, y, torch.tensor([classes-1]))
    for key, item in zip(['probabilities', 'prediction', 'loss'], expected):
        exact(value[key], item)
    value['loss'].backward(); expected[-1].backward()
    exact(x.grad, y.grad)
    for (name, a), (other, b) in zip(model.network.named_parameters(), original.named_parameters()):
        assert name == other
        exact(a.grad, b.grad)
    assert all(p.grad is not None for p in model.network.text_encoder.parameters())
    assert model.network.prompt_learner.ctx.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        first = model(x.detach())['probabilities']
        for label in range(classes):
            exact(model(x.detach(), torch.tensor([label]))['probabilities'], first)


def test_native_long_bag_top8192_order_and_actual_chunk_compression_independent_oracle():
    model = FOCUSReference(weights(), prompts()).eval().network
    # More than the native token budget; repeated sequence windows exercise
    # actual compression without changing the original8192/8/.8 settings.
    features = torch.randn(1125, 512).repeat_interleave(8, 0)
    texts = torch.randn(2, 512)
    values = F.normalize(features, dim=-1)
    keep = []
    for start in range(0, len(features), 8):
        similarity = values[start:start+8] @ values[start:start+8].T
        ids = torch.where(similarity.mean(1) <= similarity.mean()+similarity.std(unbiased=False))[0]+start
        keep.append(ids if len(ids) else torch.tensor([start]))
    keep = torch.cat(keep)
    importance = torch.zeros(len(features))
    importance[keep] = (features @ texts.T).mean(1)[keep]
    expected_ids = importance.topk(8192).indices.sort().values
    selected, ids = model.adaptive_token_selection(features, texts)
    exact(ids, expected_ids); exact(selected, features[expected_ids])
    assert len(ids) == 8192 and (ids[1:] > ids[:-1]).all()
    expected = []
    for start in range(0, len(selected), 8):
        part = selected[start:start+8]
        normalized = F.normalize(part, dim=-1)
        mask = (normalized[:-1]*normalized[1:]).sum(-1) < .8
        expected.append(torch.cat([part[:1], part[1:][mask]]))
    compressed = model.spatial_token_compression(selected, texts)
    exact(compressed, torch.cat(expected)[:8192])
    assert 1 <= len(compressed) < len(selected)


class OriginalLoader(Dataset):
    def __init__(self, bundle, rows):
        self.bundle, self.rows = bundle, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = self.bundle.focus.load(row)
        return x.new_empty((0, x.shape[1])), x, torch.tensor([row['target']])


def original_collate(items):
    return items[0]


def test_actual_original_train_loop_full_text_and_adam_updates_rng_order(tmp_path):
    bundle = Bundle(create_focus_demo(tmp_path/'data'))
    # Two complete training bags cover both labels, no TEST bags or hidden patch labels.
    rows = [next(r for r in bundle.records('train') if r['target'] == k) for k in [0, 1]]
    bundle.records_original = bundle.records
    bundle.records = lambda subset: rows if subset == 'train' else bundle.records_original(subset)
    model = FOCUSMIL(bundle); model.train()
    original = original_model(model.core.network.state_dict(), bundle.focus.prompts, bundle.dim)
    a = model.make_optimizer(.0001, .00001)
    b = torch.optim.Adam(original.parameters(), lr=.0001, weight_decay=.00001)
    helper = definitions('utils/core_utils.py', ['Accuracy_Logger', 'train_loop'],
        {'torch': torch, 'np': np, 'calculate_error': lambda x, y: 1-float(x.eq(y).float().mean())})
    seed_everything(41); rng = rng_state()
    model.start_epoch(0); model.epoch(bundle, a, 'cpu')
    actual_rng = rng_state()
    restore_rng(rng)
    loader = DataLoader(OriginalLoader(bundle, rows), batch_size=1,
        sampler=RandomSampler(rows), num_workers=0, collate_fn=original_collate)
    helper['train_loop'](None, 0, original, loader, b, 2)
    exact(model.core.network.state_dict(), original.state_dict())
    exact(a.state_dict(), b.state_dict()); exact(actual_rng, rng_state())
    assert model.optimizer_updates == 2 and set(model.train_orders[0]) == {r['slide_id'] for r in rows}


@pytest.mark.parametrize('mutation', ['classes', 'order', 'synthetic', 'native_mismatch', 'unknown_alignment', 'missing_order'])
def test_metadata_rejects_wrong_classes_prompts_alignment_and_missing_provenance(tmp_path, mutation):
    bundle = Bundle(create_focus_demo(tmp_path/'data'))
    metadata = copy.deepcopy(bundle.focus.manifest['metadata'])
    if mutation == 'classes': metadata['classes'] = metadata['classes'][::-1]
    if mutation == 'order': metadata['prompts'] = metadata['prompts'][::-1]
    if mutation == 'synthetic': metadata['synthetic'] = False
    if mutation == 'native_mismatch': metadata['alignment_policy'] = 'native_conch'
    if mutation == 'unknown_alignment': metadata['alignment_policy'] = 'automatic'
    if mutation == 'missing_order': metadata.pop('feature_order_provenance')
    with pytest.raises(ValueError):
        validate_metadata(bundle, metadata)


def test_ordered_feature_mutation_and_manifest_tampering_refused(tmp_path):
    bundle = Bundle(create_focus_demo(tmp_path/'data'))
    row = bundle.records('train')[0]
    path = bundle.root/row['path']
    with h5py.File(path, 'r+') as f:
        f['features'][:] = f['features'][:][::-1]
    with pytest.raises(ValueError, match='bytes changed'):
        bundle.focus.load(row)
    manifest = bundle.focus.manifest
    manifest['metadata']['prompts'][0]['text'] = 'changed unbound prompt'
    manifest['contract'] = digest_object({k: v for k, v in manifest.items() if k != 'contract'})
    write_json(bundle.focus.path, manifest)
    with pytest.raises(ValueError, match='embedded metadata'):
        Bundle(bundle.root)


def test_native_val_error_firstbest_ties_and_original_minimum_stopping(tmp_path):
    bundle = Bundle(create_focus_demo(tmp_path/'data')); model = FOCUSMIL(bundle)
    frame = pd.DataFrame({'target': [0, 1], 'p_0': [.6, .1], 'p_1': [.4, .9], 'focus_ce': [4., 5.]})
    assert model.validation_metrics(frame) == {'focus_val_error': 0., 'focus_val_ce': 4.5}
    assert model.selection_objective(model.validation_metrics(frame), 0, -np.inf) == 0.
    assert not model.selection_improved(0., 0.) and model.selection_improved(0., -1.)
    assert model.stop_after_epoch == 41 and model.settings['max_context_length'] == 8192
    source = definitions('utils/core_utils.py', ['EarlyStopping'], {'np': np, 'torch': torch})
    early = source['EarlyStopping'](patience=20, stop_epoch=40)
    early.save_checkpoint = lambda *args: None
    for epoch in range(41):
        early(epoch, 0., None)
        assert not early.early_stop
    early(41, 0., None)
    assert early.early_stop and early.counter == 41


def test_invalid_generic_subsampling_clipping_selection_and_memory_refused(tmp_path):
    bundle = Bundle(create_focus_demo(tmp_path/'data')); model = FOCUSMIL(bundle, max_dense_bytes=1)
    with pytest.raises(ValueError, match='max_dense_bytes'):
        model.prepare(bundle, 7, 'cpu')
    for cfg in [{'max_instances': 100}, {'grad_clip': 5}, {'selection_metric': 'macro_auc'}]:
        with pytest.raises(ValueError):
            model.configure_training({'max_instances': None, 'grad_clip': None, 'selection_metric': 'focus_val_error', **cfg})


def test_changed_fixed_prompt_buffers_and_nontrain_checkpoint_order_refused(tmp_path):
    bundle = Bundle(create_focus_demo(tmp_path/'data')); model = FOCUSMIL(bundle)
    state = model.state_dict()
    original = state['core.network.prompt_learner.token_prefix'].clone()
    state['core.network.prompt_learner.token_prefix'] = original+1
    with pytest.raises(ValueError, match='fixed prompt buffers'):
        model.load_state_dict(state)
    state['core.network.prompt_learner.token_prefix'] = original
    state['_extra_state'] = {**state['_extra_state'], 'epoch_index': 0, 'completed_epochs': 1,
        'optimizer_updates': len(bundle.records('train')),
        'train_orders': [[bundle.records('test')[0]['slide_id']]*len(bundle.records('train'))]}
    with pytest.raises(ValueError, match='non-TRAIN'):
        model.load_state_dict(state)


def test_complete_full_text_training_exact_resume_reload_and_input_contract(tmp_path):
    data = create_focus_demo(tmp_path/'data')
    config = {'epochs': 2, 'threads': 1}
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    train(data, 'FOCUS', full, seed=93, config=config)
    train(data, 'FOCUS', resumed, seed=93, config=config, stop_after=1)
    assert read_json(resumed/'status.json')['state'] == 'interrupted'
    train(data, 'FOCUS', resumed, seed=93, config=config, resume=True)
    a = torch.load(full/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'history', 'best', 'bad', 'rng']:
        exact(a[key], b[key])
    assert a['model']['_extra_state']['optimizer_updates'] == 2*len(Bundle(data).records('train'))
    assert not (full/'test_predictions.csv').exists()
    evaluate(full)
    original = (full/'test_predictions.csv').read_bytes(); evaluate(full)
    assert (full/'test_predictions.csv').read_bytes() == original
    assert read_json(full/'method_protocol.json')['trainable_text_parameters'] == 85547520


def test_cli_fixture_setup_alias_and_prepare_refuses_overwrite(tmp_path):
    from milplus.cli import dispatch, parser
    from milplus.registry import resolve_for_bundle

    args = parser().parse_args(['demo', '--focus', '--patients', '20', '--output', str(tmp_path/'cli')])
    bundle = Bundle(dispatch(args)['created'])
    assert bundle.dim == 512 and bundle.config['synthetic'] and bundle.config['evaluation_unit'] == 'slide'
    assert resolve_for_bundle('FOCUS', bundle)['id'] == 'focus_source'
    with pytest.raises(FileExistsError):
        prepare(bundle, bundle.focus.manifest['source']['path'], bundle.focus.path)
    for flag in ['--codmil', '--raw-images', '--raw-regions', '--smmile', '--vision-language']:
        value = ['binary'] if flag == '--smmile' else ['concept_anchors'] if flag == '--vision-language' else []
        with pytest.raises(ValueError, match='mixed demo'):
            dispatch(parser().parse_args(['demo', '--focus', flag, *value, '--output', str(tmp_path/'bad')]))
    assert digest_file(weights()) == WEIGHTS_SHA

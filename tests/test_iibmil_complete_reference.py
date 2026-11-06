"""Full IIB training mechanism checks, including literal original data and step oracles."""
import copy
import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate
import yaml

from milplus.data import Bundle
from milplus.engine import rng_state, restore_rng, seed_everything
from milplus.iibmil import CONDITIONS, IIBMIL, check_inputs
from milplus.iibmil_data import Bags, Inputs, patch_names, select_rows
from milplus.iibmil_demo import create_demo
from milplus.iibmil_source import author_components, checked_source, source_config
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, inspect_method, resolve_for_bundle


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert type(a) is type(b) and a == b


def literal(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp('iib_full_native_features')
    return {c: create_demo(root / str(c), c, 1280) for c in [2, 3]}


@pytest.mark.parametrize('classes', [2, 3])
def test_TRAIN_initializer_and_actual500row_draws_equal_literal_author_dataset(datasets, tmp_path, classes):
    bundle = Bundle(datasets[classes])
    inputs = Inputs(bundle)
    source, _ = checked_source()
    module = literal(source / 'main/datasets/wsi_feat_dataset.py', '_iib_literal_dataset')
    annotations = []
    for row in bundle.records('train'):
        features, _ = bundle.load(row)
        names = inputs.names[row['slide_id']]
        path = tmp_path / (row['slide_id'] + '.pkl')
        # Own trusted pickle fixture only; no external untrusted input loaded.
        with path.open('wb') as f:
            pickle.dump([{'val': value.numpy(), 'feat_name': name} for value, name in zip(features, names)], f)
        annotations.append(f"{row['slide_id']},{path},{row['target']}\n")
    table = tmp_path / 'TRAIN.txt'
    table.write_text(''.join(annotations))
    args = SimpleNamespace(DATASET=SimpleNamespace(DATASET_SCALE='synthetic', FEATURE_LEN=500,
        PATCH_SCORE_PATH=None, PATCH_LEVEL=True), MODEL=SimpleNamespace(NUM_CLASSES=classes), is_training=True)
    original = module.build(str(table), 'train', args)
    actual = Bags(inputs, 'train')
    exact(inputs.initial_confidence(), original.confidence)
    assert len(inputs.bank_keys) == sum(inputs.counts[r['slide_id']] for r in bundle.records('train'))
    assert {k[0] for k in inputs.bank_keys}.isdisjoint(r['slide_id'] for r in bundle.records('test'))
    for index in [0, 1, 2]:  # overflow641, padded437, overflow501
        for seed in [1, 5, 17]:
            np.random.seed(seed)
            a = actual[index]
            after = np.random.get_state()
            np.random.seed(seed)
            value, mask, target = original[index]
            exact(after, np.random.get_state())
            exact(a['features'], value)
            exact(a['mask'], mask)
            exact(a['bank_indices'], torch.tensor(target['index']))
            sid = a['slide_id']
            assert [inputs.names[sid][i] for i in a['original_rows'].tolist()] == target['patch_name']
            assert value.shape == (500, 1280) and target['batchY'] == [[1] * classes] * 500


@pytest.mark.parametrize('classes', [2, 3])
@pytest.mark.parametrize('epoch', [9, 10])
def test_complete_dual_objective_step_matches_independent_literal_source(datasets, monkeypatch, classes, epoch):
    torch.set_num_threads(2)
    bundle = Bundle(datasets[classes])
    condition = 'tcga_nsclc' if classes == 2 else 'tcga_rcc'
    profile = 'iibmil_' + condition + '_complete_reference'
    root, _ = checked_source()
    _, nested, _ = author_components()
    package, misc = ModuleType('util'), ModuleType('util.misc')
    misc.NestedTensor = nested
    monkeypatch.setitem(sys.modules, 'util', package)
    monkeypatch.setitem(sys.modules, 'util.misc', misc)
    source = literal(root / 'main/models/iibmil.py', '_iib_literal_step_network')
    loss_source = literal(root / 'main/util/loss.py', '_iib_literal_step_loss')
    cfg = source_config(condition)
    seed_everything(10511)
    actual = IIBMIL(bundle, profile)
    actual.configure_training({'epochs':12, 'max_instances':None, 'grad_clip':.01})
    optimizer = actual.make_optimizer(cfg['TRAIN']['LR'], 1e-5)
    seed_everything(10511)
    original = source.IIBMIL(classes, 1280, 256, 3 if classes == 2 else 4, 5)
    original.train()
    original_confidence = actual.inputs.initial_confidence()
    original_partial = loss_source.PartialLoss(original_confidence, {})
    original_optimizer = torch.optim.AdamW(original.parameters(), lr=cfg['TRAIN']['LR'], weight_decay=1e-5)
    original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, 100, gamma=.1)
    exact(actual.native.state_dict(), original.state_dict())
    actual.start_epoch(epoch)
    original_partial.set_conf_ema_m(epoch)
    assert actual.partial.conf_ema_m == original_partial.conf_ema_m
    # Three actual500-token bags include every class, valid padding and one
    # singleton-class subbatch; native network widths/depths are not reduced.
    batch = default_collate([Bags(actual.inputs, 'train')[i] for i in range(classes)])
    before_confidence = original_confidence.clone()
    valid_indices = batch['bank_indices'][batch['mask']]
    seed_everything(10512)
    result = actual.train_batch(batch, optimizer, torch.device('cpu'))
    after_actual = rng_state()
    seed_everything(10512)
    patch, embeddings, prototype_scores, bag = original(nested(batch['features'], batch['mask']))
    loss = F.cross_entropy(bag, batch['target']) + original_partial(patch[batch['mask']], valid_indices)
    original_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), .01, error_if_nonfinite=True)
    original_optimizer.step()
    with torch.no_grad():
        for c in range(classes):
            selected = np.argwhere(batch['target'].numpy() == c)
            if len(selected):
                original.update_prototypes(c, selected, patch.detach(), embeddings.detach(), batch['mask'], .99)
        if epoch >= 10:
            scores = prototype_scores.reshape(classes, 500, classes)[batch['mask']]
            original_partial.confidence_update(scores, valid_indices, torch.ones_like(scores))
    exact(after_actual, rng_state())
    assert result['loss'] == float(loss.detach())
    exact(actual.native.state_dict(), original.state_dict())
    exact(actual.confidence, original_partial.confidence)
    exact(optimizer.state_dict(), original_optimizer.state_dict())
    for (name, a), (other, b) in zip(actual.native.named_parameters(), original.named_parameters()):
        assert name == other and (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            exact(a.grad, b.grad)
    assert actual.native.patch_encoder.patch_to_embedding.weight.grad is None
    assert actual.native.patch_encoder.mlp_head[1].weight.grad is not None
    assert actual.native.wsi_classifier.weight.grad is not None
    assert torch.count_nonzero(actual.native.prototypes) > 0
    unchanged = torch.ones(len(before_confidence), dtype=torch.bool)
    unchanged[valid_indices] = False
    exact(actual.confidence[unchanged], before_confidence[unchanged])
    if epoch < 10:
        exact(actual.confidence, before_confidence)
    else:
        assert not torch.equal(actual.confidence, before_confidence)
    actual.finish_epoch(bundle, 'cpu', {})
    original_scheduler.step()
    exact(actual.scheduler.state_dict(), original_scheduler.state_dict())


def test_absent_class_tail_and_padded_instances_keep_unselected_bank_rows(datasets):
    torch.set_num_threads(2)
    model = IIBMIL(Bundle(datasets[3]), 'iibmil_tcga_rcc_complete_reference')
    model.configure_training({'epochs':12, 'max_instances':None, 'grad_clip':.01})
    optimizer = model.make_optimizer(1e-5, 1e-5)
    batch = default_collate([Bags(model.inputs, 'train')[1]])  # class1, only437real rows
    model.start_epoch(10)
    before = model.confidence.clone()
    result = model.train_batch(batch, optimizer, torch.device('cpu'))
    assert result['valid_instances'] == 437 and model.confidence_updates == 437
    assert torch.count_nonzero(model.native.prototypes[[0, 2]]) == 0
    valid = batch['bank_indices'][batch['mask']]
    unchanged = torch.ones(len(before), dtype=torch.bool)
    unchanged[valid] = False
    exact(before[unchanged], model.confidence[unchanged])
    assert len(torch.unique(batch['bank_indices'])) == 437  # repeated padding not a new instance
    assert len(torch.unique(valid)) == 437


def test_complete_checkpoint_restores_confidence_prototype_schedule_and_next_draws(datasets, tmp_path):
    torch.set_num_threads(2)
    bundle = Bundle(datasets[3])
    seed_everything(10531)
    first = IIBMIL(bundle, 'iibmil_tcga_rcc_complete_reference')
    cfg = {'epochs':12, 'max_instances':None, 'grad_clip':.01}
    first.configure_training(cfg)
    optimizer = first.make_optimizer(1e-5, 1e-5)
    first.start_epoch(10)
    first.epoch(bundle, optimizer, torch.device('cpu'))
    first.finish_epoch(bundle, 'cpu', cfg)
    path = tmp_path / 'complete.pt'
    torch.save({'model':first.state_dict(), 'optimizer':optimizer.state_dict(), 'rng':rng_state()}, path)
    first.start_epoch(11)
    first.epoch(bundle, optimizer, torch.device('cpu'))
    first.finish_epoch(bundle, 'cpu', cfg)
    expected_rng = rng_state()
    second = IIBMIL(bundle, 'iibmil_tcga_rcc_complete_reference')
    second.configure_training(cfg)
    other = second.make_optimizer(1e-5, 1e-5)
    checkpoint = torch.load(path, weights_only=False)
    second.load_state_dict(checkpoint['model'])
    other.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint['rng'])
    second.start_epoch(11)
    second.epoch(bundle, other, torch.device('cpu'))
    second.finish_epoch(bundle, 'cpu', cfg)
    exact(first.state_dict(), second.state_dict())
    exact(optimizer.state_dict(), other.state_dict())
    exact(expected_rng, rng_state())
    assert first.partial.confidence is first.confidence and second.partial.confidence is second.confidence
    assert len(first.reference_history) == 2 and first.updates == 4  # fullB4 plus tail1, twice
    assert len(first.draw_history) == 10


def test_prediction_never_changes_TRAIN_state_and_preserves_actual_patients(datasets):
    torch.set_num_threads(2)
    bundle = Bundle(datasets[3])
    model = IIBMIL(bundle, 'iibmil_tcga_rcc_complete_reference')
    before = copy.deepcopy(model.state_dict())
    predictions = model.predict_subset(bundle, 'test', torch.device('cpu'), {})
    exact(before, model.state_dict())
    assert set(predictions.slide_id) == {r['slide_id'] for r in bundle.records('test')}
    assert set(predictions.global_patient_id) == {r['global_patient_id'] for r in bundle.records('test')}
    assert len(predictions) == 6 and np.allclose(predictions[['p_0','p_1','p_2']].sum(1), 1.)
    assert torch.all(Bags(model.inputs, 'test')[0]['bank_indices'] == -1)


def test_explicit_FM1024_and_five_class_ports_preserve_full_native_architecture(tmp_path):
    root = create_demo(tmp_path / 'five_class_port', 5, 1024)
    bundle = Bundle(root)
    profile = resolve_for_bundle('IIB-MIL', bundle)
    assert profile['id'] == 'iibmil_tcga_rcc_complete_reference'
    model = build_bundle_model(profile['id'], bundle, {})
    assert model.native.patch_encoder.patch_dim == 1024
    assert model.native.patch_encoder.z_dim == 256 and model.native.patch_encoder.depth == 4
    assert model.native.query_embed.weight.shape == (5,256)
    assert model.native.wsi_classifier.out_features == 5
    assert model.native.wsi_aggregator.decoder.layers[0].linear1.weight.shape == (512,256)
    assert model.native.patch_encoder._modules['trans_0'].layers[0].linear1.weight.shape == (2048,256)
    assert model.protocol()['FM_input_port'] and model.protocol()['output_class_port']
    assert model.protocol()['reference_reconstruction'] and not model.protocol()['source_training_epoch_available']
    model.configure_training({'epochs':12, 'max_instances':None, 'grad_clip':.01})
    optimizer = model.make_optimizer(1e-5, 1e-5)
    model.start_epoch(10)
    batch = default_collate([Bags(model.inputs, 'train')[i] for i in range(4)])
    terms = model.train_batch(batch, optimizer, torch.device('cpu'))
    assert np.isfinite(terms['loss']) and model.native.wsi_classifier.weight.grad.shape == (5,1280)
    assert model.native.patch_encoder.mlp_head[1].weight.grad.shape == (5,256)
    assert model.confidence.shape[1] == 5 and model.confidence_updates > 0


def test_original_patch_identity_rejects_duplicates_and_binds_actual_feature_bytes(tmp_path):
    root = create_demo(tmp_path / 'identity', 3)
    bundle = Bundle(root)
    before = Inputs(bundle).identity
    row = bundle.records('train')[0]
    path = root / row['path']
    with h5py.File(path, 'a') as f:
        f['features'][0,0] += 1.
    assert Inputs(Bundle(root)).identity != before
    with h5py.File(path, 'a') as f:
        f['patch_ids'][1] = f['patch_ids'][0]
    with pytest.raises(ValueError, match='Duplicate original patch IDs'):
        Inputs(Bundle(root))


def test_invalid_missing_contract_and_disabled_core_training_fail(datasets):
    bundle = Bundle(datasets[2])
    assert inspect_method('IIB-MIL')['id'] == 'iibmil_tcga_nsclc_complete_reference'
    check_compatibility(resolve_for_bundle('IIB-MIL', bundle), bundle)
    model = IIBMIL(bundle, 'iibmil_tcga_nsclc_complete_reference')
    for value in [0., -1., float('nan'), True]:
        with pytest.raises(ValueError, match='Both IIB-MIL'):
            IIBMIL(bundle, 'iibmil_tcga_nsclc_complete_reference', instance_coefficient=value)
    for config in [{'epochs':10,'max_instances':None,'grad_clip':.01},
                   {'epochs':12,'max_instances':50,'grad_clip':.01},
                   {'epochs':12,'max_instances':None,'grad_clip':None}]:
        with pytest.raises(ValueError):
            model.configure_training(config)
    with pytest.raises(ValueError, match='batched training'):
        model(torch.randn(500,1280))
    bundle.config['training_unit'] = 'patient'
    with pytest.raises(ValueError, match='individual WSI'):
        check_inputs(bundle)


@pytest.mark.parametrize('classes', [2, 3])
def test_public_demo_has_native_overflow_padding_and_full_tail_batches(tmp_path, classes):
    output = tmp_path / 'public_demo'
    command = [sys.executable, '-m', 'milplus', 'demo-iibmil', '--output', str(output), '--classes', str(classes)]
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    bundle = Bundle(output)
    assert bundle.config['synthetic'] and bundle.dim == 1280
    assert len(bundle.records('train')) == (17 if classes == 2 else 5)
    assert {len(bundle.load(row)[0]) for row in bundle.records('train')} == {437,501,641}
    assert bundle.config['evaluation_unit'] == 'patient'

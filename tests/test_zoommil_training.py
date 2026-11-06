"""Original epoch oracle, complete recipe state and actual interrupted fits."""
import copy
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, train
from milplus.io import digest_file, read_json
from milplus.upstream import load_definitions
from milplus.zoommil import ZoomMIL
from milplus.zoommil_demo import create_zoommil_demo
from milplus.zoommil_source import classes


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root = create_zoommil_demo(tmp_path_factory.mktemp('zoommil_recipe')/'data', dimension=32, roots=4)
    return Bundle(root)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(i, j) for i, j in zip(a, b))
    return a == b


def settings(epochs=3):
    return {'epochs': epochs, 'patience': epochs, 'lr': .0001, 'weight_decay': 0.,
            'grad_clip': None, 'max_instances': None, 'selection_metric': 'zoommil_weighted_f1'}


@pytest.mark.parametrize('num_workers', [0, 8])
def test_zoommil_complete_epoch_matches_actual_original_loader_trainer_and_scheduler(bundle, tmp_path, num_workers):
    torch.set_num_threads(1)
    # Load the actual released Dataset, trainer methods and metric/saver bodies.
    # TensorBoard construction, plotting and CLI setup are outside these paths.
    data = load_definitions('zoommil', 'zoommil/dataset/dataset.py', ['BaseDataset', 'PatchFeatureDataset'],
        {'torch': torch, 'np': np, 'pd': pd, 'Dataset': Dataset, 'h5py': h5py, 'os': os})
    logger = load_definitions('zoommil', 'zoommil/utils/logger.py', ['MetricLogger'],
        {'np': np, 'accuracy_score': accuracy_score, 'f1_score': f1_score,
         'cohen_kappa_score': cohen_kappa_score, 'confusion_matrix': confusion_matrix})
    saver = load_definitions('zoommil', 'zoommil/utils/saver.py', ['ModelSaver'], {'torch': torch, 'np': np, 'os': os})
    trainer = load_definitions('zoommil', 'zoommil/trainer.py', ['Trainer'],
        {'torch': torch, 'tqdm': tqdm, 'MetricLogger': logger.MetricLogger})
    frame = bundle.rows[['slide_id', 'target']].rename(columns={'target': 'type'})
    path = tmp_path/'source_labels.csv'; frame.to_csv(path, index=False)
    levels = bundle.zoommil.meta['schema']['levels']
    datasets = {}
    for subset in ['train', 'val']:
        datasets[subset] = data.PatchFeatureDataset(csv_path=path,
            split=pd.Series([row['slide_id'] for row in bundle.records(subset)]),
            label_dict={index: index for index in range(len(bundle.classes))},
            data_path=bundle.root/'pyramids', low_mag=str(levels[0]['magnification'])+'x',
            mid_mag=str(levels[1]['magnification'])+'x', high_mag=str(levels[2]['magnification'])+'x')
    model, _ = classes()
    reference = model.ZoomMIL(bundle.dim, dropout=.25, k_sample=12, k_sigma=.002, n_cls=len(bundle.classes))
    device = reference.device; reference.to(device)
    actual = ZoomMIL(bundle, num_workers=num_workers).to(device); actual.network.load_state_dict(reference.state_dict(), strict=True)
    actual.configure_training(settings())
    optimizer = actual.make_optimizer(.0001, 0.)
    source_optimizer = torch.optim.Adam(reference.parameters(), lr=.0001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    source_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(source_optimizer, mode='min', factor=.8, patience=5)
    original_train = datasets['train']; n = len(original_train)
    weights = torch.DoubleTensor([n/len(original_train.slide_cls_ids[original_train.get_label(i)]) for i in range(n)])
    context = SimpleNamespace(model=reference, optimizer=source_optimizer, device=device, ce_loss=nn.CrossEntropyLoss(),
        train_loader=DataLoader(original_train, batch_size=1, sampler=WeightedRandomSampler(weights, n), num_workers=num_workers),
        val_loader=DataLoader(datasets['val'], batch_size=1, sampler=SequentialSampler(datasets['val']), num_workers=num_workers),
        model_saver=saver.ModelSaver(tmp_path, save_metric='f1'))
    torch.manual_seed(1011)
    for epoch in range(3):
        before = rng_state()
        expected_train = trainer.Trainer._train_one_epoch(context)
        expected_loss, expected_metrics = trainer.Trainer._val_one_epoch(context)
        source_scheduler.step(expected_loss)
        expected_rng = rng_state()
        restore_rng(before)
        actual.start_epoch(epoch)
        losses = actual.epoch(bundle, optimizer, device)
        predictions = actual.predict_subset(bundle, 'val', device, {})
        metrics = actual.validation_metrics(predictions); actual.after_validation(metrics)
        assert sum(item['loss'] for item in losses)/len(losses) == expected_train
        assert metrics['zoommil_source_ce'] == expected_loss
        assert metrics['zoommil_weighted_f1'] == expected_metrics['val_weighted_f1']
        assert all(torch.equal(value, actual.network.state_dict()[name]) for name, value in reference.state_dict().items())
        assert equal(source_optimizer.state_dict(), optimizer.state_dict())
        assert equal(source_scheduler.state_dict(), actual._scheduler.state_dict())
        assert equal(expected_rng, rng_state())
        actual.validate_optimizer_state()
    assert actual.optimizer_updates == n*3 and len(actual.train_orders) == 3


@pytest.mark.parametrize('profile', ['zoommil_bright_source_repaired', 'zoommil_crc_source_repaired'])
def test_zoommil_actual_fit_resume_and_selected_reload_are_exact(bundle, tmp_path, profile):
    cfg = {'epochs': 3, 'threads': 1, 'model': {'num_workers': 0}}
    train(bundle.root, profile, tmp_path/'continuous', config=cfg, seed=2031)
    train(bundle.root, profile, tmp_path/'resumed', config=cfg, seed=2031, stop_after=1)
    assert read_json(tmp_path/'resumed/status.json')['state'] == 'interrupted'
    train(bundle.root, profile, tmp_path/'resumed', config=cfg, seed=2031, resume=True)
    a = torch.load(tmp_path/'continuous/last.pt', weights_only=False, map_location='cpu')
    b = torch.load(tmp_path/'resumed/last.pt', weights_only=False, map_location='cpu')
    for key in ['model', 'optimizers', 'rng', 'history', 'epoch', 'best', 'bad']:
        assert equal(a[key], b[key]), key
    extra = a['model']['_extra_state']
    assert extra['completed_epochs'] == 3 and extra['optimizer_updates'] == len(bundle.records('train'))*3
    assert len(extra['lr_history']) == 3 and extra['scheduler']['last_epoch'] == 3
    metric = 'zoommil_source_ce' if 'crc' in profile else 'zoommil_weighted_f1'
    best = torch.load(tmp_path/'continuous/best.pt', weights_only=False, map_location='cpu')
    expected = min(a['history'], key=lambda row: row['validation'][metric]) if 'crc' in profile else max(a['history'], key=lambda row: row['validation'][metric])
    assert best['epoch'] == expected['epoch']
    metrics = evaluate(tmp_path/'continuous')
    sha = digest_file(tmp_path/'continuous/test_predictions.csv')
    assert evaluate(tmp_path/'continuous') == metrics and digest_file(tmp_path/'continuous/test_predictions.csv') == sha
    assert evaluate(tmp_path/'resumed') == metrics and digest_file(tmp_path/'resumed/test_predictions.csv') == sha
    prediction = pd.read_csv(tmp_path/'continuous/test_predictions.csv')
    assert prediction.slide_id.is_unique and len(prediction) == len(bundle.records('test'))


def test_zoommil_plateau_callback_uses_ce_not_f1_and_restores_state(bundle):
    model = ZoomMIL(bundle); model.configure_training(settings(8)); optimizer = model.make_optimizer(.0001, 0.)
    reference_optimizer = torch.optim.Adam(model.network.parameters(), lr=.0001)
    reference = torch.optim.lr_scheduler.ReduceLROnPlateau(reference_optimizer, mode='min', factor=.8, patience=5)
    # Callback-level simulated validation events only, not completed fit proof.
    for epoch in range(8):
        model.epoch_index = epoch
        metrics = {'zoommil_source_ce': 1., 'zoommil_weighted_f1': epoch/8}
        reference.step(1.); model.after_validation(metrics)
        assert equal(reference.state_dict(), model._scheduler.state_dict())
    assert optimizer.param_groups[0]['lr'] == .0001*.8


@pytest.mark.parametrize('bad', ['truncation', 'clipping', 'early_stop', 'wrong_selection'])
def test_zoommil_rejects_altered_source_recipe(bundle, bad):
    cfg = settings()
    cfg[{'truncation': 'max_instances', 'clipping': 'grad_clip', 'early_stop': 'patience', 'wrong_selection': 'selection_metric'}[bad]] = {
        'truncation': 1, 'clipping': 5., 'early_stop': 1, 'wrong_selection': 'macro_auc'}[bad]
    with pytest.raises(ValueError):
        ZoomMIL(bundle).configure_training(cfg)


def test_zoommil_missing_assets_memory_guard_and_incomplete_checkpoint_rejected(bundle):
    model = ZoomMIL(bundle, max_dense_bytes=1)
    with pytest.raises(ValueError, match='no row truncation'):
        model._loader('train')
    state = copy.deepcopy(ZoomMIL(bundle).get_extra_state())
    state['completed_epochs'] = 1
    with pytest.raises(ValueError, match='complete epoch'):
        ZoomMIL(bundle).set_extra_state(state)
    plain = copy.copy(bundle); plain.zoommil = None
    with pytest.raises(ValueError, match='three-scale'):
        ZoomMIL(plain)

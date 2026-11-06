"""Genuine full original network, source trainer bodies and optimizer recovery."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils import data
import yaml

from milplus.convl import ConVLMIL
from milplus.convl_data import KIND, prepare
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import DEFAULTS, restore_rng, rng_state, seed_everything
from milplus.io import write_json
from milplus.upstream import load_definitions

WEIGHTS = Path('/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth')


def full_native_bundle(root):
    root = create_demo(root, patients=20, dimension=512, survival=True)
    original = Bundle(root)
    # Small genuine software cohort with an odd TRAIN population. Every
    # registered slide/patch is retained; no native model/text width is reduced.
    patients = original.rows.drop_duplicates('global_patient_id')
    chosen = pd.concat([patients.loc[patients.subset == subset].iloc[:count]
                        for subset, count in [('train', 3), ('val', 2), ('test', 2)]])
    ids = set(chosen.patient_id)
    slides = pd.read_csv(root/'slides.csv'); slides = slides.loc[slides.patient_id.isin(ids)]
    slides.to_csv(root/'slides.csv', index=False)
    split = pd.read_csv(root/'split.csv'); split.loc[split.patient_id.isin(ids)].to_csv(root/'split.csv', index=False)
    features = pd.read_csv(root/'features.csv'); features.loc[features.slide_id.isin(slides.slide_id)].to_csv(root/'features.csv', index=False)
    bundle = Bundle(root)
    labels = bundle.rows.drop_duplicates('global_patient_id')[['global_patient_id', 'time', 'event']].copy()
    labels['bin'] = [i % 4 for i in range(len(labels))]; labels.to_csv(root/'original_bins.csv', index=False)
    write_json(root/'metadata.json', {'schema_version': 1, 'kind': KIND, 'synthetic': True,
        'alignment_policy': 'synthetic_native_dimension_fixture', 'checkpoint': str(WEIGHTS), 'concept_cohort': 'BRCA',
        'feature_preprocessing': 'Synthetic original full512 vectors, all registered variable-length bags',
        'feature_provenance': 'Explicit synthetic software bags, not clinical CONCH extraction',
        'patient_order_provenance': 'True stable multi-slide patient groups', 'task_semantics': 'Synthetic months/event1',
        'binning': {'policy': 'provided_author_four_bin_labels', 'table': 'original_bins.csv',
                    'provenance': 'Explicit synthetic provided discrete label fixture, not historical patient labels'}})
    prepare(bundle, 'metadata.json', 'convl/manifest.json')
    task = yaml.safe_load((root/'task.yaml').read_text()); task['encoders']['toy']['convl'] = {'manifest': 'convl/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return Bundle(root)


def config():
    return {**DEFAULTS, 'epochs': 2, 'threads': 1, 'patience': 8, 'lr': .0002,
            'weight_decay': .00001, 'grad_clip': None, 'selection_metric': 'convl_val_nll'}


def reference_dataset(bundle, subset):
    namespace = {'torch': torch, 'np': np, 'pd': pd, 'Path': Path, 'data': data, 'os': os, 'h5py': h5py}
    cls = load_definitions('convl', 'datasets/ProgData.py', ['ProgData'], namespace).ProgData
    original = cls.__new__(cls)
    patients = bundle.rows.loc[bundle.rows.subset == subset].drop_duplicates('global_patient_id')
    original.patient_df = patients
    # Named correction of the author's slide-index/unique-patient mismatch;
    # actual original __getitem__ reads every registered H5 through patient_dict.
    original.case_id = pd.Series(patients.global_patient_id.tolist())
    original.survival_months = pd.Series(patients.time.tolist())
    original.censorship = pd.Series((1-patients.event).tolist())
    original.label = pd.Series([bundle.convl.meta['patients'][p]['bin'] for p in patients.global_patient_id])
    original.patient_dict = {p: rows.slide_id.to_numpy() for p, rows in bundle.rows.groupby('global_patient_id', sort=False)}
    original.feature_high_path = bundle.root/'features'
    original.cache_manager, original.dataset_name = None, 'BRCA'
    return original


def source_trainer():
    # Execute real author method bodies on the complete actual model. The
    # Lightning base is only a definition-time namespace; no fake network,
    # objective, optimizer or truncated text tower is introduced.
    return load_definitions('convl', 'models/model_interface.py', ['ModelInterfaceProg'],
        {'pl': SimpleNamespace(LightningModule=nn.Module), 'torch': torch}).ModelInterfaceProg


def assert_core_equal(actual, expected):
    a, b = actual.state_dict(), expected.state_dict()
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)


def assert_optimizer_equal(actual, expected):
    assert actual.optimizer.buffer == expected.optimizer.buffer
    for ga, gb in zip(actual.param_groups, expected.param_groups):
        assert {k: v for k, v in ga.items() if k != 'params'} == {k: v for k, v in gb.items() if k != 'params'}
        for a, b in zip(ga['params'], gb['params']):
            sa, sb = actual.optimizer.state.get(a, {}), expected.optimizer.state.get(b, {})
            assert set(sa) == set(sb)
            for key in sa:
                torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0)
            aa, bb = actual.state.get(a, {}), expected.state.get(b, {})
            assert set(aa) == set(bb)
            for key in aa:
                torch.testing.assert_close(aa[key], bb[key], rtol=0, atol=0)


def assert_rng_equal(actual, expected):
    assert actual['python'] == expected['python']
    assert actual['numpy'][0] == expected['numpy'][0] and actual['numpy'][2:] == expected['numpy'][2:]
    assert np.array_equal(actual['numpy'][1], expected['numpy'][1])
    assert torch.equal(actual['torch'], expected['torch'])


def test_full_original_twoepoch_NLL_patient_accumulation_odd_tail_optimizer_and_rng(tmp_path):
    from milplus.convl_optim import create_source_optimizer
    torch.set_num_threads(1); seed_everything(2024)
    bundle = full_native_bundle(tmp_path/'data')
    actual = ConVLMIL(bundle, 'convl_yaml_fp32_reference'); actual.configure_training(config())
    opt = actual.make_optimizer(.0002, .00001)
    original = copy.deepcopy(actual.core)
    original_opt = create_source_optimizer(original)
    holder = SimpleNamespace(model=original, loss=actual.source_nll, training_step_outputs=[], val_step_outputs=[])
    trainer = source_trainer()
    for epoch in range(2):
        start = rng_state()
        actual.train(); actual.start_epoch(epoch)
        losses = actual.epoch(bundle, opt, 'cpu')
        got_val = actual.predict_subset(bundle, 'val', 'cpu', config())
        got_rng = rng_state()
        restore_rng(start); original.train(); holder.training_step_outputs=[]; holder.val_step_outputs=[]
        loader = data.DataLoader(reference_dataset(bundle, 'train'), batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
        for index, batch in enumerate(loader):
            loss = trainer.training_step(holder, batch, index)
            if index % 2 == 0:
                original_opt.zero_grad()
            (loss/2).backward()
            if (index+1) % 2 == 0 or index+1 == len(loader):
                original_opt.step()
        original.eval()
        with torch.no_grad():
            for index, batch in enumerate(data.DataLoader(reference_dataset(bundle, 'val'), batch_size=1, num_workers=0)):
                trainer.validation_step(holder, batch, index)
        assert [row['loss'] for row in losses] == [row['loss'] for row in holder.training_step_outputs]
        assert float(got_val.convl_nll.mean()) == float(np.mean([row['loss'] for row in holder.val_step_outputs]))
        assert got_val.risk.tolist() == [row['risk'] for row in holder.val_step_outputs]
        assert actual.optimizer_updates == (epoch+1)*2
        assert [len(w['patients']) for w in actual.update_traces[-1]] == [2, 1]
        assert all(w['normalization'] == 2 for w in actual.update_traces[-1])
        assert_core_equal(actual.core, original); assert_optimizer_equal(opt, original_opt)
        assert_rng_equal(got_rng, rng_state())


def test_cuda_precision_route_is_explicit_and_no_generic_sampling_or_auc_shortcut(tmp_path):
    torch.set_num_threads(1)
    bundle = full_native_bundle(tmp_path/'data')
    model = ConVLMIL(bundle, 'convl_yaml_fp16_source_repaired')
    with pytest.raises(ValueError, match='FP16 mixed precision requires CUDA'):
        model.configure_training(config())
    for override in [{'max_instances': 17}, {'grad_clip': 1.}, {'selection_metric': 'c_index'}]:
        with pytest.raises(ValueError, match='complete patient bags'):
            model.configure_training({**config(), **override})


def test_original_NLL_selection_is_strict_minimum_and_excludes_auxiliary_losses():
    # These are the actual method's public selection/diagnostic hooks, not an
    # alternate trainer. Counterexample to using source concept loss or C-index.
    frame = pd.DataFrame({'convl_nll': [2., 4.], 'concept_loss': [-100., 100.], 'risk': [10., -10.]})
    metrics = ConVLMIL.validation_metrics(None, frame)
    assert metrics == {'convl_val_nll': 3.}
    assert ConVLMIL.selection_objective(None, metrics, 0, 100.) == -3.

"""Real patient manifests, native feature identity and original survival bins."""
from pathlib import Path

import h5py
import pandas as pd
import pytest

from milplus.convl_data import KIND, ConVLInputs, prepare
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.io import read_json, write_json

WEIGHTS = Path('/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth')


def setup(tmp_path, policy='train_observed_event_quartiles'):
    root = create_demo(tmp_path/'data', patients=20, dimension=512, survival=True)
    bundle = Bundle(root)
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True,
        'alignment_policy': 'synthetic_native_dimension_fixture', 'checkpoint': str(WEIGHTS),
        'feature_preprocessing': 'Explicit synthetic512 Gaussian vectors, not extracted WSI',
        'feature_provenance': 'Synthetic software fixture', 'patient_order_provenance': 'Actual registered slide-ID order',
        'task_semantics': 'Synthetic patient survival, months/event1', 'concept_cohort': 'BRCA',
        'binning': {'policy': policy, 'provenance': 'Explicit synthetic software labels, not original clinical labels'}}
    if policy == 'provided_author_four_bin_labels':
        patients = bundle.rows.drop_duplicates('global_patient_id')[['global_patient_id', 'time', 'event']].copy()
        patients['bin'] = [i % 4 for i in range(len(patients))]
        patients.to_csv(root/'original_bins.csv', index=False)
        metadata['binning']['table'] = 'original_bins.csv'
    write_json(root/'metadata.json', metadata)
    manifest = prepare(bundle, 'metadata.json', 'convl_manifest.json')
    return root, bundle, manifest


def test_every_original_slide_in_a_true_patient_is_concatenated_in_registered_order(tmp_path):
    _, bundle, manifest = setup(tmp_path)
    assets = ConVLInputs(bundle, manifest)
    gid = next(gid for gid, rows in bundle.rows.groupby('global_patient_id') if len(rows) == 2)
    rows = bundle.rows.loc[bundle.rows.global_patient_id == gid].to_dict('records')
    features, label, censorship = assets.patient(rows)
    a, _ = bundle.load(rows[0]); b, _ = bundle.load(rows[1])
    assert features.shape == (len(a)+len(b), 512)
    import torch
    assert torch.equal(features, torch.cat([a, b]))
    assert label.tolist() == [assets.meta['patients'][gid]['bin']]
    assert censorship.tolist() == [1-int(rows[0]['event'])]
    with pytest.raises(ValueError, match='all patient slides'):
        assets.patient(rows[:1])
    with pytest.raises(ValueError, match='stable order'):
        assets.patient(rows[::-1])
    altered = [dict(row) for row in rows]
    altered[0]['event'] = 1-int(altered[0]['event'])
    with pytest.raises(ValueError, match='patient identity, event/time'):
        assets.patient(altered)


def test_exact_original_provided_four_bin_labels_are_not_replaced_by_fitted_quantiles(tmp_path):
    root, bundle, manifest = setup(tmp_path, 'provided_author_four_bin_labels')
    assets = ConVLInputs(bundle, manifest)
    expected = pd.read_csv(root/'original_bins.csv').set_index('global_patient_id')
    assert assets.meta['finite_time_cuts'] is None
    for patient, rows in bundle.rows.groupby('global_patient_id'):
        _, label, _ = assets.patient(rows.to_dict('records'))
        assert label.item() == int(expected.loc[patient, 'bin'])
    assert assets.meta['binning_files']


def test_val_test_times_never_change_train_quantiles_or_training_labels(tmp_path):
    root, bundle, manifest = setup(tmp_path)
    before = read_json(manifest)
    patients = set(bundle.rows.loc[bundle.rows.subset != 'train', 'patient_id'])
    slides = pd.read_csv(root/'slides.csv'); slides.loc[slides.patient_id.isin(patients), 'time'] += 10000
    slides.to_csv(root/'slides.csv', index=False)
    after = read_json(prepare(Bundle(root), 'metadata.json', 'new_manifest.json'))
    assert before['finite_time_cuts'] == after['finite_time_cuts']
    assert before['training_patients'] == after['training_patients']
    assert all(before['patients'][p] == after['patients'][p] for p in before['training_patients'])


def test_real_unrelated_fm_cannot_be_presented_as_aligned_conch(tmp_path):
    root, bundle, _ = setup(tmp_path)
    metadata = read_json(root/'metadata.json'); metadata['alignment_policy'] = 'native_conch'
    write_json(root/'incompatible.json', metadata)
    with pytest.raises(ValueError, match='actual aligned native512 CONCH'):
        prepare(bundle, 'incompatible.json', 'bad.json')
    assert not (root/'bad.json').exists()


def test_original_feature_mutation_and_manifest_overwrite_are_rejected(tmp_path):
    _, bundle, manifest = setup(tmp_path)
    assets = ConVLInputs(bundle, manifest)
    rows = bundle.rows.loc[bundle.rows.global_patient_id == bundle.rows.global_patient_id.iloc[0]].to_dict('records')
    assets.patient(rows)
    with h5py.File(bundle.root/rows[0]['path'], 'r+') as archive:
        archive['features'][0, 0] += .25
    with pytest.raises(ValueError, match='original full patch features'):
        assets.patient(rows)
    with pytest.raises(FileExistsError, match='preserve'):
        prepare(bundle, 'metadata.json', manifest)


def test_original_author_label_table_bytes_remain_bound_during_execution(tmp_path):
    root, bundle, manifest = setup(tmp_path, 'provided_author_four_bin_labels')
    assets = ConVLInputs(bundle, manifest)
    table = pd.read_csv(root/'original_bins.csv'); table.loc[0, 'bin'] = (table.loc[0, 'bin']+1) % 4
    table.to_csv(root/'original_bins.csv', index=False)
    with pytest.raises(ValueError, match='immutable manifest, concepts or metadata'):
        assets.check_unchanged()

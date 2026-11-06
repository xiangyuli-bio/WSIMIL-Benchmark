"""Native visual and patient contracts without fabricated pretrained checkpoints."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from milplus.feather_assets import native_lock
from milplus.feather_data import feature_population, prepare, verify_visual_metadata


def bundle(tmp_path, encoder='uni'):
    dim = {'uni': 1024, 'conch_v15': 768, 'uni_v2': 1536}[encoder]
    rows = []
    rng = np.random.default_rng(31)
    for subset in ['train', 'val', 'test']:
        for label in range(2):
            name = subset + str(label)
            path = tmp_path / (name + '.npy')
            np.save(path, rng.normal(size=(257 + label, dim)).astype('float32'))
            rows.append({'slide_id': name, 'patient_id': name, 'global_patient_id': name,
                'path': str(path), 'subset': subset, 'target': label})
    return SimpleNamespace(root=tmp_path, config={'task_id': 'explicit_synthetic_contracts'},
        task_type='classification', dim=dim, classes=['a', 'b'], fm=encoder, split='fixed',
        encoder={}, rows=pd.DataFrame(rows),
        load=lambda row: (torch.from_numpy(np.load(row['path'], allow_pickle=False)), None))


def visual(encoder):
    row = native_lock()['native_encoders'][encoder]
    return {'schema_version': 1, 'encoder_id': encoder,
        'feature_dim': {'uni': 1024, 'conch_v15': 768, 'uni_v2': 1536}[encoder],
        'model_id': row['model_id'], 'revision': row['revision'],
        'weights_sha256': next(iter(row['files'].values()))['sha256'],
        'extraction_provenance': 'Explicit software-test metadata; no real feature extraction asserted'}


@pytest.mark.parametrize('encoder', ['uni', 'uni_v2', 'conch_v15'])
def test_all_native_visual_spaces_bind_exact_published_encoder_identity(tmp_path, encoder):
    b = bundle(tmp_path, encoder)
    v = visual(encoder)
    assert verify_visual_metadata(b, encoder, v) == v
    records = feature_population(b)
    assert len(records) == 6
    assert all(v['dimension'] == b.dim and v['instances'] >= 257 for v in records.values())


@pytest.mark.parametrize('key', ['weights_sha256', 'revision', 'model_id', 'encoder_id', 'extraction_provenance'])
def test_shape_only_or_wrong_visual_identity_is_rejected(tmp_path, key):
    b = bundle(tmp_path)
    v = visual('uni')
    v[key] = '' if key == 'extraction_provenance' else 'unrelated'
    with pytest.raises(ValueError, match='equal dimensions'):
        verify_visual_metadata(b, 'uni', v)


def test_titan_conch_file_cannot_silently_replace_standalone_conch_features(tmp_path):
    b = bundle(tmp_path, 'conch_v15')
    v = visual('conch_v15')
    v['weights_sha256'] = native_lock()['other_CONCHv15_file_identity_not_proven_feature_equivalent']['sha256']
    with pytest.raises(ValueError, match='equal dimensions'):
        verify_visual_metadata(b, 'conch_v15', v)


def test_global_patient_leak_and_identical_bags_in_different_paths_rejected(tmp_path):
    b = bundle(tmp_path)
    b.rows.loc[b.rows.subset == 'test', 'global_patient_id'] = 'train0'
    with pytest.raises(ValueError, match='patient identity'):
        feature_population(b)
    b.rows['global_patient_id'] = b.rows.slide_id
    train = Path(b.rows.iloc[0].path)
    test = Path(b.rows.iloc[-1].path)
    test.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match='Byte-identical'):
        feature_population(b)


def test_missing_original_asset_never_creates_a_complete_native_input_manifest(tmp_path):
    b = bundle(tmp_path)
    target = tmp_path / 'binding.json'
    with pytest.raises(FileNotFoundError):
        prepare(b, {'schema_version': 1, 'pretrained_assets': 'missing.json',
            'visual_extraction_manifest': 'metadata.json', 'provenance': 'test'}, target)
    assert not target.exists()

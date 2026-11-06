
import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from milplus.data import Bundle, load_array
from milplus.demo import create_demo
from milplus.metrics import harrell_c_index, paired_bootstrap
from milplus.registry import methods, profiles, require_runnable


@pytest.fixture
def bundle_path(tmp_path):
    return create_demo(tmp_path / "task", patients=20)


def test_catalog_scope():
    assert len(methods()) == 164
    assert len({m['id'] for m in methods()}) == 164
    assert len(profiles()) >= 12
    for method in methods():
        if not method['profiles']:
            with pytest.raises(ValueError, match="pending_adapter"):
                require_runnable(method['id'])


def test_deep_bundle_check(bundle_path):
    b = Bundle(bundle_path)
    report = b.check(deep=True, require_coords=True)
    assert report['patients'] == 20
    assert report['slides'] == 30


def test_patient_leakage_rejected(bundle_path):
    p = bundle_path / 'split.csv'
    f = pd.read_csv(p)
    f = pd.concat([f, f.iloc[[0]].assign(subset='test')])
    f.to_csv(p, index=False)
    with pytest.raises(ValueError, match='leakage'):
        Bundle(bundle_path)


def test_global_alias_leakage(bundle_path):
    b = Bundle(bundle_path)
    f = pd.read_csv(bundle_path / 'slides.csv')
    f['global_patient_id'] = f.patient_id
    pair = [b.records('train')[0]['patient_id'], b.records('test')[0]['patient_id']]
    f.loc[f.patient_id.isin(pair), 'global_patient_id'] = 'alias'
    f.to_csv(bundle_path / 'slides.csv', index=False)
    with pytest.raises(ValueError, match='Global patient'):
        Bundle(bundle_path)


def test_coordinate_mismatch(bundle_path):
    b = Bundle(bundle_path)
    p = bundle_path / b.records('train')[0]['path']
    with h5py.File(p, 'a') as f:
        del f['coords']
        f['coords'] = np.zeros((2, 2))
    with pytest.raises(ValueError, match='alignment'):
        b.check(deep=True)


def test_nan_and_dimension_rejected(bundle_path):
    b = Bundle(bundle_path)
    row = b.records('train')[0]
    with h5py.File(bundle_path / row['path'], 'a') as f:
        f['features'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        b.load(row)


def test_empty_features_rejected(bundle_path):
    b = Bundle(bundle_path)
    row = b.records('train')[0]
    with h5py.File(bundle_path / row['path'], 'w') as f:
        f['features'] = np.zeros((0, 16))
    with pytest.raises(ValueError, match='nonempty'):
        b.load(row)


@pytest.mark.parametrize('subset', ['train', 'val', 'test'])
@pytest.mark.parametrize('key', ['features', 'coords'])
def test_complex_inputs_never_silently_discard_imaginary_values(bundle_path, subset, key):
    import warnings
    bundle = Bundle(bundle_path)
    row = bundle.records(subset)[0]
    with h5py.File(bundle_path/row['path'], 'r+') as h:
        values = h[key][:].astype(np.complex64)
        values[0, 0] += 1j
        del h[key]
        h[key] = values
    with warnings.catch_warnings(record=True) as recorded, pytest.raises(ValueError, match='nonreal|alignment'):
        bundle.check(deep=True)
    assert not recorded


def test_safe_feature_formats(tmp_path):
    x = np.arange(20).reshape(5, 4).astype(np.float32)
    np.save(tmp_path/'f.npy', x)
    torch.save(torch.from_numpy(x), tmp_path/'f.pt')
    torch.save({'features': torch.from_numpy(x)}, tmp_path/'dict.pt')
    for name in ('f.npy', 'f.pt', 'dict.pt'):
        np.testing.assert_array_equal(load_array(tmp_path/name), x)


def test_bootstrap_pairing_and_cluster_multiplicity():
    f = pd.DataFrame({'slide_id': [f's{i}' for i in range(8)], 'global_patient_id': [f'p{i//2}' for i in range(8)],
                      'target': [0,0,1,1,0,0,1,1], 'subset': ['test']*8,
                      'p_0': [.8,.9,.2,.1,.7,.6,.4,.2], 'p_1': [.2,.1,.8,.9,.3,.4,.6,.8]})
    result = paired_bootstrap(f, f.copy(), 2, unit='patient', repetitions=40)
    assert result['ci95'] == [0, 0]
    with pytest.raises(ValueError, match='Duplicate'):
        paired_bootstrap(pd.concat([f, f]), f, 2)
    with pytest.raises(ValueError, match='identity'):
        paired_bootstrap(f, f.assign(target=1-f.target), 2)


def test_cindex_direction():
    assert harrell_c_index([1,2,3], [1,1,1], [3,2,1]) == 1
    assert harrell_c_index([1,2,3], [1,1,1], [1,2,3]) == 0
    assert harrell_c_index([1,2], [0,0], [1,2]) is None

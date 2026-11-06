import copy
import json

import h5py
import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.mgtrans_data import physical_alignment, prepare, read_pair
from milplus.mgtrans_demo import create_mgtrans_demo


@pytest.fixture
def data(tmp_path):
    return create_mgtrans_demo(tmp_path/'data', patients=18, dimension=1024, classes=2, roots=16)


def test_actual_scale_order_geometry_and_deep_inventory(data):
    bundle = Bundle(data)
    row = bundle.rows.iloc[0].to_dict()
    low, coarse, high, fine, mask = bundle.mgtrans.load(row)
    assert low.shape == (16, 1024) and high.shape == (64, 1024)
    assert mask.dtype == torch.bool and mask.shape == (16, 64)
    assert (mask.sum(0) == 1).all() and (mask.sum(1) == 4).all()
    for i, point in enumerate(coarse):
        covered = fine[mask[i]]
        assert ((covered >= point) & (covered < point+2048)).all()
    assert bundle.check(deep=True)['feature_inventory_sha256']


@pytest.mark.parametrize('defect', ['duplicate', 'missing_level', 'dimension', 'nonfinite', 'orphan', 'grid', 'primary', 'bounds'])
def test_measured_scale_corruption_is_rejected(data, defect):
    bundle = Bundle(data); row = bundle.rows.iloc[0].to_dict()
    record = bundle.mgtrans.meta['slides'][row['slide_id']]
    path = record['path']
    with h5py.File(path, 'r+') as archive:
        if defect == 'duplicate':
            archive['coords_high'][1] = archive['coords_high'][0]
        elif defect == 'missing_level':
            del archive['features_high']
        elif defect == 'dimension':
            values = archive['features_high'][:, :-1]; del archive['features_high']; archive['features_high'] = values
        elif defect == 'nonfinite':
            archive['features_high'][0, 0] = float('nan')
        elif defect == 'orphan':
            archive['coords_high'][0] = np.asarray([32768, 0])
            row.pop('width_level0', None); row.pop('height_level0', None)
        elif defect == 'grid':
            archive['coords_high'][0] = np.asarray([13, 17])
        elif defect == 'bounds':
            row['width_level0'] = 100
        else:
            archive['features_low'][0, 0] += 1
    with pytest.raises(ValueError):
        read_pair(bundle, row, path, bundle.mgtrans.schema)
    if defect != 'bounds':
        with pytest.raises(ValueError, match='changed'):
            bundle.mgtrans.load(row)


@pytest.mark.parametrize('field', ['subset', 'target', 'global_patient_id'])
def test_manifest_does_not_accept_changed_task_identity(data, field):
    path = data/'mgtrans/manifest.json'; content = json.loads(path.read_text())
    first = next(iter(content['slides']))
    content['slides'][first][field] = 12345 if field == 'target' else 'changed'
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='task label, patient mapping or split'):
        Bundle(data)


def test_alignment_is_physical_and_permutation_equivariant_and_budgeted():
    coarse = np.asarray([[2048, 0], [0, 0]], dtype=np.int64)
    fine = np.asarray([[1024, 0], [2048, 0], [0, 0], [3072, 1024]], dtype=np.int64)
    expected = np.asarray([[False, True, False, True], [True, False, True, False]])
    np.testing.assert_array_equal(physical_alignment(coarse, fine, 2048, 1024), expected)
    with pytest.raises(ValueError, match='memory'):
        physical_alignment(coarse, fine, 2048, 1024, maximum_bytes=7)


@pytest.mark.parametrize('defect', ['scale', 'coordinate_frame', 'provenance'])
def test_prepare_rejects_ambiguous_scale_or_coordinate_contract(data, defect):
    bundle = Bundle(data); schema = copy.deepcopy(bundle.mgtrans.schema)
    if defect == 'scale':
        schema['levels'][1]['patch_size_level0'] = 2048
    elif defect == 'coordinate_frame':
        schema['coordinate_frame'] = 'sequence_pseudo_grid'
    else:
        schema['weights_provenance'] = ''
    with pytest.raises(ValueError):
        prepare(bundle, data/'mgtrans_index.csv', schema, data/'other_manifest.json')

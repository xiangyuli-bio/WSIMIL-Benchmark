import copy
import json

import h5py
import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.kmgmamba_data import prepare, read_pair
from milplus.kmgmamba_demo import create_kmgmamba_demo


@pytest.fixture
def data(tmp_path):
    return create_kmgmamba_demo(tmp_path/'data', patients=18, dimension=1024, classes=2, roots=16)


def test_actual_scale_order_geometry_and_deep_inventory(data):
    bundle = Bundle(data)
    row = bundle.rows.iloc[0].to_dict()
    low, coarse, high, fine = bundle.kmgmamba.load(row)
    assert low.shape == (16, 1024) and high.shape == (64, 1024)
    assert coarse.shape == (16, 2) and fine.shape == (64, 2)
    assert not torch.equal(low, high[:16])
    assert bundle.check(deep=True)['feature_inventory_sha256']


@pytest.mark.parametrize('defect', ['duplicate', 'missing_level', 'dimension', 'nonfinite', 'primary', 'bounds'])
def test_measured_scale_corruption_is_rejected(data, defect):
    bundle = Bundle(data); row = bundle.rows.iloc[0].to_dict()
    record = bundle.kmgmamba.meta['slides'][row['slide_id']]
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
        elif defect == 'bounds':
            row['width_level0'] = 100
        else:
            archive['features_low'][0, 0] += 1
    with pytest.raises(ValueError):
        read_pair(bundle, row, path, bundle.kmgmamba.schema)
    if defect != 'bounds':
        with pytest.raises(ValueError, match='changed'):
            bundle.kmgmamba.load(row)


@pytest.mark.parametrize('field', ['subset', 'target', 'global_patient_id'])
def test_manifest_does_not_accept_changed_task_identity(data, field):
    path = data/'kmgmamba/manifest.json'; content = json.loads(path.read_text())
    first = next(iter(content['slides']))
    content['slides'][first][field] = 12345 if field == 'target' else 'changed'
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='task label, patient mapping or split'):
        Bundle(data)


@pytest.mark.parametrize('defect', ['scale', 'coordinate_frame', 'provenance'])
def test_prepare_rejects_ambiguous_scale_or_coordinate_contract(data, defect):
    bundle = Bundle(data); schema = copy.deepcopy(bundle.kmgmamba.schema)
    if defect == 'scale':
        schema['levels'][1]['patch_size_level0'] = 2048
    elif defect == 'coordinate_frame':
        schema['coordinate_frame'] = 'sequence_pseudo_grid'
    else:
        schema['weights_provenance'] = ''
    with pytest.raises(ValueError):
        prepare(bundle, data/'kmgmamba_index.csv', schema, data/'other_manifest.json')

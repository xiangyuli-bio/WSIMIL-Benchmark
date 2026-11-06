"""Actual overlap, group-token, provenance and tamper rejection contracts."""
import copy
import shutil

import h5py
import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.io import write_json
from milplus.sammil_data import SAMMILAssets, read_groups, validate_schema
from milplus.sammil_demo import create_sammil_demo
from milplus.upstream import load_definitions


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    return create_sammil_demo(tmp_path_factory.mktemp('sam_data')/'data',
        patients=18, dimension=32, classes=2, patches=32)


def test_original_process_info_overlap_and_sentinel_tokens_and_means(data):
    bundle = Bundle(data); row = bundle.records('train')[0]; value = bundle.sammil.load(row)
    x, coords = bundle.load(row)
    source = load_definitions('sammil', 'WSI_preprocess/03_extract_sam_info.py',
                              ['process_info'], dict(np=np))
    combined = value['coords'].numpy()
    groups = {name: coords[indices].numpy() for name, indices in zip(value['group_ids'], value['members'])}
    _, flags, areas = source.process_info(combined, [{'area': a} for a in value['areas']], groups)
    assert np.array_equal(flags, value['is_group_feat'].numpy())
    assert np.array_equal(areas, value['relative_area'].numpy())
    assert torch.equal(x, value['features'][:len(x)])
    for index, members in enumerate(value['members']):
        assert np.array_equal(value['features'][len(x)+index].numpy(), np.mean(x[members].numpy(), axis=0))
    assert value['relative_area'][10] == 7000  # Overlap uses the minimum area.
    assert torch.equal(value['relative_area'][-3:], torch.zeros(3, dtype=torch.int64))
    assert bundle.config['synthetic'] is True and bundle.sammil.meta['synthetic'] is True


@pytest.mark.parametrize('corruption', ['coordinate_order', 'unknown_member', 'duplicate_member',
    'empty_group', 'invalid_area', 'noninteger_area', 'missing_group'])
def test_invalid_measured_membership_rejected(data, tmp_path, corruption):
    bundle = Bundle(data); row = bundle.records('train')[0]
    path = tmp_path/'bad.h5'; shutil.copy2(bundle.sammil.meta['slides'][row['slide_id']]['path'], path)
    with h5py.File(path, 'r+') as archive:
        if corruption == 'coordinate_order':
            archive['coords'][:] = archive['coords'][:][::-1]
        elif corruption == 'unknown_member':
            archive['groups/seg_0'][0] = [999999, 999999]
        elif corruption == 'duplicate_member':
            archive['groups/seg_0'][1] = archive['groups/seg_0'][0]
        elif corruption == 'empty_group':
            del archive['groups/seg_0']; archive['groups/seg_0'] = np.empty((0, 2), dtype=np.int64)
            archive['groups/seg_0'].attrs['area'] = 7000
        elif corruption == 'invalid_area':
            archive['groups/seg_0'].attrs['area'] = 1
        elif corruption == 'noninteger_area':
            archive['groups/seg_0'].attrs['area'] = 7000.5
        elif corruption == 'missing_group':
            del archive['groups']
    with pytest.raises(ValueError):
        read_groups(bundle, row, path)


@pytest.mark.parametrize('key,value', [('sam_model', 'vit_b'), ('sam_weights_sha256', 'unknown'),
    ('membership_provenance', ''), ('group_feature', 'invented_spatial_grid'),
    ('overlap', 'largest_mask')])
def test_schema_refuses_silent_input_substitution(data, key, value):
    bundle = Bundle(data); schema = copy.deepcopy(bundle.sammil.schema); schema[key] = value
    with pytest.raises(ValueError):
        validate_schema(bundle, schema)


@pytest.mark.parametrize('corruption', ['label', 'split', 'patient', 'synthetic', 'preparer', 'members'])
def test_manifest_and_identity_tampering_rejected(data, tmp_path, corruption):
    bundle = Bundle(data); meta = copy.deepcopy(bundle.sammil.meta)
    slide = next(iter(meta['slides']))
    if corruption == 'label': meta['slides'][slide]['target'] += 1
    if corruption == 'split': meta['slides'][slide]['subset'] = 'test'
    if corruption == 'patient': meta['slides'][slide]['global_patient_id'] = 'another_patient'
    if corruption == 'synthetic': meta['synthetic'] = False
    if corruption == 'preparer': meta['preparer_sha256'] = '0'*64
    if corruption == 'members': meta['slides'][slide]['groups_sha256'] = '0'*64
    path = tmp_path/'manifest.json'; write_json(path, meta)
    with pytest.raises(ValueError):
        assets = SAMMILAssets(bundle, path)
        assets.load(next(row for row in bundle.rows.to_dict('records') if row['slide_id'] == slide))

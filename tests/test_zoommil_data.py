"""Real hierarchy coverage/coordinate contracts; all fixtures stay synthetic."""
import copy

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from milplus.data import Bundle
from milplus.io import read_json
from milplus.zoommil_data import ZoomMILAssets, prepare, read_hierarchy, schema_contract
from milplus.zoommil_demo import create_zoommil_demo


@pytest.fixture
def pyramid(tmp_path):
    root = create_zoommil_demo(tmp_path/'data', dimension=32, roots=4)
    bundle = Bundle(root)
    schema = read_json(root/'zoommil_schema.json')
    row = bundle.records('train')[0]
    path = root/'pyramids'/f'{row["slide_id"]}.h5'
    return bundle, schema, row, path


def test_zoommil_complete_native_hierarchy_and_inventory(pyramid):
    bundle, schema, row, path = pyramid
    features = bundle.zoommil.load(row)
    assert [tuple(value.shape) for value in features] == [(4, 32), (16, 32), (256, 32)]
    assert torch.equal(features[0], bundle.load(row)[0])
    assert bundle.zoommil.depths == [0, 1, 3]
    assert schema['synthetic'] is True
    report = bundle.check(deep=True)
    assert report['feature_inventory_sha256']
    assert path.is_file()


def test_zoommil_actual_coordinate_reordering_preserves_every_feature(pyramid):
    bundle, schema, row, path = pyramid
    before = bundle.zoommil.load(row)
    with h5py.File(path, 'a') as archive:
        for level in schema['levels'][1:]:
            feature = archive[level['feature_key']][:]
            coords = archive[level['coords_key']][:]
            order = np.random.default_rng(36).permutation(len(feature))
            archive[level['feature_key']][:] = feature[order]
            archive[level['coords_key']][:] = coords[order]
    actual, _, counts = read_hierarchy(bundle, row, path, schema)
    assert counts == [4, 16, 256]
    assert all(torch.equal(a, b) for a, b in zip(actual, before))
    with pytest.raises(ValueError, match='changed'):
        bundle.zoommil.load(row)  # old bound preparation cannot be reused


@pytest.mark.parametrize('attribute', [False, True])
def test_zoommil_physical_trident_style_coordinates_match_source_exactly(pyramid, attribute):
    bundle, schema, row, path = pyramid
    before = bundle.zoommil.load(row)
    schema = copy.deepcopy(schema)
    schema.update(coordinate_frame='level0_top_left_pixels', coarse_patch_size_level0=256.,
                  patch_size_provenance='Synthetic known256pixel field, not real WSI calibration')
    if attribute:
        schema.pop('coarse_patch_size_level0')
        schema['coarse_patch_size_attribute'] = 'actual_coarse_side'
    with h5py.File(path, 'a') as archive:
        archive.attrs['actual_coarse_side'] = 256.
        for level in schema['levels']:
            coords = archive[level['coords_key']][:]
            xy = coords[:, [1, 0]].astype(np.float64)
            for depth in range((coords.shape[1]-2)//2):
                xy += coords[:, [3+2*depth, 2+2*depth]]/2**(depth+1)
            del archive[level['coords_key']]
            archive[level['coords_key']] = xy*256.
    actual, _, counts = read_hierarchy(bundle, row, path, schema)
    assert counts == [4, 16, 256]
    assert all(torch.equal(a, b) for a, b in zip(actual, before))
    with h5py.File(path, 'a') as archive:
        archive[schema['levels'][2]['coords_key']][0, 0] += 1.
    with pytest.raises(ValueError, match='physical patch grid'):
        read_hierarchy(bundle, row, path, schema)


@pytest.mark.parametrize('bad', ['duplicate', 'missing_child', 'wrong_root', 'quadrant', 'dimension', 'base_features', 'float_addresses'])
def test_zoommil_rejects_false_incomplete_or_misaligned_hierarchies(pyramid, bad):
    bundle, schema, row, path = pyramid
    key = schema['levels'][2]['coords_key']
    with h5py.File(path, 'a') as archive:
        if bad == 'duplicate':
            archive[key][1] = archive[key][0]
        elif bad == 'wrong_root':
            archive[key][0, 0] = 10000
        elif bad == 'quadrant':
            archive[key][0, 2] = 2
        elif bad == 'base_features':
            archive[schema['levels'][0]['feature_key']][0, 0] += 4.
        elif bad == 'float_addresses':
            value = archive[key][:].astype(np.float32)
            del archive[key]; archive[key] = value
        else:
            key = schema['levels'][2]['feature_key']
            value = archive[key][:]
            value = value[:-1] if bad == 'missing_child' else value[:, :-1]
            del archive[key]; archive[key] = value
    with pytest.raises(ValueError):
        read_hierarchy(bundle, row, path, schema)


@pytest.mark.parametrize('bad', ['ratio', 'provenance', 'synthetic', 'keys', 'physical_no_size'])
def test_zoommil_rejects_unknown_scale_provenance_and_calibration(pyramid, bad):
    bundle, schema, _, _ = pyramid
    schema = copy.deepcopy(schema)
    if bad == 'ratio':
        schema['levels'][2]['magnification'] = 7.
    elif bad == 'provenance':
        schema['coordinate_provenance'] = ''
    elif bad == 'synthetic':
        schema['synthetic'] = False
    elif bad == 'keys':
        schema['levels'][2]['feature_key'] = schema['levels'][1]['feature_key']
    else:
        schema['coordinate_frame'] = 'level0_top_left_pixels'
    with pytest.raises(ValueError):
        schema_contract(bundle, schema)


def test_zoommil_prepare_rejects_labels_and_duplicate_physical_slide_paths(pyramid):
    bundle, schema, _, _ = pyramid
    frame = pd.read_csv(bundle.root/'zoommil_index.csv')
    frame['clinical_label'] = 'forbidden_in_geometry_index'
    path = bundle.root/'bad.csv'; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='no outcomes'):
        prepare(bundle, path, schema, bundle.root/'bad_manifest.json')
    frame = frame.drop(columns='clinical_label')
    frame.loc[1, 'path'] = frame.loc[0, 'path']; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='one physical'):
        prepare(bundle, path, schema, bundle.root/'bad_manifest.json')


def test_zoommil_manifest_rejects_changed_fm_and_inputs(pyramid):
    bundle, _, row, _ = pyramid
    bundle.encoder['weights'] = 'different_FM_weights'
    with pytest.raises(ValueError, match='encoder identity'):
        ZoomMILAssets(bundle)
    bundle.encoder['weights'] = 'synthetic'
    with h5py.File(bundle.root/row['path'], 'a') as archive:
        archive['features'][0, 0] += 1.
    with pytest.raises(ValueError, match='changed'):
        bundle.zoommil.load(row)

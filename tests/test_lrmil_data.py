import copy
import json

import h5py
import pytest
import torch

from milplus.data import Bundle
from milplus.lrmil_data import LRMILAssets, prepare, read_slide
from milplus.lrmil_demo import create_lrmil_demo


@pytest.fixture
def inputs(tmp_path):
    root, manifest = create_lrmil_demo(tmp_path/'data', patients=18)
    bundle = Bundle(root)
    return bundle, LRMILAssets(bundle, manifest)


def test_full_measured_rgb_layers_and_exact_cross_resolution_regions(inputs):
    bundle, assets = inputs
    row = bundle.records('train')[0]
    result = assets.load(row)
    assert bundle.config['synthetic'] is True
    assert result['images'].shape == (2, 256, 256, 3) and result['images'].dtype == torch.uint8
    assert result['teacher_final'].shape == (32, 768)
    assert {key: value.shape for key, value in result['teacher_layers'].items()} == {
        0: (32, 768), 1: (32, 768), 10: (32, 768)}
    assert result['links_5x_to_20x'] == [list(range(16)), list(range(16, 32))]
    assert not torch.equal(result['teacher_layers'][0], result['teacher_layers'][1])
    assert assets.inventory(row, deep=True)['low_nodes'] == 2


def test_streamed_patch_and_low_reads_equal_the_entire_original_physical_input(inputs):
    bundle, assets = inputs
    row = bundle.records('train')[0]
    full = assets.load(row)
    torch.testing.assert_close(assets.high(row), full['teacher_final'], atol=0, rtol=0)
    torch.testing.assert_close(assets.low(row), full['images'], atol=0, rtol=0)
    for index in range(2):
        patch = assets.patch(row, index)
        links = full['links_5x_to_20x'][index]
        torch.testing.assert_close(patch['image'], full['images'][index], atol=0, rtol=0)
        torch.testing.assert_close(patch['coords_5x'], full['coords_5x'][index], atol=0, rtol=0)
        torch.testing.assert_close(patch['coords_20x'], full['coords_20x'][links], atol=0, rtol=0)
        torch.testing.assert_close(patch['teacher_final'], full['teacher_final'][links], atol=0, rtol=0)
        for layer in [0, 1, 10]:
            torch.testing.assert_close(patch['teacher_layers'][layer], full['teacher_layers'][layer][links], atol=0, rtol=0)


def test_streaming_reader_rechecks_real_archive_when_file_changes(inputs):
    bundle, assets = inputs
    row = bundle.records('train')[0]
    assets.patch(row, 0)
    path = assets.meta['slides'][row['slide_id']]['path']
    with h5py.File(path, 'r+') as archive:
        archive['images'][0, 0, 0, 0] = (int(archive['images'][0, 0, 0, 0])+1)%256
    for operation in [lambda: assets.low(row), lambda: assets.patch(row, 0), lambda: assets.high(row)]:
        with pytest.raises(ValueError, match='changed'):
            operation()


@pytest.mark.parametrize('defect', ['missing_layer', 'wrong_dimension', 'duplicate_low',
                                   'primary_disagrees', 'nonfinite', 'wrong_mpp', 'unmatched_low'])
def test_actual_intermediate_or_physical_corruption_is_rejected(inputs, defect):
    bundle, assets = inputs
    row = bundle.records('train')[0]
    path = assets.meta['slides'][row['slide_id']]['path']
    with h5py.File(path, 'r+') as archive:
        if defect == 'missing_layer':
            del archive['layer_10']
        elif defect == 'wrong_dimension':
            values = archive['layer_1'][:, :-1]
            del archive['layer_1']
            archive['layer_1'] = values
        elif defect == 'duplicate_low':
            archive['coords_low'][1] = archive['coords_low'][0]
        elif defect == 'primary_disagrees':
            archive['teacher_final'][0, 0] += 1
        elif defect == 'nonfinite':
            archive['layer_0'][0, 0] = float('nan')
        elif defect == 'wrong_mpp':
            archive.attrs['low_mpp'] = 4.
        else:
            archive['coords_low'][1] = [4096, 0]
    with pytest.raises(ValueError):
        read_slide(bundle, row, path, assets.schema)
    with pytest.raises(ValueError, match='changed'):
        assets.load(row)


@pytest.mark.parametrize('key', ['global_patient_id', 'target', 'subset'])
def test_no_reuse_after_patient_label_or_split_change(inputs, key):
    bundle, assets = inputs
    value = json.loads(assets.path.read_text())
    first = next(iter(value['slides']))
    value['slides'][first][key] = 12345 if key == 'target' else 'changed'
    assets.path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='patient mapping or split'):
        LRMILAssets(bundle, assets.path)


def test_true40x_level0_coords_convert_to_source20x_without_resizing_images(inputs):
    bundle, assets = inputs
    row = bundle.records('train')[0]
    expected = assets.load(row)
    path = assets.meta['slides'][row['slide_id']]['path']
    with h5py.File(path, 'r+') as archive:
        archive.attrs['level0_mpp'] = .25
        archive['coords_low'][:] *= 2
        archive['coords_high'][:] *= 2
    primary = bundle.root/row['path']
    with h5py.File(primary, 'r+') as archive:
        archive['coords'][:] *= 2
    row = {**row, 'width_level0': 2*row['width_level0'], 'height_level0': 2*row['height_level0']}
    actual = read_slide(bundle, row, path, assets.schema)
    torch.testing.assert_close(actual['coords_5x'], expected['coords_5x'], atol=0, rtol=0)
    torch.testing.assert_close(actual['coords_20x'], expected['coords_20x'], atol=0, rtol=0)
    torch.testing.assert_close(actual['images'], expected['images'], atol=0, rtol=0)
    assert actual['links_5x_to_20x'] == expected['links_5x_to_20x']


@pytest.mark.parametrize('key,value', [('teacher_layers', [0, 1]), ('teacher_representation', 'CONCH_contrastive512'),
                                      ('image_provenance', ''), ('coordinate_frame', 'pseudo_grid')])
def test_missing_intermediates_or_ambiguous_preparation_is_rejected(inputs, key, value):
    bundle, assets = inputs
    schema = {**copy.deepcopy(assets.schema), key: value}
    with pytest.raises(ValueError):
        prepare(bundle, bundle.root/'lrmil_index.csv', schema, bundle.root/'new.json')

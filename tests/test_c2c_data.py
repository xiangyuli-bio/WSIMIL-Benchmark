"""RGB identity, historical checkpoints and source decoder/packing contracts."""
import copy
import shutil

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from milplus.c2c_assets import ASSETS, default_root, verify
from milplus.c2c_data import C2CAssets, describe, validate_schema
from milplus.c2c_demo import create_c2c_demo
from milplus.c2c_pack import pack
from milplus.data import Bundle
from milplus.io import write_json


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    return create_c2c_demo(tmp_path_factory.mktemp('c2c_rgb')/'data', patients=10, patches=16, patch_pixels=64)


def test_original_rgb_shape_patch_identity_and_frozen_feature_refusal(data):
    bundle = Bundle(data); row = bundle.records('train')[0]
    with bundle.c2c.open(row) as pixels:
        assert pixels.shape == (16, 64, 64, 3) and pixels.dtype == np.uint8
    assert bundle.c2c.count(row) == 16 and bundle.c2c.inventory(row, True)['shape'] == [16, 64, 64, 3]
    assert bundle.check(deep=True)['slides'] == 10
    with pytest.raises(ValueError, match='not frozen'):
        bundle.load(row)


@pytest.mark.parametrize('key,value', [('color', 'BGR'), ('normalization', 'imagenet_mean_std'),
    ('dtype', 'float32'), ('patch_pixels', 224), ('coordinate_frame', 'invented_grid'),
    ('image_provenance', '')])
def test_incompatible_image_preprocessing_is_not_silently_substituted(data, key, value):
    bundle = Bundle(data); schema = copy.deepcopy(bundle.c2c.schema); schema[key] = value
    with pytest.raises(ValueError):
        validate_schema(bundle, schema)
        describe(bundle, bundle.records('train')[0], schema)


@pytest.mark.parametrize('corruption', ['duplicate_patch', 'missing_patch', 'float_pixels', 'features_only', 'external_pixels'])
def test_invalid_original_rgb_or_identity_rejected(data, tmp_path, corruption):
    bundle = Bundle(data); row = copy.deepcopy(bundle.records('train')[0])
    path = tmp_path/'bad.h5'; shutil.copy2(row['path'], path); row['path'] = str(path)
    with h5py.File(path, 'r+') as archive:
        if corruption == 'duplicate_patch':
            archive['patch_ids'][1] = archive['patch_ids'][0]
        if corruption == 'missing_patch':
            ids = archive['patch_ids'][:-1]; del archive['patch_ids']; archive['patch_ids'] = ids
        if corruption == 'float_pixels':
            shape = archive['pixels'].shape; del archive['pixels']; archive['pixels'] = np.zeros(shape, dtype=np.float32)
        if corruption == 'features_only':
            del archive['pixels']; archive['features'] = np.zeros((16, 1024), dtype=np.float32)
        if corruption == 'external_pixels':
            del archive['pixels']; archive['pixels'] = h5py.ExternalLink(bundle.records('train')[0]['path'], '/pixels')
    with pytest.raises(ValueError):
        describe(bundle, row, bundle.c2c.schema)


@pytest.mark.parametrize('field', ['patient', 'label', 'split', 'assets', 'synthetic'])
def test_patient_label_split_checkpoint_and_synthetic_identities_bound(data, tmp_path, field):
    bundle = Bundle(data); meta = copy.deepcopy(bundle.c2c.meta); slide = next(iter(meta['slides']))
    if field == 'patient': meta['slides'][slide]['global_patient_id'] = 'another_patient'
    if field == 'label': meta['slides'][slide]['target'] += 1
    if field == 'split': meta['slides'][slide]['subset'] = 'test'
    if field == 'assets': meta['assets']['resnet18-5c106cde.pth'] = 'f'*64
    if field == 'synthetic': meta['synthetic'] = False
    path = tmp_path/'changed.json'; write_json(path, meta)
    with pytest.raises(ValueError): C2CAssets(bundle, path)


def test_no_newer_or_random_initialization_fallback(tmp_path):
    for name in ASSETS:
        if name.endswith('.pth'):
            (tmp_path/name).write_bytes(b'not the original old checkpoint')
        else:
            (tmp_path/name).symlink_to(default_root()/name)
    with pytest.raises(ValueError, match='exact historical'):
        verify(tmp_path)


def test_original_opencv_decoder_pack_preserves_rgb_and_csv_row_order(tmp_path):
    records, expected = [], []
    for n in range(3):
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2] = 20+n, 80+2*n, 160+n
        path = tmp_path/f'patch_{2-n}.png'
        assert cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        records.append({'wsi': 'slide/with_special_path', 'path': path.name}); expected.append(pixels)
    table = tmp_path/'source.csv'; pd.DataFrame(records).to_csv(table, index=False)
    index = pd.read_csv(pack(table, tmp_path/'packed'))
    assert index.slide_id.tolist() == ['slide/with_special_path']
    with h5py.File(index.path[0], 'r') as archive:
        assert np.array_equal(archive['pixels'][:], np.stack(expected))
        assert archive['patch_ids'].asstr()[:].tolist() == [str(tmp_path/row['path']) for row in records]
    assert len(list((tmp_path/'packed').glob('*.h5'))) == 1
    with pytest.raises(ValueError, match='Preserve'):
        pack(table, tmp_path/'packed')

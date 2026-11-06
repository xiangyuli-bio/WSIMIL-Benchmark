"""Actual synthetic TIFF I/O parity; no fixtures are natural WSI evidence."""
import ast
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image
import pytest
import tifffile
import torch
from torch.utils import data
from torchvision import transforms

from milplus.io import digest_file, write_json
from milplus.raw_region_cache import KIND, MANIFEST_KIND, coordinate_sha256, fingerprint
from milplus.raw_regions import RawRegions
from milplus.upstream import pinned_text, verified_source


@pytest.fixture
def cached(tmp_path):
    rng = np.random.default_rng(172)
    pixels = rng.integers(0, 256, (672, 896, 3), dtype=np.uint8)
    wsi = tmp_path / 'actual_synthetic.tiff'
    tifffile.imwrite(wsi, pixels, tile=(224, 224), photometric='rgb', metadata=None)
    coords = np.asarray([(x * 224, y * 224) for y in range(3) for x in range(4)], dtype=np.int64)
    np.save(tmp_path / 'coords.npy', coords)
    path = tmp_path / 'original_rgb.h5'
    pixel_hash = hashlib.sha256()
    with openslide.OpenSlide(str(wsi)) as slide, h5py.File(path, 'x') as archive:
        archive.attrs['synthetic'] = True
        images = archive.create_dataset('imgs', (12, 256, 256, 3), dtype='uint8', compression='lzf')
        archive['coords'] = coords
        ids = []
        for i, coord in enumerate(coords):
            region = np.asarray(slide.read_region(tuple(coord), 0, (256, 256)).convert('RGB'))
            images[i] = region
            pixel_hash.update(region.tobytes())
            ids.append(hashlib.sha256(json.dumps(['one', i, 0, int(coord[0]), int(coord[1]), 256], separators=(',', ':')).encode()).hexdigest())
        archive['patch_ids'] = np.asarray(ids, dtype='S64')
        quickhash = slide.properties['openslide.quickhash-1']
    source = {'kind': 'actual_wsi_read_region_original_clam_coordinates', 'source_slide_id': 'one',
        'source_wsi': str(wsi), 'source_quickhash1': quickhash, 'source_fingerprint_before': fingerprint(wsi),
        'patch_level': 0, 'patch_pixels': 256, 'actual_level0_dimensions': [896, 672],
        'actual_level_dimensions': [896, 672], 'actual_openslide_downsample': 1.0,
        'coordinates_int64_le_sha256': coordinate_sha256(coords)}
    receipt = {'slide_id': 'one', 'path': str(path), 'instances': 12, 'source': source,
        'archive_sha256': digest_file(path), 'destination_fingerprint': fingerprint(path),
        'raw_rgb_pixel_sha256': pixel_hash.hexdigest(), 'source_fingerprint_after': fingerprint(wsi)}
    write_json(tmp_path / 'receipt.json', receipt)
    item = {'slide_id': 'one', 'path': str(path), 'sha256': receipt['archive_sha256'], 'instances': 12,
        'source': source, 'source_receipt': 'receipt.json', 'source_receipt_sha256': digest_file(tmp_path / 'receipt.json'),
        'destination_fingerprint': receipt['destination_fingerprint'],
        'coordinates_int64_le_sha256': source['coordinates_int64_le_sha256'],
        'raw_rgb_pixel_sha256': receipt['raw_rgb_pixel_sha256']}
    manifest = {'schema_version': 1, 'kind': MANIFEST_KIND, 'synthetic': True,
        'provenance': 'Explicit synthetic tiled TIFF software fixture, not natural WSI',
        'pixel_layout': 'NHWC', 'pixel_dtype': 'uint8', 'color': 'RGB', 'pixel_key': 'imgs',
        'coordinate_key': 'coords', 'patch_ids_key': 'patch_ids', 'slides': [item]}
    write_json(tmp_path / 'manifest.json', manifest)
    config = {'kind': KIND, 'coordinate_unit': 'level0_pixels', 'coordinate_order': 'xy',
        'level': 0, 'mult': 8 / 7, 'provenance': 'Same complete native source256 regions, measured original rows',
        'cache_manifest': 'manifest.json', 'backbone': {'architecture': 'resnet34', 'initialization': 'explicit_random'}}
    row = {'slide_id': 'one', 'path': str(path), 'coords_path': 'coords.npy', 'target': 1}
    bundle = SimpleNamespace(root=tmp_path, dim=3, task_type='classification', classes=['negative', 'positive'],
        rows=pd.DataFrame([row]), config={'synthetic': True}, encoder={'raw_regions': config})
    return bundle, row, manifest, coords, wsi


def original_class(filename, name):
    root, commit = verified_source('campanella_rnn')
    tree = ast.parse(pinned_text(root, commit, filename))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(torch=torch, np=np, random=random, data=data, openslide=openslide, sys=sys, Image=Image)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


def test_full_original_cnn_and_ten_step_rnn_datasets_bitwise_equal(cached):
    bundle, row, _, coords, wsi = cached
    reader = RawRegions(bundle)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5] * 3, [.1] * 3)])
    library = bundle.root / 'source.pt'
    torch.save({'slides': [str(wsi)], 'targets': [1], 'grid': [list(map(tuple, coords))], 'level': 0, 'mult': 8 / 7}, library)
    source_cnn = original_class('MIL_train.py', 'MILdataset')(str(library), normalize)
    source_cnn.setmode(1)
    source_rnn = original_class('RNN_train.py', 'rnndata')(str(library), 10, False, normalize)
    for index in range(12):
        assert torch.equal(reader.read(row, index, 'cnn'), source_cnn[index])
    sequence, target = source_rnn[0]
    assert target == 1 and len(sequence) == 10
    for index in range(10):
        assert torch.equal(reader.read(row, index, 'rnn'), sequence[index])
    assert reader.size('cnn') == reader.size('rnn') == 256
    assert reader.inventory(row, deep=True)['sha256'] == digest_file(row['path'])
    assert reader.describe(row)['kind'] == KIND
    with pytest.raises(ValueError, match='not an OpenSlide'):
        reader.slide(row)
    reader.cache.close()


@pytest.mark.parametrize('key,value', [('level', 1), ('mult', 1), ('mult', 1.003)])
def test_cannot_crop_or_resize_cache_to_hide_source_region_mismatch(cached, key, value):
    bundle, _, _, _, _ = cached
    bundle.encoder['raw_regions'][key] = value
    with pytest.raises(ValueError, match='exact original source level/size'):
        RawRegions(bundle)


def test_synthetic_cache_cannot_be_relabelled_as_real_input(cached):
    bundle, _, _, _, _ = cached
    bundle.config['synthetic'] = False
    with pytest.raises(ValueError, match='lossless RGB/source manifest'):
        RawRegions(bundle)


def test_coordinate_order_and_changed_coordinate_file_rejected(cached):
    bundle, row, _, coords, _ = cached
    reader = RawRegions(bundle)
    reader.read(row, 0, 'cnn')
    np.save(bundle.root / 'coords.npy', coords[::-1])
    with pytest.raises(ValueError, match='coordinate file changed'):
        reader.read(row, 0, 'cnn')
    fresh = RawRegions(bundle)
    with pytest.raises(ValueError, match='coordinate population/order'):
        fresh.describe(row)
    reader.cache.close()


def test_source_receipt_and_cache_population_hashes_are_bound(cached):
    bundle, _, manifest, _, _ = cached
    manifest['slides'][0]['source_receipt_sha256'] = '0' * 64
    write_json(bundle.root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='receipt hash mismatch'):
        RawRegions(bundle)


def test_file_change_after_real_delivery_is_not_silently_reaccepted(cached):
    bundle, row, _, _, _ = cached
    reader = RawRegions(bundle)
    reader.describe(row)
    with h5py.File(row['path'], 'r+') as archive:
        archive['imgs'][0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match='file identity changed'):
        reader.read(row, 0, 'cnn')


def test_manifest_requires_complete_unique_population(cached):
    bundle, _, manifest, _, _ = cached
    manifest['slides'].append(manifest['slides'][0])
    write_json(bundle.root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='every original slide exactly once'):
        RawRegions(bundle)


@pytest.mark.parametrize('quickhash', [None, '', False])
def test_optional_source_quickhash_absence_is_recorded_not_invented(cached, quickhash):
    bundle, row, manifest, _, _ = cached
    receipt_path = bundle.root / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    # Simulate a reader without the optional property on an explicitly
    # synthetic TIFF fixture; every actual file/pixel/coordinate hash remains.
    receipt['source']['source_quickhash1'] = quickhash
    write_json(receipt_path, receipt)
    manifest['slides'][0]['source'] = receipt['source']
    manifest['slides'][0]['source_receipt_sha256'] = digest_file(receipt_path)
    write_json(bundle.root / 'manifest.json', manifest)
    if quickhash is None:
        reader = RawRegions(bundle)
        info = reader.inventory(row, deep=True)
        assert info['source_quickhash1'] is None and info['source_quickhash_available'] is False
        assert reader.read(row, 0, 'cnn').shape == (3, 224, 224)
        reader.cache.close()
    else:
        with pytest.raises(ValueError, match='exact original source'):
            RawRegions(bundle)

"""Full native two-stage trajectories/recovery through the explicit cache I/O."""
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import openslide
import pandas as pd
import pytest
import torch
import yaml

import test_campanella as original
from milplus.data import Bundle, resolve
from milplus.engine import train
from milplus.io import digest_file, digest_object, write_json
from milplus.raw_region_cache import KIND, MANIFEST_KIND, coordinate_sha256, fingerprint
from milplus.raw_regions import grid_sha256

original_bundle = original.bundle


def cached_bundle(path, pretrained=True, **options):
    source = original_bundle(path.with_name(path.name + '_source'), pretrained=pretrained, mult=8 / 7, **options)
    return convert_source(source, path)


def convert_source(source, root):
    root.mkdir(exist_ok=False)
    task = yaml.safe_load((source.root / 'task.yaml').read_text())
    task['task_id'] += '_lossless_native256_cache'
    for name in ['slides.csv', 'split.csv']:
        shutil.copy2(source.root / name, root / name)
    rows, metadata = [], []
    for row in source.rows.to_dict('records'):
        wsi = source.root / row['path']
        coords = source.raw_regions.grid(row)
        path = root / (row['slide_id'] + '.h5')
        pixels = hashlib.sha256()
        before = fingerprint(wsi)
        with openslide.OpenSlide(str(wsi)) as slide, h5py.File(path, 'x') as archive:
            archive.attrs['synthetic'] = True
            images = archive.create_dataset('imgs', (len(coords), 256, 256, 3), dtype='uint8', compression='lzf')
            archive['coords'] = coords
            ids = []
            for index, coordinate in enumerate(coords):
                rgb = np.asarray(slide.read_region(tuple(coordinate), 0, (256, 256)).convert('RGB'))
                images[index] = rgb
                pixels.update(rgb.tobytes())
                identity = [row['slide_id'], index, 0, int(coordinate[0]), int(coordinate[1]), 256]
                ids.append(hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest())
            archive['patch_ids'] = np.asarray(ids, dtype='S64')
            provenance = {'kind': 'actual_wsi_read_region_original_clam_coordinates',
                'source_slide_id': row['slide_id'], 'source_wsi': str(wsi),
                'source_quickhash1': slide.properties['openslide.quickhash-1'], 'source_fingerprint_before': before,
                'patch_level': 0, 'patch_pixels': 256, 'actual_level0_dimensions': list(slide.dimensions),
                'actual_level_dimensions': list(slide.level_dimensions[0]), 'actual_openslide_downsample': 1.0,
                'coordinates_int64_le_sha256': coordinate_sha256(coords)}
        assert fingerprint(wsi) == before
        receipt = {'slide_id': row['slide_id'], 'path': str(path), 'instances': len(coords), 'source': provenance,
            'archive_sha256': digest_file(path), 'destination_fingerprint': fingerprint(path),
            'raw_rgb_pixel_sha256': pixels.hexdigest(), 'source_fingerprint_after': before}
        receipt_path = root / (row['slide_id'] + '.receipt.json')
        write_json(receipt_path, receipt)
        metadata.append({'slide_id': row['slide_id'], 'path': str(path), 'sha256': receipt['archive_sha256'],
            'instances': len(coords), 'source': provenance, 'source_receipt': str(receipt_path),
            'source_receipt_sha256': digest_file(receipt_path), 'destination_fingerprint': receipt['destination_fingerprint'],
            'coordinates_int64_le_sha256': coordinate_sha256(coords), 'raw_rgb_pixel_sha256': pixels.hexdigest()})
        rows.append({'slide_id': row['slide_id'], 'path': str(path), 'sha256': receipt['archive_sha256'],
            'coords_path': str(source.root / row['coords_path'])})
    pd.DataFrame(rows).to_csv(root / 'images.csv', index=False)
    write_json(root / 'source_cache.json', {'schema_version': 1, 'kind': MANIFEST_KIND, 'synthetic': True,
        'provenance': 'Actual synthetic TIFF read_region software-only fixture; no natural WSI or clinical result',
        'pixel_layout': 'NHWC', 'pixel_dtype': 'uint8', 'color': 'RGB', 'pixel_key': 'imgs',
        'coordinate_key': 'coords', 'patch_ids_key': 'patch_ids', 'slides': metadata})
    task['encoders']['raw_resnet34']['raw_regions'].update(kind=KIND, cache_manifest='source_cache.json')
    (root / 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return Bundle(root)


@pytest.mark.parametrize('interruption', [1, 2])
def test_cached_native_two_phase_original_exact_resume_case_unchanged(tmp_path, monkeypatch, interruption):
    # Execute the complete unchanged old case, including full native ResNet34,
    # ten recurrence steps, all old checkpoint/Adam/SGD/RNG/TEST assertions.
    monkeypatch.setattr(original, 'bundle', cached_bundle)
    original.test_full_two_phase_exact_resume_before_and_after_transition(tmp_path, interruption)


def test_cached_and_direct_original_regions_produce_identical_full_native_training(tmp_path):
    source = original_bundle(tmp_path / 'source', mult=8 / 7)
    cached = convert_source(source, tmp_path / 'cached')
    config = {'threads': 1, 'model': {'cnn_epochs': 1, 'rnn_epochs': 2, 'test_every': 1,
        'cnn_batch_size': 4, 'rnn_batch_size': 4}}
    direct_run = train(source.root, 'campanella_rnn', tmp_path / 'direct_run', seed=6, config=config)
    cached_run = train(cached.root, 'campanella_rnn', tmp_path / 'cached_run', seed=6, config=config)
    a, b = [torch.load(path / 'last.pt', map_location='cpu', weights_only=False) for path in [direct_run, cached_run]]
    # TIFF and H5 must have distinct input identities; compare their exact
    # original identity formula against their own real file stamps first.
    # Only that validated storage identity is removed from numerical parity.
    # All model tensors, selection indices/probabilities, moments and RNG stay
    # subject to the original exact recursive equality below.
    for checkpoint, bundle in [(a, source), (b, cached)]:
        by_slide = {row['slide_id']: row for row in bundle.rows.to_dict('records')}
        cache = checkpoint['model']['_extra_state']['top_cache']
        for slide, item in cache.items():
            row = by_slide[slide]
            path = resolve(bundle.root, row['path'])
            expected = digest_object({'grid': grid_sha256(bundle.raw_regions.grid(row)),
                'selector': checkpoint['model']['_extra_state']['cnn_best_sha256'],
                'image_stamp': [path.stat().st_size, path.stat().st_mtime_ns]})
            assert item['identity'] == expected
    for slide in a['model']['_extra_state']['top_cache']:
        left = a['model']['_extra_state']['top_cache'][slide]
        right = b['model']['_extra_state']['top_cache'][slide]
        assert left['identity'] != right['identity'], 'Distinct TIFF/H5 identities must never be conflated'
        del left['identity'], right['identity']
    for field in ['model', 'optimizers', 'rng', 'best', 'bad', 'history']:
        original.equal(a[field], b[field])
    assert a['model']['_extra_state']['cnn_updates'] == 1
    assert a['model']['_extra_state']['rnn_updates'] == 2
    assert json.loads((cached_run / 'method_protocol.json').read_text())['region_storage']['kind'] == KIND

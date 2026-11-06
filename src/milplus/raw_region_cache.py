"""Explicit lossless measured-region cache, never a fake whole-slide image.

Each cache row is the complete actual read_region output at its original
coordinate/level/size. A size mismatch cannot be repaired by cropping, filling,
resampling or repeated instances. Source resize/normalization stays in the
original RawRegions reader after this lossless I/O boundary.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .data import resolve
from .io import digest_file

KIND = 'measured_rgb_region_cache'
MANIFEST_KIND = 'campanella_lossless_original_read_region_cache'


def fingerprint(path):
    stat = Path(path).stat()
    return {name: getattr(stat, name) for name in ['st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino']}


def coordinate_sha256(coords):
    return hashlib.sha256(np.ascontiguousarray(coords, dtype='<i8').tobytes()).hexdigest()


class MeasuredRegionCache:
    def __init__(self, reader, path):
        self.reader = reader
        self.path = path
        self.metadata = metadata = json.loads(path.read_text())
        self.manifest_sha256 = digest_file(path)
        self.files = [path]
        self._validated = {}
        self._handle, self._handle_path = None, None
        synthetic = reader.bundle.config.get('synthetic', False)
        if (metadata.get('schema_version') != 1 or metadata.get('kind') != MANIFEST_KIND
                or metadata.get('synthetic') is not bool(synthetic)
                or not isinstance(metadata.get('provenance'), str) or not metadata['provenance'].strip()
                or metadata.get('pixel_layout') != 'NHWC' or metadata.get('pixel_dtype') != 'uint8'
                or metadata.get('color') != 'RGB' or metadata.get('pixel_key') != 'imgs'
                or metadata.get('coordinate_key') != 'coords' or metadata.get('patch_ids_key') != 'patch_ids'):
            raise ValueError('Measured region cache needs an explicit complete lossless RGB/source manifest')
        rows = metadata.get('slides')
        if (not isinstance(rows, list) or len(rows) != len(reader.bundle.rows)
                or any(not isinstance(row, dict) for row in rows)
                or len({row.get('slide_id') for row in rows}) != len(rows)
                or {row.get('slide_id') for row in rows} != set(reader.bundle.rows.slide_id)):
            raise ValueError('Measured region cache must cover every original slide exactly once')
        self.rows = {row['slide_id']: row for row in rows}
        for original in reader.bundle.rows.to_dict('records'):
            row = self.rows[original['slide_id']]
            receipt_path = resolve(path.parent, row['source_receipt'])
            if digest_file(receipt_path) != row.get('source_receipt_sha256'):
                raise ValueError('Measured region source receipt hash mismatch')
            receipt = json.loads(receipt_path.read_text())
            source = receipt.get('source', {})
            archive = resolve(reader.bundle.root, original['path'])
            if (archive != resolve(path.parent, row['path']) or archive != Path(receipt.get('path', '')).resolve()
                    or row.get('sha256') != receipt.get('archive_sha256')
                    or row.get('instances') != receipt.get('instances')
                    or row.get('coordinates_int64_le_sha256') != source.get('coordinates_int64_le_sha256')
                    or row.get('destination_fingerprint') != receipt.get('destination_fingerprint')
                    or row.get('source') != source or row.get('raw_rgb_pixel_sha256') != receipt.get('raw_rgb_pixel_sha256')
                    or receipt.get('slide_id') != original['slide_id']
                    or source.get('source_slide_id') != original['slide_id']
                    or source.get('kind') != 'actual_wsi_read_region_original_clam_coordinates'
                    or source.get('patch_level') != reader.level
                    or source.get('patch_pixels') != reader.size('cnn')
                    or source.get('patch_pixels') != reader.size('rnn')
                    or type(row.get('instances')) is not int or not 1 <= row['instances'] <= reader.max_rows
                    or not source.get('source_wsi') or 'source_quickhash1' not in source
                    or (source['source_quickhash1'] is not None and
                        (not isinstance(source['source_quickhash1'], str) or not source['source_quickhash1'].strip()))
                    or not isinstance(source.get('source_fingerprint_before'), dict)
                    or set(source['source_fingerprint_before']) != {'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino'}
                    or any(type(value) is not int or value < 0 for value in source['source_fingerprint_before'].values())
                    or source.get('source_fingerprint_before') != receipt.get('source_fingerprint_after')
                    or len(source.get('actual_level0_dimensions', [])) != 2
                    or len(source.get('actual_level_dimensions', [])) != 2):
                raise ValueError('Measured cache must retain exact original source level/size/coordinates/slide identity')
            downsample = source.get('actual_openslide_downsample')
            if type(downsample) not in {int, float} or not np.isfinite(downsample) or downsample <= 0:
                raise ValueError('Measured cache requires the actual source downsample')
            if 'downsample' in reader.config and reader.config['downsample'] != downsample:
                raise ValueError('Measured cache actual source downsample differs')
            if fingerprint(archive) != row['destination_fingerprint']:
                raise ValueError('Measured cache changed after its verified source delivery')
            self.files.append(receipt_path)

    def describe(self, original, coords):
        row = self.rows[original['slide_id']]
        path = resolve(self.reader.bundle.root, original['path'])
        stamp = fingerprint(path)
        if stamp != row['destination_fingerprint']:
            raise ValueError('Measured cache file identity changed')
        coordinate_fingerprint = fingerprint(resolve(self.reader.bundle.root, original['coords_path']))
        old = self._validated.get(original['slide_id'])
        if old is not None:
            if old['coordinate_fingerprint'] != coordinate_fingerprint:
                raise ValueError('Measured cache bound coordinate file changed')
            return dict(old['description'])
        if original['slide_id'] not in self._validated:
            if len(coords) != row['instances'] or coordinate_sha256(coords) != row['coordinates_int64_le_sha256']:
                raise ValueError('Measured cache original coordinate population/order mismatch')
            with h5py.File(path, 'r') as archive:
                for key in ['imgs', 'coords', 'patch_ids']:
                    if key not in archive or not isinstance(archive.get(key, getlink=True), h5py.HardLink):
                        raise ValueError('Measured cache data must be embedded, not soft/external links')
                    dataset = archive[key]
                    if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external:
                        raise ValueError('Measured cache cannot hide virtual or external pixel storage')
                size = row['source']['patch_pixels']
                if (archive['imgs'].shape != (row['instances'], size, size, 3)
                        or archive['imgs'].dtype != np.uint8 or archive['coords'].shape != coords.shape
                        or not np.issubdtype(archive['coords'].dtype, np.integer)
                        or not np.array_equal(archive['coords'][:], coords)
                        or bool(archive.attrs.get('synthetic', False)) != bool(self.reader.bundle.config.get('synthetic', False))
                        or archive['patch_ids'].shape != (row['instances'],)):
                    raise ValueError('Measured cache RGB/coordinate/patch identity header mismatch')
        source = row['source']
        dims = source['actual_level0_dimensions']
        if (coords[:, 0] >= dims[0]).any() or (coords[:, 1] >= dims[1]).any():
            raise ValueError('Measured cache coordinate origin outside actual WSI dimensions')
        from .raw_regions import grid_sha256
        description = {'kind': KIND, 'instances': row['instances'], 'level': self.reader.level,
            'downsample': source['actual_openslide_downsample'], 'level0_dimensions': dims,
            'coordinate_unit': 'level0_pixels', 'coordinate_order': 'xy',
            'cnn_read_size': self.reader.size('cnn'), 'rnn_read_size': self.reader.size('rnn'),
            'output_tile_size': 224, 'grid_sha256': grid_sha256(coords), 'backbone': self.reader.backbone,
            'source_cache_manifest_sha256': self.manifest_sha256, 'source_receipt_sha256': row['source_receipt_sha256'],
            'source_wsi': source['source_wsi'], 'source_quickhash1': source['source_quickhash1'],
            'source_quickhash_available': source['source_quickhash1'] is not None,
            'source_wsi_fingerprint': source['source_fingerprint_before'],
            'raw_rgb_pixel_sha256': row['raw_rgb_pixel_sha256'],
            'edge_behavior': 'literal cached OpenSlide RGBA->RGB output, no new padding/cropping or synthesis'}
        self._validated[original['slide_id']] = {'description': description,
            'coordinate_fingerprint': coordinate_fingerprint}
        return dict(description)

    def read(self, original, index, coords):
        self.describe(original, coords)
        path = resolve(self.reader.bundle.root, original['path'])
        if self._handle_path != path:
            self.close()
            self._handle, self._handle_path = h5py.File(path, 'r'), path
        coordinate = coords[index]
        source = self.rows[original['slide_id']]['source']
        identity = [original['slide_id'], index, self.reader.level, int(coordinate[0]), int(coordinate[1]), source['patch_pixels']]
        expected_id = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        if self._handle['patch_ids'][index].decode() != expected_id:
            raise ValueError('Measured cache patch ID differs from original region/row')
        return Image.fromarray(self._handle['imgs'][index])

    def validate_archive_digest(self, row, actual):
        if actual != self.rows[row['slide_id']]['sha256']:
            raise ValueError('Measured cache complete source archive SHA mismatch')

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle, self._handle_path = None, None

    def __del__(self):
        self.close()

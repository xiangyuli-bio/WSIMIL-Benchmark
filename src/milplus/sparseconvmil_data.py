"""Measured original RGB tiles and aligned spatial coordinates, never FM caches."""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'sparseconvmil_original_spatial_rgb_bags'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
        'encoder': {key: value for key, value in bundle.encoder.items() if key != 'sparseconvmil'}})


def validate_schema(bundle, schema):
    if (bundle.task_type != 'classification' or bundle.dim != 3
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('layout') != 'NHWC' or schema.get('dtype') != 'uint8'
            or schema.get('color') != 'RGB' or schema.get('normalization') != 'ToTensor_only'
            or schema.get('coordinate_frame') not in {'level0_top_left_pixels', 'tile_grid_indices'}
            or type(schema.get('patch_pixels')) is not int or schema['patch_pixels'] < 32
            or any(not isinstance(schema.get(key), str) or not schema[key].strip()
                   for key in ['pixel_key', 'patch_ids_key', 'coords_key', 'image_provenance',
                               'patch_provenance', 'coordinate_provenance'])
            or any(key in bundle.encoder for key in ['c2c', 'raw_images', 'raw_regions', 'spatial',
                'vision_language', 'dual_scale', 'multiscale', 'regions', 'smmile', 'sammil',
                'lrmil', 'hipt', 'omics'])):
        raise ValueError('SparseConvMIL needs original uint8 RGB images, measured coordinates and ToTensor-only preprocessing')
    return schema


def describe(bundle, row, schema):
    path = resolve(bundle.root, row['path'])
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        raise ValueError('SparseConvMIL images must be an embedded HDF5 bag, not cached FM features')
    with h5py.File(path, 'r') as archive:
        for name in [schema['pixel_key'], schema['patch_ids_key'], schema['coords_key']]:
            if name not in archive or isinstance(archive.get(name, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError('SparseConvMIL requires embedded image, patch-ID and coordinate datasets')
            value = archive[name]
            if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
                raise ValueError('SparseConvMIL cannot use unregistered external or virtual image/coordinate storage')
        pixels = archive[schema['pixel_key']]
        size = schema['patch_pixels']
        if pixels.ndim != 4 or pixels.shape[0] < 1 or pixels.shape[1:] != (size, size, 3) or pixels.dtype != np.uint8:
            raise ValueError('SparseConvMIL requires complete RGB tile bytes at the declared original size')
        ids = archive[schema['patch_ids_key']].asstr()[:].tolist()
        if (not isinstance(ids, list) or len(ids) != len(pixels)
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('SparseConvMIL original tile identities must be unique and image-aligned')
        coordinates = archive[schema['coords_key']][:]
        if (coordinates.shape != (len(pixels), 2) or not np.issubdtype(coordinates.dtype, np.integer)
                or (coordinates < 0).any() or len(np.unique(coordinates, axis=0)) != len(coordinates)):
            raise ValueError('SparseConvMIL needs actual unique integer coordinates aligned to original tiles')
        shape = list(pixels.shape)
    return {'shape': shape, 'patch_ids': ids, 'patch_ids_sha256': digest_object(ids),
            'coordinate_sha256': digest_object(coordinates.tolist())}


def prepare(bundle, schema, output):
    schema = validate_schema(bundle, schema)
    output = resolve(bundle.root, output)
    if output.exists():
        raise ValueError('SparseConvMIL manifest exists; prepare a new input version')
    records = {}
    for row in bundle.rows.to_dict('records'):
        records[row['slide_id']] = {**describe(bundle, row, schema),
            'path': str(resolve(bundle.root, row['path']).resolve()),
            'sha256': digest_file(resolve(bundle.root, row['path'])),
            **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}
    write_json(output, {'schema_version': 1, 'kind': KIND, 'schema': schema,
        'encoder_identity': encoder_identity(bundle), 'slides': records,
        'synthetic': bool(bundle.config.get('synthetic', False)), 'preparer_sha256': digest_file(Path(__file__))})
    return output


class SparseConvMILAssets:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('synthetic') != bool(bundle.config.get('synthetic', False))
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('SparseConvMIL images, input preparer or feature identity changed')
        self.schema = validate_schema(bundle, self.meta['schema'])
        for row in bundle.rows.to_dict('records'):
            record = self.meta['slides'][row['slide_id']]
            if (record['path'] != str(resolve(bundle.root, row['path']).resolve())
                    or any(record[key] != row[key] for key in ['global_patient_id', 'subset', 'target'])):
                raise ValueError('SparseConvMIL original images, patient labels or partition changed')
        self.contract = digest_file(self.path)
        self.files, self._verified = [self.path], {}

    def _check(self, row):
        path = resolve(self.bundle.root, row['path']); stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            expected = self.meta['slides'][row['slide_id']]
            if digest_file(path) != expected['sha256']:
                raise ValueError('SparseConvMIL original pixel/coordinate content changed')
            actual = describe(self.bundle, row, self.schema)
            if any(actual[key] != expected[key] for key in actual):
                raise ValueError('SparseConvMIL patch order, coordinates or shape changed')
            self._verified[str(path)] = signature

    @contextmanager
    def open(self, row):
        self._check(row)
        with h5py.File(resolve(self.bundle.root, row['path']), 'r') as archive:
            yield archive[self.schema['pixel_key']], archive[self.schema['coords_key']]

    def inventory(self, row, deep=False):
        if deep:
            self._check(row)
        return {key: self.meta['slides'][row['slide_id']][key]
                for key in ['shape', 'sha256', 'patch_ids_sha256', 'coordinate_sha256']}

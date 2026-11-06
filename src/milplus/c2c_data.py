"""Original RGB patch bags and immutable historical assets for end-to-end C2C."""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .c2c_assets import ASSETS, verify
from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'c2c_original_rgb_patch_bags'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
        'encoder': {key: value for key, value in bundle.encoder.items() if key != 'c2c'}})


def validate_schema(bundle, schema):
    if (bundle.task_type != 'classification' or bundle.dim != 3
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('layout') != 'NHWC' or schema.get('dtype') != 'uint8'
            or schema.get('color') != 'RGB' or schema.get('normalization') != 'albumentations046_ToTensor_no_normalize'
            or schema.get('coordinate_frame') not in {'path_identity_only', 'level0_top_left_pixels'}
            or type(schema.get('patch_pixels')) is not int or schema['patch_pixels'] < 32
            or any(not isinstance(schema.get(key), str) or not schema[key].strip()
                   for key in ['pixel_key', 'patch_ids_key', 'image_provenance', 'patch_provenance', 'asset_root'])
            or any(key in bundle.encoder for key in ['raw_images', 'raw_regions', 'spatial', 'vision_language',
                'dual_scale', 'multiscale', 'regions', 'smmile', 'sammil', 'lrmil', 'hipt', 'omics'])):
        raise ValueError('C2C needs original RGB patch bags, explicit ToTensor-only preprocessing and historical ResNet18 assets')
    if schema['coordinate_frame'] == 'level0_top_left_pixels' and not isinstance(schema.get('coords_key'), str):
        raise ValueError('C2C declared physical coordinates need an actual aligned coordinate dataset')
    return schema


def describe(bundle, row, schema):
    path = resolve(bundle.root, row['path'])
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        raise ValueError('C2C RGB patch input must be a packed, embedded HDF5 archive, not cached FM features')
    with h5py.File(path, 'r') as archive:
        for name in [schema['pixel_key'], schema['patch_ids_key']]:
            if name not in archive or isinstance(archive.get(name, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError('C2C pixels and original patch IDs must be embedded datasets')
            value = archive[name]
            if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
                raise ValueError('C2C archives cannot contain unregistered external pixel or patch-ID storage')
        pixels = archive[schema['pixel_key']]
        size = schema['patch_pixels']
        if pixels.ndim != 4 or pixels.shape[0] < 1 or pixels.shape[1:] != (size, size, 3) or pixels.dtype != np.uint8:
            raise ValueError('C2C must receive complete uint8 RGB patch images at the declared size')
        ids = archive[schema['patch_ids_key']].asstr()[:].tolist()
        if (not isinstance(ids, list) or len(ids) != len(pixels)
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('C2C original patch IDs must be unique and aligned to every actual image')
        coordinates = None
        if schema['coordinate_frame'] == 'level0_top_left_pixels':
            name = schema['coords_key']
            if name not in archive or isinstance(archive.get(name, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError('C2C physical coordinates must be embedded and real')
            coordinates = archive[name][:]
            if (coordinates.shape != (len(pixels), 2) or not np.issubdtype(coordinates.dtype, np.integer)
                    or (coordinates < 0).any() or len(np.unique(coordinates, axis=0)) != len(coordinates)):
                raise ValueError('C2C actual coordinate/patch row alignment changed')
        shape = list(pixels.shape)
    return {'shape': shape, 'patch_ids': ids, 'patch_ids_sha256': digest_object(ids),
        'coordinate_sha256': None if coordinates is None else digest_object(coordinates.tolist())}


def prepare(bundle, schema, output):
    schema = validate_schema(bundle, schema)
    output = resolve(bundle.root, output)
    if output.exists():
        raise ValueError('C2C manifest exists; prepare a new version instead of overwriting evidence')
    asset_root = verify(resolve(bundle.root, schema['asset_root'])).resolve()
    records = {}
    for row in bundle.rows.to_dict('records'):
        description = describe(bundle, row, schema)
        records[row['slide_id']] = {**description,
            'path': str(resolve(bundle.root, row['path']).resolve()),
            'sha256': digest_file(resolve(bundle.root, row['path'])),
            **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}
    write_json(output, {'schema_version': 1, 'kind': KIND, 'schema': schema,
        'encoder_identity': encoder_identity(bundle), 'asset_root': str(asset_root),
        'assets': {name: sha256 for name, (_, sha256) in ASSETS.items()}, 'slides': records,
        'synthetic': bool(bundle.config.get('synthetic', False)), 'preparer_sha256': digest_file(Path(__file__))})
    return output


class C2CAssets:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('synthetic') != bool(bundle.config.get('synthetic', False))
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('assets') != {name: sha256 for name, (_, sha256) in ASSETS.items()}
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('C2C input/source/preparer identity changed')
        self.schema = validate_schema(bundle, self.meta['schema'])
        self.assets = verify(self.meta['asset_root'])
        for row in bundle.rows.to_dict('records'):
            record = self.meta['slides'][row['slide_id']]
            if (record['path'] != str(resolve(bundle.root, row['path']).resolve())
                    or any(record[key] != row[key] for key in ['global_patient_id', 'subset', 'target'])):
                raise ValueError('C2C original images, patient mapping, labels or split changed')
        self.contract = digest_file(self.path)
        self.files = [self.path, *[self.assets/name for name in ASSETS]]
        self._verified = {}

    def _check(self, row):
        path = resolve(self.bundle.root, row['path']); stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            expected = self.meta['slides'][row['slide_id']]
            if digest_file(path) != expected['sha256']:
                raise ValueError('C2C original image content or patch identity changed')
            actual = describe(self.bundle, row, self.schema)
            if any(actual[key] != expected[key] for key in actual):
                raise ValueError('C2C actual image shape, ordered patch identity or physical coordinates changed')
            self._verified[str(path)] = signature

    @contextmanager
    def open(self, row):
        self._check(row)
        with h5py.File(resolve(self.bundle.root, row['path']), 'r') as archive:
            # A loader with workers0 streams this handle; neither all images in
            # a real WSI nor an artificial fixed-size bag is forced into RAM.
            yield archive[self.schema['pixel_key']]

    def count(self, row):
        return self.meta['slides'][row['slide_id']]['shape'][0]

    def inventory(self, row, deep=False):
        if deep:
            self._check(row)
        record = self.meta['slides'][row['slide_id']]
        return {key: record[key] for key in ['shape', 'sha256', 'patch_ids_sha256', 'coordinate_sha256']}

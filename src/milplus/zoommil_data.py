"""Bound actual multiscale H5 features and source recursive spatial addresses.

The root order comes from the registered low-magnification features. Children
are reordered by their actual recursive coordinates, never invented from row
count. The source child order is(0,0),(1,0),(0,1),(1,1).
"""
import copy
import hashlib
import math
import os
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'zoommil_three_level_source_hierarchy'


def encoder_identity(bundle):
    return digest_object({key: value for key, value in bundle.encoder.items() if key != 'zoommil'})


def schema_contract(bundle, schema):
    if (not isinstance(schema, dict) or bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide'
            or schema.get('kind') != KIND or schema.get('schema_version') != 1
            or type(schema.get('synthetic')) is not bool
            or schema['synthetic'] != bool(bundle.config.get('synthetic', False))
            or schema.get('coordinate_frame') not in {'source_recursive_row_col_indices', 'level0_top_left_pixels'}
            or type(schema.get('patch_pixels')) is not int or schema['patch_pixels'] < 1):
        raise ValueError('ZoomMIL needs slide classification and actual source recursive multiscale geometry')
    for key in ['coordinate_provenance', 'weights_provenance', 'preprocessing', 'multiscale_provenance']:
        if not isinstance(schema.get(key), str) or not schema[key].strip():
            raise ValueError('ZoomMIL requires actual '+key)
    if schema['coordinate_frame'] == 'level0_top_left_pixels':
        if not isinstance(schema.get('patch_size_provenance'), str) or not schema['patch_size_provenance'].strip():
            raise ValueError('ZoomMIL physical coordinates require actual patch-size/calibration provenance')
        attribute = schema.get('coarse_patch_size_attribute')
        if attribute is not None and (not isinstance(attribute, str) or not attribute.strip()):
            raise ValueError('ZoomMIL per-slide physical patch-size attribute is invalid')
        size = schema.get('coarse_patch_size_level0')
        if attribute is None and (type(size) not in {int, float} or not math.isfinite(size) or size <= 0):
            raise ValueError('ZoomMIL needs actual coarse patch side in level0 pixels or its per-slide H5 attribute')
    levels = schema.get('levels')
    if not isinstance(levels, list) or len(levels) != 3:
        raise ValueError('The released ZoomMIL model needs three actual magnification levels')
    magnifications, keys = [], []
    for level in levels:
        mag = level.get('magnification')
        if type(mag) not in {int, float} or not math.isfinite(mag) or mag <= 0:
            raise ValueError('ZoomMIL requires positive finite magnifications')
        for key in ['feature_key', 'coords_key']:
            if not isinstance(level.get(key), str) or not level[key].strip():
                raise ValueError('ZoomMIL level H5 keys are required')
            keys.append(level[key])
        magnifications.append(mag)
    if len(keys) != len(set(keys)) or not all(a < b for a, b in pairwise(magnifications)):
        raise ValueError('ZoomMIL levels need distinct keys and increasing actual magnifications')
    depths = [math.log2(mag/magnifications[0]) for mag in magnifications]
    if any(abs(depth-round(depth)) > 1e-10 for depth in depths):
        raise ValueError('ZoomMIL needs complete power-of-two source pyramids')
    return [round(depth) for depth in depths]


def _array_hash(value):
    value = np.asarray(value)
    canonical = np.asarray(value, dtype=np.float64 if value.dtype.kind == 'f' else np.int64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def physical_to_recursive(coordinates, depths, coarse_side):
    """Derive source addresses only from actual complete nested physical grids.

    The origin is a translation of the observed actual low grid, never row IDs.
    Every finer top-left must align to its stated physical pitch; descendant
    membership/complete coverage is checked by read_hierarchy afterwards.
    """
    if not math.isfinite(coarse_side) or coarse_side <= 0:
        raise ValueError('ZoomMIL physical coarse patch side must be positive and finite')
    if any(value.ndim != 2 or value.shape[1] != 2 or not len(value)
           or value.dtype.kind not in 'iuf' or not np.isfinite(value).all() for value in coordinates):
        raise ValueError('ZoomMIL physical coordinates must be actual finite nonempty [N,2] top-left xy positions')
    origin = coordinates[0].min(axis=0)
    recursive = []
    for value, depth in zip(coordinates, depths):
        grid = (value-origin)/(coarse_side/2**depth)
        rounded = np.rint(grid)
        if (not np.allclose(grid, rounded, rtol=0., atol=1e-7) or (rounded < 0).any()
                or (rounded > np.iinfo(np.int64).max//2).any()):
            raise ValueError('ZoomMIL coordinates do not form the actual declared nested physical patch grid')
        xy = rounded.astype(np.int64)
        root = xy//2**depth
        prefix = [root[:, 1], root[:, 0]]
        for shift in reversed(range(depth)):
            prefix.extend([(xy[:, 1] >> shift) & 1, (xy[:, 0] >> shift) & 1])
        recursive.append(np.stack(prefix, axis=1))
    return recursive


def read_hierarchy(bundle, row, path, schema):
    depths = schema_contract(bundle, schema)
    levels = schema['levels']
    features, coordinates = [], []
    with h5py.File(path, 'r') as archive:
        for level in levels:
            features.append(np.asarray(archive[level['feature_key']][:]))
            coordinates.append(np.asarray(archive[level['coords_key']][:]))
        side = (archive.attrs[schema['coarse_patch_size_attribute']]
                if schema.get('coarse_patch_size_attribute') else schema.get('coarse_patch_size_level0'))
    raw_coordinates = coordinates
    primary, primary_coords = bundle.load(row)
    if schema['coordinate_frame'] == 'level0_top_left_pixels':
        if (primary_coords is not None and not torch.equal(primary_coords,
                torch.from_numpy(np.asarray(coordinates[0], dtype=np.float32)))):
            raise ValueError('ZoomMIL low physical coordinates differ from the actual registered feature/coordinate rows')
        coordinates = physical_to_recursive(coordinates, depths, float(side))
    low = coordinates[0]
    if (low.shape != (len(primary), 2) or low.dtype.kind not in 'iu' or (low < 0).any()
            or len(set(map(tuple, low.tolist()))) != len(low)):
        raise ValueError('ZoomMIL root coordinates must be actual unique source row/col indices')
    roots = {tuple(coord): index for index, coord in enumerate(low.tolist())}
    outputs, orders, counts = [], [], []
    for depth, feature, coords in zip(depths, features, coordinates):
        count = len(low)*4**depth
        if (feature.shape != (count, bundle.dim) or feature.dtype.kind != 'f' or not np.isfinite(feature).all()
                or coords.shape != (count, 2+2*depth) or coords.dtype.kind not in 'iu'):
            raise ValueError('ZoomMIL requires every finite full-dimension child and its complete spatial address')
        if depth and not np.isin(coords[:, 2:], [0, 1]).all():
            raise ValueError('ZoomMIL recursive child coordinates must identify actual quadrants')
        try:
            key = np.asarray([roots[tuple(coord[:2])] for coord in coords.tolist()], dtype=np.int64)
        except KeyError as error:
            raise ValueError('ZoomMIL child belongs to an unregistered spatial root') from error
        for offset in range(depth):
            key = key*4+coords[:, 2+offset*2]+2*coords[:, 3+offset*2]
        order = np.argsort(key, kind='stable')
        if not np.array_equal(key[order], np.arange(count)):
            raise ValueError('ZoomMIL hierarchy has duplicate, missing or mismatched spatial descendants')
        value = torch.from_numpy(np.asarray(feature[order], dtype=np.float32))
        outputs.append(value); orders.append(_array_hash(order)); counts.append(count)
    if not torch.equal(outputs[0], primary):
        raise ValueError('ZoomMIL root features differ from the actual registered frozen-FM slide features')
    geometry = digest_object({'raw_coordinates': [_array_hash(value) for value in raw_coordinates],
                              'recursive_coordinates': [_array_hash(value) for value in coordinates],
                              'canonical_orders': orders, 'depths': depths, 'counts': counts})
    return outputs, geometry, counts


def prepare(bundle, index, schema, output):
    schema_contract(bundle, schema)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('ZoomMIL preparation needs a new immutable output path')
    frame = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (set(frame) != {'slide_id', 'path'} or frame.slide_id.duplicated().any()
            or set(frame.slide_id) != set(bundle.rows.slide_id) or frame.path.eq('').any()):
        raise ValueError('ZoomMIL hierarchy index must contain exactly all registered slides, no outcomes')
    paths = {item.slide_id: resolve(Path(index).resolve().parent, item.path).resolve() for item in frame.itertuples()}
    if len(set(paths.values())) != len(paths):
        raise ValueError('ZoomMIL slides cannot share one physical hierarchy file')
    entries = {}
    for row in bundle.rows.to_dict('records'):
        path = paths[row['slide_id']]
        _, geometry, counts = read_hierarchy(bundle, row, path, schema)
        entries[row['slide_id']] = {'path': os.path.relpath(path, output.parent), 'sha256': digest_file(path),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'geometry_sha256': geometry, 'nodes_per_level': counts}
    write_json(output, {'schema_version': 1, 'schema': copy.deepcopy(schema),
                        'encoder_sha256': encoder_identity(bundle), 'slides': entries})
    return output


class ZoomMILAssets:
    def __init__(self, bundle, manifest=None):
        registration = bundle.encoder.get('zoommil', {})
        if not isinstance(registration, dict):
            raise TypeError('ZoomMIL encoder registration must be a manifest mapping')
        manifest = manifest or registration.get('manifest')
        if not isinstance(manifest, (str, Path)):
            raise TypeError('ZoomMIL requires encoder.zoommil.manifest with actual multiscale geometry')
        self.path = resolve(bundle.root, manifest).resolve()
        self.bundle, self.meta = bundle, read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('encoder_sha256') != encoder_identity(bundle)
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('ZoomMIL manifest changed encoder identity or slide inventory')
        self.depths = schema_contract(bundle, self.meta['schema'])
        self.contract = digest_file(self.path)
        self.files = [self.path, *[resolve(self.path.parent, item['path']).resolve() for item in self.meta['slides'].values()]]
        if len(set(self.files)) != len(self.files):
            raise ValueError('ZoomMIL slides need distinct real hierarchy files')
        self.files = list(dict.fromkeys([*self.files,
            *[resolve(bundle.root, row['path']).resolve() for row in bundle.rows.to_dict('records')]]))

    def load(self, row):
        entry = self.meta['slides'][row['slide_id']]
        path = resolve(self.path.parent, entry['path'])
        if (digest_file(self.path) != self.contract or digest_file(path) != entry['sha256']
                or digest_file(resolve(self.bundle.root, row['path'])) != entry['primary_sha256']):
            raise ValueError('ZoomMIL hierarchy, preparation manifest or registered FM features changed')
        values, geometry, counts = read_hierarchy(self.bundle, row, path, self.meta['schema'])
        if geometry != entry['geometry_sha256'] or counts != entry['nodes_per_level']:
            raise ValueError('ZoomMIL prepared spatial order or full feature shapes changed')
        return values

    def inventory(self, row, deep=False):
        entry = self.meta['slides'][row['slide_id']]
        if deep:
            self.load(row)
        return {'manifest_sha256': self.contract, 'hierarchy_sha256': entry['sha256'],
                'geometry_sha256': entry['geometry_sha256'], 'nodes_per_level': entry['nodes_per_level']}

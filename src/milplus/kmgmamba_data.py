"""Measured dual-scale KMG-Mamba features and explicit extraction provenance."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'kmgmamba_measured_dual_scale'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'split': bundle.split, 'dimension': bundle.dim,
        'encoder': {key: value for key, value in bundle.encoder.items() if key != 'kmgmamba'}})


def validate_schema(bundle, schema):
    if (schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('coordinate_frame') != 'level0_top_left_pixels'
            or any(not isinstance(schema.get(k), str) or not schema[k].strip() for k in
                   ['coordinate_provenance', 'weights_provenance', 'preprocessing', 'patch_size_provenance'])):
        raise ValueError('KMG-Mamba requires explicit level0 coordinates, two physical scales and extraction provenance')
    levels = schema.get('levels')
    if not isinstance(levels, list) or len(levels) != 2:
        raise ValueError('KMG-Mamba needs exactly two measured feature scales, coarse then fine')
    if (any(type(level.get('patch_size_level0')) is not int or level['patch_size_level0'] < 1
            or type(level.get('magnification')) not in {int, float} or not np.isfinite(level['magnification'])
            or level['magnification'] <= 0 or not level.get('feature_key') or not level.get('coords_key') for level in levels)
            or levels[0]['patch_size_level0'] != 2*levels[1]['patch_size_level0']
            or levels[1]['magnification'] != 2*levels[0]['magnification']
            or len({level['feature_key'] for level in levels}) != 2
            or len({level['coords_key'] for level in levels}) != 2):
        raise ValueError('KMG-Mamba source geometry requires distinct 2:1 physical patch scales')
    return schema


def read_pair(bundle, row, path, schema):
    features, coordinates = [], []
    with h5py.File(path, 'r') as archive:
        for level in schema['levels']:
            if level['feature_key'] not in archive or level['coords_key'] not in archive:
                raise ValueError('KMG-Mamba archive is missing a declared measured scale')
            x, c = archive[level['feature_key']][:], archive[level['coords_key']][:]
            if (x.ndim != 2 or not len(x) or x.shape[1] != bundle.dim or not np.issubdtype(x.dtype, np.floating)
                    or not np.isfinite(x).all() or not np.isfinite(x.astype(np.float32)).all()
                    or c.shape != (len(x), 2)
                    or not (np.issubdtype(c.dtype, np.integer) or np.issubdtype(c.dtype, np.floating))
                    or not np.isfinite(c).all() or (c < 0).any() or not np.equal(c, np.floor(c)).all()
                    or (c > np.iinfo(np.int64).max-level['patch_size_level0']).any()):
                raise ValueError('KMG-Mamba needs finite full-dimensional features and aligned integer physical coordinates')
            c = c.astype(np.int64)
            if len(np.unique(c, axis=0)) != len(c):
                raise ValueError('KMG-Mamba original measured patches must not have duplicate coordinates')
            if row.get('width_level0') and row.get('height_level0'):
                extent = np.asarray([int(row['width_level0']), int(row['height_level0'])])
                if (c+level['patch_size_level0'] > extent).any():
                    raise ValueError('KMG-Mamba measured patch exceeds registered WSI dimensions')
            features.append(x.astype(np.float32)); coordinates.append(c)
    primary, coords = bundle.load(row, require_coords=True)
    if not torch.equal(primary, torch.from_numpy(features[0])) or not torch.equal(coords, torch.from_numpy(coordinates[0])):
        raise ValueError('KMG-Mamba primary features/coordinates must match the actual coarse scale and row order')
    return (torch.from_numpy(features[0]), torch.from_numpy(coordinates[0]),
            torch.from_numpy(features[1]), torch.from_numpy(coordinates[1]))


def prepare(bundle, index, schema, output):
    schema = validate_schema(bundle, schema)
    index, output = resolve(bundle.root, index), resolve(bundle.root, output)
    if output.exists():
        raise ValueError('KMG-Mamba manifest already exists; create a new explicit input version')
    table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (not {'slide_id', 'path'} <= set(table) or table[['slide_id', 'path']].eq('').any().any()
            or table.slide_id.duplicated().any() or set(table.slide_id) != set(bundle.rows.slide_id)):
        raise ValueError('KMG-Mamba scale index must contain exactly the task slide set without duplicates')
    records, used = {}, set()
    lookup = table.set_index('slide_id')
    for row in bundle.rows.to_dict('records'):
        path = resolve(bundle.root, lookup.loc[row['slide_id'], 'path']).resolve()
        if path in used:
            raise ValueError('KMG-Mamba slides cannot share a single feature archive')
        used.add(path)
        loaded = read_pair(bundle, row, path, schema)
        records[row['slide_id']] = {'path': str(path), 'sha256': digest_file(path),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'global_patient_id': row['global_patient_id'], 'subset': row['subset'], 'target': row['target'],
            'nodes_per_level': [len(loaded[0]), len(loaded[2])]}
    write_json(output, {'schema_version': 1, 'kind': KIND, 'schema': schema,
        'encoder_identity': encoder_identity(bundle), 'index': str(index), 'index_sha256': digest_file(index),
        'slides': records, 'preparer_sha256': digest_file(Path(__file__))})
    return output


class KMGMambaAssets:
    def __init__(self, bundle):
        self.bundle = bundle
        self.path = resolve(bundle.root, bundle.encoder['kmgmamba']['manifest'])
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)
                or digest_file(self.meta['index']) != self.meta.get('index_sha256')):
            raise ValueError('KMG-Mamba input identity, task slides or preparation changed')
        self.schema = validate_schema(bundle, self.meta['schema'])
        for row in bundle.rows.to_dict('records'):
            record = self.meta['slides'][row['slide_id']]
            if any(record[key] != row[key] for key in ['global_patient_id', 'subset', 'target']):
                raise ValueError('KMG-Mamba task label, patient mapping or split changed')
        self.files = [self.path, Path(self.meta['index']),
                      *[Path(v['path']) for v in self.meta['slides'].values()]]
        self.contract = digest_file(self.path)

    def load(self, row):
        record = self.meta['slides'][row['slide_id']]
        if (digest_file(record['path']) != record['sha256']
                or digest_file(resolve(self.bundle.root, row['path'])) != record['primary_sha256']):
            raise ValueError('KMG-Mamba registered full features or coordinates changed')
        result = read_pair(self.bundle, row, record['path'], self.schema)
        if record['nodes_per_level'] != [len(result[0]), len(result[2])]:
            raise ValueError('KMG-Mamba full measured patch sequence changed')
        return result

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        record = self.meta['slides'][row['slide_id']]
        return {'manifest_sha256': self.contract, 'source_sha256': record['sha256'],
                'primary_sha256': record['primary_sha256'], 'nodes_per_level': record['nodes_per_level']}

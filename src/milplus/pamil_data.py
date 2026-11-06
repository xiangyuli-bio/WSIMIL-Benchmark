"""Input-bound TRAIN instance annotations for PAMIL's active prototype push.

This is extra supervision. It is never inferred from slide labels or requested
for validation/test bags. Unsafe NumPy object dictionaries are not supported.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_array, resolve
from .io import digest_file, digest_object, write_json


def encoder_identity(bundle):
    return digest_object({key: value for key, value in bundle.encoder.items() if key not in {'pamil', 'index'}})


def load_labels(path, count, classes):
    labels = load_array(path, 'instance_labels')
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('PAMIL instance labels must be aligned integer [N]')
    if (labels < -1).any() or (labels >= classes).any():
        raise ValueError('PAMIL labels must be -1=unannotated/background or an ordered task class index')
    return labels.astype(np.int64)


def input_binding(bundle, row):
    x, _ = bundle.load(row, require_coords=True)
    cp = resolve(bundle.root, row.get('coords_path') or row['path'])
    return {'instances': len(x), 'features_sha256': digest_file(resolve(bundle.root, row['path'])),
            'coordinates_sha256': digest_file(cp)}


def prepare_annotations(bundle, annotations, output, provenance):
    """Freeze supplied labels against one FM, split and patch order; never fit labels."""
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('PAMIL requires explicit instance annotation provenance')
    source = Path(annotations).resolve()
    table = pd.read_csv(source, dtype=str, keep_default_na=False)
    if not {'slide_id', 'path'} <= set(table) or table[['slide_id', 'path']].eq('').any().any():
        raise ValueError('PAMIL annotations CSV needs nonempty slide_id,path')
    if table.slide_id.duplicated().any():
        raise ValueError('Duplicate PAMIL annotation slide_id')
    rows = {row['slide_id']: row for row in bundle.records('train')}
    if set(table.slide_id) != set(rows):
        raise ValueError('PAMIL annotations must contain exactly TRAIN slides, no validation/test labels')
    records, paths = {}, set()
    for item in table.to_dict('records'):
        row = rows[item['slide_id']]
        binding = input_binding(bundle, row)
        path = resolve(source.parent, item['path']).resolve()
        if path in paths:
            raise ValueError('Different PAMIL slides must not share one physical annotation file')
        paths.add(path)
        load_labels(path, binding['instances'], len(bundle.classes))
        records[item['slide_id']] = {**binding, 'path': str(path), 'sha256': digest_file(path)}
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError('PAMIL refuses to overwrite frozen annotation manifest')
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {'schema_version': 1, 'fm': bundle.fm,
        'weights_digest': bundle.encoder.get('weights_digest'), 'encoder_sha256': encoder_identity(bundle), 'split': bundle.split,
        'classes': bundle.classes, 'scope': 'training_instance_labels_extra_supervision',
        'provenance': provenance, 'records': records})
    return destination


class PAMILAnnotations:
    def __init__(self, bundle):
        config = bundle.encoder.get('pamil')
        if not isinstance(config, dict) or set(config) != {'manifest'} or not isinstance(config['manifest'], str):
            raise ValueError('PAMIL requires encoder.pamil.manifest for train-only extra instance supervision')
        self.bundle = bundle
        self.path = resolve(bundle.root, config['manifest']).resolve()
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        expected = {'schema_version': 1, 'fm': bundle.fm, 'weights_digest': bundle.encoder.get('weights_digest'),
                    'encoder_sha256': encoder_identity(bundle), 'split': bundle.split, 'classes': bundle.classes,
                    'scope': 'training_instance_labels_extra_supervision'}
        if any(m.get(k) != value for k, value in expected.items()):
            raise ValueError('PAMIL annotation FM/weights/split/classes/scope mismatch')
        if not isinstance(m.get('provenance'), str) or not m['provenance'].strip():
            raise ValueError('PAMIL annotation provenance is required')
        self.rows = {row['slide_id']: row for row in bundle.records('train')}
        self.records = m.get('records')
        if not isinstance(self.records, dict) or set(self.records) != set(self.rows):
            raise ValueError('PAMIL manifest must cover exactly TRAIN slides')
        self.files = [self.path]
        paths = set()
        for record in self.records.values():
            if not isinstance(record, dict) or not isinstance(record.get('path'), str) or not record['path']:
                raise ValueError('PAMIL annotation record requires a path')
            p = resolve(self.path.parent, record['path']).resolve()
            if p in paths:
                raise ValueError('Different PAMIL slides must not share one physical annotation file')
            paths.add(p)
            self.files.append(p)

    def load(self, row):
        key = row['slide_id']
        if key not in self.rows or row.get('subset') != 'train':
            raise ValueError('PAMIL annotation access is restricted to training slides')
        item = self.records[key]
        binding = input_binding(self.bundle, row)
        if any(item.get(k) != value for k, value in binding.items()):
            raise ValueError('PAMIL annotation binding differs from features/coordinate rows')
        path = resolve(self.path.parent, item['path'])
        if digest_file(path) != item.get('sha256'):
            raise ValueError('PAMIL annotation hash mismatch')
        return load_labels(path, binding['instances'], len(self.bundle.classes))

    def inventory(self, row, deep):
        if row['subset'] != 'train':
            return {'instance_annotations': 'not_provided_not_used'}
        item = self.records[row['slide_id']]
        path = resolve(self.path.parent, item['path'])
        if digest_file(path) != item.get('sha256'):
            raise ValueError('PAMIL annotation hash mismatch')
        if deep:
            self.load(row)
        return {'instance_annotations': item, 'scope': self.manifest['scope']}

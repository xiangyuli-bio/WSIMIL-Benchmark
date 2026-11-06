"""SAM region membership, original coordinates, area and mean-token provenance."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'sammil_measured_region_membership'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
        'encoder': {key: value for key, value in bundle.encoder.items() if key != 'sammil'}})


def read_groups(bundle, row, path):
    features, coords = bundle.load(row, require_coords=True)
    coords = coords.numpy()
    if (len(features) < 5 or (coords < 0).any() or not np.equal(coords, np.floor(coords)).all()
            or len(np.unique(coords, axis=0)) != len(coords)):
        raise ValueError('SAM-MIL base features need unique real patch coordinates and no pre-appended group tokens')
    lookup = {tuple(value): i for i, value in enumerate(coords)}
    groups, areas, ids = [], [], []
    with h5py.File(path, 'r') as archive:
        if 'coords' not in archive or not np.array_equal(archive['coords'][:], coords):
            raise ValueError('SAM-MIL membership coordinate order differs from the registered base features')
        if 'groups' not in archive or not isinstance(archive['groups'], h5py.Group):
            raise ValueError('SAM-MIL archive must contain actual SAM groups')
        for name in sorted(archive['groups'], key=lambda value: int(value.removeprefix('seg_'))):
            item = archive['groups'][name]
            if (not name.startswith('seg_') or not name[4:].isdigit() or not isinstance(item, h5py.Dataset)):
                raise ValueError('SAM-MIL group identities must be explicit seg_ID datasets')
            values = item[:]
            area = item.attrs.get('area')
            if (values.ndim != 2 or values.shape[1] != 2 or not len(values)
                    or not np.issubdtype(values.dtype, np.integer) or (values < 0).any()
                    or len(np.unique(values, axis=0)) != len(values)
                    or not isinstance(area, (int, np.integer)) or area < 2
                    or any(tuple(coord) not in lookup for coord in values)):
                raise ValueError('SAM-MIL requires nonempty measured memberships and original positive integer SAM mask areas')
            groups.append([lookup[tuple(coord)] for coord in values]); areas.append(int(area)); ids.append(name)
    if not groups:
        raise ValueError('SAM-MIL requires genuine segmentation groups; no fabricated or empty region fallback')
    # Author preprocessing appends one mean token for each nonempty SAM mask.
    # Overlap is retained. An ordinary patch receives the minimum covering area;
    # uncovered patches retain sentinel1, group tokens sentinel0.
    x = features.numpy()
    tokens = [np.mean(x[indices], axis=0, keepdims=True) for indices in groups]
    augmented = np.concatenate([x, *tokens], axis=0)
    relative = np.ones(len(x), dtype=np.int64)
    for indices, area in zip(groups, areas):
        for index in indices:
            relative[index] = area if relative[index] == 1 else min(relative[index], area)
    return {'features': torch.from_numpy(augmented),
        'is_group_feat': torch.tensor([0]*len(x)+[1]*len(groups), dtype=torch.int64),
        'relative_area': torch.from_numpy(np.concatenate([relative, np.zeros(len(groups), dtype=np.int64)])),
        'group_ids': ids, 'members': groups, 'areas': areas,
        'coords': torch.from_numpy(np.concatenate([coords, np.full((len(groups), 2), -1)], axis=0))}


def validate_schema(bundle, schema):
    if (bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide'
            or schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('coordinate_frame') != 'level0_top_left_pixels'
            or schema.get('group_feature') != 'arithmetic_mean_of_registered_member_patch_features'
            or schema.get('overlap') != 'minimum_original_mask_area'
            or schema.get('sam_model') != 'vit_h'
            or not isinstance(schema.get('sam_weights_sha256'), str)
            or len(schema['sam_weights_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in schema['sam_weights_sha256'])
            or any(not isinstance(schema.get(key), str) or not schema[key].strip()
                   for key in ['segmentation_provenance', 'membership_provenance', 'feature_provenance'])):
        raise ValueError('SAM-MIL needs explicit SAM-vit-H weights, measured membership, original area and feature provenance')
    return schema


def prepare(bundle, index, schema, output):
    schema = validate_schema(bundle, schema)
    index, output = resolve(bundle.root, index), resolve(bundle.root, output)
    if output.exists():
        raise ValueError('SAM-MIL manifest exists; prepare a new explicit version')
    table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (not {'slide_id', 'path'} <= set(table) or table[['slide_id', 'path']].eq('').any().any()
            or table.slide_id.duplicated().any() or set(table.slide_id) != set(bundle.rows.slide_id)):
        raise ValueError('SAM-MIL membership index must contain every registered task slide once')
    lookup, records, seen = table.set_index('slide_id'), {}, set()
    for row in bundle.rows.to_dict('records'):
        path = resolve(bundle.root, lookup.loc[row['slide_id'], 'path']).resolve()
        if path in seen:
            raise ValueError('Distinct SAM-MIL slides cannot share one membership archive')
        seen.add(path); value = read_groups(bundle, row, path)
        records[row['slide_id']] = {'path': str(path), 'sha256': digest_file(path),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'groups_sha256': digest_object({key: value[key] for key in ['group_ids', 'members', 'areas']}),
            'base_nodes': len(value['features'])-len(value['members']), 'group_nodes': len(value['members']),
            **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}
    write_json(output, {'schema_version': 1, 'kind': KIND, 'schema': schema,
        'encoder_identity': encoder_identity(bundle), 'index': str(index), 'index_sha256': digest_file(index),
        'slides': records, 'preparer_sha256': digest_file(Path(__file__)),
        'synthetic': bool(bundle.config.get('synthetic', False))})
    return output


class SAMMILAssets:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('synthetic') != bool(bundle.config.get('synthetic', False))
                or digest_file(self.meta['index']) != self.meta.get('index_sha256')
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('SAM-MIL input identities, preparer, membership index or task changed')
        self.schema = validate_schema(bundle, self.meta['schema'])
        for row in bundle.rows.to_dict('records'):
            record = self.meta['slides'][row['slide_id']]
            if any(record[key] != row[key] for key in ['global_patient_id', 'subset', 'target']):
                raise ValueError('SAM-MIL patient mapping, labels or split changed')
        self.contract = digest_file(self.path)
        self.files = [self.path, Path(self.meta['index']),
            *[Path(record['path']) for record in self.meta['slides'].values()]]

    def load(self, row):
        record = self.meta['slides'][row['slide_id']]
        if (digest_file(record['path']) != record['sha256']
                or digest_file(resolve(self.bundle.root, row['path'])) != record['primary_sha256']):
            raise ValueError('SAM-MIL original frozen features or measured segmentation membership changed')
        value = read_groups(self.bundle, row, record['path'])
        if digest_object({key: value[key] for key in ['group_ids', 'members', 'areas']}) != record['groups_sha256']:
            raise ValueError('SAM-MIL membership row order, mask area or group identity changed')
        return value

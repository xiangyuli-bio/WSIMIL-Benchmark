"""One-to-one measured low-resolution RGB patches for released HDMIL.

The low-resolution patches cover the SAME physical rectangles as the frozen
high-resolution features, not neighboring tiles or a fabricated coarse grid.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'hdmil_matched_low_resolution_rgb'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'split': bundle.split, 'dimension': bundle.dim,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'hdmil'}})


def validate_schema(schema):
    if (schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('coordinate_frame') != 'level0_top_left_pixels'
            or schema.get('rgb_dtype') != 'uint8' or schema.get('rgb_layout') != 'NHWC'
            or schema.get('low_patch_pixels') != 16 or schema.get('high_patch_pixels') != 256
            or schema.get('low_magnification') != 1.25 or schema.get('high_magnification') != 20
            or any(not isinstance(schema.get(k), str) or not schema[k].strip() for k in
                   ['rgb_key', 'coords_key', 'coordinate_provenance', 'image_provenance', 'weights_provenance'])):
        raise ValueError('HDMIL requires measured uint8 RGB16 at1.25x paired to256px20x feature patches and explicit provenance')
    return schema


def read_rgb(bundle, row, path, schema):
    features, coords = bundle.load(row, require_coords=True)
    with h5py.File(path, 'r') as f:
        if any(schema[k] not in f for k in ['rgb_key', 'coords_key']):
            raise ValueError('HDMIL RGB archive lacks its registered images or coordinates')
        images, measured = f[schema['rgb_key']][:], f[schema['coords_key']][:]
        # Per-slide measured physical size allows mixed scanners; magnification
        # cannot be inferred from pixel dimensions alone.
        scale = f.attrs.get('level0_mpp')
        high_mpp, low_mpp = f.attrs.get('high_mpp'), f.attrs.get('low_mpp')
        if (any(not isinstance(v, (int, float, np.number)) or not np.isfinite(v) or v <= 0
                for v in [scale, high_mpp, low_mpp])
                or not np.isclose(low_mpp, 16*high_mpp, rtol=1e-9, atol=0)):
            raise ValueError('HDMIL each slide must declare measured high/low/level0 MPP and equal physical fields of view')
        side = 256*float(high_mpp)/float(scale)
    if (images.dtype != np.uint8 or images.shape != (len(features), 16, 16, 3)
            or measured.shape != (len(features), 2) or not np.issubdtype(measured.dtype, np.integer)
            or (measured < 0).any() or len(np.unique(measured, axis=0)) != len(features)
            or not torch.equal(coords, torch.from_numpy(measured))):
        raise ValueError('HDMIL RGB count, dtype, shape or measured feature row order differs')
    if len(features) < 2:
        raise ValueError('HDMIL MobileNet BatchNorm requires at least two true patches; no synthetic padding')
    if row.get('width_level0') and row.get('height_level0'):
        extent = np.asarray([int(row['width_level0']), int(row['height_level0'])])
        if (measured+side > extent).any():
            raise ValueError('HDMIL measured RGB/feature patch exceeds the original WSI')
    return features, torch.from_numpy(images), dict(level0_mpp=float(scale), high_mpp=float(high_mpp),
                                                   low_mpp=float(low_mpp), side_level0=side)


def prepare(bundle, index, schema, output):
    schema = validate_schema(schema)
    index, output = resolve(bundle.root, index), resolve(bundle.root, output)
    if output.exists():
        raise ValueError('HDMIL manifest exists; prepare a new explicit version')
    table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (not {'slide_id', 'path'} <= set(table) or table[['slide_id', 'path']].eq('').any().any()
            or table.slide_id.duplicated().any() or set(table.slide_id) != set(bundle.rows.slide_id)):
        raise ValueError('HDMIL RGB index must cover the exact task slides once')
    lookup, records, used = table.set_index('slide_id'), {}, set()
    for row in bundle.rows.to_dict('records'):
        path = resolve(bundle.root, lookup.loc[row['slide_id'], 'path']).resolve()
        if path in used:
            raise ValueError('HDMIL different slides cannot share a low-resolution image archive')
        used.add(path)
        features, _, physical = read_rgb(bundle, row, path, schema)
        records[row['slide_id']] = dict(path=str(path), sha256=digest_file(path),
            primary_sha256=digest_file(resolve(bundle.root, row['path'])), nodes=len(features), physical=physical,
            **{k: row[k] for k in ['global_patient_id', 'subset', 'target']})
    write_json(output, dict(schema_version=1, kind=KIND, schema=schema, encoder_identity=encoder_identity(bundle),
        index=str(index), index_sha256=digest_file(index), slides=records, preparer_sha256=digest_file(Path(__file__))))
    return output


class HDMILAssets:
    def __init__(self, bundle):
        self.bundle = bundle
        self.path = resolve(bundle.root, bundle.encoder['hdmil']['manifest'])
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)
                or digest_file(self.meta['index']) != self.meta.get('index_sha256')):
            raise ValueError('HDMIL input source, split, preparation or encoder identity changed')
        self.schema = validate_schema(self.meta['schema'])
        for row in bundle.rows.to_dict('records'):
            if any(self.meta['slides'][row['slide_id']][k] != row[k] for k in ['global_patient_id', 'subset', 'target']):
                raise ValueError('HDMIL registered patient, label or split differs')
        self.contract = digest_file(self.path)
        self.files = [self.path, Path(self.meta['index']), *[Path(x['path']) for x in self.meta['slides'].values()]]

    def load(self, row):
        record = self.meta['slides'][row['slide_id']]
        if (digest_file(record['path']) != record['sha256']
                or digest_file(resolve(self.bundle.root, row['path'])) != record['primary_sha256']):
            raise ValueError('HDMIL registered RGB or feature archive changed')
        x, images, physical = read_rgb(self.bundle, row, record['path'], self.schema)
        if len(x) != record['nodes'] or physical != record['physical']:
            raise ValueError('HDMIL actual measured physical patch mapping changed')
        return x, images

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        record = self.meta['slides'][row['slide_id']]
        return dict(manifest_sha256=self.contract, rgb_sha256=record['sha256'],
                    primary_sha256=record['primary_sha256'], nodes=record['nodes'], physical=record['physical'])

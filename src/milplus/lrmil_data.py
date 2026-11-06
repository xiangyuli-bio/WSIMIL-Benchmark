"""Measured RGB5x and full HR CONCH intermediates for complete LRMIL training."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import resolve
from .focus_network import WEIGHTS_BYTES, WEIGHTS_SHA
from .io import digest_file, digest_object, read_json, write_json
from .lrmil_source import SOURCE_LAYERS

KIND = 'lrmil_measured_lr_rgb_and_hr_intermediates'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'split': bundle.split, 'dimension': bundle.dim,
        'encoder': {key: value for key, value in bundle.encoder.items() if key != 'lrmil'}})


def validate_schema(bundle, schema):
    if (bundle.task_type != 'classification' or bundle.dim != 768
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('coordinate_frame') != 'level0_top_left_pixels'
            or schema.get('rgb_dtype') != 'uint8' or schema.get('rgb_layout') != 'NHWC'
            or schema.get('low_patch_pixels') != 256 or schema.get('high_patch_pixels') != 256
            or schema.get('low_magnification') != 5 or schema.get('high_magnification') != 20
            or schema.get('teacher_representation') != 'CONCH_native_visual_trunk_normalized_CLS_768'
            or schema.get('teacher_layers') != list(SOURCE_LAYERS)
            or schema.get('pretrained_weights_sha256') != WEIGHTS_SHA
            or any(not isinstance(schema.get(key), str) or not schema[key].strip() for key in
                ['rgb_key', 'low_coords_key', 'high_coords_key', 'teacher_final_key',
                 'teacher_layer_prefix', 'coordinate_provenance', 'image_provenance',
                 'teacher_feature_provenance', 'weights_provenance', 'pretrained_checkpoint'])):
        raise ValueError('LRMIL needs full native768 CONCH intermediates and measured RGB256 at5x/20x')
    return schema


def read_slide(bundle, row, path, schema):
    primary, primary_coords = bundle.load(row, require_coords=True)
    with h5py.File(path, 'r') as archive:
        keys = [schema[key] for key in ['rgb_key', 'low_coords_key', 'high_coords_key', 'teacher_final_key']]
        keys += [schema['teacher_layer_prefix']+str(layer) for layer in SOURCE_LAYERS]
        if any(key not in archive for key in keys):
            raise ValueError('LRMIL archive is missing actual RGB, coordinates or a selected teacher layer')
        rgb, low, high, final = [archive[key][:] for key in keys[:4]]
        layers = {layer: archive[schema['teacher_layer_prefix']+str(layer)][:] for layer in SOURCE_LAYERS}
        physical = {key: archive.attrs.get(key) for key in ['level0_mpp', 'high_mpp', 'low_mpp']}
    if (any(not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0
            for value in physical.values())
            or not np.isclose(physical['low_mpp'], 4*physical['high_mpp'], rtol=1e-9, atol=0)):
        raise ValueError('LRMIL measured low/high MPP must differ by four')
    if (rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (256, 256, 3)
            or len(rgb) < 2 or low.shape != (len(rgb), 2) or high.shape != (len(primary), 2)
            or any(not np.issubdtype(coords.dtype, np.integer) or (coords < 0).any()
                   or len(np.unique(coords, axis=0)) != len(coords) for coords in [low, high])
            or not torch.equal(primary_coords, torch.from_numpy(high))
            or any(value.shape != (len(high), 768) or not np.issubdtype(value.dtype, np.floating)
                   or not np.isfinite(value).all() or not np.isfinite(value.astype(np.float32)).all()
                   for value in [final, *layers.values()])
            or not torch.equal(primary, torch.from_numpy(final.astype(np.float32)))):
        raise ValueError('LRMIL measured RGB, full768 feature layers or primary row order differs')
    factor = float(physical['level0_mpp']/physical['high_mpp'])
    low_hr, high_hr = low.astype(np.float64)*factor, high.astype(np.float64)*factor
    if any(not np.isfinite(coords).all() or (coords > np.iinfo(np.int64).max-1024).any()
           or not np.equal(coords, np.floor(coords)).all() for coords in [low_hr, high_hr]):
        raise ValueError('LRMIL released matching requires exact integer coordinates in measured20x pixels')
    low_hr, high_hr = low_hr.astype(np.int64), high_hr.astype(np.int64)
    links = []
    for x, y in low_hr:
        mask = (high_hr[:, 0] >= x) & (high_hr[:, 0] <= x+768)
        mask &= (high_hr[:, 1] >= y) & (high_hr[:, 1] <= y+768)
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError('LRMIL low-resolution patch has no actual matched high-resolution teacher region')
        # The actual released patch objective permits partial tissue grids.
        # Retain every measured match without manufacturing the missing patches.
        links.append(indices.tolist())
    if row.get('width_level0') and row.get('height_level0'):
        extent = np.asarray([int(row['width_level0']), int(row['height_level0'])], dtype=np.float64)
        if ((low+256*physical['low_mpp']/physical['level0_mpp'] > extent).any()
                or (high+256*physical['high_mpp']/physical['level0_mpp'] > extent).any()):
            raise ValueError('LRMIL measured patch extends beyond the original WSI')
    return {'images': torch.from_numpy(rgb), 'coords_5x': torch.from_numpy(low_hr),
        'coords_20x': torch.from_numpy(high_hr), 'teacher_final': primary,
        'teacher_layers': {key: torch.from_numpy(value.astype(np.float32)) for key, value in layers.items()},
        'links_5x_to_20x': links,
        'physical': {key: float(value) for key, value in physical.items()}}


def prepare(bundle, index, schema, output):
    schema = validate_schema(bundle, schema)
    index, output = resolve(bundle.root, index), resolve(bundle.root, output)
    if output.exists():
        raise ValueError('LRMIL manifest exists; prepare a new explicit version')
    checkpoint = resolve(bundle.root, schema['pretrained_checkpoint']).resolve()
    if (not checkpoint.is_file() or checkpoint.stat().st_size != WEIGHTS_BYTES
            or digest_file(checkpoint) != WEIGHTS_SHA):
        raise ValueError('LRMIL registered actual pretrained CONCH checkpoint is missing or altered')
    table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (not {'slide_id', 'path'} <= set(table) or table[['slide_id', 'path']].eq('').any().any()
            or table.slide_id.duplicated().any() or set(table.slide_id) != set(bundle.rows.slide_id)):
        raise ValueError('LRMIL input index must contain exactly the registered task slides once')
    lookup = table.set_index('slide_id')
    records, used = {}, set()
    for row in bundle.rows.to_dict('records'):
        path = resolve(bundle.root, lookup.loc[row['slide_id'], 'path']).resolve()
        if path in used:
            raise ValueError('LRMIL different slides cannot share a single input archive')
        used.add(path)
        data = read_slide(bundle, row, path, schema)
        records[row['slide_id']] = {'path': str(path), 'sha256': digest_file(path),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'low_nodes': len(data['images']), 'high_nodes': len(data['teacher_final']),
            'links_sha256': digest_object(data['links_5x_to_20x']), 'physical': data['physical'],
            **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}
    write_json(output, {'schema_version': 1, 'kind': KIND, 'schema': schema,
        'encoder_identity': encoder_identity(bundle), 'index': str(index), 'index_sha256': digest_file(index),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': WEIGHTS_SHA,
        'slides': records, 'preparer_sha256': digest_file(Path(__file__))})
    return output


class LRMILAssets:
    def __init__(self, bundle, manifest):
        self.bundle = bundle
        self.path = resolve(bundle.root, manifest)
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)
                or digest_file(self.meta['index']) != self.meta.get('index_sha256')
                or digest_file(self.meta['checkpoint']) != self.meta.get('checkpoint_sha256')
                or self.meta['checkpoint_sha256'] != WEIGHTS_SHA):
            raise ValueError('LRMIL inputs, pretrained initialization or preparer identity changed')
        self.schema = validate_schema(bundle, self.meta['schema'])
        for row in bundle.rows.to_dict('records'):
            record = self.meta['slides'][row['slide_id']]
            if any(record[key] != row[key] for key in ['global_patient_id', 'subset', 'target']):
                raise ValueError('LRMIL task labels, patient mapping or split changed')
        self.contract = digest_file(self.path)
        self.files = [self.path, Path(self.meta['index']), Path(self.meta['checkpoint']),
                      *[Path(record['path']) for record in self.meta['slides'].values()]]
        self._verified = {}

    def _check_archive(self, row):
        record = self.meta['slides'][row['slide_id']]
        path = Path(record['path'])
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            if digest_file(path) != record['sha256']:
                raise ValueError('LRMIL actual measured images or teacher features changed')
            self._verified[str(path)] = signature
        return record

    def high(self, row):
        self._check_archive(row)
        record = self.meta['slides'][row['slide_id']]
        if digest_file(resolve(self.bundle.root, row['path'])) != record['primary_sha256']:
            raise ValueError('LRMIL primary high-resolution feature input changed')
        return self.bundle.load(row, require_coords=True)[0]

    def low(self, row, start=0, stop=None):
        record = self._check_archive(row)
        if stop is None:
            stop = record['low_nodes']
        if not 0 <= start < stop <= record['low_nodes']:
            raise ValueError('LRMIL actual LR patch range is invalid')
        with h5py.File(record['path'], 'r') as archive:
            images = archive[self.schema['rgb_key']][start:stop]
        return torch.from_numpy(images)

    def patch(self, row, index):
        record = self._check_archive(row)
        if type(index) is not int or not 0 <= index < record['low_nodes']:
            raise ValueError('LRMIL actual LR patch index is invalid')
        factor = record['physical']['level0_mpp']/record['physical']['high_mpp']
        with h5py.File(record['path'], 'r') as archive:
            low = (archive[self.schema['low_coords_key']][index]*factor).astype(np.int64)
            high = (archive[self.schema['high_coords_key']][:]*factor).astype(np.int64)
            mask = (high[:, 0] >= low[0]) & (high[:, 0] <= low[0]+768)
            mask &= (high[:, 1] >= low[1]) & (high[:, 1] <= low[1]+768)
            links = np.flatnonzero(mask)
            if not len(links):
                raise ValueError('LRMIL actual low-resolution patch lost its teacher matches')
            return {'image': torch.from_numpy(archive[self.schema['rgb_key']][index]),
                'coords_5x': torch.from_numpy(low), 'coords_20x': torch.from_numpy(high[links]),
                'teacher_final': torch.from_numpy(archive[self.schema['teacher_final_key']][links].astype(np.float32)),
                'teacher_layers': {layer: torch.from_numpy(archive[self.schema['teacher_layer_prefix']+str(layer)][links].astype(np.float32))
                                   for layer in SOURCE_LAYERS}, 'links': torch.from_numpy(links)}

    def coordinates(self, row):
        record = self._check_archive(row)
        factor = record['physical']['level0_mpp']/record['physical']['high_mpp']
        with h5py.File(record['path'], 'r') as archive:
            return tuple(torch.from_numpy((archive[self.schema[key]][:]*factor).astype(np.int64))
                         for key in ['low_coords_key', 'high_coords_key'])

    def load(self, row):
        record = self.meta['slides'][row['slide_id']]
        if (digest_file(record['path']) != record['sha256']
                or digest_file(resolve(self.bundle.root, row['path'])) != record['primary_sha256']):
            raise ValueError('LRMIL actual measured images or teacher features changed')
        result = read_slide(self.bundle, row, record['path'], self.schema)
        if (len(result['images']) != record['low_nodes'] or len(result['teacher_final']) != record['high_nodes']
                or digest_object(result['links_5x_to_20x']) != record['links_sha256']
                or result['physical'] != record['physical']):
            raise ValueError('LRMIL measured cross-resolution matching changed')
        return result

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        record = self.meta['slides'][row['slide_id']]
        return {'manifest_sha256': self.contract, 'source_sha256': record['sha256'],
            'primary_sha256': record['primary_sha256'], 'low_nodes': record['low_nodes'],
            'high_nodes': record['high_nodes'], 'links_sha256': record['links_sha256'],
            'checkpoint_sha256': self.meta['checkpoint_sha256']}

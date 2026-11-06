"""Measured RGB224 and observed TRAIN LPA for the complete LNPL reference.

Preparation never invents patch labels, weak scores, patient IDs or an FM.
The weak-classifier recipe is a disclosed reconstruction: its original training
configuration was not included in the LNPL release or the cited SuNCEt source.
"""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .io import digest_file, digest_object, read_json, write_json

KIND = 'lnplmil_measured_RGB224_observed_LPA_reference'


def embedded(file, key):
    if key not in file or isinstance(file.get(key, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
        raise ValueError('LNPL actual RGB/annotations must use registered embedded arrays')
    value = file[key]
    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
        raise ValueError('LNPL cannot silently follow virtual/external data arrays')
    return value


def inspect(path, row, schema):
    with h5py.File(path, 'r') as file:
        pixels = embedded(file, schema['pixels_key'])
        if (pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (224, 224, 3)
                or len(pixels) < 200):
            raise ValueError('LNPL needs N>=200 actual original RGB224 patches, not feature pseudo-images')
        ids = embedded(file, schema['patch_ids_key']).asstr()[:].tolist()
        coords = embedded(file, schema['coords_key'])[:]
        if (len(ids) != len(pixels) or len(set(ids)) != len(ids)
                or any(not isinstance(x, str) or not x.strip() for x in ids)
                or coords.dtype != np.int64 or coords.shape != (len(pixels), 2)
                or (coords < 0).any() or (coords >= 240000).any()
                or len(np.unique(coords, axis=0)) != len(coords)):
            raise ValueError('LNPL patch IDs/level0 coordinates must bind actual RGB row order and original positional range')
        indices = embedded(file, schema['labelled_indices_key'])[:]
        labels = embedded(file, schema['patch_labels_key'])[:]
        if (indices.dtype != np.int64 or labels.dtype != np.int64 or indices.ndim != 1
                or labels.shape != indices.shape or len(set(indices.tolist())) != len(indices)
                or len(indices) and (indices.min() < 0 or indices.max() >= len(pixels)
                                    or labels.min() < 0 or labels.max() >= len(schema['weak_classes']))):
            raise ValueError('Observed LPA must bind unique actual patch indices and explicit class IDs')
        if row['subset'] != 'train' and len(indices):
            raise ValueError('Only TRAIN patients may supply observed weak-classifier LPA')
        return {'shape': list(pixels.shape), 'patch_ids_sha256': digest_object(ids),
                'coords_sha256': digest_object(coords.tolist()), 'labelled_indices': indices.tolist(),
                'observed_patch_labels': labels.tolist()}


def prepare(root, rows, schema, output):
    """Bind explicit raw rows before this complete method is registered in Bundle."""
    from .lnplmil_weak import assets
    root = Path(root).expanduser().resolve()
    required = {'pixels_key', 'patch_ids_key', 'coords_key', 'labelled_indices_key', 'patch_labels_key',
                'image_provenance', 'annotation_provenance', 'patch_provenance', 'initial_assets'}
    if (not isinstance(schema, dict) or schema.get('kind') != KIND or schema.get('schema_version') != 1
            or schema.get('layout') != 'NHWC' or schema.get('color') != 'RGB'
            or schema.get('patch_pixels') != 224 or schema.get('coordinate_frame') != 'level0_top_left_pixels'
            or schema.get('magnification') not in {20, 40} or not isinstance(schema.get('mpp'), (int, float))
            or not np.isfinite(schema['mpp']) or schema['mpp'] <= 0
            or any(not isinstance(schema.get(k), str) or not schema[k].strip() for k in required)
            or not isinstance(schema.get('weak_classes'), list) or len(schema['weak_classes']) not in {2, 9}
            or len(set(schema['weak_classes'])) != len(schema['weak_classes'])
            or any(not isinstance(c, str) or not c.strip() for c in schema['weak_classes'])
            or not isinstance(schema.get('positive_weak_class_ids'), list)
            or not schema['positive_weak_class_ids']
            or len(set(schema['positive_weak_class_ids'])) != len(schema['positive_weak_class_ids'])
            or any(type(i) is not int or not 0 <= i < len(schema['weak_classes'])
                   for i in schema['positive_weak_class_ids'])
            or len(schema['positive_weak_class_ids']) == len(schema['weak_classes'])):
        raise ValueError('Declare actual RGB224/LPA, measured scale, explicit weak classes/positive mapping and reference assets')
    if not isinstance(rows, list) or not rows:
        raise ValueError('LNPL requires explicit slide/patient/partition rows')
    slides, patients, observed = {}, {}, set()
    for row in rows:
        if (any(not isinstance(row.get(k), str) or not row[k].strip()
                for k in ['slide_id', 'global_patient_id', 'patient_id', 'path'])
                or row.get('subset') not in {'train', 'val', 'test'}
                or type(row.get('target')) is not int or row['target'] not in {0, 1}
                or row['slide_id'] in slides):
            raise ValueError('LNPL raw rows need unique slides, real patient identities, binary targets and fixed partitions')
        pid = row['global_patient_id']
        if pid in patients and patients[pid] != row['subset']:
            raise ValueError('A patient cannot cross LNPL TRAIN/VAL/TEST boundaries')
        patients[pid] = row['subset']
        path = (root / row['path']).resolve()
        record = inspect(path, row, schema)
        if row['subset'] == 'train':
            observed.update(record['observed_patch_labels'])
        slides[row['slide_id']] = {**row, **record, 'path': str(path), 'sha256': digest_file(path)}
    if observed != set(range(len(schema['weak_classes']))):
        raise ValueError('Observed TRAIN LPA must actually cover every declared weak class')
    if any({r['target'] for r in slides.values() if r['subset'] == s} != {0, 1} for s in ['train', 'val', 'test']):
        raise ValueError('The complete binary LNPL reference needs both slide classes in all fixed subsets')
    asset_root = (root / schema['initial_assets']).resolve()
    assets(asset_root)
    if type(schema.get('synthetic', False)) is not bool:
        raise ValueError('Declare the LNPL synthetic-data flag as an actual boolean')
    manifest = {'kind': KIND, 'schema_version': 1, 'root': str(root), 'schema': schema,
                'synthetic': schema.get('synthetic', False),
                'slides': slides, 'initial_assets': str(asset_root),
                'preparer_sha256': digest_file(Path(__file__)),
                'annotation_usage': 'observed_TRAIN_only_no_slide_label_inheritance'}
    output = root / output
    if output.exists():
        raise FileExistsError('Do not replace a prior measured LNPL manifest')
    write_json(output, manifest)
    return output


class RawInputs:
    def __init__(self, manifest):
        from .lnplmil_weak import assets, ASSETS
        self.path = Path(manifest).expanduser().resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('kind') != KIND or self.meta.get('schema_version') != 1
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))):
            raise ValueError('LNPL prepared actual input/reader contract changed')
        self.schema = self.meta['schema']
        self.asset_root = assets(self.meta['initial_assets'])
        self.contract = digest_file(self.path)
        self.rows = list(self.meta['slides'].values())
        self._verified = {}
        self.files = [self.path] + [self.asset_root / name for name in ASSETS]
        # Detect altered patient/LPA metadata even when an attacker edits the JSON.
        patients = {}
        for row in self.rows:
            pid = row['global_patient_id']
            if pid in patients and patients[pid] != row['subset']:
                raise ValueError('LNPL prepared patients cross partitions')
            patients[pid] = row['subset']
            if row['subset'] != 'train' and row['labelled_indices']:
                raise ValueError('Held-out LPA may not enter the LNPL reference')
        self.labelled = [(row['slide_id'], index, target)
                         for row in self.rows if row['subset'] == 'train'
                         for index, target in zip(row['labelled_indices'], row['observed_patch_labels'])]
        if not self.labelled:
            raise ValueError('LNPL needs genuine TRAIN LPA before scoring')

    def verify(self, row):
        path = Path(row['path'])
        info = path.stat()
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            actual = inspect(path, row, self.schema)
            if digest_file(path) != row['sha256'] or any(actual[k] != row[k] for k in actual):
                raise ValueError('LNPL RGB, observed annotations or ordered patch identities changed')
            self._verified[str(path)] = signature

    @contextmanager
    def open(self, row):
        self.verify(row)
        with h5py.File(row['path'], 'r') as file:
            yield file

    def labelled_batch(self, entries):
        images, targets = [], []
        for slide, index, target in entries:
            row = self.meta['slides'][slide]
            if row['subset'] != 'train' or (index, target) not in set(zip(row['labelled_indices'], row['observed_patch_labels'])):
                raise ValueError('A weak minibatch cannot invent labels or include held-out patches')
            with self.open(row) as file:
                images.append(file[self.schema['pixels_key']][index])
            targets.append(target)
        return images, targets


def bundle_contract(bundle):
    if (bundle.task_type != 'classification' or len(bundle.classes) != 2 or bundle.dim != 3
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or any(k in bundle.encoder for k in ['raw_images', 'raw_regions', 'hybrid_supervision',
                'macnet', 'pmil_prototypical', 'icmil_coupled', 'sparseconvmil', 'attention2majority_raw'])):
        raise ValueError('The full LNPL diagnosis protocol requires its own binary slide/RGB224/TRAIN-LPA input')


def prepare_bundle(bundle, schema, output):
    bundle_contract(bundle)
    schema = dict(schema)
    synthetic = bundle.config.get('synthetic', False)
    if type(synthetic) is not bool or schema.get('synthetic', synthetic) != synthetic:
        raise ValueError('LNPL synthetic flag must agree with the explicit task bundle')
    schema['synthetic'] = synthetic
    rows = []
    for row in bundle.rows.to_dict('records'):
        rows.append({**{k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'subset']},
                     'path': str((bundle.root / row['path']).resolve()), 'target': int(row['target'])})
    return prepare(bundle.root, rows, schema, output)


class BundleInputs(RawInputs):
    def __init__(self, bundle, manifest):
        bundle_contract(bundle)
        super().__init__(bundle.root / manifest)
        if (Path(self.meta['root']) != bundle.root or set(self.meta['slides']) != set(bundle.rows.slide_id)
                or type(bundle.config.get('synthetic', False)) is not bool
                or self.meta['synthetic'] != bundle.config.get('synthetic', False)):
            raise ValueError('LNPL raw manifest and public bundle no longer describe the same task')
        for row in bundle.rows.to_dict('records'):
            original = self.meta['slides'][row['slide_id']]
            if (original['path'] != str((bundle.root / row['path']).resolve())
                    or any(original[k] != row[k] for k in ['patient_id', 'global_patient_id', 'subset', 'target'])):
                raise ValueError('LNPL raw patients/labels/partitions differ from the public task')

    def inventory(self, row, deep=False):
        record = self.meta['slides'][row['slide_id']]
        if deep:
            self.verify(record)
        return record

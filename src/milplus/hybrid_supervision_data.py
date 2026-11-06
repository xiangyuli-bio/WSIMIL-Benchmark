"""Measured original512RGB bags, real fine masks and patient-safe hybrid inputs."""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json
from .hybrid_supervision_source import validate_weights

KIND = 'hybrid_supervision_paper_original_RGB512'


def identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
                          'encoder': {k: v for k, v in bundle.encoder.items() if k != 'hybrid_supervision'}})


def schema_contract(bundle, schema):
    required = ['pixel_key', 'patch_ids_key', 'coords_key', 'fine_indices_key', 'fine_masks_key',
                'image_provenance', 'patch_provenance', 'fine_annotation_provenance', 'initial_weights']
    if (bundle.task_type != 'classification' or len(bundle.classes) != 2 or bundle.dim != 3
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or not isinstance(schema, dict) or schema.get('schema_version') != 1
            or schema.get('kind') != KIND or schema.get('layout') != 'NHWC'
            or schema.get('dtype') != 'uint8' or schema.get('color') != 'RGB'
            or schema.get('patch_pixels') != 512 or schema.get('stride_pixels') != 384
            or schema.get('mpp') != .5 or schema.get('coordinate_frame') != 'level0_top_left_pixels'
            or schema.get('normalization') != 'RGB_uint8_to_float32_div255_no_ImageNet_normalization'
            or any(not isinstance(schema.get(k), str) or not schema[k].strip() for k in required)
            or any(k in bundle.encoder for k in ['macnet', 'pmil_prototypical', 'icmil_coupled',
                'raw_images', 'raw_regions', 'c2c', 'sparseconvmil', 'attention2majority_raw'])):
        raise ValueError('Hybrid paper input needs binary slide labels, actual512RGB, stride384/mpp0.5 and real fine masks')
    return schema


def embedded(file, key):
    if key not in file or isinstance(file.get(key, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
        raise ValueError('Hybrid RGB/masks/identities must be embedded rather than unregistered external links')
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external:
        raise ValueError('Hybrid RGB and annotation arrays cannot use virtual/external storage')
    return dataset


def describe(bundle, row, schema):
    path = resolve(bundle.root, row['path'])
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        raise ValueError('Use actual HDF5 RGB512 and original patch/annotation identities')
    with h5py.File(path, 'r') as f:
        pixels = embedded(f, schema['pixel_key'])
        if pixels.ndim != 4 or pixels.shape[0] < 16 or pixels.shape[1:] != (512, 512, 3) or pixels.dtype != np.uint8:
            raise ValueError('Hybrid Top16 needs at least16 actual original512RGB patches, without resize/padding/FM substitutes')
        ids = embedded(f, schema['patch_ids_key']).asstr()[:].tolist()
        coords = embedded(f, schema['coords_key'])[:]
        if (len(ids) != len(pixels) or len(set(ids)) != len(ids)
                or any(not isinstance(s, str) or not s.strip() for s in ids)
                or coords.dtype != np.int64 or coords.shape != (len(pixels), 2)
                or (coords < 0).any() or len(np.unique(coords, axis=0)) != len(pixels)):
            raise ValueError('Hybrid coordinates and unique patch IDs must preserve actual stored RGB row order')
        indices = embedded(f, schema['fine_indices_key'])[:]
        masks = embedded(f, schema['fine_masks_key'])
        if (indices.dtype != np.int64 or indices.ndim != 1 or len(np.unique(indices)) != len(indices)
                or masks.dtype != np.uint8 or masks.shape != (len(indices), 512, 512)
                or len(indices) and (indices.min() < 0 or indices.max() >= len(pixels))):
            raise ValueError('Hybrid fine masks must bind unique actual512RGB patch indices')
        if row['subset'] != 'train' and len(indices):
            raise ValueError('Only TRAIN patients may supply hybrid fine masks to the fitting protocol')
        positive_masks = 0
        for i in range(len(masks)):
            a = masks[i]
            if not np.isin(a, [0, 255]).all():
                raise ValueError('Hybrid original fine masks are observed binary0/255 labels, not generated pseudo masks')
            if row['target'] == 0 and a.any():
                raise ValueError('Hybrid negative slide labels contradict positive fine annotations')
            positive_masks += int(a.any())
        return {'shape': list(pixels.shape), 'patch_ids_sha256': digest_object(ids),
                'coordinates_sha256': digest_object(coords.tolist()),
                'fine_indices': indices.tolist(), 'fine_positive_mask_count': positive_masks}


def prepare(bundle, schema, output):
    schema_contract(bundle, schema)
    output = resolve(bundle.root, output)
    if output.exists():
        raise FileExistsError('Create a new measured hybrid manifest without replacing prior evidence')
    weights, weights_sha = validate_weights(resolve(bundle.root, schema['initial_weights']))
    slides = {}
    for row in bundle.rows.to_dict('records'):
        path = resolve(bundle.root, row['path']).resolve()
        slides[row['slide_id']] = {**describe(bundle, row, schema), 'path': str(path), 'sha256': digest_file(path),
                                 **{k: row[k] for k in ['global_patient_id', 'subset', 'target']}}
    if not any(r['fine_positive_mask_count'] for r in slides.values()):
        raise ValueError('Hybrid initialization needs genuine observed positive TRAIN fine masks')
    write_json(output, {'schema_version': 1, 'kind': KIND, 'schema': schema,
        'encoder_identity': identity(bundle), 'initial_weights': str(weights.resolve()),
        'initial_weights_sha256': weights_sha, 'slides': slides,
        'synthetic': bool(bundle.config.get('synthetic', False)), 'preparer_sha256': digest_file(Path(__file__))})
    return output


class HybridInputs:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.meta = read_json(self.path)
        if (self.meta.get('kind') != KIND or self.meta.get('schema_version') != 1
                or self.meta.get('encoder_identity') != identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('synthetic') != bool(bundle.config.get('synthetic', False))
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('Hybrid prepared source/data/reader identity changed')
        self.schema = schema_contract(bundle, self.meta['schema'])
        self.weights, sha = validate_weights(self.meta['initial_weights'])
        if sha != self.meta['initial_weights_sha256']:
            raise ValueError('Hybrid original DLA34 weight bytes changed')
        for row in bundle.rows.to_dict('records'):
            record = self.meta['slides'][row['slide_id']]
            if (record['path'] != str(resolve(bundle.root, row['path']).resolve())
                    or any(record[k] != row[k] for k in ['global_patient_id', 'subset', 'target'])):
                raise ValueError('Hybrid pixels/fine annotations/patient labels/partitions changed')
        self.files = [self.path, self.weights]
        self.contract = digest_file(self.path)
        self._verified = {}

    def verify(self, row):
        path = resolve(self.bundle.root, row['path'])
        st = path.stat()
        signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            record = self.meta['slides'][row['slide_id']]
            if digest_file(path) != record['sha256'] or any(record[k] != v for k, v in describe(self.bundle, row, self.schema).items()):
                raise ValueError('Hybrid actual RGB/fine annotation/patch identity bytes changed')
            self._verified[str(path)] = signature

    @contextmanager
    def open(self, row):
        self.verify(row)
        with h5py.File(resolve(self.bundle.root, row['path']), 'r') as f:
            yield f

    def count(self, row):
        return self.meta['slides'][row['slide_id']]['shape'][0]

    def inventory(self, row, deep=False):
        if deep:
            self.verify(row)
        return self.meta['slides'][row['slide_id']]

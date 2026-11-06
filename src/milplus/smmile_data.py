"""SMMILe's physical NIC grid and frozen, input-bound superpixel assets.

Axes deliberately follow the author: [feature, x, y], rather than image [y, x].
No label or cohort fitting enters the SLIC preparation.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import load_array, resolve
from .io import digest_file, digest_object, write_json
from .upstream import load_definitions, verified_source

DEFAULT_CELLS = 1_000_000
DEFAULT_BYTES = 512 * 1024**2


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


class NICGrid:
    """Map registered patch rows onto the exact physical lattice, retaining holes."""
    def __init__(self, coords, stride, coordinate_frame='level0_pixels',
                 max_cells=DEFAULT_CELLS, max_bytes=DEFAULT_BYTES):
        if coordinate_frame not in {'level0_pixels', 'micrometers'}:
            raise ValueError('SMMILe coordinates require level0_pixels or micrometers')
        if isinstance(stride, bool) or not isinstance(stride, (int, float)) or not math.isfinite(stride) or stride <= 0:
            raise ValueError('SMMILe patch_stride must be finite positive in the declared coordinate frame')
        self.max_cells = positive_int(max_cells, 'max_grid_cells')
        self.max_bytes = positive_int(max_bytes, 'max_dense_bytes')
        xy = np.asarray(coords)
        real_dtype = np.issubdtype(xy.dtype, np.integer) or np.issubdtype(xy.dtype, np.floating)
        if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not real_dtype:
            raise ValueError('SMMILe requires nonempty numeric [N,2] coordinates')
        xy = xy.astype(np.float64)
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 2**53-1).any() or len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError('SMMILe coordinates must be finite, nonnegative and unique')
        if coordinate_frame == 'level0_pixels' and (not np.equal(xy, np.floor(xy)).all() or stride != int(stride)):
            raise ValueError('Level0 pixel coordinates and patch_stride must be integers')
        self.origin = xy.min(0)
        lattice = (xy-self.origin)/stride
        rounded = np.rint(lattice)
        if not np.allclose(lattice, rounded, rtol=0, atol=1e-6):
            raise ValueError('Coordinates do not lie on the declared SMMILe patch lattice')
        # Check Python integer sizes before any int64 conversion or dense allocation.
        self.shape = tuple(int(rounded[:, i].max())+1 for i in range(2))
        self.cells = math.prod(self.shape)
        if self.cells > self.max_cells or max(self.shape) > np.iinfo(np.int64).max:
            raise ValueError('SMMILe physical grid exceeds max_grid_cells; cropping/rescaling is never implicit')
        self.indices = rounded.astype(np.int64)
        if len(np.unique(self.indices, axis=0)) != len(xy):
            raise ValueError('Different coordinates collide on the SMMILe lattice')
        self.order = np.lexsort((self.indices[:, 1], self.indices[:, 0]))
        self.inverse = np.argsort(self.order)
        self.reserve(1, np.dtype(np.uint8).itemsize)
        self.mask = np.zeros(self.shape, dtype=np.uint8)
        self.mask[tuple(self.indices.T)] = 1

    def reserve(self, channels, itemsize, copies=1):
        positive_int(channels, 'channels')
        positive_int(itemsize, 'itemsize')
        positive_int(copies, 'copies')
        if self.cells * (channels*itemsize*copies + 24) > self.max_bytes:
            raise ValueError('SMMILe dense grid exceeds max_dense_bytes before allocation')

    def scatter(self, features):
        if features.ndim != 2 or len(features) != len(self.indices) or features.shape[1] < 1:
            raise ValueError('NIC features must be aligned nonempty [N,D]')
        if isinstance(features, torch.Tensor):
            if not features.is_floating_point() or not torch.isfinite(features).all():
                raise ValueError('NIC features must be finite floating point')
            self.reserve(features.shape[1], features.element_size())
            output = features.new_zeros((features.shape[1], *self.shape))
            idx = torch.from_numpy(self.indices).to(features.device)
            output[:, idx[:, 0], idx[:, 1]] = features.T
        else:
            features = np.asarray(features)
            if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
                raise ValueError('NIC features must be finite floating point')
            self.reserve(features.shape[1], features.dtype.itemsize)
            output = np.zeros((features.shape[1], *self.shape), dtype=features.dtype)
            output[:, self.indices[:, 0], self.indices[:, 1]] = features.T
        return output

    def restore_rows(self, scores):
        """Undo np.where(mask==1) ordering for instance outputs and coordinates."""
        if scores.ndim < 1 or len(scores) != len(self.indices):
            raise ValueError('NIC instance scores must have one row per original patch')
        if isinstance(scores, torch.Tensor):
            return scores[torch.from_numpy(self.inverse).to(scores.device)]
        return scores[self.inverse]


def author_preprocessing():
    return load_definitions('smmile', 'superpixel_generation.py',
                            ['get_nic_with_coord', 'generate_adjacency_matrix'], {'np': np})


def load_spatial_inputs(bundle, row):
    """Preserve stored coordinate precision instead of using Bundle's float32 view."""
    path = resolve(bundle.root, row.get('coords_path') or row['path'])
    xy = load_array(path, 'coords')
    real_dtype = np.issubdtype(xy.dtype, np.integer) or np.issubdtype(xy.dtype, np.floating)
    if xy.ndim != 2 or xy.shape[1] != 2 or not real_dtype or not np.isfinite(xy).all():
        raise ValueError('SMMILe requires finite coordinates aligned with stored patch rows')
    x, _ = bundle.load(row, require_coords=True)
    if xy.shape != (len(x), 2):
        raise ValueError('SMMILe coordinate/feature row alignment mismatch')
    return x, torch.from_numpy(np.asarray(xy, dtype=np.float64))


def generate_superpixels(features, layout, patches_per_superpixel=16, compactness=50.):
    """Literal author SLIC input and transpose convention, with declared edge repairs."""
    positive_int(patches_per_superpixel, 'patches_per_superpixel')
    if isinstance(compactness, bool) or not isinstance(compactness, (int, float)) or not math.isfinite(compactness) or compactness <= 0:
        raise ValueError('SMMILe SLIC compactness must be finite and positive')
    try:
        from skimage.segmentation import slic
    except ImportError as error:
        raise ImportError('SMMILe preparation needs the smmile optional dependency') from error
    x = np.asarray(features)
    # Source np.ones()*nan creates a float64 dense canvas; retain that convention.
    if x.ndim != 2 or len(x) != len(layout.indices) or not x.shape[1] or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError('SMMILe SLIC needs finite aligned floating features')
    layout.reserve(x.shape[1], 8, copies=3)
    span = x.max()-x.min()
    if not np.isfinite(span):
        raise ValueError('SMMILe feature dynamic range overflows its stored floating dtype')
    normalized = np.zeros_like(x) if span == 0 else (x-x.min())/span*255
    grid = layout.scatter(normalized.astype(np.float64))
    segments = max(10, int(len(x)/patches_per_superpixel))
    # Guard both the requested and the returned dense adjacency allocations.
    possible = min(segments, len(x)) + 1
    if possible**2 * 8 > layout.max_bytes:
        raise ValueError('SMMILe requested adjacency exceeds max_dense_bytes')
    sp = slic(np.transpose(grid, (1, 2, 0)), n_segments=segments,
              mask=layout.mask, compactness=compactness, start_label=1)
    # Author builds adjacency on transposed map, then transposes the map back in Dataset.
    if (int(sp.max())+1)**2 * 8 > layout.max_bytes:
        raise ValueError('SMMILe returned adjacency exceeds max_dense_bytes')
    adjacency = author_preprocessing().generate_adjacency_matrix(sp.T)
    validate_superpixels(sp, adjacency, layout)
    return np.asarray(sp, dtype=np.int64), np.asarray(adjacency, dtype=np.uint8)


def validate_superpixels(sp, adjacency, layout):
    if sp.shape != layout.shape or not np.issubdtype(sp.dtype, np.integer) or (sp < 0).any():
        raise ValueError('SMMILe superpixel map must be nonnegative integer [x,y] on the physical grid')
    if (sp[layout.mask == 0] != 0).any() or (sp[layout.mask == 1] < 1).any():
        raise ValueError('SMMILe superpixel map must match the exact tissue mask and holes')
    labels = np.unique(sp[layout.mask == 1])
    # Bound the allocation by observed cells, never by a possibly corrupt label ID.
    if not np.array_equal(labels, np.arange(1, len(labels)+1)):
        raise ValueError('SMMILe superpixel IDs must be contiguous positive labels')
    count = int(labels.max())+1
    if count**2 * 8 > layout.max_bytes:
        raise ValueError('SMMILe adjacency exceeds max_dense_bytes')
    if adjacency.shape != (count, count) or not np.issubdtype(adjacency.dtype, np.integer):
        raise ValueError('SMMILe adjacency must be integer [max_label+1,max_label+1]')
    if not np.isin(adjacency, [0, 1]).all():
        raise ValueError('SMMILe adjacency must be binary')
    expected = author_preprocessing().generate_adjacency_matrix(sp.T)
    if not np.array_equal(adjacency, expected):
        raise ValueError('SMMILe adjacency differs from the pinned author geometry')


def prepare_superpixels(bundle, output, stride, coordinate_frame='level0_pixels',
                        patches_per_superpixel=16, compactness=50., provenance='',
                        max_cells=DEFAULT_CELLS, max_bytes=DEFAULT_BYTES):
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('SMMILe preprocessing provenance is required')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a new SMMILe superpixel output directory')
    # Never alter task.yaml or an existing experiment. A failed build has no complete manifest.
    output.mkdir(parents=True)
    records = []
    source_commit = verified_source('smmile')[1]
    for i, row in enumerate(bundle.rows.to_dict('records')):
        x, coords = load_spatial_inputs(bundle, row)
        layout = NICGrid(coords.numpy(), stride, coordinate_frame, max_cells, max_bytes)
        sp, adj = generate_superpixels(x.numpy(), layout, patches_per_superpixel, compactness)
        path = output/f'{i:06d}.h5'
        with h5py.File(path, 'x') as handle:
            handle['superpixels'] = sp
            handle['adjacency'] = adj
        records.append({'slide_id': row['slide_id'], 'path': path.name, 'sha256': digest_file(path),
                        'features': array_identity(x.numpy()), 'coordinates': array_identity(coords.numpy()),
                        'grid_shape': list(layout.shape), 'origin': layout.origin.tolist(),
                        'superpixels': int(sp.max())})
    manifest = {'schema_version': 1, 'source_commit': source_commit, 'fm': bundle.fm,
                'weights': bundle.encoder.get('weights'), 'dimension': bundle.dim,
                'synthetic': bool(bundle.config.get('synthetic', False)),
                'coordinate_frame': coordinate_frame, 'patch_stride': stride,
                'patches_per_superpixel': patches_per_superpixel, 'compactness': compactness,
                'preprocessing': provenance, 'axes': 'feature,x,y; source transpose before adjacency',
                'max_grid_cells': max_cells, 'max_dense_bytes': max_bytes,
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scikit-image']},
                'normalization': 'per-slide global minmax*255; constant tensor -> zeros (explicit numerical repair)',
                'records': records}
    write_json(output/'manifest.json', manifest)
    return str(output/'manifest.json')


class SuperpixelTable:
    def __init__(self, bundle):
        self.bundle = bundle
        meta = bundle.encoder['smmile']
        if not isinstance(meta, dict) or set(meta) != {'manifest'}:
            raise ValueError('encoder.smmile requires exactly one manifest path')
        self.path = resolve(bundle.root, meta['manifest']).resolve()
        self.files = [self.path]
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get('schema_version') != 1 or m.get('source_commit') != verified_source('smmile')[1]:
            raise ValueError('SMMILe manifest schema or author source version mismatch')
        if m.get('fm') != bundle.fm or m.get('dimension') != bundle.dim or m.get('weights') != bundle.encoder.get('weights'):
            raise ValueError('SMMILe manifest FM identity mismatch')
        if type(m.get('synthetic')) is not bool or m['synthetic'] != bool(bundle.config.get('synthetic', False)):
            raise ValueError('SMMILe synthetic identity mismatch')
        if not isinstance(m.get('preprocessing'), str) or not m['preprocessing'].strip():
            raise ValueError('SMMILe preprocessing provenance is required')
        records = m.get('records', [])
        if len(records) != len(bundle.rows) or {r['slide_id'] for r in records} != set(bundle.rows.slide_id):
            raise ValueError('SMMILe manifest must cover every registered slide exactly once')
        self.records = {r['slide_id']: r for r in records}

    def load(self, row):
        x, coords = load_spatial_inputs(self.bundle, row)
        record = self.records[row['slide_id']]
        if array_identity(x.numpy()) != record['features'] or array_identity(coords.numpy()) != record['coordinates']:
            raise ValueError('SMMILe superpixels are stale or patch rows/coordinates changed')
        m = self.manifest
        layout = NICGrid(coords.numpy(), m['patch_stride'], m['coordinate_frame'], m['max_grid_cells'], m['max_dense_bytes'])
        if list(layout.shape) != record['grid_shape'] or layout.origin.tolist() != record['origin']:
            raise ValueError('SMMILe grid metadata mismatch')
        path = resolve(self.path.parent, record['path'])
        if digest_file(path) != record['sha256']:
            raise ValueError('SMMILe superpixel asset hash mismatch')
        with h5py.File(path, 'r') as handle:
            if set(handle) != {'superpixels', 'adjacency'} or handle['superpixels'].shape != layout.shape:
                raise ValueError('Invalid SMMILe superpixel asset datasets')
            # Inspect shape before reading arbitrary HDF5 data into memory.
            for key in handle:
                ds = handle[key]
                if math.prod(ds.shape)*ds.dtype.itemsize > layout.max_bytes:
                    raise ValueError('SMMILe HDF5 dataset exceeds max_dense_bytes')
            sp, adj = handle['superpixels'][:], handle['adjacency'][:]
        validate_superpixels(sp, adj, layout)
        if int(sp.max()) != record['superpixels']:
            raise ValueError('SMMILe superpixel count mismatch')
        return x, coords, layout, sp, adj

    def inventory(self, row, deep=False):
        record = self.records[row['slide_id']]
        path = resolve(self.path.parent, record['path'])
        actual = digest_file(path)
        if actual != record['sha256']:
            raise ValueError('SMMILe superpixel asset hash mismatch')
        if deep:
            self.load(row)
        return {'path': str(path), 'sha256': actual, 'record_sha256': digest_object(record)}

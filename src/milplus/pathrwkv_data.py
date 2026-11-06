"""PathRWKV source sampling, physical coordinates and task-specific targets."""
import contextlib

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_array, resolve
from .io import digest_object
from .spatial_attention import spatial_metadata


class PathInputs:
    def __init__(self, bundle, grids=1200):
        self.bundle, self.grids = bundle, grids
        self.spatial = dict(spatial_metadata(bundle))
        if self.spatial.get('coordinate_order') not in {'xy', 'yx'}:
            raise ValueError('PathRWKV requires explicit encoder.spatial.coordinate_order: xy or yx')
        self.contract = digest_object({'spatial': self.spatial, 'grids': grids,
            'source_order': 'yx', 'source_pitch': 224,
            'mapping': 'declared physical origin/pitch, floor to native source grid'})

    def coordinates(self, values):
        if not (np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer)):
            raise ValueError('PathRWKV physical coordinates must be real numbers')
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError('PathRWKV requires aligned finite real coordinates')
        pitch = self.spatial['patch_size']
        values = values-(pitch/2 if self.spatial['origin'] == 'center' else 0)
        if (values < 0).any():
            raise ValueError('PathRWKV coordinates precede the declared top-left physical origin')
        grid = np.floor(values/pitch)
        if (grid >= self.grids).any():
            raise ValueError('PathRWKV positions exceed the declared grid; set a sufficient explicit grid budget')
        if self.spatial['coordinate_order'] == 'xy':
            grid = grid[:, ::-1]
        return torch.from_numpy(np.array(grid*224, dtype=np.float32, copy=True))

    @contextlib.contextmanager
    def arrays(self, row):
        path = resolve(self.bundle.root, row['path'])
        with contextlib.ExitStack() as stack:
            if path.suffix in {'.h5', '.hdf5'}:
                h = stack.enter_context(h5py.File(path, 'r'))
                if not isinstance(h.get('features', getlink=True), h5py.HardLink) or not isinstance(h['features'], h5py.Dataset):
                    raise ValueError('PathRWKV features must be a direct HDF5 dataset, without external links')
                features = h['features']
                if features.is_virtual or features.external:
                    raise ValueError('PathRWKV requires self-contained HDF5 feature storage')
            elif path.suffix == '.npy':
                features = np.load(path, mmap_mode='r', allow_pickle=False)
            else:
                features = load_array(path)
            if row.get('coords_path'):
                cp = resolve(self.bundle.root, row['coords_path'])
                if cp.suffix in {'.h5', '.hdf5'}:
                    ch = stack.enter_context(h5py.File(cp, 'r'))
                    if not isinstance(ch.get('coords', getlink=True), h5py.HardLink) or not isinstance(ch['coords'], h5py.Dataset):
                        raise ValueError('PathRWKV coordinates must be a direct HDF5 dataset')
                    coords = ch['coords']
                    if coords.is_virtual or coords.external:
                        raise ValueError('PathRWKV requires self-contained HDF5 coordinate storage')
                else:
                    coords = load_array(cp, 'coords')
            elif path.suffix in {'.h5', '.hdf5'}:
                if not isinstance(h.get('coords', getlink=True), h5py.HardLink) or not isinstance(h['coords'], h5py.Dataset):
                    raise ValueError('PathRWKV needs direct, row-aligned HDF5 coords or an explicit coordinate path')
                coords = h['coords']
                if coords.is_virtual or coords.external:
                    raise ValueError('PathRWKV requires self-contained HDF5 coordinate storage')
            else:
                raise ValueError('PathRWKV needs an explicit real coordinate path')
            if len(features.shape) != 2 or features.shape[0] < 1 or features.shape[1] != self.bundle.dim or coords.shape != (len(features), 2):
                raise ValueError('PathRWKV features/coordinates have wrong shape or FM dimension')
            yield features, coords

    @staticmethod
    def take(array, indices):
        if isinstance(indices, slice):
            return np.asarray(array[indices])
        indices = np.asarray(indices, dtype=np.int64)
        order = np.argsort(indices)
        return np.asarray(array[indices[order]])[np.argsort(order)]

    def tensors(self, features, coords, indices):
        values = self.take(features, indices)
        raw_coords = self.take(coords, indices)
        if not (np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer)):
            raise ValueError('PathRWKV features must be real numbers')
        values = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        if not torch.isfinite(values).all():
            raise ValueError('PathRWKV features must remain finite in float32')
        return values, self.coordinates(raw_coords)

    def sampled(self, row, max_tiles):
        with self.arrays(row) as (x, c):
            # Exactly the original Dataset's randperm -> truncate/zero-pad order.
            ids = torch.randperm(len(x))[:max_tiles].numpy()
            features, coords = self.tensors(x, c, ids)
        if len(features) < max_tiles:
            missing = max_tiles-len(features)
            features = torch.cat([features, features.new_zeros(missing, self.bundle.dim)])
            coords = torch.cat([coords, coords.new_zeros(missing, 2)])
        return features, coords

    def chunks(self, row, size):
        # PT retains one CPU bag, H5 and NPY only materialize the current chunk.
        with self.arrays(row) as (x, c):
            for start in range(0, len(x), size):
                yield self.tensors(x, c, slice(start, start+size))

    def full(self, row):
        with self.arrays(row) as (x, c):
            return self.tensors(x, c, slice(None))


class TaskSet:
    """Use the full heterogeneous contract, or the source's explicit single-task mode."""
    def __init__(self, bundle):
        self.bundle = bundle
        self.multi = bundle.multitask
        if self.multi is not None:
            self.specs = self.multi.specs
        elif bundle.task_type in {'classification', 'ordinal'}:
            self.specs = [{'name': 'target', 'type': 'classification', 'classes': bundle.classes,
                           'unit': bundle.config.get('evaluation_unit', 'slide')}]
        elif bundle.task_type == 'survival':
            if bundle.config.get('time_unit') not in {'days', 'months', 'years'}:
                raise ValueError('PathRWKV survival requires an explicit time_unit')
            self.specs = [{'name': 'survival', 'type': 'survival', 'unit': 'patient',
                           'time_unit': bundle.config.get('time_unit')}]
        elif bundle.task_type == 'regression':
            self.specs = [{'name': name, 'type': 'regression', 'unit': bundle.config.get('evaluation_unit', 'patient')}
                          for name in bundle.regression_targets.columns]
        else:
            raise ValueError('PathRWKV supports classification/ordinal, scalar regression heads, survival and heterogeneous multitask')
        if any('.' in s['name'] or not s['name'] for s in self.specs):
            raise ValueError('PathRWKV task names must be valid nonempty PyTorch module keys')
        kinds = {'classification': 'cls', 'regression': 'reg', 'survival': 'surv'}
        self.source_tasks = {s['name']: {'type': kinds[s['type']],
            'labels': {name: i for i, name in enumerate(s['classes'])} if s['type'] == 'classification' else None}
            for s in self.specs}
        self.contract = digest_object({'tasks': self.specs, 'bundle': bundle.fingerprint})

    def tensors(self, rows, device='cpu'):
        if self.multi is not None:
            return self.multi.tensors(rows, device)
        values, observed = {}, {}
        if self.bundle.task_type in {'classification', 'ordinal'}:
            values['target'] = torch.tensor([r['target'] for r in rows], dtype=torch.long, device=device)
        elif self.bundle.task_type == 'survival':
            values['survival'] = torch.tensor([[r['time'], r['event']] for r in rows], dtype=torch.float32, device=device)
        else:
            tensors = torch.stack([self.bundle.regression_targets.tensor(r['global_patient_id']) for r in rows]).to(device)
            values = {s['name']: tensors[:, i] for i, s in enumerate(self.specs)}
        observed = {name: torch.ones(len(rows), dtype=torch.bool, device=device) for name in values}
        return values, observed


class TrainingSlides(Dataset):
    def __init__(self, inputs, tasks, rows, max_tiles):
        self.inputs, self.tasks, self.rows, self.max_tiles = inputs, tasks, rows, max_tiles

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, c = self.inputs.sampled(row, self.max_tiles)
        target, mask = self.tasks.tensors([row])
        return {'image_features': x, 'coords_yx': c,
                'task_labels': {k: v[0] for k, v in target.items()},
                'observed': {k: v[0] for k, v in mask.items()}}

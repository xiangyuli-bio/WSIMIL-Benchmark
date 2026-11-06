"""TRAIN-only, indexed cross-bag reservoirs without copying the feature corpus."""
import random

import h5py
import numpy as np
import torch

from .data import load_array, resolve
from .io import digest_object


class TrainClassBank:
    def __init__(self, bundle):
        self.bundle = bundle
        self.rows = bundle.records('train')
        self.by_slide = {}
        self.by_class = {}
        for row in self.rows:
            path = resolve(bundle.root, row['path'])
            if path.suffix in {'.h5', '.hdf5'}:
                with h5py.File(path, 'r') as f:
                    shape = f['features'].shape
            elif path.suffix == '.npy':
                shape = np.load(path, mmap_mode='r', allow_pickle=False).shape
            else:
                shape = load_array(path).shape
            if len(shape) != 2 or shape[0] < 1 or shape[1] != bundle.dim:
                raise ValueError('C2Aug reservoir requires nonempty feature bags in the declared FM space')
            entry = {**row, 'count': int(shape[0])}
            self.by_slide[row['slide_id']] = entry
            self.by_class.setdefault(row['target'], []).append(entry)
        self.max_count = max(r['count'] for r in self.by_slide.values())
        self.fingerprint = digest_object([
            {k: r[k] for k in ['slide_id', 'target', 'path', 'count']}
            for r in self.by_slide.values()])
        self.reads = 0

    def candidates(self, target, exclude_slide):
        if exclude_slide not in self.by_slide or self.by_slide[exclude_slide]['target'] != target:
            raise ValueError('C2Aug augmentation is restricted to the original TRAIN slide and class')
        return [r for r in self.by_class[target] if r['slide_id'] != exclude_slide]

    def validate_views(self, views):
        for slide, row in self.by_slide.items():
            required = max(views*row['count'], self.max_count-row['count'])
            if sum(r['count'] for r in self.candidates(row['target'], slide)) < required:
                raise ValueError('C2Aug needs enough OTHER same-class TRAIN rows for disjoint views; '
                                 'reduce max_views explicitly or provide more TRAIN bags for views/expansion, no duplicate-view fallback')

    def sample(self, target, count, exclude_slide, device='cpu'):
        rows = self.candidates(target, exclude_slide)
        ends = np.cumsum([r['count'] for r in rows], dtype=np.int64)
        total = int(ends[-1]) if len(ends) else 0
        if type(count) is not int or not 0 <= count <= total:
            raise ValueError('C2Aug disjoint sampling exceeds the OTHER same-class TRAIN reservoir')
        # random.sample(range(...)) uses O(requested rows) memory for sparse
        # samples, rather than allocating a permutation of a multi-TB corpus.
        selected = np.asarray(random.sample(range(total), count), dtype=np.int64)
        groups = np.searchsorted(ends, selected, side='right')
        starts = np.r_[0, ends[:-1]]
        output = torch.empty(count, self.bundle.dim)
        for group in np.unique(groups):
            positions = np.flatnonzero(groups == group)
            local = selected[positions]-starts[group]
            order = np.argsort(local)
            path = resolve(self.bundle.root, rows[group]['path'])
            if path.suffix in {'.h5', '.hdf5'}:
                with h5py.File(path, 'r') as f:
                    values = f['features'][local[order]]
            elif path.suffix == '.npy':
                values = np.load(path, mmap_mode='r', allow_pickle=False)[local[order]]
            else:
                values = load_array(path)[local[order]]
            if not (np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating)) or not np.isfinite(values).all():
                raise ValueError('C2Aug sampled nonreal/nonfinite TRAIN features')
            output[torch.from_numpy(positions[order])] = torch.from_numpy(np.asarray(values, dtype=np.float32))
            self.reads += 1
        return output.to(device)

"""Explicit, independently indexed scales in a common physical coordinate frame."""
import numpy as np
import pandas as pd
import torch

from .data import load_array, resolve
from .io import digest_file


class MultiscaleTable:
    def __init__(self, bundle):
        self.bundle = bundle
        self.config = bundle.encoder['multiscale']
        self.kind = self.config['kind']
        if self.kind not in {'aligned_centers', 'quadtree'}:
            raise ValueError('multiscale kind must be aligned_centers or quadtree')
        if self.config.get('coordinate_frame') != 'micrometers' or not self.config.get('preprocessing'):
            raise ValueError('Multiscale needs center coordinates in micrometers and preprocessing provenance')
        self.levels = self.config['levels']
        if len(self.levels) != 3 or len({v['name'] for v in self.levels}) != 3:
            raise ValueError('Multiscale requires three uniquely named levels, coarse to fine')
        self.fovs = np.array([v['field_of_view_um'] for v in self.levels], dtype=float)
        if not np.isfinite(self.fovs).all() or (self.fovs <= 0).any() or (np.diff(self.fovs) >= 0).any():
            raise ValueError('Scale field_of_view_um must be finite, positive and strictly decreasing')
        if self.kind == 'quadtree' and not np.allclose(self.fovs[:-1], 2*self.fovs[1:], rtol=0, atol=.001):
            raise ValueError('Quadtree requires successive 2:1 physical fields of view')
        self.files = [resolve(bundle.root, level['index']) for level in self.levels]
        if self.files[0].resolve() != resolve(bundle.root, bundle.encoder['index']).resolve():
            raise ValueError('Encoder base index must be the coarsest multiscale index')
        self.tables = []
        all_paths = set()
        for path in self.files:
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            if not {'slide_id', 'path'} <= set(table) or table[['slide_id', 'path']].eq('').any().any():
                raise ValueError('Scale index needs nonempty slide_id and path')
            if table.slide_id.duplicated().any() or set(table.slide_id) != set(bundle.rows.slide_id):
                raise ValueError('Scale slide sets must exactly match the task, without duplicates')
            paths = [str(resolve(bundle.root, p).resolve()) for p in table.path]
            if len(set(paths)) != len(paths) or all_paths.intersection(paths):
                raise ValueError('Scales must reference distinct feature files, not reuse one scale')
            all_paths.update(paths)
            self.tables.append(table.set_index('slide_id').to_dict('index'))

    def rows(self, row):
        return [{'slide_id': row['slide_id'], **table[row['slide_id']]} for table in self.tables]

    def load(self, row):
        rows = self.rows(row)
        loaded = [self.bundle.load(r, require_coords=True) for r in rows]
        features, centers = map(list, zip(*loaded))
        parents = [None]
        for level in range(1, 3):
            if self.kind == 'aligned_centers':
                if centers[level].shape != centers[0].shape or not torch.allclose(
                        centers[level], centers[0], rtol=0, atol=.01):
                    raise ValueError(f"{row['slide_id']}: scales must have the same centers in the same row order")
                parents.append(None)
                continue
            if not rows[level].get('parent_path'):
                raise ValueError('Quadtree child index requires parent_path')
            parent = load_array(resolve(self.bundle.root, rows[level]['parent_path']), 'parents')
            count = len(features[level-1])
            if (parent.shape != (len(features[level]),) or not np.issubdtype(parent.dtype, np.integer)
                    or (parent < 0).any() or (parent >= count).any()):
                raise ValueError('Parent indices must be integer [N] referencing the previous scale')
            if not np.all(np.bincount(parent, minlength=count) == 4):
                raise ValueError('Every quadtree parent must have exactly four children')
            delta = centers[level].numpy().astype(float) - centers[level-1].numpy()[parent]
            quarter = self.fovs[level-1] / 4
            if not np.allclose(np.abs(delta), quarter, rtol=0, atol=.01):
                raise ValueError('Quadtree child centers do not match parent physical extent')
            quadrants = (delta[:, 0] > 0).astype(int)*2 + (delta[:, 1] > 0)
            if len(np.unique(parent*4+quadrants)) != len(parent):
                raise ValueError('Duplicate quadtree child quadrant')
            parents.append(torch.from_numpy(parent.astype(np.int64)))
        return features, centers, parents

    def inventory(self, row, deep):
        if deep:
            self.load(row)
        result = []
        for level, r in zip(self.levels, self.rows(row)):
            entry = {'level': level['name']}
            for field, hash_field in [('path', 'sha256'), ('coords_path', 'coords_sha256'),
                                      ('parent_path', 'parents_sha256')]:
                if r.get(field):
                    p = resolve(self.bundle.root, r[field])
                    digest = digest_file(p)
                    if r.get(hash_field) and r[hash_field] != digest:
                        raise ValueError(f'Multiscale hash mismatch: {p}')
                    entry[field] = str(p)
                    entry[hash_field] = digest
            result.append(entry)
        return result

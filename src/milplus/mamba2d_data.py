"""Real physical grid contract for original dense 2DMamba, including padded cells."""
import math

import numpy as np
import torch

from .data import load_array, resolve
from .spatial_attention import spatial_metadata


class PhysicalGrid:
    def __init__(self, bundle):
        self.bundle = bundle
        self.spatial = spatial_metadata(bundle)
        if self.spatial['origin'] != 'top_left':
            raise ValueError('2DMamba requires declared top-left physical patch coordinates')
        meta = bundle.encoder.get('mamba2d', {})
        extent = meta.get('max_coordinates')
        if not isinstance(extent, list) or len(extent) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                or v < 0 or v > 2**53-1 for v in extent):
            raise ValueError('2DMamba requires explicit inclusive physical max_coordinates: [x,y]')
        if not isinstance(meta.get('provenance'), str) or not meta['provenance'].strip():
            raise ValueError('2DMamba grid extent requires provenance, not inferred TEST-label-dependent bounds')
        self.pitch = self.spatial['patch_size']
        self.extent = np.asarray(extent)
        self.width, self.height = [math.floor(v/self.pitch)+1 for v in extent]
        self.padded_width, self.padded_height = [1 << (v-1).bit_length() for v in [self.width, self.height]]
        if bundle.task_type not in {'classification', 'survival'}:
            raise ValueError('2DMamba supports original classification and discrete survival tasks')
        if bundle.config.get('training_unit', 'slide') != 'slide' and bundle.task_type == 'classification':
            raise ValueError('2DMamba classification trains individual slide bags')
        if bundle.task_type == 'survival' and bundle.rows.groupby('global_patient_id').size().max() != 1:
            raise ValueError('2DMamba native survival requires one registered WSI per patient; distinct coordinate frames cannot be concatenated')

    def inputs(self, row):
        x, _ = self.bundle.load(row, require_coords=True)
        path = resolve(self.bundle.root, row.get('coords_path') or row['path'])
        coordinates = np.asarray(load_array(path, 'coords'))
        if coordinates.shape != (len(x), 2) or coordinates.dtype.kind not in 'iuf' or not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > self.extent).any():
            raise ValueError('2DMamba coordinates must be finite nonnegative physical points within the declared canvas')
        if self.spatial['unit'] == 'level0_pixels' and not np.equal(coordinates, np.floor(coordinates)).all():
            raise ValueError('2DMamba level0 pixel coordinates must be integer-valued')
        grid = np.floor_divide(coordinates, self.pitch).astype(np.int64)
        if len(np.unique(grid, axis=0)) != len(grid):
            raise ValueError('2DMamba physical coordinates collide in one native grid cell; no silent overwrite')
        return x, torch.from_numpy(grid), coordinates

    def metadata(self):
        return {'width': self.width, 'height': self.height, 'power2_width': self.padded_width,
                'power2_height': self.padded_height, 'pitch': self.pitch,
                'physical_max_coordinates': self.extent.tolist(), 'spatial': self.spatial,
                'holes': 'original pad tokens remain in both the scan and final attention denominator'}

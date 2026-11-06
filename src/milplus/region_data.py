"""Explicit, physically checked 4x4 regions for AdvMIL's region projection."""
import numpy as np
import torch

from .data import load_array, resolve
from .io import digest_file


class RegionTable:
    def __init__(self, bundle):
        self.bundle = bundle
        self.meta = dict(bundle.encoder['regions'])
        if set(self.meta) != {'coordinate_frame', 'patch_stride', 'preprocessing'}:
            raise ValueError('regions requires coordinate_frame, patch_stride and preprocessing')
        if self.meta['coordinate_frame'] not in {'level0_pixels', 'micrometers'}:
            raise ValueError('Region coordinates must declare level0_pixels or micrometers')
        stride = self.meta['patch_stride']
        if isinstance(stride, bool) or not isinstance(stride, (int, float)) or not np.isfinite(stride) or stride <= 0:
            raise ValueError('Region patch_stride must be finite positive in coordinate_frame units')
        if not isinstance(self.meta['preprocessing'], str) or not self.meta['preprocessing'].strip():
            raise ValueError('Region preprocessing provenance is required')
        if 'region_ids_path' not in bundle.rows or bundle.rows.region_ids_path.eq('').any():
            raise ValueError('Each slide requires aligned region_ids_path')

    def load(self, row):
        x, coords = self.bundle.load(row, require_coords=True)
        ids = load_array(resolve(self.bundle.root, row['region_ids_path']), 'region_ids')
        if ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
            raise ValueError('Region IDs must be aligned nonnegative integer [N]')
        c = coords.numpy().astype(np.float64)
        if len(np.unique(c, axis=0)) != len(c):
            raise ValueError('Duplicate region coordinates within a slide')
        ordered = []
        for region in np.unique(ids):
            idx = np.flatnonzero(ids == region)
            grid = c[idx]
            ux, uy = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            if len(idx) != 16 or len(ux) != 4 or len(uy) != 4:
                raise ValueError('AdvMIL requires complete 4x4 regions, without fabricated or missing patches')
            if not all(np.allclose(np.diff(axis), self.meta['patch_stride'], rtol=1e-5, atol=1e-5) for axis in (ux, uy)):
                raise ValueError('Region coordinates do not match declared patch_stride')
            ordered.append(idx[np.lexsort((grid[:, 0], grid[:, 1]))])
        return x[torch.from_numpy(np.stack(ordered))]

    def inventory(self, row, deep=False):
        path = resolve(self.bundle.root, row['region_ids_path'])
        checksum = digest_file(path)
        if row.get('region_ids_sha256') and row['region_ids_sha256'] != checksum:
            raise ValueError('Region ID hash mismatch')
        result = {'path': str(path), 'sha256': checksum}
        if deep:
            result['regions'] = len(self.load(row))
        return result


def patient_regions(bundle, rows, device, budget=None, train=False):
    if bundle.regions is None:
        raise ValueError('AdvMIL requires explicit 4x4 regions')
    if budget is not None and (type(budget) is not int or budget < 16):
        raise ValueError('Region patch budget must be an integer >=16 or null')
    # Complete regions are concatenated, never formed across slide boundaries.
    regions = torch.cat([bundle.regions.load(row) for row in rows])
    if budget is not None and len(regions) > budget // 16:
        count = budget // 16
        indices = torch.randperm(len(regions))[:count].sort().values if train else torch.linspace(0, len(regions)-1, count).long()
        regions = regions[indices]
    return regions.reshape(1, -1, bundle.dim).to(device)

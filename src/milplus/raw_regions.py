"""Input-bound real WSI coordinates and streamed RGB regions, never frozen FM bags."""
import hashlib
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from .data import resolve
from .io import digest_file, digest_object
from .raw_images import positive_integer

RESNET34_V1_SHA256 = 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'


def grid_sha256(coords):
    a = np.ascontiguousarray(coords, dtype='<i8')
    h = hashlib.sha256(digest_object({'shape': list(a.shape), 'dtype': '<i8', 'order': 'original xy rows'}).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


class RawRegions:
    def __init__(self, bundle):
        self.bundle = bundle
        self.config = c = bundle.encoder['raw_regions']
        if bundle.dim != 3 or bundle.task_type != 'classification' or len(bundle.classes) != 2:
            raise ValueError('Raw Campanella regions require RGB dimension3 and explicit binary classes in source0/1 order')
        if not isinstance(c, dict) or c.get('kind') not in {'openslide', 'measured_rgb_region_cache'} or c.get('coordinate_unit') != 'level0_pixels' or c.get('coordinate_order') != 'xy':
            raise ValueError('Declare raw_regions OpenSlide and real level0_pixels coordinates in xy order')
        if any(k in bundle.encoder for k in ['raw_images', 'regions', 'multiscale', 'smmile', 'pamil', 'dual_scale', 'vision_language', 'spatial']):
            raise ValueError('Streamed raw regions cannot be combined with a frozen-feature/whole-level input contract')
        if type(c.get('level')) is not int or c['level'] < 0 or not isinstance(c.get('provenance'), str) or not c['provenance'].strip():
            raise ValueError('Raw regions require an actual nonnegative WSI level and coordinate preprocessing provenance')
        self.level = c['level']
        self.mult = c.get('mult')
        if type(self.mult) not in {int, float} or not np.isfinite(self.mult) or self.mult <= 0:
            raise ValueError('Raw region mult must be finite and positive')
        self.max_rows = positive_integer(c.get('max_grid_rows', 2_000_000), 'max_grid_rows')
        self.max_grid_cache_bytes = positive_integer(c.get('max_grid_cache_bytes', 128*1024**2), 'max_grid_cache_bytes')
        self.max_tile_bytes = positive_integer(c.get('max_tile_bytes', 256*1024**2), 'max_tile_bytes')
        if min(self.size('cnn'), self.size('rnn')) < 1 or 4*max(self.size('cnn'), self.size('rnn'))**2 > self.max_tile_bytes:
            raise ValueError('Raw source region exceeds max_tile_bytes before allocation, or rounds to zero')
        if 'coords_path' not in bundle.rows or bundle.rows.coords_path.eq('').any():
            raise ValueError('Each raw WSI needs its real input-bound coords_path; no generated sequence grid')
        if 'cluster_labels_path' in bundle.rows and bundle.rows.cluster_labels_path.ne('').any():
            raise ValueError('Raw WSI regions cannot include a frozen-feature cluster label index')
        self.backbone = b = c.get('backbone', {})
        if not isinstance(b, dict) or b.get('architecture') != 'resnet34' or b.get('initialization') not in {'imagenet1k_v1', 'explicit_random'}:
            raise ValueError('Declare native ResNet34 ImageNet1K_V1 or explicitly named random initialization')
        self.files = [resolve(bundle.root, p) for p in bundle.rows.coords_path]
        self.weights = None
        if b['initialization'] == 'imagenet1k_v1':
            if b.get('sha256') != RESNET34_V1_SHA256 or not isinstance(b.get('path'), str):
                raise ValueError('Local ImageNet1K_V1 ResNet34 weights require the official complete SHA256')
            self.weights = resolve(bundle.root, b['path'])
            self.check_initialization()
            self.files.append(self.weights)
        self._grids, self._grid_cache_bytes = OrderedDict(), 0
        self._transform = None
        self.cache = None
        if c['kind'] == 'measured_rgb_region_cache':
            from .raw_region_cache import MeasuredRegionCache
            if not isinstance(c.get('cache_manifest'), str) or not c['cache_manifest']:
                raise ValueError('Measured regions require their actual source cache manifest')
            self.cache = MeasuredRegionCache(self, resolve(bundle.root, c['cache_manifest']))
            self.files.extend(self.cache.files)

    def size(self, stage):
        if stage not in {'cnn', 'rnn'}:
            raise ValueError('Source region stage must be cnn or rnn')
        return int(np.round(224*self.mult)) if stage == 'cnn' else int(224*self.mult)

    def slide(self, row):
        if self.cache is not None:
            raise ValueError('A measured-region cache is not an OpenSlide whole-image reader')
        import openslide
        p = resolve(self.bundle.root, row['path'])
        if p.suffix.lower() not in {'.svs', '.tif', '.tiff', '.ndpi', '.scn', '.bif'}:
            raise ValueError('Raw ROI inventory currently supports single-file WSI formats only')
        return openslide.OpenSlide(str(p))

    def _shape(self, shape, dtype):
        if len(shape) != 2 or shape[1] != 2 or not 1 <= shape[0] <= self.max_rows or not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            raise ValueError('Raw xy coordinates require bounded nonempty integer[N,2], without fabricated feature rows')
        if shape[0]*16 > self.max_grid_cache_bytes:
            raise ValueError('Raw coordinate grid exceeds bounded max_grid_cache_bytes before allocation')

    def grid(self, row):
        p = resolve(self.bundle.root, row['coords_path'])
        stamp = (p.stat().st_size, p.stat().st_mtime_ns)
        old = self._grids.get(str(p))
        if old is not None and old[0] == stamp:
            self._grids.move_to_end(str(p))
            return old[1]
        if old is not None:
            self._grid_cache_bytes -= old[1].nbytes
            del self._grids[str(p)]
        if p.suffix == '.npy':
            a = np.load(p, allow_pickle=False, mmap_mode='r')
            self._shape(a.shape, a.dtype)
        elif p.suffix in {'.h5', '.hdf5'}:
            with h5py.File(p, 'r') as f:
                if 'coords' not in f or isinstance(f.get('coords', getlink=True), h5py.ExternalLink):
                    raise ValueError('Raw HDF5 coords must be embedded in the inventoried file')
                ds = f['coords']
                if not isinstance(ds, h5py.Dataset) or ds.is_virtual or ds.external or Path(ds.file.filename).resolve() != p.resolve():
                    raise ValueError('Raw HDF5 coords cannot use virtual or external storage')
                self._shape(ds.shape, ds.dtype)
                a = ds[:]
        else:
            raise ValueError('Real raw coordinate grids support NPY or embedded HDF5 coords')
        if (a < 0).any() or (a > 2**53).any() or len(np.unique(a, axis=0)) != len(a):
            raise ValueError('Raw WSI coordinates must be unique nonnegative exact level0 pixel locations')
        a = np.array(a, dtype=np.int64)
        a.setflags(write=False)
        while self._grid_cache_bytes+a.nbytes > self.max_grid_cache_bytes:
            _, (_, evicted) = self._grids.popitem(last=False)
            self._grid_cache_bytes -= evicted.nbytes
        self._grids[str(p)] = (stamp, a)
        self._grid_cache_bytes += a.nbytes
        return a

    def describe(self, row):
        coords = self.grid(row)
        if self.cache is not None:
            return self.cache.describe(row, coords)
        with self.slide(row) as slide:
            if self.level >= slide.level_count:
                raise ValueError('Declared raw region WSI level does not exist')
            dims, downsample = list(slide.dimensions), float(slide.level_downsamples[self.level])
            if (coords[:, 0] >= dims[0]).any() or (coords[:, 1] >= dims[1]).any():
                raise ValueError('Raw coordinate origin lies outside the real level0 WSI canvas')
            if 'downsample' in self.config and self.config['downsample'] != downsample:
                raise ValueError('Raw region downsample differs from actual WSI metadata')
        return {'kind': 'openslide_regions', 'instances': len(coords), 'level': self.level, 'downsample': downsample,
                'level0_dimensions': dims, 'coordinate_unit': 'level0_pixels', 'coordinate_order': 'xy',
                'cnn_read_size': self.size('cnn'), 'rnn_read_size': self.size('rnn'), 'output_tile_size': 224,
                'grid_sha256': grid_sha256(coords), 'backbone': self.backbone,
                'edge_behavior': 'native OpenSlide RGBA to RGB, including native out-of-bounds region padding'}

    def read(self, row, index, stage):
        grid = self.grid(row)
        if type(index) is not int or not 0 <= index < len(grid):
            raise ValueError('Raw region index is outside the registered coordinate grid')
        size = self.size(stage)
        if self.cache is not None:
            image = self.cache.read(row, index, grid)
        else:
            with self.slide(row) as slide:
                if self.level >= slide.level_count:
                    raise ValueError('Declared raw region WSI level does not exist')
                image = slide.read_region(tuple(map(int, grid[index])), self.level, (size, size)).convert('RGB')
        if self.mult != 1:
            from PIL import Image
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
        if self._transform is None:
            from torchvision.transforms import Compose, Normalize, ToTensor
            self._transform = Compose([ToTensor(), Normalize([.5]*3, [.1]*3)])
        return self._transform(image)

    def inventory(self, row, deep):
        info = self.describe(row)
        if deep:
            info['sha256'] = digest_file(resolve(self.bundle.root, row['path']))
            if self.cache is not None:
                self.cache.validate_archive_digest(row, info['sha256'])
            info['coords_sha256'] = digest_file(resolve(self.bundle.root, row['coords_path']))
            for key in ['sha256', 'coords_sha256']:
                if row.get(key) and row[key] != info[key]:
                    raise ValueError('Raw region image/coordinate content hash mismatch')
        return info

    def check_initialization(self):
        if self.weights is not None and (not self.weights.is_file() or digest_file(self.weights) != RESNET34_V1_SHA256):
            raise ValueError('Raw ResNet34 initialization weights/hash mismatch')

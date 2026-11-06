"""Explicit full-slide RGB inputs, distinct from frozen patch feature bags."""
import math
from pathlib import Path

import h5py
import numpy as np

from .data import resolve
from .io import digest_file


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class RawImages:
    """Read one complete declared resolution level; patching remains stochastic in TRAIN."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.config = c = bundle.encoder['raw_images']
        if not isinstance(c, dict) or c.get('kind') not in {'decoded_rgb', 'openslide'}:
            raise ValueError('raw_images.kind must be decoded_rgb or openslide')
        if bundle.dim != 3 or bundle.task_type != 'ordinal':
            raise ValueError('Raw RGB MONAI input requires dimension3 and explicit ordered class labels')
        if any(key in bundle.encoder for key in ['multiscale', 'spatial', 'regions', 'smmile', 'pamil', 'dual_scale', 'vision_language', 'raw_regions']):
            raise ValueError('Full raw RGB images cannot masquerade as a registered frozen-feature/spatial bag')
        if type(c.get('level')) is not int or c['level'] < 0 or not isinstance(c.get('provenance'), str) or not c['provenance'].strip():
            raise ValueError('Raw images require a nonnegative source level and preprocessing provenance')
        self.level = c['level']
        self.max_bytes = positive_integer(c.get('max_image_bytes', 512*1024**2), 'max_image_bytes')
        if c['kind'] == 'decoded_rgb' and (isinstance(c.get('downsample'), bool) or not isinstance(c.get('downsample'), (float, int)) or not np.isfinite(c['downsample']) or c['downsample'] < 1):
            raise ValueError('Decoded RGB requires its real positive level0 downsample, not inferred coordinates')
        if any(key in bundle.rows and bundle.rows[key].ne('').any() for key in ['coords_path', 'cluster_labels_path']):
            raise ValueError('Raw full RGB inputs cannot include a frozen feature/cluster coordinate index')
        self.backbone = b = c.get('backbone', {})
        if not isinstance(b, dict) or b.get('architecture') != 'resnet50' or b.get('initialization') not in {'imagenet1k_v1', 'explicit_random'}:
            raise ValueError('Declare ResNet50 ImageNet1K_V1 weights or explicit_random initialization')
        self.files = []
        if b['initialization'] == 'imagenet1k_v1':
            if not isinstance(b.get('sha256'), str) or len(b['sha256']) != 64 or not isinstance(b.get('path'), str):
                raise ValueError('Local ImageNet initialization requires path and full SHA256; no implicit download')
            if b['sha256'] != '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a':
                raise ValueError('ImageNet1K_V1 initialization must match the official ResNet50 checkpoint')
            self.weights = resolve(bundle.root, b['path'])
            if not self.weights.is_file() or digest_file(self.weights) != b['sha256']:
                raise ValueError('Raw-image backbone initialization weights/hash mismatch')
            self.files.append(self.weights)
        else:
            self.weights = None

    def _check_shape(self, shape, dtype):
        if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) < 1 or np.dtype(dtype) != np.dtype('uint8'):
            raise ValueError('Decoded RGB must be complete uint8 [3,H,W], not features, cropped tiles or normalized floats')
        if math.prod(shape) > self.max_bytes:
            raise ValueError('Raw image exceeds max_image_bytes before decoding/tiling allocation')

    def _slide(self, path):
        import openslide
        # Multi-file formats need an explicit companion-file inventory not yet defined here.
        if path.suffix.lower() not in {'.svs', '.tif', '.tiff', '.ndpi', '.scn', '.bif'}:
            raise ValueError('Raw OpenSlide input currently requires a supported single-file WSI')
        return openslide.OpenSlide(str(path))

    def describe(self, row):
        p = resolve(self.bundle.root, row['path'])
        if self.config['kind'] == 'openslide':
            with self._slide(p) as slide:
                if self.level >= slide.level_count:
                    raise ValueError('Declared WSI resolution level does not exist')
                w, h = slide.level_dimensions[self.level]
                info = {'shape': [3, h, w], 'downsample': float(slide.level_downsamples[self.level]),
                        'level0_dimensions': list(slide.dimensions), 'vendor': slide.detect_format(str(p))}
                if 'downsample' in self.config and self.config['downsample'] != info['downsample']:
                    raise ValueError('Declared downsample does not match the actual WSI resolution level')
        else:
            if p.suffix == '.npy':
                a = np.load(p, allow_pickle=False, mmap_mode='r')
                shape, dtype = a.shape, a.dtype
            elif p.suffix in {'.h5', '.hdf5'}:
                with h5py.File(p, 'r') as f:
                    if 'pixels' not in f:
                        raise ValueError('Decoded image HDF5 requires pixels, not a features dataset')
                    if isinstance(f.get('pixels', getlink=True), h5py.ExternalLink):
                        raise ValueError('Raw HDF5 pixels must be embedded, not an unregistered external link')  # noqa: TRY004 -- invalid file content, not a caller argument type
                    pixels = f['pixels']
                    if not isinstance(pixels, h5py.Dataset) or pixels.is_virtual or pixels.external or Path(pixels.file.filename).resolve() != p.resolve():
                        raise ValueError('Raw HDF5 pixels must be embedded, not virtual or external storage')
                    shape, dtype = pixels.shape, pixels.dtype
            else:
                raise ValueError('Decoded RGB supports complete CHW NPY or HDF5 pixels; use openslide for WSI files')
            self._check_shape(shape, dtype)
            info = {'shape': list(shape), 'downsample': self.config['downsample']}
        self._check_shape(info['shape'], np.uint8)
        return {**info, 'kind': self.config['kind'], 'level': self.level, 'color': 'RGB uint8, no stain/ICC conversion',
                'backbone': self.backbone}

    def load(self, row):
        info = self.describe(row)
        p = resolve(self.bundle.root, row['path'])
        if self.config['kind'] == 'openslide':
            with self._slide(p) as slide:
                # Match the tutorial's full level read; no pyramid level or region is silently substituted.
                a = np.asarray(slide.read_region((0, 0), self.level, tuple(info['shape'][:0:-1])).convert('RGB')).transpose(2, 0, 1)
        elif p.suffix == '.npy':
            a = np.load(p, allow_pickle=False, mmap_mode='r')
        else:
            with h5py.File(p, 'r') as f:
                a = f['pixels'][:]
        self._check_shape(a.shape, a.dtype)
        return a, info

    def inventory(self, row, deep):
        info = self.describe(row)
        if deep:
            info['sha256'] = digest_file(resolve(self.bundle.root, row['path']))
            if row.get('sha256') and row['sha256'] != info['sha256']:
                raise ValueError('Raw image content hash mismatch')
        return info

    def check_initialization(self):
        if self.weights is not None and digest_file(self.weights) != self.backbone['sha256']:
            raise ValueError('Raw backbone initialization content changed')

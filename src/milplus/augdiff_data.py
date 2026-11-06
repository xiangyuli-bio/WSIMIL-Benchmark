"""TRAIN-bound seven real image-view feature pools for AugDiff pretraining.

View features must come from actual image augmentations and the same frozen FM.
This loader never generates alternative views by adding feature-space noise.
"""
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_array, resolve
from .io import digest_file, digest_object, read_json

VIEWS = ('images', 'affinecvimg_0', 'blurimg_0', 'Colorimg_0',
         'elasticimg_0', 'HEDJitter_0', 'Rotationimg_0')
KIND = 'augdiff_train_seven_image_views'
INDEPENDENT = 'independent_augmentation_pools'
PAIRED = 'paired_patch_views'


def _ordered_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class AugDiffInputs:
    def __init__(self, bundle, manifest=None):
        if bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('AugDiff source recipe requires individual slide classification')
        self.bundle = bundle
        manifest = manifest or bundle.encoder['augdiff']['manifest']
        self.path = resolve(bundle.root, manifest).resolve()
        self.manifest = meta = read_json(self.path)
        if (meta.get('schema_version') != 1 or meta.get('kind') != KIND
                or type(meta.get('synthetic')) is not bool
                or meta['synthetic'] != bool(bundle.config.get('synthetic', False))
                or meta.get('dimension') != bundle.dim or meta.get('fm') != bundle.fm):
            raise ValueError('AugDiff seven-view manifest schema, FM, full dimension or synthetic flag differs')
        self.pairing = meta.get('view_pairing_policy', INDEPENDENT)
        if self.pairing not in {INDEPENDENT, PAIRED}:
            raise ValueError('AugDiff view pairing policy must declare independent pools or paired patch views')
        provenance = 'row_correspondence_provenance' if self.pairing == PAIRED else 'view_identity_provenance'
        for key in ['augmentation_provenance', 'weights_provenance', 'feature_preprocessing',
                    provenance, 'diffusion_split_provenance']:
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                raise ValueError('AugDiff requires actual '+key)
        if meta.get('encoder_identity') != digest_object({k: v for k, v in bundle.encoder.items() if k != 'augdiff'}):
            raise ValueError('AugDiff seven views must use the exact registered frozen FM and extraction settings')
        train = {r['slide_id']: r for r in bundle.records('train')}
        records = meta.get('records')
        if (not isinstance(records, list) or len(records) != len(train)
                or len({r.get('slide_id') for r in records}) != len(records)
                or {r.get('slide_id') for r in records} != set(train)):
            raise ValueError('AugDiff DAE input must contain exactly supervised TRAIN slides; no VAL/TEST leakage')
        self.records, patient_subsets, files = {}, {}, [self.path]
        for record in records:
            row = train[record['slide_id']]
            subset = record.get('diffusion_subset')
            if subset not in {'train', 'val'}:
                raise ValueError('AugDiff requires explicit patient-grouped internal DAE train/val subsets')
            patient = row['global_patient_id']
            if patient_subsets.setdefault(patient, subset) != subset:
                raise ValueError('AugDiff internal DAE split leaks a patient across train and validation')
            expected, _ = bundle.load(row)
            order = record.get('ordered_patch_ids_sha256')
            if self.pairing == PAIRED and not _ordered_hash(order):
                raise ValueError('AugDiff requires actual ordered patch-ID hash for all corresponding image views')
            if order is not None and not _ordered_hash(order):
                raise ValueError('AugDiff optional original ordered patch-ID hash is malformed')
            views = record.get('views')
            if not isinstance(views, dict) or set(views) != set(VIEWS):
                raise ValueError('AugDiff requires original plus all six actual image-augmentation feature sets')
            result = {**record, 'global_patient_id': patient, 'count': len(expected), 'views': {}}
            for name in VIEWS:
                entry = views[name]
                path = resolve(bundle.root, entry['path']).resolve()
                identity = entry.get('ordered_patch_ids_sha256')
                if (entry.get('sha256') != digest_file(path)
                        or (self.pairing == PAIRED and identity != order)
                        or (identity is not None and not _ordered_hash(identity))):
                    raise ValueError('AugDiff view bytes or ordered patch registration differs')
                features = np.asarray(load_array(path, key=entry.get('key', 'features')))
                if (features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != bundle.dim
                        or (self.pairing == PAIRED and features.shape != tuple(expected.shape))
                        or not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all()):
                    raise ValueError('AugDiff view must retain finite patches, full dimension and declared pairing')
                if name == 'images' and (path != resolve(bundle.root, row['path']).resolve()
                        or not torch.equal(torch.from_numpy(features).float(), expected)):
                    raise ValueError('AugDiff original view must be the actual immutable supervised slide feature file')
                result['views'][name] = {**entry, 'path': str(path), 'count': len(features)}
                files.append(path)
            self.records[row['slide_id']] = result
        if set(patient_subsets.values()) != {'train', 'val'}:
            raise ValueError('AugDiff DAE requires nonempty patient-disjoint internal train and validation')
        self.files = list(dict.fromkeys(files))
        self.hashes = {str(p): digest_file(p) for p in self.files}
        self.stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in self.files}
        self.contract = digest_object({'manifest_and_view_files': self.hashes,
            'patient_subsets': patient_subsets, 'feature_dimension': bundle.dim,
            'original_supervised_train_slide_ids': sorted(train), 'view_pairing_policy': self.pairing})

    def check_unchanged(self):
        if any(digest_file(path) != sha for path, sha in self.hashes.items()):
            raise ValueError('AugDiff TRAIN manifest or registered seven-view bytes changed')

    def patches(self, subset):
        if subset not in {'train', 'val'}:
            raise ValueError('AugDiff internal diffusion subset must be train or val')
        return ViewPatches(self, subset)


class ViewPatches(Dataset):
    """Flat pool of all7 views, bounded to one loaded file; no patch truncation."""
    def __init__(self, assets, subset):
        self.assets, self.subset, self.parts, self.edges = assets, subset, [], [0]
        self.cache_path, self.cache = None, None
        for key in sorted(assets.records):
            record = assets.records[key]
            if record['diffusion_subset'] != subset:
                continue
            for name in VIEWS:
                self.parts.append(record['views'][name])
                self.edges.append(self.edges[-1]+record['views'][name]['count'])
        if not self.parts:
            raise ValueError('AugDiff pretraining patch pool is empty')

    def __len__(self):
        return self.edges[-1]

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError('AugDiff patch index outside registered seven-view pool')
        part = bisect_right(self.edges, index)-1
        entry = self.parts[part]
        path = Path(entry['path'])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.assets.stats[str(path)]:
            raise ValueError('AugDiff view file changed during pretraining')
        if path.suffix in {'.h5', '.hdf5'}:
            import h5py
            with h5py.File(path, 'r') as archive:
                value = archive[entry.get('key', 'features')][index-self.edges[part]]
            return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
        if self.cache_path != path:
            if path.suffix == '.npy':
                value = np.load(path, allow_pickle=False, mmap_mode='r')
            elif path.suffix in {'.pt', '.pth'}:
                value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                if isinstance(value, dict):
                    value = value[entry.get('key', 'features')]
                value = value.numpy()
            else:
                raise ValueError('AugDiff patch pool requires NPY/H5 or memory-mappable tensor files')
            if value.shape != (self.edges[part+1]-self.edges[part], self.assets.bundle.dim):
                raise ValueError('AugDiff registered view shape changed during pretraining')
            self.cache_path, self.cache = path, value
        # Explicit copy: no writable alias to read-only source features.
        return torch.from_numpy(np.array(self.cache[index-self.edges[part]], dtype=np.float32, copy=True))

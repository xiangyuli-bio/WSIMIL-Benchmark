"""Stable TRAIN-only INS instance IDs, bounded feature access and native samplers."""
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import resolve
from .io import digest_object


class TrainPatchStore:
    def __init__(self, bundle, cache_bytes=128*1024**2, max_open_files=16):
        if any(type(v) is not int or v < 1 for v in [cache_bytes, max_open_files]):
            raise ValueError('INS patch cache and file budgets must be positive')
        self.bundle, self.rows = bundle, bundle.records('train')
        self.cache_bytes, self.max_open_files = cache_bytes, max_open_files
        self.cache, self.handles, self.used_bytes = OrderedDict(), OrderedDict(), 0
        counts = []
        for row in self.rows:
            path = resolve(bundle.root, row['path'])
            if path.suffix in {'.h5', '.hdf5'}:
                with h5py.File(path, 'r') as f:
                    shape = f['features'].shape
            elif path.suffix == '.npy':
                a = np.load(path, allow_pickle=False, mmap_mode='r')
                shape = a.shape
                del a
            else:
                shape = bundle.load(row)[0].shape
            if len(shape) != 2 or shape[1] != bundle.dim or shape[0] < 1:
                raise ValueError('INS indexed features differ from the declared complete FM shape')
            counts.append(shape[0])
        self.counts = np.asarray(counts, dtype=np.int64)
        self.offsets = np.concatenate([[0], self.counts.cumsum()])
        self.labels = torch.repeat_interleave(torch.tensor([row['target'] for row in self.rows]), torch.from_numpy(self.counts))
        self.contract = digest_object({'bundle': bundle.fingerprint, 'fm': bundle.fm, 'dimension': bundle.dim,
            'training_rows': [{'slide_id': r['slide_id'], 'target': r['target'], 'patches': int(n)} for r, n in zip(self.rows, self.counts)],
            'order': 'manifest slide order, then original feature-row order; TRAIN only'})

    def __len__(self):
        return int(self.offsets[-1])

    def patch(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError('INS global instance index is outside the actual TRAIN inventory')
        slide = int(np.searchsorted(self.offsets, index, side='right')-1)
        local = int(index-self.offsets[slide])
        row = self.rows[slide]
        path = resolve(self.bundle.root, row['path'])
        name = str(path)
        if path.suffix in {'.h5', '.hdf5'}:
            if name not in self.handles:
                while len(self.handles) >= self.max_open_files:
                    self.handles.popitem(last=False)[1].close()
                self.handles[name] = h5py.File(path, 'r')
            self.handles.move_to_end(name)
            x = np.asarray(self.handles[name]['features'][local])
            value = torch.from_numpy(np.array(x, copy=True)).float()
        elif path.suffix == '.npy':
            a = np.load(path, allow_pickle=False, mmap_mode='r')
            value = torch.from_numpy(np.array(a[local], copy=True)).float()
            del a
        else:
            if name not in self.cache:
                x = self.bundle.load(row)[0]
                amount = x.numel()*x.element_size()
                while self.cache and self.used_bytes+amount > self.cache_bytes:
                    _, previous = self.cache.popitem(last=False)
                    self.used_bytes -= previous.numel()*previous.element_size()
                if amount <= self.cache_bytes:
                    self.cache[name] = x
                    self.used_bytes += amount
                else:
                    # Large PT cannot be partially decoded: explicit bounded
                    # transient full load, no truncation or persistent caching.
                    value = x[local].float().clone()
            if name in self.cache:
                self.cache.move_to_end(name)
                value = self.cache[name][local].float().clone()
        if value.shape != (self.bundle.dim,) or not torch.isfinite(value).all():
            raise ValueError('INS feature row shape or finite-value contract changed')
        return value

    def close(self):
        for stream in self.handles.values():
            stream.close()
        self.handles.clear()
        self.cache.clear()
        self.used_bytes = 0


class InstanceDataset(Dataset):
    def __init__(self, store, positive_only):
        self.store = store
        self.indices = (store.labels == 1).nonzero(as_tuple=True)[0] if positive_only else torch.arange(len(store))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        return self.store.patch(index), index


class BagDataset(Dataset):
    def __init__(self, store):
        self.store = store

    def __len__(self):
        return len(self.store.rows)

    def __getitem__(self, i):
        row = self.store.rows[i]
        return self.store.bundle.load(row)[0], row['target'], row['slide_id']

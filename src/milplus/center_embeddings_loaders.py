"""Source exhaustive candidate scans and ordered top32 raw-image bags."""
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .center_embeddings_assets import transforms
from .data import resolve


class CandidateTiles(Dataset):
    def __init__(self, assets, rows, training=False):
        self.assets, self.rows = assets, rows
        self.counts = [assets.count(r) for r in rows]
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.groups = np.repeat(np.arange(len(rows)), self.counts)
        self.handle, self.open_slide = None, None
        self.transform = transforms(assets.assets, training)

    def close(self):
        if self.handle is not None:
            self.handle.close(); self.handle, self.open_slide = None, None

    def __del__(self):
        self.close()

    def pixel(self, slide, index):
        row = self.rows[slide]
        if self.open_slide != slide:
            self.close(); self.assets.check(row)
            self.handle = h5py.File(resolve(self.assets.bundle.root, row['path']), 'r')
            self.open_slide = slide
        value = self.handle[self.assets.schema['pixel_key']][index]
        return self.transform(Image.fromarray(value))

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        slide = int(np.searchsorted(self.offsets, index, side='right')-1)
        return self.pixel(slide, int(index-self.offsets[slide])), self.rows[slide]['target']


class SelectedBags(Dataset):
    def __init__(self, candidates, selected, order=None):
        self.candidates = candidates
        # The author dictionary preserves selected per-slide score order.
        groups = {}
        for index in selected:
            slide = int(candidates.groups[index])
            groups.setdefault(slide, []).append(int(index-candidates.offsets[slide]))
        if set(groups) != set(range(len(candidates.rows))):
            raise ValueError('Center Embeddings source top-k omitted a registered slide')
        self.selected = groups
        self.order = list(groups) if order is None else list(order)
        if len(self.order) != len(groups) or set(self.order) != set(groups):
            raise ValueError('Center Embeddings bag order changed source coverage')

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        slide = self.order[index]; patch_ids = self.selected[slide]
        images = torch.zeros(len(patch_ids), 3, 256, 256)
        labels = torch.zeros(len(patch_ids))
        target = self.candidates.rows[slide]['target']
        for i, patch in enumerate(patch_ids):
            images[i] = self.candidates.pixel(slide, patch); labels[i] = target
        return images, target, labels

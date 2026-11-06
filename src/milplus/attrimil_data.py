"""Exact original nine-neighbor geometry and class-balanced author sampler data."""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .spatial_attention import checked_points, spatial_metadata


def neighbor_step(bundle):
    meta = spatial_metadata(bundle)
    settings = bundle.encoder.get("attrimil", {})
    if not isinstance(settings, dict) or set(settings) - {"neighbor_step"}:
        raise ValueError("encoder.attrimil accepts only explicit physical neighbor_step")
    step = settings.get("neighbor_step", [meta["patch_size"], meta["patch_size"]])
    if (not isinstance(step, list) or len(step) != 2
            or any(type(v) not in {int, float} or not np.isfinite(v) or v <= 0 for v in step)):
        raise ValueError("AttriMIL neighbor_step must contain two finite positive coordinate distances")
    return step


def nearest_indices(coords, step):
    """Self,left,right,up,down,left-up,left-down,right-up,right-down.

    The source searches exact coordinate matches and substitutes self when
    absent. A hash table changes lookup cost only, preserving order and indices.
    """
    points = checked_points(coords)
    lookup = {tuple(p.tolist()): i for i, p in enumerate(points)}
    sx, sy = step
    offsets = [(0, 0), (0, -sy), (0, sy), (-sx, 0), (sx, 0),
               (-sx, -sy), (sx, -sy), (-sx, sy), (sx, sy)]
    result = np.empty((len(points), 9), dtype=np.int64)
    for index, point in enumerate(points):
        for slot, (dx, dy) in enumerate(offsets):
            neighbor = np.array([point[0] + dx, point[1] + dy])
            result[index, slot] = lookup.get(tuple(neighbor.tolist()), index)
    return torch.from_numpy(result)


class SlideBags(Dataset):
    def __init__(self, bundle, subset):
        self.bundle, self.rows, self.step = bundle, bundle.records(subset), neighbor_step(bundle)
        self.cache = {}
        labels = [int(r["target"]) for r in self.rows]
        self.slide_cls_ids = [np.flatnonzero(np.asarray(labels) == c) for c in range(len(bundle.classes))]

    def __len__(self):
        return len(self.rows)

    def getlabel(self, index):
        return int(self.rows[index]["target"])

    def __getitem__(self, index):
        row = self.rows[index]
        features, coords = self.bundle.load(row, require_coords=True)
        if index not in self.cache:
            self.cache[index] = nearest_indices(coords, self.step)
        return features, torch.tensor([self.getlabel(index)]), coords, self.cache[index]


class RecordingWeightedSampler(WeightedRandomSampler):
    def __iter__(self):
        self.order = list(super().__iter__())
        yield from self.order


def single_bag(batch):
    if len(batch) != 1:
        raise ValueError("AttriMIL original recipe requires one unmerged slide bag")
    return batch[0]

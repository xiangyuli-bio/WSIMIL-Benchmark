"""Actual ordered frozen-feature instances and TRAIN-only IIB confidence identity."""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_array, resolve
from .io import digest_object


def patch_names(bundle, row, count):
    if row.get('patch_ids_path'):
        names = load_array(resolve(bundle.root, row['patch_ids_path']), 'patch_ids')
        provenance = 'provided_original_patch_ids'
    else:
        path = resolve(bundle.root, row['path'])
        names = None
        if path.suffix in {'.h5', '.hdf5'}:
            with h5py.File(path, 'r') as handle:
                if 'patch_ids' in handle:
                    names = handle['patch_ids'][:]
        provenance = 'provided_original_patch_ids' if names is not None else 'content_bound_original_feature_row_identity'
        if names is None:
            names = np.asarray([f'feature_row_{i}' for i in range(count)])
    names = np.asarray(names)
    if names.shape != (count,):
        raise ValueError('IIB-MIL original patch IDs must match every feature row')
    if names.dtype.kind not in {'U', 'S', 'O'}:
        raise ValueError('IIB-MIL patch identities must be original string IDs')
    output = []
    for value in names.tolist():
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str) or not value:
            raise ValueError('IIB-MIL patch identity is missing or non-string')
        output.append(value)
    if len(set(output)) != len(output):
        raise ValueError('Duplicate original patch IDs within an IIB-MIL slide would corrupt confidence updates')
    return output, provenance


class Inputs:
    """Immutable real row inventory; no validation/test entry in confidence bank."""
    def __init__(self, bundle):
        if bundle.task_type != 'classification':
            raise ValueError('Released IIB-MIL defines classification, not a survival or multimodal objective')
        self.bundle = bundle
        self.rows = {row['slide_id']: row for subset in ['train', 'val', 'test'] for row in bundle.records(subset)}
        self.names, self.counts, self.provenance = {}, {}, {}
        self.offsets, self.bank_keys, self.bank_targets = {}, [], []
        for subset in ['train', 'val', 'test']:
            for row in bundle.records(subset):
                sid = row['slide_id']
                value, _ = bundle.load(row)
                names, provenance = patch_names(bundle, row, len(value))
                self.names[sid], self.counts[sid], self.provenance[sid] = names, len(value), provenance
                if subset == 'train':
                    self.offsets[sid] = len(self.bank_keys)
                    self.bank_keys.extend((sid, name, i) for i, name in enumerate(names))
                    self.bank_targets.extend([int(row['target'])] * len(value))
        if not self.bank_keys:
            raise ValueError('IIB-MIL needs real TRAIN instances for its confidence bank')
        self.identity = digest_object({'bundle': bundle.fingerprint, 'bank_keys': self.bank_keys,
                                      'bank_targets': self.bank_targets, 'counts': self.counts,
                                      'provenance': self.provenance,
                                      'actual_features':bundle.check(deep=True)['feature_inventory_sha256']})

    def initial_confidence(self):
        classes = len(self.bundle.classes)
        # Exact public WSIFeatDataset.build initialization for2/3classes. The
        # same explicitly recorded epsilon distribution defines other-class ports.
        value = torch.full((len(self.bank_keys), classes), 1e-6/(classes-1), dtype=torch.float32)
        target = torch.tensor(self.bank_targets, dtype=torch.long)
        value[torch.arange(len(value)), target] = 1. - 1e-6
        return value


def select_rows(features, training):
    """Literal public one-view permutation/prefix/zero-padding policy, length500."""
    values = np.asarray(features)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('IIB-MIL needs a nonempty real finite feature matrix')
    indices = np.arange(len(values), dtype=np.int64)
    if training and np.random.random_sample() > .5:
        indices = np.random.permutation(indices)
    indices = indices[:500]
    selected = values[indices]
    valid = np.ones(len(indices), dtype=np.bool_)
    if len(indices) < 500:
        residual = 500-len(indices)
        # Author appends float64 zeros, then explicitly casts the whole bag to
        # float32. Retain that value/order path and the repeated first-patch ID.
        selected = np.concatenate([selected, np.zeros((residual, values.shape[1]))])
        indices = np.concatenate([indices, np.repeat(indices[0], residual)])
        valid = np.concatenate([valid, np.zeros(residual, dtype=np.bool_)])
    return torch.from_numpy(selected).float(), torch.from_numpy(valid), torch.from_numpy(indices)


class Bags(Dataset):
    def __init__(self, inputs, subset):
        self.inputs, self.subset = inputs, subset
        self.rows = inputs.bundle.records(subset)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, _ = self.inputs.bundle.load(row)
        if len(features) != self.inputs.counts[row['slide_id']]:
            raise ValueError('IIB-MIL real feature/patch identity inventory changed')
        value, mask, original_rows = select_rows(features.numpy(), self.subset == 'train')
        offset = self.inputs.offsets.get(row['slide_id'])
        bank = original_rows+offset if offset is not None else torch.full_like(original_rows, -1)
        return {'features': value, 'mask': mask, 'target': int(row['target']),
                'slide_id': row['slide_id'], 'original_rows': original_rows, 'bank_indices': bank}

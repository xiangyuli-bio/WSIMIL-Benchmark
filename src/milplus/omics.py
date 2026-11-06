"""Explicit patient-aligned molecular inputs; never infer genes or impute missing assays."""
import csv
import json

import numpy as np
import pandas as pd
import torch

from .data import resolve


class OmicsTable:
    def __init__(self, bundle):
        config = bundle.config['omics']
        self.normalization = config.get('normalization', 'train_standard')
        if self.normalization not in {'train_standard', 'none', 'train_global_minmax_preserve_zero'}:
            raise ValueError('omics normalization must be train_standard, none, or train_global_minmax_preserve_zero (PIBD only)')
        if not config.get('preprocessing'):
            raise ValueError('omics.preprocessing must document the upstream assay transformation')
        self.columns = config.get('columns', [])
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError('omics.columns must explicitly list unique molecular feature names')
        if 'global_patient_id' in self.columns:
            raise ValueError('Patient identifiers cannot be molecular features')
        table_path = resolve(bundle.root, config['table'])
        self.files = [table_path]
        with table_path.open(newline='') as stream:
            header = next(csv.reader(stream))
        if len(set(header)) != len(header):
            raise ValueError('Duplicate omics CSV columns')
        frame = pd.read_csv(table_path, dtype=str, keep_default_na=False)
        required = {'global_patient_id', *self.columns}
        if not required <= set(frame):
            raise ValueError(f'Missing omics columns: {required - set(frame)}')
        if frame.global_patient_id.eq('').any() or frame.global_patient_id.duplicated().any():
            raise ValueError('Omics requires one unique nonempty global_patient_id per patient')
        expected = set(bundle.rows.global_patient_id)
        if set(frame.global_patient_id) != expected:
            raise ValueError('Omics and task patient sets differ; align/filter explicitly')
        self.values = frame.set_index('global_patient_id')[self.columns].apply(pd.to_numeric, errors='raise')
        values = self.values.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(values.astype(np.float32)).all():
            raise ValueError('Omics values must be finite and representable as float32; no implicit imputation')
        self.groups = []
        if config.get('pathways'):
            path = resolve(bundle.root, config['pathways'])
            self.files.append(path)
            self.groups = json.loads(path.read_text())
            if not isinstance(self.groups, list) or not self.groups:
                raise ValueError('Pathways JSON must be a nonempty ordered list of name/features objects')
            names = []
            for group in self.groups:
                if not isinstance(group, dict) or not group.get('name') or not group.get('features'):
                    raise ValueError('Each pathway needs name and nonempty features')
                genes = group['features']
                if len(set(genes)) != len(genes) or not set(genes) <= set(self.columns):
                    raise ValueError('Pathway features must be unique within each group and present in omics.columns')
                names.append(group['name'])
            if len(set(names)) != len(names):
                raise ValueError('Pathway names must be unique')
        self.group_indices = [[self.columns.index(gene) for gene in group['features']] for group in self.groups]

    def tensor(self, global_patient_id):
        return torch.tensor(self.values.loc[global_patient_id].to_numpy(dtype=np.float32))

    def training_statistics(self, bundle):
        if self.normalization == 'train_global_minmax_preserve_zero':
            raise ValueError('Global minmax zero restoration requires the PIBD adapter')
        ids = sorted({row['global_patient_id'] for row in bundle.records('train')})
        values = self.values.loc[ids].to_numpy(dtype=np.float64)
        if self.normalization == 'none':
            return torch.zeros(len(self.columns)), torch.ones(len(self.columns))
        mean, scale = values.mean(axis=0), values.std(axis=0)
        scale[scale < np.finfo(np.float32).eps] = 1.
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32)

"""Explicit patient-level continuous targets and project-stratified regression metrics."""
import csv

import numpy as np
import pandas as pd
import torch

from .data import resolve


class RegressionTargets:
    def __init__(self, bundle):
        cfg = bundle.config.get('regression_targets', {})
        self.columns = cfg.get('columns', [])
        if (not isinstance(self.columns, list) or not self.columns or
                any(not isinstance(c, str) or not c.strip() for c in self.columns) or
                len(set(self.columns)) != len(self.columns) or {'global_patient_id', 'project'} & set(self.columns)):
            raise ValueError('regression_targets.columns requires unique explicit output names')
        if not cfg.get('preprocessing') or not cfg.get('label_provenance'):
            raise ValueError('Regression targets require preprocessing and label_provenance')
        self.transform = cfg.get('transform')
        if self.transform not in {'none', 'log10_1p'}:
            raise ValueError('Declare regression_targets.transform: none or log10_1p')
        path = resolve(bundle.root, cfg['table'])
        self.files = [path]
        with path.open(newline='') as stream:
            header = next(csv.reader(stream))
        if len(header) != len(set(header)):
            raise ValueError('Duplicate regression target CSV columns')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not {'global_patient_id', 'project', *self.columns} <= set(frame):
            raise ValueError('Regression table needs global_patient_id, project and every target column')
        if frame.global_patient_id.duplicated().any() or frame[['global_patient_id', 'project']].eq('').any().any():
            raise ValueError('Regression targets need unique patients and nonempty project identifiers')
        if set(frame.global_patient_id) != set(bundle.rows.global_patient_id):
            raise ValueError('Regression target and task patient sets differ')
        frame = frame.set_index('global_patient_id')
        values = frame[self.columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Regression targets must be finite; missing assays are not zeros')
        if self.transform == 'log10_1p':
            if (values < 0).any():
                raise ValueError('log10_1p expression requires nonnegative assay values')
            values = np.log10(1 + values)
        if not np.isfinite(values.astype(np.float32)).all():
            raise ValueError('Regression targets are outside float32 range')
        self.values = pd.DataFrame(values, index=frame.index, columns=self.columns)
        self.projects = frame.project

    def tensor(self, patient):
        return torch.tensor(self.values.loc[patient].to_numpy(dtype=np.float32))


def regression_metrics(frame, outputs, unit='patient'):
    target = [f'target_{i}' for i in range(len(outputs))]
    raw = [f'raw_{i}' for i in range(len(outputs))]
    prediction = [f'prediction_{i}' for i in range(len(outputs))]
    if unit == 'patient':
        grouped = frame.groupby('global_patient_id', sort=True)
        if (grouped[['project', *target]].nunique() > 1).any().any():
            raise ValueError('Conflicting patient regression targets or projects')
        frame = grouped[['project', *target]].first().join(grouped[raw+prediction].mean()).reset_index()
    elif unit != 'slide':
        raise ValueError('Regression evaluation_unit must be patient or slide')
    arrays = frame[target+raw+prediction].to_numpy(dtype=float)
    if not np.isfinite(arrays).all():
        raise ValueError('Nonfinite regression prediction or target')
    correlations, per_project = [], {}
    for project, group in frame.groupby('project', sort=True):
        per_gene = {}
        for i, name in enumerate(outputs):
            y, p = group[f'target_{i}'].to_numpy(), group[f'prediction_{i}'].to_numpy()
            value = None
            if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0:
                value = float(np.corrcoef(y, p)[0, 1])
                correlations.append(value)
            per_gene[name] = value
        per_project[str(project)] = {'n': len(group), 'pearson': per_gene}
    return {'n': len(frame), 'unit': unit, 'outputs': list(outputs),
            'mse': float(np.square(frame[raw].to_numpy()-frame[target].to_numpy()).mean()),
            'mae': float(np.abs(frame[prediction].to_numpy()-frame[target].to_numpy()).mean()),
            'mean_pearson': float(np.mean(correlations)) if correlations else None,
            'defined_correlations': len(correlations),
            'undefined_correlations': len(per_project)*len(outputs)-len(correlations),
            'per_project': per_project}


def regression_epoch(model, bundle, groups, optimizer, device, config):
    losses = []
    for start in range(0, len(groups), model.batch_size):
        window = groups[start:start+model.batch_size]
        optimizer.zero_grad(set_to_none=True)
        k = model.training_k()  # The author's random top-k is shared across the whole batch.
        result = model.forward_batch_units(bundle, window, device, config, train=True, k=k)
        target = torch.stack([bundle.regression_targets.tensor(rows[0]['global_patient_id'])
                              for rows in window]).to(device)
        loss = torch.nn.functional.mse_loss(result['raw'], target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite regression loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        losses.append({'loss': float(loss.detach())})
    return losses

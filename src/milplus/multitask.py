"""Explicit slide-indexed heterogeneous targets, with missingness kept separate."""
import csv
import math

import numpy as np
import pandas as pd
import torch

from .data import resolve
from .metrics import classification_metrics, harrell_c_index


class MultiTaskTargets:
    def __init__(self, bundle):
        cfg = bundle.config.get('multitask', {})
        if cfg.get('schema_version') != 1 or not isinstance(cfg.get('tasks'), list) or not cfg['tasks']:
            raise ValueError('multitask requires schema_version1 and a nonempty ordered tasks list')
        self.bundle = bundle
        self.specs = []
        for spec in cfg['tasks']:
            if not isinstance(spec, dict) or not spec:
                raise ValueError('Each multitask target needs an explicit specification')
            item = dict(spec)
            name = item.get('name')
            if not isinstance(name, str) or not name.strip() or '.' in name:
                raise ValueError('Multitask names must be nonempty PyTorch module keys, without dots')
            if name in {s['name'] for s in self.specs}:
                raise ValueError('Duplicate multitask target name')
            if item.get('type') not in {'classification', 'regression', 'survival'}:
                raise ValueError('Multitask target type must be classification, regression or survival')
            if item.get('unit') not in {'slide', 'patient'} or not item.get('label_provenance'):
                raise ValueError('Every target requires unit: slide/patient and label_provenance')
            if item['type'] == 'classification':
                classes = item.get('classes')
                if not isinstance(classes, list) or len(classes) < 2 or any(not isinstance(s, str) or not s for s in classes) or len(set(classes)) != len(classes):
                    raise ValueError('Classification targets require ordered unique class names')
            if item['type'] == 'survival':
                if item['unit'] != 'patient' or item.get('time_unit') not in {'days', 'months', 'years'}:
                    raise ValueError('Survival targets require patient unit and explicit time unit')
                columns = [item.get('time_column'), item.get('event_column')]
            else:
                columns = [item.get('column')]
            if any(not isinstance(c, str) or not c.strip() for c in columns) or len(set(columns)) != len(columns):
                raise ValueError('Each target requires explicit distinct CSV columns')
            item['columns'] = columns
            self.specs.append(item)
        path = resolve(bundle.root, cfg['table'])
        self.files = [path]
        with path.open(newline='') as stream:
            header = next(csv.reader(stream))
        if len(header) != len(set(header)):
            raise ValueError('Duplicate multitask CSV columns')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {'slide_id', *[c for s in self.specs for c in s['columns']]}
        if not required <= set(frame) or frame.slide_id.eq('').any() or frame.slide_id.duplicated().any():
            raise ValueError('Multitask CSV requires unique slide_id and every declared target column')
        if set(frame.slide_id) != set(bundle.rows.slide_id):
            raise ValueError('Multitask table must exactly cover the declared slides; no silent row dropping')
        self.values = {}
        for row in frame.to_dict('records'):
            values = {}
            for spec in self.specs:
                raw = [row[c] for c in spec['columns']]
                if all(v == '' for v in raw):
                    values[spec['name']] = None
                    continue
                if any(v == '' for v in raw):
                    raise ValueError('Survival time and event must be jointly observed or jointly missing')
                if spec['type'] == 'classification':
                    if raw[0] not in spec['classes']:
                        raise ValueError(f"Unknown class in multitask target {spec['name']}; missing is empty, not an undeclared label")
                    value = spec['classes'].index(raw[0])
                else:
                    try:
                        nums = [float(v) for v in raw]
                    except ValueError as error:
                        raise ValueError('Multitask numeric targets must be finite numbers or empty') from error
                    if any(not math.isfinite(v) or abs(v) > np.finfo(np.float32).max for v in nums):
                        raise ValueError('Multitask numeric target outside finite float32 range')
                    if spec['type'] == 'survival':
                        if nums[0] <= 0 or nums[1] not in {0., 1.}:
                            raise ValueError('Survival time must be positive; event is0/1')
                        value = tuple(nums)
                    else:
                        value = nums[0]
                values[spec['name']] = value
            if all(v is None for v in values.values()):
                raise ValueError('Every included slide must have at least one observed target')
            self.values[row['slide_id']] = values
        for spec in self.specs:
            if not any(self.values[r['slide_id']][spec['name']] is not None for r in bundle.records('train')):
                raise ValueError('Every declared task requires an observed TRAIN target')
            if spec['unit'] == 'patient':
                grouped = {}
                for r in bundle.rows.to_dict('records'):
                    value = self.values[r['slide_id']][spec['name']]
                    if value is not None:
                        grouped.setdefault(r['global_patient_id'], set()).add(value)
                if any(len(v) > 1 for v in grouped.values()):
                    raise ValueError('Conflicting patient outcomes in multitask target')

    def tensors(self, rows, device='cpu'):
        result, observed = {}, {}
        for spec in self.specs:
            values = [self.values[r['slide_id']][spec['name']] for r in rows]
            observed[spec['name']] = torch.tensor([v is not None for v in values], device=device)
            sentinel = (-1., -1.) if spec['type'] == 'survival' else -1
            dtype = torch.long if spec['type'] == 'classification' else torch.float32
            result[spec['name']] = torch.tensor([sentinel if v is None else v for v in values], dtype=dtype, device=device)
        return result, observed

    def statistics(self):
        return {subset: {spec['name']: sum(self.values[r['slide_id']][spec['name']] is not None
                for r in self.bundle.records(subset)) for spec in self.specs} for subset in ['train', 'val', 'test']}


def task_metrics(frame, spec, index):
    """Standard metrics from explicit masks; undefined endpoints remain null."""
    prefix = f't{index}_'
    known = frame[frame[prefix+'observed'].astype(bool)].copy()
    if known.empty:
        return {'n': 0, 'unit': spec['unit'], 'defined': False}
    if spec['type'] == 'classification':
        columns = {prefix+'target': 'target', **{prefix+f'p{i}': f'p_{i}' for i in range(len(spec['classes']))}}
        return classification_metrics(known.rename(columns=columns), len(spec['classes']), spec['unit'])
    if spec['unit'] == 'patient':
        group = known.groupby('global_patient_id', sort=True)
        target_cols = [prefix+'time', prefix+'event'] if spec['type'] == 'survival' else [prefix+'target']
        if (group[target_cols].nunique() > 1).any().any():
            raise ValueError('Conflicting known patient outcomes during multitask evaluation')
        prediction = prefix+('risk' if spec['type'] == 'survival' else 'prediction')
        known = group[target_cols].first().join(group[prediction].mean()).reset_index()
    if spec['type'] == 'survival':
        score = harrell_c_index(known[prefix+'time'], known[prefix+'event'], known[prefix+'risk'])
        return {'n': len(known), 'unit': spec['unit'], 'c_index': score}
    target = known[prefix+'target'].to_numpy(float)
    prediction = known[prefix+'prediction'].to_numpy(float)
    error = prediction-target
    absolute = np.abs(error)
    return {'n': len(known), 'unit': spec['unit'], 'mse': float((error**2).mean()),
            'mae': float(absolute.mean()), 'smooth_l1': float(np.where(absolute < 1, .5*error**2, absolute-.5).mean())}


def multitask_metrics(bundle, frame):
    return {'n': len(frame), 'unit': 'slide',
            'tasks': {spec['name']: task_metrics(frame, spec, i) for i, spec in enumerate(bundle.multitask.specs)}}

"""Explicit complete multilabel supervision and patient-aware independent probabilities."""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


class MultilabelTargets:
    def __init__(self, bundle):
        if any(bundle.config.get(key, 'slide') not in {'slide', 'patient'} for key in ['training_unit', 'evaluation_unit']):
            raise ValueError('Multilabel units must be explicitly slide or patient')
        meta = bundle.config.get('multilabel')
        if not isinstance(meta, dict) or set(meta) != {'columns', 'label_provenance', 'empty_target_semantics'}:
            raise ValueError('multilabel requires columns, label_provenance and empty_target_semantics')
        columns = meta['columns']
        if not all(isinstance(c, str) and c.strip() for c in bundle.classes):
            raise ValueError('Multilabel endpoints require nonempty ordered names')
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns) or len(columns) != len(bundle.classes) or len(set(columns)) != len(columns):
            raise ValueError('Multilabel columns must uniquely follow the declared class order')
        if not all(isinstance(c, str) and c and c in bundle.rows for c in columns):
            raise ValueError('Missing multilabel target columns')
        for key in ['label_provenance', 'empty_target_semantics']:
            if not isinstance(meta[key], str) or not meta[key].strip():
                raise ValueError('Multilabel label provenance and explicit all-negative semantics are required')
        # Read as strings: blank/unknown/untested is never silently turned into negative.
        if not bundle.rows[columns].isin(['0', '1']).all().all():
            raise ValueError('Every multilabel target must explicitly be 0 or 1; unknown/missing is unsupported')
        self.columns = columns
        values = bundle.rows[columns].to_numpy(dtype=np.int64)
        bundle.rows['targets'] = [tuple(map(int, row)) for row in values]
        patient_level = bundle.config.get('evaluation_unit', 'slide') == 'patient' or bundle.config.get('training_unit', 'slide') == 'patient'
        if patient_level and (bundle.rows.groupby('global_patient_id').targets.nunique() > 1).any():
            raise ValueError('Patient-level multilabel operations require consistent patient targets')
        for subset in ['train', 'val', 'test']:
            y = values[bundle.rows.subset.eq(subset)]
            if any(len(np.unique(y[:, i])) != 2 for i in range(len(columns))):
                raise ValueError(f'{subset} requires positive and negative support for every multilabel endpoint')


def multilabel_metrics(frame, classes, unit='slide'):
    targets = [f'target_{i}' for i in range(classes)]
    scores = [f'p_{i}' for i in range(classes)]
    if classes < 2 or not set(targets+scores) <= set(frame) or frame.empty:
        raise ValueError('Multilabel metrics require nonempty aligned target and independent score columns')
    y, p = frame[targets].to_numpy(dtype=float), frame[scores].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Multilabel targets must be complete binary values')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Multilabel probabilities must independently lie in [0,1]')
    if unit == 'patient':
        if (frame.groupby('global_patient_id')[targets].nunique() > 1).any().any():
            raise ValueError('Patient multilabel aggregation requires consistent targets')
        groups = frame.groupby('global_patient_id', sort=True)
        y, p = groups[targets].first().to_numpy(dtype=int), groups[scores].mean().to_numpy(dtype=float)
    elif unit != 'slide':
        raise ValueError('Multilabel evaluation unit must be slide or patient')
    y = y.astype(np.int64)
    pred = p > .5  # Literal source threshold; .5 is negative.
    complete = all(len(np.unique(y[:, i])) == 2 for i in range(classes))
    bounded = np.clip(p, np.finfo(np.float64).eps, 1-np.finfo(np.float64).eps)
    return {'n': len(y), 'unit': unit, 'classes': classes, 'threshold': .5,
            'subset_accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(y, pred, average='micro', zero_division=0)),
            'log_loss': float(-(y*np.log(bounded)+(1-y)*np.log1p(-bounded)).mean()),
            'macro_auc': float(roc_auc_score(y, p, average='macro')) if complete else None,
            'micro_auc': float(roc_auc_score(y, p, average='micro')) if len(np.unique(y)) == 2 else None,
            'macro_ap': float(average_precision_score(y, p, average='macro')) if complete else None,
            'micro_ap': float(average_precision_score(y, p, average='micro')) if len(np.unique(y)) == 2 else None}

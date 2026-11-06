"""DSPA ordinal decisions on recorded student probabilities, with patient fusion.

The six-logit interface and decision rule are explicit reconstruction choices.
No cumulative-head rounding or generic argmax silently replaces that choice.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, roc_auc_score

DECISIONS = {'half_up_expected_grade_reference', 'argmax_logits_reference'}


def decide(probabilities, rule):
    p = np.asarray(probabilities, dtype=np.float64)
    if (rule not in DECISIONS or p.ndim != 2 or p.shape[1] != 6 or len(p) == 0
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1., rtol=1e-5, atol=1e-6)):
        raise ValueError('DSPA ordinal decision needs recorded finite six-grade probabilities')
    # Fixed addition order also avoids a BLAS GEMV/GEMM-dependent tie decision.
    scores = np.zeros(len(p), dtype=np.float64)
    for grade in range(6):
        scores = scores + p[:, grade] * grade
    predicted = (np.floor(scores + .5).clip(0, 5).astype(np.int64)
                 if rule == 'half_up_expected_grade_reference' else p.argmax(axis=1))
    return scores, predicted


def decision(frame):
    if 'dspamil_decision' not in frame or frame.empty:
        raise ValueError('DSPA predictions need their explicit ordinal decision provenance')
    values = set(frame.dspamil_decision)
    if len(values) != 1 or not values <= DECISIONS:
        raise ValueError('DSPA predictions contain mixed or unknown ordinal decisions')
    return next(iter(values))


def point_predictions(frame, *, rule, unit='slide'):
    columns = [f'p_{i}' for i in range(6)]
    if 'dspamil_decision' in frame and decision(frame) != rule:
        raise ValueError('DSPA declared decision differs from the recorded prediction provenance')
    if (rule not in DECISIONS or unit not in {'slide', 'patient'} or frame.empty
            or not {'slide_id', 'target', *columns} <= set(frame)
            or frame.slide_id.isna().any() or frame.slide_id.astype(str).eq('').any()
            or frame.slide_id.duplicated().any()):
        raise ValueError('DSPA needs unique actual slides, six probabilities and a declared decision/evaluation unit')
    data = frame.copy()
    labels = data.target.to_numpy(dtype=np.float64)
    probabilities = data[columns].to_numpy(dtype=np.float64)
    if (not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
            or (labels < 0).any() or (labels > 5).any()
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1., rtol=1e-5, atol=1e-6)):
        raise ValueError('DSPA recorded targets/probabilities must be finite actual six-grade distributions')
    if unit == 'patient':
        if ('global_patient_id' not in data or data.global_patient_id.isna().any()
                or data.global_patient_id.astype(str).str.strip().eq('').any()
                or (data.groupby('global_patient_id').target.nunique() != 1).any()):
            raise ValueError('DSPA patient probability fusion requires consistent global patient grades')
        data = data.groupby('global_patient_id', sort=True).agg(
            target=('target', 'first'), **{key: (key, 'mean') for key in columns}).reset_index()
    y = data.target.to_numpy(dtype=np.int64)
    p = data[columns].to_numpy(dtype=np.float64)
    scores, predicted = decide(p, rule)
    return data, y, p, scores, predicted


def metrics(frame, *, rule=None, unit='slide'):
    rule = decision(frame) if rule is None else rule
    data, y, p, scores, predicted = point_predictions(frame, rule=rule, unit=unit)
    qwk = cohen_kappa_score(y, predicted, labels=list(range(6)), weights='quadratic')
    auc = (float(roc_auc_score(y, p, labels=list(range(6)), multi_class='ovr', average='macro'))
           if set(y) == set(range(6)) else None)
    return {'n': len(y), 'unit': unit, 'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(np.mean([np.mean(predicted[y == c] == c) for c in np.unique(y)])),
        'macro_f1': float(f1_score(y, predicted, labels=list(range(6)), average='macro', zero_division=0)),
        'mae': float(np.abs(y - predicted).mean()), 'qwk': float(qwk) if np.isfinite(qwk) else None,
        'macro_auc': auc, 'log_loss': float(log_loss(y, p, labels=list(range(6)))),
        'prediction_rule': rule, 'decision_arithmetic': 'float64_six_grade_left_fold_over_recorded_student_probabilities',
        'patient_fusion': 'mean_six_grade_probabilities_before_ordinal_decision' if unit == 'patient' else None,
        'original_unpublished_ordinal_interface': 'explicit_six_logit_reconstruction'}

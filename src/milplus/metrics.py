"""Metrics and paired patient cluster bootstrap; seeds are never sample units."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def aggregate_patients(frame):
    pcols = [c for c in frame if c.startswith("p_")]
    if (frame.groupby("global_patient_id").target.nunique() > 1).any():
        raise ValueError("Patient probability aggregation requires one target per patient")
    metadata = frame.groupby("global_patient_id", sort=True).target.first()
    out = frame.groupby("global_patient_id", sort=True)[pcols].mean()
    if 'ranking_score' in frame:
        out['ranking_score'] = frame.groupby('global_patient_id', sort=True).ranking_score.mean()
    out["target"] = metadata
    return out.reset_index()


def classification_metrics(frame, classes, unit="slide"):
    decision = None
    if 'decision_rule' in frame:
        if classes != 2 or set(frame.decision_rule) != {'binary_positive_at_half'}:
            raise ValueError('Unsupported or inconsistent explicit binary decision rule')
        decision = 'binary_positive_at_half'
    if 'ranking_score' in frame and (classes != 2 or not np.isfinite(frame.ranking_score.to_numpy(dtype=float)).all()):
        raise ValueError('Explicit ranking_score requires finite binary prediction scores')
    if unit == "patient":
        frame = aggregate_patients(frame)
    elif unit != "slide":
        raise ValueError("evaluation unit must be slide or patient")
    y = frame.target.to_numpy(dtype=int)
    p = frame[[f"p_{i}" for i in range(classes)]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    p = p / p.sum(1, keepdims=True)  # remove float32 serialization rounding, after validation
    ranking = frame.ranking_score.to_numpy(dtype=float) if 'ranking_score' in frame else None
    if ranking is not None and (classes != 2 or not np.isfinite(ranking).all()):
        raise ValueError('Explicit ranking_score requires finite binary prediction scores')
    pred = (p[:, 1] >= .5).astype(int) if decision else p.argmax(1)
    complete = len(np.unique(y)) == classes
    result = {"n": len(y), "unit": unit, "accuracy": float(accuracy_score(y, pred)),
              "balanced_accuracy": float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])),
              "macro_f1": float(f1_score(y, pred, labels=list(range(classes)), average="macro", zero_division=0)),
              "log_loss": float(log_loss(y, p, labels=list(range(classes)))),
              "macro_auc": None, "auprc": None, "qwk": None}
    if complete:
        binary_score = p[:, 1] if ranking is None else ranking
        result["macro_auc"] = float(roc_auc_score(y, binary_score) if classes == 2 else
                                     roc_auc_score(y, p, average="macro", multi_class="ovr"))
        result["auprc"] = float(average_precision_score(y, binary_score) if classes == 2 else
                                average_precision_score(np.eye(classes)[y], p, average="macro"))
        qwk = cohen_kappa_score(y, pred, labels=list(range(classes)), weights="quadratic")
        result["qwk"] = float(qwk) if np.isfinite(qwk) else None
    return result


def ordinal_metrics(frame, classes, unit='slide'):
    """Rounded expected grades from cumulative BCE heads; never fabricate class probabilities."""
    if type(classes) is not int or classes < 2 or unit not in {'slide', 'patient'}:
        raise ValueError('Ordinal metrics require ordered classes and a declared evaluation unit')
    if unit == 'patient':
        if (frame.groupby('global_patient_id').target.nunique() > 1).any():
            raise ValueError('Patient ordinal aggregation requires consistent targets')
        frame = frame.groupby('global_patient_id', sort=True).agg(target=('target', 'first'), ordinal_score=('ordinal_score', 'mean')).reset_index()
    y = frame.target.to_numpy(dtype=float)
    scores = frame.ordinal_score.to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all() or (y < 0).any() or (y >= classes).any():
        raise ValueError('Ordinal targets must be ordered integer class IDs')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > classes-1).any():
        raise ValueError('Expected ordinal grade must lie within the declared grade range')
    pred = np.rint(scores).astype(int)
    qwk = cohen_kappa_score(y, pred, labels=list(range(classes)), weights='quadratic')
    return {'n': len(y), 'unit': unit, 'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])),
            'macro_f1': float(f1_score(y, pred, labels=list(range(classes)), average='macro', zero_division=0)),
            'mae': float(np.abs(y-pred).mean()), 'qwk': float(qwk) if np.isfinite(qwk) else None,
            'macro_auc': None, 'log_loss': None,
            'prediction_rule': 'round(sum(sigmoid(cumulative logits))); no normalized class probability assumption'}


def harrell_c_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    concordant, pairs = 0.0, 0
    for i in range(len(time)):
        eligible = (time > time[i]) & bool(event[i])
        pairs += int(eligible.sum())
        concordant += float((risk[i] > risk[eligible]).sum() + 0.5 * (risk[i] == risk[eligible]).sum())
    return None if pairs == 0 else concordant / pairs


def paired_bootstrap(left, right, classes, metric="macro_auc", unit="slide", repetitions=2000, seed=2026):
    """One matched run per side. Whole patients sampled with replacement."""
    ordinal = ('ordinal_score' in left or 'ordinal_score' in right or
               (metric in {'qwk', 'mae'} and ('dspamil_decision' in left or 'dspamil_decision' in right)))
    if ordinal and metric not in {'qwk', 'accuracy', 'balanced_accuracy', 'macro_f1', 'mae'}:
        raise ValueError('Ordinal threshold heads support grade metrics, not an invented class-probability metric')

    def metric_function(frame, classes, unit):
        if 'dspamil_decision' in frame:
            if classes != 6:
                raise ValueError('DSPA paired inference preserves all six ordered Grade Groups')
            from .dspamil_metrics import metrics as dspa_metrics
            return dspa_metrics(frame, unit=unit)
        if not ordinal:
            return classification_metrics(frame, classes, unit)
        if 'ordinal_score' in frame:
            return ordinal_metrics(frame, classes, unit)
        columns = [f'p_{i}' for i in range(classes)]
        if not set(columns) <= set(frame):
            raise ValueError('Paired score definition requires cumulative ordinal scores or class probabilities')
        p = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError('Invalid class probabilities for paired ordinal grade comparison')
        # Preserve each head's actual decision rule. A CE classifier uses argmax,
        # including probability fusion before argmax when patient evaluation is declared.
        point = aggregate_patients(frame) if unit == 'patient' else frame.copy()
        if 'decision_rule' in frame:
            if classes != 2 or set(frame.decision_rule) != {'binary_positive_at_half'}:
                raise ValueError('Unsupported paired binary decision rule')
            point['ordinal_score'] = (point.p_1 >= .5).astype(float)
        else:
            point['ordinal_score'] = point[columns].to_numpy(dtype=float).argmax(1).astype(float)
        result = ordinal_metrics(point, classes, 'slide' if unit == 'patient' else unit)
        result['unit'] = unit
        return result
    keys = ["slide_id", "global_patient_id", "target", "subset"]
    for frame in (left, right):
        if frame.slide_id.duplicated().any():
            raise ValueError("Duplicate slides: do not pool seeds as independent observations")
        if set(frame.subset) != {"test"}:
            raise ValueError("Paired inference requires locked test predictions")
    a, b = (f.sort_values("slide_id").reset_index(drop=True) for f in (left, right))
    if not a[keys].equals(b[keys]):
        raise ValueError("Predictions differ in slide/patient/target/test identity")
    if repetitions < 20:
        raise ValueError("At least 20 bootstrap replicates required")
    ma, mb = metric_function(a, classes, unit)[metric], metric_function(b, classes, unit)[metric]
    if ma is None or mb is None:
        raise ValueError("Metric undefined on the original test sample")
    patients = a.global_patient_id.unique()
    indices = {p: np.flatnonzero(a.global_patient_id.to_numpy() == p) for p in patients}
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        chosen = rng.choice(patients, len(patients), replace=True)
        # Give each draw a distinct cluster ID; repeated draws must not be collapsed.
        aa, bb = [], []
        for draw, patient in enumerate(chosen):
            for frame, dest in ((a, aa), (b, bb)):
                part = frame.iloc[indices[patient]].copy()
                part["global_patient_id"] = str(draw)
                part["slide_id"] = str(draw) + ":" + part.slide_id.astype(str)
                dest.append(part)
        va = metric_function(pd.concat(aa), classes, unit)[metric]
        vb = metric_function(pd.concat(bb), classes, unit)[metric]
        if va is not None and vb is not None:
            values.append(va - vb)
    if len(values) < repetitions // 2:
        raise ValueError("Too many undefined bootstrap replicates; insufficient class support")
    return {"metric": metric, "unit": unit, "difference_left_minus_right": ma - mb,
            "ci95": np.quantile(values, [.025, .975]).tolist(), "valid_replicates": len(values),
            "requested_replicates": repetitions, "patients": len(patients), "seed": seed}

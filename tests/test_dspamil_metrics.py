"""Independent ordinal and patient counterexamples, including paired inference."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from milplus.dspamil_metrics import metrics, point_predictions
from milplus.engine import score
from milplus.metrics import paired_bootstrap


def predictions(rule='half_up_expected_grade_reference'):
    records = []
    for patient in range(18):
        grade = patient % 6
        p = np.eye(6)[grade]
        if grade in {0, 5}:
            p = np.asarray([.4, 0, 0, 0, 0, .6])
        for slide in range(2):
            records.append({'slide_id': f'p{patient}s{slide}', 'global_patient_id': f'p{patient}',
                'target': grade, 'subset': 'test', 'dspamil_decision': rule,
                **{f'p_{i}': float(x) for i, x in enumerate(p)}})
    return pd.DataFrame(records)


def independent_qwk(targets, predicted):
    matrix = np.zeros((6, 6))
    for true, guess in zip(targets, predicted):
        matrix[int(true), int(guess)] += 1
    n = len(targets)
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / n
    distance = (np.arange(6)[:, None] - np.arange(6)[None, :]) ** 2
    return 1 - (matrix * distance).sum() / (expected * distance).sum()


def test_native_grade_decision_and_common_score_keep_the_declared_rule():
    data = predictions()
    expected = [3 if grade in {0, 5} else grade for grade in data.target]
    result = metrics(data)
    assert result['qwk'] == independent_qwk(data.target, expected)
    argmax = metrics(predictions('argmax_logits_reference'))
    assert argmax['qwk'] == independent_qwk(data.target, data[[f'p_{i}' for i in range(6)]].to_numpy().argmax(1))
    assert result['qwk'] != argmax['qwk']
    bundle = SimpleNamespace(config={'evaluation_unit': 'slide'})
    assert score(bundle, data) == result
    assert result['macro_auc'] is not None and result['log_loss'] > 0
    with pytest.raises(ValueError, match='recorded prediction provenance'):
        metrics(data, rule='argmax_logits_reference')


def test_patient_probabilities_fuse_before_expected_grade_and_half_up_rounding():
    data = pd.DataFrame([
        {'slide_id': 'a0', 'global_patient_id': 'a', 'target': 0, 'p_0': .6, 'p_1': .4},
        {'slide_id': 'a1', 'global_patient_id': 'a', 'target': 0, 'p_0': .5, 'p_1': .5},
        {'slide_id': 'b0', 'global_patient_id': 'b', 'target': 5, 'p_0': 0., 'p_1': 0., 'p_5': 1.},
    ]).fillna(0.)
    for grade in range(2, 5):
        data[f'p_{grade}'] = 0.
    point, targets, p, scores, predicted = point_predictions(data,
        rule='half_up_expected_grade_reference', unit='patient')
    assert point.global_patient_id.tolist() == ['a', 'b']
    assert p[0, 0] == .55 and p[0, 1] == .45
    assert scores.tolist() == [.45, 5.] and predicted.tolist() == [0, 5]
    assert metrics(data, rule='half_up_expected_grade_reference', unit='patient')['qwk'] == 1.
    data.loc[data.slide_id == 'a1', 'target'] = 1
    with pytest.raises(ValueError, match='consistent global patient'):
        metrics(data, rule='half_up_expected_grade_reference', unit='patient')


def test_logits_convenience_and_export_use_identical_arithmetic_near_half_grade():
    from milplus.dspamil_training import grade_predictions
    logits = torch.log(torch.tensor([[1 / 6, 1e-20, 1e-20, 5 / 6, 1e-20, 1e-20]], dtype=torch.float32))
    p = logits.softmax(-1).numpy()[0]
    frame = pd.DataFrame([{'slide_id': 'boundary', 'target': 2,
        **{f'p_{i}': float(v) for i, v in enumerate(p)}}])
    unused, targets, probabilities, scores, predicted = point_predictions(frame,
        rule='half_up_expected_grade_reference')
    expected = int(np.floor(sum(float(p[i]) * i for i in range(6)) + .5))
    assert predicted.tolist() == grade_predictions(logits, 'half_up_expected_grade_reference').tolist() == [expected]


@pytest.mark.parametrize('unit', ['slide', 'patient'])
def test_paired_patient_bootstrap_preserves_DSPA_decision_and_repeated_draws(unit):
    left = predictions()
    right = predictions('argmax_logits_reference')
    result = paired_bootstrap(left, right, 6, metric='qwk', unit=unit, repetitions=40, seed=73)
    expected = metrics(left, unit=unit)['qwk'] - metrics(right, unit=unit)['qwk']
    assert result['difference_left_minus_right'] == expected
    assert result['patients'] == 18 and result['valid_replicates'] == 40
    assert len(result['ci95']) == 2
    # A normal CE head retains its argmax rule when compared with DSPA.
    ce = right.drop(columns=['dspamil_decision'])
    ordinary = paired_bootstrap(left, ce, 6, metric='qwk', unit=unit, repetitions=40, seed=73)
    assert ordinary['difference_left_minus_right'] == expected and ordinary['ci95'] == result['ci95']


@pytest.mark.parametrize('kind', ['mixed_decision', 'duplicate_slide', 'nonfinite', 'negative_probability'])
def test_ambiguous_prediction_identity_or_distribution_is_rejected(kind):
    data = predictions()
    if kind == 'mixed_decision':
        data.loc[0, 'dspamil_decision'] = 'argmax_logits_reference'
    elif kind == 'duplicate_slide':
        data.loc[0, 'slide_id'] = data.loc[1, 'slide_id']
    elif kind == 'nonfinite':
        data.loc[0, 'p_0'] = np.nan
    else:
        data.loc[0, 'p_0'] = -.1
    with pytest.raises(ValueError):
        metrics(data)

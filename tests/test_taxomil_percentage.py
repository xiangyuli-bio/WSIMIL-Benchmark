"""Regression from actual full native BRACS epoch11 VAL (four of14 correct)."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from milplus.taxomil import text_metrics
from milplus.upstream import load_definitions


def test_generated_percentages_match_literal_source_division_before_multiplication():
    literal = load_definitions('taxomil', 'trainer/test.py', ['calculate_metrics'],
        {'f1_score': f1_score, 'cohen_kappa_score': cohen_kappa_score}).calculate_metrics
    frame = pd.DataFrame({
        'taxomil_coarse_target': ['benign'] * 14,
        'taxomil_fine_target': ['normal'] * 14,
        'taxomil_coarse_text': ['benign'] * 4 + ['malignant'] * 10,
        'taxomil_fine_text': ['normal'] * 2 + ['other'] * 12,
    })
    actual = text_metrics(frame)
    expected = {}
    for level in ['coarse', 'fine']:
        truth = frame['taxomil_' + level + '_target'].tolist()
        predictions = frame['taxomil_' + level + '_text'].tolist()
        accuracy, weighted_f1, kappa = literal(predictions, truth)
        correct = sum(a == b for a, b in zip(truth, predictions))
        # This exact input exposed the old adapter's algebraically equivalent
        # but bit-different multiply-first expression during real training.
        assert accuracy != 100 * correct / len(truth)
        assert actual['taxomil_' + level + '_accuracy'] == accuracy
        assert actual['taxomil_' + level + '_weighted_f1'] == weighted_f1
        assert actual['taxomil_' + level + '_kappa'] == kappa
        expected[level] = accuracy
    assert actual['taxomil_score'] == (expected['coarse'] + expected['fine']) / 2

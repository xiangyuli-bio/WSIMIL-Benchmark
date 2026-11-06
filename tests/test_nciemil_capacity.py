"""Native capacity and preparation-control guards, not model-budget evidence."""
import json
from types import SimpleNamespace

import pytest

from milplus.nciemil_capacity import native_selection_minimum, validate_selection_capacity
import milplus.nciemil_prepare as preparation
import milplus.nciemil_cache as cache


@pytest.mark.parametrize('k', [128, 512])
def test_native_selection_capacity_exact_boundary_and_whole_larger_bags(k):
    assert native_selection_minimum(k) == 2 * k
    records = {'boundary': {'count': 2*k}, 'larger': {'count': 16848}}
    before = {name: dict(row) for name, row in records.items()}
    assert validate_selection_capacity(records, k) is None
    assert records == before


@pytest.mark.parametrize('k', [128, 512])
@pytest.mark.parametrize('count_kind', ['minimum_minus_one', 'smallest_brca_bag'])
def test_one_small_original_bag_rejects_full_cohort_without_padding_or_filtering(k, count_kind):
    count = 2*k-1 if count_kind == 'minimum_minus_one' else 128
    with pytest.raises(ValueError, match='at least2K real distinct patches'):
        validate_selection_capacity({'sufficient': {'count': 16848}, 'original_small': {'count': count}}, k)


@pytest.mark.parametrize('k', [1, 64, 256, 0, '128'])
def test_arbitrary_k_cannot_work_around_native_capacity(k):
    with pytest.raises(ValueError, match='K must be128 or512'):
        native_selection_minimum(k)


class ControlInputs:
    """Only orchestration markers; no fake clinical feature/model evidence."""
    def __init__(self, counts):
        self.records = {str(i): {'count': count} for i, count in enumerate(counts)}


def settings():
    return dict(epochs=3, lr=.0001, weight_decay=.00001, bag_batch_size=2, seed=2026)


def test_invalid_k_is_rejected_before_reading_input_or_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'FilterInputs', lambda data: pytest.fail('No input should be loaded'))
    with pytest.raises(ValueError, match='K must be128 or512'):
        preparation.prepare('UNUSED_CONTROL_INPUT', 'UNUSED_WEIGHTS', tmp_path/'run', settings(), 64)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('resume', [False, True])
def test_small_cohort_fails_before_any_filter_update_checkpoint_or_output(tmp_path, monkeypatch, resume):
    monkeypatch.setattr(preparation, 'FilterInputs', ControlInputs)
    monkeypatch.setattr(preparation, 'train_filter', lambda *a, **kw: pytest.fail('Must not train an impossible cohort'))
    monkeypatch.setattr(preparation, 'extract_selected', lambda *a, **kw: pytest.fail('Must not score patches'))
    with pytest.raises(ValueError, match='at least2K real distinct patches'):
        preparation.prepare(ControlInputs([300, 128]), 'UNUSED_WEIGHTS', tmp_path/'run', settings(),
                            128, resume=resume)
    assert not (tmp_path/'run').exists()


def test_cache_has_identical_capacity_guard_before_checkpoint_or_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'FilterInputs', ControlInputs)
    monkeypatch.setattr(cache, 'SwinFilter', lambda *a, **kw: pytest.fail('Must not construct a model'))
    with pytest.raises(ValueError, match='at least2K real distinct patches'):
        cache.extract_selected(ControlInputs([128]), tmp_path/'no_checkpoint', 'UNUSED_WEIGHTS',
                               tmp_path/'no_output', 128)
    assert not (tmp_path/'no_output').exists()


@pytest.mark.parametrize('k', [128, 512])
def test_valid_orchestration_forwards_exact_original_inputs_settings_and_native_k(tmp_path, monkeypatch, k):
    """Control spies only: no claim that this executes a Swin training budget."""
    monkeypatch.setattr(preparation, 'FilterInputs', ControlInputs)
    inputs = ControlInputs([2*k, 16848])
    calls = []
    def train(actual_inputs, weights, weight_sha, output, **kwargs):
        calls.append(('train', actual_inputs, weights, weight_sha, kwargs))
        output.mkdir()
        (output/'status.json').write_text(json.dumps({'state': 'completed'}))
    def extract(actual_inputs, run, weights, output, actual_k, device):
        calls.append(('extract', actual_inputs, weights, actual_k, device))
        return output
    monkeypatch.setattr(preparation, 'train_filter', train)
    monkeypatch.setattr(preparation, 'extract_selected', extract)
    result = preparation.prepare(inputs, 'CONTROL_WEIGHTS', tmp_path/'run', settings(), k, device='cpu')
    assert result['k'] == k and result['state'] == 'completed'
    assert calls[0] == ('train', inputs, 'CONTROL_WEIGHTS', preparation.WEIGHTS_SHA256,
                       dict(device='cpu', resume=False, stop_after=None, **settings()))
    assert calls[1] == ('extract', inputs, 'CONTROL_WEIGHTS', k, 'cpu')
    assert list(inputs.records.values()) == [{'count': 2*k}, {'count': 16848}]

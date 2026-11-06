"""Explicit covariance precision option; preserve literal source default."""
import numpy as np
import pytest
import torch
from sklearn.mixture import BayesianGaussianMixture
from threadpoolctl import threadpool_limits

from milplus.cdpmil import CDPMILSource


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_assignment_and_original_float32_means_match_independent_reference(dtype):
    torch.set_num_threads(1)
    x = torch.from_numpy(np.random.default_rng(73).normal(size=(87, 8)).astype(np.float32))
    model = CDPMILSource(8, 2, clusters=3, aggregation_dtype=dtype)
    original = x.numpy()
    reference = original.astype(dtype)
    with threadpool_limits(limits=1):
        labels = BayesianGaussianMixture(n_components=3, random_state=0, max_iter=30,
            weight_concentration_prior=.1).fit(reference).predict(reference)
        expected = np.array([np.mean(original[labels == i], axis=0) for i in np.unique(labels)])
        actual = model.aggregate(x)
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert model.protocol()['aggregation_dtype'] == dtype


def test_default_stays_literal_float32_and_precision_repair_is_not_implicit():
    assert CDPMILSource(8, 2).aggregation_dtype == 'float32'
    with pytest.raises(ValueError, match='aggregation_dtype'):
        CDPMILSource(8, 2, aggregation_dtype='float16')


def test_full_covariance_and_budgets_preserved_and_cache_binds_precision(monkeypatch):
    import milplus.cdpmil as module
    calls = []
    class Mixture:
        def __init__(self, **kwargs):
            assert kwargs == {'n_components': 10, 'random_state': 0, 'max_iter': 30,
                              'weight_concentration_prior': .1}
        def fit(self, value):
            calls.append(value.dtype)
            return self
        def predict(self, value):
            return np.arange(len(value)) % 10
    monkeypatch.setattr(module, 'BayesianGaussianMixture', Mixture)
    x = torch.randn(30, 16)
    model = CDPMILSource(16, 2, aggregation_dtype='float64')
    first = model.aggregate(x)
    assert torch.equal(first, model.aggregate(x.clone())) and calls == [np.dtype('float64')]
    # Even an explicit in-memory diagnostic switch cannot reuse wrong-precision
    # assignments from the disposable content cache.
    model.aggregation_dtype = 'float32'
    model.aggregate(x)
    assert calls == [np.dtype('float64'), np.dtype('float32')]

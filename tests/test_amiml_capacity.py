"""Reject impossible real populations before clustering without changing native R."""
import h5py
import pytest
import torch

from milplus.amiml import AMIML
from milplus.amiml_data import check_cluster_capacity, patient_bags
from milplus.amiml_demo import create_amiml_demo
from milplus.data import Bundle


def shorten(bundle, row, count):
    with h5py.File(bundle.root / row['path'], 'r+') as handle:
        values = handle['features'][:count]
        del handle['features']
        handle['features'] = values


def test_native_R100_short_bags_reject_before_kmeans_or_any_update(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    bundle = Bundle(create_amiml_demo(tmp_path / 'data', dimension=16, per_cluster=12))
    model = AMIML(bundle, R=100)
    def forbidden(*args, **kwargs):
        raise AssertionError('Impossible population must not start full TRAIN clustering')
    monkeypatch.setattr('milplus.amiml.train_kmeans', forbidden)
    with pytest.raises(ValueError, match='needs at least 400'):
        model.prepare(bundle, 2026, 'cpu')
    assert model.optimizer_steps == [0] * 4
    assert not model.prepared


def test_capacity_is_patient_sum_not_individual_slide_size(tmp_path):
    bundle = Bundle(create_amiml_demo(tmp_path / 'data', dimension=16, per_cluster=12))
    patient = next(rows for rows in patient_bags(bundle, 'train') if len(rows) == 2)
    for row in patient:
        shorten(bundle, row, 24)
    check_cluster_capacity(bundle, 11)  # 24+24 >=4*11, even though each slide <44.


def test_heldout_short_bags_are_not_silently_filtered(tmp_path):
    bundle = Bundle(create_amiml_demo(tmp_path / 'data', dimension=16, per_cluster=12))
    patient = next(rows for rows in patient_bags(bundle, 'test') if len(rows) == 1)
    shorten(bundle, patient[0], 43)
    with pytest.raises(ValueError, match=r'\(test\) has 43, needs at least 44'):
        check_cluster_capacity(bundle, 11)


@pytest.mark.parametrize('R', [0, -1, True, 1.2])
def test_capacity_requires_explicit_positive_integer_R(tmp_path, R):
    bundle = Bundle(create_amiml_demo(tmp_path / 'data', dimension=16, per_cluster=12))
    with pytest.raises(ValueError, match='positive integer'):
        check_cluster_capacity(bundle, R)

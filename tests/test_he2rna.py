"""HE2RNA native output/gradient, spatial input, target isolation and regression tests."""
import copy

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.he2rna import HE2RNARegression, load_he2rna, spatial_supertiles
from milplus.models import build_bundle_model
from milplus.regression import regression_epoch, regression_metrics
from milplus.upstream import source_root


@pytest.fixture(autouse=True)
def source_and_cache(tmp_path, monkeypatch):
    if not source_root('he2rna').is_dir():
        pytest.skip('Fetch pinned HE2RNA source')
    monkeypatch.setenv('MILPLUS_HE2RNA_CACHE', str(tmp_path/'cache'))
    torch.set_num_threads(1)


def fixture(tmp_path, **options):
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, regression=True))
    cfg = {'layers': [16, 12], 'ks': [1, 2, 4], 'dropout': 0., 'input_mode': 'tiles', **options}
    model = build_bundle_model('he2rna', bundle, cfg)
    model.prepare(bundle, 2, 'cpu')
    return model, bundle


@pytest.mark.parametrize('training', [False, True])
def test_native_positive_unpadded_output_and_gradient(tmp_path, training):
    model, bundle = fixture(tmp_path)
    author = load_he2rna().HE2RNA(bundle.dim, 3, layers=[16, 12], ks=[1, 2, 4], dropout=0.)
    author.load_state_dict(model.network.state_dict())
    model.train(training); author.train(training)
    x = torch.rand(7, bundle.dim, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    np.random.seed(77); a = model(x)['raw']
    np.random.seed(77); b = author(xx.T[None])
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    target = torch.randn(1, 3)
    (a-target).square().mean().backward(); (b-target).square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=0, rtol=0)
    for p, q in zip(model.network.parameters(), author.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_signed_fm_rows_are_valid_and_prediction_relu_is_not_training_relu(tmp_path):
    model, bundle = fixture(tmp_path, layers=[])
    with torch.no_grad():
        model.network.conv0.weight.fill_(1.)
        model.network.conv0.bias.fill_(0.)
    x = -torch.rand(7, bundle.dim, requires_grad=True)
    result = model(x, k=2)
    assert (result['raw'] < 0).all() and (result['prediction'] == 0).all()
    result['raw'].square().mean().backward()
    assert model.network.conv0.weight.grad.abs().sum() > 0


def test_coordinates_not_features_drive_supertiles_and_cache_is_verified(tmp_path, monkeypatch):
    x = torch.tensor([[1., 3.], [100., 99.], [2., 4.], [101., 100.]])
    coords = torch.tensor([[0., 0.], [0., 1.], [100., 100.], [100., 101.]])
    result = spatial_supertiles(x, coords, count=2, seed=8)
    expected = torch.stack([x[:2].mean(0), x[2:].mean(0)])
    torch.testing.assert_close(result.sort(dim=0).values, expected.sort(dim=0).values)
    torch.testing.assert_close(result, spatial_supertiles(x, coords, count=2, seed=8), rtol=0, atol=0)
    file = next((tmp_path/'cache').glob('*.npz'))
    np.savez(file, features=np.zeros((2, 2), dtype=np.float32), sha256='bad')
    with pytest.raises(ValueError, match='cache checksum'):
        spatial_supertiles(x, coords, count=2, seed=8)
    with pytest.raises(ValueError, match='physical coordinates'):
        spatial_supertiles(x, None, count=2)
    with pytest.raises(ValueError, match='distinct physical'):
        spatial_supertiles(x, torch.zeros(4, 2), count=2)


def test_train_bias_uses_training_targets_not_validation_or_test(tmp_path):
    model, bundle = fixture(tmp_path)
    expected = torch.stack([bundle.regression_targets.tensor(g[0]['global_patient_id'])
                            for g in units(bundle, 'train')]).mean(0)
    torch.testing.assert_close(model.network.layers[-1].bias, expected)
    held = bundle.rows.query("subset != 'train'").global_patient_id.unique()
    bundle.regression_targets.values.loc[held] = 1e5
    other = build_bundle_model('he2rna', bundle, {'layers': [], 'ks': [1], 'input_mode': 'tiles'})
    other.prepare(bundle, 77, 'cpu')
    torch.testing.assert_close(other.network.layers[-1].bias, expected)


def test_random_k_shared_by_batch_and_mean_mse_update_matches_native(tmp_path):
    model, bundle = fixture(tmp_path, batch_size=3)
    reference = copy.deepcopy(model)
    original_load = bundle.load
    bundle.load = lambda row, **kwargs: tuple(a[:8] if a is not None else None for a in original_load(row))
    groups = units(bundle, 'train')[:5]
    options = {'max_instances': None, 'grad_clip': 1e5}
    optimizer, reference_optimizer = [torch.optim.Adam(m.parameters(), lr=.0003) for m in [model, reference]]
    np.random.seed(121)
    losses = regression_epoch(model, bundle, groups, optimizer, 'cpu', options)
    np.random.seed(121)
    expected = []
    for start in [0, 3]:
        window = groups[start:start+3]
        x = torch.stack([bundle.load(rows[0])[0].T for rows in window])
        y = torch.stack([bundle.regression_targets.tensor(rows[0]['global_patient_id']) for rows in window])
        # Native padding rule requires nonnegative features. Compare direct conv + native sampled top-k.
        k = reference.training_k()
        raw = reference.network.conv(x).topk(k, dim=2).values.mean(2)
        loss = (raw-y).square().mean()
        reference_optimizer.zero_grad(); loss.backward(); reference_optimizer.step()
        expected.append(float(loss.detach()))
    np.testing.assert_allclose([r['loss'] for r in losses], expected, rtol=1e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('defect', ['duplicate', 'missing', 'nonfinite', 'negative', 'project'])
def test_expression_target_contract_rejects_invalid_data(tmp_path, defect):
    root = create_demo(tmp_path/'data', patients=20, regression=True)
    frame = pd.read_csv(root/'expression.csv')
    if defect == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == 'missing':
        frame = frame.iloc[1:]
    elif defect == 'project':
        frame.loc[0, 'project'] = ''
    else:
        frame.loc[0, 'synthetic_gene_A'] = {'nonfinite': float('inf'), 'negative': -1.}[defect]
    frame.to_csv(root/'expression.csv', index=False)
    with pytest.raises(ValueError):
        Bundle(root)


def test_targets_transform_and_identity_hash(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, regression=True)
    bundle = Bundle(root)
    original = pd.read_csv(root/'expression.csv')
    np.testing.assert_allclose(bundle.regression_targets.values.to_numpy(),
                               np.log10(1+original[bundle.regression_targets.columns].to_numpy()))
    original.loc[0, 'synthetic_gene_A'] += 1
    original.to_csv(root/'expression.csv', index=False)
    assert Bundle(root).fingerprint != bundle.fingerprint


def test_project_correlations_are_reported_separately_and_patients_not_duplicated():
    frame = pd.DataFrame({'global_patient_id': ['p0', 'p0', 'p1', 'p2', 'p3'],
        'project': ['a', 'a', 'a', 'b', 'b'], 'target_0': [1., 1., 2., 3., 4.],
        'raw_0': [1., 1., 2., 4., 3.], 'prediction_0': [1., 1., 2., 4., 3.]})
    result = regression_metrics(frame, ['gene'], unit='patient')
    assert result['n'] == 4 and result['mean_pearson'] == pytest.approx(0.)
    frame.loc[frame.project == 'b', 'prediction_0'] = 1.
    result = regression_metrics(frame, ['gene'], unit='patient')
    assert result['defined_correlations'] == 1 and result['undefined_correlations'] == 1
    assert result['per_project']['b']['pearson']['gene'] is None


@pytest.mark.parametrize('mode', ['tiles', 'supertiles'])
def test_regression_train_save_resume_evaluate_and_changed_target_rejection(tmp_path, mode):
    root = create_demo(tmp_path/'data', patients=20, regression=True)
    config = {'epochs': 2, 'threads': 1, 'model': {'layers': [16, 12], 'ks': [1, 2, 4],
              'supertile_count': 4, 'input_mode': mode, 'batch_size': 4}}
    full = train(root, 'he2rna', tmp_path/'full', config=config)
    part = train(root, 'he2rna', tmp_path/'part', config=config, stop_after=1)
    train(root, 'he2rna', part, config=config, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, part]]
    torch.testing.assert_close(a['model'], b['model'], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizers'], b['optimizers'], rtol=0, atol=0)
    assert a['history'] == b['history']
    assert not (full/'test_predictions.csv').exists()
    metrics = evaluate(full)
    assert metrics['n'] == 4 and metrics['defined_correlations'] == 3
    before = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(before, pd.read_csv(full/'test_predictions.csv'))
    data = pd.read_csv(root/'expression.csv'); data.loc[0, 'synthetic_gene_A'] += 1
    data.to_csv(root/'expression.csv', index=False)
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(full)


def test_small_bag_and_wrong_task_are_not_silently_adapted(tmp_path):
    model, bundle = fixture(tmp_path)
    with pytest.raises(ValueError, match='N >= max'):
        model(torch.rand(2, bundle.dim))
    with pytest.raises(ValueError, match='supertiles'):
        HE2RNARegression(bundle.dim, bundle.regression_targets, input_mode='invented')
    with pytest.raises(ValueError, match='does not support'):
        train(bundle.root, 'abmil', tmp_path/'wrong')
    cfg = yaml.safe_load((bundle.root/'task.yaml').read_text())
    cfg['training_unit'] = 'patient'
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(cfg))
    grouped = units(Bundle(bundle.root), 'train')
    assert len(grouped) == 12 and any(len(g) > 1 for g in grouped)
    # No test silently treats signed features as missing or discards their whole rows.
    with h5py.File(bundle.root/bundle.rows.iloc[0].path, 'r') as f:
        assert (f['features'][:] < 0).any()

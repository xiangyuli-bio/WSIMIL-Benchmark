"""Author adaptive graphs/epoch, complete native network and source-state recovery."""
import copy
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.deepgraphsurv import DeepGraphSurv
from milplus.deepgraphsurv_data import prepare
from milplus.deepgraphsurv_demo import create_deepgraphsurv_demo
from milplus.deepgraphsurv_network import NativeAGCN, TF1Adam, source_cox
from milplus.deepgraphsurv_source import graph_functions
from milplus.engine import evaluate, train
from milplus.io import digest_file, read_json, write_json
from milplus.upstream import verified_source


def fixture(tmp_path, dimension=16, maximum=12):
    torch.set_num_threads(1)
    root = create_deepgraphsurv_demo(tmp_path/'data', patients=20, dimension=dimension, maximum=maximum)
    return root, Bundle(root)


def config():
    return {'epochs': 6, 'patience': 100, 'max_instances': None, 'grad_clip': None,
            'selection_metric': 'deepgraphsurv_val_patient_ci', 'model': {'batch_size': 4}}


def test_graph_is_literal_feature_distance_not_spatial_knn(tmp_path):
    _, b = fixture(tmp_path)
    row = b.rows.iloc[0].to_dict()
    x, lap = b.deepgraphsurv.load(row, rebuild=True)
    a = x.numpy()
    distances = np.asarray([np.linalg.norm(a[i]-a[j]) for i in range(len(a)) for j in range(i+1)])
    threshold = np.sort(distances)[int(.3*len(distances))]
    adjacency = (np.linalg.norm(a[:, None]-a[None], axis=-1) < threshold).astype(float)
    np.fill_diagonal(adjacency, 0)
    adjacency += np.eye(len(a))
    degree = adjacency.sum(1)**-.5
    normalized = degree[:, None]*adjacency*degree[None, :]
    expected = np.diag(normalized.sum(0))-normalized
    np.testing.assert_allclose(lap.numpy(), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('dimension', [128, 1024, 1536])
def test_native_architecture_keeps_all_fm_channels_and_gradients(dimension):
    torch.set_num_threads(1)
    torch.manual_seed(514)
    m = NativeAGCN(dimension)
    x = torch.randn(2, 8, dimension, requires_grad=True)
    lap = torch.zeros(2, 8, 8)
    out = m(x, lap, torch.tensor([7, 8]))
    value = source_cox(out['output'], torch.tensor([0., 1.]))
    value.backward()
    assert out['graph_feature'].shape == (2, 128)
    assert out['output'].shape == (2, 1) and out['output'].abs().max() <= 1
    assert m.attention1.metric.shape == (dimension, dimension)
    assert m.backbone1.weight.shape == (3*dimension, 256)
    assert len(m.source_parameter_map()) == 30
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_tied_time_cox_retains_original_ordered_sum():
    risks = torch.tensor([.2, -.1, .4], requires_grad=True)
    events = torch.tensor([1., 0., 1.])
    expected = -sum((risks[i]-torch.log(torch.exp(risks[:i+1]).sum()))*events[i] for i in range(3))
    assert torch.allclose(source_cox(risks, events), expected)
    g = torch.autograd.grad(expected, risks)[0]
    assert torch.allclose(torch.autograd.grad(source_cox(risks, events), risks)[0], g)


def test_inference_dropout_retained_and_native_padding_attention_mass():
    torch.set_num_threads(1)
    m = NativeAGCN(16).eval()
    x, lap, n = torch.randn(2, 10, 16), torch.zeros(2, 10, 10), torch.tensor([4, 9])
    a, b = m(x, lap, n), m(x, lap, n)
    assert not torch.equal(a['output'], b['output'])
    assert a['attention'][0, :, :4].sum() < .5
    assert torch.allclose(a['attention'].sum(2), torch.ones(2, 1))


def test_tf1_adam_uses_global_float32_powers_and_native_epsilon():
    p, q = torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))
    optimizer = TF1Adam([p, q])
    p.grad, q.grad = torch.tensor([1e-8]), None
    optimizer.step()
    # A skipped parameter does not get a separate bias-correction clock.
    assert optimizer.param_groups[0]['updates'] == 1
    assert q not in optimizer.state
    p.grad, q.grad = torch.tensor([1e-8]), torch.tensor([2e-8])
    optimizer.step()
    assert optimizer.param_groups[0]['updates'] == 2
    # Native TF1 scalar state after two updates: float32 multiplication at each
    # update, not NumPy's promoted float64 scalar-power expression.
    np.testing.assert_array_equal(optimizer.param_groups[0]['beta1_power'], np.float32(.7289999))
    assert p.item() > .9999  # eps is outside uncorrected sqrt(v), not torch.Adam.


def test_original_complete_epoch_owns_mixed_windows_and_resume(tmp_path):
    _, b = fixture(tmp_path)
    torch.manual_seed(451)
    m = DeepGraphSurv(b, batch_size=4)
    m.configure_training(config())
    optimizer = m.make_optimizer(.0002, 0.)
    m.epoch(b, optimizer)
    m.finish_epoch(b, 'cpu', config())
    checkpoint = copy.deepcopy({'model': m.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()})
    expected = m.epoch(b, optimizer)
    final = copy.deepcopy(m.state_dict())
    restored = DeepGraphSurv(b, batch_size=4)
    restored.configure_training(config())
    restored_optimizer = restored.make_optimizer(.0002, 0.)
    restored.load_state_dict(checkpoint['model'])
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng'])
    assert restored.epoch(b, restored_optimizer) == expected
    for key, value in final.items():
        other = restored.state_dict()[key]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert all(0 in s['events'] and 1 in s['events'] for s in m.last_trace)
    assert m.optimizer_steps == 6


def test_full_engine_six_epochs_selected_reload_and_repeat(tmp_path):
    root, _ = fixture(tmp_path)
    run = tmp_path/'run'
    train(root, 'DeepGraphSurv', run, config=config())
    selected = torch.load(run/'best.pt', weights_only=False)
    assert selected['epoch'] == 5
    assert len(read_json(run/'history.json')) == 6
    assert all(not r['validation'].get('validation_performed', True) for r in read_json(run/'history.json')[:5])
    metrics = evaluate(run)
    sha = digest_file(run/'test_predictions.csv')
    assert evaluate(run) == metrics and digest_file(run/'test_predictions.csv') == sha
    protocol = read_json(run/'method_protocol.json')
    assert protocol['source_commit'] == '3149752deba68067f1a39b9f3d212cb16ffaf560'


def test_graph_feature_and_manifest_drift_rejected(tmp_path):
    _, b = fixture(tmp_path)
    row = b.rows.iloc[0].to_dict()
    path = b.root/row['path']
    with h5py.File(path, 'r+') as stream:
        stream['features'][0, 0] += 1
    with pytest.raises(ValueError, match='contents changed'):
        b.deepgraphsurv.load(row)


def test_fm_identity_and_clinical_order_bound(tmp_path):
    root, _ = fixture(tmp_path)
    p = root/'task.yaml'
    task = yaml.safe_load(p.read_text())
    task['encoders']['toy']['weights'] = 'changed_encoder'
    p.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='FM identity'):
        Bundle(root)


def test_original_small_graphs_need_explicit_task_filter(tmp_path):
    root, b = fixture(tmp_path)
    task = yaml.safe_load((root/'task.yaml').read_text())
    del task['encoders']['toy']['deepgraphsurv']
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    row = b.rows.iloc[0].to_dict()
    with h5py.File(root/row['path'], 'w') as stream:
        stream['features'] = np.ones((3, 16), np.float32)
    with pytest.raises(ValueError, match='fewer than4'):
        prepare(Bundle(root), read_json(root/'deepgraphsurv_schema.json'), root/'new_graphs')


def test_no_generic_weight_decay_or_short_selection_recipe(tmp_path):
    _, b = fixture(tmp_path)
    m = DeepGraphSurv(b, batch_size=4)
    with pytest.raises(ValueError, match='collected layer'):
        m.make_optimizer(.0002, .01)
    bad = config()
    bad['epochs'] = 5
    with pytest.raises(ValueError, match='at least6epochs'):
        m.configure_training(bad)


def test_patient_averages_native_exp_risk_and_singleton_diagnostic_stays_undefined(tmp_path):
    root, b = fixture(tmp_path)
    row = b.rows[b.rows.subset == 'val'].iloc[0].to_dict()
    extra_slide = row['slide_id']+'_second'
    extra_path = 'features/'+extra_slide+'.h5'
    x, _ = b.load(row)
    with h5py.File(root/extra_path, 'w') as stream:
        stream['features'] = (x+.1).numpy()
    import pandas as pd
    slides = pd.read_csv(root/'slides.csv')
    extra = slides.loc[slides.slide_id == row['slide_id']].iloc[0].to_dict()
    extra['slide_id'] = extra_slide
    pd.concat([slides, pd.DataFrame([extra])], ignore_index=True).to_csv(root/'slides.csv', index=False)
    features = pd.read_csv(root/'features.csv')
    pd.concat([features, pd.DataFrame([{'slide_id': extra_slide, 'path': extra_path}])], ignore_index=True).to_csv(root/'features.csv', index=False)
    task = yaml.safe_load((root/'task.yaml').read_text())
    del task['encoders']['toy']['deepgraphsurv']
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    schema = read_json(root/'deepgraphsurv_schema.json')
    schema['slide_order']['val'].append(extra_slide)
    manifest = prepare(Bundle(root), schema, root/'multi_graphs')
    task['encoders']['toy']['deepgraphsurv'] = {'manifest': str(manifest.relative_to(root))}
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    b = Bundle(root)
    m = DeepGraphSurv(b, batch_size=4)
    captured = {}
    original = m.author.retrieve_feature

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['prognosis'] = result[1].copy()
        return result

    m.author.retrieve_feature = capture
    frame = m.predict_subset(b, 'val', 'cpu', config())
    assert len(frame) == 4
    order = b.deepgraphsurv.order['val']
    positions = [i for i, slide in enumerate(order) if b.deepgraphsurv.rows[slide]['global_patient_id'] == row['global_patient_id']]
    prediction = frame[frame.global_patient_id == row['global_patient_id']].iloc[0]
    assert prediction.risk == float(-captured['prognosis'][positions].mean())
    assert len(prediction.source_slides.split('|')) == 2
    assert frame.attrs['native_metrics']['deepgraphsurv_batch_average_c_index'] is None
    assert frame.attrs['native_metrics']['deepgraphsurv_patient_c_index'] is not None


def test_native_all_observed_windows_do_not_invent_training_updates(tmp_path):
    root, _ = fixture(tmp_path)
    import pandas as pd
    slides = pd.read_csv(root/'slides.csv')
    slides['event'] = 1
    slides.to_csv(root/'slides.csv', index=False)
    m = DeepGraphSurv(Bundle(root), batch_size=4)
    optimizer = m.make_optimizer(.0002, 0.)
    with pytest.raises(ValueError):
        m.epoch(m.bundle, optimizer)
    assert m.optimizer_steps == 0 and not optimizer.state


def test_real_cli_dispatch_and_preparation(tmp_path):
    from milplus.cli import dispatch, parser
    root, b = fixture(tmp_path)
    # Actual CLI parser and dispatcher, with already prepared task. Fresh
    # preparation writes a distinct frozen graph manifest through the CLI.
    p = parser()
    args = p.parse_args(['prepare-deepgraphsurv', '--data', str(root), '--schema',
                        str(root/'deepgraphsurv_schema.json'), '--output', str(tmp_path/'cli_graphs')])
    assert Path(dispatch(args)).is_file()
    assert b.deepgraphsurv is not None


@pytest.mark.parametrize('dimension', [128, 1024, 1536])
def test_original_tensorflow_native_model_loss_all_gradients_bn_and_adam(tmp_path, dimension):
    python = Path(os.environ.get('MILPLUS_TF_PYTHON', '/root/milplus_validation_envs/camil_tf/bin/python'))
    if not python.is_file():
        pytest.skip('Set MILPLUS_TF_PYTHON to the isolated TensorFlow2.15.1 compatibility environment')
    torch.set_num_threads(1)
    root, sha = verified_source('deepgraphsurv')
    rng, source = np.random.RandomState(515), graph_functions()
    x = np.full((2, 12, dimension), -1, np.float32)
    lap = np.full((2, 12, 12), -1, np.float32)
    sizes = np.asarray([10, 12], np.int32)
    for i, n in enumerate(sizes):
        x[i, :n] = rng.randn(n, dimension).astype(np.float32)
        adj, _ = source.get_adjacency(x[i, :n])
        lap[i, :n, :n] = source.compute_laplacian(adj, normalized=False)
    fixture = tmp_path/'input.npz'
    np.savez(fixture, x=x, laplacian=lap, sizes=sizes)
    output = tmp_path/'tf'
    script = Path(__file__).resolve().parents[1]/'scripts/deepgraphsurv_tf_reference.py'
    with (tmp_path/'original_tf.log').open('w') as log:
        subprocess.run([str(python), str(script), '--source', str(root), '--fixture', str(fixture),
                        '--output', str(output)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'})
    report = read_json(output/'reference.json')
    assert report['state'] == 'original_TF1_native_network_all_gradients_and_Adam_step_passed'
    assert report['source_commit'] == sha and report['input_shape'] == [2, 12, dimension]
    arrays = np.load(output/'reference.npz')
    model = NativeAGCN(dimension)
    mapping = model.source_parameter_map()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.from_numpy(arrays['before::'+mapping[name]]))
    inputs = torch.from_numpy(x).requires_grad_(True)
    masks = [torch.from_numpy(arrays[f'mask::backbone_net/sgc_ll_{i}/dropout/GreaterEqual']) for i in [1, 2, 3]]
    prediction = model(inputs, torch.from_numpy(lap), torch.from_numpy(sizes), dropout_masks=masks)
    loss = source_cox(prediction['output'], torch.from_numpy(arrays['events']))
    np.testing.assert_allclose(prediction['output'].detach(), arrays['risk'], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(loss.detach(), arrays['loss'], rtol=2e-5, atol=2e-6)
    loss.backward()
    np.testing.assert_allclose(inputs.grad, arrays['input_gradient'], rtol=2e-4, atol=2e-6)
    errors = {}
    for name, p in model.named_parameters():
        native = arrays['gradient::'+mapping[name]]
        np.testing.assert_allclose(p.grad, native, rtol=2e-4, atol=2e-6)
        errors[name] = float(np.max(np.abs(p.grad.numpy()-native)))
    for i in [1, 2, 3]:
        bn = getattr(model, f'backbone{i}').bn
        prefix = f'backbone_net/sgc_ll_{i}/sgc_ll_{i}/backbone_net/sgc_ll_{i}/sgc_ll_{i}/moments/'
        for actual, name in [(bn.moving_mean, 'Squeeze'), (bn.moving_var, 'Squeeze_1')]:
            np.testing.assert_allclose(actual, arrays['after_forward::'+prefix+name+'/ExponentialMovingAverage'].reshape(-1), rtol=2e-5, atol=2e-6)
    optimizer = TF1Adam(model.parameters())
    for name, p in model.named_parameters():
        p.grad = torch.from_numpy(arrays['gradient::'+mapping[name]]).clone()
    optimizer.step()
    for name, p in model.named_parameters():
        native = mapping[name]
        np.testing.assert_allclose(p.detach(), arrays['after::'+native], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(optimizer.state[p]['m'], arrays['after::'+native+'/Adam'], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(optimizer.state[p]['v'], arrays['after::'+native+'/Adam_1'], rtol=2e-6, atol=1e-7)
    for key in ['beta1_power', 'beta2_power']:
        np.testing.assert_array_equal(optimizer.param_groups[0][key], arrays['after::'+key])
    write_json(tmp_path/'deepgraphsurv_tensorflow_comparison.json',
        {'state': 'passed', 'source_commit': sha, 'dimension': dimension,
         'reference_sha256': digest_file(output/'reference.npz'), 'tensorflow': report['tensorflow'],
         'parameter_gradient_errors': errors, 'trainable_tensors': 30,
         'scope': 'Actual unmodified TF1 author network/Cox, identical TF-generated masks, all gradients/EMA and isolated actual TF1Adam apply_gradients with identical original derivatives. No natural WSI or paper-score claim.'})


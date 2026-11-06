"""Native raw RGB MONAI models, ordinal semantics and complete published training loop."""
import ast
import copy
import gc
import json
import os
import subprocess
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from monai.metrics import CumulativeAverage
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from torch import nn
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train, units
from milplus.io import digest_file, write_json
from milplus.metrics import ordinal_metrics
from milplus.monai_mil import MODES, MONAIMIL, ImageBags, monai_epoch, ordinal_targets
from milplus.raw_demo import create_raw_demo
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)
    yield
    gc.collect()


def fixture(tmp_path, profile='monai_transformer_tutorial', **options):
    root = create_raw_demo(tmp_path/'data', patients=15, classes=3)
    b = Bundle(root)
    cfg = {'tile_size': 32, 'tile_count': 4, 'trans_blocks': 1, 'batch_size': 4, 'eval_chunk': 3, **options}
    return MONAIMIL(b, profile, **cfg), b, cfg


def equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def source(name, path):
    root, sha = verified_source(name)
    return subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)


def reference_model(model):
    namespace = {'__name__': '_independent_monai_reference'}
    code = source('monai_mil', 'monai/networks/nets/milmodel.py')
    exec(compile(code, 'native_milmodel.py', 'exec'), namespace)  # noqa: S102 -- complete pinned upstream source, independent importer
    ref = namespace['MILModel'](model.thresholds, model.mode, pretrained=False, trans_blocks=model.trans_blocks,
                                trans_dropout=model.trans_dropout)
    ref.load_state_dict(model.network.state_dict())
    return ref


@pytest.mark.parametrize('profile', list(MODES))
@pytest.mark.parametrize('training', [False, True])
def test_native_resnet_all_five_heads_outputs_bce_all_gradients(tmp_path, profile, training):
    model, _, _ = fixture(tmp_path, profile)
    ref = reference_model(model)
    model.train(training)
    ref.train(training)
    x = torch.rand(2, 3, 3, 32, 32, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    targets = torch.tensor([[0., 0.], [1., 1.]])
    before = rng_state()
    logits = model(x)['ordinal_logits']
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    after = rng_state()
    restore_rng(before)
    expected = ref(xx)
    native_loss = nn.BCEWithLogitsLoss()(expected, targets)
    native_loss.backward()
    equal(logits, expected)
    equal(loss, native_loss)
    equal(x.grad, xx.grad)
    equal(after, rng_state())
    assert model.network.net.conv1.weight.grad.abs().sum() > 0
    for (name, p), (refname, q) in zip(model.network.named_parameters(), ref.named_parameters()):
        assert name == refname and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            equal(p.grad, q.grad)
    equal(model.network.state_dict(), ref.state_dict())


@pytest.mark.parametrize('profile', ['monai_transformer_tutorial', 'monai_pyramid_tutorial'])
def test_full_native_four_blocks_six_grade_model_and_resnet_gradient(tmp_path, profile):
    b = Bundle(create_raw_demo(tmp_path/'data', patients=30, classes=6))
    model = MONAIMIL(b, profile, tile_size=32, tile_count=2)
    ref = reference_model(model)
    assert model.network.net.layer4[2].conv3.out_channels == 2048
    assert model.network.myfc.out_features == 5
    if model.mode == 'att_trans':
        assert len(model.network.transformer.layers) == 4
        assert model.network.transformer.layers[0].self_attn.embed_dim == 2048
    else:
        assert [len(m.layers) if hasattr(m, 'layers') else len(m[1].layers) for m in model.network.transformer] == [4]*4
        assert model.network.transformer[3].layers[0].self_attn.embed_dim == 2304
    model.train()
    ref.train()
    x = torch.rand(1, 2, 3, 32, 32, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    target = torch.tensor([[1., 1., 1., 0., 0.]])
    a = model(x)['ordinal_logits']
    loss = nn.functional.binary_cross_entropy_with_logits(a, target)
    loss.backward()
    b = ref(xx)
    native_loss = nn.functional.binary_cross_entropy_with_logits(b, target)
    native_loss.backward()
    equal(a, b)
    equal(x.grad, xx.grad)
    for p, q in zip(model.network.parameters(), ref.parameters()):
        if p.grad is not None:
            equal(p.grad, q.grad)


@pytest.mark.parametrize('profile', list(MODES))
def test_eval_streamed_full_pyramid_maps_preserve_multiple_bag_order(tmp_path, profile):
    model, _, _ = fixture(tmp_path, profile, eval_chunk=2)
    ref = reference_model(model).eval()
    model.eval()
    x = torch.rand(2, 7, 3, 32, 32)
    with torch.no_grad():
        actual = model(x, chunked=True)['ordinal_logits']
        # Native forward on the whole bag, independent of our chunk collector.
        expected = ref(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if model.mode == 'att_trans_pyramid':
        assert all(v.shape[-2:] == (1, 1) and len(v) == 14 for v in model.network.extra_outputs.values())
    model.train()
    with pytest.raises(ValueError, match='Chunking TRAIN'):
        model(x, chunked=True)


@pytest.mark.parametrize('training', [True, False])
def test_author_tutorial_image_transform_many_seeds_and_white_count_padding(tmp_path, training):
    model, bundle, _ = fixture(tmp_path, tile_count=12)
    row = bundle.records('train')[0]
    image, _ = bundle.raw_images.load(row)
    # Obtain the actual transform-construction AST from the released main_worker, not our builder.
    code = source('monai_tutorials', 'pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py')
    worker = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'main_worker')
    assignment = next(n for n in worker.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == ('train_transform' if training else 'valid_transform') for t in n.targets))
    node = copy.deepcopy(assignment)
    # Raw pixels have already been decoded; the rest is exactly the source Compose.
    node.value.args[0].elts = node.value.args[0].elts[2:]
    scope = {'np': np, 'Compose': Compose, 'GridPatchd': GridPatchd, 'RandGridPatchd': RandGridPatchd,
             'RandFlipd': RandFlipd, 'RandRotate90d': RandRotate90d, 'ScaleIntensityRanged': ScaleIntensityRanged,
             'SplitDimd': SplitDimd, 'ToTensord': ToTensord, 'LoadImaged': LoadImaged,
             'args': types.SimpleNamespace(tile_size=32, tile_count=12)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'source_transform.py', 'exec'), scope)  # noqa: S102 -- actual author transform expression
    pipeline = scope['train_transform' if training else 'valid_transform']
    for seed in range(12):
        seed_everything(seed)
        before = rng_state()
        expected_seed = int(torch.randint(0, 2**32-1, (1,), dtype=torch.int64).item()) if training else None
        if training:
            pipeline.set_random_state(expected_seed)
        native = pipeline({'image': image, 'label': np.array([0., 0.], dtype=np.float32)})
        expected = torch.stack([r['image'].as_subclass(torch.Tensor) for r in native])
        after = rng_state()
        restore_rng(before)
        actual, info = model.images(row, training=training)
        equal(actual, expected)
        equal(rng_state(), after)
        assert info['padded_white_tiles'] == (3 if training else 0)
        assert len(actual) == (12 if training else 8)
        if training:
            assert actual[-3:].eq(1).all()  # Native count padding, despite spatial pad_mode=None.


def native_epoch():
    code = source('monai_tutorials', 'pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py')
    # Only replace the hard-coded CUDA transfer to execute the original optimization loop on CPU.
    code = code.replace('.cuda(args.rank)', '.to(args.rank)')
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
    scope = {'torch': torch, 'nn': nn, 'CumulativeAverage': CumulativeAverage, 'time': time, 'autocast': torch.autocast}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'source_train_epoch.py', 'exec'), scope)  # noqa: S102 -- complete pinned author training function
    return scope['train_epoch']


@pytest.mark.parametrize('profile', list(MODES))
def test_complete_author_epochs_tail_batches_adamw_cosine_bn_and_rng(tmp_path, profile):
    model, bundle, _ = fixture(tmp_path, profile)
    model.configure_training({'epochs': 2, 'max_instances': None, 'grad_clip': None, 'selection_metric': 'qwk'})
    ref = reference_model(model)
    optimizer = model.make_optimizer(3e-5, 0.)
    params = ref.parameters()
    if model.mode.startswith('att_trans'):
        params = [{'params': list(ref.attention.parameters())+list(ref.myfc.parameters())+list(ref.net.parameters())},
                  {'params': list(ref.transformer.parameters()), 'lr': 6e-6, 'weight_decay': .1}]
    refopt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(refopt, T_max=2, eta_min=0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    args = types.SimpleNamespace(rank=torch.device('cpu'), amp=False, epochs=2)
    epoch = native_epoch()
    for i in range(2):
        model.train()
        model.start_epoch(i)
        before = rng_state()
        monai_epoch(model, bundle, units(bundle, 'train'), optimizer, torch.device('cpu'), {})
        model.finish_epoch(bundle, 'cpu', {})
        after = rng_state()
        restore_rng(before)
        loader = DataLoader(ImageBags(model, bundle.records('train')), batch_size=4, shuffle=True, num_workers=0)
        epoch(ref, loader, refopt, scaler, i, args)
        scheduler.step()
        equal(model.network.state_dict(), ref.state_dict())
        for p, q in zip(model.network.parameters(), ref.parameters()):
            if p.grad is not None:
                equal(p.grad, q.grad)
        equal(optimizer.state_dict(), refopt.state_dict())
        equal(model.scheduler.state_dict(), scheduler.state_dict())
        equal(rng_state(), after)
    assert model.optimizer_steps == 6 and model.training_bags == 18
    assert all(int(s['step']) == 6 for s in optimizer.state.values())
    write_json(tmp_path/'monai_author_training.json', {'profile': profile, 'state': 'passed', 'epochs': 2,
        'optimizer_updates': 6, 'training_bags': 18, 'native_backbone': 'full ResNet50', 'trans_blocks': 1,
        'scope': 'complete source tutorial train_epoch; actual RGB augmentation, BCE, tail batch, AdamW groups, BN, cosine and RNG; CPU synthetic'})


@pytest.mark.parametrize('profile', list(MODES))
def test_exact_resume_optimizer_scheduler_rng_best_reload_and_ordinal_csv(tmp_path, profile):
    data = create_raw_demo(tmp_path/'data', patients=15)
    cfg = {'epochs': 2, 'threads': 1, 'model': {'tile_size': 32, 'tile_count': 2, 'trans_blocks': 1, 'eval_chunk': 3}}
    full = train(data, profile, tmp_path/'full', config=cfg)
    train(data, profile, tmp_path/'resume', config=cfg, stop_after=1)
    resumed = train(data, profile, tmp_path/'resume', config=cfg, resume=True)
    a = torch.load(full/'last.pt', weights_only=False)
    b = torch.load(resumed/'last.pt', weights_only=False)
    for key in ['epoch', 'model', 'optimizers', 'rng', 'history', 'best', 'bad']:
        equal(a[key], b[key])
    assert a['model']['_extra_state']['optimizer_steps'] == 6
    assert not json.loads((full/'status.json').read_text())['test_evaluated']
    metrics = evaluate(full)
    sha = digest_file(full/'test_predictions.csv')
    equal(metrics, evaluate(full))
    assert sha == digest_file(full/'test_predictions.csv')
    frame = pd.read_csv(full/'test_predictions.csv')
    assert all(not c.startswith('p_') for c in frame)
    assert metrics['macro_auc'] is None and metrics['log_loss'] is None
    assert frame.prediction.tolist() == np.rint(frame.ordinal_score).astype(int).tolist()
    assert (frame.raw_tiles == 8).all() and (frame.raw_padded_white_tiles == 0).all()
    assert all(len(x) == 64 for x in frame.raw_locations_sha256)


def test_cumulative_grade_targets_and_nonmonotonic_threshold_prediction():
    from monai.transforms import MapTransform
    code = source('monai_tutorials', 'pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py')
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == 'LabelEncodeIntegerGraded')
    scope = {'MapTransform': MapTransform, 'KeysCollection': object, 'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'native_grade_encoding.py', 'exec'), scope)  # noqa: S102 -- native target encoding
    encode = scope['LabelEncodeIntegerGraded'](num_classes=5)
    expected = np.stack([encode({'label': i})['label'] for i in range(6)])
    equal(ordinal_targets(list(range(6)), 5, 'cpu'), torch.from_numpy(expected))
    equal(ordinal_targets([0, 1, 3, 5], 5, 'cpu'), torch.tensor([[0,0,0,0,0], [1,0,0,0,0], [1,1,1,0,0], [1,1,1,1,1]]).float())
    # The original unconstrained heads need not be monotonic: sum, rather than argmax, defines the grade.
    logits = torch.tensor([[8., -8., 8., -8., 8.], [-8., -8., -8., -8., -8.]])
    score = logits.sigmoid().sum(1)
    frame = pd.DataFrame({'target': [3, 0], 'ordinal_score': score.numpy(), 'global_patient_id': ['a', 'b']})
    metrics = ordinal_metrics(frame, 6)
    assert metrics['accuracy'] == metrics['qwk'] == 1 and metrics['macro_auc'] is None
    with pytest.raises(ValueError):
        ordinal_targets([1.5], 5, 'cpu')


def test_ordinal_patient_cluster_bootstrap_uses_grade_definition():
    from milplus.metrics import paired_bootstrap
    y = np.arange(18) % 3
    a = pd.DataFrame({'slide_id': [f's{i}' for i in range(18)], 'global_patient_id': [f'p{i//2}' for i in range(18)],
                      'target': y, 'subset': 'test', 'ordinal_score': y.astype(float)})
    b = a.copy()
    b.loc[b.index % 5 == 0, 'ordinal_score'] = 1.
    result = paired_bootstrap(a, b, 3, metric='qwk', repetitions=40)
    assert result['difference_left_minus_right'] > 0 and result['valid_replicates'] >= 20
    classifier = b.drop(columns='ordinal_score')
    probability = np.eye(3)[np.rint(b.ordinal_score).astype(int)]
    for i in range(3):
        classifier[f'p_{i}'] = probability[:, i]
    # A native ordinal head and a CE/softmax head must be comparable on the same WSI grades.
    equal(result, paired_bootstrap(a, classifier, 3, metric='qwk', repetitions=40))
    with pytest.raises(ValueError, match='grade metrics'):
        paired_bootstrap(a, classifier, 3, metric='macro_auc', repetitions=40)
    with pytest.raises(ValueError, match='score definition'):
        paired_bootstrap(a, b.drop(columns='ordinal_score'), 3, metric='qwk', repetitions=40)


def test_raw_decoded_npy_and_hdf5_pixels_equivalence(tmp_path):
    a = Bundle(create_raw_demo(tmp_path/'a', patients=15, hdf5=False))
    b = Bundle(create_raw_demo(tmp_path/'b', patients=15, hdf5=True))
    equal(a.raw_images.load(a.records('train')[0])[0], b.raw_images.load(b.records('train')[0])[0])
    assert a.check(True)['features'][0]['raw_image']['shape'] == [3, 103, 107]
    assert b.check(True)['features'][0]['sha256']


@pytest.mark.parametrize('kind', ['link', 'link_chain', 'virtual', 'external_storage'])
def test_raw_hdf5_does_not_hide_unhashed_external_pixels(tmp_path, kind):
    data = create_raw_demo(tmp_path/'data', patients=15, hdf5=True)
    index = pd.read_csv(data/'images.csv')
    path = data/index.iloc[0]['path']
    external = data/'outside.h5'
    with h5py.File(external, 'w') as f:
        f['pixels'] = np.full((3, 103, 107), 90, dtype=np.uint8)
    with h5py.File(path, 'w', libver='latest') as f:
        if kind == 'link':
            f['pixels'] = h5py.ExternalLink(str(external), 'pixels')
        elif kind == 'link_chain':
            f['indirect'] = h5py.ExternalLink(str(external), 'pixels')
            f['pixels'] = h5py.SoftLink('/indirect')
        elif kind == 'virtual':
            layout = h5py.VirtualLayout((3, 103, 107), dtype=np.uint8)
            layout[:] = h5py.VirtualSource(str(external), 'pixels', shape=(3, 103, 107))
            f.create_virtual_dataset('pixels', layout)
        else:
            ds = f.create_dataset('pixels', (3, 103, 107), dtype=np.uint8,
                                  external=[(str(data/'outside.bin'), 0, h5py.h5f.UNLIMITED)])
            ds[:] = 90
    index.loc[0, 'sha256'] = digest_file(path)
    index.to_csv(data/'images.csv', index=False)
    with pytest.raises(ValueError, match='embedded'):
        Bundle(data).check(True)


def test_real_openslide_decoder_on_synthetic_two_level_tiled_tiff(tmp_path):
    import tifffile
    data = create_raw_demo(tmp_path/'data', patients=15)
    rgb = np.random.default_rng(1).integers(0, 200, (128, 128, 3), dtype=np.uint8)
    path = data/'synthetic_pyramid.tiff'
    with tifffile.TiffWriter(path) as out:
        out.write(rgb, photometric='rgb', tile=(32, 32), metadata=None)
        out.write(rgb[::2, ::2], photometric='rgb', tile=(32, 32), subfiletype=1, metadata=None)
    config = yaml.safe_load((data/'task.yaml').read_text())
    c = config['encoders']['raw_resnet']['raw_images']
    c.update(kind='openslide', level=1, downsample=2.)
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    bundle = Bundle(data)
    row = {**bundle.records('train')[0], 'path': str(path)}
    pixels, info = bundle.raw_images.load(row)
    equal(pixels, rgb[::2, ::2].transpose(2, 0, 1))
    assert info['level0_dimensions'] == [128, 128] and info['downsample'] == 2


def test_official_imagenet_v1_weights_are_used_and_trainable(tmp_path):
    weights = Path(os.environ.get('MILPLUS_MONAI_WEIGHTS', '/data/milplus_model_weights/resnet50-0676ba61.pth'))
    assert weights.is_file(), 'Provide official ResNet50 ImageNet1K_V1 weights for this full validation gate'
    data = create_raw_demo(tmp_path/'data', patients=15, initial_weights=weights)
    model = MONAIMIL(Bundle(data), 'monai_attention_tutorial', tile_size=32, tile_count=2)
    expected = torch.load(weights, weights_only=True, map_location='cpu')
    equal(model.network.net.conv1.weight, expected['conv1.weight'])
    model(torch.rand(1, 2, 3, 32, 32))['ordinal_logits'].sum().backward()
    assert model.network.net.conv1.weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in model.network.net.parameters())


@pytest.mark.parametrize('bad', ['float_pixels', 'channels', 'max_bytes', 'fake_features', 'wrong_weight_hash', 'unknown_level', 'coords'])
def test_reject_invalid_raw_contracts_without_fallback(tmp_path, bad):
    data = create_raw_demo(tmp_path/'data', patients=15)
    config = yaml.safe_load((data/'task.yaml').read_text())
    raw = config['encoders']['raw_resnet']['raw_images']
    first = pd.read_csv(data/'images.csv').iloc[0]
    p = data/first['path']
    if bad == 'float_pixels':
        np.save(p, np.zeros((3, 103, 107), dtype=np.float32))
    elif bad == 'channels':
        np.save(p, np.zeros((4, 103, 107), dtype=np.uint8))
    elif bad == 'max_bytes':
        raw['max_image_bytes'] = 100
    elif bad == 'fake_features':
        np.save(p, np.zeros((20, 2048), dtype=np.float32))
    elif bad == 'wrong_weight_hash':
        raw['backbone'].update(initialization='imagenet1k_v1', path=str(p), sha256='0'*64)
    elif bad == 'unknown_level':
        raw['level'] = -1
    else:
        rows = pd.read_csv(data/'images.csv')
        rows['coords_path'] = 'fake.npy'
        rows.to_csv(data/'images.csv', index=False)
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        Bundle(data).check(deep=True)


def test_mutations_are_refused_and_raw_and_frozen_methods_are_not_interchangeable(tmp_path):
    raw = Bundle(create_raw_demo(tmp_path/'raw', patients=15))
    frozen = Bundle(create_demo(tmp_path/'frozen', patients=20))
    with pytest.raises(ValueError):
        check_compatibility(require_runnable('ABMIL'), raw)
    with pytest.raises(ValueError):
        check_compatibility(require_runnable('monai'), frozen)
    first = raw.records('train')[0]
    with pytest.raises(ValueError):
        raw.load(first)
    p = raw.root/first['path']
    pixels = np.load(p)
    pixels[0, 50, 50] ^= 1
    np.save(p, pixels)
    with pytest.raises(ValueError, match='hash'):
        raw.check(True)


def test_prediction_is_label_independent_and_empty_foreground_and_memory_fail_clearly(tmp_path):
    model, bundle, _ = fixture(tmp_path)
    model.eval()
    row = bundle.records('test')[0]
    with torch.no_grad():
        a = model.forward_unit(bundle, [row], 'cpu', {})
        b = model.forward_unit(bundle, [{**row, 'target': 1-row['target']}], 'cpu', {})
    equal(a['ordinal_logits'], b['ordinal_logits'])
    equal(a['raw_input'], b['raw_input'])
    with pytest.raises(ValueError, match='before allocation'):
        MONAIMIL(bundle, max_dense_bytes=1024)
    np.save(bundle.root/row['path'], np.full((3, 103, 107), 255, np.uint8))
    with pytest.raises(ValueError, match='foreground'):
        model.images(row)


def test_raw_demo_cli_inspection_and_incompatible_modalities(tmp_path):
    from milplus.cli import dispatch, parser
    from milplus.registry import inspect_method
    args = parser().parse_args(['demo', '--raw-images', '--patients', '15', '--classes', '3', '--output', str(tmp_path/'data')])
    assert dispatch(args)['created'] == tmp_path/'data'
    b = Bundle(tmp_path/'data')
    spec = inspect_method('monai')
    assert spec['id'] == 'monai_transformer_tutorial' and spec['tasks'] == ['ordinal']
    check_compatibility(spec, b)
    with pytest.raises(ValueError, match='separate'):
        dispatch(parser().parse_args(['demo', '--raw-images', '--survival', '--output', str(tmp_path/'bad')]))


def test_amp_cannot_silently_substitute_cpu_prediction_precision(tmp_path):
    model, bundle, _ = fixture(tmp_path, 'monai_mean_tutorial', amp=True)
    with pytest.raises(ValueError, match='actual CUDA'):
        model.configure_training({'epochs': 2, 'max_instances': None, 'grad_clip': None, 'selection_metric': 'qwk'})
    with pytest.raises(ValueError, match='actual CUDA'):
        model.forward_unit(bundle, [bundle.records('test')[0]], 'cpu', {})
    with pytest.raises(ValueError, match='actual CUDA'):
        model.prediction_fields({'ordinal_logits': torch.zeros(1, 2)}, {'target': 0})

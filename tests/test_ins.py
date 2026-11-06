"""Actual author kernels, weak-only data and complete source-repaired INS runs."""
import ast
import copy
import math
from datetime import timedelta

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import models

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.ins import INSFrozenMIL
from milplus.ins_data import InstanceDataset, TrainPatchStore
from milplus.ins_demo import create_ins_demo
from milplus.ins_network import COMMIT, INSFrozenReference
from milplus.io import digest_file, read_json
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def source(relative, names, namespace, replacements=()):
    root, sha = verified_source('ins')
    assert sha == COMMIT
    code = pinned_text(root, sha, relative)
    for old, new in replacements:
        assert old in code
        code = code.replace(old, new)
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    result = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent_original_INS_'+relative, 'exec'), result)  # noqa: S102 -- exact reviewed pinned Git definitions
    return result


def independent_source():
    backbone = source('resnet.py', ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet18', 'Identity', 'SupConResNet'],
        {'torch': torch, 'nn': nn, 'F': F, 'models': models}, [('print(x.shape)', 'pass')])
    backbone['model_dict'] = {'resnet18': (backbone['resnet18'], 512)}
    heads = source('alexnet.py', ['AlexNet_CIFAR10_attention', 'AlexNet_CIFAR10_dsmil', 'teacher_Attention_head', 'teacher_DSMIL_head'],
        {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'np': np})
    network = source('model.py', ['INS', 'concat_all_gather'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np,
         'teacher_Attention_head': heads['teacher_Attention_head'], 'teacher_DSMIL_head': heads['teacher_DSMIL_head']},
        [('pretrained = True', 'pretrained = False'),
         ('bag_prediction = torch.softmax(bag_prediction, dim=1)', 'bag_prediction = self.bag_classifier(bag_prediction)'),
         ('torch.randperm(batch_size_all).cuda()', 'torch.randperm(batch_size_all).to(x.device)'),
         ("print('val:',args.moco_queue, batch_size)", 'pass')])
    losses = source('utils/utils_loss.py', ['partial_loss', 'SupConLoss'], {'torch': torch, 'nn': nn, 'F': F},
        [('F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()', 'F.one_hot(prot_pred, batchY.shape[1]).float().to(batchY.device).detach()'),
         ('torch.zeros(logits.shape[0], dtype=torch.long).cuda()', 'torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)')])
    return backbone['SupConResNet'], network['INS'], losses


@pytest.mark.parametrize('dimension', [1024, 1536])
def test_native_original_resnet_projection_forward_and_full_bag_gradient(dimension):
    model = INSFrozenReference(dimension, [0]*64+[1]*64)
    encoder, _, _ = independent_source()
    ref = encoder(num_class=2, feat_dim=128, name='resnet18', pretrained=False)
    ref.load_state_dict(model.network.encoder_q.state_dict())
    x = torch.randn(4, dimension, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual = model.instance_logits(x)
    expected, _embedding = ref(other.reshape(4, 32, dimension//32).unsqueeze(1).repeat(1, 3, 1, 1))
    exact(actual, expected)
    F.cross_entropy(actual, torch.tensor([0, 1, 0, 1])).backward()
    F.cross_entropy(expected, torch.tensor([0, 1, 0, 1])).backward()
    exact(x.grad, other.grad)
    for p, q in zip(model.network.encoder_q.parameters(), ref.parameters()):
        exact(p.grad, q.grad)
    assert model.network.encoder_q.encoder.layer4[1].conv2.out_channels == 512
    assert model.network.encoder_q.head[2].out_features == 128
    model.zero_grad()
    result = model(x.detach())
    assert result.shape == (1, 2)
    F.cross_entropy(result, torch.tensor([1])).backward()
    assert model.network.bag_classifier.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.network.model_teacherHead.parameters())
    assert all(p.grad is None for p in model.network.model_teacherHead2.parameters())


def test_actual_original_world1_gloo_queue_proto_confidence_all_gradients_and_sgd(tmp_path):
    assert not torch.distributed.is_initialized()
    torch.distributed.init_process_group('gloo', init_method='file://'+str(tmp_path/'gloo'),
        rank=0, world_size=1, timeout=timedelta(seconds=30))
    try:
        seed_everything(82)
        labels = torch.tensor([0]*32+[1]*32)
        model = INSFrozenReference(32, labels, warmup_epochs=1)
        encoder, original, kernels = independent_source()
        ref = original(model.args, encoder)
        ref.load_state_dict(model.network.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.001)
        ref_optimizer = torch.optim.SGD(ref.parameters(), lr=.01, momentum=.9, weight_decay=.001)
        x = torch.randn(64, 32)
        index = torch.arange(64)
        state = rng_state()
        actual = model.instance_objective(x, x, index, epoch=1)
        after = rng_state()
        restore_rng(state)
        partial = torch.ones(64, 2)
        partial[:32, 1] = 0
        view = x.reshape(64, 32, 1)[:, None].repeat(1, 3, 1, 1)
        output, features, pseudo, _candidates, scores = ref(view, view, partial, model.args)
        # First call has no historical observed keys. Keep original current q/k
        # kernel outputs and explicitly discard only unfilled random slots.
        features, pseudo = features[:128], pseudo[:128]
        target = torch.stack([torch.where(labels == 0, 1., .5), torch.where(labels == 0, 0., .5)], 1)
        cls = kernels['partial_loss'](target)
        cls.confidence_update(scores, index, partial, point=labels)
        mask = pseudo[:64].argmax(1)[:, None].eq(pseudo.argmax(1)[None]).float()
        contrast = kernels['SupConLoss']()(features, mask=mask, batch_size=64)
        expected = cls(output, index)+.5*contrast
        exact(actual['output'], output)
        exact(actual['features'], features)
        exact(actual['prototype_scores'], scores)
        exact(actual['mask'], mask)
        exact(model.confidence, cls.confidence)
        exact(actual['loss'], expected)
        exact(model.network.state_dict(), ref.state_dict())
        exact(rng_state(), after)
        optimizer.zero_grad();actual['loss'].backward()
        ref_optimizer.zero_grad();expected.backward()
        for (name, p), (other_name, q) in zip(model.network.named_parameters(), ref.named_parameters()):
            assert name == other_name
            exact(p.grad, q.grad)
        optimizer.step();ref_optimizer.step()
        exact(model.network.state_dict(), ref.state_dict())
        # Wrapper adds no trainable parameters. Native source state IDs match.
        exact(optimizer.state_dict(), ref_optimizer.state_dict())
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('format', ['h5', 'npy', 'pt'])
def test_real_train_patch_ids_and_bounded_exact_feature_rows(tmp_path, format):
    bundle = Bundle(create_ins_demo(tmp_path/'data', dimension=32))
    if format != 'h5':
        import pandas as pd
        frame = pd.read_csv(bundle.root/'features.csv')
        for i, row in frame.iterrows():
            record = bundle.rows[bundle.rows.slide_id == row.slide_id].to_dict('records')[0]
            x, _ = bundle.load(record)
            p = (bundle.root/row.path).with_suffix('.'+format)
            if format == 'npy':
                np.save(p, x.numpy())
            else:
                torch.save({'features': x}, p)
            frame.loc[i, 'path'] = str(p.relative_to(bundle.root))
        frame.to_csv(bundle.root/'features.csv', index=False)
        bundle = Bundle(bundle.root)
    store = TrainPatchStore(bundle, cache_bytes=100, max_open_files=2)
    expected = torch.cat([bundle.load(row)[0] for row in bundle.records('train')])
    for index in [0, 1, len(expected)//2, len(expected)-1, 0]:
        exact(store.patch(index), expected[index])
        assert store.used_bytes <= 100 and len(store.handles) <= 2
    assert {r['slide_id'] for r in store.rows}.isdisjoint({r['slide_id'] for r in bundle.records('test')})
    for index in [-1, len(expected)]:
        with pytest.raises(ValueError, match='TRAIN'):
            store.patch(index)
    positive = InstanceDataset(store, True)
    assert all(store.labels[int(i)] == 1 for i in positive.indices)
    sampler = DistributedSampler(positive, num_replicas=1, rank=0, seed=0)
    sampler.set_epoch(3)
    expected_order = torch.randperm(len(positive), generator=torch.Generator().manual_seed(3)).tolist()
    assert list(sampler) == expected_order
    store.close()
    assert not store.handles and not store.cache


def test_queue_real_indices_coldstart_wrap_and_negative_confidence():
    model = INSFrozenReference(32, [0]*64+[1]*64, queue_size=128, warmup_epochs=1)
    features = torch.randn(128, 32)
    for epoch, ids in [(0, torch.arange(64, 128)), (1, torch.arange(64)), (1, torch.arange(64, 128))]:
        value = model.instance_objective(features[ids], features[ids], ids, epoch)
        assert torch.isfinite(value['loss'])
    assert int(model.network.queue_ptr) == 64 and model.queue_valid.all()
    exact(model.queue_indices[:64], torch.arange(64, 128))
    exact(model.queue_indices[64:], torch.arange(64))
    exact(model.confidence[:64], torch.tensor([[1., 0.]]).repeat(64, 1))
    assert (model.confidence[64:] != .5).any()
    assert value['contrastive'].requires_grad
    assert not model.network.queue.requires_grad and not model.network.prototypes.requires_grad


def test_complete_unified_training_exact_resume_no_test_and_repeat_prediction(tmp_path):
    data = create_ins_demo(tmp_path/'data', dimension=32)
    cfg = {'epochs': 3, 'threads': 1, 'model': {'warmup_epochs': 1, 'batch_size': 64, 'eval_chunk': 7}}
    complete = train(data, 'INS', tmp_path/'complete', seed=21, config=cfg)
    partial = train(data, 'INS', tmp_path/'partial', seed=21, config=cfg, stop_after=1)
    assert read_json(partial/'status.json')['state'] == 'interrupted'
    train(data, 'INS', partial, seed=21, config=cfg, resume=True)
    a = torch.load(complete/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(partial/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'history']:
        exact(a[key], b[key])
    assert not (complete/'test_predictions.csv').exists()
    best = torch.load(complete/'best.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == max(a['history'][1:], key=lambda r: r['validation']['macro_auc'])['epoch']
    assert all(r['train']['instance_update'] == r['train']['bag_update'] == 1 for r in a['history'])
    metrics = evaluate(complete)
    sha = digest_file(complete/'test_predictions.csv')
    assert metrics['unit'] == 'slide'
    assert evaluate(complete) == metrics and digest_file(complete/'test_predictions.csv') == sha
    assert evaluate(partial) == metrics and digest_file(partial/'test_predictions.csv') == sha
    protocol = read_json(complete/'method_protocol.json')
    assert protocol['completed_epochs'] == 3 and protocol['instance_updates'] > 0 and protocol['bag_updates'] > 0


def test_bounded_prediction_matches_complete_native_mean_and_does_not_mutate_state(tmp_path):
    bundle = Bundle(create_ins_demo(tmp_path/'data', dimension=32))
    model = INSFrozenMIL(bundle, warmup_epochs=1, eval_chunk=7)
    model.eval()
    x, _ = bundle.load(bundle.records('test')[0])
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        actual = model(x)['logits']
        expected = model.core(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    exact(before, model.state_dict())


def test_checkpoint_rejects_heldout_ids_wrong_weak_labels_and_confidence(tmp_path):
    bundle = Bundle(create_ins_demo(tmp_path/'data', dimension=32))
    model = INSFrozenMIL(bundle, warmup_epochs=1)
    for mutation in ['index', 'labels', 'confidence']:
        state = copy.deepcopy(model.state_dict())
        if mutation == 'index':
            state['core.queue_valid'][0] = True
            state['core.queue_indices'][0] = len(model.store)
        elif mutation == 'labels':
            state['core.training_bag_labels'][0] = 1-state['core.training_bag_labels'][0]
        else:
            state['core.confidence'][0] = torch.tensor([.2, .2])
        with pytest.raises(ValueError, match='TRAIN|labels|confidence'):
            model.load_state_dict(state)


def test_native_lr_confidence_schedule_and_protocol(tmp_path):
    bundle = Bundle(create_ins_demo(tmp_path/'data', dimension=32))
    model = INSFrozenMIL(bundle)
    cfg = {'epochs': 150, 'selection_metric': 'macro_auc', 'grad_clip': None, 'max_instances': None}
    model.configure_training(cfg)
    optimizer = model.make_optimizer(.01, .001)
    model.start_epoch(0)
    assert optimizer.param_groups[0]['lr'] == .01
    model.start_epoch(149)
    assert optimizer.param_groups[0]['lr'] == .01
    model.core.finish_epoch(19, 150)
    assert model.core.confidence_momentum == .95+19/150*(.8-.95)
    assert model.minimum_selection_epoch == 20 and model.stop_after_epoch == 150
    assert model.settings['batch_size'] == 64 and model.settings['queue_size'] == 8192
    assert model.settings['projection'] == 128 and model.settings['warmup_epochs'] == 20


@pytest.mark.parametrize('dimension', [16, 33])
def test_feature_layout_rejects_incompatible_dimensions_without_padding(dimension):
    with pytest.raises(ValueError, match='divisible'):
        INSFrozenReference(dimension, [0, 1])


def test_invalid_recipe_and_memory_budget_refused(tmp_path):
    bundle = Bundle(create_ins_demo(tmp_path/'data', dimension=32))
    model = INSFrozenMIL(bundle, max_dense_bytes=1)
    with pytest.raises(ValueError, match='max_dense_bytes'):
        model.prepare(bundle, 7, 'cpu')
    base = {'epochs': 21, 'selection_metric': 'macro_auc', 'grad_clip': None, 'max_instances': None}
    for change in [{'epochs': 20}, {'grad_clip': 5}, {'max_instances': 64}, {'selection_metric': 'accuracy'}]:
        with pytest.raises(ValueError):
            model.configure_training({**base, **change})


def test_source_weak_feature_views_are_identical_without_instance_ground_truth(tmp_path):
    bundle = Bundle(create_ins_demo(tmp_path/'data', dimension=32))
    store = TrainPatchStore(bundle)
    data = InstanceDataset(store, False)
    batch, index = next(iter(DataLoader(data, batch_size=64)))
    labels = store.labels[index]
    assert set(labels.tolist()) <= {0, 1} and batch.shape == (64, 32)
    assert 'instance_label' not in bundle.rows and 'point' not in bundle.rows
    with pytest.raises(ValueError, match='positive-bag'):
        INSFrozenReference(32, labels, warmup_epochs=20).instance_objective(batch, batch, torch.arange(64), 0)


def test_cli_ins_fixture_alias_and_rejects_mixed_input_conditions(tmp_path):
    from milplus.cli import dispatch, parser
    from milplus.registry import resolve_for_bundle

    args = parser().parse_args(['demo', '--ins', '--patients', '20', '--output', str(tmp_path/'cli')])
    result = dispatch(args)
    bundle = Bundle(result['created'])
    assert bundle.dim == 1024 and bundle.task_type == 'classification'
    assert bundle.config['evaluation_unit'] == 'slide' and bundle.config['synthetic']
    assert resolve_for_bundle('INS', bundle)['id'] == 'ins_frozen_source_repaired'
    for extra in [['--survival'], ['--sclwc'], ['--classes', '3']]:
        args = parser().parse_args(['demo', '--ins', '--output', str(tmp_path/'invalid'), *extra])
        with pytest.raises(ValueError, match='binary frozen-feature'):
            dispatch(args)

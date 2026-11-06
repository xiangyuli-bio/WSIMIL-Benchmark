"""Independent compiled-kernel and complete author-training comparisons."""
import copy
import math
import random
from typing import TypeVar

import numpy as np
import pytest
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from milplus.data import Bundle
from milplus.engine import restore_rng, rng_state, seed_everything
from milplus.sparseconvmil import SparseConvMIL
from milplus.sparseconvmil_demo import create_sparseconvmil_demo
from milplus.sparseconvmil_source import backend, definitions
from milplus.upstream import load_definitions


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('channels', [32, 128])
def test_actual_compiled_submanifold_and_duplicate_input_gradients(channels):
    torch.set_num_threads(2); scn, identity = backend()
    assert identity['backend'] == 'original_compiled_sparseconvnet'
    coordinates = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 3, 0], [0, 0, 0]])
    source = torch.arange(15, dtype=torch.float32).reshape(5, 3).div(7).requires_grad_()
    conv = scn.SubmanifoldConvolution(2, 3, channels, 3, True)
    sparse = scn.InputLayer(2, [4, 4], mode=4)([coordinates, source])
    actual = conv(sparse)
    actual.features.square().sum().backward()
    actual_x = source.grad.clone(); actual_w = conv.weight.grad.clone(); actual_b = conv.bias.grad.clone()
    x = source.detach().clone().requires_grad_()
    weight = conv.weight.detach().clone().requires_grad_(); bias = conv.bias.detach().clone().requires_grad_()
    # Independent dense formula evaluated ONLY on the backend's actual active
    # sites. Duplicate source tile embeddings are averaged, not summed/dropped.
    points = actual.get_spatial_locations().tolist()
    values = {tuple(point): x[(coordinates == torch.tensor(point)).all(1)].mean(0) for point in points}
    expected = []
    for px, py, bag in points:
        value = bias
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (px+dx, py+dy, bag)
                if key in values:
                    value = value+values[key]@weight[(dx+1)*3+dy+1, 0]
        expected.append(value)
    expected = torch.stack(expected)
    torch.testing.assert_close(actual.features, expected, rtol=2e-6, atol=2e-6)
    expected.square().sum().backward()
    torch.testing.assert_close(actual_x, x.grad, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(actual_w, weight.grad, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(actual_b, bias.grad, rtol=3e-6, atol=3e-6)
    assert len(points) == 4 and (2, 2, 0) not in values


def test_original_rotation_truncation_and_rng_are_executed():
    source, _ = definitions()
    original = source.SparseConvMIL(nn.Identity(), nn.Identity(), nn.Identity(), 10, True, True, True)
    coordinates = torch.tensor([[0., 0.], [1.7, 0.2], [0.4, 4.8], [2.8, 3.2]])
    random.seed(71); before = random.getstate()
    got = original.data_augment_tiles_locations(coordinates.clone()); after = random.getstate()
    random.setstate(before)
    angle = random.uniform(-180., 180.)
    matrix = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    matrix = matrix@torch.diag(torch.tensor([random.choice([-1., 1.]), random.choice([-1., 1.])]))
    matrix = matrix@torch.diag(torch.tensor([.6*random.random()+.7, .6*random.random()+.7]))
    sizes = coordinates.max(0).values-coordinates.min(0).values
    translation = torch.tensor([random.randint(0, int(size)) for size in sizes])
    expected = ((coordinates-translation).float()@matrix).long()
    expected -= expected.min(0, keepdim=True).values
    assert torch.equal(got, expected) and random.getstate() == after


@pytest.mark.parametrize('profile', ['sparseconvmil_raw_source_c32', 'sparseconvmil_raw_source_c128'])
def test_complete_author_perform_epoch_weights_adam_sampling_rng(tmp_path, profile):
    torch.set_num_threads(2)
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10, patches=16)
    bundle = Bundle(data)
    seed_everything(79)
    adapted = SparseConvMIL(bundle, profile)
    cfg = {'epochs': 2, 'lr': .002, 'weight_decay': 1e-6, 'threads': 2,
           'selection_metric': 'sparseconvmil_completed_epoch', 'max_instances': None, 'grad_clip': None}
    adapted.configure_training(cfg); optimizer = adapted.make_optimizer(.002, 1e-6)
    source_network = copy.deepcopy(adapted.network)
    source_optimizer = torch.optim.Adam(source_network.parameters(), lr=.002, weight_decay=1e-6)
    # Materialize the very same original bytes as author x_y PNG folders.
    # Explicitly bind the source dataset's input order and class IDs to the
    # benchmark manifest; this is data alignment, not a trainer/model rewrite.
    folders, labels, ordered = tmp_path/'png', [], []
    folders.mkdir()
    for row in bundle.records('train'):
        folder = folders/row['slide_id']; folder.mkdir()
        paths, coordinates = [], []
        with bundle.sparseconvmil.open(row) as (pixels, coords):
            for i in range(len(pixels)):
                x, y = map(int, coords[i]); path = folder/f'{x}_{y}.png'
                Image.fromarray(pixels[i]).save(path); paths.append(str(path)); coordinates.append([x, y])
        ordered.append((row, paths, coordinates)); labels.append(row['slide_id']+','+str(row['target']))
    label_file = tmp_path/'labels.csv'; label_file.write_text('\n'.join(labels)+'\n')
    ds = load_definitions('sparseconvmil', 'dataset.py', ['pil_loader', 'Dataset'],
        {'torch': torch, 'os': __import__('os'), 'random': random, 'Image': Image,
         'T_co': TypeVar('T_co', covariant=True), 'transforms': transforms})
    original = ds.Dataset(str(folders), str(label_file), 5)
    original.slides_ids = [row['slide_id'] for row, _, _ in ordered]
    original.tiles_files = [paths for _, paths, _ in ordered]
    original.tiles_locations = [coords for _, _, coords in ordered]
    original.slide_labels = {row['slide_id']: row['target'] for row, _, _ in ordered}
    train = load_definitions('sparseconvmil', 'training.py', ['perform_epoch'],
        {'torch': torch, 'np': np, 'metrics': __import__('sklearn.metrics', fromlist=['balanced_accuracy_score'])},
        replacements=[('data.cuda()', 'data.to(next(mil_model.parameters()).device)'),
                      ('locations.cuda()', 'locations.to(next(mil_model.parameters()).device)'),
                      ('slides_labels.cuda()', 'slides_labels.to(next(mil_model.parameters()).device)')])
    for epoch in range(2):
        initial_rng = rng_state()
        adapted.start_epoch(epoch); values = adapted.epoch(bundle, optimizer, torch.device('cpu'))
        final_rng = rng_state()
        restore_rng(initial_rng)
        loss, _ = train.perform_epoch(source_network,
            DataLoader(original, batch_size=2, shuffle=True, num_workers=0), source_optimizer, nn.CrossEntropyLoss())
        equal(adapted.network.state_dict(), source_network.state_dict())
        equal(optimizer.state_dict(), source_optimizer.state_dict()); equal(final_rng, rng_state())
        assert np.isclose(np.mean([v['loss'] for v in values]), loss)
        assert adapted.completed_epochs == epoch+1 and adapted.optimizer_updates == (epoch+1)*3
        assert len(adapted.samples_history[-1]) == 6
    before = rng_state()
    a = adapted.predict_subset(bundle, 'val', 'cpu', cfg)
    b = adapted.predict_subset(bundle, 'val', 'cpu', cfg)
    assert a.equals(b); equal(before, rng_state())


def test_complete_network_and_source_profile_refusals():
    from milplus.sparseconvmil_source import build_model
    with pytest.raises(ValueError, match='complete end-to-end'):
        build_model(2, 'dense_or_frozen_substitute')
    with pytest.raises(ValueError, match='positive integer'):
        build_model(2, 'sparseconvmil_raw_source_c32', downsample=0)

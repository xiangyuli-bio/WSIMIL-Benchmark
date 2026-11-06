"""Actual full original raw256/K32 trainer versus the complete adapter."""
import contextlib
import copy
import io
import random
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from milplus.center_embeddings import CenterEmbeddingMIL
from milplus.center_embeddings_assets import default_root, transforms
from milplus.center_embeddings_demo import create_center_embeddings_demo
from milplus.center_embeddings_pack import norm_coord
from milplus.center_embeddings_source import build_model, definitions, staircase
from milplus.data import Bundle
from milplus.engine import rng_state, seed_everything
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
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


class PixelSlide:
    """Synthetic OpenSlide boundary backed by actual registered256RGB bytes."""
    def __init__(self, row, level=2):
        self.path, self.level = row['path'], level
        with h5py.File(self.path, 'r') as archive:
            coords = archive['coords'][:]
        self.indices = {norm_coord(center, level): i for i, center in enumerate(coords)}

    def read_region(self, point, level, size):
        assert level == self.level and size == (256, 256)
        with h5py.File(self.path, 'r') as archive:
            return Image.fromarray(archive['pixels'][self.indices[point]])


def original_dataset(bundle, subset):
    cls = load_definitions('center_embeddings', 'research_mil/loaders/datasets.py', ['MILdataset'],
        {'Dataset': Dataset, 'np': np, 'random': random, 'torch': torch}).MILdataset
    result = cls.__new__(cls)
    result.mode, result.level, result.mult, result.size = 0, 2, 1, 256
    result.targets, result.slideIDX, result.grid, result.slidenames = [], [], [], []
    rows = bundle.records(subset); result.slides = [PixelSlide(r) for r in rows]
    for i, row in enumerate(rows):
        with h5py.File(row['path'], 'r') as archive:
            coords = archive['coords'][:].tolist()
        result.grid.extend(coords); result.slideIDX.extend([i]*len(coords))
        result.targets.append(row['target']); result.slidenames.append(row['slide_id'])
    result.transform = transforms(default_root(), training=subset == 'train')
    return result


def original_trainer(bundle, profile, repaired=True):
    _, _, _, selector = definitions()
    replacements = []
    if repaired:
        replacements.append(('loss_center = self.center_loss(bag_feat, labels)',
                             'loss_center = self.center_loss(bag_embed, labels)'))
    if 'soft_assignment' in profile:
        replacements.append(('logits   = self.model(input)[3]\n                output   = self.model.module.pooling.probabilities(logits)',
                             'logits   = self.model(input)[2]\n                output   = self.center_loss.get_assignment(logits)'))
    # All full original epoch/start/VAL/inference bodies are executed, with
    # only the explicitly declared typo and author-documented inference switch.
    cls = load_definitions('center_embeddings', 'research_mil/trainers/globaltaskmil_trainer.py', ['GlobalTaskMIL'],
        {'GlobalTrainer': object, 'torch': torch, 'np': np, 'copy': copy,
         'tqdm': lambda iterator, **kwargs: Progress(iterator),
         'AverageMeter': load_definitions('center_embeddings', 'research_mil/utils/misc.py', ['AverageMeter'], {}).AverageMeter,
         'group_argtopk': selector.group_argtopk, 'calc_err': selector.calc_err}, replacements).GlobalTaskMIL
    result = cls.__new__(cls)
    result.cfg = {'training': {'val_batch_size': 1, 'train_batch_size': 1, 'k': 32}}
    result.device, result.nclass = torch.device('cpu'), 2
    model, result.center_loss = build_model(default_root())
    result.model = torch.nn.DataParallel(model)
    features = list(model.features.parameters())
    new = [p for n, p in model.named_parameters() if not n.startswith('features.')]
    result.optimizer = torch.optim.Adam([{'params': features, 'lr': .0001}, {'params': new, 'lr': .01}], weight_decay=.0001)
    result.optimizerpool = torch.optim.Adam(result.center_loss.parameters(), lr=.0001)
    result.train_dset, result.val_dset = [original_dataset(bundle, split) for split in ['train', 'val']]
    result.train_loader = DataLoader(result.train_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    result.val_loader = DataLoader(result.val_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    result.writer = SimpleNamespace(add_scalar=lambda *args: None)
    return result


class Progress:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return iter(self.iterator)

    def set_description(self, value):
        pass


@pytest.mark.parametrize('profile', ['center_embeddings_bag_classifier_repaired', 'center_embeddings_soft_assignment_repaired'])
def test_actual_original_full_two_epoch_trainer_and_all_weights_moments_bn_rng(tmp_path, profile):
    torch.set_num_threads(1)
    data = create_center_embeddings_demo(tmp_path/'data', patients=6)
    bundle = Bundle(data)
    seed_everything(818)
    port = CenterEmbeddingMIL(bundle, profile, train_batch_size=1)
    cfg = {'epochs': 2, 'lr': .0001, 'weight_decay': .0001, 'threads': 1,
           'max_instances': None, 'grad_clip': None, 'selection_metric': 'center_embeddings_val_balanced_accuracy'}
    port.configure_training(cfg); optimizers = port.make_optimizers(.0001, .0001)
    accuracy = []
    for epoch in range(2):
        port.start_epoch(epoch); port.epoch(bundle, optimizers, 'cpu')
        frame = port.predict_subset(bundle, 'val', 'cpu', cfg)
        accuracy.append(port.validation_metrics(frame)['center_embeddings_val_balanced_accuracy'])
    expected_network = copy.deepcopy(port.network.state_dict())
    expected_centers = copy.deepcopy(port.center_loss.state_dict())
    expected_optimizers = [copy.deepcopy(o.state_dict()) for o in optimizers]
    expected_rng = rng_state()
    # Release large activations/model before the independent full source pass.
    del optimizers, port
    import gc
    gc.collect()
    seed_everything(818)
    original = original_trainer(bundle, profile)
    original_accuracy = []
    with contextlib.redirect_stdout(io.StringIO()):
        for epoch in range(2):
            original._on_train_start(); original._train_epoch(epoch)
            original_accuracy.append(original._valid_epoch(epoch)['acc'])
    equal(original.model.module.state_dict(), expected_network)
    equal(original.center_loss.state_dict(), expected_centers)
    equal(original.optimizer.state_dict(), expected_optimizers[0])
    equal(original.optimizerpool.state_dict(), expected_optimizers[1])
    equal(rng_state(), expected_rng)
    assert accuracy == original_accuracy


def test_original_staircase_is_not_generic_multistep_decay():
    groups = [{'lr': .0001}, {'lr': .01}]
    rates = []
    for epoch in range(20):
        staircase(groups, epoch); rates.append([g['lr'] for g in groups])
    assert rates[0] == rates[8] == [.0001, .01]
    assert rates[9] == [.001*.1, .01*.1]
    assert rates[14] == [.001*.1**2, .01*.1**2]
    assert rates[16] == rates[-1] == [.001*.1**3, .01*.1**3]


def test_student_t_center_distribution_is_original_not_softmax():
    torch.manual_seed(81)
    center = definitions()[2].CenterLoss(2, 512, device='cpu')
    features = torch.randn(4, 512)
    distance = (features[:, None]-center.centers).square().sum(2)
    expected = (1/(1+distance))
    expected = expected/expected.sum(1, keepdim=True)
    assert torch.equal(center.get_assignment(features), expected)


def test_top32_score_order_and_source_tie_semantics_are_retained():
    groups = np.repeat([0, 1], [35, 40]); values = np.zeros(75)
    actual = definitions()[3].group_argtopk(groups, values, 32)
    assert actual == list(range(3, 35))+list(range(43, 75))

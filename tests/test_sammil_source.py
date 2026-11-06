"""Independent author-loop, loss-kernel and real mask/gradient checks."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace
from collections import OrderedDict
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus.engine import seed_everything, rng_state, restore_rng
from milplus.sammil import SAMMIL
from milplus.sammil_source import build_model, definitions, full_step
from milplus.sammil_demo import create_sammil_demo
from milplus.data import Bundle
from milplus.upstream import load_definitions


def tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            tensors_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            tensors_equal(a, b)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    else:
        assert left == right


class Meter:
    """Non-algorithmic author logging binding; no model/loss/sampling behavior."""
    def __init__(self):
        self.avg, self.total, self.count = 0., 0., 0

    def update(self, value, count):
        self.total += float(value)*count; self.count += count; self.avg = self.total/self.count


def test_actual_author_complete_train_loop_all_adam_teacher_gradients_rng(tmp_path):
    torch.set_num_threads(2)
    root = create_sammil_demo(tmp_path/'data', patients=18, dimension=1024, classes=2, patches=32)
    bundle = Bundle(root); seed_everything(76)
    model = SAMMIL(bundle, 'sammil_attn_source_control')
    cfg = dict(epochs=2, lr=.0002, weight_decay=1e-5, patience=20,
               selection_metric='sammil_val_auc', max_instances=None, grad_clip=None)
    model.configure_training(cfg); optimizer = model.make_optimizer(cfg['lr'], cfg['weight_decay'])
    student, teacher = copy.deepcopy(model.student), copy.deepcopy(model.teacher)
    original_optimizer = torch.optim.Adam(student.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(original_optimizer, cfg['epochs'], 0.)
    _, utility = definitions()
    original = load_definitions('sammil', 'engine.py', ['train_loop'],
        dict(torch=torch, np=np, time=time, OrderedDict=OrderedDict, AverageMeter=Meter,
             calculate_consistency_loss=utility.calculate_consistency_loss))
    args = SimpleNamespace(model='sam', baseline='attn', patch_shuffle=False, group_shuffle=False,
        cl_alpha=0., cls_alpha=1., group_alpha=.5, consistency_alpha=1000., num_group=5,
        con_batch_size=2048, loss='ce', clip_grad=0., accumulation_steps=1,
        lr_supi=False, log_iter=1000000, no_log=True, wandb=False)
    state = rng_state()
    for epoch in range(2):
        original.train_loop(args, student, teacher, model._loader('train')[0], original_optimizer,
            torch.device('cpu'), nullcontext, F.cross_entropy, None, original_scheduler, 0, None, epoch)
    expected_rng = rng_state()
    restore_rng(state)
    for epoch in range(2):
        model.start_epoch(epoch); model.epoch(bundle, optimizer, 'cpu')
    tensors_equal(student.state_dict(), model.student.state_dict())
    tensors_equal(teacher.state_dict(), model.teacher.state_dict())
    tensors_equal(original_optimizer.state_dict(), optimizer.state_dict())
    tensors_equal(original_scheduler.state_dict(), model._scheduler.state_dict())
    for original_parameter, parameter in zip(teacher.parameters(), model.teacher.parameters()):
        assert (original_parameter.grad is None) == (parameter.grad is None)
        if original_parameter.grad is not None:
            assert torch.equal(original_parameter.grad, parameter.grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in teacher.parameters())
    tensors_equal(expected_rng, rng_state())


@pytest.mark.parametrize('classes', [2, 3, 7])
def test_native_source_network_width_and_all_prediction_gradients(classes):
    seed_everything(76); torch.set_num_threads(2)
    model = build_model(1536, classes, repaired=True)
    source, _ = definitions(True)
    oracle = source.sam_mil(input_dim=1536, mlp_dim=512, n_classes=classes,
        baseline='attn', dropout=.25, da_act='relu', temp_t=.1)
    oracle.load_state_dict(model.state_dict())
    model.eval(); oracle.eval()
    x = torch.randn(1, 67, 1536); x2 = x.clone()
    actual = model(x)[0]; expected = oracle(x2)[0]
    assert torch.equal(actual, expected)
    actual.square().sum().backward(); expected.square().sum().backward()
    for p, q in zip(model.parameters(), oracle.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)
    assert model.patch_to_emb[0].out_features == 512
    assert model.online_encoder.attn_model.attention.D == 128


def test_original_consistency_kernel_matches_independent_per_instance_distances_and_gradient():
    _, utility = definitions()
    x = torch.tensor([.07, .12, .2, .27, .34], dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(); areas = torch.tensor([[7, 7, 7, 0, 0]])
    actual = utility.calculate_consistency_loss(x, areas, 2)
    # The author initializes a default-float32 accumulator and adds in-place,
    # including when the attention tensor is float64. Preserve that observable
    # casting behavior rather than asserting an invented float64 kernel.
    total, count = torch.zeros(1, dtype=torch.float32), 0
    for values in [y[3:], y[:3]]:  # Sorted areas: sentinel0, then area7.
        for start in range(0, len(values), 2):
            # An independently constructed distance vector from each instance
            # to every member of its area group. Source dim=1 reduces each
            # ROW of [query_chunk,all_group_members], not each column.
            distances = [torch.linalg.vector_norm(value-values)
                         for value in values[start:min(start+2, len(values))]]
            total.add_(sum(distances)); count += len(distances)
    expected = (total/count).squeeze()
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual, x)[0]; b = torch.autograd.grad(expected, y)[0]
    assert torch.allclose(a, b, atol=1e-14, rtol=1e-14)


def test_repaired_mask_preserves_actual_group_ids_and_compact_mapping():
    model = build_model(1024, 3, True)
    flags = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    areas = torch.tensor([[0, 7000, 7000, 0, 11000, 11000, 11000, 15000, 15000, 15000, 15000, 15000]])
    attention = torch.linspace(.01, .2, 12).unsqueeze(0)
    seed_everything(76)
    kept, ids = model.get_mask(12, 0, attention, is_group_feat=flags, relative_area=areas)
    assert ids.shape == (1, 12) and len(ids.unique()) == 12
    assert {0, 3} <= set(ids[0, :kept].tolist())
    assert torch.equal(ids[0, :2], torch.tensor([0, 3]))
    # Original batched-index error is executable and explicitly stays in the
    # source-control route. This assertion is not a speculative static audit.
    original = build_model(1024, 2, False)
    seed_everything(76)
    with pytest.raises((IndexError, RuntimeError)):
        original.get_mask(12, 0, attention, is_group_feat=flags, relative_area=areas)


@pytest.mark.parametrize('repaired', [False, True])
def test_complete_objective_exposes_global_group_and_consistency_gradient_paths(repaired):
    seed_everything(76); torch.set_num_threads(2)
    student = build_model(1024, 2, repaired); teacher = copy.deepcopy(student)
    x = torch.randn(1, 33, 1024); flags = torch.tensor([[0]*30+[1]*3])
    areas = torch.tensor([[7000]*10+[11000]*10+[15000]*10+[0]*3])
    result = full_step(student, teacher, x, torch.tensor([1]), flags, areas, repaired=repaired)
    assert result['group_sizes'] == [6, 6, 6, 6, 9]
    assert result['consistency'] > 0
    assert torch.equal(result['loss'], result['global_ce']+.5*result['group_ce']+1000*result['consistency'])
    consistency_student = torch.autograd.grad(result['consistency'], tuple(student.parameters()),
        retain_graph=True, allow_unused=True)
    consistency_teacher = torch.autograd.grad(result['consistency'], tuple(teacher.parameters()),
        retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in consistency_student) == repaired
    assert any(g is not None and g.abs().sum() > 0 for g in consistency_teacher) != repaired
    expected_main = F.cross_entropy(result['global_logits'] if repaired else result['group_logits'][-1], torch.tensor([1]))
    assert torch.equal(result['global_ce'], expected_main)


def test_released_binary_route_rejects_silent_multiclass_substitution():
    with pytest.raises(ValueError, match='binary'):
        build_model(1024, 3, repaired=False)

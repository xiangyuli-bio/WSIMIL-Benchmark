"""Actual full original architecture control checks, explicitly unpretrained.

These checks cannot establish successful FEATHER pretrained loading. They
verify the independently specified downstream recipe and exact recovery.
"""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus.feather_source import architecture_classes, original_configuration
from milplus.feather_training import (
    class_weights, fit, knn_probabilities, new_progress, reference_recipe, setup)


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def case(encoder):
    torch.set_num_threads(2)
    torch.manual_seed(903)
    model = architecture_classes().ABMIL(**original_configuration(encoder, 2))
    dim = original_configuration(encoder, 2)['in_dim']
    generator = torch.Generator().manual_seed(24)
    features = {str(i): torch.randn(257 + i, dim, generator=generator) for i in range(9)}
    train = [{'slide_id': str(i), 'target': int(i == 4)} for i in range(5)]
    val = [{'slide_id': str(i), 'target': i % 2} for i in range(5, 9)]
    return model, train, val, lambda row: features[row['slide_id']]


def snapshot(model, optimizer, scheduler, scaler, generator):
    return deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'optimizer_step_called': bool(getattr(optimizer, '_opt_called', False)),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'sampler': generator.get_state(), 'rng': torch.get_rng_state(),
        'gradients': {n: None if p.grad is None else p.grad.detach().clone()
                      for n, p in model.named_parameters()}})


@pytest.mark.parametrize('encoder', ['conch_v15', 'uni', 'uni_v2'])
def test_full_native_graph_no_validation10epoch_recipe_matches_independent_literal_loop(encoder):
    model, rows, val, load = case(encoder)
    initial = deepcopy(model.state_dict())
    initial_rng = torch.get_rng_state()
    recipe = reference_recipe('finetune_no_val10')
    optimizer, scheduler, scaler, generator = setup(model, device='cpu', seed=2026, recipe=recipe)
    batch_proofs = []
    state = new_progress()
    def batch_callback(progress, row, loss):
        batch_proofs.append((row['slide_id'], loss, snapshot(model, optimizer, scheduler, scaler, generator)))
    result = fit(model, optimizer, scheduler, scaler, generator, rows, [], load, state,
        device='cpu', classes=2, recipe=recipe, on_checkpoint=lambda *a: None, on_batch=batch_callback)
    final = snapshot(model, optimizer, scheduler, scaler, generator)
    assert result == 'completed' and state['epochs'] == 10 and state['train_batches'] == 50
    assert state['best_epoch'] is None and all(v['validation_loss'] is None for v in state['history'])
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    # Independent explicit paper-constant/reference-policy implementation. No
    # production helper controls this loop, weighting, update or schedule.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5,
        betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generator = torch.Generator().manual_seed(2026)
    weights = torch.tensor([.25, .25, .25, .25, 1.], dtype=torch.float64)
    cursor = 0
    for epoch in range(10):
        indices = torch.multinomial(weights, 5, replacement=True, generator=generator).tolist()
        assert state['history'][epoch]['train_draws'] == [rows[i]['slide_id'] for i in indices]
        for i in indices:
            row = rows[i]
            model.train()
            x = F.dropout(load(row), p=.1, training=True)
            optimizer.zero_grad(set_to_none=True)
            output, unused = model(x.unsqueeze(0))
            loss = F.cross_entropy(output['logits'], torch.tensor([row['target']]))
            loss.backward()
            optimizer.step()
            slide, expected_loss, expected = batch_proofs[cursor]
            assert row['slide_id'] == slide and float(loss.detach()) == expected_loss
            exact(snapshot(model, optimizer, scheduler, scaler, generator), expected)
            cursor += 1
        model.zero_grad(set_to_none=True)
        scheduler.step()
    exact(snapshot(model, optimizer, scheduler, scaler, generator), final)


@pytest.mark.parametrize('interrupt', [1, 4, 5, 6, 49, 50])
def test_native_graph10epoch_recovery_preserves_actual_weighted_order_all_states_and_rng(interrupt):
    model, rows, val, load = case('uni')
    initial = deepcopy(model.state_dict())
    initial_rng = torch.get_rng_state()
    recipe = reference_recipe('finetune_no_val10')
    optimizer, scheduler, scaler, generator = setup(model, device='cpu', seed=101, recipe=recipe)
    state = new_progress()
    fit(model, optimizer, scheduler, scaler, generator, rows, [], load, state,
        device='cpu', classes=2, recipe=recipe, on_checkpoint=lambda *a: None)
    final_state = deepcopy(state)
    final = snapshot(model, optimizer, scheduler, scaler, generator)
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    optimizer, scheduler, scaler, generator = setup(model, device='cpu', seed=101, recipe=recipe)
    state = new_progress()
    saved = []
    def save(progress, best):
        saved.append((deepcopy(progress), snapshot(model, optimizer, scheduler, scaler, generator)))
    assert fit(model, optimizer, scheduler, scaler, generator, rows, [], load, state,
        device='cpu', classes=2, recipe=recipe, on_checkpoint=save, stop_after_batch=interrupt) == 'interrupted'
    progress, checkpoint = saved[-1]
    # Reconstruct the complete actual author graph and all mutable state.
    model, unused_rows, unused_val, unused_load = case('uni')
    optimizer, scheduler, scaler, generator = setup(model, device='cpu', seed=101, recipe=recipe)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer._opt_called = checkpoint['optimizer_step_called']
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    generator.set_state(checkpoint['sampler'])
    for n, p in model.named_parameters():
        p.grad = None if checkpoint['gradients'][n] is None else checkpoint['gradients'][n].clone()
    torch.set_rng_state(checkpoint['rng'])
    assert fit(model, optimizer, scheduler, scaler, generator, rows, [], load, progress,
        device='cpu', classes=2, recipe=recipe, on_checkpoint=lambda *a: None) == 'completed'
    exact(progress, final_state)
    exact(snapshot(model, optimizer, scheduler, scaler, generator), final)


def test_reference_policies_record_undisclosed_choices_and_never_train_on_val():
    val = reference_recipe('finetune_val20')
    assert (val['epochs'], val['min_epochs'], val['patience']) == (20, 10, 5)
    assert val['unspecified_details_are_reference_choices'] is True
    assert val['source_patch_projection_dropout'] == 0.
    model, rows, val_rows, load = case('uni')
    recipe = reference_recipe('finetune_no_val10')
    args = setup(model, device='cpu', seed=31, recipe=recipe)
    with pytest.raises(ValueError, match='held-out VAL'):
        fit(model, *args, rows, val_rows, load, new_progress(), device='cpu', classes=2,
            recipe=recipe, on_checkpoint=lambda *a: None)
    with pytest.raises(ValueError, match='every TRAIN class'):
        class_weights([{'target': 0}], 2)


def test_knn20_matches_independent_exact_distances_votes_and_ties():
    rng = np.random.default_rng(31)
    train = rng.normal(size=(31, 512)).astype('float32')
    train[1] = train[0]
    labels = np.asarray([i % 3 for i in range(31)], dtype=np.int64)
    queries = np.stack([train[0], train[5], np.zeros(512, dtype='float32')])
    probabilities, neighbors = knn_probabilities(train, labels, queries, 3)
    for i, q in enumerate(queries):
        ordered = sorted(range(31), key=lambda j: (sum((float(x) - float(y)) ** 2
            for x, y in zip(train[j], q)), j))[:20]
        assert neighbors[i] == ordered
        expected = np.asarray([sum(labels[j] == k for j in ordered) / 20 for k in range(3)])
        assert np.array_equal(probabilities[i], expected)
    assert neighbors[0][:2] == [0, 1]
    with pytest.raises(ValueError, match='at least20'):
        knn_probabilities(train[:19], labels[:19], queries, 3)
    with pytest.raises(ValueError, match='at least20'):
        knn_probabilities(train, labels, queries, 3, k=5)

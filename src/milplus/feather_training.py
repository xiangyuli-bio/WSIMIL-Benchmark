"""Explicit FEATHER downstream reference control, not a recovered author trainer.

The source release contains the model and checkpoint loader, not its trainer.
Paper-specified constants are retained. Unspecified sampling, AMP, scheduling,
selection and KNN choices are named reference choices in every run request.
"""
import copy
import math

import numpy as np
import torch
from torch.nn import functional as F


def reference_recipe(mode='finetune_val20'):
    if mode not in {'finetune_val20', 'finetune_no_val10', 'frozen_knn20'}:
        raise ValueError('Choose a complete FEATHER finetuning or frozen-KNN reference')
    return {'mode': mode, 'epochs': 10 if mode == 'finetune_no_val10' else 20,
        'min_epochs': 10, 'patience': 5, 'optimizer': 'AdamW', 'lr': 1e-4,
        'weight_decay': 1e-5, 'betas': [.9, .999], 'eps': 1e-8,
        'batch_size': 1, 'feature_dropout': .1, 'source_gated_branch_dropout': .25,
        'source_patch_projection_dropout': 0., 'scheduler': 'CosineAnnealingLR',
        'scheduler_step': 'once_after_each_complete_epoch', 'eta_min': 0.,
        'sampling': 'inverse_TRAIN_slide_class_frequency_with_replacement',
        'samples_per_epoch': 'number_of_TRAIN_slides',
        'sampler_rng': 'separate_cpu_torch_generator_seeded_with_run_seed',
        'validation_selection': 'strictly_lower_unweighted_VAL_cross_entropy',
        'patience_semantics': 'all_nonimproving_epochs_count_stop_only_after_min10',
        'precision': 'cuda_float16_GradScaler_or_explicit_cpu_float32_reference',
        'knn': {'k': 20, 'distance': 'float64_squared_euclidean', 'weights': 'uniform',
            'normalization': 'none', 'feature': 'original_512D_preclassifier_slide_embedding',
            'tie_neighbors': 'stable_TRAIN_slide_id_order', 'tie_class': 'lowest_common_class_index'},
        'author_full_trainer_available': False, 'unspecified_details_are_reference_choices': True,
        'historical_paper_score_reproduction': False}


def new_progress():
    return {'epochs': 0, 'train_batches': 0, 'optimizer_updates': 0,
        'cursor': 0, 'order': None, 'epoch_loss_sum': 0., 'epoch_batches': 0,
        'best_epoch': None, 'best_validation_loss': None, 'bad_epochs': 0,
        'history': [], 'completed': False}


def class_weights(train_rows, classes):
    counts = torch.bincount(torch.tensor([int(r['target']) for r in train_rows]), minlength=classes)
    if len(train_rows) == 0 or len(counts) != classes or bool((counts == 0).any()):
        raise ValueError('FEATHER random class-weighted sampling needs every TRAIN class')
    return torch.tensor([1. / int(counts[int(r['target'])]) for r in train_rows], dtype=torch.float64)


def setup(model, *, device, seed, recipe):
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'],
        weight_decay=recipe['weight_decay'], betas=tuple(recipe['betas']), eps=recipe['eps'],
        foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe['epochs'], eta_min=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=str(device).startswith('cuda:'))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return optimizer, scheduler, scaler, generator


@torch.no_grad()
def validation_loss(model, rows, load, device):
    model.eval()
    total = 0.
    for row in rows:
        x = load(row).to(device)
        result, unused = model(x.unsqueeze(0))
        # The validation selection reference deliberately computes CE in FP32.
        loss = F.cross_entropy(result['logits'].float(), torch.tensor([int(row['target'])], device=device))
        if not torch.isfinite(loss):
            raise FloatingPointError('FEATHER nonfinite validation loss')
        total += float(loss)
    if not rows:
        raise ValueError('The VAL20 reference cannot select on an empty validation set')
    return total / len(rows)


def fit(model, optimizer, scheduler, scaler, generator, train_rows, val_rows, load,
        state, *, device, classes, recipe, on_checkpoint, stop_after_batch=None,
        on_batch=None):
    if recipe['mode'] not in {'finetune_val20', 'finetune_no_val10'}:
        raise ValueError('Frozen FEATHER KNN never enters optimizer training')
    if recipe['mode'] == 'finetune_no_val10' and val_rows:
        raise ValueError('No-validation FEATHER must not consume held-out VAL labels')
    weights = class_weights(train_rows, classes)
    while state['epochs'] < recipe['epochs'] and not state['completed']:
        if state['order'] is None:
            state['order'] = list(torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(train_rows), replacement=True, generator=generator))
        if len(state['order']) != len(train_rows) or not 0 <= state['cursor'] <= len(train_rows):
            raise ValueError('FEATHER recovery changed the actual class-weighted TRAIN draw plan')
        while state['cursor'] < len(state['order']):
            model.train()
            row = train_rows[state['order'][state['cursor']]]
            x = F.dropout(load(row).to(device), p=recipe['feature_dropout'], training=True)
            target = torch.tensor([int(row['target'])], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=str(device).startswith('cuda:')):
                result, unused = model(x.unsqueeze(0))
                loss = F.cross_entropy(result['logits'], target)
            if not torch.isfinite(loss):
                raise FloatingPointError('FEATHER nonfinite actual minibatch loss')
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= scale_before
            state['optimizer_updates'] += int(updated)
            state['train_batches'] += 1
            state['epoch_batches'] += 1
            state['epoch_loss_sum'] += float(loss.detach())
            state['cursor'] += 1
            if on_batch is not None:
                on_batch(state, row, float(loss.detach()))
            if stop_after_batch is not None and state['train_batches'] == stop_after_batch:
                on_checkpoint(state, False)
                return 'interrupted'
        model.zero_grad(set_to_none=True)
        old_lr = optimizer.param_groups[0]['lr']
        selected = recipe['mode'] == 'finetune_val20'
        val = validation_loss(model, val_rows, load, device) if selected else None
        state['epochs'] += 1
        improved = selected and (state['best_validation_loss'] is None or val < state['best_validation_loss'])
        if improved:
            state['best_validation_loss'] = val
            state['best_epoch'] = state['epochs']
            state['bad_epochs'] = 0
        elif selected:
            state['bad_epochs'] += 1
        scheduler.step()
        state['history'].append({'epoch': state['epochs'], 'lr_used': old_lr,
            'next_lr': optimizer.param_groups[0]['lr'],
            'train_loss': state['epoch_loss_sum'] / state['epoch_batches'],
            'train_draws': [train_rows[i]['slide_id'] for i in state['order']],
            'validation_loss': val, 'improved': improved, 'bad_epochs': state['bad_epochs']})
        state['order'] = None
        state['cursor'] = 0
        state['epoch_batches'] = 0
        state['epoch_loss_sum'] = 0.
        state['completed'] = (state['epochs'] == recipe['epochs'] or
            (selected and state['epochs'] >= recipe['min_epochs'] and state['bad_epochs'] >= recipe['patience']))
        on_checkpoint(state, improved)
    return 'completed'


def knn_probabilities(train_features, train_targets, query_features, classes, *, k=20):
    """Reference uniform KNN; no validation/test samples enter its reference bank."""
    train = np.asarray(train_features, dtype=np.float64)
    queries = np.asarray(query_features, dtype=np.float64)
    labels = np.asarray(train_targets)
    if (k != 20 or train.ndim != 2 or queries.ndim != 2 or train.shape[1] != 512
            or queries.shape[1] != 512 or len(train) < k
            or labels.shape != (len(train),) or not np.issubdtype(labels.dtype, np.integer)
            or not np.isfinite(train).all() or not np.isfinite(queries).all()
            or not isinstance(classes, int) or classes < 2 or (labels < 0).any() or (labels >= classes).any()):
        raise ValueError('FEATHER KNN20 needs actual finite512D embeddings and at least20 TRAIN slides')
    probabilities = []
    neighbors = []
    for query in queries:
        squared = np.sum((train - query) ** 2, axis=1)
        if not np.isfinite(squared).all():
            raise FloatingPointError('FEATHER KNN reference distance overflow')
        indices = np.argsort(squared, kind='stable')[:k]
        neighbors.append(indices.tolist())
        probabilities.append(np.bincount(labels[indices], minlength=classes) / k)
    return np.asarray(probabilities, dtype=np.float64).reshape(-1, classes), neighbors

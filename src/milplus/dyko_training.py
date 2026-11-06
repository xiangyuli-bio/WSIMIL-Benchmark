"""Explicit Lightning2.3-style automatic-optimization control for DyKo.

The model and CE+KL remain the original source. This module exposes sampling,
partial accumulated gradients and validation stopping so a fresh process can
recover the full computation. Control-loop tests do not validate TITAN weights.
"""
import copy
import math

import torch
from torch.utils.data import DataLoader

from .dyko_source import classification_loss


def original_loader_order(length, shuffle):
    """Actual PyTorch loader base-seed and sampler-seed consumption.

    Source items are preloaded frozen tensors, with no stochastic getitem.
    Single-process index materialization preserves the main-process RNG stream;
    worker parallelism itself is not claimed to be reproduced.
    """
    if type(length) is not int or length < 1:
        raise ValueError('DyKo source loader population cannot be empty')
    loader = DataLoader(list(range(length)),batch_size=1,shuffle=shuffle,num_workers=0)
    return [int(index.item()) for index in loader]


def new_progress():
    return {'epochs':0,'train_batches':0,'optimizer_updates':0,'order':None,'position':0,
            'epoch_train':[],'history':[],'best_accuracy':None,'best_epoch':None,'bad_epochs':0,
            'completed':False,'last_validation':None}


def accumulated_gradients(model):
    return {name:None if parameter.grad is None else parameter.grad.detach().clone()
            for name,parameter in model.named_parameters()}


def restore_gradients(model, values):
    parameters = dict(model.named_parameters())
    if set(parameters) != set(values):
        raise ValueError('Full DyKo accumulated-gradient parameter identity changed')
    for name,parameter in parameters.items():
        value = values[name]
        if value is not None and (value.shape != parameter.shape or value.dtype != parameter.dtype):
            raise ValueError('Full DyKo accumulated-gradient dimensions or dtype changed')
        parameter.grad = None if value is None else value.to(parameter.device).clone()


def fit_control(model, optimizer, train_rows, val_rows, forward, state, *, epochs=1000,
                patience=40, loss_function=classification_loss, on_checkpoint=None,
                stop_after_epoch=None, stop_after_batch=None, on_train_batch=None):
    """Fixed accumulation4, including a short final group normalized by4.

    A step retains its gradient buffers until the next group's zero_grad,
    matching Lightning automatic optimization. No scheduler, clipping, AMP,
    backbone freezing or altered loss is introduced.
    """
    if (type(epochs) is not int or epochs < 1 or type(patience) is not int or patience < 1
            or not train_rows or not val_rows):
        raise ValueError('DyKo needs nonempty actual TRAIN/VAL data and positive stopping budgets')
    if state['completed']:
        raise ValueError('DyKo optimization is already complete')
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 0 < stop_after_epoch < epochs):
        raise ValueError('Requested DyKo interruption must be inside its declared max epochs')
    if stop_after_batch is not None and (type(stop_after_batch) is not int or stop_after_batch <= state['train_batches']):
        raise ValueError('Requested DyKo batch interruption must follow actual recovered progress')
    while state['epochs'] < epochs:
        model.train()
        if state['order'] is None:
            state['order'] = original_loader_order(len(train_rows),True)
            state['position'] = 0;state['epoch_train'] = []
        order = state['order']
        if sorted(order) != list(range(len(train_rows))) or not 0 <= state['position'] <= len(order):
            raise ValueError('Actual DyKo source sampling order or recovery position is invalid')
        for position in range(state['position'],len(order)):
            row = train_rows[order[position]]
            # Lightning evaluates training_step before its group zero_grad.
            results,label = forward(row,True)
            loss = loss_function(results,label)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Original DyKo CE+KL objective became nonfinite')
            if position % 4 == 0:
                optimizer.zero_grad()
            (loss/4).backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('Actual full DyKo gradient became nonfinite')
            boundary = (position+1) % 4 == 0 or position+1 == len(order)
            if boundary:
                optimizer.step();state['optimizer_updates'] += 1
            state['train_batches'] += 1;state['position'] = position+1
            state['epoch_train'].append({'loss':float(loss.detach()),
                'correct':int(results['Y_hat'].reshape(-1).item() == label.reshape(-1).item())})
            if on_train_batch:
                on_train_batch(state)
            if stop_after_batch is not None and state['train_batches'] == stop_after_batch:
                if on_checkpoint:
                    on_checkpoint(state,False)
                return 'interrupted'
        # LightningModule.on_validation_model_zero_grad releases the final
        # accumulated buffers before entering evaluation. Preserve this actual
        # boundary in epoch checkpoints as well as in the next training epoch.
        model.zero_grad()
        model.eval();predictions = []
        # The original validation DataLoader consumes a new base seed. This
        # consumption remains part of the subsequent model/dropout RNG state.
        val_order = original_loader_order(len(val_rows),False)
        with torch.no_grad():
            for index in val_order:
                row = val_rows[index];results,label = forward(row,False)
                probability = results['Y_prob'].detach().cpu()
                if not bool(torch.isfinite(probability).all()):
                    raise FloatingPointError('Actual full DyKo validation predictions became nonfinite')
                predictions.append({'row_index':index,'source_target':int(label.item()),
                    'source_prediction':int(results['Y_hat'].reshape(-1).item()),'probabilities':probability.tolist()})
        accuracy = sum(r['source_prediction'] == r['source_target'] for r in predictions)/len(predictions)
        if not math.isfinite(accuracy):
            raise FloatingPointError('Original standard validation accuracy is nonfinite')
        # Source self.log converts this Python accuracy to its FP32 monitor.
        # Keep the raw standard score and the actual monitor value separately.
        monitor = float(torch.tensor(accuracy,dtype=torch.float32))
        improved = state['best_accuracy'] is None or monitor > state['best_accuracy']
        state['epochs'] += 1
        if improved:
            state['best_accuracy'] = monitor;state['best_epoch'] = state['epochs'];state['bad_epochs'] = 0
        else:
            state['bad_epochs'] += 1
        state['history'].append({'epoch':state['epochs'],
            'train_loss':sum(r['loss'] for r in state['epoch_train'])/len(state['epoch_train']),
            'train_accuracy':sum(r['correct'] for r in state['epoch_train'])/len(state['epoch_train']),
            'val_standard_accuracy':accuracy,'val_acc_monitor':monitor,'optimizer_updates':state['optimizer_updates']})
        state['last_validation'] = predictions;state['order'] = None;state['position'] = 0;state['epoch_train'] = []
        state['completed'] = state['epochs'] == epochs or state['bad_epochs'] >= patience
        model.train()
        if on_checkpoint:
            on_checkpoint(state,improved)
        if state['completed']:
            return 'completed'
        if stop_after_epoch is not None and state['epochs'] == stop_after_epoch:
            return 'interrupted'
    raise AssertionError('DyKo control loop exited outside its declared stopping state')

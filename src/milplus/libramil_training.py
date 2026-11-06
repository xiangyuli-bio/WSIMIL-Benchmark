"""Libra-MIL source recipe primitives used before full CLI admission.

Do not confuse source-component validation with an installed benchmark profile.
The original accuracy/AUC checkpoint quirk is preserved and reported explicitly.
"""
import copy
import math

import numpy as np
import torch

from .libramil_source import checked_source
from .upstream import load_definitions


def original_training():
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    from typing import List, Tuple, Union
    from tqdm import tqdm
    checked_source()
    metrics = load_definitions('libramil', 'utils/general.py', ['compute_metrics'],
        {'np': np, 'List': List, 'Tuple': Tuple, 'Union': Union,
         'roc_auc_score': roc_auc_score, 'accuracy_score': accuracy_score,
         'f1_score': f1_score, 'label_binarize': label_binarize})
    return load_definitions('libramil', 'main.py', ['train', 'evaluate'],
        {'torch': torch, 'np': np, 'tqdm': tqdm, 'compute_metrics': metrics.compute_metrics})


def source_optimizer(model, train_bags, epochs=20, lr=.0001):
    if (type(train_bags) is not int or train_bags < 1 or type(epochs) is not int or epochs < 1
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Libra-MIL requires finite positive LR and complete positive bag/epoch budgets')
    from transformers import get_scheduler
    # The released source passes only lr to AdamW. Its actual default decay is
    # .01, NOT the1e-5 often used by other MIL methods.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=100,
                              num_training_steps=train_bags*epochs)
    return optimizer, scheduler


class SourceSelection:
    """Literal source main.py, including its stale best_auc on accuracy gain."""
    def __init__(self):
        self.best_acc, self.best_auc, self.selected_epoch = 0., 0., None
        self.history = []

    def observe(self, epoch, accuracy, auc):
        if (type(epoch) is not int or epoch != len(self.history)
                or not all(isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 1
                           for x in [accuracy, auc])):
            raise ValueError('Libra-MIL needs consecutive finite source validation accuracy/AUROC')
        improved = False
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.selected_epoch = epoch
            improved = True
            # Original source does not update best_auc in this branch.
        elif accuracy == self.best_acc and auc > self.best_auc:
            self.best_auc = auc
            self.selected_epoch = epoch
            improved = True
        self.history.append({'epoch': epoch, 'accuracy': float(accuracy),
                             'auc': float(auc), 'selected': improved})
        return improved

    def state_dict(self):
        return copy.deepcopy({'best_acc': self.best_acc, 'best_auc': self.best_auc,
            'selected_epoch': self.selected_epoch, 'history': self.history})

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {'best_acc', 'best_auc', 'selected_epoch', 'history'}:
            raise ValueError('Libra-MIL source checkpoint selection state missing')
        reference = SourceSelection()
        for row in state['history']:
            if set(row) != {'epoch', 'accuracy', 'auc', 'selected'}:
                raise ValueError('Libra-MIL source validation history changed')
            if reference.observe(row['epoch'], row['accuracy'], row['auc']) != row['selected']:
                raise ValueError('Libra-MIL source selection branch disagrees with history')
        if reference.state_dict() != state:
            raise ValueError('Libra-MIL saved source accuracy/AUROC branch state is inconsistent')
        self.__dict__.update(copy.deepcopy(reference.__dict__))

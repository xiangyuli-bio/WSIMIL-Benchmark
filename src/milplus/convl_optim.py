"""Actual author Lookahead/RAdam updates with explicit portable state repair."""
import copy
import math
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import torch
from torch.optim import Optimizer

from .convl_source import checked_source
from .upstream import load_definitions


def original_optimizer_classes():
    checked_source()
    radam = load_definitions('convl', 'optimizers/radam.py', ['RAdam'],
                            {'torch': torch, 'math': math, 'Optimizer': Optimizer})
    lookahead = load_definitions('convl', 'optimizers/lookahead.py', ['Lookahead'],
        {'torch': torch, 'Optimizer': Optimizer, 'defaultdict': defaultdict,
         'Dict': Dict, 'Any': Any, 'State': Dict[str, Any],
         'OptLossClosure': Optional[Callable], 'OptFloat': Optional[float]})
    return radam.RAdam, lookahead.Lookahead


def portable_lookahead_class():
    _, OriginalLookahead = original_optimizer_classes()

    class PortableLookahead(OriginalLookahead):
        def __init__(self, optimizer, k=5, alpha=.5):
            # The old source skips Optimizer.__init__, so modern state hooks
            # are absent. Initialize only bookkeeping before its exact init.
            Optimizer.__init__(self, optimizer.param_groups, optimizer.defaults)
            OriginalLookahead.__init__(self, optimizer, k=k, alpha=alpha)

        def state_dict(self):
            return {**OriginalLookahead.state_dict(self),
                    'rectification_buffer': copy.deepcopy(self.optimizer.buffer)}

        def load_state_dict(self, state):
            if set(state) != {'fast_state', 'slow_state', 'param_groups', 'rectification_buffer'}:
                raise ValueError('ConVL requires both actual Lookahead states and the original RAdam buffer')
            groups = state['param_groups']
            if (len(groups) != len(self.param_groups)
                    or any(type(g.get('counter')) is not int or not 0 <= g['counter'] < self.k for g in groups)
                    or len(state['rectification_buffer']) != 10
                    or any(len(row) != 3 for row in state['rectification_buffer'])):
                raise ValueError('ConVL Lookahead counters or RAdam rectification state changed')
            OriginalLookahead.load_state_dict(self, {k: v for k, v in state.items() if k != 'rectification_buffer'})
            # Source load_state_dict creates separate wrapper/base group lists.
            # Reconnect the actual groups, preserving all source counters and
            # weights; otherwise resumed Lookahead phases silently diverge.
            self.param_groups = self.optimizer.param_groups
            self.fast_state = self.optimizer.state
            self.optimizer.buffer = copy.deepcopy(state['rectification_buffer'])

    return PortableLookahead


def create_source_optimizer(model, lr=.0002, weight_decay=.00001):
    radam, _ = original_optimizer_classes()
    lookahead = portable_lookahead_class()
    factory = load_definitions('convl', 'optimizers/optim_factory.py',
        ['add_weight_decay', 'create_optimizer'],
        {'torch': torch, 'optim': torch.optim, 'has_apex': False, 'RAdam': radam, 'Lookahead': lookahead})
    config = SimpleNamespace(opt='lookahead_radam', lr=lr, weight_decay=weight_decay,
                             opt_eps=None, opt_betas=None, momentum=None)
    return factory.create_optimizer(config, model)

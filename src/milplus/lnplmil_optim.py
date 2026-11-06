"""Original LNPL Lookahead/RAdam equations with portable exact checkpoint state."""
import copy
from collections import defaultdict
import math
from types import SimpleNamespace

import torch
from torch.optim import Optimizer

from .lnplmil_source import checked_source
from .upstream import load_definitions

def original_classes():
    checked_source()
    radam = load_definitions('lnplmil','MyOptimizer/radam.py',['RAdam'],
                             {'torch':torch,'math':math,'Optimizer':Optimizer})
    look = load_definitions('lnplmil','MyOptimizer/lookahead.py',['Lookahead'],
                            {'torch':torch,'Optimizer':Optimizer,'defaultdict':defaultdict})
    return radam.RAdam, look.Lookahead

def portable_class():
    _, Original = original_classes()
    class PortableLookahead(Original):
        def __init__(self, base_optimizer, alpha=.5, k=6):
            # Only modern Torch bookkeeping; the author update equations stay.
            Optimizer.__init__(self,base_optimizer.param_groups,base_optimizer.defaults)
            Original.__init__(self,base_optimizer,alpha=alpha,k=k)

        def state_dict(self):
            parameters = [p for group in self.param_groups for p in group['params']]
            indices = {id(p):i for i,p in enumerate(parameters)}
            return {'fast':self.base_optimizer.state_dict(),
                'slow':{indices[id(p)]:copy.deepcopy(v) for p,v in self.state.items()},
                'rectification_buffer':copy.deepcopy(self.base_optimizer.buffer)}

        def load_state_dict(self, state):
            if set(state)!={'fast','slow','rectification_buffer'}:
                raise ValueError('LNPL needs all actual fast/slow Lookahead and RAdam cache states')
            params = [p for g in self.param_groups for p in g['params']]
            if len(state['rectification_buffer'])!=10 or any(len(v)!=3 for v in state['rectification_buffer']):
                raise ValueError('LNPL RAdam rectification cache changed')
            groups = state['fast']['param_groups']
            if len(groups)!=len(self.param_groups):
                raise ValueError('LNPL original bias/1D weight-decay partition changed')
            for current,saved in zip(self.param_groups,groups):
                if (len(current['params'])!=len(saved['params']) or saved['lookahead_alpha']!=.5
                        or saved['lookahead_k']!=6 or type(saved['lookahead_step']) is not int
                        or saved['lookahead_step']<0):
                    raise ValueError('LNPL Lookahead phase or parameter group changed')
            slow = defaultdict(dict)
            for i,values in state['slow'].items():
                if type(i) is not int or not 0<=i<len(params) or set(values)!={'slow_buffer'}:
                    raise ValueError('LNPL portable slow-state parameter identity changed')
                value = values['slow_buffer']
                if value.shape!=params[i].shape or value.dtype!=params[i].dtype or not torch.isfinite(value).all():
                    raise ValueError('LNPL actual slow-weight tensor changed')
                slow[params[i]] = {'slow_buffer':value.to(params[i].device).clone()}
            self.base_optimizer.load_state_dict(copy.deepcopy(state['fast']))
            self.param_groups = self.base_optimizer.param_groups
            self.base_optimizer.buffer = copy.deepcopy(state['rectification_buffer'])
            self.state = slow
    return PortableLookahead

def create_optimizer(net):
    RAdam,_ = original_classes()
    factory = load_definitions('lnplmil','MyOptimizer/optim_factory.py',
        ['add_weight_decay','create_optimizer'],
        {'torch':torch,'optim':torch.optim,'has_apex':False,'RAdam':RAdam,'Lookahead':portable_class()})
    return factory.create_optimizer(SimpleNamespace(opt='lookahead_radam',lr=.0002,
        weight_decay=.00001,opt_eps=None,opt_betas=None,momentum=None),net)

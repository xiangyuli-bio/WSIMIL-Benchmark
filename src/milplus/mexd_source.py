"""Original MExD TransMIL, conditional denoiser, diffusion and RAdam operators.

The author code has incompatible pretrain/inference MoE classes. The complete
paper-reference controller therefore names its reconstruction explicitly and
loads the unchanged author adapters/noise/posterior definitions privately.
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer, required

from .upstream import load_definitions
from .vendor.transmil.nystrom_attention import NystromAttention

COMMIT='1d4d438e2b19d0fee9cb55328faa84b6ca37fe0b'


def components():
    graph=load_definitions('mexd','classification/Models/TransMIL/net.py',
        ['TransLayer','PPEG','TransMIL'],{'torch':torch,'nn':nn,'F':F,'np':np,
                                     'NystromAttention':NystromAttention})
    denoiser=load_definitions('mexd','classification/model_v1.py',
        ['ConditionalLinear','ConditionalModel'],{'torch':torch,'nn':nn,'F':F})
    functions=['make_beta_schedule','extract','q_sample','p_sample',
               'p_sample_t_1to0','y_0_reparam','p_sample_loop']
    diffusion=load_definitions('mexd','classification/diffusion_utils.py',
        functions,{'math':math,'torch':torch})
    radam=load_definitions('mexd','Opt/radam.py',['RAdam'],
        {'math':math,'torch':torch,'Optimizer':Optimizer,'required':required}).RAdam
    return graph.TransMIL,denoiser.ConditionalModel,diffusion,radam


def denoiser_config(classes,feature_dimension):
    return SimpleNamespace(
        diffusion=SimpleNamespace(timesteps=500),
        data=SimpleNamespace(dataset='WSI',num_classes=classes),
        model=SimpleNamespace(data_dim=512,feature_dim=512,hidden_dim=384,
                              arch='linear',ori_data_dim=feature_dimension))


def complete_radam(parameters,lr,weight_decay):
    """Retain the actual author optimizer, including its unregistered step cache."""
    original=components()[3]
    class RestorableRAdam(original):
        def state_dict(self):
            state=super().state_dict();state['source_step_cache']=copy.deepcopy(self.buffer)
            return state
        def load_state_dict(self,state):
            state=copy.deepcopy(state)
            self.buffer=state.pop('source_step_cache')
            return super().load_state_dict(state)
    return RestorableRAdam(parameters,lr=lr,weight_decay=weight_decay)

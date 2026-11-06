"""Pinned MSMMIL: atrous three-branch scans and the complete Lookahead/RAdam recipe."""
import math
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from .mamba_mil import backend_functions
from .upstream import load_definitions


def author_network(hidden=256, backend='torch', scan_chunk=64):
    no_out, with_out, identity = backend_functions(backend, scan_chunk, source='msmmil')
    def forbidden(*args, **kwargs):
        raise RuntimeError('MSMMIL requires the published fast-path graph; single-branch fallback is disabled')
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'Optional': Optional, 'Tensor': Tensor,
                 'rearrange': rearrange, 'repeat': repeat, 'mamba_inner_fn_no_out_proj': no_out,
                 'mamba_inner_fn': with_out, 'selective_scan_fn': forbidden}
    namespace['AFWMamba'] = load_definitions('msmmil',
        'mamba/mamba_ssm/modules/atrousfourway_mamba.py',
        ['TransposeTokenReEmbedding', 'BiAttn', 'AtrousFourWayMamba'], namespace).AtrousFourWayMamba
    replacements = [] if hidden == 256 else [('512', str(2*hidden)), ('256', str(hidden))]
    network = load_definitions('msmmil', 'models/MKMIL.py', ['initialize_weights', 'MKMIL'],
                               namespace, replacements).MKMIL
    return network, identity


def author_radam():
    return load_definitions('msmmil', 'MyOptimizer/radam.py', ['RAdam'],
        {'torch': torch, 'math': math, 'Optimizer': torch.optim.Optimizer}).RAdam


class MSMMIL(nn.Module):
    def __init__(self, bundle, hidden=256, layers=2, rate=10, dropout=.5,
                 accumulation=2, backend='torch', scan_chunk=64):
        super().__init__()
        if bundle.task_type != 'classification':
            raise ValueError('MSMMIL author recipe currently supports classification only')
        if any(type(v) is not int or v < 1 for v in (hidden, layers, rate, accumulation, scan_chunk)):
            raise ValueError('MSMMIL dimensions, layers, rate, accumulation and scan_chunk must be positive integers')
        if hidden < 2 or rate % 2 or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('MSMMIL needs hidden>=2, even rate, and finite dropout in [0,1)')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('MSMMIL trains individual slide bags; patient bag merging is not registered')
        self.dim, self.hidden, self.layers, self.rate = bundle.dim, hidden, layers, rate
        network, self.backend_identity = author_network(hidden, backend, scan_chunk)
        self.network = network(len(bundle.classes), dropout, 'relu', n_features=bundle.dim,
                               layer=layers, rate=rate, type='AFWMamba')
        self.accumulation, self.steps, self.slow = accumulation, 0, {}

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('MSMMIL preserves the entire registered patch sequence; max_instances is not supported')

    def forward(self, x, coords=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('MSMMIL expects finite nonempty [instances, registered FM dimension] input')
        result = self.network(data=x[None])
        if not torch.isfinite(result['logits']).all():
            raise ValueError('Nonfinite MSMMIL logits')
        return result

    def make_optimizer(self, lr, weight_decay):
        groups = load_definitions('msmmil', 'MyOptimizer/optim_factory.py', ['add_weight_decay'], {})
        return author_radam()(groups.add_weight_decay(self.network, weight_decay), lr=lr, weight_decay=0.)

    @torch.no_grad()
    def after_step(self):
        self.steps += 1
        if self.steps % 6 == 0:
            for name, p in self.network.named_parameters():
                if p.grad is not None:
                    # Preserve first creation at step six. Stable names repair source
                    # state_dict's transient object IDs, allowing an exact fresh-process resume.
                    slow = self.slow.get(name, p.detach().clone()).to(p.device)
                    slow.add_(p-slow, alpha=.5)
                    p.copy_(slow)
                    self.slow[name] = slow

    def get_extra_state(self):
        return {'backend': self.backend_identity, 'steps': self.steps,
                'slow': {name: value.cpu() for name, value in self.slow.items()}}

    def set_extra_state(self, state):
        if state['backend'] != self.backend_identity:
            raise ValueError('MSMMIL backend identity changed since checkpoint creation')
        self.steps, self.slow = state['steps'], state['slow']

    def protocol(self):
        return {'source': 'pinned MSMMIL models/MKMIL.py, AFWMamba branch',
                'hidden': self.hidden, 'layers': self.layers, 'rate': self.rate,
                'backend': self.backend_identity, 'optimizer_steps': self.steps,
                'architecture': 'D->512->256, two residual AFWMamba blocks, 256->512, LayerNorm, attention128; widths shown at native default',
                'branches': 'checkerboard concatenation, alternating row concatenation and original sequence; independent scan parameters, shared BiAttn called three times',
                'order': 'stored patch order; even-rate pseudo-grid with author zero padding, flips and inverse restoration; no physical-coordinate claim',
                'objective': 'cross entropy', 'optimizer': 'author RAdam with bias/1D no decay; Lookahead k=6 alpha=.5, first slow buffer at step6',
                'initialization': 'literal author recursive initializer, including zeroing marked dt_proj biases',
                'accumulation': self.accumulation,
                'tail': 'last short window also divided by configured accumulation, matching released Lightning recipe',
                'resume_repair': 'slow buffers keyed by stable parameter names instead of transient Python object IDs',
                'precision': 'FP32, as released precision=32; APEX AMP behavior not validated',
                'selection': 'validation log loss; patience40; no test-based model selection',
                'gpu_validation': 'not_run', 'paper_reproduction': 'not_run'}


def msmmil_epoch(model, bundle, groups, optimizer, device, config):
    from .engine import load_unit
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, rows in enumerate(groups):
        x, _ = load_unit(bundle, rows, device)
        loss = F.cross_entropy(model(x)['logits'], torch.tensor([rows[0]['target']], device=device))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite MSMMIL loss')
        (loss/model.accumulation).backward()
        losses.append({'loss': float(loss.detach())})
        if (index+1) % model.accumulation == 0 or index+1 == len(groups):
            if config.get('grad_clip') is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
            elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite MSMMIL gradients')
            optimizer.step()
            model.after_step()
            optimizer.zero_grad(set_to_none=True)
    return losses

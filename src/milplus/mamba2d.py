"""Complete original physical 2DMamba classification and patient NLL training recipes."""
import copy
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch
from torch.nn import functional as F

from .discrete_survival import DiscreteSurvival
from .io import digest_file, digest_object
from .mamba2d_data import PhysicalGrid
from .mamba2d_source import VARIANTS, author_model, original_nll, original_optimizer


class Mamba2DMIL(DiscreteSurvival):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'native whole-rectangle attention, including explicit non-tissue pad tokens'

    def __init__(self, bundle, profile='mamba2dmil_paper', hidden=128, state_size=16, layers=1,
                 dropout=.25, pad_token='trainable', inner_layernorms=False, bins=4,
                 backend='torch', cuda_binary=None, workspace_budget_bytes=2**31):
        super().__init__(bins=bins, alpha=0., accumulation_steps=1)
        if profile not in VARIANTS or any(type(v) is not int or v < 1 for v in [hidden, state_size, layers, workspace_budget_bytes]):
            raise ValueError('2DMamba requires a registered profile and positive integer dimensions/resource budget')
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('2DMamba dropout must lie in [0,1)')
        if type(inner_layernorms) is not bool or pad_token not in {'zero', 'trainable'}:
            raise ValueError('2DMamba requires explicit native pad token and boolean inner layernorms')
        self.bundle, self.profile = bundle, profile
        self.grid = PhysicalGrid(bundle)
        self.survival = bundle.task_type == 'survival'
        self.hidden, self.state_size, self.layers = hidden, state_size, layers
        self.dim, self.backend = bundle.dim, backend
        # Estimate dominant scan-state storage before constructing the dense grid.
        # This is an explicit planning guard, not a guarantee on total allocator/autograd memory.
        self.estimated_scan_bytes = (8*self.grid.padded_width*self.grid.padded_height*hidden*2*state_size*4*layers
                                     if backend == 'torch' else 8*self.grid.width*self.grid.height*hidden*2*4*layers)
        if self.estimated_scan_bytes > workspace_budget_bytes:
            raise ValueError('2DMamba dense padded scan exceeds the explicit workspace budget; increase it for available memory or use the separately verified native CUDA backend')
        if backend == 'cuda' and layers != 1:
            raise ValueError('Original 2DMamba CUDA rebuilds a sparse grid at each flattened layer; only its native one-layer configuration is registered')
        self.workspace_budget_bytes = workspace_budget_bytes
        model, self.backend_identity = author_model(profile, backend, cuda_binary)
        args = SimpleNamespace(in_dim=bundle.dim, mambamil_dim=hidden, drop_out=dropout,
            patch_encoder_batch_size=128, mambamil_layer=layers, mambamil_state_dim=state_size,
            mambamil_inner_layernorms=inner_layernorms, pscan=True, cuda_pscan=backend == 'cuda',
            model_type='2DMambaMIL', mamba_2d_max_w=self.grid.width-1,
            mamba_2d_max_h=self.grid.height-1, mamba_2d_pad_token=pad_token,
            mamba_2d_patch_size=1, n_classes=bins if self.survival else len(bundle.classes),
            survival=self.survival, pos_emb_type=None, pos_emb_dropout=0.)
        # Keep the author's backend diagnostics, while preserving machine-readable CLI output.
        with redirect_stdout(sys.stderr):self.network = model(args)
        self.nll = original_nll()
        self.optimizer_steps, self.completed_epochs = 0, 0
        self.scheduler, self.saved_scheduler = None, None
        self.scheduler_epochs = 20
        self.trace = None
        self.network.attention.register_forward_hook(self._capture_attention)
        self.contract = digest_object({'profile': profile, 'backend': self.backend_identity,
            'grid': self.grid.metadata(), 'model': vars(args), 'budget': workspace_budget_bytes})

    def _capture_attention(self, module, inputs, output):
        if not self.training:self.trace = output.detach().reshape(-1).softmax(0)

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('2DMamba original recipes use complete physical bags and unclipped gradients')
        required = 'c_index' if self.survival else 'mamba2d_bag_auc'
        if cfg['selection_metric'] != required:
            raise ValueError('2DMamba requires its native validation selection metric: '+required)
        self.scheduler_epochs = cfg['epochs']

    def prepare(self, bundle, seed, device):
        if self.survival:super().prepare(bundle, seed, device)

    def make_optimizer(self, lr, weight_decay):
        optimizer = original_optimizer(self.network, lr, weight_decay)
        if not self.survival:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler_epochs, eta_min=0)
            if self.saved_scheduler is not None:self.scheduler.load_state_dict(self.saved_scheduler)
        return optimizer

    def finish_epoch(self, bundle, device, cfg):
        if self.scheduler is not None:self.scheduler.step()
        self.completed_epochs += 1

    def selection_improved(self, objective, best):
        return objective > best if self.survival else objective >= best

    def validation_metrics(self, frame):
        if self.survival:return {}
        from .metrics import classification_metrics
        return {'mamba2d_bag_auc': classification_metrics(frame, len(self.bundle.classes))['macro_auc']}

    def forward(self, x, coords=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('2DMamba requires finite nonempty registered patch features')
        if coords is None or coords.shape != (len(x), 2) or coords.dtype != torch.int64 or (coords < 0).any():
            raise ValueError('2DMamba requires validated integer physical grid indices')
        if (coords[:, 0] >= self.grid.width).any() or (coords[:, 1] >= self.grid.height).any() or len(torch.unique(coords, dim=0)) != len(x):
            raise ValueError('2DMamba grid cells collide or exceed the declared physical canvas')
        if self.backend == 'cuda' and x.device.type != 'cuda':
            raise ValueError('Native 2DMamba CUDA cannot execute on CPU; no hidden backend switch')
        self.trace = None
        output = self.network(x, coords)
        if self.survival:
            hazards, survival = output[:2]
            if not torch.isfinite(hazards).all() or not torch.isfinite(survival).all():
                raise ValueError('Nonfinite 2DMamba hazards')
            return {'logits': -survival.sum(1, keepdim=True), 'hazards': hazards, 'survival': survival}
        if not torch.isfinite(output[0]).all():raise ValueError('Nonfinite 2DMamba logits')
        return {'logits': output[0]}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:raise ValueError('2DMamba does not concatenate unrelated WSI coordinate frames')
        x, grid, _ = self.grid.inputs(rows[0])
        return self(x.to(device), grid.to(device))

    def objective(self, result, time, event):
        return self.nll(result['hazards'], result['survival'], self.time_label([time]),
                        self.bin_edges.new_tensor([1-event]), alpha=0.)

    def get_extra_state(self):
        return {'contract': self.contract, 'optimizer_steps': self.optimizer_steps,
            'completed_epochs': self.completed_epochs,
            'scheduler': copy.deepcopy(self.scheduler.state_dict() if self.scheduler is not None else self.saved_scheduler)}

    def set_extra_state(self, state):
        if set(state) != {'contract', 'optimizer_steps', 'completed_epochs', 'scheduler'} or state['contract'] != self.contract or any(
                type(state[k]) is not int or state[k] < 0 for k in ['optimizer_steps', 'completed_epochs']):
            raise ValueError('2DMamba checkpoint input/backend contract or counters changed')
        self.optimizer_steps, self.completed_epochs = state['optimizer_steps'], state['completed_epochs']
        self.saved_scheduler = copy.deepcopy(state['scheduler'])
        if self.scheduler is not None:self.scheduler.load_state_dict(self.saved_scheduler)

    def export_instances(self, result, row, directory):
        directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object(row['slide_id'])[:24]+'.h5')
        _, grid, coords = self.grid.inputs(row)
        if self.trace is None or len(self.trace) != self.grid.width*self.grid.height:
            raise ValueError('2DMamba native whole-grid attention was not captured')
        attention = self.trace.cpu()
        index = grid[:, 1]*self.grid.width+grid[:, 0]
        mask = torch.zeros(len(self.trace), dtype=torch.bool);mask[index] = True
        with h5py.File(path, 'w') as h:
            h['coords'] = coords;h['grid_indices'] = grid.numpy()
            h['attention'] = attention[index].numpy()
            h['whole_grid_attention'] = attention.reshape(self.grid.height, self.grid.width).numpy()
            h['tissue_mask'] = mask.reshape(self.grid.height, self.grid.width).numpy()
            h.attrs['semantics'] = self.instance_score_semantics
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path),
            'instances': len(grid), 'grid_cells': len(self.trace), 'tissue_attention_mass': float(attention[index].sum()),
            'pad_attention_mass': float(attention[~mask].sum())}

    def protocol(self):
        return {**(super().protocol() if self.survival else {}), 'profile': self.profile,
            'backend': self.backend_identity, 'grid': self.grid.metadata(), 'hidden': self.hidden,
            'state_size': self.state_size, 'layers': self.layers, 'estimated_scan_bytes': self.estimated_scan_bytes,
            'workspace_budget_bytes': self.workspace_budget_bytes,
            'scan': VARIANTS[self.profile], 'optimizer': 'original AdamW all trainable network parameters, source Xavier initialization',
            'scan_contract': ('original separately fingerprinted fused CUDA forward/backward; no CPU repair is applied'
                if self.backend == 'cuda' else {
                    'source': 'literal published parallel PyTorch forward and custom backward, including verified defects',
                    'autograd': 'bitwise original PyTorch forward; indexed updates functionalized for correct automatic derivatives',
                    'paper': 'two vertical downsweep A-to-A_raw repairs plus automatic derivatives, independently checked against Eq5/Eq6'
                }[VARIANTS[self.profile]]),
            'training_order': 'source sequential slide order' if not self.survival else 'source random patient sampler without replacement',
            'objective': 'source NLL(alpha0)/32, optimizer step per patient; no accumulation' if self.survival else 'source full-bag CE',
            'scheduler': None if self.survival else 'epoch cosine, T_max=requested epochs, eta_min0',
            'selection': 'strict validation patient C-index' if self.survival else 'validation slide macro AUROC, last tied best',
            'optimizer_steps': self.optimizer_steps, 'completed_epochs': self.completed_epochs,
            'repairs': ['explicit physical grid registration and collision/budget checks',
                        'TRAIN-only observed-time bins, one registered WSI per patient',
                        'first valid checkpoint retained even at zero validation metric',
                        'TEST sealed from model selection; complete scheduler/optimizer/RNG restore'],
            'source_gradient_warning': 'original CPU custom backward fails finite-difference checks' if self.profile == 'mamba2dmil_source' and self.backend == 'torch' else None,
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}


def mamba2d_epoch(model, bundle, groups, optimizer, device, cfg):
    if model.survival:
        # Match both DataLoader base-seed and RandomSampler seed consumption, without I/O workers.
        from torch.utils.data import DataLoader, RandomSampler
        groups = DataLoader(groups, batch_size=1, sampler=RandomSampler(groups), num_workers=0,
                            collate_fn=lambda batch: batch[0])
    losses = []
    for rows in groups:
        output = model.forward_unit(bundle, rows, device, cfg, train=True)
        raw_loss = model.objective(output, rows[0]['time'], rows[0]['event']) if model.survival else F.cross_entropy(
            output['logits'], torch.tensor([rows[0]['target']], device=device))
        loss = raw_loss/32 if model.survival else raw_loss
        if not torch.isfinite(loss):raise ValueError('Nonfinite 2DMamba training objective')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite 2DMamba gradients')
        optimizer.step();optimizer.zero_grad();model.optimizer_steps += 1
        losses.append({'loss': float(raw_loss.detach()), 'optimization_loss': float(loss.detach())})
    return losses

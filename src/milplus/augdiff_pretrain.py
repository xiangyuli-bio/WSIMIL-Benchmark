"""Complete native-width seven-view DAE stage, EMA selection and restart state.

This implements the full pretraining mechanism. Epoch budget is explicit because
the public paper/config does not specify it. MIL consumes the selected raw
weights, matching the released loader, NOT the EMA inference weights.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augdiff_source import COMMIT, LAYOUT, LDM_COMMIT, build_diffusion, epsilon_loss
from .io import atomic_torch_save, digest_file, digest_object, write_json


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('AugDiff cannot exactly resume CUDA diffusion RNG on a CPU host')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def backend_flags():
    return {'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}


def training_contract(assets, epochs, batch_size, precision, seed, device, torch_version=None, backend=None):
    """Compute the exact preparation identity without allocating a second DAE."""
    return digest_object({'source': COMMIT, 'ldm': LDM_COMMIT, 'layout': LAYOUT,
        'assets': assets.contract, 'epochs': epochs, 'batch_size': batch_size, 'precision': precision,
        'base_lr': 5e-8, 'scaled_lr': 5e-8*batch_size, 'devices': 1, 'accumulate_grad': 1, 'seed': seed,
        'device_type': torch.device(device).type, 'torch_version': str(torch.__version__) if torch_version is None else torch_version,
        'backend_flags': backend_flags() if backend is None else backend,
        'runtime': {name: digest_file(Path(__file__).with_name(name))
            for name in ['augdiff_source.py', 'augdiff_data.py', 'augdiff_pretrain.py']}})


class DiffusionTrainer:
    def __init__(self, assets, epochs, device='cpu', batch_size=1200, precision='fp32', seed=2026):
        if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
                or precision not in {'fp32', 'fp16'} or type(seed) is not int):
            raise ValueError('AugDiff DAE needs explicit positive full epoch/batch budgets and fp32/fp16 precision')
        self.device = torch.device(device)
        if precision == 'fp16' and self.device.type != 'cuda':
            raise ValueError('AugDiff fp16 DAE requires CUDA; CPU fp32 is a disclosed device port')
        self.assets, self.epochs, self.batch_size, self.precision, self.seed = assets, epochs, batch_size, precision, seed
        self.backend = backend_flags()
        self.lr = 5e-8 * batch_size
        random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
        self.contract = training_contract(assets, epochs, batch_size, precision, seed, self.device, backend=self.backend)
        self.model = build_diffusion(assets.bundle.dim).to(self.device)
        self.model.learning_rate = self.lr
        self.optimizer = self.model.configure_optimizers()
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise TypeError('AugDiff full DAE must use source AdamW')
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=precision == 'fp16')
        self.completed_epochs, self.optimizer_updates, self.best_epoch, self.bad_epochs = 0, 0, -1, 0
        self.best_score, self.orders, self.history = float('inf'), [], []

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.precision == 'fp16')

    def checkpoint(self):
        return {'contract': self.contract, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(), 'rng': rng_state(), 'completed_epochs': self.completed_epochs,
            'optimizer_updates': self.optimizer_updates, 'best_epoch': self.best_epoch, 'best_score': self.best_score,
            'bad_epochs': self.bad_epochs, 'orders': self.orders, 'history': self.history}

    def resume(self, path):
        # Private training artifact produced by this trainer. Public arbitrary
        # checkpoints are not accepted as an unvalidated pretrained substitute.
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get('contract') != self.contract:
            raise ValueError('AugDiff diffusion source/input/budget/device-precision contract changed')
        self.assets.check_unchanged()
        if (type(state['completed_epochs']) is not int or not 0 <= state['completed_epochs'] <= self.epochs
                or len(state['orders']) != state['completed_epochs'] or len(state['history']) != state['completed_epochs']):
            raise ValueError('AugDiff diffusion checkpoint epoch history is incomplete')
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer']); self.scaler.load_state_dict(state['scaler'])
        for key in ['completed_epochs', 'optimizer_updates', 'best_epoch', 'best_score', 'bad_epochs', 'orders', 'history']:
            setattr(self, key, state[key])
        expected = self.completed_epochs * ((len(self.assets.patches('train'))+self.batch_size-1)//self.batch_size)
        # An AMP overflow may skip an optimizer step, so the update count is
        # checked against recorded actual steps rather than guessed from epochs.
        if self.optimizer_updates != sum(x['optimizer_updates'] for x in self.history) or not 0 <= self.optimizer_updates <= expected:
            raise ValueError('AugDiff DAE actual optimizer-update history is inconsistent')
        for order in self.orders:
            if sorted(order) != list(range(len(self.assets.patches('train')))):
                raise ValueError('AugDiff DAE checkpoint omitted/duplicated registered TRAIN view patches')
        restore_rng(state['rng'])

    @torch.no_grad()
    def validate(self):
        self.model.eval(); raw, ema, count = 0., 0., 0
        loader = DataLoader(self.assets.patches('val'), batch_size=self.batch_size, shuffle=False, num_workers=0)
        for x in loader:
            x = x.to(self.device)
            # Original validation_step performs two separate stochastic calls;
            # they deliberately draw their own times/noise, raw then EMA.
            with self.autocast():
                _, metric = epsilon_loss(self.model, x)
            raw += float(metric['val/loss_simple']) * len(x)
            with self.model.ema_scope(), self.autocast():
                _, metric = epsilon_loss(self.model, x)
            ema += float(metric['val/loss_simple']) * len(x)
            count += len(x)
        # The original store/restore cache has no numerical role after restore.
        # Release its extra full-model clone before the next training epoch.
        self.model.model_ema.collected_params = []
        return {'raw_mse': raw/count, 'ema_mse': ema/count, 'patches': count}

    def fit(self, output, stop_after=None, resume=None):
        if backend_flags() != self.backend:
            raise ValueError('AugDiff DAE backend flags changed after constructing the preparation recipe')
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        if resume is not None:
            self.resume(resume)
        elif (output/'last.pt').exists():
            raise FileExistsError('AugDiff DAE output already contains training; pass an explicit resume checkpoint')
        self.assets.check_unchanged()
        if stop_after is not None and (type(stop_after) is not int or not self.completed_epochs < stop_after <= self.epochs):
            raise ValueError('AugDiff stop_after must advance complete epochs within the explicit DAE budget')
        end = self.epochs if stop_after is None else stop_after
        dataset = self.assets.patches('train')
        for epoch in range(self.completed_epochs, end):
            self.model.train(); self.model.current_epoch = epoch
            order = torch.randperm(len(dataset)).tolist()
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=order, num_workers=0)
            total, count, updates = 0., 0, 0
            for features in loader:
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    loss, _ = epsilon_loss(self.model, features.to(self.device))
                if not bool(torch.isfinite(loss)):
                    raise ValueError('AugDiff full diffusion objective is nonfinite')
                self.scaler.scale(loss).backward()
                before = self.scaler.get_scale()
                self.scaler.step(self.optimizer); self.scaler.update()
                stepped = self.scaler.get_scale() >= before
                updates += int(stepped); self.optimizer_updates += int(stepped)
                self.model.global_step = self.optimizer_updates
                # Match original on_train_batch_end even on an AMP overflow.
                self.model.on_train_batch_end()
                total += float(loss.detach()) * len(features); count += len(features)
            validation = self.validate()
            score = validation['ema_mse']
            if not np.isfinite(score):
                raise ValueError('AugDiff EMA validation objective is nonfinite')
            improved = score < self.best_score
            if improved:
                self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
            else:
                self.bad_epochs += 1
            self.completed_epochs = epoch+1; self.orders.append(order)
            self.history.append({'epoch': epoch, 'train_mse': total/count, 'optimizer_updates': updates,
                                 'validation': validation, 'selected': improved})
            # Both artifacts include RAW and EMA weights. Downstream sampler
            # uses the raw weights of the checkpoint selected by EMA VAL loss.
            state = self.checkpoint()
            atomic_torch_save(state, output/'last.pt')
            if improved:
                atomic_torch_save(state, output/'selected.pt')
            write_json(output/'protocol.json', self.protocol())
        return self.protocol()

    def protocol(self):
        return {'contract': self.contract, 'source': COMMIT, 'ldm': LDM_COMMIT, 'layout': LAYOUT,
            'full_dae_parameters': sum(p.numel() for p in self.model.parameters()),
            'unet_base_width': 256, 'channel_multipliers': [1, 2, 4], 'num_res_blocks': 2,
            'diffusion_timesteps': 1000, 'feature_dimension': self.assets.bundle.dim,
            'image_views': 7, 'diffusion_train_patch_count': len(self.assets.patches('train')),
            'diffusion_val_patch_count': len(self.assets.patches('val')),
            'optimizer': 'source AdamW, weight_decay0.01', 'base_lr': 5e-8, 'scaled_lr': self.lr,
            'batch_size': self.batch_size, 'precision': self.precision, 'max_epochs': self.epochs,
            'device_type': self.device.type, 'seed': self.seed, 'train_assets_contract': self.assets.contract,
            'torch_version': str(torch.__version__),
            'backend_flags': self.backend,
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'selected_epoch': self.best_epoch, 'selection': 'first strict minimum val/loss_simple_ema',
            'mil_sampling_weights': 'selected checkpoint RAW weights, no ema_scope, as released',
            'budget_disclosure': 'paper/config does not publish epoch budget; caller explicitly declares it',
            'split_port': 'internal patient-disjoint subsets restricted to supervised TRAIN, not original unsorted per-view glob80/20',
            'historical_architecture_reproduced': False, 'paper_reproduction': 'not_run'}

"""Complete released-source RGB/ResNet/sparse-convolution/Adam recipe."""
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import digest_object
from .sparseconvmil_source import COMMIT, PROFILES, build_model


class Tiles(Dataset):
    def __init__(self, owner, rows, trace=None):
        self.owner, self.rows, self.trace = owner, rows, trace

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with self.owner.assets.open(row) as (pixels, coordinates):
            # Literal author Dataset: choices WITH replacement, even when the
            # WSI has more tiles. Do not replace by randperm or all-tile bags.
            indices = random.choices(range(len(pixels)), k=self.owner.tiles_per_slide)
            images = [torch.from_numpy(np.array(pixels[i], copy=True).transpose(2, 0, 1).copy()).float().div(255)
                      for i in indices]
            locations = torch.tensor(np.asarray([coordinates[i] for i in indices]), dtype=torch.long)
        if self.trace is not None:
            self.trace.append({'slide_id': row['slide_id'], 'indices': list(indices)})
        return torch.stack(images), locations, row['target'], row['slide_id']


class SparseConvMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='sparseconvmil_raw_source_c32',
                 tiles_per_slide=5, batch_size=2, downsample=10, evaluation_seed=1979):
        super().__init__()
        if profile not in PROFILES or bundle.sparseconvmil is None:
            raise ValueError('SparseConvMIL requires its complete original RGB and spatial input contract')
        if any(type(v) is not int or v < 1 for v in [tiles_per_slide, batch_size, downsample]):
            raise ValueError('SparseConvMIL tile count, bag batch size and spatial downsample must be positive integers')
        if type(evaluation_seed) is not int or evaluation_seed < 0:
            raise ValueError('SparseConvMIL deterministic benchmark evaluation needs an explicit nonnegative seed')
        self.bundle, self.assets, self.profile = bundle, bundle.sparseconvmil, profile
        self.tiles_per_slide, self.batch_size, self.downsample = tiles_per_slide, batch_size, downsample
        self.evaluation_seed = evaluation_seed
        self.network, self.backend_identity = build_model(len(bundle.classes), profile, downsample)
        self.contract = digest_object({'source': COMMIT, 'profile': profile,
            'bundle': bundle.fingerprint, 'input': self.assets.contract, 'backend': self.backend_identity,
            'tiles_per_slide': tiles_per_slide, 'batch_size': batch_size,
            'downsample': downsample, 'evaluation_seed': evaluation_seed})
        self.recipe = None
        self.completed_epochs, self.optimizer_updates = 0, 0
        self.epoch_index = -1
        self.train_orders, self.samples_history = [], []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'sparseconvmil_completed_epoch':
            raise ValueError('SparseConvMIL source has no generic feature cap, clipping or validation-based selection')
        self.recipe = {key: cfg[key] for key in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 1 or groups[0]['lr'] != self.recipe['lr']
                or groups[0]['weight_decay'] != self.recipe['weight_decay']
                or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('SparseConvMIL original complete-network Adam changed')
        if self.optimizer_updates and (len(self._optimizer.state) != len(list(self.network.parameters()))
                or any(int(state['step']) != self.optimizer_updates for state in self._optimizer.state.values())):
            raise ValueError('SparseConvMIL optimizer did not update all original network parameters')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('SparseConvMIL must resume at a complete epoch boundary')
        self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('SparseConvMIL input or complete-network Adam changed')
        self.network.train()
        samples, order, losses = [], [], []
        loader = DataLoader(Tiles(self, bundle.records('train'), samples),
            batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        for data, locations, labels, ids in loader:
            data, locations, labels = data.to(device), locations.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = self.network(data, locations)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('SparseConvMIL full original CE became nonfinite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError('SparseConvMIL full original backward became nonfinite')
            optimizer.step(); self.optimizer_updates += 1
            order.extend(ids)
            losses.append({'loss': float(loss.detach()), 'batch_slides': len(ids),
                           'sampled_tiles_per_slide': self.tiles_per_slide})
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(order); self.samples_history.append(samples)
        self.validate_optimizer_state()
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('SparseConvMIL needs its registered validation/test image partition')
        from .engine import restore_rng, rng_state
        saved = rng_state()
        records = []
        self.network.eval()
        try:
            # The public trainer has no evaluation or checkpoint-selection
            # implementation. Benchmark inference is an explicit extension:
            # fixed per-slide seeded original sampling + original coordinate
            # transforms, with all global RNG restored afterward. Thus VAL
            # cannot alter the next source training epoch or choose a model.
            for row in bundle.records(subset):
                seed = (int(digest_object({'slide_id': row['slide_id'], 'subset': subset,
                    'seed': self.evaluation_seed})[:12], 16)) % (2**32)
                random.seed(seed)
                sample = Tiles(self, [row])[0]
                images, locations = sample[0][None].to(device), sample[1][None].to(device)
                logits = self.network(images, locations)
                record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'target']}
                record.update(subset=subset, source_slides=row['slide_id'], prediction=int(logits.argmax(1)))
                record.update({f'p_{i}': float(value) for i, value in enumerate(logits.softmax(-1)[0])})
                records.append(record)
        finally:
            restore_rng(saved)
        return pd.DataFrame(records)

    def validation_metrics(self, predictions):
        # Released main() trains the entire requested budget and keeps final
        # weights. Never select by an invented validation AUROC/accuracy rule.
        return {'sparseconvmil_completed_epoch': self.completed_epochs}

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'train_orders': self.train_orders, 'samples_history': self.samples_history})

    def set_extra_state(self, state):
        if state.get('contract') != self.contract:
            raise ValueError('SparseConvMIL backend, raw image, spatial sampling or recipe identity changed')
        epochs, updates = state.get('completed_epochs'), state.get('optimizer_updates')
        batches = (len(self.bundle.records('train'))+self.batch_size-1)//self.batch_size
        if (type(epochs) is not int or epochs < 0 or updates != epochs*batches
                or len(state.get('train_orders', [])) != epochs
                or len(state.get('samples_history', [])) != epochs
                or (self.recipe is not None and state.get('recipe') != self.recipe)):
            raise ValueError('SparseConvMIL checkpoint lost complete source epochs, optimizer or sampling state')
        rows = {row['slide_id']: self.assets.meta['slides'][row['slide_id']]['shape'][0]
                for row in self.bundle.records('train')}
        for order, samples in zip(state['train_orders'], state['samples_history']):
            if sorted(order) != sorted(rows) or [s['slide_id'] for s in samples] != order:
                raise ValueError('SparseConvMIL checkpoint lost actual source shuffled slide order')
            for sample in samples:
                if (len(sample['indices']) != self.tiles_per_slide
                        or any(type(i) is not int or not 0 <= i < rows[sample['slide_id']] for i in sample['indices'])):
                    raise ValueError('SparseConvMIL checkpoint lost original with-replacement tile sampling')
        for key, value in state.items():
            if key != 'contract':
                setattr(self, key, copy.deepcopy(value))
        self.epoch_index = epochs-1

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'backend': self.backend_identity,
            'network': 'full trained ResNet18, two original3x3 submanifold sparse convolutions, active-only mean and32-hidden classifier',
            'initialization': 'released public training.py default random ResNet18; no pretrained/FM substitute',
            'source_recipe': '200epochs, Adam2e-3/wd1e-6, slide batch2,5tiles/slide with replacement; no scheduler/clipping/early stop',
            'recipe': self.recipe, 'tiles_per_slide': self.tiles_per_slide, 'batch_size': self.batch_size,
            'downsample': self.downsample, 'preprocessing': 'original RGB/255 ToTensor only; no resize/normalization',
            'spatial': 'actual declared original coordinates; source random translation, rotation, flips, per-axis resize and truncation; duplicate-site mode4 averaging',
            'source_rotation_units': 'preserved source uniform(-180,180) argument to math.sin/cos; no silent degrees-to-radians correction',
            'coordinate_augmentation_eval': 'source forward augments even in eval; explicit fixed-per-slide seeded benchmark inference restores training RNG',
            'selection': 'last complete epoch, independent of VAL values; author public main has no validation implementation',
            'ports': 'explicit model/device placement and modern torchvision keyword-only constructor; DataLoader workers0 for exact epoch-boundary recovery',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'synthetic': bool(self.bundle.config.get('synthetic', False)), 'real_wsi_validation': 'not_run',
            'paper_scores': 'not_reproduced', 'paper_recipe_difference': 'TCGA appendix uses pretrainedResNet34/200cropped128px tiles/batch10/Adam1e-4 wd1e-4/200epochs and additional augmentation; public entry is a separately identified source condition'}

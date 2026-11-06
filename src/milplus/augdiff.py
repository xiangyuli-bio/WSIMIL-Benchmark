"""Complete reconstructed AugDiff MIL recipe using the actual full frozen DAE."""
import copy
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .augdiff_pretrain import restore_rng, rng_state
from .augdiff_source import (
    COMMIT,
    LAYOUT,
    LDM_COMMIT,
    augment,
    build_classifier,
    classifier_optimizer,
    classifier_result,
)
from .focus import SlideRows, one_row
from .io import digest_object

PROFILES = {f'augdiff_{head}_{variant}': (head, variant == 'schedule_repaired_reconstruction')
    for head in ['amil', 'lossattn', 'dsmil']
    for variant in ['released_time_reconstruction', 'schedule_repaired_reconstruction']}


class AugDiffMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile, steps=20, nu=.4, precision='auto', max_dense_bytes=32*1024**3):
        super().__init__()
        if profile not in PROFILES or bundle.augdiff is None or bundle.task_type != 'classification':
            raise ValueError('AugDiff requires a bound actually trained full DAE and slide classification')
        if (bundle.config.get('training_unit', 'slide') != 'slide' or steps not in {20, 30}
                or type(steps) is not int or isinstance(nu, bool) or nu not in {.2, .4}
                or precision not in {'auto', 'fp32', 'fp16'}
                or type(max_dense_bytes) is not int or max_dense_bytes < 1):
            raise ValueError('AugDiff requires complete bags, explicit T20/30, nu0.2/0.4 and valid precision')
        self.bundle, self.assets, self.profile = bundle, bundle.augdiff, profile
        self.head, self.repaired = PROFILES[profile]
        self.network = build_classifier(self.head, bundle.dim, len(bundle.classes))
        # Source inference DAE is a global outside the classifier and optimizer.
        # Strict immutable assets bind its full weights; it is not resaved on
        # every MIL epoch and is loaded lazily only for TRAIN/VAL augmentation.
        object.__setattr__(self, '_dae', None)
        self.settings = {'steps': steps, 'nu': nu, 'precision': precision, 'max_dense_bytes': max_dense_bytes}
        self.contract = digest_object({'source': COMMIT, 'ldm': LDM_COMMIT, 'layout': LAYOUT,
            'profile': profile, 'assets': self.assets.contract, 'settings': self.settings,
            'bundle': bundle.fingerprint, 'train_repetitions': 7, 'val_repetitions': 7,
            'accumulation_divisor_including_tail': 2, 'dataloader_port': 'TRAIN RandomSampler, VAL sequential, workers0'})
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders, self.window_history = [], []
        self.validation_epoch, self.validation_rng = -1, None
        self._pending_scaler = None

    def _apply(self, fn, recurse=True):
        if hasattr(self, '_optimizer'):
            raise ValueError('Move AugDiff to its target device before creating its optimizer')
        result = super()._apply(fn, recurse)
        if self._dae is not None:
            self._dae._apply(fn, recurse)
        return result

    def train(self, mode=True):
        result = super().train(mode)
        if self._dae is not None:
            self._dae.eval()
        return result

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'augdiff_source_ce'):
            raise ValueError('AugDiff requires complete bags, no generic clipping and original repeated-VAL CE selection')

    def precision(self, device):
        device = torch.device(device)
        value = self.settings['precision']
        value = ('fp16' if device.type == 'cuda' else 'fp32') if value == 'auto' else value
        if value == 'fp16' and device.type != 'cuda':
            raise ValueError('AugDiff fp16 requires CUDA; use the explicit CPU fp32 port')
        return value

    def autocast(self, device):
        device = torch.device(device)
        return torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=self.precision(device) == 'fp16')

    def make_optimizer(self, lr, weight_decay):
        self._configured_lr, self._configured_decay = lr, weight_decay
        self._optimizer = classifier_optimizer(self.network, lr=lr, weight_decay=weight_decay)
        device = next(self.network.parameters()).device
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.precision(device) == 'fp16')
        if self._pending_scaler is not None:
            self.scaler.load_state_dict(self._pending_scaler)
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 2 or [g['weight_decay'] for g in groups] != [0., self._configured_decay]
                or any(g['lr'] != self._configured_lr or g['lookahead_k'] != 6
                       or g['lookahead_alpha'] != .5 or g['lookahead_step'] != self.optimizer_updates for g in groups)
                or {id(p) for g in groups for p in g['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('AugDiff checkpoint optimizer changed source classifier ownership, LR or Lookahead state')

    def reserve(self, count):
        # Full source inference blocks and complete rows. Fail before allocating
        # instead of substituting a smaller DAE or truncating a whole-slide bag.
        estimate = 4*self.assets.metadata['pretrain_protocol']['full_dae_parameters'] + count*(self.bundle.dim+1024)*128
        if estimate > self.settings['max_dense_bytes']:
            raise ValueError('AugDiff complete frozen DAE/feature graph exceeds the explicit memory budget')

    def result(self, row, device, augmented):
        features = self.assets.load_slide(row).to(device)
        self.reserve(len(features))
        with self.autocast(device):
            if augmented:
                if self._dae is None:
                    object.__setattr__(self, '_dae', self.assets.load_dae(device))
                features = augment(self._dae, features, self.settings['steps'], self.settings['nu'], self.repaired)
            target = torch.tensor([row['target']], dtype=torch.long, device=device)
            result = classifier_result(self.network, features, target)
        if not torch.isfinite(result['loss']) or not torch.isfinite(result['logits']).all():
            raise ValueError('AugDiff native classifier objective is nonfinite')
        return result

    def start_epoch(self, epoch):
        self.assets.check_unchanged(); self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        dataset = SlideRows(bundle.records('train')*7)
        loader = DataLoader(dataset, batch_size=1, sampler=RandomSampler(dataset),
                            num_workers=0, collate_fn=one_row)
        losses, order, windows, current = [], [], [], []
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(loader):
            result = self.result(row, device, augmented=True)
            # Lightning1.2.3 normalizes by the configured accumulation count,
            # including the short final window. Do not renormalize that tail.
            self.scaler.scale(result['loss']/2).backward()
            order.append(row['slide_id']); current.append(row['slide_id'])
            losses.append({'loss': float(result['loss'].detach())})
            if (index+1) % 2 == 0 or index+1 == len(loader):
                scale = self.scaler.get_scale()
                self.scaler.step(optimizer); self.scaler.update()
                stepped = self.scaler.get_scale() >= scale
                self.optimizer_updates += int(stepped)
                windows.append({'slides': current, 'divisor': 2, 'stepped': stepped})
                current = []; optimizer.zero_grad(set_to_none=True)
        self.train_orders.append(order); self.window_history.append(windows)
        self.completed_epochs = self.epoch_index+1
        return losses

    @contextmanager
    def validation_stream(self):
        if self.validation_epoch == self.epoch_index and self.validation_rng is not None:
            caller = rng_state()
            try:
                restore_rng(self.validation_rng)
                yield
            finally:
                restore_rng(caller)
        else:
            self.validation_rng = rng_state()
            yield
            self.validation_epoch = self.epoch_index

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'}:
            raise ValueError('AugDiff evaluation subset must be val or test')
        self.eval(); self.assets.check_unchanged()
        rows = bundle.records(subset)
        repeated = rows*7 if subset == 'val' else rows
        # TEST is clean and must never load or invoke the frozen DAE.
        values = {r['slide_id']: [] for r in rows}

        def compute():
            loader = DataLoader(SlideRows(repeated), batch_size=1,
                sampler=SequentialSampler(repeated), num_workers=0, collate_fn=one_row)
            for row in loader:
                result = self.result(row, device, augmented=subset == 'val')
                logits = result['logits'].float()
                # Numerical repair of original softmax/log CPU reconstruction:
                # identical CE mathematically, stable for half-precision tails.
                ce = -logits.log_softmax(-1)[0, row['target']]
                values[row['slide_id']].append((logits.softmax(-1)[0].cpu(), float(ce)))

        if subset == 'val':
            with self.validation_stream():
                compute()
        else:
            compute()
        records = []
        for row in rows:
            draws = values[row['slide_id']]
            probability = torch.stack([value[0] for value in draws]).mean(0)
            result = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'target']}
            result.update(source_slides=row['slide_id'], augdiff_draws=len(draws),
                          augdiff_source_ce=float(np.mean([value[1] for value in draws])))
            result.update({f'p_{i}': float(v) for i, v in enumerate(probability)})
            records.append(result)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        values = frame['augdiff_source_ce'].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('AugDiff original repeated-VAL CE is missing or nonfinite')
        return {'augdiff_source_ce': float(values.mean())}

    def selection_objective(self, metrics, epoch, current):
        return -metrics['augdiff_source_ce']

    def get_extra_state(self):
        return {'contract': self.contract, 'epoch_index': self.epoch_index,
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'train_orders': copy.deepcopy(self.train_orders), 'window_history': copy.deepcopy(self.window_history),
            'validation_epoch': self.validation_epoch, 'validation_rng': copy.deepcopy(self.validation_rng),
            'scaler': copy.deepcopy(self.scaler.state_dict() if hasattr(self, 'scaler') else self._pending_scaler)}

    def set_extra_state(self, state):
        keys = {'contract', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders',
                'window_history', 'validation_epoch', 'validation_rng', 'scaler'}
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.contract
                or any(type(state[key]) is not int for key in ['epoch_index', 'completed_epochs', 'optimizer_updates', 'validation_epoch'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or state['validation_epoch'] != state['epoch_index']
                or len(state['train_orders']) != state['completed_epochs']
                or len(state['window_history']) != state['completed_epochs']):
            raise ValueError('AugDiff full stage/input/sampler/epoch checkpoint contract differs')
        expected = Counter(row['slide_id'] for row in self.bundle.records('train')*7)
        if state['completed_epochs'] and (not isinstance(state['validation_rng'], dict)
                or set(state['validation_rng']) != {'python', 'numpy', 'torch', 'cuda'}
                or not isinstance(state['scaler'], dict)):
            raise ValueError('AugDiff completed checkpoint lost its validation RNG or AMP state')
        attempts, updates = 0, 0
        for order, windows in zip(state['train_orders'], state['window_history']):
            if Counter(order) != expected or len(windows) != (len(order)+1)//2:
                raise ValueError('AugDiff checkpoint must include seven complete TRAIN-only passes and all windows')
            if [slide for window in windows for slide in window['slides']] != order:
                raise ValueError('AugDiff checkpoint gradient windows differ from actual TRAIN order')
            for index, window in enumerate(windows):
                wanted = min(2, len(order)-index*2)
                if (set(window) != {'slides', 'divisor', 'stepped'} or len(window['slides']) != wanted
                        or window['divisor'] != 2 or type(window['stepped']) is not bool):
                    raise ValueError('AugDiff checkpoint accumulation/tail/AMP step audit differs')
                attempts += 1; updates += int(window['stepped'])
        if state['optimizer_updates'] != updates or not 0 <= updates <= attempts:
            raise ValueError('AugDiff checkpoint actual optimizer-update count differs')
        for key in ['epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders', 'window_history',
                    'validation_epoch', 'validation_rng']:
            setattr(self, key, copy.deepcopy(state[key]))
        self._pending_scaler = copy.deepcopy(state['scaler'])
        if hasattr(self, 'scaler'):
            self.scaler.load_state_dict(self._pending_scaler)

    def protocol(self):
        return {'method': self.profile, 'source': COMMIT, 'ldm': LDM_COMMIT, 'layout': LAYOUT,
            'settings': self.settings, 'prepared_dae_contract': self.assets.contract,
            'dae': 'actual full256base/1,2,4levels/2resblocks/allattention/1000step identity-first-stage epsilon DAE',
            'full_dae_parameters': self.assets.metadata['pretrain_protocol']['full_dae_parameters'],
            'autocast_checkpoint_port': 'Restore original forward autocast device/enabled/dtype/cache when recomputing checkpointed attention in backward; no skipped blocks or precision fallback',
            'dae_inference': 'frozen selected RAW weights, no EMA inference; no DAE parameters in MIL optimizer',
            'training': 'complete original classifier CE; each TRAIN bag repeated7, one bag at a time, accumulation2 including taildivisor2',
            'optimizer': 'actual source RAdam and Lookahead alpha0.5/k6; source bias/1D zero-decay groups; constant LR',
            'sampling': 'T20/30 nu0.2/0.4 eta0 semantic feature forward diffusion then DDIM; no random generation or row truncation',
            'timestep_variant': 'explicit per-step repaired schedule' if self.repaired else 'released stale timestep control',
            'validation': 'seven stochastic draws per slide; original mean draw CE selects first strict minimum; patience10/defaultmax200',
            'statistics_port': 'mean7probabilities, one row per true slide; noisy replicas do not become independent cases',
            'test': 'single original clean feature bag; no diffusion augmentation or DAE allocation',
            'source_deficits': 'missing DataInterface and actual feature UNet/weights; explicit full-width feature-layout reconstruction and workers0 sampling port, no historical cold-start/prefetch claim',
            'numerical_port': 'stable float32 logsoftmax CE for validation instead of half softmax/log/CPU list conversion',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'train_orders_sha256': digest_object(self.train_orders), 'gradient_windows_sha256': digest_object(self.window_history),
            'checkpoint': 'classifier/fullRAdam+portableLookahead/AMP/windoworders/validationstream/allRNG and immutable nativeDAE asset contract',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

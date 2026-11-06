"""Complete released HDMIL: DMIN self-distillation then frozen-teacher LIPN."""
import copy
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler

from .hdmil_source import COMMIT, RECIPES, build_dmin, build_lipn, dmin_loss, lipn_loss
from .io import digest_object


def tensor_state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode()); result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class Slides(Dataset):
    def __init__(self, assets, rows):
        self.assets, self.rows = assets, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return *self.assets.load(row), row['target'], index


class HDMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='hdmil_brca_source', dmin_epochs=200, lipn_epochs=200,
                 dmin_selection_start=50, dmin_patience=20, lipn_patience=5, lipn_lr=1e-4):
        super().__init__()
        if (profile not in RECIPES or bundle.task_type != 'classification'
                or bundle.config.get('training_unit', 'slide') != 'slide' or bundle.hdmil is None
                or len(bundle.classes) != RECIPES[profile]['classes']):
            raise ValueError('HDMIL requires registered paired RGB/features, slide classification and the selected source class count')
        if (any(type(v) is not int or v < 1 for v in
                [dmin_epochs, lipn_epochs, dmin_selection_start, dmin_patience, lipn_patience])
                or dmin_selection_start > dmin_epochs or not np.isfinite(lipn_lr) or lipn_lr <= 0):
            raise ValueError('HDMIL must include both complete declared stages and DMIN selection before transfer')
        self.bundle, self.assets, self.profile = bundle, bundle.hdmil, profile
        self.stage_config = dict(dmin_epochs=dmin_epochs, lipn_epochs=lipn_epochs,
            dmin_selection_start=dmin_selection_start, dmin_patience=dmin_patience,
            lipn_patience=lipn_patience, lipn_lr=lipn_lr)
        self.native = RECIPES[profile].copy()
        if any(r['nodes'] < self.native['k_sample'] for r in self.assets.meta['slides'].values()):
            raise ValueError('HDMIL original top-k instance loss requires enough genuine patches; no repeat padding')
        self.dmin = build_dmin(profile, bundle.dim)
        # Keep checkpoint ownership fixed while preserving stage-one source RNG.
        from .engine import rng_state, restore_rng
        rng = rng_state()
        self.teacher = build_dmin(profile, bundle.dim, 'lipn')
        self.processor = build_lipn(profile, bundle.dim)
        restore_rng(rng)
        self.contract = digest_object(dict(source=COMMIT, bundle=bundle.fingerprint,
            assets=self.assets.contract, profile=profile, stage_config=self.stage_config))
        self.stage, self.stage_epoch, self.completed_epochs = 'dmin', 0, 0
        self.epoch_index, self.stage_seed, self.recipe = -1, None, None
        self.transfer_pending, self.training_done = False, False
        self.best_dmin, self.best_dmin_hash, self.best_dmin_epoch = None, None, None
        self.stage_best, self.stage_bad = dict(dmin=None, lipn=None), dict(dmin=0, lipn=0)
        self.updates, self.train_orders, self.selection_history = dict(dmin=0, lipn=0), [], []
        self.apply_phase()

    def apply_phase(self):
        for p in self.dmin.parameters():
            p.requires_grad_(self.stage == 'dmin')
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for p in self.processor.parameters():
            p.requires_grad_(self.stage == 'lipn')

    def configure_training(self, cfg):
        if (cfg['epochs'] != self.stage_config['dmin_epochs']+self.stage_config['lipn_epochs']
                or cfg['grad_clip'] is not None or cfg['max_instances'] is not None
                or cfg['selection_metric'] != 'hdmil_source_ce'
                or cfg['patience'] != self.stage_config['dmin_patience']):
            raise ValueError('HDMIL requires declared two-stage budgets, original selection and unsampled full bags')
        self.recipe = {k: cfg[k] for k in ['epochs', 'patience', 'lr', 'weight_decay', 'selection_metric']}

    def prepare(self, bundle, seed, device):
        if bundle.fingerprint != self.bundle.fingerprint or type(seed) is not int:
            raise ValueError('HDMIL preparation must bind the actual task and seed')
        self.stage_seed = seed

    def make_optimizers(self, lr, weight_decay):
        self.optimizers = [torch.optim.Adam(self.dmin.parameters(), lr=lr, weight_decay=weight_decay),
                           torch.optim.Adam(self.processor.parameters(), lr=self.stage_config['lipn_lr'], weight_decay=weight_decay)]
        return self.optimizers

    def validate_optimizer_state(self):
        for optimizer, module, lr in zip(self.optimizers, [self.dmin, self.processor],
                [self.recipe['lr'], self.stage_config['lipn_lr']]):
            groups = optimizer.param_groups
            if (len(groups) != 1 or groups[0]['lr'] != lr or groups[0]['weight_decay'] != self.recipe['weight_decay']
                    or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                    or {id(p) for p in groups[0]['params']} != {id(p) for p in module.parameters()}):
                raise ValueError('HDMIL complete stage Adam state or parameter ownership changed')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs or self.training_done:
            raise ValueError('HDMIL epoch must continue the saved complete stage state')
        if self.transfer_pending:
            if self.best_dmin is None or tensor_state_hash(self.best_dmin) != self.best_dmin_hash:
                raise ValueError('HDMIL LIPN requires the actual validation-selected DMIN teacher')
            from .engine import seed_everything
            # Author runs the two scripts separately with the same seed. Match
            # the stage-two constructor order before loading selected weights.
            seed_everything(self.stage_seed)
            fresh_teacher = build_dmin(self.profile, self.bundle.dim, 'lipn')
            fresh_processor = build_lipn(self.profile, self.bundle.dim)
            fresh_teacher.load_state_dict(self.best_dmin, strict=True)
            self.teacher.load_state_dict(fresh_teacher.state_dict(), strict=True)
            self.processor.load_state_dict(fresh_processor.state_dict(), strict=True)
            self.optimizers[1].state.clear()
            self.stage, self.stage_epoch, self.transfer_pending = 'lipn', 0, False
            self.apply_phase()
        self.epoch_index = epoch

    def _loader(self, subset):
        rows = self.bundle.records(subset)
        dataset = Slides(self.assets, rows)
        if subset == 'train':
            counts = np.bincount([r['target'] for r in rows], minlength=len(self.bundle.classes))
            weights = torch.tensor([len(rows)/counts[r['target']] for r in rows], dtype=torch.float64)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        else:
            sampler = SequentialSampler(dataset)
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0), rows

    def epoch(self, bundle, optimizers, device):
        if bundle.fingerprint != self.bundle.fingerprint or any(a is not b for a, b in zip(optimizers, self.optimizers)):
            raise ValueError('HDMIL task or stage optimizer changed')
        self.train(); loader, rows = self._loader('train')
        optimizer = optimizers[int(self.stage == 'lipn')]
        losses, order = [], []
        for features, images, labels, index in loader:
            features, images, labels = features.to(device), images.to(device), labels.to(device)
            optimizer.zero_grad()
            if self.stage == 'dmin':
                loss, components = dmin_loss(self.dmin, features, labels, self.native['mask_ratio'])
            else:
                loss, components = lipn_loss(self.teacher, self.processor, features, images)
            if not torch.isfinite(loss):
                raise FloatingPointError('HDMIL original objective is nonfinite; source NaN skipping is not accepted as a complete run')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params']):
                raise FloatingPointError('HDMIL source stage produced nonfinite gradients')
            optimizer.step()
            self.updates[self.stage] += 1
            order.append(rows[int(index)]['slide_id'])
            losses.append(dict(loss=float(loss.detach()), **{k: float(v.detach()) for k, v in components.items()}))
        self.stage_epoch += 1; self.completed_epochs += 1
        self.train_orders.append(dict(stage=self.stage, slides=order))
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('HDMIL prediction requires registered validation/test inputs')
        if subset == 'test' and self.stage != 'lipn':
            raise ValueError('HDMIL full test prediction requires completed LIPN training')
        self.eval(); loader, rows = self._loader(subset); records = []
        for features, images, labels, index in loader:
            features, images, labels = features.to(device), images.to(device), labels.to(device)
            logits = (self.dmin(features) if self.stage == 'dmin' else
                      self.teacher(features, self.processor(images))[0])
            if not torch.isfinite(logits).all():
                raise FloatingPointError('HDMIL source inference produced nonfinite logits')
            row = rows[int(index)]
            record = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], target=row['target'],
                prediction=int(logits.argmax(1)), hdmil_source_ce=float(F.cross_entropy(logits, labels)))
            record.update({f'p_{i}': float(v) for i, v in enumerate(logits.softmax(1)[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        if not len(frame) or not frame.slide_id.is_unique or not np.isfinite(frame.hdmil_source_ce).all():
            raise ValueError('HDMIL requires finite mean slide CE on the entire validation partition')
        return dict(hdmil_source_ce=float(frame.hdmil_source_ce.mean()))

    def after_validation(self, metrics):
        value = metrics['hdmil_source_ce']
        eligible = self.stage != 'dmin' or self.stage_epoch >= self.stage_config['dmin_selection_start']
        if eligible:
            best = self.stage_best[self.stage]
            if best is None or value < best:
                self.stage_best[self.stage], self.stage_bad[self.stage] = value, 0
                if self.stage == 'dmin':
                    self.best_dmin = snapshot(self.dmin)
                    self.best_dmin_hash = tensor_state_hash(self.best_dmin)
                    self.best_dmin_epoch = self.stage_epoch
            else:
                self.stage_bad[self.stage] += 1
        done = (self.stage_epoch == self.stage_config[self.stage+'_epochs'] or
                self.stage_bad[self.stage] >= self.stage_config[self.stage+'_patience'])
        self.transfer_pending = done and self.stage == 'dmin'
        self.training_done = done and self.stage == 'lipn'
        self.selection_history.append(dict(epoch=self.epoch_index, stage=self.stage, stage_epoch=self.stage_epoch,
            value=value, eligible=eligible, best=self.stage_best[self.stage], bad=self.stage_bad[self.stage], done=done))

    def selection_objective(self, metrics, epoch, current):
        return -metrics['hdmil_source_ce']

    def selection_improved(self, objective, best):
        return self.stage == 'lipn' and objective > best

    def get_extra_state(self):
        return copy.deepcopy({k: getattr(self, k) for k in ['contract', 'recipe', 'stage_seed', 'stage', 'stage_epoch',
            'completed_epochs', 'epoch_index', 'transfer_pending', 'training_done', 'best_dmin', 'best_dmin_hash',
            'best_dmin_epoch', 'stage_best', 'stage_bad', 'updates', 'train_orders', 'selection_history']})

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) != set(self.get_extra_state()) or state['contract'] != self.contract
                or state['stage'] not in {'dmin', 'lipn'} or type(state['stage_seed']) is not int
                or any(type(state[k]) is not int for k in ['stage_epoch', 'completed_epochs', 'epoch_index'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or len(state['train_orders']) != state['completed_epochs']
                or len(state['selection_history']) != state['completed_epochs']
                or any(type(state[k]) is not bool for k in ['transfer_pending', 'training_done'])):
            raise ValueError('HDMIL source, task, seed or complete stage history changed')
        if self.recipe is not None and self.recipe != state['recipe']:
            raise ValueError('HDMIL resumed two-stage recipe changed')
        if state['best_dmin'] is not None and tensor_state_hash(state['best_dmin']) != state['best_dmin_hash']:
            raise ValueError('HDMIL saved selected DMIN tensor identity changed')
        train = {r['slide_id'] for r in self.bundle.records('train')}
        updates = dict(dmin=0, lipn=0); epochs = dict(dmin=0, lipn=0)
        best, bad, done, phase = dict(dmin=None, lipn=None), dict(dmin=0, lipn=0), False, 'dmin'
        best_epoch = None
        for index, (order, record) in enumerate(zip(state['train_orders'], state['selection_history'])):
            if done:
                if phase == 'lipn':
                    raise ValueError('HDMIL checkpoint continued past its declared completed stage')
                phase, done = 'lipn', False
            epochs[phase] += 1
            if order.get('stage') != phase or len(order['slides']) != len(train) or not set(order['slides']) <= train:
                raise ValueError('HDMIL weighted replacement sample membership changed')
            updates[phase] += len(train)
            value = record.get('value')
            if type(value) not in {int, float} or not np.isfinite(value) or value < 0:
                raise ValueError('HDMIL complete validation CE history is invalid')
            eligible = phase != 'dmin' or epochs[phase] >= self.stage_config['dmin_selection_start']
            if eligible:
                if best[phase] is None or value < best[phase]:
                    best[phase], bad[phase] = value, 0
                    if phase == 'dmin':
                        best_epoch = epochs[phase]
                else:
                    bad[phase] += 1
            done = epochs[phase] == self.stage_config[phase+'_epochs'] or bad[phase] >= self.stage_config[phase+'_patience']
            if record != dict(epoch=index, stage=phase, stage_epoch=epochs[phase], value=value,
                              eligible=eligible, best=best[phase], bad=bad[phase], done=done):
                raise ValueError('HDMIL source first-minimum selection or stopping trace changed')
        if (state['updates'] != updates or state['stage_best'] != best or state['stage_bad'] != bad
                or state['stage'] != phase or state['stage_epoch'] != epochs[phase]
                or state['transfer_pending'] != (done and phase == 'dmin')
                or state['training_done'] != (done and phase == 'lipn') or state['best_dmin_epoch'] != best_epoch
                or (best_epoch is not None) != (state['best_dmin'] is not None)):
            raise ValueError('HDMIL stage transition, teacher selection or actual Adam counts changed')
        for key in set(state)-{'contract'}:
            setattr(self, key, copy.deepcopy(state[key]))
        self.apply_phase()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.stage == 'lipn' and tensor_state_hash(self.teacher.state_dict()) != self.best_dmin_hash:
            raise ValueError('HDMIL frozen LIPN teacher differs from its selected DMIN')
        return result

    def protocol(self):
        return dict(method=self.profile, source=COMMIT, input_dimension=self.bundle.dim, native=self.native,
            architecture='complete native512/256 gated attention, degree12/16 Chebyshev KAN bag+instance heads; original MobileNetV4 tiny RGB processor',
            stages=self.stage_config, stage=self.stage, recipe=self.recipe, completed_epochs=self.completed_epochs,
            actual_optimizer_updates=self.updates.copy(), selected_dmin_epoch=self.best_dmin_epoch,
            selected_dmin_sha256=self.best_dmin_hash, train_orders_sha256=digest_object(self.train_orders),
            source_objectives='DMIN: .7CE+.3SmoothTop1 instance +.5KL+.5representationMSE+2maskMSE; LIPN: L1 hard straight-through masks + union-rate MSE',
            preserved_source_details='training attention uses global normalizer; retention uses first2classes even RCC; LIPN teacher remains frozen but TRAIN-mode dropout active; source stage2 separately resets same seed',
            scope='complete released cached-feature and matched-low-RGB training/inference, not an online high-resolution FM extraction speed benchmark',
            source_recipe_conflict='task README defaults used; run_scripts.sh stale mask_ratio0.1 differs',
            numerical_guard='nonfinite objective/gradient fails explicitly instead of silently skipping source NaN batches',
            real_wsi_validation='not_run', paper_reproduction='not_run')

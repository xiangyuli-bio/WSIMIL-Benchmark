"""Complete DMIN preparation followed by original AHDMIL joint training."""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

from .ahdmil_source import COMMIT, RECIPES, build_joint, dmin_objective, inference, lipn_objective
from .hdmil import Slides, snapshot, tensor_state_hash
from .hdmil_source import COMMIT as PRETRAIN_COMMIT, build_dmin, dmin_loss
from .io import digest_object


class AHDMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='ahdmil_brca_source_repaired', dmin_epochs=200,
                 joint_epochs=200, dmin_selection_start=50, dmin_patience=20,
                 joint_patience=5, joint_lr=1e-5):
        super().__init__()
        if (profile not in RECIPES or bundle.task_type != 'classification'
                or bundle.config.get('training_unit', 'slide') != 'slide' or bundle.hdmil is None
                or len(bundle.classes) != RECIPES[profile]['classes']):
            raise ValueError('AHDMIL requires real registered paired RGB/features and the source task class count')
        if (any(type(v) is not int or v < 1 for v in [dmin_epochs, joint_epochs,
                dmin_selection_start, dmin_patience, joint_patience]) or dmin_selection_start > dmin_epochs
                or not np.isfinite(joint_lr) or joint_lr <= 0):
            raise ValueError('AHDMIL requires both declared full stages and an eligible pretrained DMIN')
        self.bundle, self.assets, self.profile = bundle, bundle.hdmil, profile
        self.native = RECIPES[profile].copy()
        if any(row['nodes'] < self.native['k_sample'] for row in self.assets.meta['slides'].values()):
            raise ValueError('AHDMIL pretraining needs enough genuine patches for original instance top-k')
        self.stage_config = dict(dmin_epochs=dmin_epochs, joint_epochs=joint_epochs,
            dmin_selection_start=dmin_selection_start, dmin_patience=dmin_patience,
            joint_patience=joint_patience, joint_lr=joint_lr)
        self.pretrain = build_dmin(self.native['pretrain_profile'], bundle.dim)
        from .engine import rng_state, restore_rng
        rng = rng_state()
        self.dmin, self.lipn = build_joint(profile, bundle.dim)
        restore_rng(rng)
        self.contract = digest_object(dict(source=COMMIT, pretrain_source=PRETRAIN_COMMIT,
            profile=profile, task=bundle.fingerprint, assets=self.assets.contract, stages=self.stage_config))
        self.recipe, self.stage_seed = None, None
        self.stage, self.stage_epoch, self.completed_epochs, self.epoch_index = 'dmin', 0, 0, -1
        self.transfer_pending, self.training_done = False, False
        self.selected_dmin, self.selected_dmin_hash, self.selected_dmin_epoch = None, None, None
        self.stage_best, self.stage_bad = dict(dmin=None, joint=None), dict(dmin=0, joint=0)
        self.lower_ratio, self.upper_ratio = .3, 1.
        self.joint_generation, self.generation_updates, self.ema_calls = 0, 0, 0
        self.updates = dict(pretrain=0, lipn=0, dmin=0)
        self.train_orders, self.selection_history, self.reinitializations = [], [], []
        self.apply_phase()

    def apply_phase(self):
        self.pretrain.requires_grad_(self.stage == 'dmin')
        self.dmin.requires_grad_(self.stage == 'joint')
        self.lipn.requires_grad_(self.stage == 'joint')

    def configure_training(self, cfg):
        if (cfg['epochs'] != self.stage_config['dmin_epochs']+self.stage_config['joint_epochs']
                or cfg['grad_clip'] is not None or cfg['max_instances'] is not None
                or cfg['selection_metric'] != 'ahdmil_source_ce' or cfg['patience'] != self.stage_config['dmin_patience']):
            raise ValueError('AHDMIL requires both budgets, full bags and original stage selection')
        self.recipe = {k: cfg[k] for k in ['epochs', 'patience', 'lr', 'weight_decay', 'selection_metric']}

    def prepare(self, bundle, seed, device):
        if bundle.fingerprint != self.bundle.fingerprint or type(seed) is not int:
            raise ValueError('AHDMIL must bind the registered task and seed')
        self.stage_seed = seed

    def make_optimizers(self, lr, weight_decay):
        self.optimizers = [torch.optim.Adam(module.parameters(), lr=rate, weight_decay=weight_decay)
            for module, rate in [(self.pretrain, lr), (self.lipn, self.stage_config['joint_lr']),
                                 (self.dmin, self.stage_config['joint_lr'])]]
        return self.optimizers

    def validate_optimizer_state(self):
        for optimizer, module, rate in zip(self.optimizers, [self.pretrain, self.lipn, self.dmin],
                [self.recipe['lr'], self.stage_config['joint_lr'], self.stage_config['joint_lr']]):
            groups = optimizer.param_groups
            if (len(groups) != 1 or groups[0]['lr'] != rate or groups[0]['weight_decay'] != self.recipe['weight_decay']
                    or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                    or {id(p) for p in groups[0]['params']} != {id(p) for p in module.parameters()}):
                raise ValueError('AHDMIL full Adam hyperparameters or parameter ownership changed')

    def _reset_joint(self):
        if self.selected_dmin is None or tensor_state_hash(self.selected_dmin) != self.selected_dmin_hash:
            raise ValueError('AHDMIL must initialize from the actual VAL-selected complete DMIN')
        fresh_dmin, fresh_lipn = build_joint(self.profile, self.bundle.dim)
        fresh_dmin.load_state_dict(self.selected_dmin, strict=True)
        self.dmin.load_state_dict(fresh_dmin.state_dict(), strict=True)
        self.lipn.load_state_dict(fresh_lipn.state_dict(), strict=True)
        self.optimizers[1].state.clear(); self.optimizers[2].state.clear()
        self.generation_updates = 0

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs or self.training_done:
            raise ValueError('AHDMIL must continue the saved full stage state')
        if self.transfer_pending:
            from .engine import seed_everything
            seed_everything(self.stage_seed)
            self._reset_joint()
            self.stage, self.stage_epoch, self.transfer_pending = 'joint', 0, False
            self.lipn.cross_EMA(); self.ema_calls += 1
            self.apply_phase()
        self.epoch_index = epoch

    def _loader(self, subset):
        rows = self.bundle.records(subset); dataset = Slides(self.assets, rows)
        if subset == 'train':
            counts = np.bincount([row['target'] for row in rows], minlength=len(self.bundle.classes))
            weights = torch.tensor([len(rows)/counts[row['target']] for row in rows], dtype=torch.float64)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        else:
            sampler = SequentialSampler(dataset)
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0), rows

    @staticmethod
    def _update(loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError('AHDMIL nonfinite objective; source skipped batches are not counted as completed updates')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise FloatingPointError('AHDMIL nonfinite gradient')
        optimizer.step()

    def epoch(self, bundle, optimizers, device):
        if bundle.fingerprint != self.bundle.fingerprint or len(optimizers) != 3 or any(a is not b for a,b in zip(optimizers,self.optimizers)):
            raise ValueError('AHDMIL task or all three optimizers changed')
        self.train(); loader, rows = self._loader('train'); losses, order, manners = [], [], []
        for features, images, labels, index in loader:
            features, images, labels = features.to(device), images.to(device), labels.to(device)
            if self.stage == 'dmin':
                optimizers[0].zero_grad()
                loss, components = dmin_loss(self.pretrain, features, labels, self.native['mask_ratio'])
                self._update(loss, optimizers[0]); self.updates['pretrain'] += 1
            else:
                optimizers[1].zero_grad()
                lipn_loss, components, manner = lipn_objective(self.dmin, self.lipn,
                    features, images, self.native['attn_ratio'])
                self._update(lipn_loss, optimizers[1]); self.updates['lipn'] += 1
                optimizers[2].zero_grad()
                dmin_ce = dmin_objective(self.dmin, self.lipn, features, images, labels)
                self._update(dmin_ce, optimizers[2]); self.updates['dmin'] += 1
                self.generation_updates += 1; manners.append(manner)
                loss = lipn_loss.detach()+dmin_ce.detach()
                components.update(lipn_loss=lipn_loss, dmin_ce=dmin_ce)
            order.append(rows[int(index)]['slide_id'])
            losses.append(dict(loss=float(loss.detach()), **{k:float(v.detach()) for k,v in components.items()}))
        if self.stage == 'joint':
            self.lipn.cross_EMA(); self.ema_calls += 1
        self.stage_epoch += 1; self.completed_epochs += 1
        self.train_orders.append(dict(stage=self.stage, slides=order, manners=manners,
                                      joint_generation=self.joint_generation))
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val','test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('AHDMIL prediction requires the registered held-out partition')
        if subset == 'test' and self.stage != 'joint':
            raise ValueError('AHDMIL test prediction requires a trained joint model')
        self.eval(); loader, rows = self._loader(subset); records = []
        for features, images, labels, index in loader:
            features, images, labels = features.to(device), images.to(device), labels.to(device)
            if self.stage == 'dmin':
                logits, rate = self.pretrain(features), None
            else:
                logits, rate = inference(self.dmin, self.lipn, features, images)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('AHDMIL nonfinite prediction')
            row = rows[int(index)]
            record = {k:row[k] for k in ['slide_id','patient_id','global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], target=row['target'],
                prediction=int(logits.argmax(1)), ahdmil_source_ce=float(F.cross_entropy(logits,labels)))
            if rate is not None:
                record['ahdmil_retention'] = float(rate)
            record.update({f'p_{i}':float(v) for i,v in enumerate(logits.softmax(1)[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        if not len(frame) or not frame.slide_id.is_unique or not np.isfinite(frame.ahdmil_source_ce).all():
            raise ValueError('AHDMIL requires finite mean CE on the full validation partition')
        result = dict(ahdmil_source_ce=float(frame.ahdmil_source_ce.mean()))
        if self.stage == 'joint':
            if not np.isfinite(frame.ahdmil_retention).all() or not frame.ahdmil_retention.between(0,1).all():
                raise ValueError('AHDMIL retention must be the actual selected patch fraction')
            result['ahdmil_retention'] = float(frame.ahdmil_retention.mean())
        return result

    def after_validation(self, metrics):
        value = metrics['ahdmil_source_ce']; rate = metrics.get('ahdmil_retention')
        if not np.isfinite(value) or value < 0:
            raise ValueError('AHDMIL validation CE must be finite and nonnegative')
        lower_before = self.lower_ratio
        eligible = (self.stage_epoch >= self.stage_config['dmin_selection_start'] if self.stage == 'dmin'
                    else rate is not None and np.isfinite(rate) and self.lower_ratio <= rate <= self.upper_ratio)
        best = self.stage_best[self.stage]
        improved = eligible and (best is None or value < best)
        if improved:
            self.stage_best[self.stage], self.stage_bad[self.stage] = value, 0
            if self.stage == 'dmin':
                self.selected_dmin = snapshot(self.pretrain)
                self.selected_dmin_hash, self.selected_dmin_epoch = tensor_state_hash(self.selected_dmin), self.stage_epoch
        elif self.stage == 'joint' or eligible:
            self.stage_bad[self.stage] += 1
        reset = False
        stopped = self.stage_bad[self.stage] >= self.stage_config[self.stage+'_patience']
        if self.stage == 'joint' and stopped and self.stage_best['joint'] is None:
            reset = True; stopped = False; self.stage_bad['joint'] = 0
            self._reset_joint(); self.joint_generation += 1
            # Explicit repair of original args.lower_ratio typo. Preserve the
            # original decrement and epoch condition, including no clipping.
            if self.native['task'] == 'rcc' and self.stage_epoch > 50:
                self.lower_ratio -= .05
            self.reinitializations.append(dict(epoch=self.epoch_index, joint_epoch=self.stage_epoch,
                generation=self.joint_generation, lower_before=lower_before, lower_after=self.lower_ratio))
        done = stopped or self.stage_epoch == self.stage_config[self.stage+'_epochs']
        self.transfer_pending = done and self.stage == 'dmin'
        self.training_done = done and self.stage == 'joint'
        self.selection_history.append(dict(epoch=self.epoch_index, stage=self.stage, stage_epoch=self.stage_epoch,
            value=value, rate=rate, eligible=bool(eligible), improved=bool(improved), best=self.stage_best[self.stage],
            bad=self.stage_bad[self.stage], lower_before=lower_before, lower_after=self.lower_ratio,
            reset=reset, generation=self.joint_generation, done=done))
        if self.training_done and self.stage_best['joint'] is None:
            raise RuntimeError('AHDMIL full joint budget ended without a source-eligible checkpoint; no forced best selection')

    def selection_objective(self, metrics, epoch, current):
        return -metrics['ahdmil_source_ce']

    def selection_improved(self, objective, best):
        return self.stage == 'joint' and self.selection_history[-1]['eligible'] and objective > best

    def get_extra_state(self):
        return copy.deepcopy({k:getattr(self,k) for k in ['contract','recipe','stage_seed','stage','stage_epoch',
            'completed_epochs','epoch_index','transfer_pending','training_done','selected_dmin','selected_dmin_hash',
            'selected_dmin_epoch','stage_best','stage_bad','lower_ratio','upper_ratio','joint_generation',
            'generation_updates','ema_calls','updates','train_orders','selection_history','reinitializations']})

    def set_extra_state(self, state):
        if (not isinstance(state,dict) or set(state) != set(self.get_extra_state()) or state['contract'] != self.contract
                or state['stage'] not in {'dmin','joint'} or type(state['stage_seed']) is not int
                or type(state['completed_epochs']) is not int or state['completed_epochs'] < 0
                or state['epoch_index'] != state['completed_epochs']-1
                or len(state['train_orders']) != state['completed_epochs'] or len(state['selection_history']) != state['completed_epochs']
                or state['upper_ratio'] != 1. or any(type(state[k]) is not bool for k in ['transfer_pending','training_done'])):
            raise ValueError('AHDMIL source, task or full stage history changed')
        if self.recipe is not None and self.recipe != state['recipe']:
            raise ValueError('AHDMIL resumed recipe changed')
        if state['selected_dmin'] is not None and tensor_state_hash(state['selected_dmin']) != state['selected_dmin_hash']:
            raise ValueError('AHDMIL selected pretrained weights were changed')
        train = {row['slide_id'] for row in self.bundle.records('train')}
        phase, done, lower, generation, generation_updates = 'dmin', False, .3, 0, 0
        best, bad, epochs = dict(dmin=None,joint=None), dict(dmin=0,joint=0), dict(dmin=0,joint=0)
        selected_epoch, resets = None, []
        for index,(order,record) in enumerate(zip(state['train_orders'],state['selection_history'])):
            if done:
                if phase == 'joint':
                    raise ValueError('AHDMIL continued beyond its complete joint budget')
                phase, done = 'joint', False
            epochs[phase] += 1
            manners = order.get('manners',[])
            if (order.get('stage') != phase or len(order['slides']) != len(train) or not set(order['slides']) <= train
                    or order.get('joint_generation') != generation
                    or (phase == 'dmin' and manners) or (phase == 'joint' and (len(manners) != len(train)
                    or not set(manners) <= {'attn_train','hard_train'}
                    or (self.native['attn_ratio'] == 0 and set(manners) != {'hard_train'})))):
                raise ValueError('AHDMIL source sample order or distillation branch history changed')
            value, rate = record.get('value'), record.get('rate')
            if type(value) not in {int,float} or not np.isfinite(value) or value < 0:
                raise ValueError('AHDMIL invalid validation CE trace')
            if phase == 'joint' and (type(rate) not in {int,float} or not np.isfinite(rate) or not 0 <= rate <= 1):
                raise ValueError('AHDMIL invalid actual retention trace')
            if phase == 'dmin' and rate is not None:
                raise ValueError('AHDMIL pretraining trace must not invent LIPN retention')
            before = lower
            eligible = epochs[phase] >= self.stage_config['dmin_selection_start'] if phase == 'dmin' else lower <= rate <= 1
            improved = eligible and (best[phase] is None or value < best[phase])
            if improved:
                best[phase],bad[phase] = value,0
                if phase == 'dmin':selected_epoch = epochs[phase]
            elif phase == 'joint' or eligible:bad[phase] += 1
            if phase == 'joint':generation_updates += len(train)
            stopped = bad[phase] >= self.stage_config[phase+'_patience']; reset = False
            if phase == 'joint' and stopped and best[phase] is None:
                reset,stopped = True,False;bad[phase] = 0;generation += 1;generation_updates = 0
                if self.native['task'] == 'rcc' and epochs[phase] > 50:lower -= .05
                resets.append(dict(epoch=index,joint_epoch=epochs[phase],generation=generation,lower_before=before,lower_after=lower))
            done = stopped or epochs[phase] == self.stage_config[phase+'_epochs']
            expected = dict(epoch=index,stage=phase,stage_epoch=epochs[phase],value=value,rate=rate,
                eligible=bool(eligible),improved=bool(improved),best=best[phase],bad=bad[phase],lower_before=before,
                lower_after=lower,reset=reset,generation=generation,done=done)
            if record != expected:
                raise ValueError('AHDMIL checkpoint changed source selection, reset or stopping trace')
        updates = dict(pretrain=epochs['dmin']*len(train),lipn=epochs['joint']*len(train),dmin=epochs['joint']*len(train))
        if (state['updates'] != updates or state['stage'] != phase or state['stage_epoch'] != epochs[phase]
                or state['stage_best'] != best or state['stage_bad'] != bad or state['lower_ratio'] != lower
                or state['joint_generation'] != generation or state['generation_updates'] != generation_updates
                or state['ema_calls'] != (epochs['joint']+1 if epochs['joint'] else 0)
                or state['reinitializations'] != resets or state['selected_dmin_epoch'] != selected_epoch
                or (selected_epoch is not None) != (state['selected_dmin'] is not None)
                or state['transfer_pending'] != (done and phase == 'dmin')
                or state['training_done'] != (done and phase == 'joint')):
            raise ValueError('AHDMIL optimizer counts, selected pretraining or full stage transition changed')
        for key in set(state)-{'contract'}:
            setattr(self,key,copy.deepcopy(state[key]))
        self.apply_phase()

    def protocol(self):
        return dict(method=self.profile, source=COMMIT, pretraining_source=PRETRAIN_COMMIT,
            native=self.native, stages=self.stage_config, recipe=self.recipe, stage=self.stage,
            architecture='native512/256 gated ChebyKAN12/16 plus two full RGB16 MobileNetV4 experts',
            actual_optimizer_updates=self.updates.copy(), completed_epochs=self.completed_epochs,
            selected_pretrained_epoch=self.selected_dmin_epoch, selected_pretrained_sha256=self.selected_dmin_hash,
            cross_ema_calls=self.ema_calls, reinitializations=copy.deepcopy(self.reinitializations),
            native_selection='strict meanVALCE and0.3..1 retention; no eligible best triggers reset; RCC after50 decrements lower bound0.05',
            source_repairs=['baseline AHDMIL/HADMIL import name','RCC args.lower_ratio changed to self.lower_ratio'],
            preserved_details='train-mode teacher dropout and second expert BN update; raw expert logit average then hard mask; simultaneous parameter-only cross-EMA; all-class RCC union',
            train_orders_sha256=digest_object(self.train_orders),
            scope='automatic complete original HDMIL DMIN preparation then complete source AHDMIL joint recipe; cached high features and measured matched low RGB',
            real_wsi_validation='not_run', paper_reproduction='not_run')

"""Complete HR teacher -> patch encoder KD -> LR student source recipe."""
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import atomic_torch_save, digest_file, digest_object, read_json, write_json
from .lrmil_source import (COMMIT, author_components, native_backbone, normalize_rgb,
                           patch_objective, student_objective, teacher_metadata)

PHASES = ('teacher', 'patch', 'student')


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def collate_list(items):
    return items


class PatchExamples(Dataset):
    def __init__(self, assets, rows, metadata):
        self.assets, self.rows, self.metadata = assets, rows, metadata
        self.pairs = [(i, j) for i, row in enumerate(rows)
                      for j in range(assets.meta['slides'][row['slide_id']]['low_nodes'])]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row_index, patch_index = self.pairs[index]
        row = self.rows[row_index]
        data = self.assets.patch(row, patch_index)
        return {'img5': normalize_rgb(data['image'][None])[0],
            'coords_5x': data['coords_5x'], 'coords_20x': data['coords_20x'],
            'teacher_final': data['teacher_final'], 'teacher_layers': data['teacher_layers'],
            'teacher_attn': self.metadata[row['slide_id']]['attention'][data['links']],
            'unit': row['slide_id']+':'+str(patch_index)}


class LRMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True
    STATE_KEYS = ('contract', 'recipe', 'run_seed', 'phase', 'phase_epoch', 'epoch_index',
        'completed_epochs', 'transfer_pending', 'training_done', 'phase_best', 'phase_bad',
        'updates', 'student_instance_updates', 'best_teacher', 'best_teacher_hash',
        'best_encoder', 'best_encoder_hash', 'selected_epochs', 'train_orders',
        'selection_history', 'last_patch_loss')

    def __init__(self, bundle, profile='lrmil_source_repaired',
                 stage_epochs=(200, 200, 200), patch_batch_size=64, num_workers=0,
                 inference_batch_size=32, gated_attention=False,
                 lambda_layer=1., lambda_attn=1., temperature_bag=1.,
                 temperature_attn=1., lambda_bag=.5, lambda_soft=.5, lambda_hard=.3,
                 topk=4):
        super().__init__()
        if (profile != 'lrmil_source_repaired' or bundle.task_type != 'classification'
                or bundle.dim != 768 or bundle.lrmil is None
                or bundle.config.get('training_unit', 'slide') != 'slide'):
            raise ValueError('LRMIL requires native768 CONCH HR intermediates and measured LR RGB')
        if (not isinstance(stage_epochs, (list, tuple)) or len(stage_epochs) != 3
                or any(type(v) is not int or v < 1 for v in stage_epochs)
                or any(type(v) is not int or v < 1 for v in
                       [patch_batch_size, inference_batch_size, topk])
                or type(num_workers) is not int or num_workers < 0
                or type(gated_attention) is not bool
                or any(type(v) not in {int, float} or not np.isfinite(v) or v <= 0 for v in
                       [lambda_layer, lambda_attn, temperature_bag, temperature_attn,
                        lambda_bag, lambda_soft, lambda_hard])):
            raise ValueError('LRMIL needs three positive explicit budgets and every source loss active')
        self.bundle, self.assets, self.profile = bundle, bundle.lrmil, profile
        self.stage_epochs = dict(zip(PHASES, stage_epochs))
        self.options = dict(patch_batch_size=patch_batch_size, num_workers=num_workers,
            inference_batch_size=inference_batch_size, gated_attention=gated_attention,
            lambda_layer=lambda_layer, lambda_attn=lambda_attn,
            temperature_bag=temperature_bag, temperature_attn=temperature_attn,
            lambda_bag=lambda_bag, lambda_soft=lambda_soft, lambda_hard=lambda_hard, topk=topk)
        self.authors, self.losses, self.helpers = author_components()
        self.teacher = self.authors.MILClassifier(768, len(bundle.classes), gated_attention)
        # Later phases reset and initialize independently, as in the separate
        # source scripts. Their placeholder construction must not advance phase1 RNG.
        from .engine import rng_state, restore_rng
        rng = rng_state()
        self.encoder = native_backbone(self.assets.meta['checkpoint'])
        self.student = self.authors.StudentMIL(768, len(bundle.classes), gated_attention)
        restore_rng(rng)
        self.contract = digest_object({'source': COMMIT, 'bundle': bundle.fingerprint,
            'assets': self.assets.contract, 'profile': profile,
            'native_encoder': self.encoder._milplus_lrmil_identity,
            'stages': self.stage_epochs, 'options': self.options})
        self.phase, self.phase_epoch, self.epoch_index = 'teacher', 0, -1
        self.completed_epochs, self.run_seed, self.recipe = 0, None, None
        self.transfer_pending, self.training_done = False, False
        self.phase_best, self.phase_bad = dict.fromkeys(PHASES), dict.fromkeys(PHASES, 0)
        self.updates, self.student_instance_updates = dict.fromkeys(PHASES, 0), 0
        self.best_teacher, self.best_teacher_hash = None, None
        self.best_encoder, self.best_encoder_hash = None, None
        self.selected_epochs = dict.fromkeys(PHASES)
        self.train_orders, self.selection_history, self.last_patch_loss = [], [], None
        self._metadata = {}
        self.output = None
        self.apply_phase()

    def bind_output(self, output):
        self.output = Path(output)

    def apply_phase(self):
        for phase, module in [('teacher', self.teacher), ('patch', self.encoder), ('student', self.student)]:
            active = self.phase == phase
            for parameter in module.parameters():
                parameter.requires_grad_(active)
                if not active:
                    parameter.grad = None
            module.train(self.training and active)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'phase'):
            self.apply_phase()
        return self

    def configure_training(self, config):
        if (config['epochs'] != sum(self.stage_epochs.values()) or config['weight_decay'] != 0
                or config['max_instances'] is not None or config['grad_clip'] is not None
                or config['selection_metric'] != 'lrmil_stage_selection'):
            raise ValueError('LRMIL requires all three declared stages, full bags and original Adam/no clipping')
        self.recipe = {key: config[key] for key in
                       ['epochs', 'lr', 'weight_decay', 'patience', 'selection_metric']}

    def prepare(self, bundle, seed, device):
        if bundle.fingerprint != self.bundle.fingerprint or type(seed) is not int:
            raise ValueError('LRMIL actual data and independent-stage seed changed')
        self.run_seed = seed

    def make_optimizers(self, lr, weight_decay):
        self.optimizers = [torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
                           for module in [self.teacher, self.encoder, self.student]]
        return self.optimizers

    def validate_optimizer_state(self):
        for phase, module, optimizer in zip(PHASES, [self.teacher, self.encoder, self.student], self.optimizers):
            groups = optimizer.param_groups
            if (len(groups) != 1 or groups[0]['lr'] != self.recipe['lr']
                    or groups[0]['weight_decay'] != 0 or groups[0]['betas'] != (.9, .999)
                    or groups[0]['eps'] != 1e-8
                    or {id(p) for p in groups[0]['params']} != {id(p) for p in module.parameters()}):
                raise ValueError('LRMIL complete per-stage Adam ownership or hyperparameters changed')
            for name, parameter in module.named_parameters():
                count = (self.student_instance_updates if phase == 'student'
                         and name.startswith('inst_classifier.') else self.updates[phase])
                value = optimizer.state.get(parameter)
                if (count and value is None) or (value is not None and int(value.get('step', -1)) != count):
                    raise ValueError('LRMIL actual per-parameter Adam update count changed')

    def rows(self, subset):
        return sorted(self.bundle.records(subset), key=lambda row: row['slide_id'])

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs or self.training_done:
            raise ValueError('LRMIL must continue the actual saved complete phase sequence')
        if self.transfer_pending:
            from .engine import seed_everything
            if self.phase == 'teacher':
                if self.best_teacher is None or state_hash(self.best_teacher) != self.best_teacher_hash:
                    raise ValueError('LRMIL patch KD needs the actual VAL-selected HR teacher')
                self.teacher.load_state_dict(self.best_teacher, strict=True)
                seed_everything(self.run_seed)
                fresh = native_backbone(self.assets.meta['checkpoint'])
                self.encoder.load_state_dict(fresh.state_dict(), strict=True)
                del fresh
                self.optimizers[1].state.clear()
                self.phase = 'patch'
            elif self.phase == 'patch':
                if self.best_encoder is None or state_hash(self.best_encoder) != self.best_encoder_hash:
                    raise ValueError('LRMIL student needs the actual TRAIN-selected distilled LR encoder')
                self.encoder.visual.load_state_dict(self.best_encoder, strict=True)
                seed_everything(self.run_seed)
                fresh = self.authors.StudentMIL(768, len(self.bundle.classes), self.options['gated_attention'])
                self.student.load_state_dict(fresh.state_dict(), strict=True)
                self.optimizers[2].state.clear()
                self.phase = 'student'
            else:
                raise ValueError('LRMIL has no fourth training phase')
            self.phase_epoch, self.transfer_pending = 0, False
            self._metadata.clear()
        self.epoch_index = epoch
        self.apply_phase()

    @torch.no_grad()
    def metadata(self, row, device):
        if row['subset'] != 'train':
            raise ValueError('LRMIL teacher guidance for optimization is TRAIN-only')
        if row['slide_id'] not in self._metadata:
            self.teacher.eval()
            result = teacher_metadata(self.teacher, self.assets.high(row).to(device))
            self._metadata[row['slide_id']] = {key: value.detach().cpu() for key, value in result.items()}
        return self._metadata[row['slide_id']]

    @torch.no_grad()
    def low_features(self, row, device):
        if self.phase != 'student' or self.best_encoder_hash is None:
            raise ValueError('LRMIL LR bags require the actual selected, frozen patch encoder')
        if self.output is None:
            raise ValueError('LRMIL must bind its run directory before feature extraction')
        stem = self.output/'lrmil_lr_features'/self.best_encoder_hash/digest_object(row['slide_id'])
        path, record_path = stem.with_suffix('.pt'), stem.with_suffix('.json')
        record = self.assets._check_archive(row)
        expected = {'encoder_sha256': self.best_encoder_hash, 'assets_sha256': self.assets.contract,
                    'slide_id': row['slide_id'], 'source_sha256': record['sha256'],
                    'rows': record['low_nodes'], 'dimension': 768}
        if path.exists() or record_path.exists():
            if not path.is_file() or not record_path.is_file():
                raise ValueError('LRMIL LR feature cache is incomplete; preserve it and use a new run')
            meta = read_json(record_path)
            if {key: meta.get(key) for key in expected} != expected or digest_file(path) != meta.get('sha256'):
                raise ValueError('LRMIL LR feature cache differs from its selected encoder or true RGB')
            value = torch.load(path, weights_only=True, map_location='cpu')
        else:
            self.encoder.eval()
            batches = []
            for start in range(0, record['low_nodes'], self.options['inference_batch_size']):
                images = self.assets.low(row, start, min(start+self.options['inference_batch_size'], record['low_nodes']))
                batches.append(self.encoder(normalize_rgb(images).to(device))['final'].cpu())
            value = torch.cat(batches)
            if value.shape != (record['low_nodes'], 768) or not torch.isfinite(value).all():
                raise FloatingPointError('LRMIL actual distilled LR encoder returned invalid full-dimensional features')
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(value, path)
            write_json(record_path, {**expected, 'sha256': digest_file(path),
                                    'synthetic': bool(self.bundle.config.get('synthetic', False))})
        if value.shape != (record['low_nodes'], 768) or not torch.isfinite(value).all():
            raise ValueError('LRMIL recorded LR feature tensor has invalid dimensions or values')
        return value.to(device)

    def epoch(self, bundle, optimizers, device):
        if bundle.fingerprint != self.bundle.fingerprint or len(optimizers) != 3 or any(
                left is not right for left, right in zip(optimizers, self.optimizers)):
            raise ValueError('LRMIL actual inputs or complete three-Adam state changed')
        self.train()
        rows = self.rows('train')
        optimizer = optimizers[PHASES.index(self.phase)]
        losses, order, instance_updates = [], [], 0
        if self.phase == 'patch':
            metadata = {row['slide_id']: self.metadata(row, device) for row in rows}
            dataset = PatchExamples(self.assets, rows, metadata)
            loader = DataLoader(dataset, batch_size=self.options['patch_batch_size'], shuffle=True,
                num_workers=self.options['num_workers'], collate_fn=collate_list)
            for items in loader:
                images = torch.stack([item['img5'] for item in items]).to(device)
                moved = [{key: ({layer: value.to(device) for layer, value in item[key].items()}
                         if key == 'teacher_layers' else item[key].to(device))
                         for key in ['coords_5x', 'coords_20x', 'teacher_final', 'teacher_layers', 'teacher_attn']}
                         for item in items]
                terms = patch_objective(self.encoder(images), moved,
                    self.options['lambda_layer'], self.options['lambda_attn'])
                self._step(terms['loss'], optimizer)
                losses.append({key: float(value.detach()) for key, value in terms.items()})
                order.extend(item['unit'] for item in items)
            self.last_patch_loss = sum(row['loss'] for row in losses)/len(losses)
        else:
            loader = DataLoader(list(range(len(rows))), batch_size=1, shuffle=True)
            for indices in loader:
                row = rows[int(indices[0])]
                if self.phase == 'teacher':
                    logits, _, _ = self.teacher(self.assets.high(row).to(device))
                    loss = self.losses.ce_loss(logits, torch.tensor(row['target'], device=device))
                    terms = {'loss': loss, 'teacher_ce': loss}
                else:
                    low, high = self.assets.coordinates(row)
                    meta = {key: value.to(device) for key, value in self.metadata(row, device).items()}
                    terms, _, guided = student_objective(self.student, self.low_features(row, device),
                        torch.tensor(row['target'], device=device), meta, low.to(device), high.to(device),
                        **{key: self.options[key] for key in ['temperature_bag', 'temperature_attn',
                                                            'lambda_bag', 'lambda_soft', 'lambda_hard', 'topk']})
                    instance_updates += int(guided.sum().abs() > 1e-6)
                self._step(terms['loss'], optimizer)
                order.append(row['slide_id'])
                losses.append({key: float(value.detach()) for key, value in terms.items()})
        if not losses:
            raise ValueError('LRMIL phase has no actual training units')
        self.updates[self.phase] += len(losses)
        self.student_instance_updates += instance_updates
        self.completed_epochs += 1
        self.phase_epoch += 1
        self.train_orders.append(order)
        self._last_instance_updates = instance_updates
        return losses

    def _step(self, loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError('LRMIL original complete phase objective is nonfinite')
        optimizer.zero_grad()
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise FloatingPointError('LRMIL original phase gradient is nonfinite')
        optimizer.step()

    def should_validate(self, epoch):
        return self.phase != 'patch'

    def finish_epoch(self, bundle, device, cfg):
        if self.phase == 'patch':
            self._select(self.last_patch_loss)

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle.fingerprint != self.bundle.fingerprint or subset not in {'val', 'test'}:
            raise ValueError('LRMIL predictions require actual registered validation/test slides')
        if self.phase == 'patch' or (subset == 'test' and self.phase != 'student'):
            raise ValueError('LRMIL full test inference requires the selected LR student phase')
        self.eval()
        rows, records = self.rows(subset), []
        loader = DataLoader(list(range(len(rows))), batch_size=1, shuffle=False)
        for indices in loader:
            row = rows[int(indices[0])]
            logits = (self.teacher(self.assets.high(row).to(device))[0] if self.phase == 'teacher'
                      else self.student(self.low_features(row, device))[0])
            if not torch.isfinite(logits).all():
                raise FloatingPointError('LRMIL native prediction produced nonfinite logits')
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], target=row['target'],
                          prediction=int(logits.argmax()))
            record.update({f'p_{index}': float(value) for index, value in enumerate(logits.softmax(-1))})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        if not len(frame) or not frame.slide_id.is_unique:
            raise ValueError('LRMIL must evaluate every actual slide once')
        classes = len(self.bundle.classes)
        probability = frame[[f'p_{i}' for i in range(classes)]].to_numpy()
        value = (roc_auc_score(frame.target, probability[:, 1]) if classes == 2 else
                 roc_auc_score(frame.target, probability, multi_class='ovr'))
        return {'lrmil_stage_selection': float(value)}

    def after_validation(self, metrics):
        self._select(metrics['lrmil_stage_selection'])

    def _select(self, value):
        if not np.isfinite(value):
            raise ValueError('LRMIL source phase selection value is undefined')
        previous = self.phase_best[self.phase]
        better = previous is None or (value < previous if self.phase == 'patch' else value > previous)
        if better:
            self.phase_best[self.phase], self.phase_bad[self.phase] = float(value), 0
            self.selected_epochs[self.phase] = self.phase_epoch
            if self.phase == 'teacher':
                self.best_teacher = snapshot(self.teacher)
                self.best_teacher_hash = state_hash(self.best_teacher)
                # Original save_teacher_meta iterates shuffled TRAIN, then VAL
                # and TEST loaders. Preserve those RNG draws without reading,
                # predicting or exporting TEST features/labels during fitting.
                for subset in ['train', 'val', 'test']:
                    list(DataLoader(list(range(len(self.rows(subset)))), batch_size=1, shuffle=subset == 'train'))
            elif self.phase == 'patch':
                self.best_encoder = snapshot(self.encoder.visual)
                self.best_encoder_hash = state_hash(self.best_encoder)
        else:
            self.phase_bad[self.phase] += 1
        done = (self.phase_epoch == self.stage_epochs[self.phase] or
                (self.phase != 'patch' and self.phase_bad[self.phase] >= self.recipe['patience']))
        self.transfer_pending = done and self.phase != 'student'
        self.training_done = done and self.phase == 'student'
        self.selection_history.append({'epoch': self.epoch_index, 'phase': self.phase,
            'phase_epoch': self.phase_epoch, 'value': float(value), 'best': self.phase_best[self.phase],
            'bad': self.phase_bad[self.phase], 'done': done,
            'instance_updates': self._last_instance_updates})

    def selection_improved(self, objective, best):
        return self.phase == 'student' and objective > best

    def get_extra_state(self):
        return copy.deepcopy({key: getattr(self, key) for key in self.STATE_KEYS})

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) != set(self.STATE_KEYS)
                or state['contract'] != self.contract or state['phase'] not in PHASES
                or type(state['run_seed']) is not int
                or any(type(state[key]) is not int for key in
                       ['epoch_index', 'completed_epochs', 'phase_epoch', 'student_instance_updates'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or len(state['train_orders']) != state['completed_epochs']
                or len(state['selection_history']) != state['completed_epochs']
                or not isinstance(state['recipe'], dict)
                or (self.recipe is not None and state['recipe'] != self.recipe)
                or (self.run_seed is not None and state['run_seed'] != self.run_seed)):
            raise ValueError('LRMIL checkpoint source, inputs, phase or history structure changed')
        if self.output is not None and (self.output/'request.json').exists():
            if state['run_seed'] != read_json(self.output/'request.json')['seed']:
                raise ValueError('LRMIL independent-phase seed differs from the run request')
        for name in ['teacher', 'encoder']:
            saved, digest = state['best_'+name], state['best_'+name+'_hash']
            if (saved is None) != (digest is None) or (saved is not None and state_hash(saved) != digest):
                raise ValueError('LRMIL selected teacher or encoder snapshot changed')
        phase, epochs, best, bad = 'teacher', dict.fromkeys(PHASES, 0), dict.fromkeys(PHASES), dict.fromkeys(PHASES, 0)
        selected, updates, instance_updates, done = dict.fromkeys(PHASES), dict.fromkeys(PHASES, 0), 0, False
        slides = [row['slide_id'] for row in self.rows('train')]
        patches = [row['slide_id']+':'+str(index) for row in self.rows('train')
                   for index in range(self.assets.meta['slides'][row['slide_id']]['low_nodes'])]
        for index, (order, record) in enumerate(zip(state['train_orders'], state['selection_history'])):
            if done:
                if phase == 'student':
                    raise ValueError('LRMIL trace continues after completed student training')
                phase = PHASES[PHASES.index(phase)+1]
            if len(order) != len(patches if phase == 'patch' else slides) or sorted(order) != sorted(patches if phase == 'patch' else slides):
                raise ValueError('LRMIL source full-unit sampling order is incomplete')
            epochs[phase] += 1
            updates[phase] += math.ceil(len(patches)/self.options['patch_batch_size']) if phase == 'patch' else len(slides)
            value = record['value']
            if type(value) not in {float, int} or not np.isfinite(value) or (phase != 'patch' and not 0 <= value <= 1):
                raise ValueError('LRMIL actual phase selection metric changed')
            improved = best[phase] is None or (value < best[phase] if phase == 'patch' else value > best[phase])
            if improved:
                best[phase], bad[phase], selected[phase] = value, 0, epochs[phase]
            else:
                bad[phase] += 1
            count = record['instance_updates']
            if type(count) is not int or not 0 <= count <= (len(slides) if phase == 'student' else 0):
                raise ValueError('LRMIL active source instance-head update trace changed')
            instance_updates += count
            done = epochs[phase] == self.stage_epochs[phase] or (phase != 'patch' and bad[phase] >= state['recipe']['patience'])
            if record != {'epoch': index, 'phase': phase, 'phase_epoch': epochs[phase], 'value': value,
                    'best': best[phase], 'bad': bad[phase], 'done': done, 'instance_updates': count}:
                raise ValueError('LRMIL actual first-optimum/stopping/phase trace changed')
        if (state['phase'] != phase or state['phase_epoch'] != epochs[phase]
                or state['phase_best'] != best or state['phase_bad'] != bad
                or state['updates'] != updates or state['student_instance_updates'] != instance_updates
                or state['selected_epochs'] != selected
                or state['transfer_pending'] != (done and phase != 'student')
                or state['training_done'] != (done and phase == 'student')
                or (selected['teacher'] is None) != (state['best_teacher'] is None)
                or (selected['patch'] is None) != (state['best_encoder'] is None)
                or state['last_patch_loss'] != next((record['value'] for record in
                    reversed(state['selection_history']) if record['phase'] == 'patch'), None)):
            raise ValueError('LRMIL actual optimizer counts, transfer or source optimum changed')
        for key in self.STATE_KEYS:
            setattr(self, key, copy.deepcopy(state[key]))
        self._metadata.clear()
        self.apply_phase()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.phase != 'teacher' and state_hash(self.teacher.state_dict()) != self.best_teacher_hash:
            raise ValueError('LRMIL frozen teacher differs from its actual selected snapshot')
        if self.phase == 'student' and state_hash(self.encoder.visual.state_dict()) != self.best_encoder_hash:
            raise ValueError('LRMIL frozen LR encoder differs from its actual selected snapshot')
        return result

    def protocol(self):
        return {'method': self.profile, 'source': COMMIT, 'native_encoder': self.encoder._milplus_lrmil_identity,
            'phases': self.stage_epochs, 'phase': self.phase, 'options': self.options, 'recipe': self.recipe,
            'completed_epochs': self.completed_epochs, 'actual_optimizer_updates': self.updates.copy(),
            'actual_student_instance_updates': self.student_instance_updates,
            'selected_epochs': self.selected_epochs.copy(), 'selected_teacher_sha256': self.best_teacher_hash,
            'selected_encoder_sha256': self.best_encoder_hash, 'train_orders_sha256': digest_object(self.train_orders),
            'objectives': 'HRteacherCE -> layer0/1/10 regionMSE + attentionWeightedFeatureMSE -> LRstudentCE + bagKL + softAttentionKL + hardInstanceCE',
            'selection': 'Teacher and student: first highest VAL-AUC, patience20; patch: first lowest TRAIN mean-batch objective, fixed full declared budget',
            'budget_disclosure': 'Source CLI provides no max-epoch/batch/lambda-layer defaults. Explicit registered reference caps200/200/200, patchBatch64, patch coefficients1/1, temperatures1/1; lr1e-4 and patience20, student weights.5/.5/.3 match paper. Caps are declared engineering choices, not claimed historical paper values.',
            'repairs': 'Native NHWC token layout; teacher metadata bag/attention output order; own patient identities and no cross-fold parameter carry; actual selected-encoder LR feature generation fills missing released preprocessing step',
            'source_preserved': 'Source position slice; layers0/1/10 and extra patch attentionMSE; raw-logit guidance and source KL batchmean/T-squared, gate and binary hard head; no loss/width/patch/epoch fallback',
            'inference': 'Student prediction only uses distilled LR RGB/features; HR teacher guidance is TRAIN-only; full input integrity checking remains separate from any future end-to-end efficiency benchmark',
            'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

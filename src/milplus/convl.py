"""Complete original ConVL patient-survival training, precision and recovery."""
import copy
import math

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .convl_optim import create_source_optimizer
from .convl_precision import validate_scaler_history
from .convl_source import COMMIT, build_original
from .focus_network import WEIGHTS_SHA
from .io import digest_object
from .upstream import load_definitions
from .vlsa import Patients, one_patient

PROFILES = {'convl_yaml_fp16_source_repaired', 'convl_yaml_fp32_reference'}


class ConVLMIL(nn.Module):
    owns_sampling = True
    # Shared engine uses strict improvement and bad>=patience, matching the
    # original Lightning EarlyStopping(min_delta0,mode=min,patience8) callbacks.
    handles_early_stopping = False

    def __init__(self, bundle, profile='convl_yaml_fp16_source_repaired'):
        super().__init__()
        if profile not in PROFILES or getattr(bundle, 'convl', None) is None:
            raise ValueError('ConVL needs its original native CONCH patient survival inputs and explicit precision route')
        self.bundle, self.inputs, self.profile = bundle, bundle.convl, profile
        self.use_amp = profile == 'convl_yaml_fp16_source_repaired'
        self.core = build_original(self.inputs.weights, self.inputs.meta['concept_cohort'])
        loss = load_definitions('convl', 'utils/utils.py', ['nll_loss', 'NLLSurvLoss'], {'torch': torch})
        self.source_nll = loss.NLLSurvLoss(alpha=0.)
        self.contract = digest_object({'source': COMMIT, 'inputs': self.inputs.contract,
            'bundle': bundle.fingerprint, 'profile': profile, 'accumulate_grad_batches': 2,
            'native_dimensions': [512, 256, 16, 768, 12, 12, 4]})
        self.recipe = None
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders, self.update_traces = [], []
        self._scaler = None
        self._saved_scaler = None

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'convl_val_nll'):
            raise ValueError('ConVL uses complete patient bags, no clipping and original NLL-only VAL selection')
        device = next(self.parameters()).device
        if self.use_amp and device.type != 'cuda':
            raise ValueError('Original ConVL FP16 mixed precision requires CUDA; choose the explicit FP32 reference on CPU')
        self.recipe = {key: cfg[key] for key in ['epochs', 'lr', 'weight_decay', 'patience', 'threads', 'selection_metric']}
        self._precision(device)

    def _precision(self, device):
        device = torch.device(device)
        if self.use_amp and device.type != 'cuda':
            raise ValueError('ConVL FP16 source route requires an actual allocated CUDA device')
        if self._scaler is None:
            self._scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            if self._saved_scaler is not None:
                self._scaler.load_state_dict(copy.deepcopy(self._saved_scaler))
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=self.use_amp)

    def make_optimizer(self, lr, weight_decay):
        if self.recipe is None:
            raise ValueError('Configure the complete ConVL recipe before creating its original optimizer')
        self._optimizer = create_source_optimizer(self.core, lr=lr, weight_decay=weight_decay)
        return self._optimizer

    def _active_parameters(self):
        unused = {id(p) for p in self.core.fc_patch.parameters()}
        unused.update(id(p) for p in self.core.clip_model.text.token_embedding.parameters())
        unused.add(id(self.core.clip_model.text.cls_emb))
        return {p for p in self.core.parameters() if id(p) not in unused}

    def validate_optimizer_state(self):
        opt = self._optimizer
        expected_no_decay, expected_decay = [], []
        for name, parameter in self.core.named_parameters():
            (expected_no_decay if parameter.ndim == 1 or name.endswith('.bias') else expected_decay).append(parameter)
        if (opt.k != 5 or opt.alpha != .5 or opt.param_groups is not opt.optimizer.param_groups
                or len(opt.param_groups) != 2 or type(opt.optimizer).__name__ != 'RAdam'):
            raise ValueError('ConVL lost its actual original Lookahead/RAdam optimizer')
        for group, parameters, decay in zip(opt.param_groups, [expected_no_decay, expected_decay], [0., self.recipe['weight_decay']]):
            if ([id(p) for p in group['params']] != [id(p) for p in parameters]
                    or group['lr'] != self.recipe['lr'] or group['weight_decay'] != decay
                    or group['betas'] != (.9, .999) or group['eps'] != 1e-8
                    or group['counter'] != self.optimizer_updates % 5):
                raise ValueError('ConVL original decay partition, LR or Lookahead phase changed')
        expected = self._active_parameters() if self.optimizer_updates else set()
        if set(opt.optimizer.state) != expected:
            raise ValueError('ConVL lost active text/prompt/attention/head RAdam moments or updated unused tensors')
        for parameter, state in opt.optimizer.state.items():
            if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or state['step'] != self.optimizer_updates
                    or any(state[k].shape != parameter.shape or not torch.isfinite(state[k]).all()
                           for k in ['exp_avg', 'exp_avg_sq'])):
                raise ValueError('ConVL original RAdam moments or successful update count changed')
        slow_expected = set(self.core.parameters()) if self.optimizer_updates else set()
        if set(opt.state) != slow_expected:
            raise ValueError('ConVL original Lookahead slow weights are incomplete')
        for parameter, state in opt.state.items():
            if set(state) != {'slow_param'} or state['slow_param'].shape != parameter.shape or not torch.isfinite(state['slow_param']).all():
                raise ValueError('ConVL original Lookahead slow tensor changed')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('ConVL recovery requires a complete original patient-epoch boundary')
        self.inputs.check_unchanged(); self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer or self.recipe is None:
            raise ValueError('ConVL original patient inputs, recipe or optimizer ownership changed')
        self.core.train()
        loader = DataLoader(Patients(bundle, 'train'), batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False, collate_fn=one_patient)
        losses, order, updates, window = [], [], [], []
        for index, rows in enumerate(loader):
            patient = rows[0]['global_patient_id']; order.append(patient); window.append(patient)
            features, label, censorship = self.inputs.patient(rows, device)
            with self._precision(device):
                output = self.core(data=features)
                nll = self.source_nll(output['hazards'], output['S'], label.long(), censorship)
                loss = nll+output['concept_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('ConVL original NLL+distribution+clustering objective is nonfinite')
            # Lightning2.3.0 zeroes after training_step at the first member of
            # each accumulation window, and divides *all* losses by2, including
            # the last odd patient. It steps on that final partial window.
            if index % 2 == 0:
                optimizer.zero_grad()
            self._scaler.scale(loss/2).backward()
            losses.append({'loss': float(loss.detach()), 'nll': float(nll.detach()),
                'concept_loss': float(output['concept_loss'].detach()),
                'distribution_alignment': float(output['dist_loss_val'].detach()),
                'instance_clustering': float(output['loss_clst'].detach()),
                'concept_clustering': float(output['loss_concept_clst'].detach())})
            if (index+1) % 2 == 0 or index+1 == len(loader):
                self._scaler.unscale_(optimizer)
                if any(p.grad is None for p in self._active_parameters()):
                    raise FloatingPointError('ConVL lost an original active text/prompt/attention/head gradient')
                finite = all(torch.isfinite(p.grad).all() for p in self._active_parameters())
                if not self.use_amp and not finite:
                    raise FloatingPointError('ConVL FP32 reference accumulated gradients are nonfinite')
                scale_before = self._scaler.get_scale()
                self._scaler.step(optimizer); self._scaler.update()
                scale_after = self._scaler.get_scale()
                skipped = scale_after < scale_before
                if bool(skipped) != (not finite):
                    raise RuntimeError('ConVL AMP overflow and actual skipped-step state disagree')
                self.optimizer_updates += int(not skipped)
                updates.append({'patients': window.copy(), 'normalization': 2, 'skipped': bool(skipped),
                    'scale_before': scale_before, 'scale_after': scale_after})
                window.clear()
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(order); self.update_traces.append(updates)
        self.validate_optimizer_state()
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle.fingerprint != self.bundle.fingerprint or subset not in {'val', 'test'}:
            raise ValueError('ConVL predictions require the registered independent VAL or TEST patient partition')
        self.core.eval(); self.inputs.check_unchanged()
        loader = DataLoader(Patients(bundle, subset), batch_size=1, num_workers=0,
                            pin_memory=False, collate_fn=one_patient)
        records = []
        for rows in loader:
            features, label, censorship = self.inputs.patient(rows, device)
            with self._precision(device):
                output = self.core(data=features)
                nll = self.source_nll(output['hazards'], output['S'], label.long(), censorship)
            if not torch.isfinite(nll) or not torch.isfinite(output['S']).all():
                raise FloatingPointError('ConVL original evaluation survival probabilities/NLL are nonfinite')
            row = rows[0]
            record = {key: row[key] for key in ['patient_id', 'global_patient_id', 'subset', 'time', 'event']}
            record.update(slide_id='patient:'+row['global_patient_id'], source_slides=';'.join(r['slide_id'] for r in rows),
                risk=float(-output['S'].sum(1)[0]), convl_nll=float(nll), convl_bin=int(label.item()),
                convl_patient_patches=len(features))
            record.update({f'hazard_{i}': float(v) for i, v in enumerate(output['hazards'][0])})
            record.update({f'survival_{i}': float(v) for i, v in enumerate(output['S'][0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        return {'convl_val_nll': float(frame.convl_nll.mean())}

    def selection_objective(self, metrics, epoch, current):
        return -metrics['convl_val_nll']

    def get_extra_state(self):
        scaler = self._scaler.state_dict() if self._scaler is not None else self._saved_scaler
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
            'optimizer_updates': self.optimizer_updates, 'train_orders': self.train_orders,
            'update_traces': self.update_traces, 'scaler': scaler})

    def set_extra_state(self, state):
        keys = {'contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders', 'update_traces', 'scaler'}
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.contract
                or any(type(state[k]) is not int for k in ['epoch_index', 'completed_epochs', 'optimizer_updates'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or (self.recipe is not None and state['recipe'] != self.recipe)):
            raise ValueError('ConVL checkpoint source/input/precision/recipe or epoch boundary changed')
        ids = [rows[0]['global_patient_id'] for rows in Patients(self.bundle, 'train').rows]
        epochs, updates = state['completed_epochs'], 0
        recipe_keys = {'epochs', 'lr', 'weight_decay', 'patience', 'threads', 'selection_metric'}
        recipe = state['recipe']
        if epochs and (not isinstance(recipe, dict) or set(recipe) != recipe_keys
                or any(type(recipe[k]) is not int or recipe[k] < 1 for k in ['epochs', 'patience', 'threads'])
                or recipe['epochs'] < epochs or recipe['selection_metric'] != 'convl_val_nll'
                or not isinstance(recipe['lr'], (int, float)) or not math.isfinite(recipe['lr']) or recipe['lr'] <= 0
                or not isinstance(recipe['weight_decay'], (int, float)) or not math.isfinite(recipe['weight_decay']) or recipe['weight_decay'] < 0):
            raise ValueError('ConVL checkpoint lost its actual complete configured source recipe')
        if len(state['train_orders']) != epochs or len(state['update_traces']) != epochs:
            raise ValueError('ConVL lost complete patient and accumulation history')
        for order, traces in zip(state['train_orders'], state['update_traces']):
            if len(order) != len(ids) or set(order) != set(ids) or len(traces) != math.ceil(len(ids)/2):
                raise ValueError('ConVL checkpoint omitted or repeated an actual TRAIN patient')
            for index, trace in enumerate(traces):
                if (set(trace) != {'patients', 'normalization', 'skipped', 'scale_before', 'scale_after'}
                        or trace['patients'] != order[index*2:index*2+2] or trace['normalization'] != 2
                        or type(trace['skipped']) is not bool
                        or (not self.use_amp and (trace['skipped'] or trace['scale_before'] != 1. or trace['scale_after'] != 1.))
                        or not all(isinstance(trace[k], (int, float)) and math.isfinite(trace[k]) and trace[k] > 0
                                   for k in ['scale_before', 'scale_after'])
                        or trace['skipped'] != (trace['scale_after'] < trace['scale_before'])):
                    raise ValueError('ConVL original complete two-patient/odd-tail/scaler update history changed')
                updates += int(not trace['skipped'])
        if state['optimizer_updates'] != updates:
            raise ValueError('ConVL successful optimizer count disagrees with actual overflow/accumulation history')
        scaler_keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
        if epochs and ((self.use_amp and (not isinstance(state['scaler'], dict) or set(state['scaler']) != scaler_keys))
                       or (not self.use_amp and state['scaler'] != {})):
            raise ValueError('ConVL lost actual AMP scale/growth tracker or changed precision route')
        validate_scaler_history(self.use_amp, state['update_traces'], state['scaler'])
        for key in ['recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders', 'update_traces']:
            setattr(self, key, copy.deepcopy(state[key]))
        self._saved_scaler = copy.deepcopy(state['scaler'])
        if self._scaler is not None and self._saved_scaler is not None:
            self._scaler.load_state_dict(self._saved_scaler)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'input_contract': self.inputs.contract,
            'pretrained_CONCH_weights_sha256': WEIGHTS_SHA, 'native_input_dimension': 512,
            'full_text': 'Actual full pretrained768hidden/12layer/12head/512projection text, original contextual prefix/suffix,16dynamic prompts and unmasked EOT source training path',
            'native_network': 'Actual shared512to256 projection,4headCGA, visual/concept heads, source sigmoid-KL and both max-based clustering terms; original unused768D fc_patch retained',
            'objective': 'Actual NLLSurvLoss(alpha0) on blended hazards + source .1distribution -.5instance_clustering -.3concept_clustering',
            'recipe': self.recipe, 'source_default': '200maxepochs, patientbatch1, accumulation2 including odd-tail divisor2, source bias/1D zero-decay Lookahead(k5,alpha.5)/RAdam(lr2e-4,wd1e-5), no scheduler/clipping, VALNLL-only strict first-tie selection and patience8',
            'precision': 'Original CUDA FP16 autocast+dynamic GradScaler' if self.use_amp else 'Explicit full-native FP32 precision reference',
            'overflow': 'Actual AMP skipped updates and scale/growth state recorded; no skipped window reported as an optimizer update',
            'patient_io_port': 'Every true patient once, all slides in registered stable order; source slide-index/unique-patient-length mismatch repaired; worker0 recoverable I/O',
            'binning': self.inputs.meta['metadata']['binning'],
            'source_entry_ports': 'Pinned train.py references missing Model.encoder/Data.data_shuffle fields; unified entry binds the released full YAML recipe explicitly',
            'checkpoint': 'All active full text/prompt/CGA/head and allocated unused tensors, actual Lookahead fast/slow weights/counters, RAdam moments/buffer, AMP scaler, patient/order/window traces and global engine RNG',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'formal_admission': False}

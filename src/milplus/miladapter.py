"""All actual author MIL-Adapter branches with full original few-shot recipe."""
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from .io import digest_object
from .miladapter_source import COMMIT, PROFILES, build_model, fewshot_indices


class MILAdapter(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='miladapter_taskres_abmil', k_shots=8, sampling_seed=0):
        super().__init__()
        if bundle.miladapter is None or profile not in PROFILES:
            raise ValueError('MIL-Adapter requires its registered complete source condition and prepared assets')
        if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32:
            raise ValueError('MIL-Adapter NumPy support sampling seed must be a nonnegative uint32')
        self.bundle, self.assets, self.profile = bundle, bundle.miladapter, profile
        pool = bundle.records('train')
        indices = fewshot_indices([r['target'] for r in pool], k_shots, sampling_seed)
        self.support = [pool[i] for i in indices]
        self.support_ids = [r['slide_id'] for r in self.support]
        features = [self.assets.load(r).numpy() for r in self.support]
        self.network = build_model(profile, self.assets.prototypes, features, [r['target'] for r in self.support])
        self.route, self.aggregator, self.init = PROFILES[profile]
        if self.aggregator == 'WIKGMIL' and any(len(x) < 6 for x in features):
            raise ValueError('Original WiKG topk6 needs at least six real patches; no fabricated padding')
        self.k_shots, self.sampling_seed = k_shots, sampling_seed
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'assets': self.assets.contract,
            'bundle': bundle.fingerprint, 'k_shots': k_shots, 'sampling_seed': sampling_seed,
            'support_ids': self.support_ids})
        self.recipe = None
        self.completed_epochs, self.optimizer_updates, self.epoch_index = 0, 0, -1
        self.current_order = self.support_ids.copy(); self.train_orders = []
        self.active_parameter_names = []
        self._fixed = {name: value.detach().cpu().clone() for name, value in self.network.state_dict().items()
            if name == 'text_embeddings' or (name == 'classifier' and self.route in {'CLIPAdapter', 'TIPAdapter'})
            or name == 'cache_values'}

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'miladapter_completed_epoch'):
            raise ValueError('MIL-Adapter uses full source bags, no clipping, no VAL model selection and final weights only')
        self.recipe = {k: cfg[k] for k in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr,
            betas=(.9, .999), weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self.recipe['epochs'])
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 1 or groups[0]['weight_decay'] != self.recipe['weight_decay']
                or groups[0]['initial_lr'] != self.recipe['lr'] or groups[0]['betas'] != (.9, .999)
                or groups[0]['eps'] != 1e-8
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('MIL-Adapter complete source AdamW configuration changed')
        state = self.scheduler.state_dict()
        if (state['T_max'] != self.recipe['epochs'] or state['eta_min'] != 0
                or state['last_epoch'] != self.completed_epochs or state['_step_count'] != self.completed_epochs+1
                or state['_last_lr'] != [groups[0]['lr']]):
            raise ValueError('MIL-Adapter original cosine phase or actual learning rate changed')
        params = dict(self.network.named_parameters())
        if self.optimizer_updates:
            expected = {params[n] for n in self.active_parameter_names}
            if set(self._optimizer.state) != expected or any(int(s['step']) != self.optimizer_updates for s in self._optimizer.state.values()):
                raise ValueError('MIL-Adapter AdamW moments or genuine per-parameter update counts changed')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('MIL-Adapter resumes only at complete source epoch boundaries')
        self.epoch_index = epoch

    def forward(self, features):
        if features.ndim != 2 or not len(features) or features.shape[1] != self.bundle.dim:
            raise ValueError('MIL-Adapter requires full original native512D CONCH/768D KEEP patch bags')
        if self.aggregator == 'WIKGMIL' and len(features) < 6:
            raise ValueError('Original WiKG needs at least six patches; no fallback aggregator')
        logits, embedding = self.network(features)
        if logits.shape != (len(self.bundle.classes),) or not torch.isfinite(logits).all():
            raise FloatingPointError('Original MIL-Adapter produced invalid C-way logits')
        return {'logits': logits.unsqueeze(0), 'embedding': embedding}

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('MIL-Adapter input or actual AdamW identity changed')
        self.network.train()
        # Original shuffle_data shuffles the previous epoch's ordered list.
        # Starting every epoch from the initial support order changes its RNG
        # semantics even when the same seed is used.
        random.shuffle(self.current_order)
        by_id = {r['slide_id']: r for r in self.support}; losses = []
        for slide in self.current_order:
            row = by_id[slide]; features = self.assets.load(row).to(device)
            optimizer.zero_grad()
            result = self(features)
            target = torch.tensor(row['target'], device=device)
            loss = nn.functional.cross_entropy(result['logits'][0], target, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('MIL-Adapter complete source CE sum became nonfinite')
            loss.backward()
            names = [n for n, p in self.network.named_parameters() if p.grad is not None]
            if self.active_parameter_names and names != self.active_parameter_names:
                raise ValueError('MIL-Adapter active original parameter set changed')
            self.active_parameter_names = names
            if any(not torch.isfinite(p.grad).all() for p in self.network.parameters() if p.grad is not None):
                raise FloatingPointError('MIL-Adapter source gradient became nonfinite')
            optimizer.step(); self.optimizer_updates += 1
            losses.append({'loss': float(loss.detach()), 'patches': len(features),
                           'accuracy': float(result['logits'][0].argmax() == target)})
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(self.current_order.copy())
        self.scheduler.step()
        self.validate_optimizer_state()
        return losses

    def should_validate(self, epoch):
        # The author trains to the requested final epoch, then validates once.
        return epoch+1 == self.recipe['epochs']

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle.fingerprint != self.bundle.fingerprint or subset not in {'val', 'test'}:
            raise ValueError('MIL-Adapter requires its independent registered VAL/TEST partitions')
        self.network.eval(); records = []
        for row in bundle.records(subset):
            features = self.assets.load(row).to(device)
            result = self(features); logits = result['logits'][0]
            record = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'target']}
            record.update(subset=subset, source_slides=row['slide_id'], prediction=int(logits.argmax()),
                miladapter_patches=len(features))
            record.update({f'p_{i}': float(v) for i, v in enumerate(logits.softmax(0))})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        balanced = np.mean([np.mean(frame[frame.target == c].prediction.to_numpy() == c)
                            for c in range(len(self.bundle.classes))])
        return {'miladapter_completed_epoch': self.completed_epochs, 'miladapter_balanced_accuracy': float(balanced)}

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'current_order': self.current_order, 'train_orders': self.train_orders,
            'active_parameter_names': self.active_parameter_names,
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None})

    def set_extra_state(self, state):
        expected = {'contract', 'recipe', 'completed_epochs', 'optimizer_updates',
                    'current_order', 'train_orders', 'active_parameter_names', 'scheduler'}
        if not isinstance(state, dict) or set(state) != expected or state.get('contract') != self.contract:
            raise ValueError('MIL-Adapter support/prototype/recipe/checkpoint identity changed')
        epochs, updates = state['completed_epochs'], state['optimizer_updates']
        if (type(epochs) is not int or epochs < 0 or type(updates) is not int
                or updates != epochs*len(self.support) or len(state['train_orders']) != epochs
                or (self.recipe is not None and state['recipe'] != self.recipe)):
            raise ValueError('MIL-Adapter complete epoch or actual update count changed')
        for order in [state['current_order'], *state['train_orders']]:
            if len(order) != len(self.support_ids) or set(order) != set(self.support_ids):
                raise ValueError('MIL-Adapter support order lost or introduced non-TRAIN slides')
        if state['current_order'] != (state['train_orders'][-1] if epochs else self.support_ids):
            raise ValueError('MIL-Adapter previous-epoch shuffle state changed')
        if epochs:
            schedule = state['scheduler']
            if (not isinstance(schedule, dict) or schedule['last_epoch'] != epochs
                    or schedule['_step_count'] != epochs+1 or schedule['T_max'] != state['recipe']['epochs']
                    or not state['active_parameter_names']
                    or not set(state['active_parameter_names']) <= set(dict(self.network.named_parameters()))):
                raise ValueError('MIL-Adapter source cosine or active parameter checkpoint changed')
        for key, value in state.items():
            if key not in {'contract', 'scheduler'}:
                setattr(self, key, copy.deepcopy(value))
        if hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(state['scheduler'])
        self.epoch_index = epochs-1

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for key, value in self._fixed.items():
            if 'network.'+key not in state_dict or not torch.equal(state_dict['network.'+key].cpu(), value):
                raise ValueError('MIL-Adapter frozen actual text prototypes or support onehot labels changed')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'adapter': self.route,
            'aggregator': self.aggregator, 'initialization': self.init,
            'native_dimension': self.bundle.dim, 'prototype_provenance': self.assets.text_meta,
            'recipe': self.recipe, 'source_default_recipe': '20epochs AdamW1e-3/wd1e-5/betas.9,.999; cosine; CE sum, batch1 complete bags; no clipping/earlystop; final weights only',
            'support_policy': 'Exact source classwise NumPy choice without replacement k2/4/8/16; eligible TRAIN-only pool, single slide per patient, independent VAL/TEST extension',
            'k_shots': self.k_shots, 'sampling_seed': self.sampling_seed, 'support_slides': self.support_ids,
            'source_seed_boundary': 'ZSMIL resets torch.manual_seed42 before classifier+aggregator; other adapters use run seed; NumPy support selection happens first',
            'source_mechanisms': 'Full author native aggregators; TaskRes .25*trainable prototype residual; CLIPnativeD->D/4->D .2residual; TIP BGAP-initialized trainable support cache, beta5 alpha1 frozen onehot/text',
            'declared_ports': ['source .cuda constructors to normal device movement',
                'ZSMIL ZS undefined logit_scale -> existing self.logit_scale',
                'CLIP/TIP ABMIL tuple -> actual first embedding',
                'KEEP CLIP fixed512 input -> actual768, reduction4 preserved (named native-width port)', 'plain constant text tensor registered as unchanged buffer'],
            'unsupported_source_branches': 'LR never constructed; TITAN prototypes absent; TaskRes/CLIP/TIP only BGAP/BGMP/ABMIL/TransMIL',
            'selection': 'Final epoch only; validation balanced accuracy reported, not used for selection',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'resume': 'Full actual aggregator/adaptation/cache keys, fixed prototypes, support IDs/order, AdamW, cosine, previous shuffled order and all global RNG',
            'synthetic': bool(self.bundle.config.get('synthetic', False)),
            'real_wsi_validation': 'not_run', 'paper_scores': 'not_reproduced'}

"""Full source TRIAGE-MIL: MASS/hypergraph data, native epoch and survival selection."""
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import digest_object
from .triagemil_source import COMMIT, trainer


class SelectedBags(Dataset):
    def __init__(self, assets, subset):
        self.assets, self.ids = assets, assets.order[subset]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.assets.load(self.assets.by_patient[self.ids[index]])


class TRIAGEMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='triagemil_source', hidden=512, heads=6, hypergraph_layers=4,
                 attention_dropout=.22, learnable_temperature=True, batch_size=4,
                 accumulation_steps=1, amp=True, early_stopping=True, early_min_epochs=60):
        super().__init__()
        if profile != 'triagemil_source' or bundle.task_type != 'survival' or bundle.triagemil is None:
            raise ValueError('TRIAGE-MIL requires its original survival profile and prepared MASS/hypergraph assets')
        if not isinstance(bundle.config.get('time_unit'), str) or not bundle.config['time_unit'].strip():
            raise ValueError('TRIAGE-MIL requires actual clinical time units')
        if (any(type(v) is not int or v < 1 for v in [hidden, heads, hypergraph_layers, batch_size,
                                                     accumulation_steps, early_min_epochs]) or
                any(type(v) is not bool for v in [learnable_temperature, amp, early_stopping]) or
                isinstance(attention_dropout, bool) or not isinstance(attention_dropout, (float, int)) or
                not math.isfinite(attention_dropout) or not 0 <= attention_dropout < 1):
            raise ValueError('Invalid TRIAGE-MIL native architecture, batch/AMP or stopping parameters')
        self.bundle, self.assets, self.profile = bundle, bundle.triagemil, profile
        self.author = trainer()
        self.network = self.author.HypergraphGatedMILSurvival(bundle.dim, hidden=hidden, heads=heads,
            num_hyper_layers=hypergraph_layers, attn_dropout=attention_dropout, learnable_temp=learnable_temperature)
        self.batch_size, self.accumulation_steps, self.amp = batch_size, accumulation_steps, amp
        self.early_stopping, self.minimum_epochs = early_stopping, early_min_epochs
        self.maximum_epochs, self.patience = 200, 40
        self.completed_epochs = self.optimizer_steps = self.bad_epochs = 0
        self.best_ci, self.last_ci, self.training_done = None, None, False
        self.settings = {'profile': profile, 'dimension': bundle.dim, 'hidden': hidden, 'heads': heads,
            'hypergraph_layers': hypergraph_layers, 'attention_dropout': float(attention_dropout),
            'learnable_temperature': learnable_temperature, 'batch_size': batch_size,
            'accumulation_steps': accumulation_steps, 'amp_requested': amp, 'early_stopping': early_stopping,
            'early_min_epochs': early_min_epochs}
        self.contract = digest_object({'source': COMMIT, 'assets': self.assets.contract,
                                       'bundle': bundle.fingerprint, 'settings': self.settings})

    def configure_training(self, cfg):
        if (cfg['selection_metric'] != 'triagemil_selection' or cfg['max_instances'] is not None or
                cfg['grad_clip'] is not None):
            raise ValueError('TRIAGE-MIL preserves complete frozen MASS bags and native un-clipped Cox/VAL selection')
        self.maximum_epochs, self.patience = cfg['epochs'], cfg['patience']

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        # Counts only real successful steps, including source no-event and AMP
        # skips. The callback changes no gradients, parameters or optimizer state.
        optimizer.register_step_post_hook(self._count_update)
        return optimizer

    def _count_update(self, optimizer, args, kwargs):
        self.optimizer_steps += 1

    def loader(self, subset):
        return DataLoader(SelectedBags(self.assets, subset), batch_size=self.batch_size if subset == 'train' else 1,
            shuffle=subset == 'train', num_workers=0, collate_fn=self.author.collate_bags)

    def forward(self, features, H, return_attention=False):
        if (features.ndim != 2 or features.shape[1] != self.bundle.dim or not torch.isfinite(features).all() or
                not isinstance(H, torch.Tensor) or not H.is_sparse or
                H.shape[0] != len(features) or H.shape[1] < 1 or not H._nnz() or
                not torch.isfinite(H.values()).all()):
            raise ValueError('TRIAGE-MIL requires full-dimensional actual MASS features and a nonempty sparse hypergraph')
        return self.network(features, H, return_attn=return_attention)

    def epoch(self, bundle, optimizer, device='cpu'):
        self.train()
        before = self.optimizer_steps
        # Execute the original complete trainer, including per-epoch scaler,
        # batch-local risk sets, accumulation and batches without observed events.
        value = self.author.train_one_epoch(self.network, self.loader('train'), optimizer,
            torch.device(device), {'amp': self.amp, 'gradient_accumulation_steps': self.accumulation_steps})
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite original TRIAGE-MIL epoch objective')
        return [{'loss': float(value), 'actual_optimizer_updates': self.optimizer_steps-before}]

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        self.eval()
        # Original evaluate executes its actual DataLoader and float32 clinical
        # times, including its NaN guard for constant risk or no observed events.
        ci, native = self.author.evaluate(self.network, self.loader(subset), torch.device(device))
        rows = []
        for item in native.to_dict('records'):
            original = self.assets.by_patient[item['patient_id']]
            rows.append({'slide_id': original['slide_id'], 'patient_id': original['patient_id'],
                'global_patient_id': original['global_patient_id'], 'source_slides': original['slide_id'],
                'subset': subset, 'time': original['time'], 'event': original['event'],
                'native_time': item['time'], 'risk': item['risk']})
        frame = pd.DataFrame(rows)
        frame.attrs['triagemil_native_c_index'] = float(ci) if np.isfinite(ci) else None
        return frame

    def validation_metrics(self, frame):
        value = frame.attrs['triagemil_native_c_index']
        metrics = {'triagemil_native_c_index': value}
        if set(frame.subset) == {'val'}:
            # -1 is a selection-only sentinel outside the C-index domain, not
            # an invented C-index. selection_improved forbids saving it. The
            # true undefined native metric is separately exported as null.
            metrics.update(triagemil_selection=-1. if value is None else value,
                           triagemil_selection_defined=value is not None)
        return metrics

    def after_validation(self, metrics):
        self.last_ci = metrics['triagemil_native_c_index']
        if self.last_ci is not None and (self.best_ci is None or self.last_ci > self.best_ci):
            self.best_ci, self.bad_epochs = self.last_ci, 0
        else:
            self.bad_epochs += 1
        self.training_done = (self.early_stopping and self.completed_epochs >= self.minimum_epochs and
                              self.bad_epochs >= self.patience)
        if (self.training_done or self.completed_epochs == self.maximum_epochs) and self.best_ci is None:
            raise RuntimeError('Original TRIAGE-MIL finished without a defined validation-selected checkpoint')

    def selection_improved(self, objective, best):
        return objective >= 0 and objective > best

    def get_extra_state(self):
        return {'contract': self.contract, 'completed_epochs': self.completed_epochs,
            'optimizer_steps': self.optimizer_steps, 'bad_epochs': self.bad_epochs,
            'best_ci': self.best_ci, 'last_ci': self.last_ci, 'training_done': self.training_done,
            'maximum_epochs': self.maximum_epochs, 'patience': self.patience}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self.get_extra_state()) or state.get('contract') != self.contract:
            raise ValueError('TRIAGE-MIL checkpoint source/assets/configuration contract differs')
        if (any(type(state[k]) is not int or state[k] < 0 for k in
                ['completed_epochs', 'optimizer_steps', 'bad_epochs']) or
                any(type(state[k]) is not int or state[k] < 1 for k in ['maximum_epochs', 'patience']) or
                state['completed_epochs'] > state['maximum_epochs'] or
                type(state['training_done']) is not bool or
                any(state[k] is not None and (isinstance(state[k], bool) or not isinstance(state[k], (float, int)) or
                    not math.isfinite(state[k]) or not 0 <= state[k] <= 1) for k in ['best_ci', 'last_ci'])):
            raise ValueError('TRIAGE-MIL checkpoint counters or native selection state are malformed')
        done = (self.early_stopping and state['completed_epochs'] >= self.minimum_epochs and
                state['bad_epochs'] >= state['patience'])
        if state['training_done'] != done:
            raise ValueError('TRIAGE-MIL checkpoint stop flag disagrees with its original stopping rule')
        for key, value in state.items():
            if key != 'contract':setattr(self, key, copy.deepcopy(value))

    def protocol(self):
        return {'source_commit': COMMIT, 'settings': self.settings, 'assets_sha256': self.assets.contract,
            'preprocessing': self.assets.meta['settings'], 'completed_epochs': self.completed_epochs,
            'maximum_epochs': self.maximum_epochs, 'patience': self.patience,
            'early_stop_triggered': self.training_done,
            'optimizer_steps': self.optimizer_steps, 'best_val_c_index': self.best_ci,
            'last_val_c_index': self.last_ci, 'bad_epochs': self.bad_epochs,
            'selection': 'strict> native VAL lifelines C-index, first ties; undefined metrics do not select',
            'undefined_selection': 'null native C-index retained; selection-only sentinel-1 never saves a checkpoint',
            'native_defaults': 'AdamW2e-4/wd1e-4,200maxepochs,batch4,source accumulated minibatch Cox, no-event skips, no scheduler; bad>=40 stop after minimum60epochs',
            'Cox': 'source float32 times, decreasing argsort/logcumsumexp and observed-event mean; source tie handling not relabeled Breslow',
            'AMP': 'source requested AMP on CUDA, scaler freshly initialized each epoch; float32 Cox; no CPU mixed-precision substitution',
            'source_behaviors': ['MASS defaults: requested4096 and four axes.35/.35/.15/.15 with source quality fallback',
                'fewer clean tiles use actual K_effective; no fake nodes or unbound square coordinates',
                'source semantic clustering/containment/intra/inter hyperedges and full-D native model',
                'actual original epoch/evaluation and DataLoader RNG; successful optimizer steps counted without changing math'],
            'integration_changes': ['private verified original definitions and explicit file/clinical mapping',
                'requires actual coordinates, complete MASS assets and real sparse hypergraph; no source missing-coordinate/H=None fallback',
                'one explicitly selected diagnostic WSI per patient; no silent prefix-first file resolution',
                'TEST sealed until explicit evaluate; optimizer/RNG/stopping state added for exact epoch-boundary resume',
                'original clinical table precision retained alongside float32 native times'],
            'synthetic': bool(self.bundle.config.get('synthetic', False))}

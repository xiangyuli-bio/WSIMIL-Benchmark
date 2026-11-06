"""Complete released three-scale ZoomMIL training and explicit saver repairs."""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler

from .io import digest_object
from .zoommil_source import COMMIT, build_model

PROFILES = {'zoommil_bright_source_repaired': (12, 'zoommil_weighted_f1'),
            'zoommil_crc_source_repaired': (16, 'zoommil_source_ce')}


class SlideDataset(Dataset):
    def __init__(self, assets, rows):
        self.assets, self.rows = assets, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (*self.assets.load(row), row['target'], index)


class ZoomMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='zoommil_bright_source_repaired', k=None, sigma=.002,
                 dropout=.25, scheduler_factor=.8, scheduler_patience=5, num_workers=8, max_dense_bytes=32*1024**3):
        super().__init__()
        if (profile not in PROFILES or bundle.task_type != 'classification'
                or bundle.config.get('training_unit', 'slide') != 'slide' or bundle.zoommil is None):
            raise ValueError('ZoomMIL requires its registered three-scale slide classification assets')
        if (type(scheduler_factor) not in {int, float} or not 0 < scheduler_factor < 1
                or type(scheduler_patience) is not int or scheduler_patience < 0
                or type(num_workers) is not int or num_workers < 0
                or type(max_dense_bytes) is not int or max_dense_bytes < 1):
            raise ValueError('ZoomMIL scheduler and full-memory budget are invalid')
        self.bundle, self.assets, self.profile = bundle, bundle.zoommil, profile
        self.k = PROFILES[profile][0] if k is None else k
        self.selection_metric = PROFILES[profile][1]
        self.network = build_model(bundle.dim, len(bundle.classes), self.k, sigma, dropout)
        self.settings = {'k': self.k, 'sigma': sigma, 'dropout': dropout,
                         'scheduler_factor': scheduler_factor, 'scheduler_patience': scheduler_patience,
                         'num_workers': num_workers,
                         'max_dense_bytes': max_dense_bytes}
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'settings': self.settings,
                                       'assets': self.assets.contract, 'bundle': bundle.fingerprint})
        self.recipe, self.pending_scheduler = None, None
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders, self.lr_history = [], []

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['patience'] != cfg['epochs'] or cfg['selection_metric'] != self.selection_metric):
            raise ValueError('ZoomMIL uses full hierarchy, no clipping/early stop and its explicit source selection rule')
        self.recipe = {'epochs': cfg['epochs'], 'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'],
                       'selection_metric': self.selection_metric}

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(.9, .999),
                                           eps=1e-8, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='min',
            factor=self.settings['scheduler_factor'], patience=self.settings['scheduler_patience'])
        if self.pending_scheduler is not None:
            self._scheduler.load_state_dict(self.pending_scheduler)
        return self._optimizer

    def validate_optimizer_state(self):
        group = self._optimizer.param_groups
        expected_lr = self.lr_history[-1]['after_lr'] if self.lr_history else self.recipe['lr']
        if (len(group) != 1 or group[0]['lr'] != expected_lr or group[0]['betas'] != (.9, .999)
                or group[0]['eps'] != 1e-8 or group[0]['weight_decay'] != self.recipe['weight_decay']
                or {id(p) for p in group[0]['params']} != {id(p) for p in self.network.parameters()}
                or self._scheduler.last_epoch != self.completed_epochs):
            raise ValueError('ZoomMIL optimizer, scheduler or parameter ownership changed')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('ZoomMIL epochs must advance the complete restored source recipe')
        self.epoch_index = epoch

    def _reserve(self, row):
        n1, n2, n3 = self.assets.meta['slides'][row['slide_id']]['nodes_per_level']
        k1, b2, b3 = min(n1, self.k), n2//n1, n3//n1
        k2, b32 = min(k1*b2, self.k), n3//n2
        elements = ((n1+n2+n3)*self.bundle.dim + k1*b2*n2 + k1*b3*n3
                    + k2*b32*k1*b3 + 100*k1*n1 + 100*k2*k1*b2)
        if elements*4 > self.settings['max_dense_bytes']:
            raise ValueError('ZoomMIL full source Kronecker/MonteCarlo graph exceeds the declared memory budget; no row truncation')

    def _loader(self, subset):
        rows = self.bundle.records(subset)
        for row in rows:
            self._reserve(row)
        dataset = SlideDataset(self.assets, rows)
        if subset == 'train':
            counts = np.bincount([row['target'] for row in rows], minlength=len(self.bundle.classes))
            if (counts == 0).any():
                raise ValueError('ZoomMIL weighted sampler needs every training class')
            weights = torch.tensor([len(rows)/counts[row['target']] for row in rows], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, len(rows), replacement=True)
        else:
            sampler = SequentialSampler(dataset)
        # Keep the source default8 readers. Explicit workers0 is also available
        # for constrained software tests; features have no random transform.
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=self.settings['num_workers']), rows

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('ZoomMIL train inputs or actual Adam changed')
        self.train(); optimizer.zero_grad()
        loader, rows = self._loader('train')
        losses, order = [], []
        for batch in loader:
            features = tuple(value.to(device) for value in batch[:3])
            target = batch[3].to(device)
            logits, _, _ = self.network(features)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('ZoomMIL source objective became nonfinite')
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            self.optimizer_updates += 1
            order.append(rows[int(batch[4])]['slide_id'])
            losses.append({'loss': float(loss.detach())})
        self.train_orders.append(order)
        self.completed_epochs = self.epoch_index+1
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('ZoomMIL inference needs the registered validation/test subset')
        self.eval(); loader, rows = self._loader(subset); predictions = []
        for batch in loader:
            row = rows[int(batch[4])]
            logits, label, probability = self.network(tuple(value.to(device) for value in batch[:3]))
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], target=row['target'], prediction=int(label),
                          zoommil_source_ce=float(F.cross_entropy(logits, batch[3].to(device))))
            record.update({f'p_{index}': float(value) for index, value in enumerate(probability[0])})
            predictions.append(record)
        return pd.DataFrame(predictions)

    def validation_metrics(self, frame):
        loss = frame['zoommil_source_ce'].to_numpy(dtype=float)
        if not len(loss) or not np.isfinite(loss).all() or not frame.slide_id.is_unique:
            raise ValueError('ZoomMIL source validation needs finite CE and one record per true slide')
        return {'zoommil_source_ce': sum(loss.tolist())/len(loss),
                'zoommil_weighted_f1': float(f1_score(frame.target, frame.prediction, average='weighted'))}

    def after_validation(self, metrics):
        before = self._optimizer.param_groups[0]['lr']
        self._scheduler.step(metrics['zoommil_source_ce'])
        self.lr_history.append({'epoch': self.epoch_index, 'validation_ce': metrics['zoommil_source_ce'],
                                'before_lr': before, 'after_lr': self._optimizer.param_groups[0]['lr']})

    def selection_objective(self, metrics, epoch, current):
        value = metrics[self.selection_metric]
        return -value if self.selection_metric == 'zoommil_source_ce' else value

    def get_extra_state(self):
        scheduler = self._scheduler.state_dict() if hasattr(self, '_scheduler') else self.pending_scheduler
        return {'contract': self.contract, 'recipe': copy.deepcopy(self.recipe),
                'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
                'optimizer_updates': self.optimizer_updates, 'train_orders': copy.deepcopy(self.train_orders),
                'lr_history': copy.deepcopy(self.lr_history), 'scheduler': copy.deepcopy(scheduler)}

    def set_extra_state(self, state):
        keys = {'contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders', 'lr_history', 'scheduler'}
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.contract
                or any(type(state[key]) is not int for key in ['epoch_index', 'completed_epochs', 'optimizer_updates'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or len(state['train_orders']) != state['completed_epochs'] or len(state['lr_history']) != state['completed_epochs']):
            raise ValueError('ZoomMIL complete epoch/source/input state changed')
        train = {row['slide_id'] for row in self.bundle.records('train')}
        if (any(len(order) != len(train) or not set(order) <= train for order in state['train_orders'])
                or state['optimizer_updates'] != len(train)*state['completed_epochs']):
            raise ValueError('ZoomMIL checkpoint omitted full weighted draws or actual Adam updates')
        if self.recipe is not None and state['recipe'] != self.recipe:
            raise ValueError('ZoomMIL checkpoint changed the declared complete training budget')
        if state['completed_epochs']:
            recipe, scheduler = state['recipe'], state['scheduler']
            if (not isinstance(recipe, dict) or set(recipe) != {'epochs', 'lr', 'weight_decay', 'selection_metric'}
                    or recipe['selection_metric'] != self.selection_metric or recipe['epochs'] < state['completed_epochs']
                    or not isinstance(scheduler, dict) or scheduler['last_epoch'] != state['completed_epochs']
                    or scheduler['factor'] != self.settings['scheduler_factor']
                    or scheduler['patience'] != self.settings['scheduler_patience']
                    or scheduler['_last_lr'] != [state['lr_history'][-1]['after_lr']]
                    or [item['epoch'] for item in state['lr_history']] != list(range(state['completed_epochs']))):
                raise ValueError('ZoomMIL actual plateau scheduler history or source recipe is incomplete')
        for key in ['recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders', 'lr_history']:
            setattr(self, key, copy.deepcopy(state[key]))
        self.pending_scheduler = copy.deepcopy(state['scheduler'])
        if hasattr(self, '_scheduler'):
            self._scheduler.load_state_dict(self.pending_scheduler)

    def protocol(self):
        return {'method': self.profile, 'source': COMMIT, 'settings': self.settings,
            'training': 'Full source512/256 three-level network,100-sample Gaussian differentiableTopK, dual auxiliary gates and original Kronecker grouping; native100fixed epochs, FP32 CE/Adam, no clipping/early stopping',
            'sampling': 'source inverse-class-frequency WeightedRandomSampler, replacementTrue, Ndraws/epoch, batch1; actual DataLoader/base seed and native8 readers by default, explicit workers0 software port available',
            'scheduler': 'actual ReduceLROnPlateau on mean slide VAL CE, factor0.8/patience5 defaults; full state restored',
            'selection': self.selection_metric,
            'saver_repairs': 'Define first valid F1==0 checkpoint; supply actually computed VAL CE missing from source loss-saver summary; keep strict first optimum',
            'magnifications': [level['magnification'] for level in self.assets.meta['schema']['levels']],
            'source_geometry': 'actual recursive addresses, complete descendants, canonical source quadrant order; no invented row-grid coordinates',
            'inference_scope': 'released complete cached-feature three-scale route; no claim of paper two-levelEff or online FM extraction/totalWSI efficiency',
            'input_dimension': self.bundle.dim, 'completed_epochs': self.completed_epochs,
            'actual_optimizer_updates': self.optimizer_updates, 'train_orders_sha256': digest_object(self.train_orders),
            'learning_rate_history': copy.deepcopy(self.lr_history), 'checkpoint': 'all model/Adam/plateau/order/counter/CPU-CUDA RNG',
            'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

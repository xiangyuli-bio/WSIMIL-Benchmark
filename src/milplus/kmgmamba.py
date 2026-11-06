"""Complete released KMG-Mamba per-slide training and validation-AUC selection."""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .io import digest_object
from .kmgmamba_source import COMMIT, PROFILES, build_model


class SlideDataset(Dataset):
    def __init__(self, assets, rows):
        self.assets, self.rows = assets, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        low, _, high, _ = self.assets.load(row)
        return low, high, row['target'], index


def collate(batch):
    if len(batch) != 1:
        raise ValueError('KMG-Mamba requires one complete slide pair per update')
    low, high, target, index = batch[0]
    return low, high, torch.tensor([target]), index


class KMGMamba(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='kmgmamba_source', num_workers=None,
                 scan_backend='torch', scan_chunk=64, sequence_length_min=0):
        super().__init__()
        if (profile not in PROFILES or bundle.task_type != 'classification'
                or bundle.config.get('training_unit', 'slide') != 'slide' or bundle.kmgmamba is None):
            raise ValueError('KMG-Mamba requires measured dual-scale features for slide classification')
        if num_workers is not None and (type(num_workers) is not int or num_workers < 0):
            raise ValueError('KMG-Mamba num_workers must be null or a nonnegative integer')
        self.bundle, self.assets, self.profile = bundle, bundle.kmgmamba, profile
        self.network, self.backend = build_model(bundle.dim, len(bundle.classes), profile,
                                                scan_backend, scan_chunk, sequence_length_min)
        self.sequence_length_min, self.num_workers = sequence_length_min, num_workers
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'assets': self.assets.contract,
            'bundle': bundle.fingerprint, 'num_workers': num_workers, 'backend': self.backend,
            'sequence_length_min': sequence_length_min})
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.recipe, self.best_auc = None, None
        self.early_bad, self.training_done = 0, False
        self.train_orders, self.selection_history = [], []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'kmgmamba_val_auc':
            raise ValueError('KMG-Mamba requires full dual-scale bags, all three source losses and validation-AUC selection')
        self.recipe = {key: cfg[key] for key in ['epochs', 'patience', 'lr', 'weight_decay', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay,
                                           betas=(.9, .999), eps=1e-8)
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 1 or groups[0]['lr'] != self.recipe['lr']
                or groups[0]['weight_decay'] != self.recipe['weight_decay']
                or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('KMG-Mamba Adam state or complete parameter ownership changed')
        if any(int(state.get('step', -1)) != self.optimizer_updates for state in self._optimizer.state.values()):
            raise ValueError('KMG-Mamba checkpoint Adam update counts changed')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('KMG-Mamba requires sequential complete epochs')
        self.epoch_index = epoch

    def _loader(self, subset, device):
        rows = self.bundle.records(subset)
        dataset = SlideDataset(self.assets, rows)
        sampler = RandomSampler(dataset) if subset == 'train' else SequentialSampler(dataset)
        readers = (4 if torch.device(device).type == 'cuda' else 0) if self.num_workers is None else self.num_workers
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=readers, collate_fn=collate), rows

    def pair_forward(self, low, high, labels):
        if (low.ndim != 2 or high.ndim != 2 or low.shape[1] != self.bundle.dim
                or high.shape[1] != self.bundle.dim or not len(low) or not len(high)):
            raise ValueError('KMG-Mamba expects two nonempty full-dimensional feature sequences')
        factor = self.sequence_length_min+1 if len(low) < self.sequence_length_min else 1
        if min(len(low), len(high))*factor < 9:
            raise ValueError('KMG-Mamba source graph needs nine nodes per scale after its explicit repeat rule')
        output = self.network(low, high, labels)
        if any(not torch.isfinite(value).all() for value in [output[0], *output[2:]]):
            raise FloatingPointError('KMG-Mamba original forward/objective is nonfinite')
        return output

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('KMG-Mamba actual input or optimizer changed')
        self.train(); loader, rows = self._loader('train', device)
        losses, order = [], []
        optimizer.zero_grad()
        for low, high, labels, index in loader:
            result = self.pair_forward(low.to(device), high.to(device), labels.to(device))
            result[2].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError('KMG-Mamba source backward produced nonfinite gradients')
            optimizer.step(); optimizer.zero_grad()
            self.optimizer_updates += 1
            order.append(rows[index]['slide_id'])
            losses.append({'loss': float(result[2].detach()), 'bag_ce': float(result[4].detach()),
                           'coarse_prototype_svm': float(result[5].detach()),
                           'fine_prototype_svm': float(result[6].detach())})
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(order)
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('KMG-Mamba prediction requires the registered validation/test inputs')
        self.eval(); loader, rows = self._loader(subset, device); records = []
        for low, high, labels, index in loader:
            result = self.pair_forward(low.to(device), high.to(device), labels.to(device))
            row = rows[index]
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], target=row['target'],
                          prediction=int(result[1]), kmgmamba_source_loss=float(result[2]))
            record.update({f'p_{i}': float(v) for i, v in enumerate(result[0][0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        if not len(frame) or not frame.slide_id.is_unique or not np.isfinite(frame.kmgmamba_source_loss).all():
            raise ValueError('KMG-Mamba validation needs finite source loss for every actual slide')
        classes = len(self.bundle.classes)
        probability = frame[[f'p_{i}' for i in range(classes)]].to_numpy()
        value = roc_auc_score(frame.target, probability[:, 1]) if classes == 2 else roc_auc_score(
            frame.target, probability, multi_class='ovr')
        if not np.isfinite(value):
            raise ValueError('KMG-Mamba source validation AUC is undefined for this split')
        return {'kmgmamba_val_auc': float(value)}

    def after_validation(self, metrics):
        auc = metrics['kmgmamba_val_auc']
        if self.best_auc is None or auc >= self.best_auc:
            self.best_auc, self.early_bad = auc, 0
        else:
            self.early_bad += 1
        self.training_done = self.early_bad >= self.recipe['patience'] and self.epoch_index > 50
        self.selection_history.append({'epoch': self.epoch_index, 'auc': auc,
            'best_auc': self.best_auc, 'bad': self.early_bad, 'done': self.training_done})

    def selection_objective(self, metrics, epoch, current):
        return metrics['kmgmamba_val_auc']

    def selection_improved(self, objective, best):
        return objective >= best

    def get_extra_state(self):
        return copy.deepcopy({key: getattr(self, key) for key in
            ['contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates',
             'best_auc', 'early_bad', 'training_done', 'train_orders', 'selection_history']})

    def set_extra_state(self, state):
        keys = {'contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates',
                'best_auc', 'early_bad', 'training_done', 'train_orders', 'selection_history'}
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.contract
                or any(type(state[k]) is not int for k in ['epoch_index', 'completed_epochs', 'optimizer_updates', 'early_bad'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or len(state['train_orders']) != state['completed_epochs']
                or len(state['selection_history']) != state['completed_epochs']
                or state['early_bad'] < 0 or type(state['training_done']) is not bool):
            raise ValueError('KMG-Mamba complete source/input/counter state changed')
        train = {row['slide_id'] for row in self.bundle.records('train')}
        if (state['optimizer_updates'] != len(train)*state['completed_epochs']
                or any(len(order) != len(train) or set(order) != train for order in state['train_orders'])):
            raise ValueError('KMG-Mamba checkpoint omitted source samples or Adam updates')
        if self.recipe is not None and state['recipe'] != self.recipe:
            raise ValueError('KMG-Mamba checkpoint changed the declared complete training recipe')
        best, bad, done = None, 0, False
        for epoch, record in enumerate(state['selection_history']):
            auc = record.get('auc')
            if type(auc) not in {int, float} or not np.isfinite(auc) or not 0 <= auc <= 1:
                raise ValueError('KMG-Mamba validation-AUC history is invalid')
            if best is None or auc >= best:
                best, bad = auc, 0
            else:
                bad += 1
            done = bad >= state['recipe']['patience'] and epoch > 50
            if record != {'epoch': epoch, 'auc': auc, 'best_auc': best, 'bad': bad, 'done': done}:
                raise ValueError('KMG-Mamba latest-tie selection or early-stop history changed')
        if (best, bad, done) != (state['best_auc'], state['early_bad'], state['training_done']):
            raise ValueError('KMG-Mamba selected optimum or stopping counters changed')
        if state['completed_epochs'] and (not isinstance(state['recipe'], dict)
                                          or state['recipe']['epochs'] < state['completed_epochs']):
            raise ValueError('KMG-Mamba saved training budget is incomplete')
        for key in keys-{'contract'}:
            setattr(self, key, copy.deepcopy(state[key]))

    def protocol(self):
        return {'method': self.profile, 'source': COMMIT, 'input_dimension': self.bundle.dim,
            'architecture': 'complete released96width,9-neighbor feature graph,top-degree prototype,8-head prototype attention,cross-scale4-state/96-channel scan,128attention head',
            'backend': self.backend, 'sequence_length_min': self.sequence_length_min,
            'objective': '.5 bagCE + .5(coarsePrototypeSmoothTop1SVM + finePrototypeSmoothTop1SVM)',
            'recipe': self.recipe, 'native_recipe': '100maxepochs,Adam2e-4/wd1e-5,batch1,RandomSampler,source4CUDA/0CPU readers; declared Plateau scheduler never stepped',
            'selection': 'VAL AUROC(binary) or macroOVR(multiclass repair), latest exact tie,patience20 and zero-based epoch>50',
            'source_inactive_modules': 'source creates but never calls depthwiseConv2d,MLP/norm2,and CrossMamba.norm; preserve parameters and initialization without claiming their gradients',
            'repairs': 'explicit device placement; multiclass profile alone changes source2 attention and prototype heads to actualC',
            'inputs': 'two genuine measured feature scales with declared provenance; preserve both unequal-length full row orders; no fabricated coordinates or cross-scale interpolation',
            'completed_epochs': self.completed_epochs, 'actual_optimizer_updates': self.optimizer_updates,
            'train_orders_sha256': digest_object(self.train_orders), 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

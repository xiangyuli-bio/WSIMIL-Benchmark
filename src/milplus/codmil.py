"""Complete CoD source-repaired two-scale/text recipe and checkpoint state."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler

from .codmil_network import COMMIT, CoDReference
from .io import digest_object

PROFILE = 'codmil_source_repaired'


class SlideRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def one_row(items):
    return items[0]


class CoDMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile=PROFILE, hidden=192, optimizer='adam',
                 weighted_sampling=False, early_stopping=True, stop_epoch=50,
                 max_dense_bytes=32*1024**3):
        super().__init__()
        if profile != PROFILE or bundle.task_type != 'classification' or bundle.codmil is None:
            raise ValueError('CoD requires classification with bound actual physical two-scale/text assets')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('CoD source recipe requires one slide per training bag')
        if (optimizer not in {'adam', 'sgd'} or type(weighted_sampling) is not bool
                or type(early_stopping) is not bool or type(stop_epoch) is not int or stop_epoch < 0
                or type(max_dense_bytes) is not int or max_dense_bytes < 1):
            raise ValueError('Invalid native CoD optimizer/sampler/stopping/memory configuration')
        self.bundle, self.assets, self.profile = bundle, bundle.codmil, profile
        self.core = CoDReference(bundle.dim, self.assets.text.shape[1], len(bundle.classes), hidden)
        self.register_buffer('text', self.assets.text.clone())
        self.settings = {'hidden': hidden, 'optimizer': optimizer, 'weighted_sampling': weighted_sampling,
            'early_stopping': early_stopping, 'stop_epoch': stop_epoch, 'max_dense_bytes': max_dense_bytes}
        self.contract = digest_object({'source': COMMIT, 'assets': self.assets.contract,
            'settings': self.settings, 'classes': bundle.classes,
            'recipe': 'complete source combined main/masked-normal CE and one-bag Adam/SGD, explicit C/D/T ports'})
        self.stop_after_epoch = stop_epoch+1
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders = []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('CoD requires complete registered scales, source selection and no generic clipping')
        if cfg['selection_metric'] != 'codmil_val_loss':
            raise ValueError('CoD source selection uses complete main plus masked-normal VAL loss')

    def make_optimizer(self, lr, weight_decay):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.settings['optimizer'] == 'sgd':
            return torch.optim.SGD(parameters, lr=lr, momentum=.9, weight_decay=weight_decay)
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    def reserve(self, row):
        record = self.assets.rows[row['slide_id']]
        estimated = (32*sum(p.numel() for p in self.parameters())+
            self.bundle.dim*(64*record['coarse_count']+8*record['fine_count']+8*len(self.text)))
        if estimated > self.settings['max_dense_bytes']:
            raise ValueError('CoD complete source feature/text graph exceeds max_dense_bytes before allocation')

    def prepare(self, bundle, seed, device):
        for row in bundle.rows.to_dict('records'):
            self.reserve(row)

    def start_epoch(self, epoch):
        self.epoch_index = epoch

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1 or config.get('max_instances') is not None:
            raise ValueError('CoD requires one complete registered physical WSI, without generic subsampling')
        row = rows[0]; self.reserve(row)
        coarse, fine, mapping = self.assets.load(row)
        result = self.core(coarse.to(device), fine.to(device), self.text, mapping.to(device),
            torch.tensor([row['target']], dtype=torch.long, device=device))
        return result

    def epoch(self, bundle, optimizer, device):
        dataset = SlideRows(bundle.records('train'))
        if self.settings['weighted_sampling']:
            counts = np.bincount([r['target'] for r in dataset.rows], minlength=len(bundle.classes))
            weights = torch.tensor([len(dataset)/counts[r['target']] for r in dataset.rows], dtype=torch.float64)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(dataset)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=one_row)
        losses, order = [], []
        for row in loader:
            result = self.forward_unit(bundle, [row], device, {}, train=True)
            result['loss'].backward()
            optimizer.step(); optimizer.zero_grad()
            self.optimizer_updates += 1; order.append(row['slide_id'])
            losses.append({'loss': float(result['loss'].detach())})
        self.train_orders.append(order)
        self.completed_epochs = self.epoch_index+1
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        self.eval()
        dataset = SlideRows(bundle.records(subset))
        loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=0, collate_fn=one_row)
        predictions = []
        for row in loader:
            result = self.forward_unit(bundle, [row], device, cfg)
            record = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'target']}
            record['source_slides'] = row['slide_id']
            record['codmil_total_loss'] = float(result['loss'])
            record.update({f'p_{i}': float(v) for i, v in enumerate(result['probabilities'][0])})
            predictions.append(record)
        return pd.DataFrame(predictions)

    def validation_metrics(self, frame):
        values = frame['codmil_total_loss'].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('CoD validation complete source objective is missing or nonfinite')
        return {'codmil_val_loss': float(values.mean())}

    def selection_objective(self, metrics, epoch, current):
        return -metrics['codmil_val_loss']

    def selection_improved(self, objective, best):
        return objective > best if self.settings['early_stopping'] else True

    def get_extra_state(self):
        return {'contract': self.contract, 'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
                'optimizer_updates': self.optimizer_updates, 'train_orders': self.train_orders}

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'contract', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders'}
                or state.get('contract') != self.contract):
            raise ValueError('CoD physical/text/source/recipe contract changed')
        if (any(type(state[k]) is not int or state[k] < 0 for k in ['completed_epochs', 'optimizer_updates'])
                or type(state['epoch_index']) is not int or state['epoch_index'] != state['completed_epochs']-1
                or not isinstance(state['train_orders'], list)):
            raise ValueError('CoD checkpoint epoch/update state is inconsistent')
        allowed = {r['slide_id'] for r in self.bundle.records('train')}
        if (len(state['train_orders']) != state['completed_epochs'] or
                any(len(order) != len(allowed) or not set(order) <= allowed for order in state['train_orders'])):
            raise ValueError('CoD checkpoint contains wrong epoch sampler length or non-TRAIN identities')
        if not self.settings['weighted_sampling'] and any(set(order) != allowed for order in state['train_orders']):
            raise ValueError('CoD source random sampling must visit every TRAIN bag exactly once')
        if state['optimizer_updates'] != sum(len(x) for x in state['train_orders']):
            raise ValueError('CoD optimizer-update count differs from complete TRAIN order')
        self.epoch_index, self.completed_epochs = state['epoch_index'], state['completed_epochs']
        self.optimizer_updates, self.train_orders = state['optimizer_updates'], state['train_orders']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'text' not in state_dict or not torch.equal(state_dict['text'].cpu(), self.assets.text):
            raise ValueError('CoD checkpoint text differs from bound actual prompt embeddings')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {**self.core.protocol(), 'profile': self.profile, 'settings': self.settings,
            'physical_assets_binding': self.assets.contract, 'full_training_recipe': 'complete_source_two_scale_and_text_masking_recipe',
            'text_condition': self.assets.manifest['text_metadata']['alignment_policy'],
            'text_role_order': ['low_classes', 'high_classes', 'normal_concepts', 'source_excluded_last'],
            'objective': 'main low+high CE plus1x masked-low class/normal-text CE, no invented patch annotations',
            'input_ports': 'C-way slices and full-D/T declared; exact native1024/1024/C2 control; original source missing-scale/path and singleton-mask repairs explicit',
            'sampling': 'Original DataLoader B1 RandomSampler, or explicit source WeightedRandomSampler N/class_count replacement; workers0, no feature augmentation. Sequential validation DataLoader retains actual source RNG base-seed consumption.',
            'optimization': 'Source Adam or declared source SGD. Launcherlr1e-4 wd1e-5,200maxepochs; scheduler instantiated but never stepped in original source, constant native LR preserved.',
            'selection': 'First strict best complete VAL loss; original patience20 and epoch_index>50 stopping condition (transFG80 branch not this CoD model). Disabling source early_stopping selects last completed epoch.',
            'evaluation_unit': self.bundle.config.get('evaluation_unit', 'slide'),
            'evaluation_extension': 'Patient fusion if explicitly requested is an outer benchmark extension, not a change to source one-slide training/VAL loss.',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'checkpoint': 'Complete source state including inactive declared heads, actual text buffer, optimizer moments, TRAIN orders, epoch and all RNG; changed physical/text assets rejected',
            'acceptance_evidence': 'tracked separately by release validation reports',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

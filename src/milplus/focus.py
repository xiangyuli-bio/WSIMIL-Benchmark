"""Complete original FOCUS trainable text/visual recipe and exact state."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler

from .focus_network import COMMIT, FOCUSReference
from .io import digest_object

PROFILE = 'focus_source'


class SlideRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def one_row(items):
    return items[0]


class FOCUSMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile=PROFILE, optimizer='adam', window_size=8, sim_threshold=.8, max_context_length=8192,
                 weighted_sampling=False, early_stopping=True, stop_epoch=40,
                 max_dense_bytes=32*1024**3):
        super().__init__()
        if profile != PROFILE or bundle.task_type != 'classification' or bundle.focus is None:
            raise ValueError('FOCUS requires classification with bound actual pretrained CONCH weights, ordered prompts and feature rows')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('FOCUS source recipe requires one slide per training bag')
        if (optimizer not in {'adam', 'sgd'} or type(weighted_sampling) is not bool
                or type(early_stopping) is not bool or type(stop_epoch) is not int or stop_epoch < 0
                or type(max_dense_bytes) is not int or max_dense_bytes < 1):
            raise ValueError('Invalid native FOCUS optimizer/sampler/stopping/memory configuration')
        self.bundle, self.assets, self.profile = bundle, bundle.focus, profile
        self.core = FOCUSReference(self.assets.weights, self.assets.prompts, bundle.dim, window_size, sim_threshold, max_context_length)
        self.settings = {'optimizer': optimizer, 'window_size': window_size, 'sim_threshold': sim_threshold,
            'max_context_length': max_context_length, 'weighted_sampling': weighted_sampling,
            'early_stopping': early_stopping, 'stop_epoch': stop_epoch, 'max_dense_bytes': max_dense_bytes}
        self.contract = digest_object({'source': COMMIT, 'assets': self.assets.contract,
            'settings': self.settings, 'classes': bundle.classes,
            'recipe': 'complete original FOCUS one-bag CE/full pretrained text and prompt updates/VAL-error firstbest'})
        self.stop_after_epoch = stop_epoch+1
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders = []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('FOCUS requires complete ordered bags, original token selection and no generic clipping')
        if cfg['selection_metric'] != 'focus_val_error':
            raise ValueError('FOCUS source selection uses VAL misclassification error, not CE or AUC')

    def make_optimizer(self, lr, weight_decay):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.settings['optimizer'] == 'sgd':
            return torch.optim.SGD(parameters, lr=lr, momentum=.9, weight_decay=weight_decay)
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    def reserve(self, row):
        record = self.assets.rows[row['slide_id']]
        estimated = 32*sum(p.numel() for p in self.parameters())+64*self.bundle.dim*record['count']
        if estimated > self.settings['max_dense_bytes']:
            raise ValueError('FOCUS full trainable text and complete visual graph exceeds max_dense_bytes')

    def prepare(self, bundle, seed, device):
        for row in bundle.rows.to_dict('records'):
            self.reserve(row)

    def start_epoch(self, epoch):
        self.epoch_index = epoch

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1 or config.get('max_instances') is not None:
            raise ValueError('FOCUS requires one complete ordered slide bag, without generic subsampling')
        row = rows[0]; self.reserve(row)
        features = self.assets.load(row)
        return self.core(features.to(device), torch.tensor([row['target']], dtype=torch.long, device=device))

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
            record['focus_ce'] = float(result['loss'])
            record.update({f'p_{i}': float(v) for i, v in enumerate(result['probabilities'][0])})
            predictions.append(record)
        return pd.DataFrame(predictions)

    def validation_metrics(self, frame):
        values = frame['focus_ce'].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('FOCUS validation complete source objective is missing or nonfinite')
        return {'focus_val_error': float((frame[[f'p_{i}' for i in range(len(self.bundle.classes))]].to_numpy().argmax(1) != frame['target'].to_numpy()).mean()),
                'focus_val_ce': float(values.mean())}

    def selection_objective(self, metrics, epoch, current):
        return -metrics['focus_val_error']

    def selection_improved(self, objective, best):
        return objective > best if self.settings['early_stopping'] else True

    def get_extra_state(self):
        return {'contract': self.contract, 'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
                'optimizer_updates': self.optimizer_updates, 'train_orders': self.train_orders}

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'contract', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders'}
                or state.get('contract') != self.contract):
            raise ValueError('FOCUS ordered feature/text/source/recipe contract changed')
        if (any(type(state[k]) is not int or state[k] < 0 for k in ['completed_epochs', 'optimizer_updates'])
                or type(state['epoch_index']) is not int or state['epoch_index'] != state['completed_epochs']-1
                or not isinstance(state['train_orders'], list)):
            raise ValueError('FOCUS checkpoint epoch/update state is inconsistent')
        allowed = {r['slide_id'] for r in self.bundle.records('train')}
        if (len(state['train_orders']) != state['completed_epochs'] or
                any(len(order) != len(allowed) or not set(order) <= allowed for order in state['train_orders'])):
            raise ValueError('FOCUS checkpoint contains wrong epoch sampler length or non-TRAIN identities')
        if not self.settings['weighted_sampling'] and any(set(order) != allowed for order in state['train_orders']):
            raise ValueError('FOCUS source random sampling must visit every TRAIN bag exactly once')
        if state['optimizer_updates'] != sum(len(x) for x in state['train_orders']):
            raise ValueError('FOCUS optimizer-update count differs from complete TRAIN order')
        self.epoch_index, self.completed_epochs = state['epoch_index'], state['completed_epochs']
        self.optimizer_updates, self.train_orders = state['optimizer_updates'], state['train_orders']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in ['token_prefix', 'token_suffix']:
            key = 'core.network.prompt_learner.'+name
            if key not in state_dict or not torch.equal(state_dict[key].cpu(), getattr(self.core.network.prompt_learner, name).cpu()):
                raise ValueError('FOCUS checkpoint fixed prompt buffers differ from registered native text assets')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {**self.core.protocol(), 'profile': self.profile, 'settings': self.settings,
            'assets_binding': self.assets.contract,
            'feature_condition': self.assets.manifest['metadata']['alignment_policy'],
            'feature_preprocessing': self.assets.manifest['metadata']['feature_preprocessing'],
            'objective': 'Original one-bag cross entropy; complete pretrained text transformer and shared prompt updated jointly with visual branch',
            'sampling': 'Original RandomSampler or source weighted replacement N/class_count, B1, workers0; full row order preserved, no generic random truncation. Sequential VAL DataLoader consumes base-seed RNG as original.',
            'optimization': 'Source Adam or explicit source SGD(momentum0.9); lr1e-4 wd1e-5,200max; instantiated ReduceLROnPlateau never stepped, source constant LR preserved.',
            'selection': 'First strict best complete VAL slide error, source patience20 and epoch_index>40; ties increment counter, no VAL-CE/AUC substitution. Disabled early-stopping selects final epoch.',
            'evaluation_unit': self.bundle.config.get('evaluation_unit', 'slide'),
            'evaluation_extension': 'Patient fusion only if explicitly requested; source one-slide training and VAL-error selection retained.',
            'checkpoint': 'All actual pretrained text/visual/prompt tensors, fixed token-prefix/suffix buffers, optimizer moments, TRAIN orders and all RNG; changed prompt/weight/ordered-input binding rejected.',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'acceptance_evidence': 'tracked separately by release validation reports',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

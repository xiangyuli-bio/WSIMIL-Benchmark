"""Complete M4 multi-endpoint source recipe; native/release admission pending."""
import copy
import math
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .io import digest_file, digest_object
from .m4_source import COMMIT, build_original
from .upstream import load_definitions


def author_training():
    return load_definitions('m4', 'train_val_multi.py', ['train', 'evaluate'],
        {'sys': sys, 'np': np, 'torch': torch, 'nn': nn, 'tqdm': tqdm,
         'roc_auc_score': roc_auc_score, 'confusion_matrix': confusion_matrix})


def author_collate():
    return load_definitions('m4', 'Dataset.py', ['collate_MIL'], {'torch': torch}).collate_MIL


class Bags(Dataset):
    def __init__(self, owner, subset):
        self.owner, self.rows, self.order = owner, owner.bundle.records(subset), []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        self.owner.check_inputs()
        value, _ = self.owner.bundle.load(row)
        if value.ndim != 2 or len(value) < 1 or value.shape[1] != self.owner.dimension:
            raise ValueError('M4 requires complete nonempty source feature bags')
        self.order.append(index)
        return value, np.asarray(row['targets'], dtype=np.int64)


class M4MIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='m4_multilabel_source', experts=5):
        super().__init__()
        if profile != 'm4_multilabel_source' or bundle.task_type != 'multilabel':
            raise ValueError('M4 requires explicitly complete independent binary endpoint supervision')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('Original M4 trains individual slide bags with patient-grouped splits')
        if bundle.config.get('evaluation_unit', 'slide') != 'slide':
            raise ValueError('Original M4 selects slide-level mean endpoint AUROC; patient aggregation is a separate extension')
        self.bundle, self.profile, self.dimension = bundle, profile, bundle.dim
        self.endpoints, self.experts = list(bundle.classes), experts
        self._label_rows = self._labels()
        self._input_files = self._inputs()
        self.contract = digest_object({'source': COMMIT, 'profile': profile,
            'bundle': bundle.fingerprint, 'dimension': self.dimension, 'endpoints': self.endpoints,
            'experts': experts, 'labels': self._label_rows, 'inputs': self._input_files,
            'label_provenance': bundle.config['multilabel']})
        self.network = build_original(self.dimension, len(self.endpoints), experts)
        self.recipe = None
        self.completed_epochs, self.epoch_index, self.optimizer_updates = 0, -1, 0
        self.train_orders, self.val_orders = [], []
        self.last_source_metrics = None
        self.scheduler = self._optimizer = None
        self._scheduler_snapshot = None
        self._prediction_metrics = None
        self._metadata_digest = digest_object({'endpoints':bundle.classes,'multilabel':bundle.config['multilabel'],
            'training_unit':bundle.config.get('training_unit','slide'),'evaluation_unit':bundle.config.get('evaluation_unit','slide')})

    def _labels(self):
        return [{key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'targets', 'path']}
                for row in self.bundle.rows.to_dict('records')]

    def _inputs(self):
        # Retain full ordered feature-file identity, not only a directory name.
        from .data import resolve
        result = {}
        for row in self.bundle.rows.to_dict('records'):
            path = resolve(self.bundle.root, row['path'])
            result[str(path)] = {'sha256': digest_file(path), 'bytes': path.stat().st_size,
                                 'mtime_ns': path.stat().st_mtime_ns}
        return result

    def check_inputs(self):
        current = digest_object({'endpoints':self.bundle.classes,'multilabel':self.bundle.config['multilabel'],
            'training_unit':self.bundle.config.get('training_unit','slide'),'evaluation_unit':self.bundle.config.get('evaluation_unit','slide')})
        if current != self._metadata_digest or digest_object(self._labels()) != digest_object(self._label_rows):
            raise ValueError('M4 patient, endpoint or split binding changed')
        from pathlib import Path
        for filename, expected in self._input_files.items():
            path = Path(filename)
            if not path.is_file() or path.stat().st_size != expected['bytes'] or path.stat().st_mtime_ns != expected['mtime_ns']:
                raise ValueError('M4 feature bytes/order changed after immutable input binding')

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'm4_val_mean_auc'
                or cfg['lr'] != .0001 or cfg['weight_decay'] != .00004
                or type(cfg['epochs']) is not int or cfg['epochs'] < 1):
            raise ValueError('M4 requires full bags, unclipped summed source BCE, Adam1e-4/4e-5 and source VAL mean-AUROC selection')
        self.recipe = {key: cfg[key] for key in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        if self.recipe is None or lr != self.recipe['lr'] or weight_decay != self.recipe['weight_decay']:
            raise ValueError('M4 original recipe must be configured before its optimizer')
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = self.recipe['epochs']
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer,
            lr_lambda=lambda index: ((1+math.cos(index*math.pi/epochs))/2)*.9+.1)
        return self._optimizer

    def _loader(self, subset):
        dataset = Bags(self, subset)
        loader = DataLoader(dataset, batch_size=1, shuffle=subset != 'test',
            num_workers=0, pin_memory=True, drop_last=subset != 'test', collate_fn=author_collate())
        return dataset, loader

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('M4 epoch does not continue complete training history')
        self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        if bundle is not self.bundle or optimizer is not self._optimizer or self.epoch_index != self.completed_epochs:
            raise ValueError('M4 bound source epoch/optimizer mismatch')
        dataset, loader = self._loader('train')
        loss = author_training().train(SimpleNamespace(num_classes=len(self.endpoints)),
            self.network, optimizer, loader, device, self.epoch_index)
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite original M4 summed task loss')
        if sorted(dataset.order) != list(range(len(dataset))):
            raise ValueError('M4 original sampler did not visit every slide exactly once')
        self.train_orders.append([dataset.rows[i]['slide_id'] for i in dataset.order])
        self.optimizer_updates += len(dataset.order)
        self.completed_epochs += 1
        return [{'loss': float(loss)}]

    def finish_epoch(self, bundle, device, cfg):
        self.scheduler.step()  # Literal main.py order: complete TRAIN, cosine step, then shuffled VAL.

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        if bundle is not self.bundle or subset not in {'val', 'test'}:
            raise ValueError('M4 prediction requires the bound validation/test split')
        dataset, loader = self._loader(subset)
        probabilities = []
        def capture(_module, _inputs, outputs):
            probabilities.append(torch.stack([value.reshape(()) for value in outputs]).detach().cpu().numpy())
        hook = self.network.register_forward_hook(capture)
        try:
            values = author_training().evaluate(SimpleNamespace(num_classes=len(self.endpoints)),
                self.network, loader, device)
        finally:
            hook.remove()
        if sorted(dataset.order) != list(range(len(dataset))) or len(probabilities) != len(dataset):
            raise ValueError('M4 prediction order lost complete source bag identity')
        source = {name: [float(v) for v in scores] for name, scores in
                  zip(['auc', 'f1', 'accuracy', 'sensitivity', 'specificity'], values)}
        source['mean_auc'] = sum(source['auc'])/len(source['auc'])
        self._prediction_metrics = (subset, source)
        if subset == 'val':
            self.last_source_metrics = source
            # Independent evaluation of a completed checkpoint is not a training epoch.
            if len(self.val_orders) < self.completed_epochs:
                self.val_orders.append([dataset.rows[i]['slide_id'] for i in dataset.order])
        rows = []
        for index, scores in zip(dataset.order, probabilities):
            row = dataset.rows[index]
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'subset']}
            record.update({f'target_{i}': int(v) for i, v in enumerate(row['targets'])})
            record.update({f'p_{i}': float(v) for i, v in enumerate(scores)})
            rows.append(record)
        return pd.DataFrame(rows)

    def validation_metrics(self, frame):
        if self._prediction_metrics is None or set(frame.subset) != {self._prediction_metrics[0]}:
            raise ValueError('M4 source prediction metrics have not been computed for this split')
        subset, values = self._prediction_metrics
        return {'m4_'+subset+'_mean_auc': values['mean_auc']}

    def selection_improved(self, objective, best):
        # Author initial best_auc=0 and strict first-tie retention. If all epochs
        # stay at zero, source saves no selected model; do not invent one.
        return objective > max(0., best)

    def forward(self, features, coords=None, label=None):
        outputs = self.network(features)
        return {'probabilities': torch.stack([p.reshape(()) for p in outputs])[None, :]}

    def _active_parameter_count(self):
        return sum(1 for name, _ in self.network.named_parameters() if '.classifier.' not in name)

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if len(groups) != 1 or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}:
            raise ValueError('M4 optimizer omitted original source parameters')
        group = groups[0]
        expected = self.recipe['lr']*((1+math.cos(self.completed_epochs*math.pi/self.recipe['epochs']))/2*.9+.1)
        if (group['weight_decay'] != .00004 or group['betas'] != (.9, .999) or group['eps'] != 1e-8
                or group['lr'] != expected or self.scheduler.last_epoch != self.completed_epochs):
            raise ValueError('M4 Adam/cosine state changed')
        state = self._optimizer.state
        if self.optimizer_updates and (len(state) != self._active_parameter_count()
                or any(int(value['step']) != self.optimizer_updates for value in state.values())):
            raise ValueError('M4 complete active Adam moment/counter state is missing')

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe, 'completed_epochs': self.completed_epochs,
            'epoch_index': self.epoch_index, 'optimizer_updates': self.optimizer_updates,
            'train_orders': self.train_orders, 'val_orders': self.val_orders,
            'last_source_metrics': self.last_source_metrics,
            'scheduler': self.scheduler.state_dict() if self.scheduler else self._scheduler_snapshot})

    def set_extra_state(self, state):
        keys = {'contract', 'recipe', 'completed_epochs', 'epoch_index', 'optimizer_updates',
                'train_orders', 'val_orders', 'last_source_metrics', 'scheduler'}
        if (not isinstance(state, dict) or set(state) != keys or state.get('contract') != self.contract
                or (self.recipe is not None and state.get('recipe') != self.recipe)):
            raise ValueError('M4 complete source/label/FM/recipe checkpoint binding mismatch')
        recipe, count = state['recipe'], state['completed_epochs']
        if (not isinstance(recipe, dict) or set(recipe) != {'epochs','lr','weight_decay','threads','selection_metric'}
                or recipe['lr'] != .0001 or recipe['weight_decay'] != .00004
                or recipe['selection_metric'] != 'm4_val_mean_auc'
                or type(recipe['epochs']) is not int or recipe['epochs'] < 1):
            raise ValueError('M4 saved original training recipe is missing or changed')
        train_ids = [row['slide_id'] for row in self.bundle.records('train')]
        val_ids = [row['slide_id'] for row in self.bundle.records('val')]
        scheduler = state['scheduler']
        if (type(count) is not int or count < 1 or count > recipe['epochs']
                or state['optimizer_updates'] != count*len(train_ids)
                or len(state['train_orders']) != count or len(state['val_orders']) != count
                or any(sorted(order) != sorted(train_ids) for order in state['train_orders'])
                or any(sorted(order) != sorted(val_ids) for order in state['val_orders'])
                or state['epoch_index'] != count-1 or not isinstance(scheduler,dict)
                or scheduler.get('last_epoch') != count or scheduler.get('_step_count') != count+1
                or scheduler.get('base_lrs') != [.0001]
                or scheduler.get('_last_lr') != [.0001*((1+math.cos(count*math.pi/recipe['epochs']))/2*.9+.1)]):
            raise ValueError('M4 checkpoint is missing complete epoch/sampler/scheduler history')
        metrics = state['last_source_metrics']
        names = {'auc','f1','accuracy','sensitivity','specificity','mean_auc'}
        if (not isinstance(metrics,dict) or set(metrics)!=names
                or any(len(metrics[key])!=len(self.endpoints) or not np.isfinite(metrics[key]).all()
                    or any(not 0<=v<=1 for v in metrics[key]) for key in names-{'mean_auc'})
                or metrics['mean_auc']!=sum(metrics['auc'])/len(self.endpoints)):
            raise ValueError('M4 checkpoint lost the complete original per-endpoint VAL metrics')
        restored = copy.deepcopy(state)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(restored['scheduler'])
        self._scheduler_snapshot = restored['scheduler']
        self.recipe = restored['recipe']
        for key in ['completed_epochs', 'epoch_index', 'optimizer_updates', 'train_orders', 'val_orders', 'last_source_metrics']:
            setattr(self, key, restored[key])

    def protocol(self):
        return {'source': 'https://github.com/Bigyehahaha/M4', 'commit': COMMIT, 'contract': self.contract,
            'dimension': self.dimension, 'native_reference_dimension': 2048, 'experts': self.experts,
            'endpoints': self.endpoints, 'expert_width': 512, 'gate_slices': 4, 'tower_hidden': 32,
            'source_loss': 'Sum over independent source sigmoid-probability BCE tasks, not a class softmax or averaged task loss',
            'recipe': self.recipe, 'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'last_source_validation': self.last_source_metrics, 'train_orders': self.train_orders, 'val_orders': self.val_orders,
            'source_unused_expert_classifiers_preserved': True, 'all_complete_slide_bags_retained': True,
            'source_default': '32epochs/5experts/10endpoints/full2048RetCCL/Adam1e-4/4e-5/cosine0.1/strictVALmeanAUROC',
            'split_port': 'Immutable benchmark-provided patient-grouped splits; author outerTEST20%seed42 and five overlapping repeatedVAL splits are distinct',
            'synthetic': bool(self.bundle.config.get('synthetic')), 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

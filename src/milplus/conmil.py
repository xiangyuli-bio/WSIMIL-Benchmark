"""Full ConMIL semantic attention/KAN/Adam/cosine recipe and exact recovery."""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .conmil_source import COMMIT, PROFILES, build_model
from .io import digest_object


class Bags(Dataset):
    def __init__(self, owner, rows, training=False, trace=None):
        self.owner, self.rows, self.training, self.trace = owner, rows, training, trace

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        value = self.owner.assets.load(row)
        indices = torch.arange(len(value))
        if self.training and self.owner.max_patches is not None and len(value) > self.owner.max_patches:
            indices = torch.randperm(len(value))[:self.owner.max_patches]
            value = value[indices]
        if self.trace is not None:
            self.trace.append({'slide_id': row['slide_id'], 'indices': indices.tolist()})
        return value, torch.tensor(row['target'], dtype=torch.long), row


def collate_one(batch):
    # Original batch size1 and list-of-bags collate; retain row identity outside
    # the mathematical input without changing any sampling RNG calls.
    features, targets, rows = zip(*batch)
    return list(features), torch.stack(targets), list(rows)


class ConMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='conmil_posneg_source', kan_grid=5, max_patches=4096):
        super().__init__()
        if profile not in PROFILES or bundle.conmil is None:
            raise ValueError('ConMIL needs bound native CONCH visual/text features and actual prompt provenance')
        if max_patches is not None and (type(max_patches) is not int or max_patches < 1):
            raise ValueError('ConMIL original train cap must be a positive integer or explicit None')
        self.bundle, self.assets, self.profile = bundle, bundle.conmil, profile
        self.repaired = profile == 'conmil_posneg_multiclass_repaired'
        self.use_negative = profile != 'conmil_pos_source'
        if self.use_negative and not len(self.assets.text.negative):
            raise ValueError('Original ConMIL POS+NEG condition requires actual encoded negative queries')
        self.network = build_model(len(bundle.classes), profile, kan_grid)
        self.register_buffer('positive_queries', self.assets.text.positive.clone())
        self.register_buffer('negative_queries', self.assets.text.negative.clone())
        self.kan_grid, self.max_patches = kan_grid, max_patches
        self.contract = digest_object({'source': COMMIT, 'profile': profile,
            'input': self.assets.contract, 'bundle': bundle.fingerprint,
            'kan_grid': kan_grid, 'max_patches': max_patches})
        self.recipe = None
        self.completed_epochs, self.optimizer_updates, self.epoch_index = 0, 0, -1
        self.train_orders, self.samples_history = [], []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'conmil_val_accuracy':
            raise ValueError('ConMIL requires its source TRAIN-only cap, unclipped full CE and VAL accuracy selection')
        self.recipe = {key: cfg[key] for key in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self.recipe['epochs'])
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 1 or groups[0]['weight_decay'] != self.recipe['weight_decay']
                or groups[0]['initial_lr'] != self.recipe['lr']
                or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('ConMIL complete original-network Adam changed')
        schedule = self.scheduler.state_dict()
        if (schedule['T_max'] != self.recipe['epochs'] or schedule['eta_min'] != 0
                or schedule['last_epoch'] not in {self.completed_epochs, self.completed_epochs-1}
                or schedule['_step_count'] != schedule['last_epoch']+1
                or schedule['_last_lr'] != [groups[0]['lr']]):
            raise ValueError('ConMIL original cosine phase or actual optimizer learning rate changed')
        if self.optimizer_updates and (len(self._optimizer.state) != len(list(self.network.parameters()))
                or any(int(value['step']) != self.optimizer_updates for value in self._optimizer.state.values())):
            raise ValueError('ConMIL source Adam lost full actual parameter updates')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('ConMIL must resume at a full original epoch boundary')
        self.epoch_index = epoch

    def forward(self, features):
        if features.ndim != 2 or not len(features) or features.shape[1] != 512:
            raise ValueError('ConMIL requires complete native512D patch features')
        logits, raw, semantic = self.network(features, self.positive_queries,
            self.negative_queries if len(self.negative_queries) else None)
        return {'logits': logits, 'attention_raw': raw.reshape(-1),
                'semantic_similarity': semantic.reshape(-1), 'attention': raw.reshape(-1).softmax(0)}

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('ConMIL registered inputs or actual Adam identity changed')
        self.network.train(); samples, order, losses = [], [], []
        loader = DataLoader(Bags(self, bundle.records('train'), training=True, trace=samples),
            batch_size=1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_one)
        for features, targets, rows in loader:
            value, target = features[0].to(device), targets[0].to(device)
            optimizer.zero_grad()
            result = self(value)
            loss = nn.functional.cross_entropy(result['logits'], target.unsqueeze(0))
            if not torch.isfinite(loss):
                raise FloatingPointError('ConMIL original complete CE became nonfinite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError('ConMIL original complete attention/KAN gradient became nonfinite')
            optimizer.step(); self.optimizer_updates += 1
            order.append(rows[0]['slide_id'])
            losses.append({'loss': float(loss.detach()), 'sampled_patches': len(value),
                           'alpha': float(self.network.alpha.detach())})
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(order); self.samples_history.append(samples)
        self.validate_optimizer_state()
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('ConMIL prediction requires its registered VAL/TEST partition')
        self.network.eval(); records = []
        # Keep the source sequential DataLoader and its base-seed RNG draw;
        # VAL/TEST always use all original patches without TRAIN truncation.
        loader = DataLoader(Bags(self, bundle.records(subset)), batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=collate_one)
        for features, targets, rows in loader:
            result = self(features[0].to(device)); logits = result['logits']
            row = rows[0]
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'target']}
            record.update(subset=subset, source_slides=row['slide_id'], prediction=int(logits.argmax(1)),
                conmil_ce=float(nn.functional.cross_entropy(logits, targets.to(device))),
                conmil_patches=len(features[0]))
            record.update({f'p_{i}': float(value) for i, value in enumerate(logits.softmax(-1)[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        return {'conmil_val_accuracy': float(np.mean(frame.target.to_numpy() == frame.prediction.to_numpy()))}

    def after_validation(self, metrics):
        self.scheduler.step()
        self.validate_optimizer_state()

    def selection_improved(self, objective, best):
        # Public trainer initializes best_acc=0 and replaces only on strict
        # improvement. Its all-zero case produces no best file. Preserve that
        # behavior for source conditions; C-way repair explicitly accepts its
        # first finite checkpoint before applying the same strict tie rule.
        return objective > (best if self.repaired else max(0., best))

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'train_orders': self.train_orders, 'samples_history': self.samples_history,
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None})

    def set_extra_state(self, state):
        expected_keys = {'contract', 'recipe', 'completed_epochs', 'optimizer_updates',
                         'train_orders', 'samples_history', 'scheduler'}
        if not isinstance(state, dict) or set(state) != expected_keys or state.get('contract') != self.contract:
            raise ValueError('ConMIL actual prompt/feature/source/recipe identity changed')
        epochs, updates = state.get('completed_epochs'), state.get('optimizer_updates')
        train = {row['slide_id'] for row in self.bundle.records('train')}
        if (type(epochs) is not int or epochs < 0 or updates != epochs*len(train)
                or len(state.get('train_orders', [])) != epochs
                or len(state.get('samples_history', [])) != epochs
                or (self.recipe is not None and state.get('recipe') != self.recipe)
                or (epochs and (not isinstance(state.get('scheduler'), dict)
                    or state['scheduler']['last_epoch'] != epochs
                    or state['scheduler']['_step_count'] != epochs+1
                    or state['scheduler']['T_max'] != state['recipe']['epochs']))):
            raise ValueError('ConMIL complete epoch, Adam or cosine scheduler checkpoint is inconsistent')
        for order, samples in zip(state['train_orders'], state['samples_history']):
            if set(order) != train or len(order) != len(train) or [s['slide_id'] for s in samples] != order:
                raise ValueError('ConMIL actual TRAIN sampler order changed')
            for sample in samples:
                count = self.assets.meta['records'][sample['slide_id']]['count']
                expected = min(count, self.max_patches) if self.max_patches is not None else count
                if (len(sample['indices']) != expected or len(set(sample['indices'])) != expected
                        or any(type(i) is not int or not 0 <= i < count for i in sample['indices'])
                        or (count == expected and sample['indices'] != list(range(count)))):
                    raise ValueError('ConMIL checkpoint lost original TRAIN-only random patch selection')
        for key, value in state.items():
            if key not in {'contract', 'scheduler'}:
                setattr(self, key, copy.deepcopy(value))
        if hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(state['scheduler'])
        self.epoch_index = epochs-1

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for key, value in [('positive_queries', self.assets.text.positive), ('negative_queries', self.assets.text.negative)]:
            if key not in state_dict or not torch.equal(state_dict[key].cpu(), value):
                raise ValueError('ConMIL fixed original positive/negative text embeddings changed')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile,
            'network': 'Full512->256projection,128gatedattention,learned alpha*1x1 positive-minus-negative similarity bias,256->128->C cubic B-spline KAN',
            'native': {'hidden': 256, 'attention': 128, 'kan_grid': self.kan_grid, 'spline_order': 3, 'dropout': .25},
            'actual_text': self.assets.text.metadata, 'source_negative_weight': .5 if self.use_negative else None,
            'max_train_patches': self.max_patches, 'sampling': 'Source shuffled one-bag DataLoader; TRAIN4096randperm default, all VAL/TEST patches, workers0 explicit recovery port; source loader RNG retained',
            'native_recipe': 'Source default50epochs Adam2e-4/wd1e-5/cosine T_max=epochs; no clipping/early stop; README20epoch/grid6 example is explicit optional configuration',
            'recipe': self.recipe, 'selection': 'Source VAL slide accuracy strict improvement from best_acc0; no AUC/CE substitution. C-way target/evaluation repair explicitly named.',
            'multiclass_repair': self.repaired, 'completed_epochs': self.completed_epochs,
            'checkpoint_storage_port': 'Materialize source zero-stride expanded immutable KAN knot grid as an identical contiguous buffer; spline values, equations, RNG and parameters unchanged',
            'optimizer_updates': self.optimizer_updates, 'resume': 'Full network, fixed text queries, B-spline grids, Adam moments, cosine state, exact sampled indices/order and all global RNG',
            'synthetic': bool(self.bundle.config.get('synthetic', False)),
            'real_wsi_validation': 'not_run', 'paper_scores': 'not_reproduced'}

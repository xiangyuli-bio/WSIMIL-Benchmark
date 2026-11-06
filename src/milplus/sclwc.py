"""Complete declared SCL-WC paper reconstruction and literal author control.

Only the paper-reference route contains the complete CDA/PNM/WSCL method.
The source control runs the actual author epoch, preserving the absent PNM and
disabled contrastive path. No source-control result is called a full method.
"""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .io import digest_object
from .sclwc_network import COMMIT, SCLPaperReference
from .upstream import load_definitions, verified_source

PAPER_PROFILE = 'sclwc_paper_reference'
SOURCE_PROFILE = 'sclwc_literal_source_control'


def author_source(dimension, device):
    _, sha = verified_source('sclwc')
    if sha != COMMIT:
        raise ValueError('Wrong SCL-WC source')
    initialize = load_definitions('sclwc', 'utils/utils.py', ['initialize_weights'], {'nn': nn}).initialize_weights
    memory = load_definitions('sclwc', 'utils/memory.py', ['Memory'], {'torch': torch, 'nn': nn, 'random': random}).Memory
    contrast = load_definitions('sclwc', 'utils/loss.py', ['SupConLoss'], {'torch': torch, 'nn': nn, 'F': F}).SupConLoss
    model = load_definitions('sclwc', 'models/public_models/SCL.py',
        ['Att_Head', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'SCL'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'initialize_weights': initialize,
         'Memory': memory, 'SupConLoss': contrast, 'input_dimension': dimension},
        replacements=[('size = self.size_dict[size_arg]', 'size = [input_dimension, 512, 256]')])
    loop = load_definitions('sclwc', 'training_methods/SCL.py', ['train_loop'],
        {'torch': torch, 'tqdm': tqdm, 'explicit_device': device},
        replacements=[('torch.device("cuda" if torch.cuda.is_available() else "cpu")', 'explicit_device')])
    return model.SCL, loop.train_loop


class SlideDataset(Dataset):
    def __init__(self, bundle, rows):
        self.bundle, self.rows = bundle, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, _ = self.bundle.load(row)
        return {'x': x, 'y': torch.tensor([row['target']], dtype=torch.long), 'slide_id': row['slide_id']}


class SCLWC(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile=PAPER_PROFILE, k=None, temperature=.07,
                 warmup_epochs=5, positive_capacity=2048, negative_capacity=16384,
                 hard_negative_capacity=2048, cosine_t_max=50, cosine_eta_min=1e-6,
                 stop_epoch=30, dropout=True):
        super().__init__()
        if profile not in {PAPER_PROFILE, SOURCE_PROFILE} or bundle.task_type != 'classification':
            raise ValueError('SCL-WC requires its declared classification profile')
        if bundle.config.get('training_unit', 'slide') != 'slide' or bundle.config.get('evaluation_unit', 'slide') != 'slide':
            raise ValueError('The SCL-WC paper/source protocol trains and evaluates individual slide bags')
        roles = bundle.config.get('sclwc', {})
        if not isinstance(roles, dict):
            raise TypeError('task.sclwc must explicitly describe the normal-tissue class')
        normal = roles.get('negative_class')
        role = roles.get('negative_class_role')
        if normal not in bundle.classes or role not in {'normal_tissue', 'reference_endpoint_class'}:
            raise ValueError('SCL-WC PNM requires an explicitly named negative_class and its declared role in task.sclwc')
        provenance = roles.get('background_target_provenance')
        if role == 'reference_endpoint_class' and (not isinstance(provenance, str) or not provenance.strip()):
            raise ValueError('Endpoint-reference PNM requires explicit background_target_provenance; never relabel a tumour subtype as normal tissue')
        self.negative_class = bundle.classes.index(normal)
        self.literal = profile == SOURCE_PROFILE
        if self.literal and (len(bundle.classes) != 2 or self.negative_class != 0):
            raise ValueError('Literal SCL source control supports only binary labels with normal class at index0')
        if (type(cosine_t_max) is not int or cosine_t_max < 1 or type(stop_epoch) is not int or stop_epoch < 0 or
                isinstance(cosine_eta_min, bool) or not isinstance(cosine_eta_min, (int, float)) or
                not np.isfinite(cosine_eta_min) or cosine_eta_min < 0 or type(dropout) is not bool):
            raise ValueError('Invalid SCL-WC cosine, stopping or dropout settings')
        self.bundle, self.profile = bundle, profile
        self.rows = bundle.records('train')
        self.allowed_train = {row['slide_id']: row['target'] for row in self.rows}
        if not self.literal and any(sum(row['target'] == c for row in self.rows) < 2 for c in range(len(bundle.classes))):
            raise ValueError('Cross-slide SCL-WC requires at least two distinct TRAIN slides per class')
        self.k = (5 if self.literal else 8) if k is None else k
        if type(self.k) is not int or self.k < 1:
            raise ValueError('SCL-WC k must be positive')
        self.settings = {'k': self.k, 'temperature': temperature, 'warmup_epochs': warmup_epochs,
            'positive_capacity': positive_capacity, 'negative_capacity': negative_capacity,
            'hard_negative_capacity': hard_negative_capacity, 'cosine_t_max': cosine_t_max,
            'cosine_eta_min': cosine_eta_min, 'stop_epoch': stop_epoch, 'dropout': dropout,
            'negative_class': normal, 'negative_class_role': role, 'background_target_provenance': provenance}
        if self.literal:
            if (temperature != .07 or warmup_epochs != 5 or
                    (positive_capacity, negative_capacity, hard_negative_capacity) != (2048, 16384, 2048)):
                raise ValueError('Literal source-control banks and inactive WSCL settings must retain native defaults')
            cls, _ = author_source(bundle.dim, torch.device('cpu'))
            self.network = cls(k_sample=self.k, n_classes=2, dropout=dropout)
        else:
            self.network = SCLPaperReference(bundle.dim, len(bundle.classes), k=self.k,
                temperature=temperature, warmup_epochs=warmup_epochs, negative_class=self.negative_class,
                positive_capacity=positive_capacity, negative_capacity=negative_capacity,
                hard_negative_capacity=hard_negative_capacity, dropout=dropout)
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'bundle': bundle.fingerprint,
            'classes': bundle.classes, 'dimension': bundle.dim, 'settings': self.settings,
            'training_slide_labels': self.allowed_train})
        self.epoch = -1
        self.completed_epochs = self.optimizer_updates = self.bad = 0
        self.best = -float('inf')
        self.training_done = False
        self.optimizer = self.scheduler = self.loaded_scheduler = None
        self.epoch_trace = []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('SCL-WC native recipe uses complete bags and no gradient clipping')
        if cfg['selection_metric'] != 'macro_auc':
            raise ValueError('SCL-WC native validation selection is slide macro AUROC')
        if not self.literal and cfg['epochs'] <= self.settings['warmup_epochs']:
            raise ValueError('SCL-WC epochs must include active WSCL after the declared warmup')
        self.patience = cfg['patience']

    def prepare(self, bundle, seed, device):
        if self.contract != digest_object({'source': COMMIT, 'profile': self.profile, 'bundle': bundle.fingerprint,
                'classes': bundle.classes, 'dimension': bundle.dim, 'settings': self.settings,
                'training_slide_labels': self.allowed_train}):
            raise ValueError('SCL-WC input contract changed')
        for row in bundle.rows.to_dict('records'):
            x, _ = bundle.load(row)
            if len(x) < (3 if self.literal else 2*self.k):
                raise ValueError('SCL-WC needs enough real patches for its top/bottom representative bags')
            if self.literal and row['subset'] == 'train' and len(x) > 16384:
                raise ValueError('Literal source control cannot enqueue more patches than its original negative bank')

    def make_optimizer(self, lr, weight_decay):
        if self.settings['cosine_eta_min'] > lr:
            raise ValueError('SCL-WC cosine eta_min must not exceed initial lr')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
            T_max=self.settings['cosine_t_max'], eta_min=self.settings['cosine_eta_min'])
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch, self.epoch_trace = epoch, []

    def forward(self, x, coords=None, label=None, instance_eval=False):
        if self.literal:
            # Deliberately omit epoch, exactly as the author train_loop does.
            result = self.network(x, label=label, instance_eval=instance_eval)
            result['logits'] = result['bag_logits']
            if instance_eval:
                bag = F.cross_entropy(result['bag_logits'], label)
                contrast = torch.as_tensor(result['contra_loss'], device=x.device)
                total = .8*bag+.2*result['inst_loss']+.01*contrast
                self.epoch_trace.append({'loss': float(total.detach()), 'mil': float(bag.detach()),
                    'instance': float(result['inst_loss'].detach()), 'wscl': float(contrast.detach()),
                    'wscl_requires_grad': float(contrast.requires_grad), 'pnm_active': 0., 'forward_epoch': 0.})
            return result
        if label is not None or instance_eval:
            raise ValueError('Paper SCL training requires the indexed TRAIN identity in sclwc_epoch')
        return self.network(x)

    def after_validation(self, metrics):
        score = metrics['macro_auc']
        if score > self.best:
            self.best, self.bad = score, 0
        else:
            self.bad += 1
        self.training_done = self.bad >= self.patience and self.epoch > self.settings['stop_epoch']
        self.completed_epochs = self.epoch+1
        # Actual train.py validates, breaks on stop, then advances cosine.
        if not self.training_done:
            self.scheduler.step()

    def get_extra_state(self):
        return {'contract': self.contract, 'epoch': self.epoch, 'completed_epochs': self.completed_epochs,
            'optimizer_updates': self.optimizer_updates, 'bad': self.bad, 'best': self.best,
            'training_done': self.training_done,
            'scheduler': copy.deepcopy(self.scheduler.state_dict() if self.scheduler is not None else self.loaded_scheduler)}

    def set_extra_state(self, state):
        if state['contract'] != self.contract:
            raise ValueError('SCL-WC source/input/recipe contract differs')
        for key in ['epoch', 'completed_epochs', 'optimizer_updates', 'bad', 'best', 'training_done']:
            setattr(self, key, state[key])
        self.loaded_scheduler = copy.deepcopy(state['scheduler'])
        if self.scheduler is not None and self.loaded_scheduler is not None:
            self.scheduler.load_state_dict(self.loaded_scheduler)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not self.literal:
            bank = state_dict.get('network._extra_state', {}).get('banks', {})
            for entries in bank.values():
                for entry in entries:
                    if self.allowed_train.get(entry['slide_id']) != entry['label']:
                        raise ValueError('SCL-WC checkpoint bank contains a held-out or mislabelled slide')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'settings': self.settings,
            'feature_dimension': self.bundle.dim, 'hidden': 512, 'attention_dim': 256,
            'original_source_control_only': self.literal, 'complete_paper_objectives': not self.literal,
            'input_condition': 'frozen patch FM; all registered channels; SSL pretraining is a separate encoder stage',
            'paper_reconstruction': None if self.literal else {
                'pnm': 'shared C-way head, positive bag target and explicitly declared background-target class/role; sum of two CE as prose and author unused helpers',
                'wscl': 'paper Eq7: sum live current top-k anchors, mean same-class historical keys; L2 normalization and separate positive-category banks',
                'memory': 'TRAIN-only actual historical top-k positive/hard-negative and bottom-k easy-negative bags; exclude current slide; empty cold-start returns inactive zero; FIFO whole bags',
                'declared_choices': 'five epoch warmup, native source capacities, all valid bank keys, current live anchors; paper does not fully specify these implementation details',
                'pnm_scope': 'both terms evaluated on each slide; for a reference-class slide both targets are the declared reference class',
                'endpoint_condition': 'When the declared role is reference_endpoint_class, this is an explicit background-target interpretation for non-normal-label tasks. Supplement §8 evaluates MSI and NSCLC but does not disclose their PNM target mapping; do not claim the undisclosed original mapping is known.'},
            'source_control_deficits': 'actual source train_loop omits epoch; no active WSCL gradient or PNM; random queues retained' if self.literal else None,
            'sampler': 'source inverse-class-frequency WeightedRandomSampler, replacement, exactly N_train slides, DataLoader batch_size=None, num_workers=0',
            'optimizer': 'Adam; cosine T_max50 eta_min1e-6; validate before schedule; first strict best VAL slide macro AUROC; stop bad>=patience and epoch>30',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'training_done': self.training_done, 'contract': self.contract,
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}


def sclwc_epoch(model, bundle, groups, optimizer, device, cfg):
    rows = model.rows
    counts = {c: sum(row['target'] == c for row in rows) for c in range(len(bundle.classes))}
    weights = torch.tensor([len(rows)/counts[row['target']] for row in rows], dtype=torch.double)
    loader = DataLoader(SlideDataset(bundle, rows), batch_size=None,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True), num_workers=0)
    if model.literal:
        _, loop = author_source(bundle.dim, torch.device(device))
        loop(model.epoch, model, loader, optimizer, None, None)
        model.optimizer_updates += len(loader)
        return model.epoch_trace
    results = []
    for batch in loader:
        slide, target = batch['slide_id'], int(batch['y'].item())
        if model.allowed_train.get(slide) != target:
            raise ValueError('SCL-WC can only update memory with labelled TRAIN slides')
        x = batch['x'].to(device)
        result = model.network(x, label=target, slide_id=slide, epoch=model.epoch)
        optimizer.zero_grad()
        result['loss'].backward()
        optimizer.step()
        model.network.enqueue(result)
        model.optimizer_updates += 1
        results.append({'loss': float(result['loss'].detach()),
            **{name: float(value.detach()) for name, value in result['terms'].items()},
            'wscl_active': float(result['contrastive_info']['active']),
            'pnm_active': 1., 'representative_k': float(model.k)})
    return results

"""Loss-Attention: shared class head, full weighted loss and temporal recipe.

Classical feature MIL follows paper Eq.(6)/supplement A2/A4. The MNIST MIL
and image-temporal routes are explicitly named frozen-WSI-feature adaptations
of distinct author training scripts, not original WSI implementations.
"""
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from .data import load_array, resolve
from .io import digest_file, digest_object
from .smmile_data import positive_int
from .upstream import load_definitions

PROFILES = {'loss_attention_classical', 'loss_attention_mil_source', 'loss_attention_temporal'}


def author_kernels():
    attention = load_definitions('loss_attention', 'MIL/model.py', ['AttentionLayer'],
        {'torch': torch, 'nn': nn, 'F': F},
        [('torch.squeeze(alpha)', 'alpha.reshape(-1)')])
    weighted = load_definitions('loss_attention', 'MIL/weight_loss.py',
        ['log_sum_exp', 'class_select', 'cross_entropy_with_weights', 'CrossEntropyLoss'],
        {'torch': torch, 'nn': nn}, [('b= torch.unsqueeze(torch.squeeze(b),dim=1)', 'b = b.unsqueeze(1)')])
    return attention.AttentionLayer, weighted.CrossEntropyLoss


def author_temporal():
    schedules = load_definitions('loss_attention', 'Image classification/main.py',
        ['rampup', 'rampdown', 'step_rampup'], {'np': np}, [('np.float(', 'float(')])
    optimizer = load_definitions('loss_attention', 'Image classification/robust_adam.py', ['Adam'],
        {'torch': torch, 'np': np, 'math': math, 'Optimizer': torch.optim.Optimizer}, [
            ("grad.add(group['weight_decay'], p.data)", "grad.add(p.data, alpha=group['weight_decay'])"),
            ('exp_avg.mul_(beta1).add_(1 - beta1, grad)', 'exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)'),
            ('exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)',
             'exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)'),
            ('p.data.addcdiv_(-step_size, exp_avg, denom)', 'p.data.addcdiv_(exp_avg, denom, value=-step_size)')])
    return schedules, optimizer.Adam


def source_loader(groups):
    groups = sorted(groups, key=lambda rows: rows[0]['slide_id'])
    if any(len(rows) != 1 for rows in groups):
        raise ValueError('Loss-Attention uses complete individual slide bags')
    return DataLoader(groups, batch_size=None, sampler=RandomSampler(groups),
                      num_workers=0, collate_fn=lambda rows: rows)


def check_update_finite(model, optimizer):
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError('Loss-Attention has a nonfinite parameter')
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Loss-Attention has nonfinite optimizer state')


def finite(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name+' must be finite numeric')
    if minimum is not None and value < minimum or maximum is not None and value >= maximum:
        raise ValueError(name+' is outside its allowed range')
    return value


class LossAttention(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'unnormalized shared-classifier logits, per-instance class probabilities, and scalar energy attention normalized over the whole bag'

    def __init__(self, bundle, profile='loss_attention_classical', hidden=None, widths=None,
                 dropout=.5, instance_weight=None, ema=None, consistency_weight=None,
                 max_bank_bytes=512*1024**2):
        super().__init__()
        if profile not in PROFILES:
            raise ValueError('Unknown Loss-Attention route')
        if bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('Loss-Attention requires individual slide classification bags')
        self.profile, self.bundle = profile, bundle
        self.temporal = profile == 'loss_attention_temporal'
        self.classical = profile == 'loss_attention_classical'
        self.classes, self.dim = len(bundle.classes), bundle.dim
        self.dropout = finite(dropout, 'dropout', 0, 1)
        self.instance_weight = (.1 if self.classical else 1. if self.temporal else 2.) if instance_weight is None else instance_weight
        if finite(self.instance_weight, 'instance_weight') <= 0:
            raise ValueError('Loss-Attention requires active weighted-instance supervision')
        self.max_bank_bytes = positive_int(max_bank_bytes, 'max_bank_bytes')
        if self.classical:
            if hidden is not None:
                raise ValueError('Classical Loss-Attention uses three widths, not a single hidden override')
            widths = [256, 128, 64] if widths is None else widths
            if not isinstance(widths, list) or len(widths) != 3:
                raise ValueError('Classical Loss-Attention requires three MLP widths')
            for width in widths:
                positive_int(width, 'MLP width')
            layers, input_dim = [], bundle.dim
            for width in widths:
                layers.extend([nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)])
                input_dim = width
            self.widths = widths
            self.project = nn.Sequential(*layers)
            head_dim = widths[-1]
        else:
            if widths is not None:
                raise ValueError('Source projection uses hidden, not classical widths')
            hidden = (512 if self.temporal else 500) if hidden is None else hidden
            positive_int(hidden, 'hidden')
            self.widths = [hidden]
            # Native MNIST feature_extractor_part2 order, with FM replacing CNN output.
            self.project = nn.Sequential(nn.Linear(bundle.dim, hidden), nn.Dropout(dropout), nn.ReLU())
            head_dim = hidden
        attention_cls, loss_cls = author_kernels()
        self.attention = attention_cls(head_dim)
        self.classifier = nn.Linear(head_dim, self.classes)
        self.weighted_criterion = loss_cls(aggregate='sum' if self.classical else 'mean')
        if self.temporal:
            self.ema = .6 if ema is None else finite(ema, 'ema', 0, 1)
            self.consistency_weight = 400. if consistency_weight is None else finite(consistency_weight, 'consistency_weight')
            if self.consistency_weight <= 0:
                raise ValueError('Temporal Loss-Attention requires active consistency')
            self.schedules, self.optimizer_class = author_temporal()
        elif ema is not None or consistency_weight is not None:
            raise ValueError('Classical/MNIST source recipes have omega=0; select temporal explicitly')
        else:
            self.ema, self.consistency_weight = None, None
        self.epoch, self.optimizer_steps, self.optimizer_resets = -1, 0, 0
        self.optimizer, self.base_lr, self.weight_decay = None, None, None
        self.bank, self.pending_attention, self.bank_counts = {}, {}, {}
        self.best_selection_key, self.current_selection_key = None, None
        self.last_schedule = None

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('Loss-Attention preserves full bag lengths, aligned EMA rows and unclipped source updates')
        if cfg['selection_metric'] != 'loss_attention_error':
            raise ValueError('Loss-Attention selects exclusively on validation bag error; classical ties use validation bag CE')
        if self.temporal and cfg['epochs'] < 2:
            raise ValueError('Temporal recipe requires a nonzero-consistency epoch after epoch0')
        if self.temporal and cfg['weight_decay'] != 0:
            raise ValueError('Original image-temporal optimizer uses zero weight decay')
        self.stop_after_epoch = cfg['epochs']  # Complete the declared source budget; selection is separate.

    def prepare(self, bundle, seed, device):
        if not self.temporal:
            return
        counts = {r['slide_id']: len(bundle.load(r)[0]) for r in bundle.records('train')}
        # One persistent Z bank plus current-epoch predictions. No artificial common bag length.
        if 2*sum(counts.values())*4 > self.max_bank_bytes:
            raise ValueError('Loss-Attention temporal banks exceed max_bank_bytes before allocation')
        self.bank_counts = counts
        self.bank = {key: torch.zeros(count, dtype=torch.float32) for key, count in counts.items()}

    def make_optimizer(self, lr, weight_decay):
        self.base_lr, self.weight_decay = lr, weight_decay
        self.optimizer = (self.optimizer_class(self.parameters(), lr=lr, eps=1e-8, weight_decay=weight_decay)
                          if self.temporal else torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay))
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch = epoch
        self.pending_attention = {}
        if self.temporal:
            if set(self.bank) != {r['slide_id'] for r in self.bundle.records('train')}:
                raise ValueError('Temporal attention targets were not prepared for TRAIN slides')
            up, down = float(self.schedules.rampup(epoch)), float(self.schedules.rampdown(epoch))
            beta1 = down*.9+(1-down)*.5
            step = self.schedules.step_rampup(epoch)
            beta2 = step*.99+(1-step)*.999
            self.optimizer.state.clear()  # The published image loop recreates Adam EVERY epoch.
            for group in self.optimizer.param_groups:
                group.update(lr=self.base_lr*up*down, betas=(beta1, beta2))
            self.optimizer_resets += 1
            self.last_schedule = {'epoch': epoch, 'rampup': up, 'rampdown': down,
                'lr': self.base_lr*up*down, 'betas': [beta1, beta2],
                'consistency_weight': 0. if epoch == 0 else self.consistency_weight*up}

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('Loss-Attention requires finite nonempty [N,D] features')
        h = self.project(x)
        context, instance_logits, attention = self.attention(h, self.classifier.weight, self.classifier.bias, 1)
        output = {'logits': self.classifier(context.mean(0, keepdim=True)),
                  'instance_logits': instance_logits, 'attention': attention}
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise ValueError('Loss-Attention produced nonfinite outputs')
        return output

    def objective(self, output, target, row=None, update_bank=False):
        bag = F.cross_entropy(output['logits'], target)
        # These weak instance targets come ONLY from the slide label, not an annotation file.
        instance = self.weighted_criterion(output['instance_logits'], target.expand(len(output['attention'])),
                                           weights=output['attention'])
        total = bag+self.instance_weight*instance
        terms = {'bag_loss': bag, 'weighted_instance_loss': instance}
        if self.temporal and update_bank:
            if row is None or row.get('subset') != 'train' or row['slide_id'] not in self.bank:
                raise ValueError('Temporal attention updates are restricted to registered TRAIN bags')
            key = row['slide_id']
            if key in self.pending_attention or self.bank_counts[key] != len(output['attention']):
                raise ValueError('Temporal rows must be complete, aligned and visited exactly once per epoch')
            previous = self.bank[key].to(output['attention'].device)
            target_attention = previous if self.epoch == 0 else previous/(1-self.ema**self.epoch)
            consistency = F.mse_loss(output['attention'], target_attention)
            total = total+self.last_schedule['consistency_weight']*consistency
            self.pending_attention[key] = output['attention'].detach().cpu().clone()
            terms['consistency_loss'] = consistency
        return total, terms

    def finish_epoch(self, bundle, device, cfg):
        if self.temporal:
            if set(self.pending_attention) != set(self.bank):
                raise ValueError('Temporal EMA cannot commit an incomplete training epoch')
            for key in self.bank:
                # Match the source multiplication/addition order and float32 EMA arithmetic.
                self.bank[key] = self.ema*self.bank[key]+(1-self.ema)*self.pending_attention[key]
            self.pending_attention = {}

    def start_prediction(self, subset):
        torch.empty((), dtype=torch.int64).random_()  # Sequential source DataLoader iterator seed.

    def prediction_fields(self, result, row):
        target = torch.tensor([row['target']], device=result['logits'].device)
        _, terms = self.objective(result, target)
        return {'loss_attention_bag_loss': float(terms['bag_loss']),
                'loss_attention_instance_loss': float(terms['weighted_instance_loss']),
                'loss_attention_error': float(int(result['logits'].argmax(1)) != row['target'])}

    def validation_metrics(self, frame):
        return {key: float(frame[key].mean()) for key in
                ['loss_attention_error', 'loss_attention_bag_loss', 'loss_attention_instance_loss']}

    def after_validation(self, metrics):
        self.current_selection_key = (-metrics['loss_attention_error'],)
        if self.classical:
            self.current_selection_key += (-metrics['loss_attention_bag_loss'],)

    def selection_improved(self, objective, best):
        improved = self.best_selection_key is None or self.current_selection_key > self.best_selection_key
        if improved:
            self.best_selection_key = self.current_selection_key
        return improved

    def get_extra_state(self):
        return {name: getattr(self, name) for name in ['epoch', 'optimizer_steps', 'optimizer_resets',
            'bank', 'bank_counts', 'best_selection_key', 'current_selection_key', 'last_schedule']}

    def set_extra_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self.bank = {key: value.cpu() for key, value in self.bank.items()}
        self.pending_attention = {}
        if self.temporal and (set(self.bank) != set(self.bank_counts) or
                any(v.shape != (self.bank_counts[k],) or not torch.isfinite(v).all() for k, v in self.bank.items())):
            raise ValueError('Invalid temporal attention checkpoint')

    def protocol(self):
        return {'method': self.profile, 'family': 'loss_attention', 'input': 'frozen FM features; explicit WSI adaptation',
            'projection_widths': self.widths, 'dropout': self.dropout,
            'attention': 'author shared bag/instance W,b; energy normalized over instances; unused legacy attention.linear retained',
            'instance_weight': self.instance_weight, 'instance_reduction': 'sum' if self.classical else 'mean (1/N)',
            'temporal': self.temporal, 'ema': self.ema, 'consistency_weight': self.consistency_weight,
            'temporal_target': 'TRAIN-only per-original-patch EMA, bias corrected exactly as image source' if self.temporal else None,
            'optimizer': 'author legacy Adam epsilon placement, state reset each epoch; 80/150/200 schedule' if self.temporal else 'Adam continuous state',
            'optimizer_steps': self.optimizer_steps, 'optimizer_resets': self.optimizer_resets,
            'epoch': self.epoch, 'last_schedule': self.last_schedule,
            'selection': 'validation slide error, then bag CE' if self.classical else 'validation slide error, first best; test remains locked',
            'best_selection_key': self.best_selection_key,
            'repairs': ['singleton alpha/logsumexp shapes', 'modern device/NumPy/Adam call syntax',
                        'reject nonfinite values instead of original optimizer zeroing NaNs', 'replace source test-set best selection with validation-only selection'],
            'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'gpu_validation': 'not_run'}

    def export_instances(self, result, row, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object({'slide_id': row['slide_id']})[:24]+'.h5')
        coordinate_path = resolve(self.bundle.root, row.get('coords_path') or row['path'])
        coords = None
        if row.get('coords_path'):
            coords = load_array(coordinate_path, 'coords')
        elif coordinate_path.suffix in {'.h5', '.hdf5'}:
            with h5py.File(coordinate_path) as h:
                if 'coords' in h:
                    coords = h['coords'][:]
        with h5py.File(path, 'w') as h:
            for key, value in {'attention': result['attention'], 'instance_logits': result['instance_logits'],
                               'instance_probabilities': result['instance_logits'].softmax(-1)}.items():
                h[key] = value.detach().cpu().numpy()
            if coords is not None:
                h['coords'] = coords
            h.attrs['method'] = self.profile
            h.attrs['score_semantics'] = self.instance_score_semantics
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path),
                'instances': len(result['attention']), 'coordinates': coords is not None}


def loss_attention_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    for rows in source_loader(groups):
        row = rows[0]
        x, _ = bundle.load(row)
        target = torch.tensor([row['target']], device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x.to(device))
        total, terms = model.objective(output, target, row=row, update_bank=model.temporal)
        if not torch.isfinite(total):
            raise ValueError('Loss-Attention objective is nonfinite')
        total.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('Loss-Attention gradient is nonfinite')
        check_update_finite(model, optimizer)
        optimizer.step()
        check_update_finite(model, optimizer)
        model.optimizer_steps += 1
        losses.append({'loss': float(total.detach()), **{k: float(v.detach()) for k, v in terms.items()}})
    return losses

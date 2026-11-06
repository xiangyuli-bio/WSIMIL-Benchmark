"""Complete C2Aug paper-reference recipe, with explicitly named source repairs.

The released Lightning route omits its bag loss and optimizer ownership of
prototypes. This profile implements both published objectives; it does not
claim to reproduce that incomplete route or the paper's numerical results.
"""
import copy
import math
import random

import numpy as np
import torch
from nystrom_attention import NystromAttention
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .c2aug_data import TrainClassBank
from .io import digest_object
from .upstream import load_definitions, verified_source

COMMIT = '14cb53bc71f5161105b0dbef45351243d7330561'
PROFILE = 'c2aug_transmil_paper_reference'
PAPER = 'https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Contrastive_Cross-Bag_Augmentation_for_Multiple_Instance_Learning-based_Whole_Slide_Image_CVPR_2026_paper.pdf'
SUPPLEMENT = 'https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Contrastive_Cross-Bag_Augmentation_CVPR_2026_supplemental.pdf'


def author_transmil(dimension, classes):
    _, sha = verified_source('c2aug')
    if sha != COMMIT:
        raise ValueError('C2Aug source differs from the reviewed commit')
    original = load_definitions('c2aug', 'models/transmil.py', ['TransLayer', 'PPEG', 'TransMIL'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'NystromAttention': NystromAttention},
        replacements=[('self.cls_token.expand(B, -1, -1).cuda()',
                       'self.cls_token.expand(B, -1, -1).to(h.device)')])
    return original.TransMIL(dimension, classes)


class PaperAugmentation(nn.Module):
    """Equations 2-4 plus expansion/compression; original-instance query."""
    def __init__(self, dimension, reduced=256, compression_max=16):
        super().__init__()
        self.reduce = nn.Linear(dimension, reduced)
        self.recover = nn.Linear(reduced, dimension)
        self.view_q = nn.Linear(reduced, reduced, bias=False)
        self.view_k = nn.Linear(reduced, reduced, bias=False)
        self.view_v = nn.Linear(reduced, reduced, bias=False)
        self.compression_q = nn.Parameter(torch.randn(1, reduced))
        self.compression_k = nn.Linear(reduced, reduced, bias=False)
        self.compression_v = nn.Linear(reduced, reduced, bias=False)
        self.compression_max = compression_max

    def fuse(self, original, views, mask=None):
        # Independent per-instance attention across disjoint same-class views.
        q = self.view_q(self.reduce(original)).unsqueeze(1)
        v = self.reduce(views)
        dots = (q*self.view_k(v)).sum(-1)/math.sqrt(q.shape[-1])
        if mask is None:
            ranks = torch.rand_like(dots).argsort(-1).argsort(-1)
            removed = torch.randint(dots.shape[1], (len(dots), 1), device=dots.device)
            mask = ranks < removed
        if mask.shape != dots.shape or mask.dtype != torch.bool or mask.all(1).any():
            raise ValueError('C2Aug row mask must retain at least one view per instance')
        weights = dots.masked_fill(mask, -9999.).softmax(-1)
        fused = (weights.unsqueeze(-1)*self.view_v(v)).sum(1)
        return self.recover(fused)

    def compress(self, original, ratio=None, padding=None):
        x = self.reduce(original)
        ratio = random.randint(2, self.compression_max) if ratio is None else ratio
        if type(ratio) is not int or not 2 <= ratio <= self.compression_max:
            raise ValueError('C2Aug compression ratio is outside the declared uniform range')
        missing = (-len(x)) % ratio
        if padding is None:
            padding = torch.randint(len(x), (missing,), device=x.device)
        if padding.shape != (missing,) or padding.dtype != torch.long or (padding < 0).any() or (padding >= len(x)).any():
            raise ValueError('C2Aug compression padding must sample real rows of the current bag')
        x = torch.cat([x, x[padding]]).reshape(-1, ratio, x.shape[-1])
        q = self.compression_q.expand(len(x), -1).unsqueeze(1)
        weights = (q@self.compression_k(x).transpose(-1, -2)/math.sqrt(x.shape[-1])).softmax(-1)
        return self.recover((weights@self.compression_v(x)).squeeze(1))

    def forward(self, x, row, bank, max_views):
        trace = {'multiview': 0, 'expansion': 0, 'compression': 0}
        if random.random() > .5:
            if random.random() > .5:
                views = random.randint(1, max_views)
                sampled = bank.sample(row['target'], views*len(x), row['slide_id'], x.device)
                x = self.fuse(x, sampled.reshape(views, len(x), -1).transpose(0, 1))
                trace['multiview'] = 1
            elif len(x) < bank.max_count:
                extra = random.randint(1, bank.max_count-len(x))
                x = torch.cat([x, bank.sample(row['target'], extra, row['slide_id'], x.device)])
                trace['expansion'] = 1
        if random.random() > .5:
            x = self.compress(x)
            trace['compression'] = 1
        return x, trace


class PaperBranch(nn.Module):
    def __init__(self, dimension, classes, reduced, compression_max, prototypes, projection):
        super().__init__()
        self.augment = PaperAugmentation(dimension, reduced, compression_max)
        self.mil = author_transmil(dimension, classes)
        self.prototypes = nn.Parameter(torch.randn(prototypes, 512))
        self.group_q = nn.Linear(512, 512, bias=False)
        self.group_k = nn.Linear(512, 512, bias=False)
        self.group_v = nn.Linear(512, 512, bias=False)
        self.group_projection = nn.Linear(512, projection)
        self.bag_projection = nn.Linear(512, projection)

    def align(self, instances):
        assignments = (F.normalize(instances, dim=-1)@F.normalize(self.prototypes, dim=-1).T).argmax(1)
        groups = []
        for i, prototype in enumerate(self.prototypes):
            selected = instances[assignments == i]
            if not len(selected):
                selected = prototype.unsqueeze(0)
            scores = self.group_q(prototype).unsqueeze(0)@self.group_k(selected).T/math.sqrt(512)
            groups.append(scores.softmax(-1)@self.group_v(selected))
        return F.normalize(self.group_projection(torch.cat(groups)), dim=-1)

    def encode(self, x):
        value = self.mil(data=x.unsqueeze(0))
        return value, F.normalize(self.bag_projection(value['feature']), dim=-1), self.align(value['contra_feature'])


def group_kl(student, teacher, center, student_temperature, teacher_temperature):
    """Forward KL(student||teacher), summed over every corresponding group."""
    log_s = (student/student_temperature).log_softmax(-1)
    log_t = ((teacher.detach()-center)/teacher_temperature).log_softmax(-1)
    return (log_s.exp()*(log_s-log_t)).sum()


def bag_infonce(student, positive, negatives, temperature):
    logits = student@torch.cat([positive.detach(), negatives.detach()]).T/temperature
    return F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=student.device))


class C2AugMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile=PROFILE, max_views=8, reduced=256,
                 compression_max=16, prototypes=8, projection=128, queue_size=256,
                 alpha=1., beta=1., temperature=.07, student_temperature=.1,
                 teacher_temperature=.04, momentum=.9999, center_momentum=.9):
        super().__init__()
        if profile != PROFILE or bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('C2Aug paper-reference profile requires individual classification slide bags')
        integers = [max_views, reduced, compression_max, prototypes, projection, queue_size]
        if any(type(v) is not int or v < 1 for v in integers) or compression_max < 2:
            raise ValueError('C2Aug view, dimension, prototype and memory budgets must be positive integers')
        positive = [alpha, beta, temperature, student_temperature, teacher_temperature]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError('C2Aug requires finite positive temperatures and both active contrastive objectives')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 < v < 1 for v in [momentum, center_momentum]):
            raise ValueError('C2Aug EMA coefficients must be finite and strictly between zero and one')
        self.bundle, self.profile, self.max_views = bundle, profile, max_views
        self.alpha, self.beta, self.temperature = alpha, beta, temperature
        self.student_temperature, self.teacher_temperature = student_temperature, teacher_temperature
        self.momentum, self.center_momentum = momentum, center_momentum
        self.bank = TrainClassBank(bundle)
        self.bank.validate_views(max_views)
        self.student = PaperBranch(bundle.dim, len(bundle.classes), reduced, compression_max, prototypes, projection)
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.register_buffer('queue', torch.zeros(queue_size, projection))
        self.register_buffer('center', torch.zeros(1, projection))
        self.queue_ids = [None]*queue_size
        self.queue_pointer, self.queue_count = 0, 0
        self.optimizer_steps, self.completed_epochs = 0, 0
        self.augmentation_counts = dict.fromkeys(['multiview', 'expansion', 'compression'], 0)
        self.epochs = 200
        self.settings = dict(zip(['max_views', 'reduced', 'compression_max', 'prototypes', 'projection', 'queue_size'], integers))
        self.settings.update(alpha=alpha, beta=beta, temperature=temperature,
            student_temperature=student_temperature, teacher_temperature=teacher_temperature,
            momentum=momentum, center_momentum=center_momentum)
        self.generator = torch.Generator().manual_seed(2023)
        self.contract = digest_object({'source': COMMIT, 'bundle': bundle.fingerprint,
            'reservoir': self.bank.fingerprint, 'profile': profile,
            'integer_options': integers, 'loss_options': positive, 'ema': [momentum, center_momentum]})

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'c2aug_val_macro_accuracy':
            raise ValueError('C2Aug requires complete bags, unclipped gradients and distinct VAL macro accuracy')
        self.epochs = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.student.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, coords=None):
        if x.ndim != 2 or len(x) < 1 or x.shape[1] != self.bundle.dim or not torch.isfinite(x).all():
            raise ValueError('C2Aug requires a finite nonempty original bag in its registered FM space')
        # Original features directly enter the MIL at inference. No augmentation,
        # bank access, prototype grouping, queue mutation or label dependency.
        result = self.student.mil(data=x.unsqueeze(0))
        return {'logits': result['logits'], 'probabilities': result['Y_prob'], 'embedding': result['feature']}

    def negatives(self, slide):
        positions = [i for i, value in enumerate(self.queue_ids) if value is not None and value != slide]
        return self.queue[positions].detach().clone()

    def training_objective(self, x, row):
        if not self.training:
            raise ValueError('C2Aug training objective requires training mode')
        sx, trace_s = self.student.augment(x, row, self.bank, self.max_views)
        output, z_s, g_s = self.student.encode(sx)
        with torch.no_grad():
            tx, trace_t = self.teacher.augment(x, row, self.bank, self.max_views)
            _, z_t, g_t = self.teacher.encode(tx)
        target = torch.tensor([row['target']], device=x.device)
        classification = F.cross_entropy(output['logits'], target)
        negatives = self.negatives(row['slide_id'])
        bag = bag_infonce(z_s, z_t, negatives, self.temperature)
        group = group_kl(g_s, g_t, self.center, self.student_temperature, self.teacher_temperature)
        return {'loss': classification+self.alpha*bag+self.beta*group,
                'classification': classification, 'bag': bag, 'group': group,
                'teacher_bag': z_t, 'teacher_groups': g_t,
                'negatives': len(negatives), 'trace_s': trace_s, 'trace_t': trace_t}

    @torch.no_grad()
    def commit_update(self, result, slide):
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.mul_(self.momentum).add_(student, alpha=1-self.momentum)
        self.center.mul_(self.center_momentum).add_(result['teacher_groups'].mean(0, keepdim=True), alpha=1-self.center_momentum)
        self.queue[self.queue_pointer].copy_(result['teacher_bag'][0])
        self.queue_ids[self.queue_pointer] = slide
        self.queue_pointer = (self.queue_pointer+1) % len(self.queue)
        self.queue_count = min(self.queue_count+1, len(self.queue))
        for name in self.augmentation_counts:
            self.augmentation_counts[name] += result['trace_s'][name]+result['trace_t'][name]
        self.optimizer_steps += 1

    def epoch(self, bundle, optimizer, device='cpu'):
        self.train()
        rows = bundle.records('train')
        loader = DataLoader(rows, batch_size=1, shuffle=True, generator=self.generator, collate_fn=lambda batch: batch[0])
        losses = []
        for row in loader:
            optimizer.zero_grad(set_to_none=True)
            x, _ = bundle.load(row)
            result = self.training_objective(x.to(device), row)
            result['loss'].backward()
            if not torch.isfinite(result['loss']) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.student.parameters()):
                raise ValueError('C2Aug nonfinite combined objective/gradient; optimizer and memory not committed')
            optimizer.step()
            self.commit_update(result, row['slide_id'])
            losses.append({k: float(result[k].detach()) for k in ['loss', 'classification', 'bag', 'group']})
        return losses

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1

    def validation_metrics(self, frame):
        prediction = frame[[f'p_{i}' for i in range(len(self.bundle.classes))]].to_numpy().argmax(1)
        accuracy = np.mean([np.mean(prediction[frame.target.to_numpy() == i] == i) for i in range(len(self.bundle.classes))])
        return {'c2aug_val_macro_accuracy': float(accuracy)}

    def get_extra_state(self):
        return {'contract': self.contract, 'queue_ids': self.queue_ids.copy(),
            'queue_pointer': self.queue_pointer, 'queue_count': self.queue_count,
            'optimizer_steps': self.optimizer_steps, 'completed_epochs': self.completed_epochs,
            'augmentation_counts': self.augmentation_counts.copy(), 'generator': self.generator.get_state()}

    def set_extra_state(self, state):
        expected = {'contract', 'queue_ids', 'queue_pointer', 'queue_count', 'optimizer_steps', 'completed_epochs', 'augmentation_counts', 'generator'}
        if set(state) != expected or state['contract'] != self.contract:
            raise ValueError('C2Aug source/FM/TRAIN bank/recipe checkpoint contract differs')
        if any(type(state[k]) is not int or state[k] < 0 for k in ['queue_pointer', 'queue_count', 'optimizer_steps', 'completed_epochs']):
            raise ValueError('C2Aug checkpoint counters invalid')
        ids = state['queue_ids']
        if not isinstance(ids, list) or len(ids) != len(self.queue) or any(x is not None and x not in self.bank.by_slide for x in ids):
            raise ValueError('C2Aug FIFO contains an unknown/non-TRAIN slide')
        pointer, count = state['queue_pointer'], state['queue_count']
        if pointer >= len(ids) or count > len(ids) or sum(x is not None for x in ids) != count:
            raise ValueError('C2Aug checkpoint FIFO occupancy invalid')
        if count != min(state['optimizer_steps'], len(ids)) or pointer != state['optimizer_steps'] % len(ids):
            raise ValueError('C2Aug FIFO cursor and committed update count differ')
        if not torch.isfinite(self.queue).all() or not torch.isfinite(self.center).all():
            raise ValueError('C2Aug checkpoint FIFO/center contains nonfinite values')
        counts = state['augmentation_counts']
        if set(counts) != set(self.augmentation_counts) or any(type(v) is not int or not 0 <= v <= 2*state['optimizer_steps'] for v in counts.values()):
            raise ValueError('C2Aug checkpoint augmentation counters invalid')
        generator = torch.Generator()
        try:
            generator.set_state(state['generator'].cpu())
        except (TypeError, AttributeError, RuntimeError) as error:
            raise ValueError('C2Aug checkpoint sampler state invalid') from error
        self.queue_ids, self.queue_pointer, self.queue_count = ids.copy(), pointer, count
        self.optimizer_steps, self.completed_epochs = state['optimizer_steps'], state['completed_epochs']
        self.augmentation_counts, self.generator = counts.copy(), generator

    def protocol(self):
        return {'profile': self.profile, 'source_commit': COMMIT, 'paper': PAPER, 'supplement': SUPPLEMENT,
            'actual_settings': self.settings.copy(), 'configured_epochs': self.epochs,
            'optimizer_steps': self.optimizer_steps, 'completed_epochs': self.completed_epochs,
            'augmentation_counts': self.augmentation_counts.copy(), 'queue_occupancy': self.queue_count,
            'recipe': 'CE + alpha*bag InfoNCE with actual other-bag FIFO negatives + beta*sum corresponding-group forward KL(student||teacher); full trainable student, EMA teacher, prototype attention, projected L2 features and teacher centering',
            'supplement_defaults': 'linear256 augmentation space,8 random views,256-entry FIFO,8 prototypes,Adam1e-4/batch1/200epochs',
            'reference_choices': 'one-head scaled cross-attention; projection128; alpha=beta=1; bag temperature.07,student.1,teacher.04,centerEMA.9,teacherEMA.9999; source-like .25 MV/.25 expansion/.5 independent compression; ratio2..16, random in-bag padding; teacher train dropout, EMA after Adam, first strict-best distinct VAL slide macroaccuracy; no early stopping by default',
            'data_boundary': 'Only TRAIN reservoir, excludes originating slide, disjoint views without replacement; indexed sampled-row reads without feature-corpus duplication. Insufficient other-bag reservoir fails explicitly.',
            'source_differences': 'Restores inactive bag objective, all-group forward KL and learnable prototype ownership; original-instance MV query instead of source learned query; no unpublished FFT transform; no train+val reservoir or TEST selection; arbitrary declared FM width and class count, CPU CLS bridge; whole-budget constant temperatures',
            'warmup': 'FIFO initially empty; first positive-only InfoNCE is exactly0, no fabricated random negative. Queue identity filter excludes repeated original bag keys.',
            'inference': 'student native MIL on original bag only; no cross-bag lookup, augmentation, grouping, labels or queue changes',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

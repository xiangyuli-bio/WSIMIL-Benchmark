"""Pinned MAMMOTH experts, author MIL heads and an explicit paper-based trainer."""
import contextlib
import io
import math
import sys
import types
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, package, pinned_text, verified_source

HEADS = {'mammoth_'+name: name for name in ['mean', 'max', 'abmil', 'clam', 'dsmil', 'transmil', 'transformer']}
HEADS['mammoth_ilra_repaired'] = 'ilra'


def author_core():
    """Execute unmodified, tracked author files in an isolated relative package."""
    root, sha = verified_source('mammoth')
    name = '_milplus_mammoth_'+sha
    if name+'.mammoth' not in sys.modules:
        package(name, root/'src/mammoth')
        for relative in ['components', 'mammoth']:
            module = types.ModuleType(name+'.'+relative)
            module.__file__, module.__package__ = str(root/f'src/mammoth/{relative}.py'), name
            sys.modules[module.__name__] = module
            exec(compile(pinned_text(root, sha, f'src/mammoth/{relative}.py'), module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified, unmodified author Git blobs
    return sys.modules[name+'.mammoth']


def author_head(kind):
    """Select actual model definitions, excluding HF auto-registration and imports."""
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'Optional': Optional,
             'Union': Union, 'Tuple': tuple, 'ABC': ABC, 'abstractmethod': abstractmethod,
             'Mammoth': author_core().Mammoth}
    if kind == 'transmil':
        from nystrom_attention import NystromAttention
        scope['NystromAttention'] = NystromAttention
    base = load_definitions('mammoth', 'millab/src/models/mil_template.py', ['MIL'], scope)
    scope['MIL'] = base.MIL
    layer_names = ['create_mlp', 'GlobalAttention', 'GlobalGatedAttention']
    layers = load_definitions('mammoth', 'millab/src/models/layers.py', layer_names, scope)
    scope.update({name: getattr(layers, name) for name in layer_names})
    # Transformer imports the MLP with a different final-dropout convention.
    if kind == 'transformer':
        scope['create_mlp'] = load_definitions('mammoth', 'src/mammoth/components.py', ['create_mlp'], scope).create_mlp
    definitions = {'abmil': (['ABMIL'], 'ABMIL'), 'clam': (['CLAMSB'], 'CLAMSB'),
        'dsmil': (['IClassifier', 'BClassifier', 'DSMIL'], 'DSMIL'),
        'transmil': (['TransLayer', 'PPEG', 'TransMIL'], 'TransMIL'),
        'transformer': (['TransLayer', 'Transformer'], 'Transformer'),
        'ilra': (['MultiHeadAttention', 'GAB', 'NLP', 'ILRA'], 'ILRA')}
    names, selected = definitions[kind]
    return getattr(load_definitions('mammoth', f'millab/src/models/{kind}.py', names, scope), selected)


class BinarySmoothTop1(nn.Module):
    """Device-neutral binary SmoothTop1SVM, including the author's hard branch."""
    def forward(self, logits, target):
        if logits.ndim != 2 or logits.shape[1] != 2 or target.shape != (len(logits),):
            raise ValueError('CLAM instance SmoothTop1 requires binary [instances,2] scores and targets')
        margin = logits + (torch.arange(2, device=logits.device)[None] != target[:, None]).to(logits.dtype)
        margin = margin-logits.gather(1, target[:, None])
        hard = (logits.max(1).values-logits.min(1).values >= math.log(1000)).detach()
        return torch.where(hard, margin.max(1).values, torch.logsumexp(margin, 1)).mean()


class PoolHead(nn.Module):
    """The paper's feature-mean and single-winning-instance max definitions."""
    def __init__(self, dim, classes, moe, kind):
        super().__init__()
        self.patch_embed = author_core().Mammoth(**moe)
        self.classifier = nn.Linear(dim, classes)
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)
        self.kind = kind

    def forward(self, h, label=None, loss_fn=None):
        z = self.patch_embed(h)
        if self.kind == 'mean':
            logits = self.classifier(z.mean(1))
        else:
            instance_logits = self.classifier(z)
            winner = instance_logits.flatten(1).argmax(1)//instance_logits.shape[-1]
            logits = instance_logits[torch.arange(len(h), device=h.device), winner]
        loss = None if label is None else loss_fn(logits, label)
        return {'logits': logits, 'loss': loss}, {}


class MammothMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile='mammoth_abmil', hidden=None, attention_dim=384,
                 experts=30, slots=10, heads=16, slot_dim=None, rank=16, auto_rank=True,
                 keep_slots=True, share_weights=True, use_layernorm=True, expert_dropout=.1,
                 dropout=.25, feature_dropout=.1, precision='fp32', k_sample=8,
                 subtyping=False, instance_loss='svm', bag_weight=.7, layers=2,
                 aggregation_heads=None, ilra_landmarks=64, minimum_epochs=10,
                 max_dense_bytes=2_000_000_000):
        super().__init__()
        if profile not in HEADS or bundle.task_type != 'classification':
            raise ValueError('MAMMOTH profiles implement the registered classification track')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('MAMMOTH classification uses single-slide bags')
        self.profile, self.kind, self.dim, self.classes = profile, HEADS[profile], bundle.dim, len(bundle.classes)
        hidden = (256 if self.kind == 'ilra' else 512) if hidden is None else hidden
        # The published ILRA YAML sets embed_dim=slot_dim=256, rejected by Mammoth.
        # This named repair keeps ILRA width and halves the routing width explicitly.
        slot_dim = (128 if self.kind == 'ilra' else 256) if slot_dim is None else slot_dim
        aggregation_heads = (4 if self.kind == 'transmil' else 8) if aggregation_heads is None else aggregation_heads
        dimensions = [self.dim, hidden, attention_dim, experts, slots, heads, slot_dim, rank,
                      k_sample, layers, aggregation_heads, ilra_landmarks, minimum_epochs, max_dense_bytes]
        if any(type(v) is not int or v < 1 for v in dimensions) or self.classes < 2:
            raise ValueError('MAMMOTH dimensions/budgets must be positive integers with at least two classes')
        if any(type(v) is not bool for v in [auto_rank, keep_slots, share_weights, use_layernorm, subtyping]):
            raise ValueError('MAMMOTH flags must be booleans')
        if not share_weights:
            raise ValueError('Pinned MAMMOTH nonshared bias broadcasts to 5D; only the author-recommended shared weights are supported')
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v < 1
               for v in [dropout, feature_dropout, expert_dropout]):
            raise ValueError('MAMMOTH dropout probabilities must be finite and in [0,1)')
        if hidden == slot_dim or any(v % heads for v in [self.dim, hidden, slot_dim]):
            raise ValueError('MAMMOTH input/output/routing widths must divide heads; output and routing widths must differ')
        if self.kind in {'ilra', 'transmil', 'transformer'} and hidden % aggregation_heads:
            raise ValueError('MAMMOTH aggregation width must divide its attention head count')
        actual_rank = int((self.dim*hidden-self.dim*slot_dim)/(slot_dim+experts*hidden)+.5) if auto_rank else rank
        if actual_rank < 1:
            raise ValueError('MAMMOTH author auto-rank gives a zero/negative rank; choose explicit valid dimensions, no rank clamp')
        if precision not in {'fp32', 'amp'} or (precision == 'amp' and not torch.cuda.is_available()):
            raise ValueError('MAMMOTH precision must be fp32 or actual CUDA amp')
        if instance_loss not in {'svm', 'ce'} or not isinstance(bag_weight, (int, float)) or not 0 < bag_weight < 1:
            raise ValueError('MAMMOTH CLAM retains active instance loss and bag_weight strictly between zero and one')
        if self.kind == 'clam' and keep_slots and experts*slots < k_sample:
            raise ValueError('MAMMOTH CLAM needs at least k_sample output slots')
        self.hidden, self.experts, self.slots, self.heads, self.slot_dim = hidden, experts, slots, heads, slot_dim
        self.max_dense_bytes, self.keep_slots, self.precision = max_dense_bytes, keep_slots, precision
        self.minimum_epochs, self.instance_loss = minimum_epochs, instance_loss
        self.epoch, self.optimizer_steps, self.skipped_updates = -1, 0, 0
        self.stop_after_epoch = minimum_epochs-1
        self.input_dropout = nn.Dropout(feature_dropout)
        self.moe_args = {'input_dim': self.dim, 'dim': hidden, 'num_experts': experts, 'num_slots': slots,
            'num_heads': heads, 'slot_dim': slot_dim, 'lora_rank': rank, 'auto_rank': auto_rank,
            'keep_slots': keep_slots, 'share_lora_weights': share_weights, 'use_layernorm': use_layernorm,
            'dropout': expert_dropout, 'slot_dropout': 0.}
        self.reserve(1)
        with contextlib.redirect_stdout(io.StringIO()):
            if self.kind in {'mean', 'max'}:
                self.network = PoolHead(hidden, self.classes, self.moe_args.copy(), self.kind)
            else:
                args = {'in_dim': self.dim, 'embed_dim': hidden, 'num_classes': self.classes,
                        'moe_args': self.moe_args.copy()}
                if self.kind in {'abmil', 'dsmil'}:
                    args.update(num_fc_layers=1, dropout=dropout, attn_dim=attention_dim)
                elif self.kind == 'clam':
                    args.update(n_fc_layers=1, dropout=dropout, attention_dim=attention_dim, k_sample=k_sample,
                                subtyping=subtyping, instance_loss_fn='ce', bag_weight=bag_weight)
                elif self.kind in {'transmil', 'transformer'}:
                    args.update(num_fc_layers=1, dropout=dropout, num_attention_layers=layers, num_heads=aggregation_heads)
                    if self.kind == 'transformer':
                        args['encoder_mlp_dim'] = 4*hidden
                else:
                    args.update(num_heads=aggregation_heads, topk=ilra_landmarks,
                                num_attention_layers=layers, ln=True, mode='classification')
                self.network = author_head(self.kind)(**args)
        if self.kind == 'clam' and instance_loss == 'svm':
            self.network.instance_loss_fn = BinarySmoothTop1()
        self.scaler = torch.amp.GradScaler('cuda', enabled=precision == 'amp')
        self.optimizer, self.scheduler, self.scheduler_state = None, None, None

    def reserve(self, n):
        # Bound routing and downstream attention *before* allocating their activations.
        m = self.experts*self.slots if self.keep_slots else n
        rank = int((self.dim*self.hidden-self.dim*self.slot_dim)/(self.slot_dim+self.experts*self.hidden)+.5) if self.moe_args['auto_rank'] else self.moe_args['lora_rank']
        params = self.dim*self.slot_dim+self.heads*rank*(self.slot_dim//self.heads+self.experts*self.hidden//self.heads)
        estimate = 64*n*self.experts*self.heads*self.slots+64*m*self.hidden+24*params
        if self.kind in {'transformer', 'transmil', 'ilra'}:
            estimate += 128*(m+256)**2+256*self.hidden**2
        if estimate > self.max_dense_bytes:
            raise ValueError('MAMMOTH routing/attention exceeds max_dense_bytes; no silent patch truncation')

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('MAMMOTH retains the complete bag and unclipped AdamW recipe')
        if self.precision == 'amp' and next(self.parameters()).device.type != 'cuda':
            raise ValueError('MAMMOTH AMP requires an actual CUDA training device')
        self.total_epochs = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_epochs)
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch = epoch

    def finish_epoch(self, bundle, device, config):
        self.scheduler.step()

    def forward(self, x, coords=None, label=None, return_routing=False):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('MAMMOTH requires finite nonempty [instances,registered FM dimension] features')
        self.reserve(len(x))
        if self.kind == 'clam' and not self.keep_slots and len(x) < self.network.k_sample:
            raise ValueError('MAMMOTH CLAM bag contains fewer instances than k_sample')
        if label is not None and (label.shape != (1,) or label.dtype != torch.long or not 0 <= label.item() < self.classes):
            raise ValueError('MAMMOTH label must be a single in-range class index')
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.precision == 'amp'):
            h = self.input_dropout(x)[None]
            if self.kind == 'clam' and label is None:
                z, _ = self.network.forward_features(h)
                result = {'logits': self.network.forward_head(z)}
            else:
                result, _ = self.network(h, label=label, loss_fn=nn.CrossEntropyLoss() if label is not None else None)
            if label is not None:
                result['total_loss'] = result.pop('loss')
            if return_routing:
                _, result['dispatch_weights'] = self.network.patch_embed(h, return_weights=True)
        if not torch.isfinite(result['logits']).all() or ('total_loss' in result and not torch.isfinite(result['total_loss'])):
            raise ValueError('Nonfinite MAMMOTH logits or objective')
        return result

    def get_extra_state(self):
        return {'epoch': self.epoch, 'optimizer_steps': self.optimizer_steps, 'skipped_updates': self.skipped_updates,
                'scheduler': self.scheduler.state_dict() if self.scheduler else self.scheduler_state,
                'scaler': self.scaler.state_dict(), 'profile': self.profile, 'precision': self.precision}

    def set_extra_state(self, state):
        if state['profile'] != self.profile or state['precision'] != self.precision:
            raise ValueError('MAMMOTH checkpoint profile or precision differs')
        if any(type(state[k]) is not int or state[k] < 0 for k in ['optimizer_steps', 'skipped_updates']) or type(state['epoch']) is not int:
            raise ValueError('Invalid MAMMOTH optimizer/epoch counters')
        self.epoch, self.optimizer_steps, self.skipped_updates = state['epoch'], state['optimizer_steps'], state['skipped_updates']
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        self.scaler.load_state_dict(state['scaler'])

    def protocol(self):
        return {'profile': self.profile, 'source': 'mahmoodlab/MAMMOTH pinned native core and MIL-Lab heads',
            'moe': self.moe_args, 'actual_rank': self.network.patch_embed.lora_rank,
            'head': self.kind, 'head_width': self.hidden, 'precision': self.precision,
            'optimizer': 'AdamW; per-epoch cosine to zero; one sampled slide per update, no clipping',
            'sampling': 'TRAIN inverse-class-frequency multinomial, replacement, one epoch=TRAIN slides',
            'feature_dropout': self.input_dropout.p, 'minimum_epochs': self.minimum_epochs,
            'optimizer_steps': self.optimizer_steps, 'amp_skipped_updates': self.skipped_updates,
            'scheduler': self.scheduler.state_dict() if self.scheduler else self.scheduler_state,
            'objective': 'CLAM .7 bag CE + .3 instance SmoothTop1/declared CE; other heads native CE (DSMIL averaged-logit CE)',
            'mean_max': 'feature mean before classifier; max takes one complete logit vector from globally winning slot',
            'selection': 'validation-only declared metric, default log_loss; paper does not specify selection metric',
            'source_differences': ['published README/YAML uses 10 slots vs paper main9/encoder-ablation6; explicit configurable slots',
                'author auto_rank rounds to nearest, paper formula floors; native code retained',
                'nonshared optional branch has a native 5D broadcast error and is rejected; recommended shared branch is unchanged',
                'ILRA named repair routing128 instead of invalid slot_dim=embed_dim=256' if self.kind == 'ilra' else 'head architecture unmodified',
                'CLAM binary SVM evaluated device-neutrally, no mandatory CUDA allocation' if self.kind == 'clam' else 'no loss replacement',
                'FP32 is explicit CPU default; paper CUDA mixed precision requires precision=amp'],
            'routing_attention': 'dispatch rows are patch-to-expert/head/slot weights; slot MIL attention is not physical patch attention',
            'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run'}


def mammoth_epoch(model, bundle, groups, optimizer, device, cfg):
    from .engine import load_unit
    labels = torch.tensor([g[0]['target'] for g in groups])
    counts = torch.bincount(labels, minlength=model.classes)
    if torch.any(counts == 0):
        raise ValueError('MAMMOTH weighted TRAIN sampling requires every registered class')
    probabilities = (1/counts.double())[labels]
    order = torch.multinomial(probabilities, len(groups), replacement=True)
    losses = []
    for index in order.tolist():
        rows = groups[index]
        x, coords = load_unit(bundle, rows, device, None, train=True)
        target = torch.tensor([rows[0]['target']], device=device)
        optimizer.zero_grad(set_to_none=True)
        result = model(x, coords, target)
        loss = result['total_loss']
        model.scaler.scale(loss).backward()
        model.scaler.unscale_(optimizer)
        if not model.scaler.is_enabled() and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite MAMMOTH training gradients')
        previous_scale = model.scaler.get_scale()
        model.scaler.step(optimizer)
        model.scaler.update()
        skipped = model.scaler.get_scale() < previous_scale
        model.optimizer_steps += int(not skipped)
        model.skipped_updates += int(skipped)
        losses.append({'loss': float(loss.detach()), 'amp_skipped_update': float(skipped)})
    return losses

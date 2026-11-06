"""PAMoE author backbones, tissue-prior routing and full batch training recipe."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from .upstream import load_definitions

PROFILES = {'pamoe_transmil_source', 'pamoe_transformer_source', 'pamoe_transmil', 'pamoe_transformer'}


def author_pamoe(prototypes, dim, hidden, repaired):
    from nystrom_attention import NystromAttention
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'copy': copy, 'random': random,
             'LayerNorm': nn.LayerNorm, 'pad_sequence': pad_sequence, 'Sampler': Sampler,
             'Tensor': torch.Tensor, 'NystromAttention': NystromAttention, 'prototypes': prototypes}
    names = ['get_activation_fn', 'FeedForwardNetwork', 'get_x_cos_similarity', 'cal_prior_loss_ce',
             'remove_zero_vectors', 'drop_patch_cal_ce']
    module = load_definitions('pamoe', 'models/pamoe_layers/pamoe_utils.py', names, scope)
    scope.update({n: getattr(module, n) for n in names})
    patches = [] if not repaired else [
        ('expert_input = x_in_reshaped[i * batch_size: (i + 1) * batch_size]',
         'expert_input = x_in.view(batch_size, self.n_experts, top_k, embed_dim)[:, i]')]
    module = load_definitions('pamoe', 'models/pamoe_layers/pamoe.py', ['PAMoE'], scope, patches)
    scope['PAMoE'] = module.PAMoE
    load_proto = [('self.proto_types = torch.load(prototype_pth, map_location=\'cpu\')',
                   "self.register_buffer('proto_types', prototypes.clone())")]
    trans = load_definitions('pamoe', 'models/models_demo/model_transmil_pamoe.py',
        ['TransLayer', 'PPEG', 'transmil_pmoe'], scope,
        replacements=[*load_proto, ('dim=512', f'dim={hidden}'), ('nn.Linear(1024, 512)', f'nn.Linear({dim}, {hidden})'),
                      ('torch.randn(1, 1, 512)', f'torch.randn(1, 1, {hidden})'),
                      ('nn.LayerNorm(512)', f'nn.LayerNorm({hidden})'), ('nn.Linear(512, self.n_classes)', f'nn.Linear({hidden}, self.n_classes)')])
    transformer = load_definitions('pamoe', 'models/models_demo/model_transformer_pamoe.py',
        ['TransformerEncoderBlock', 'TransformerEncoder'], scope, replacements=load_proto)
    loss = load_definitions('pamoe', 'utils/utils_loss.py', ['CoxPHLoss'], scope)
    sampler = load_definitions('pamoe', 'utils/utils.py', ['Sampler_custom'], scope)
    return trans.transmil_pmoe, transformer.TransformerEncoder, loss.CoxPHLoss, sampler.Sampler_custom


class PAMoE(nn.Module):
    stop_after_epoch = 25

    def __init__(self, bundle, profile='pamoe_transmil', hidden=512, batch_size=8,
                 extra_experts=2, capacity=2., prior_weight=.01, alpha_surv=0.,
                 max_train_patches=20000, layer_types=None):
        super().__init__()
        if bundle.pamoe is None:
            raise ValueError('PAMoE requires versioned tissue prototypes in the selected FM space')
        if profile not in PROFILES or bundle.task_type not in {'classification', 'survival'}:
            raise ValueError('PAMoE profile/task not supported')
        if (any(type(v) is not int or v < 1 for v in (hidden, batch_size, max_train_patches)) or
                hidden % 8 or hidden < 8 or type(extra_experts) is not int or extra_experts < 0):
            raise ValueError('Invalid PAMoE dimensions, batch or expert budget; hidden must be divisible by 8')
        self.prototype_count = len(bundle.pamoe.tensor)
        if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v)
                for v in (capacity, prior_weight, alpha_surv)) or
                not 0 < capacity <= self.prototype_count+extra_experts or prior_weight <= 0 or not 0 <= alpha_surv < 1):
            raise ValueError('PAMoE requires active prior loss, valid capacity and censor weight')
        self.profile, self.hidden, self.batch_size = profile, hidden, batch_size
        self.dim, self.task_type = bundle.dim, bundle.task_type
        self.max_train_patches, self.prior_weight, self.alpha_surv = max_train_patches, prior_weight, alpha_surv
        self.capacity, self.extra_experts = capacity, extra_experts
        self.asset_metadata = copy.deepcopy(bundle.pamoe.metadata)
        self.layer_types = ['pamoe', 'ffn', 'pamoe', 'ffn'] if layer_types is None else layer_types
        if not isinstance(self.layer_types, list) or not self.layer_types or 'pamoe' not in self.layer_types or set(self.layer_types)-{'pamoe', 'ffn'}:
            raise ValueError('Transformer PAMoE needs valid layer_types including at least one pamoe')
        if 'transmil' in profile and layer_types is not None:
            raise ValueError('TransMIL PAMoE has two fixed layers; layer_types is not applicable')
        self.repaired = not profile.endswith('_source')
        trans, transformer, cox, self.sampler_class = author_pamoe(bundle.pamoe.tensor, bundle.dim, hidden, self.repaired)
        classes = 1 if bundle.task_type == 'survival' else len(bundle.classes)
        self.network_args = {'n_classes': classes, 'capacity_factor': capacity,
                             'num_expert_proto': self.prototype_count, 'num_expert_extra': extra_experts}
        if 'transmil' in profile:
            self.network = trans(**self.network_args)
        else:
            self.network_args.update(layer_type=self.layer_types, input_dim=bundle.dim, embed_dim=hidden,
                                     num_heads=8, ff_dim_mult=4, dropout=.1, use_cls_token=True,
                                     drop_zeros=True, pamoe_use_residual=True)
            self.network = transformer(**self.network_args)
        self.criterion = cox(alpha_surv) if self.task_type == 'survival' else nn.CrossEntropyLoss()
        self.scheduler, self.scheduler_state = None, None
        self.optimizer_steps = 0
        self.fit_info, self.last_sampled_patients = {}, []

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('PAMoE uses max_train_patches only at training; do not set global max_instances')

    def prepare(self, bundle, seed, device):
        from .engine import units
        train = units(bundle, 'train')
        if len(train) < self.batch_size:
            raise ValueError('PAMoE source drop-last recipe needs at least one full training batch')
        if self.task_type == 'survival' and not any(rows[0]['event'] for rows in train):
            raise ValueError('PAMoE source Cox sampler needs observed training events')
        self.fit_info = {'prototype_asset': self.asset_metadata, 'training_patients': sorted({r[0]['global_patient_id'] for r in train})}

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def finish_epoch(self, bundle, device, config):
        self.scheduler.step()

    def get_extra_state(self):
        return {'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state,
                'fit_info': self.fit_info, 'optimizer_steps': self.optimizer_steps,
                'last_sampled_patients': self.last_sampled_patients}

    def set_extra_state(self, state):
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        self.fit_info, self.optimizer_steps = state['fit_info'], state['optimizer_steps']
        self.last_sampled_patients = state['last_sampled_patients']

    def forward_batch(self, bags, training=False):
        if not bags or any(x.ndim != 2 or x.shape[1] != self.dim or len(x) == 0 or not torch.isfinite(x).all() for x in bags):
            raise ValueError('PAMoE needs finite nonempty [patches,FM_dim] bags')
        x = pad_sequence(bags, batch_first=True, padding_value=0.)
        if training:
            x = x[:, :self.max_train_patches]
        gates, logits, prior = self.network(x)
        if not torch.isfinite(logits).all() or prior is None or not torch.isfinite(prior):
            raise ValueError('PAMoE produced nonfinite predictions or missing/nonfinite prior loss')
        return {'logits': logits, 'prior_loss': prior, 'gates': gates}

    def forward(self, x, coords=None):
        return self.forward_batch([x], training=self.training)

    def objective(self, output, rows):
        device = output['logits'].device
        if self.task_type == 'survival':
            times = torch.tensor([r[0]['time'] for r in rows], device=device)
            censor = torch.tensor([1-r[0]['event'] for r in rows], device=device)
            primary = self.criterion(S=output['logits'], c=censor, event_time=times)
        else:
            primary = self.criterion(output['logits'], torch.tensor([r[0]['target'] for r in rows], device=device))
        return primary + self.prior_weight*output['prior_loss'], primary

    def protocol(self):
        return {'profile': self.profile, 'recipe_scope': 'author networks, tissue-prior CE and source batch recipe',
                'prototype_asset': self.asset_metadata, 'hidden': self.hidden, 'batch_size': self.batch_size,
                'routing': 'corrected B,E indexing' if self.repaired else 'released flattened expert slicing',
                'prior_target': 'softmax(cosine) over all batch tokens; filtered targets not renormalized',
                'padding': 'source zero padding without masks; may affect predictions',
                'sampler': 'source event-conditioned, without replacement, drop incomplete/censor-only remainder' if self.task_type == 'survival' else 'source shuffled complete batches',
                'objective': 'source Cox sorted cumulative risk (ties order dependent)' if self.task_type == 'survival' else 'cross entropy',
                'prior_weight': self.prior_weight, 'alpha_surv': self.alpha_surv,
                'max_train_patches': self.max_train_patches, 'eval_patches': 'all; single bag per call',
                'scheduler': 'source CosineAnnealingLR T_max=5 eta_min=1e-5; cycles beyond epoch5',
                'optimization': 'AdamW per batch; source gc and lambda_reg options are inactive',
                'precision': 'FP32; no AMP or multi-device DataParallel validation',
                'early_stopping': 'validation-only patience with source minimum stop_epoch=25',
                'source_numeric_errors': 'fail visibly instead of silently skipping cases/epochs'}


def pamoe_batches(model, groups):
    if model.task_type == 'survival':
        indices = list(model.sampler_class([1-r[0]['event'] for r in groups], model.batch_size))
    else:
        indices = list(range(len(groups)))  # engine already shuffled the training units
    n = len(indices)//model.batch_size*model.batch_size
    if n == 0:
        raise ValueError('PAMoE sampler produced no complete event-containing batch')
    return [[groups[i] for i in indices[start:start+model.batch_size]] for start in range(0, n, model.batch_size)]


def pamoe_epoch(model, bundle, groups, optimizer, device, config):
    from .engine import load_unit
    records = []
    batches = pamoe_batches(model, groups)
    model.last_sampled_patients = [[rows[0]['global_patient_id'] for rows in batch] for batch in batches]
    for batch in batches:
        bags = [load_unit(bundle, rows, device)[0] for rows in batch]
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_batch(bags, training=True)
        loss, primary = model.objective(output, batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite PAMoE loss; no silent case skipping')
        loss.backward()
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite PAMoE gradient')
        optimizer.step()
        model.optimizer_steps += 1
        records.append({'loss': float(loss.detach()), 'primary_loss': float(primary.detach()),
                        'prior_loss': float(output['prior_loss'].detach())})
    return records

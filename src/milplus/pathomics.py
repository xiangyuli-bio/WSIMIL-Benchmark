"""Locked PathOmics networks, alignment pretraining and survival fine-tuning.

Source freeze behavior is explicit. Snapshot, tail accumulation and validation
isolation repairs are recorded in docs/PATHOMICS.md. No author code is vendored.
"""
import math
import random
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .multimodal import MolecularSurvival
from .stochastic import inference_rng
from .upstream import load_definitions

PROFILES = {'pathomics_concat_source', 'pathomics_image_source', 'pathomics_omics_source'}
MODEL_FILE = 'PathOmics/model_and_training_utils/PathOmics_Survival_model.py'
# These are the exact module prefixes named by bash_main.py's fine-tuning list.
HEAD_PREFIXES = ('path_attention_head.', 'omic_attention_head.', 'path_rho.', 'omic_rho.',
                 'mm.', 'classifier.', 'path_proj.', 'omic_proj.')


def author_pathomics(dim=1024, hidden=256):
    scope = {**{k: v for k, v in vars(F).items() if not k.startswith('__')}, 'torch': torch, 'nn': nn, 'F': F, 'np': np, 'random': random,
             'Optional': Optional, 'Tensor': torch.Tensor, 'Module': nn.Module, 'Parameter': nn.Parameter,
             'NonDynamicallyQuantizableLinear': nn.modules.linear.NonDynamicallyQuantizableLinear,
             'xavier_uniform_': nn.init.xavier_uniform_, 'xavier_normal_': nn.init.xavier_normal_,
             'constant_': nn.init.constant_}
    names = ['residual_block', 'DimReduction', 'Attention_Gated']
    mod = load_definitions('pathomics', 'PathOmics/model_and_training_utils/PathOmics_model_utils.py', names, scope)
    scope.update({name: getattr(mod, name) for name in names})
    mod = load_definitions('pathomics', MODEL_FILE,
        ['Omics_Attention', 'PathOmics_Surv', 'SNN_Block', 'Attn_Net_Gated',
         'multi_head_attention_forward', 'MultiheadAttention'], scope, replacements=[
            ('[1024, 256, 256]', f'[{dim}, {hidden}, {hidden}]'),
            ("'small': [256, 256]", f"'small': [{hidden}, {hidden}]"),
            ('embed_dim=256', f'embed_dim={hidden}'),
            ('d_model=256, nhead=8, dim_feedforward=512', f'd_model={hidden}, nhead=8, dim_feedforward={2*hidden}'),
            ('nn.Linear(256*2, size[2])', f'nn.Linear({hidden}*2, size[2])'),
            ('torch.LongTensor(tindex).to(self.device)', 'torch.LongTensor(tindex).to(x_path.device)'),
            ('return hazards, S, Y_hat, attention_scores, None', 'return hazards, S, Y_hat, attention_scores, logits'),
        ])
    loss = load_definitions('pathomics', 'PathOmics/model_and_training_utils/Customized_Loss.py',
                            ['nll_loss', 'NLLSurvLoss'], {'torch': torch})
    mod.NLLSurvLoss = loss.NLLSurvLoss
    return mod


def fine_trainable_names(network, policy):
    names = {name for name, _ in network.named_parameters()}
    heads = {name for name in names if name == 'logit_scale' or name.startswith(HEAD_PREFIXES)}
    if policy == 'heads':
        return heads
    if policy == 'source_zip':
        # Literal released zip(named_parameters, children): later parameters stay trainable.
        visited = {name for (name, _), _child in zip(network.named_parameters(), network.children())}
        return names - (visited - heads)
    raise ValueError('PathOmics freeze_policy must be source_zip or explicit heads repair')


class PathOmicsSurvival(MolecularSurvival):
    def __init__(self, dim, omics, profile='pathomics_concat_source', hidden=256, bins=4,
                 pretrain_epochs=25, accumulation_steps=32, finetune_lr=5e-5,
                 finetune_weight_decay=0., alpha=0., omic_bag='SNN', use_gap=True,
                 proj_ratio=1., dropout=.25, freeze_policy='source_zip', inference_seed=2026):
        if omics is None or not omics.group_indices:
            raise ValueError('PathOmics requires aligned molecular values and explicit ordered gene groups')
        super().__init__('pathomics', dim, omics, bins, alpha, accumulation_steps)
        if profile not in PROFILES:
            raise ValueError('PathOmics supports concat/image/omics; released bilinear class is missing')
        if any(type(v) is not int or v < 1 for v in (dim, hidden, pretrain_epochs, accumulation_steps)) or hidden % 8:
            raise ValueError('PathOmics dimensions/budgets must be positive integers; hidden divisible by 8')
        if type(use_gap) is not bool or omic_bag not in {'SNN', 'Attention', 'SNN_Attention'}:
            raise ValueError('Invalid PathOmics molecular encoder or GAP flag')
        if (not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (proj_ratio, dropout, finetune_lr, finetune_weight_decay)) or
                proj_ratio <= 0 or int(hidden*proj_ratio) < 1 or not 0 <= dropout < 1 or
                finetune_lr <= 0 or finetune_weight_decay < 0 or type(inference_seed) is not int):
            raise ValueError('Invalid PathOmics projection, dropout, learning rate, decay or inference seed')
        self.profile, self.hidden, self.dim = profile, hidden, dim
        self.minimum_selection_epoch = pretrain_epochs
        self.finetune_lr, self.finetune_weight_decay = finetune_lr, finetune_weight_decay
        self.freeze_policy, self.inference_seed = freeze_policy, inference_seed
        self.omic_bag, self.use_gap, self.proj_ratio, self.dropout = omic_bag, use_gap, proj_ratio, dropout
        self.fusion = profile.split('_')[1]
        author = author_pathomics(dim, hidden)
        self.network_args = {'device': 'cpu', 'fusion': self.fusion, 'omic_sizes': [len(g) for g in self.group_indices],
                             'n_classes': bins, 'dropout': dropout, 'omic_bag': omic_bag,
                             'use_GAP_in_pretrain': use_gap, 'proj_ratio': proj_ratio, 'image_group_method': 'random'}
        self.network = author.PathOmics_Surv(**self.network_args)
        self.fine_names = fine_trainable_names(self.network, freeze_policy)
        self.criterion = author.NLLSurvLoss(alpha=alpha)
        self.epoch, self.transitioned = -1, False
        self.pretrain_best, self.pretrain_best_loss, self.pretrain_best_epoch = None, float('inf'), None
        self.stage_updates = [0, 0]

    @property
    def pretraining(self):
        return self.epoch < self.minimum_selection_epoch

    def configure_training(self, config):
        if config['epochs'] <= self.minimum_selection_epoch:
            raise ValueError('PathOmics epochs must include BOTH alignment pretraining and survival fine-tuning')
        if config['max_instances'] is not None:
            raise ValueError('PathOmics source recipe uses all patches; max_instances is not supported')

    def make_optimizers(self, lr, weight_decay):
        return [torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay),
                torch.optim.Adam([p for n, p in self.network.named_parameters() if n in self.fine_names],
                                  lr=self.finetune_lr, weight_decay=self.finetune_weight_decay)]

    def start_epoch(self, epoch):
        self.epoch = epoch
        if not self.pretraining and not self.transitioned:
            if self.pretrain_best is None:
                raise RuntimeError('PathOmics has no validated alignment checkpoint to fine-tune')
            self.network.load_state_dict(self.pretrain_best)
            self.transitioned = True
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad_(self.pretraining or name in self.fine_names)

    def get_extra_state(self):
        return {'epoch': self.epoch, 'transitioned': self.transitioned, 'pretrain_best': self.pretrain_best,
                'pretrain_best_loss': self.pretrain_best_loss, 'pretrain_best_epoch': self.pretrain_best_epoch,
                'stage_updates': self.stage_updates}

    def set_extra_state(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def _call(self, x, omics, mode):
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) < len(self.group_indices) or not torch.isfinite(x).all():
            raise ValueError('PathOmics needs finite [patches,dim] with at least one patch per gene group')
        molecular = self.normalize_omics(omics)
        groups = [molecular[indices] for indices in self.group_indices]
        context = nullcontext() if self.training else inference_rng(x.device, self.inference_seed)
        with context:
            return self.network(x, groups, [], mode=mode)

    def alignment(self, x, omics):
        path, molecular = self._call(x, omics, 'pretrain')
        return F.mse_loss(path, molecular)

    def forward(self, x, omics, coords=None):
        hazards, survival, _, _, logits = self._call(x, omics, 'Finetune')
        return {'logits': -survival.sum(1, keepdim=True), 'hazard_logits': logits,
                'hazards': hazards, 'survival': survival}

    def objective(self, result, time, event):
        return self.criterion(result['hazards'], result['survival'], self.time_label([time]),
                              self.bin_edges.new_tensor([1-event]))

    def inputs(self, bundle, rows, device, train=False):
        from .engine import load_unit
        x, _ = load_unit(bundle, rows, device, train=train)
        return x, bundle.omics.tensor(rows[0]['global_patient_id']).to(device)

    def forward_unit(self, bundle, rows, device, config, train=False):
        return self(*self.inputs(bundle, rows, device, train))

    @torch.no_grad()
    def alignment_validation(self, bundle, device):
        from .engine import units
        self.eval()
        losses = [float(self.alignment(*self.inputs(bundle, rows, device))) for rows in units(bundle, 'val')]
        return {'alignment_loss': float(np.mean(losses)), 'stage': 'pretrain'}

    def after_validation(self, metrics):
        if self.pretraining:
            value = metrics['alignment_loss']
            if not math.isfinite(value):
                raise ValueError('Non-finite PathOmics validation alignment loss')
            if value < self.pretrain_best_loss:
                self.pretrain_best = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                self.pretrain_best_loss, self.pretrain_best_epoch = value, self.epoch

    def prediction_fields(self, result, row):
        return {'survival_loss': float(self.objective(result, row['time'], row['event']))}

    def validation_metrics(self, predictions):
        return {'survival_loss': float(predictions.survival_loss.mean()), 'stage': 'finetune'}

    def protocol(self):
        return {**super().protocol(), 'recipe_scope': 'source PathOmics MSE pretrain then NLL survival fine-tune',
                'profile': self.profile, 'hidden': self.hidden, 'pretrain_epochs': self.minimum_selection_epoch,
                'pretrain_selection': 'minimum validation alignment MSE; immutable CPU snapshot',
                'pretrain_best_epoch': self.pretrain_best_epoch,
                'pretrain_best_loss': self.pretrain_best_loss if math.isfinite(self.pretrain_best_loss) else None,
                'accumulation': 'pretrain=sum; finetune=mean; flush final partial window',
                'fine_tune_freeze_policy': self.freeze_policy, 'fine_tune_trainable': sorted(self.fine_names),
                'finetune_lr': self.finetune_lr, 'finetune_weight_decay': self.finetune_weight_decay,
                'omic_bag': self.omic_bag, 'use_gap': self.use_gap, 'proj_ratio': self.proj_ratio,
                'grouping': 'source random shuffle and equal-count split; N must be >= gene groups',
                'inference_grouping': f'isolated seed {self.inference_seed}; does not consume training RNG',
                'stage_updates': self.stage_updates, 'source_bilinear': 'not admitted; missing BilinearFusion definition',
                'single_modality_scope': 'output branch is image-only or omics-only; bundle still supplies paired data',
                'native_dimension': 1024, 'native_hidden': 256}


def pathomics_epoch(model, bundle, groups, optimizers, device, config):
    stage = 0 if model.pretraining else 1
    optimizer = optimizers[stage]
    records = []
    for start in range(0, len(groups), model.accumulation_steps):
        window = groups[start:start+model.accumulation_steps]
        optimizer.zero_grad(set_to_none=True)
        for rows in window:
            x, molecular = model.inputs(bundle, rows, device, train=True)
            loss = (model.alignment(x, molecular) if model.pretraining else
                    model.objective(model(x, molecular), rows[0]['time'], rows[0]['event']))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite PathOmics training objective')
            (loss if model.pretraining else loss/len(window)).backward()
            records.append({'loss': float(loss.detach()), 'stage': stage})
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Non-finite PathOmics gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.stage_updates[stage] += 1
    return records

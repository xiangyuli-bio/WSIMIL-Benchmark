"""MMP released prototype/fusion networks and complete minibatch survival recipe.

Locked author definitions stay external. Numerical and data-path repairs are
explicit in docs/MMP.md; CPU software validation is not paper reproduction.
"""
import hashlib
import math
from collections import OrderedDict
from itertools import combinations

import numpy as np
import torch
from einops import rearrange, reduce
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .multimodal import MolecularSurvival
from .upstream import load_definitions

PROFILES = {'mmp_panther_source', 'mmp_panther_uot_source', 'mmp_ot_source', 'mmp_ot_uot_source'}


def author_mmp():
    try:
        import ot
    except ImportError as exc:
        raise ImportError('MMP requires the transport extra: pip install "mil-baseline-puls[transport]"') from exc
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'ot': ot,
             'einsum': torch.einsum, 'rearrange': rearrange, 'reduce': reduce, 'combinations': combinations}
    for path, names in [
        ('src/utils/losses.py', ['NLLSurvLoss', 'nll_loss', 'CoxLoss', 'partial_ll_loss', 'SurvRankingLoss']),
        ('src/mil_models/components.py', ['SNN_Block', 'FeedForward', 'FeedForwardEnsemble',
                                        'MMAttention', 'MMAttentionLayer', 'process_surv']),
    ]:
        module = load_definitions('mmp', path, names, scope)
        scope.update({name: getattr(module, name) for name in names})
    module = load_definitions('mmp', 'src/mil_models/model_multimodal.py',
        ['init_per_path_model', 'agg_histo', 'OT_Attn', 'construct_proto_embedding', 'coattn', 'coattn_mot'], scope,
        replacements=[
            ('init_per_path_model(omic_sizes)', 'init_per_path_model(omic_sizes, hidden_dim=path_proj_dim)'),
            ('.cuda()', '.to(weight1.device)'),
            ('M_cost = cost_map_detach / cost_map_detach.max()',
             'M_cost = cost_map_detach / cost_map_detach.max() if cost_map_detach.max() > 0 else cost_map_detach * 0'),
            ('reg_m=self.ot_tau)', "reg_m=self.ot_tau, reg_type='entropy')"),
        ])
    for name in ('NLLSurvLoss', 'CoxLoss'):
        setattr(module, name, scope[name])
    return module


def histology_encoder(dim, prototypes, kind, em_iterations, tau, eps, ot_iterations, ot_log_domain, variance_floor):
    import ot
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'ot': ot}
    if kind == 'PANTHER':
        mod = load_definitions('mmp', 'src/mil_models/PANTHER/networks.py', ['mog_eval', 'DirNIWNet'], scope,
            replacements=[('return pi.float(), mu.float(), Sigma.float()',
                           f'return pi.float(), mu.float(), Sigma.float().clamp_min({variance_floor!r})')])
        scope.update(mog_eval=mod.mog_eval, DirNIWNet=mod.DirNIWNet)
        mod = load_definitions('mmp', 'src/mil_models/PANTHER/layers.py', ['PANTHERBase'], scope)
        return mod.PANTHERBase(dim, p=prototypes, L=em_iterations, tau=tau, out='allcat', ot_eps=eps,
                               load_proto=False, fix_proto=True)
    names = ['compute_distance_euclidean', 'sinkhorn', 'log_sinkhorn', 'multihead_attn']
    mod = load_definitions('mmp', 'src/mil_models/OT/otk/sinkhorn.py', names, scope)
    scope.update({name: getattr(mod, name) for name in names})
    mod = load_definitions('mmp', 'src/mil_models/OT/otk/layers.py', ['OTKernel'], scope)
    net = mod.OTKernel(dim, prototypes, heads=1, eps=eps, max_iter=ot_iterations, distance='euclidean',
                       log_domain=ot_log_domain, position_encoding=None)
    return net.requires_grad_(False)


def prototype_sample(bundle, total, seed):
    """Source equal-per-training-slide quota, without NumPy/Torch aliasing or unfilled rows."""
    rows = sorted(bundle.records('train'), key=lambda r: (r['global_patient_id'], r['slide_id']))
    quota = math.ceil(total/len(rows))
    rng = np.random.RandomState(seed)
    parts, ledger, count = [], [], 0
    for row in rows:
        if count == total:
            break
        x, _ = bundle.load(row)
        n = min(quota, len(x), total-count)
        indices = rng.permutation(len(x))[:n]
        parts.append(x[torch.from_numpy(indices)].cpu().numpy().copy())
        ledger.append({'slide_id': row['slide_id'], 'patient_id': row['global_patient_id'], 'sampled': n})
        count += n
    return np.concatenate(parts).astype(np.float32, copy=False), ledger


class MultimodalPrototyping(MolecularSurvival):
    def __init__(self, dim, omics, profile='mmp_panther_source', hidden=256, prototypes=16,
                 batch_size=64, loss='cox_source', bins=4, alpha=.5, append_embed='random',
                 histo_agg='mean', net_indiv=True, dropout=.1, em_iterations=1, tau=1., eps=1.,
                 prototype_patches=100000, cluster_iterations=50, cluster_restarts=5,
                 ot_iterations=100, ot_log_domain=True, variance_floor=1e-7,
                 ot_reg=.1, ot_tau=.5, warmup_epochs=None, cache_patients=128):
        if omics is None or not omics.group_indices:
            raise ValueError('MMP requires aligned molecular values and explicit ordered pathways')
        super().__init__('mmp', dim, omics, bins=bins, alpha=alpha)
        ints = (dim, hidden, prototypes, batch_size, em_iterations, prototype_patches,
                cluster_iterations, cluster_restarts, ot_iterations)
        if any(type(v) is not int or v < 1 for v in ints) or hidden < 2:
            raise ValueError('MMP dimensions and budgets must be positive integers, hidden >= 2')
        if profile not in PROFILES or loss not in {'cox_source', 'cox_breslow', 'nll'}:
            raise ValueError('Unsupported MMP profile/loss; ranking-loss experiment is not exposed')
        if append_embed not in {'random', 'modality', 'proto', 'none'} or histo_agg not in {'mean', 'cat'}:
            raise ValueError('Invalid MMP token embedding or aggregation')
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (tau, eps, variance_floor, ot_reg, ot_tau)):
            raise ValueError('MMP regularizers and numerical floor must be finite positive')
        if (type(net_indiv) is not bool or type(ot_log_domain) is not bool or
                not isinstance(dropout, (int, float)) or not 0 <= dropout < 1 or
                type(cache_patients) is not int or cache_patients < 0 or
                (warmup_epochs is not None and (type(warmup_epochs) is not int or warmup_epochs < 0))):
            raise ValueError('Invalid MMP dropout, boolean, cache or warmup setting')
        self.profile, self.dim, self.hidden, self.prototypes = profile, dim, hidden, prototypes
        self.batch_size, self.loss_kind, self.append_embed = batch_size, loss, append_embed
        self.histo_kind = 'PANTHER' if 'panther' in profile else 'OT'
        self.prototype_patches, self.cluster_iterations, self.cluster_restarts = prototype_patches, cluster_iterations, cluster_restarts
        self.warmup_epochs, self.cache_patients = warmup_epochs, cache_patients
        self.histology = histology_encoder(dim, prototypes, self.histo_kind, em_iterations, tau, eps,
                                           ot_iterations, ot_log_domain, variance_floor)
        self.tokenizer = load_definitions('mmp', 'src/mil_models/tokenizer.py', ['PrototypeTokenizer'],
                                          {'nn': nn, 'torch': torch}).PrototypeTokenizer(self.histo_kind, 'allcat', prototypes)
        author = author_mmp()
        args = {'omic_sizes': [len(g) for g in self.group_indices], 'histo_in_dim': dim,
                    'num_classes': bins if loss == 'nll' else 1, 'path_proj_dim': hidden, 'append_embed': append_embed,
                    'histo_agg': histo_agg, 'net_indiv': net_indiv, 'numOfproto': prototypes,
                    'histo_model': self.histo_kind, 'num_coattn_layers': 1, 'dropout': dropout}
        self.network_args = args
        self.network = author.coattn_mot(**args, ot_reg=ot_reg, ot_tau=ot_tau) if '_uot_' in profile else author.coattn(**args)
        self.criterion = author.NLLSurvLoss(alpha=alpha) if loss == 'nll' else author.CoxLoss()
        self.register_buffer('schedule', torch.tensor([50, 1, 1], dtype=torch.int64))  # epochs, warmup epochs, steps/epoch
        self.fit_info, self.token_cache = {}, OrderedDict()
        self.scheduler, self.scheduler_state = None, None
        self.numerics = {'em_iterations': em_iterations, 'tau': tau, 'eps': eps, 'variance_floor': variance_floor,
                         'ot_iterations': ot_iterations, 'ot_log_domain': ot_log_domain, 'ot_reg': ot_reg, 'ot_tau': ot_tau}

    @property
    def centers(self):
        return self.histology.priors.m if self.histo_kind == 'PANTHER' else self.histology.weight[0]

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('MMP tokenizes every patient patch; max_instances is not a source recipe option')
        self.selection_metric = config.get('selection_metric') or 'survival_loss'
        self.schedule[0] = config['epochs']
        self.schedule[1] = self.warmup_epochs if self.warmup_epochs is not None else int(config['lr'] > 5e-5)
        if int(self.schedule[1]) >= config['epochs']:
            raise ValueError('MMP needs at least one epoch after warmup; use warmup_epochs=0 for a one-epoch run')
        if config['grad_clip'] is not None and (not math.isfinite(config['grad_clip']) or config['grad_clip'] <= 0):
            raise ValueError('grad_clip must be positive finite or null')

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        if self.loss_kind == 'nll':
            super().prepare(bundle, seed, device)
        else:
            mean, scale = bundle.omics.training_statistics(bundle)
            self.omics_mean.copy_(mean)
            self.omics_scale.copy_(scale)
            self.prepared.fill_(True)
        x, ledger = prototype_sample(bundle, self.prototypes*self.prototype_patches, seed)
        if len(x) < self.prototypes or len(np.unique(x, axis=0)) < self.prototypes:
            raise ValueError('MMP training pool has fewer distinct patches than prototypes')
        km = KMeans(n_clusters=self.prototypes, n_init=self.cluster_restarts,
                    max_iter=self.cluster_iterations, random_state=seed).fit(x)
        self.centers.copy_(torch.from_numpy(km.cluster_centers_).to(self.centers))
        self.fit_info = {'seed': seed, 'sampled_patches': len(x), 'training_slides': ledger,
                         'prototype_sha256': hashlib.sha256(self.centers.cpu().numpy().tobytes()).hexdigest(),
                         'iterations': int(km.n_iter_), 'inertia': float(km.inertia_)}
        patients = bundle.rows[bundle.rows.subset == 'train'].global_patient_id.nunique()
        self.schedule[2] = math.ceil(patients/self.batch_size)
        self.token_cache.clear()

    @torch.no_grad()
    def tokenize(self, x):
        if not self.prepared:
            raise RuntimeError('Fit training prototypes and molecular normalization first')
        if x.ndim != 2 or not len(x) or x.shape[1] != self.dim or not torch.isfinite(x).all():
            raise ValueError('MMP expects finite nonempty patch features [N,D]')
        # Content-addressed bounded memory cache. No test outcome enters the key or tokens.
        key = hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes() +
                             self.centers.detach().cpu().contiguous().numpy().tobytes()).digest()
        if key in self.token_cache:
            self.token_cache.move_to_end(key)
            return self.token_cache[key].to(x.device)
        if self.histo_kind == 'PANTHER':
            flat, _ = self.histology(x[None])
            pi, mean, cov = self.tokenizer(flat)
            tokens = torch.cat([pi[..., None], mean, cov], -1)
        else:
            # Released trainer attempts Tensor(None) for OT covariance; OT fusion expects means only.
            tokens = self.histology(x[None])
        if not torch.isfinite(tokens).all():
            raise FloatingPointError('Nonfinite MMP histology tokens; inspect feature scale and prototype settings')
        if self.cache_patients:
            self.token_cache[key] = tokens.cpu()
            while len(self.token_cache) > self.cache_patients:
                self.token_cache.popitem(last=False)
        return tokens

    def batch_logits(self, tokens, molecules):
        if molecules.ndim != 2 or molecules.shape[1] != len(self.columns) or len(tokens) != len(molecules):
            raise ValueError('MMP patient token/molecular batch mismatch')
        if not self.prepared or not torch.isfinite(tokens).all() or not torch.isfinite(molecules).all():
            raise ValueError('MMP needs fitted preprocessing and finite inputs')
        normalized = (molecules-self.omics_mean)/self.omics_scale
        if not torch.isfinite(normalized).all():
            raise ValueError('MMP molecular normalization overflow')
        logits = self.network.forward_no_loss(tokens, [normalized[:, g] for g in self.group_indices])['logits']
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite MMP logits')
        return logits

    def batch_loss(self, logits, times, events):
        times = torch.as_tensor(times, dtype=torch.float64, device=logits.device).reshape(-1)
        events = torch.as_tensor(events, dtype=logits.dtype, device=logits.device).reshape(-1)
        if (len(logits) != len(times) or len(times) != len(events) or not len(times) or
                not torch.isfinite(times).all() or (times <= 0).any() or
                not torch.isin(events, events.new_tensor([0, 1])).all()):
            raise ValueError('Invalid MMP aligned event/time targets')
        if self.loss_kind == 'cox_breslow':
            from .recipes import cox_breslow_loss
            loss = cox_breslow_loss(logits.flatten(), times, events) if events.any() else logits.sum()*0
        else:
            labels = self.time_label(times) if self.loss_kind == 'nll' else times
            loss = self.criterion(logits, labels[:, None], (1-events)[:, None])['loss'].squeeze()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite MMP loss')
        return loss

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        if train or self.training:
            raise ValueError('MMP requires patient minibatches through mmp_epoch')
        x, _ = load_unit(bundle, rows, device, None, False)
        molecules = bundle.omics.tensor(rows[0]['global_patient_id']).to(device)[None]
        logits = self.batch_logits(self.tokenize(x), molecules)
        return self.result(logits) if self.loss_kind == 'nll' else {'logits': logits}

    def prediction_fields(self, result, row):
        if self.loss_kind != 'nll':
            return {}
        return {'survival_loss': float(self.batch_loss(result['hazard_logits'], [row['time']], [row['event']]))}

    def validation_metrics(self, predictions):
        if self.loss_kind == 'nll':
            return {'survival_loss': float(predictions.survival_loss.mean())}
        risk = torch.as_tensor(predictions.risk.to_numpy(), device=self.centers.device, dtype=torch.float32)[:, None]
        return {'survival_loss': float(self.batch_loss(risk, predictions.time.to_numpy(), predictions.event.to_numpy()))}

    def make_optimizer(self, lr, weight_decay):
        def exclude(name, p):
            return p.ndim < 2 or 'bn' in name or 'ln' in name or 'bias' in name or 'logit_scale' in name
        named = list(self.network.named_parameters())
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in named if p.requires_grad and exclude(n, p)], 'weight_decay': 0.},
            {'params': [p for n, p in named if p.requires_grad and not exclude(n, p)], 'weight_decay': weight_decay},
        ], lr=lr)
        def multiplier(step):
            epochs, warmup_epochs, steps = self.schedule.tolist()
            warmup, total = warmup_epochs*steps, epochs*steps
            if step < warmup:
                return step/max(1, warmup)
            if not warmup_epochs and self.warmup_epochs is None and lr <= 5e-5:
                return 1.  # released low-LR script selects constant schedule
            progress = (step-warmup)/max(1, total-warmup)
            return max(0., .5*(1+math.cos(math.pi*progress)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def get_extra_state(self):
        return {'fit_info': self.fit_info,
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        self.fit_info, self.scheduler_state = state['fit_info'], state['scheduler']
        self.token_cache.clear()
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def protocol(self):
        base = super().protocol() if self.loss_kind == 'nll' else {
            'event_definition': '1=observed; 0=censored', 'bin_fit': 'not used for Cox',
            'omics_columns': self.columns, 'pathway_names': self.group_names,
            'normalization': self.normalization, 'omics_mean': self.omics_mean.tolist(), 'omics_scale': self.omics_scale.tolist()}
        return {**base, 'profile': self.profile, 'objective': self.loss_kind,
                'risk': '-sum(survival)' if self.loss_kind == 'nll' else 'log risk (monotone equivalent to source exp(logit), avoids overflow)',
                'histology': self.histo_kind, 'prototypes': self.prototypes, 'prototype_fit': self.fit_info,
                'clustering': {'backend': 'sklearn KMeans', 'restarts': self.cluster_restarts,
                               'iterations': self.cluster_iterations, 'patches_per_prototype': self.prototype_patches,
                               'pool': 'training slides only; equal quota per slide; actual collected rows only'},
                'batch_size': self.batch_size, 'batch_tail': 'retained; one optimizer step per minibatch',
                'normalization_fit': 'unique training patients only', 'numerics': self.numerics,
                'optimizer': 'source AdamW with gain/bias decay exclusion; frozen prototypes excluded',
                'schedule': {'epochs': int(self.schedule[0]), 'warmup_epochs': int(self.schedule[1]), 'steps_per_epoch': int(self.schedule[2])},
                'selection': {'metric': getattr(self, 'selection_metric', 'survival_loss'),
                              'policy': 'PULS validation-only selection, not source final epoch; last.pt also retained'},
                'source_caveats': ['full attention preserves asymmetric histology-to-pathway double softmax',
                    'histology OT preserves positive-distance exponential, not a sign-corrected alternative',
                    'cox_source preserves ordered tie handling; choose cox_breslow explicitly for tied-risk-set repair',
                    'all-censored Cox minibatch has zero loss but still steps AdamW and scheduler'],
                'paper_reproduction': 'not_run'}


def mmp_epoch(model, bundle, groups, optimizer, device, config):
    from .engine import load_unit
    records = []
    for start in range(0, len(groups), model.batch_size):
        batch = groups[start:start+model.batch_size]
        tokens, molecular = [], []
        for rows in batch:
            x, _ = load_unit(bundle, rows, device, None, False)
            tokens.append(model.tokenize(x))
            molecular.append(bundle.omics.tensor(rows[0]['global_patient_id']).to(device))
        optimizer.zero_grad(set_to_none=True)
        logits = model.batch_logits(torch.cat(tokens), torch.stack(molecular))
        loss = model.batch_loss(logits, [rows[0]['time'] for rows in batch], [rows[0]['event'] for rows in batch])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.network.parameters(), config['grad_clip'] or math.inf, error_if_nonfinite=True)
        used_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        model.scheduler.step()
        records.append({'loss': float(loss.detach()), 'batch_patients': len(batch),
                        'observed_events': sum(rows[0]['event'] for rows in batch),
                        'gradient_norm': float(norm), 'lr': used_lr})
    return records

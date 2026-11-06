"""Pinned PIBD source recipe: prototype bottlenecks, disentanglement and joint CLUB fitting.

The source PoE scale arithmetic, unmasked zero padding and loss reductions are
intentional. See docs/PIBD.md; this profile is not a claim of paper reproduction.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .discrete_survival import DiscreteSurvival
from .upstream import load_definitions


def author_pibd(bag_size):
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'BAG_SIZE': bag_size}
    files = [
        ('utils/loss_func.py', ['NLLSurvLoss', 'nll_loss']),
        ('models/omics_encoder.py', ['SNN_Block']),
        ('models/club.py', ['CLUBMean', 'MIEstimator']),
        ('models/disentangle_transformer.py', ['Mlp', 'MIAttention', 'DropPath', 'drop_path', 'MITransformerLayer']),
    ]
    for path, names in files:
        module = load_definitions('pibd', path, names, scope)
        scope.update({name: getattr(module, name) for name in names})
    return load_definitions('pibd', 'models/model_PIBD.py',
                            ['KL_between_normals', 'PoE', 'PIB', 'PIBD'], scope, replacements=[
        ('.cuda()', '.to(next(self.parameters()).device)'),
        # Preserve the released CPU random stream, without reseeding every CUDA device.
        ('torch.manual_seed(seed)', 'torch.random.default_generator.manual_seed(seed)'),
    ])


def sampled_padded_bag(x, budget):
    """Author dataset sampling: sorted NumPy sample without replacement, then zero pad."""
    indices = np.sort(np.random.choice(len(x), min(len(x), budget), replace=False))
    selected = x[torch.as_tensor(indices, device=x.device)]
    return F.pad(selected, (0, 0, 0, budget-len(selected)))


def patient_sampling_weights(labels, events):
    # Same occupied-bin probabilities as N/count; absent classes cannot divide by zero.
    keys = labels.long().cpu() * 2 + (1-events.long().cpu())
    _, inverse, counts = keys.unique(return_inverse=True, return_counts=True)
    return len(keys) / counts[inverse].double()


class PIBDSurvival(DiscreteSurvival):
    def __init__(self, dim, omics, hidden=256, bins=4, batch_size=32, num_patches=4096,
                 micro_bag_size=512, sample_num=50, ratio_wsi=.5, ratio_omics=.8,
                 proxy_nll_weight=.1, proxy_kl_weight=.01, proxy_contrast_weight=1.,
                 disentangle_weight=.1, weighted_sampling=True):
        super().__init__(bins=bins, alpha=.5)
        integer_options = {'dim': dim, 'hidden': hidden, 'bins': bins, 'batch_size': batch_size,
                           'num_patches': num_patches, 'micro_bag_size': micro_bag_size, 'sample_num': sample_num}
        if any(type(v) is not int or v <= 0 for v in integer_options.values()):
            raise ValueError('PIBD dimensions, budgets and batch size must be positive integers')
        if bins != 4:
            raise ValueError('pibd_source preserves the author four-bin prototype identity table')
        if hidden % 4 or num_patches < micro_bag_size or num_patches % micro_bag_size:
            raise ValueError('PIBD hidden must be divisible by 4 and num_patches a positive multiple of micro_bag_size')
        if omics is None or not omics.group_indices:
            raise ValueError('PIBD requires aligned molecular values and ordered pathways')
        if type(weighted_sampling) is not bool:
            raise ValueError('weighted_sampling must be boolean')
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 < v <= 1
               for v in (ratio_wsi, ratio_omics)):
            raise ValueError('PIBD retention ratios must be finite in (0, 1]')
        weights = [proxy_nll_weight, proxy_kl_weight, proxy_contrast_weight, disentangle_weight]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights):
            raise ValueError('PIBD loss weights must be finite and nonnegative')
        if int(micro_bag_size*ratio_wsi) < 1 or int(len(omics.groups)*ratio_omics) < 1:
            raise ValueError('PIBD retention ratios select zero instances')
        self.dim, self.hidden = dim, hidden
        self.batch_size, self.num_patches, self.micro_bag_size = batch_size, num_patches, micro_bag_size
        self.sample_num, self.weighted_sampling = sample_num, weighted_sampling
        self.gamma, self.sigma = proxy_contrast_weight, disentangle_weight
        self.normalization = omics.normalization
        self.columns, self.groups, self.group_indices = omics.columns, omics.groups, omics.group_indices
        self.register_buffer('omics_offset', torch.zeros(len(self.columns), dtype=torch.float64))
        self.register_buffer('omics_multiplier', torch.ones(len(self.columns), dtype=torch.float64))
        self.register_buffer('inference_seed', torch.tensor(1, dtype=torch.int64))
        args = SimpleNamespace(omic_sizes=[len(g) for g in self.group_indices], n_classes=bins,
                               encoding_dim=dim, wsi_projection_dim=hidden, omics_format='pathways',
                               num_patches=num_patches, sample_num=sample_num, seed=1,
                               ratio_wsi=ratio_wsi, ratio_omics=ratio_omics,
                               alpha=proxy_nll_weight, beta=proxy_kl_weight)
        author = author_pibd(micro_bag_size)
        self.network = author.PIBD(args)
        self.nll = author.NLLSurvLoss(alpha=.5, reduction='sum')

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('PIBD owns two-stage patch sampling; use model.num_patches, not max_instances')
        clip = config['grad_clip']
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError('grad_clip must be positive finite or null')

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        super().prepare(bundle, seed, device)
        self.inference_seed.fill_(seed)
        if self.normalization == 'train_global_minmax_preserve_zero':
            ids = sorted({r['global_patient_id'] for r in bundle.records('train')})
            values = bundle.omics.values.loc[ids].to_numpy(dtype=np.float64)
            # sklearn MinMaxScaler(-1, 1) on the flattened RNA table, including zeros.
            low, high = values.min(), values.max()
            span = high-low
            if span < 10*np.finfo(np.float64).eps:
                span = 1.
            multiplier = 2./span
            self.omics_multiplier.fill_(multiplier)
            self.omics_offset.fill_(-1.-low*multiplier)
        else:
            mean, scale = bundle.omics.training_statistics(bundle)
            self.omics_multiplier.copy_(1/scale.double())
            self.omics_offset.copy_(-mean.double()/scale.double())

    def normalize_omics(self, values):
        if not self.prepared:
            raise RuntimeError('Fit training survival bins and molecular normalization first')
        if values.ndim != 2 or values.shape[1] != len(self.columns) or not torch.isfinite(values).all():
            raise ValueError('PIBD expects finite [patients, molecular features] values')
        normalized = values.double()*self.omics_multiplier + self.omics_offset
        if self.normalization == 'train_global_minmax_preserve_zero':
            normalized = torch.where(values == 0, 0., normalized)
        result = normalized.to(values.dtype)
        if not torch.isfinite(result).all():
            raise ValueError('PIBD normalized molecular values overflow')
        return result

    def network_output(self, x, omics, labels=None, events=None):
        if x.ndim != 3 or x.shape[2] != self.dim or len(x) != len(omics) or not torch.isfinite(x).all():
            raise ValueError('PIBD expects finite [patients, patches, feature width] bags')
        molecular = self.normalize_omics(omics)
        if self.training:
            if x.shape[1] != self.num_patches or labels is None or events is None:
                raise ValueError('PIBD training needs sampled/padded bags and observed training targets')
            if (labels.shape != (len(x),) or events.shape != (len(x),) or
                    not torch.isfinite(labels).all() or not (labels == labels.long()).all() or
                    (labels < 0).any() or (labels >= self.bins).any() or
                    not torch.isin(events, events.new_tensor([0, 1])).all()):
                raise ValueError('PIBD invalid time-bin labels or event flags')
        else:
            if labels is not None or events is not None:
                raise ValueError('PIBD inference cannot consume outcome labels')
            if min(x.shape[1], self.micro_bag_size)*2*self.bins < self.network.topk_wsi:
                raise ValueError('PIBD evaluation bag is too short for the source prototype-instance top-k')
        arguments = {f'x_omic{i+1}': molecular[:, indices] for i, indices in enumerate(self.group_indices)}
        arguments.update(x_wsi=x, y=labels, c=None if events is None else 1-events, return_attn=False)
        if self.training:
            return self.network(**arguments)
        # Source re-seeds CPU Gaussian draws inside the model. Isolate its whole
        # inference call and set the first micro-bag draw too, making reload exact.
        seed = int(self.inference_seed)
        for module in (self.network.PIB_wsi, self.network.PIB_omics, self.network.PoE):
            module.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return self.network(**arguments)

    def batch_objective(self, outputs, labels, events):
        logits, ib, contrast, mi, estimator = outputs
        survival = self.nll(h=logits, y=labels, t=None, c=1-events)
        total = (survival + self.gamma*contrast + ib + self.sigma*(mi+estimator))/len(labels)
        components = {'loss': total, 'survival_loss': survival/len(labels),
                      'prototype_ib': ib/len(labels), 'prototype_contrast': self.gamma*contrast/len(labels),
                      'mutual_information': self.sigma*mi/len(labels),
                      'estimator_loss': self.sigma*estimator/len(labels)}
        if not all(torch.isfinite(value) for value in components.values()):
            raise FloatingPointError('Nonfinite PIBD objective')
        return total, components

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        if train or self.training:
            raise ValueError('PIBD requires patient minibatch training through pibd_epoch')
        x, _ = load_unit(bundle, rows, device, None, False)
        omics = bundle.omics.tensor(rows[0]['global_patient_id']).to(device).unsqueeze(0)
        return self.result(self.network_output(x.unsqueeze(0), omics)[0])

    def prediction_fields(self, result, row):
        loss = self.nll(h=result['hazard_logits'], y=self.time_label([row['time']]),
                        t=None, c=self.bin_edges.new_tensor([1-row['event']]))
        return {'survival_loss': float(loss)}

    def validation_metrics(self, predictions):
        return {'survival_loss': float(predictions.survival_loss.mean())}

    def protocol(self):
        args = self.network.args
        return {**super().protocol(), 'profile': 'pibd_source',
                'recipe_scope': 'released joint minibatch recipe, PULS train-only bins and validation selection',
                'optimizer': 'single Adam including both CLUB estimators; no detach or alternating estimator steps',
                'batch_size': self.batch_size, 'tail': 'retain, including size-one tail with zero MI estimate',
                'loss': '(sum survival NLL + gamma*mean proxy contrast + alpha*sum proxy NLL + beta*mean KL + sigma*(mean MI + mean estimator fit))/batch size',
                'weights': {'proxy_nll': args.alpha, 'proxy_kl': args.beta, 'contrast': self.gamma, 'disentangle': self.sigma},
                'omics_columns': self.columns, 'pathways': self.groups, 'normalization': self.normalization,
                'normalization_fit': 'unique training patients only', 'omics_multiplier': self.omics_multiplier.tolist(),
                'omics_offset': self.omics_offset.tolist(),
                'num_patches': self.num_patches, 'micro_bag_size': self.micro_bag_size,
                'micro_bags': self.num_patches//self.micro_bag_size, 'sample_num': self.sample_num,
                'retention': {'wsi': args.ratio_wsi, 'omics': args.ratio_omics},
                'train_sampling': 'sorted NumPy sample then unmasked zero padding; independent torch micro-bags may overlap',
                'eval_sampling': 'all patient slides concatenated; source micro-bag sampling with isolated fixed CPU RNG',
                'weighted_sampling': self.weighted_sampling, 'sampling_strata': 'training event/time-bin combinations',
                'source_caveats': ['PoE uses prototype sigma as precision denominator and output noise scale; no unit prior expert',
                                   'top-k across prototype-instance pairs can duplicate an instance',
                                   'padding is visible to the model; no invented attention mask'],
                'inference_seed': int(self.inference_seed), 'paper_reproduction': 'not_run'}


def pibd_epoch(model, bundle, groups, optimizer, device, config):
    from .engine import load_unit
    groups = sorted(groups, key=lambda rows: rows[0]['global_patient_id'])
    labels = model.time_label([rows[0]['time'] for rows in groups])
    events = labels.new_tensor([rows[0]['event'] for rows in groups])
    if model.weighted_sampling:
        order = torch.multinomial(patient_sampling_weights(labels, events), len(groups), replacement=True).tolist()
    else:
        order = torch.randperm(len(groups)).tolist()
    records = []
    for start in range(0, len(order), model.batch_size):
        selected = order[start:start+model.batch_size]
        bags, molecules = [], []
        for i in selected:
            x, _ = load_unit(bundle, groups[i], device, None, False)
            bags.append(sampled_padded_bag(x, model.num_patches))
            molecules.append(bundle.omics.tensor(groups[i][0]['global_patient_id']).to(device))
        y, e = labels[selected], events[selected]
        optimizer.zero_grad(set_to_none=True)
        output = model.network_output(torch.stack(bags), torch.stack(molecules), y, e)
        loss, components = model.batch_objective(output, y, e)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'] or math.inf,
                                             error_if_nonfinite=True)
        optimizer.step()
        records.append({**{key: float(value.detach()) for key, value in components.items()},
                        'batch_patients': len(selected), 'gradient_norm': float(norm)})
    return records

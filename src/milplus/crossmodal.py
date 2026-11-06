"""CMTA translation/alignment and MOTCat transport, including the micro-batch recipe."""
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .discrete_survival import discrete_nll
from .multimodal import MolecularSurvival
from .stochastic import inference_rng
from .upstream import load_crossmodal

CROSSMODAL_METHODS = {'cmta', 'motcat', 'motcat_micro'}


def micro_indices(count, size):
    """Author shuffle + array_split(N // size + 1); size >= 2 avoids empty sub-bags."""
    if not isinstance(count, int) or count < 1 or not isinstance(size, int) or size < 2:
        raise ValueError('MOTCat needs a nonempty bag and integer micro_batch_size >= 2')
    indices = list(range(count))
    random.shuffle(indices)
    return [part.tolist() for part in np.array_split(indices, count // size + 1)]


def alignment_loss(path, translated_path, molecular, translated_molecular):
    # Only the alignment targets are detached; both encoders still receive survival/cross-modal gradients.
    return F.l1_loss(translated_path, path.detach()) + F.l1_loss(translated_molecular, molecular.detach())


class CrossModalSurvival(MolecularSurvival):
    def __init__(self, method, dim, omics, bins=4, alpha=0., accumulation_steps=1, l1_weight=0.,
                 fusion='concat', alignment_weight=None, ot_impl=None, ot_reg=None, ot_tau=None,
                 micro_batch_size=None, inference_seed=2026):
        if method not in CROSSMODAL_METHODS:
            raise ValueError(f'Unknown cross-modal method {method}')
        super().__init__(method, dim, omics, bins, alpha, accumulation_steps, l1_weight)
        if fusion not in {'concat', 'bilinear'}:
            raise ValueError('fusion must be concat or bilinear')
        self.fusion = fusion
        sizes = [len(indices) for indices in self.group_indices]
        if method == 'cmta':
            if any(value is not None for value in (ot_impl, ot_reg, ot_tau, micro_batch_size)):
                raise ValueError('OT and micro-batch settings are only applicable to MOTCat')
            self.alignment_weight = 1. if alignment_weight is None else alignment_weight
            if not math.isfinite(self.alignment_weight) or self.alignment_weight <= 0:
                raise ValueError('CMTA requires positive alignment_weight; zero is an unregistered ablation')
            self.network = load_crossmodal('cmta').CMTA(path_input_dim=dim, omic_sizes=sizes,
                                                       n_classes=bins, fusion=fusion)
        else:
            if alignment_weight is not None:
                raise ValueError('alignment_weight is only applicable to CMTA')
            self.ot_impl = ot_impl or 'pot-uot-l2'
            self.ot_reg = .1 if ot_reg is None else ot_reg
            self.ot_tau = .5 if ot_tau is None else ot_tau
            if self.ot_impl not in {'pot-uot-l2', 'pot-sinkhorn-l2'}:
                raise ValueError('Unsupported MOTCat transport implementation')
            if any(not math.isfinite(value) or value <= 0 for value in (self.ot_reg, self.ot_tau)):
                raise ValueError('OT regularization parameters must be finite and positive')
            if method == 'motcat' and micro_batch_size is not None:
                raise ValueError('Use the motcat_micro profile to enable micro-batch training')
            self.micro_batch_size = (4096 if micro_batch_size is None else micro_batch_size) if method == 'motcat_micro' else None
            if self.micro_batch_size is not None:
                micro_indices(1, self.micro_batch_size)
            if not isinstance(inference_seed, int):
                raise ValueError('inference_seed must be an integer')
            self.inference_seed = inference_seed
            self.network = load_crossmodal('motcat').MOTCAT_Surv(path_input_dim=dim, omic_sizes=sizes,
                n_classes=bins, fusion=fusion, ot_impl=self.ot_impl, ot_reg=self.ot_reg, ot_tau=self.ot_tau)

    def single_forward(self, x, omics):
        molecular = self.normalize_omics(omics)
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('Expected a nonempty finite feature bag')
        arguments = {f'x_omic{i+1}': molecular[indices] for i, indices in enumerate(self.group_indices)}
        output = self.network(x_path=x, **arguments)
        result = self.result(output[0])
        if self.method == 'cmta':
            result['alignment_loss'] = alignment_loss(*output[3:])
        return result

    def _micro_forward(self, x, omics):
        outputs = [self.single_forward(x[part], omics) for part in micro_indices(len(x), self.micro_batch_size)]
        survival = torch.stack([output['survival'] for output in outputs]).mean(0)
        previous = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
        # The mixture has conditional hazards derived from its mean survival, not mean constituent hazards.
        hazards = 1 - survival / previous.clamp_min(torch.finfo(survival.dtype).tiny)
        return {'logits': -survival.sum(1, keepdim=True), 'survival': survival, 'hazards': hazards,
                'micro_hazard_logits': torch.cat([output['hazard_logits'] for output in outputs])}

    def forward(self, x, omics, coords=None):
        if self.method != 'motcat_micro':
            return self.single_forward(x, omics)
        if self.training:
            return self._micro_forward(x, omics)
        with inference_rng(x.device, self.inference_seed):
            return self._micro_forward(x, omics)

    def objective(self, result, time, event):
        if 'micro_hazard_logits' in result:
            logits = result['micro_hazard_logits']
            return discrete_nll(logits, self.time_label([time]).expand(len(logits)),
                                logits.new_full((len(logits),), event), self.alpha)
        loss = super().objective(result, time, event)
        if self.method == 'cmta':
            loss = loss + self.alignment_weight * result['alignment_loss']
        return loss

    def training_losses(self, bundle, rows, device, config):
        if self.method != 'motcat_micro':
            yield from super().training_losses(bundle, rows, device, config)
            return
        from .engine import load_unit
        x, _ = load_unit(bundle, rows, device, config['max_instances'], train=True)
        omics = bundle.omics.tensor(rows[0]['global_patient_id']).to(device)
        chunks = micro_indices(len(x), self.micro_batch_size)
        for indices in chunks:
            result = self.single_forward(x[indices], omics)
            # Backpropagating each term before constructing the next graph preserves the author's mean objective.
            yield self.objective(result, rows[0]['time'], rows[0]['event']) / len(chunks)

    def protocol(self):
        result = {**super().protocol(), 'fusion': self.fusion}
        if self.method == 'cmta':
            result.update(alignment='L1(P.detach(), P_hat) + L1(G.detach(), G_hat)',
                          alignment_weight=self.alignment_weight,
                          spatial_input='author sequence-derived square PPEG grid; not physical coordinates')
        else:
            result.update(transport=self.ot_impl, ot_reg=self.ot_reg, ot_tau=self.ot_tau,
                          unbalanced_regularizer='entropy (author POT 0.9.3 behavior)',
                          transport_gradients='solver uses detached costs and marginals; WSI value path remains differentiable',
                          micro_batch_size=self.micro_batch_size,
                          patient_loss='mean of sub-bag NLLs' if self.micro_batch_size else 'full-bag NLL',
                          prediction='mean sub-bag survival/risk' if self.micro_batch_size else 'full-bag survival/risk')
            if self.micro_batch_size:
                result.update(partition='shuffle; array_split(N // size + 1)',
                              inference_seed=self.inference_seed, inference_rng='isolated from training RNG')
        return result

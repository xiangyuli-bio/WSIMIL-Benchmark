"""Complete frozen-feature LongMIL routes with explicit public/paper identities."""
import hashlib
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import resolve
from .longmil_source import (ALIBI_FILE_SHA256, PaperLongMIL, checked_source,
    grid_positions, load_full_alibi, public_definitions)

PROFILES = {
    'longmil_v1_source_classification_reference': ('v1', 'classification', False),
    'longmil_v1_eval_repaired_classification_reference': ('v1', 'classification', True),
    'longmil_v2_paper_classification_reference': ('paper_v2', 'classification', True),
    'longmil_v2_paper_survival_reference': ('paper_v2', 'survival', True),
}


def input_spec(bundle, profile, deep=True):
    if profile not in PROFILES or bundle.task_type != PROFILES[profile][1]:
        raise ValueError('Select the actual LongMIL classification or patient-survival condition')
    value = bundle.encoder.get('longmil')
    if not isinstance(value, dict) or value.get('coordinate_frame') != 'level0_xy':
        raise ValueError('Declare encoder.longmil with real level0_xy coordinates and physical patch_step_level0')
    step = value.get('patch_step_level0')
    origin = value.get('grid_origin_xy', [0., 0.])
    if type(step) not in (int, float) or not math.isfinite(step) or step <= 0:
        raise ValueError('A measured positive patch_step_level0 is required')
    if not isinstance(origin, (list, tuple)) or len(origin) != 2 or not all(
            type(v) in (int, float) and math.isfinite(v) for v in origin):
        raise ValueError('Declare the actual level0 XY patch-grid origin')
    alibi = None
    if PROFILES[profile][0] == 'v1':
        checksum = value.get('v1_alibi_sha256')
        if (not value.get('v1_alibi_path') or not isinstance(checksum, str)
                or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('Public v1 requires the verified complete original8.1GB ALiBi table')
        alibi = resolve(bundle.root, value['v1_alibi_path'])
        if not alibi.is_file():
            raise ValueError('The original full native ALiBi table is missing')
        if deep:
            load_full_alibi(str(alibi), checksum)
    if deep:
        for row in [r for subset in ['train', 'val', 'test'] for r in bundle.records(subset)]:
            _, coords = bundle.load(row, require_coords=True)
            grid_positions(coords, step, origin)
            if PROFILES[profile][0] == 'v1':
                selected = torch.round((coords - coords.amin(0)) / 896)
                if (selected >= 300).any():
                    raise ValueError('Public v1 coordinates exceed its original300x300 grid; use the explicitly extrapolating paper reference')
    return {'patch_step_level0': step, 'grid_origin_xy': list(origin),
            'v1_alibi_path': None if alibi is None else str(alibi),
            'v1_alibi_sha256': value.get('v1_alibi_sha256') if alibi is not None else None,
            'coordinate_frame': 'level0_xy'}


class LongMIL(nn.Module):
    def __init__(self, bundle, profile):
        super().__init__()
        checked_source()
        self.profile, self.task = profile, bundle.task_type
        self.version, _, self.eval_repaired = PROFILES[profile]
        self.inputs = input_spec(bundle, profile)
        outputs = 1 if self.task == 'survival' else len(bundle.classes)
        if self.version == 'v1':
            table = load_full_alibi(self.inputs['v1_alibi_path'], self.inputs['v1_alibi_sha256'])
            source = public_definitions(lambda: table)
            self.native = source.LongMIL(outputs, input_size=bundle.dim)
            # Source leaves this as an unregistered CUDA tensor. Register its
            # complete actual asset for device movement, but not checkpoint duplication.
            del self.native.alibi
            self.native.register_buffer('alibi', table, persistent=False)
        else:
            self.native = PaperLongMIL(bundle.dim, outputs)
        self.inference_seed = 2026

    def configure_training(self, cfg):
        if cfg.get('max_instances') is not None:
            raise ValueError('LongMIL reference uses complete actual bags; sampling changes the declared protocol')
        if cfg.get('grad_clip') is not None:
            raise ValueError('LongMIL reference explicitly uses no gradient clipping')

    def forward(self, features, coords=None, label=None):
        if features.device.type != 'cuda':
            raise ValueError('Full LongMIL execution requires CUDA and actual original xFormers; CPU equation oracles are separate')
        if coords is None or features.ndim != 2 or len(features) == 0:
            raise ValueError('LongMIL requires real aligned coordinates for every patch')
        positions = grid_positions(coords, self.inputs['patch_step_level0'], self.inputs['grid_origin_xy'])
        if self.version != 'v1':
            return self.native(features, positions)
        selected = torch.round((coords - coords.amin(0)) / 896)
        if (selected >= 300).any():
            raise ValueError('Original v1 positional indices exceed the actual300x300 asset')
        if self.eval_repaired:
            for block in [self.native.layer1, self.native.layer2, self.native.layer3]:
                block.attn.attn_drop = .05 if self.training else 0.
        logits, _, probability, _ = self.native(torch.cat([features, coords], dim=1))
        return {'logits': logits, 'Y_prob': probability}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if config.get('max_instances') is not None:
            raise ValueError('No implicit LongMIL bag truncation or token sampling')
        outputs = []
        for row in rows:
            features, coords = bundle.load(row, require_coords=True)
            outputs.append(self(features.to(device), coords=coords.to(device))['logits'])
        # Independent slide coordinate frames never get concatenated into a
        # fictitious tissue plane. Multiple-slide patients use mean slide risk/logit.
        return {'logits': torch.stack(outputs).mean(0)}

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        from .engine import units
        self.eval()
        output = []
        for rows in units(bundle, subset):
            row = rows[0]
            identity = '|'.join(sorted(r['slide_id'] for r in rows))
            seed = int.from_bytes(hashlib.sha256(
                f'{self.inference_seed}|{identity}|{self.profile}'.encode()).digest()[:8], 'little') % (2**63)
            dev = torch.device(device)
            devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
            # Source-v1 has attention dropout even at eval. Preserve p=.05,
            # declare per-unit inference seeds, and never perturb training RNG.
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if dev.type == 'cuda':
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                result = self.forward_unit(bundle, rows, device, config)
            record = {'slide_id': row['slide_id'] if len(rows) == 1 else 'patient:' + row['global_patient_id'],
                      'patient_id': row['patient_id'], 'global_patient_id': row['global_patient_id'],
                      'subset': subset, 'source_slides': ';'.join(r['slide_id'] for r in rows)}
            if self.task == 'survival':
                record.update(time=row['time'], event=row['event'], risk=float(result['logits'].squeeze()))
            else:
                record.update(target=row['target'])
                record.update({f'p_{i}': float(v) for i, v in enumerate(result['logits'].softmax(-1)[0])})
            output.append(record)
        return pd.DataFrame(output)

    def protocol(self):
        return {'method': 'LongMIL', 'profile': self.profile, 'source_commit':
                '457b1cf6e33a05b3c8359e1fe98114cd8241f772', 'version_condition': self.version,
                'author_v2_code_available': False, 'paper_reference_reconstruction': self.version == 'paper_v2',
                'input_projection': 384, 'local_blocks': 2, 'global_blocks': 1,
                'heads': 4 if self.version == 'v1' else 1, 'ffn_hidden': 1536,
                'local_radius': 'source_cross_head_global_minimum_mask' if self.version == 'v1' else 10,
                'full_v1_asset_shape': [90000, 90000] if self.version == 'v1' else None,
                'full_v1_asset_sha256': self.inputs['v1_alibi_sha256'],
                'paper_pool': 'actual_nonoverlapping2x2_observed_tissue_mean_and_coordinate_centroid',
                'paper_incomplete_pool_cells': 'mean_of_observed_tissue_only_no_fake_padding',
                'global_position': 'source_int8_ALiBi2D' if self.version == 'v1' else 'float_Euclidean_ALiBi2D',
                'attention_backend': 'xformers_memory_efficient_attention_cuda_no_fallback',
                'paper_local_query_chunk': 128, 'paper_global_query_chunk': 128,
                'training_attention_dropout': .05 if self.version == 'v1' else 0.,
                'evaluation_attention_dropout': .05 if self.version == 'v1' and not self.eval_repaired else 0.,
                'source_stochastic_eval_seed': 'fixed2026_plus_slide_identity_and_profile_forked_RNG',
                'paper_reported_lr': 1e-4, 'paper_reported_weight_decay': .01,
                'reference_optimizer': 'Adam', 'reference_max_epochs': 200, 'reference_patience': 20,
                'reference_selection': 'validation_macro_AUROC' if self.task == 'classification' else 'validation_patient_C_index',
                'survival_loss': 'reference_full_TRAIN_patient_cohort_Cox_Breslow' if self.task == 'survival' else None,
                'multislide_patient': 'mean_independent_slide_logits_or_risks_never_merge_coordinate_frames',
                'reference_choices_not_disclosed_by_author': ['optimizer', 'epoch_cap', 'early_stopping',
                    '2x2_reduction_and_holes', 'v2_projection_and_FFN_width_from_public_v1',
                    'global_ALiBi_slope_from_public_v1', 'survival_loss_and_patient_slide_fusion'],
                'real_WSI_experiments': 'not_run', 'paper_score_reproduction': 'not_run', 'formal_admission': False}


def longmil_epoch(model, bundle, groups, optimizer, device, cfg):
    model.train()
    if bundle.task_type == 'classification':
        losses = []
        for rows in groups:
            optimizer.zero_grad(set_to_none=True)
            value = model.forward_unit(bundle, rows, device, cfg, train=True)['logits']
            loss = F.cross_entropy(value, torch.tensor([rows[0]['target']], device=device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite LongMIL training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite LongMIL training gradient')
            optimizer.step()
            losses.append({'loss': float(loss.detach())})
        return losses
    from .engine import rng_state, restore_rng
    from .recipes import cox_breslow_loss
    states, risks = [], []
    with torch.no_grad():
        for rows in groups:
            states.append(rng_state())
            risks.append(model.forward_unit(bundle, rows, device, cfg, train=True)['logits'].flatten())
    final_rng = rng_state()
    prediction = torch.cat(risks).detach().requires_grad_()
    time = torch.tensor([rows[0]['time'] for rows in groups], device=device)
    event = torch.tensor([rows[0]['event'] for rows in groups], device=device)
    loss = cox_breslow_loss(prediction, time, event)
    gradient = torch.autograd.grad(loss, prediction)[0]
    optimizer.zero_grad(set_to_none=True)
    for i, rows in enumerate(groups):
        restore_rng(states[i])
        value = model.forward_unit(bundle, rows, device, cfg, train=True)['logits'].flatten()
        value.backward(gradient[i:i+1])
    restore_rng(final_rng)
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Nonfinite full-cohort LongMIL survival objective or gradient')
    optimizer.step()
    return [{'loss': float(loss.detach())}]

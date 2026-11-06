"""Full SMMILe source recipes, automatic best-initialized two-stage training."""
import copy
import math
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .io import digest_file, digest_object
from .smmile_data import positive_int
from .smmile_kernels import author_network, predict_instances, weak_supervision_loss
from .upstream import load_definitions

PROFILES = {'smmile': 'multiclass', 'smmile_binary': 'binary', 'smmile_multilabel': 'multilabel'}


def source_bitempered(variant):
    branch = 'multi' if variant == 'multilabel' else 'single'
    module = load_definitions('smmile', branch+'/utils/bi_tempered_loss_pytorch.py',
        ['log_t', 'bi_tempered_binary_logistic_loss', 'bi_tempered_logistic_loss'], {'torch': torch})
    def loss(probability, target):
        # The source upper epsilon1e-10 rounds to1 in float32; its power gradient at1 is undefined.
        upper = torch.nextafter(probability.new_tensor(1.), probability.new_tensor(0.))
        probability = probability.clamp(min=probability.new_tensor(1e-10), max=upper)
        return module.bi_tempered_binary_logistic_loss(probability, target, .2, 1., reduction='mean')
    return loss


def bag_target(row, variant, classes, device):
    if variant == 'multilabel':
        target = torch.tensor(row['targets'], device=device, dtype=torch.float32)
        label = torch.nonzero(target, as_tuple=False).flatten()
    elif variant == 'binary':
        target = torch.tensor([row['target']], device=device, dtype=torch.float32)
        label = target
    else:
        label = torch.tensor([row['target']], device=device)
        target = torch.zeros(classes, device=device)
        target[label] = 1
    return label, target


def sample_loader(groups, variant, weighted=True):
    """Use the actual source DataLoader/sampler RNG behavior, in registered row order."""
    groups = sorted(groups, key=lambda rows: rows[0]['slide_id'])
    if any(len(rows) != 1 for rows in groups):
        raise ValueError('SMMILe samples slide bags, not merged patient coordinate frames')
    keys = [tuple(rows[0]['targets']) if variant == 'multilabel' else rows[0]['target'] for rows in groups]
    counts = Counter(keys)
    sampler = WeightedRandomSampler(torch.tensor([len(keys)/counts[k] for k in keys], dtype=torch.double),
                                   len(keys)) if weighted else RandomSampler(groups)
    # SLIC/coordinates are frozen assets. Serial loading makes epoch-boundary resume reproducible.
    return DataLoader(groups, batch_size=None, sampler=sampler, num_workers=0, collate_fn=lambda rows: rows)


class SMMILe(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True

    def __init__(self, bundle, profile='smmile', warmup_epochs=40, size='small', dropout=True,
                 dropout_rate=.25, refs=3, groups=10, drop_times=1, inst_rate=.1,
                 mrf=None, tau=.1, consistency=False, weighted_sampling=True, bag_loss=None):
        super().__init__()
        if profile not in PROFILES:
            raise ValueError('Unknown SMMILe execution profile')
        self.profile, self.variant = profile, PROFILES[profile]
        expected = 'multilabel' if self.variant == 'multilabel' else 'classification'
        if bundle.task_type != expected or bundle.smmile is None:
            raise ValueError('SMMILe requires matching task type and input-bound superpixel assets')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('SMMILe requires individual slide bags; aggregate predictions for patient evaluation')
        self.warmup_epochs = positive_int(warmup_epochs, 'warmup_epochs')
        self.groups, self.drop_times = positive_int(groups, 'groups'), positive_int(drop_times, 'drop_times')
        if isinstance(inst_rate, bool) or not isinstance(inst_rate, (int, float)) or not math.isfinite(inst_rate) or not 0 < inst_rate <= 1:
            raise ValueError('SMMILe candidate rate must lie in (0,1]')
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau < 0:
            raise ValueError('SMMILe MRF weight must be finite nonnegative')
        self.mrf = self.variant != 'binary' if mrf is None else mrf
        if any(type(flag) is not bool for flag in [self.mrf, consistency, weighted_sampling]):
            raise ValueError('SMMILe MRF/consistency/sampling flags must be boolean')
        if consistency and (self.variant == 'multiclass' or not isinstance(bundle.config.get('smmile_negative_bag_semantics'), str)
                            or not bundle.config['smmile_negative_bag_semantics'].strip()):
            raise ValueError('Consistency requires binary/multilabel negative bags with explicit task semantics')
        self.bag_loss = ('bibce' if self.variant == 'multilabel' else 'bce') if bag_loss is None else bag_loss
        if self.bag_loss not in {'bce', 'bibce'}:
            raise ValueError('SMMILe supports the original BCE or bi-tempered augmented-bag recipe')
        self.augmented_loss = source_bitempered(self.variant) if self.bag_loss == 'bibce' else None
        self.classes, self.dim = len(bundle.classes), bundle.dim
        self.class_names = list(bundle.classes)
        self.coordinate_frame = bundle.smmile.manifest['coordinate_frame']
        self.patch_stride = bundle.smmile.manifest['patch_stride']
        self.network = author_network(self.variant, dimension=bundle.dim, classes=self.classes,
            size=size, dropout=dropout, dropout_rate=dropout_rate, refs=refs, repair=True)
        self.inst_rate, self.tau, self.consistency, self.weighted_sampling = inst_rate, tau, consistency, weighted_sampling
        self.negative_bag_semantics = bundle.config.get('smmile_negative_bag_semantics') if consistency else None
        self.validation_inst_rate = inst_rate if self.variant == 'multilabel' else .01
        self.minimum_selection_epoch = warmup_epochs
        # README fixed budgets:40 base +20 refinement. Source resets its stage2 early stop to >100.
        self.stop_after_epoch = warmup_epochs+101
        self.epoch, self.stage_transitions, self.optimizer_steps = -1, 0, 0
        self.warmup_best, self.warmup_loss, self.warmup_best_epoch = None, float('inf'), None
        self.optimizer, self.scheduler, self.loaded_scheduler, self.initial_lr = None, None, None, None

    @property
    def refinement(self):
        return self.epoch >= self.warmup_epochs

    def configure_training(self, cfg):
        if cfg['epochs'] <= self.warmup_epochs:
            raise ValueError('SMMILe budget must include both base and refinement stages')
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('SMMILe preserves the full registered lattice and the source unclipped Adam updates')
        if cfg['selection_metric'] != 'smmile_source_loss':
            raise ValueError('SMMILe source recipe selects on its native validation bag/refinement objective')

    def prepare(self, bundle, seed, device):
        for row in bundle.records('train'):
            if math.prod(bundle.smmile.records[row['slide_id']]['grid_shape']) < 2:
                raise ValueError('SMMILe training BatchNorm requires at least two physical grid cells')

    def make_optimizer(self, lr, weight_decay):
        self.initial_lr = lr
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=.5, patience=10)
        if self.loaded_scheduler is not None:
            self.scheduler.load_state_dict(self.loaded_scheduler)
        return self.optimizer

    def start_epoch(self, epoch):
        self.epoch = epoch
        if epoch == self.warmup_epochs and self.stage_transitions == 0:
            if self.warmup_best is None:
                raise ValueError('SMMILe stage2 requires a validated stage1 best checkpoint')
            self.network.load_state_dict(self.warmup_best, strict=True)
            self.optimizer.state.clear()
            for group in self.optimizer.param_groups:
                group['lr'] = self.initial_lr
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=.5, patience=10)
            self.stage_transitions += 1

    def after_validation(self, metrics):
        value = metrics['smmile_source_loss']
        if not np.isfinite(value):
            raise ValueError('SMMILe source validation loss is undefined')
        self.scheduler.step(value)
        if not self.refinement and value <= self.warmup_loss:
            self.warmup_loss, self.warmup_best_epoch = value, self.epoch
            self.warmup_best = copy.deepcopy(self.network.state_dict())

    def selection_improved(self, objective, best):
        return objective >= best  # Original EarlyStopping saves ties and resets its counter.

    def start_prediction(self, subset):
        if subset == 'val':
            # Equivalent to constructing the source SequentialSampler DataLoader iterator.
            torch.empty((), dtype=torch.int64).random_()

    def raw_forward(self, bundle, row, device, training):
        x, coords, layout, sp, adj = bundle.smmile.load(row)
        label, target = bag_target(row, self.variant, self.classes, device)
        captured = {}
        def cls_hook(module, args, value):
            captured['classification'] = value
        def det_hook(module, args, value):
            captured['detection'] = value[0]
        hooks = [self.network.cls_net.register_forward_hook(cls_hook), self.network.det_net.register_forward_hook(det_hook)]
        try:
            kwargs = {'label': label, 'instance_eval': self.refinement, 'superpixels': True, 'group_numbers': self.groups,
                      'inst_rate': self.inst_rate if training else self.validation_inst_rate}
            if self.variant != 'multilabel':
                kwargs['sp_smooth'] = True
            if training:
                kwargs.update(drop_with_score=True, drop_times=self.drop_times, mrf=self.mrf and self.refinement, tau=self.tau)
                if self.variant != 'multiclass':
                    kwargs['consistency'] = self.consistency
            raw = self.network(layout.scatter(x.to(device)), layout.mask, sp, adj, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        loss = weak_supervision_loss(raw, target, self.variant, augmented_loss=self.augmented_loss)
        cls = captured['classification'].softmax(1) if self.variant == 'multiclass' else captured['classification'].sigmoid()
        attention = captured['detection'].softmax(0)
        result = {'bag_scores': raw[1][0], 'instance_class_scores': layout.restore_rows(cls),
                  'attention': layout.restore_rows(attention), 'instance_attention_product': layout.restore_rows(cls*attention),
                  'refined_scores': [layout.restore_rows(raw[3])] if self.refinement else [],
                  'coords': coords, 'loss': loss, 'source_result': raw}
        if not training:
            result['validation_bag_loss'] = weak_supervision_loss(raw, target, self.variant,
                include_refinement=False, augmented_loss=self.augmented_loss)
        return result

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:
            raise ValueError('SMMILe cannot merge slide coordinate frames')
        row = rows[0]
        if train or row['subset'] == 'val':
            result = self.raw_forward(bundle, row, device, train)
        else:
            x, coords, layout, _sp, _adj = bundle.smmile.load(row)
            result = {**predict_instances(self.network, x.to(device), layout, self.refinement), 'coords': coords}
        raw = result['bag_scores']
        p = torch.cat([1-raw, raw]) if self.variant == 'binary' else raw
        if self.variant == 'multiclass':
            p = p/p.sum()
        return {**result, 'probabilities': p.unsqueeze(0)}

    def prediction_fields(self, result, row):
        raw = result['bag_scores']
        _label, target = bag_target(row, self.variant, self.classes, raw.device)
        record = {f'raw_p_{i}': float(value) for i, value in enumerate(raw)}
        record['smmile_bag_bce'] = float(F.binary_cross_entropy(raw, target))
        if row['subset'] == 'val':
            record['source_validation_bag_loss'] = float(result['validation_bag_loss'])
            record['source_validation_refinement_loss'] = float(result['source_result'][4]['instance_loss'])
        return record

    def validation_metrics(self, predictions):
        result = {'smmile_bag_bce': float(predictions.smmile_bag_bce.mean())}
        if 'source_validation_bag_loss' in predictions:
            total = 0.
            for bag, refinement in zip(predictions.source_validation_bag_loss, predictions.source_validation_refinement_loss):
                total += bag
                if refinement > 0:
                    total += refinement
            result['smmile_source_loss'] = total/len(predictions)
        if self.variant == 'multiclass':
            y = np.eye(self.classes)[predictions.target.to_numpy(dtype=int)]
            p = predictions[[f'raw_p_{i}' for i in range(self.classes)]].to_numpy(dtype=float)
            # Source OVR scores are not normalized. Report slide ranking separately from common metrics.
            result['smmile_raw_slide_macro_auc'] = float(roc_auc_score(predictions.target, p[:, 1]) if self.classes == 2
                                                        else roc_auc_score(y, p, average='macro'))
        return result

    def export_instances(self, result, row, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        name = digest_object({'slide_id': row['slide_id']})+'.h5'
        path = directory/name; temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        with h5py.File(temporary, 'w') as h:
            h.attrs['slide_id'], h.attrs['global_patient_id'] = row['slide_id'], row['global_patient_id']
            h.attrs['profile'], h.attrs['epoch'] = self.profile, self.epoch
            h.attrs['coordinate_frame'], h.attrs['patch_stride'] = self.coordinate_frame, self.patch_stride
            h.attrs['axis_order'] = 'x,y'
            h.attrs['positive_class_index'] = 1 if self.variant == 'binary' else -1
            h.attrs['background_index'] = -1 if self.variant == 'binary' else self.classes
            h.attrs['refinement_class_order'] = 'declared_binary_class_order' if self.variant == 'binary' else 'declared_classes_then_background'
            h['coords'] = result['coords'].cpu().numpy()
            for key in ['bag_scores', 'instance_class_scores', 'attention', 'instance_attention_product']:
                h.create_dataset(key, data=result[key].detach().cpu().numpy(), track_times=False)
            if result['refined_scores']:
                h.create_dataset('refined_scores', data=result['refined_scores'][-1].detach().cpu().numpy(), track_times=False)
        temporary.replace(path)
        return {'slide_id': row['slide_id'], 'file': name, 'sha256': digest_file(path),
                'instances': len(result['coords']), 'query_labels_used_for_predictions': False}

    def get_extra_state(self):
        return {'epoch': self.epoch, 'stage_transitions': self.stage_transitions, 'optimizer_steps': self.optimizer_steps,
                'warmup_best': self.warmup_best, 'warmup_loss': self.warmup_loss, 'warmup_best_epoch': self.warmup_best_epoch,
                'initial_lr': self.initial_lr,
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.loaded_scheduler}

    def set_extra_state(self, state):
        for key in ['epoch', 'stage_transitions', 'optimizer_steps', 'warmup_best', 'warmup_loss', 'warmup_best_epoch', 'initial_lr']:
            setattr(self, key, state[key])
        self.loaded_scheduler = state['scheduler']
        if self.scheduler is not None and self.loaded_scheduler is not None:
            self.scheduler.load_state_dict(self.loaded_scheduler)

    def protocol(self):
        return {'profile': self.profile, 'variant': self.variant, 'warmup_epochs': self.warmup_epochs,
                'class_order': self.class_names, 'coordinate_frame': self.coordinate_frame, 'patch_stride': self.patch_stride,
                'stage': 'refinement' if self.refinement else 'base', 'epoch': self.epoch,
                'stage_transitions': self.stage_transitions, 'optimizer_steps': self.optimizer_steps,
                'stage1_best_epoch': self.warmup_best_epoch,
                'stage1_best_loss': self.warmup_loss if math.isfinite(self.warmup_loss) else None,
                'stage2_initialization': 'strict validation-best stage1; reset Adam state, lr and plateau scheduler',
                'rng': 'one deterministic run stream; initial console-only validation summary omitted',
                'sampler': 'inverse observed train class/label-combination frequency with replacement' if self.weighted_sampling else 'source RandomSampler',
                'bag_loss': self.bag_loss, 'refinement_heads': self.network.n_refs,
                'inst_rate': self.inst_rate, 'validation_inst_rate': self.validation_inst_rate,
                'mrf': self.mrf, 'tau': self.tau, 'consistency': self.consistency,
                'negative_bag_semantics': self.negative_bag_semantics,
                'superpixel_groups': self.groups, 'score_drop_times': self.drop_times,
                'loader': 'serial source DataLoader/samplers; no invented instance annotations',
                'selection': 'source validation bag/SP objective plus pseudo-label refinement CE; ties save/reset',
                'stage2_selection_repair': 'retain the first refinement checkpoint; original epoch0 unconditional EarlyStopping reset can discard its best value',
                'prediction': 'label-free native scores; exclusive-class normalized probabilities reported separately',
                'repairs': 'registered spatial lattice, finite constant-score spans, input-device MRF, empty-neighbor zero energy',
                'bi_tempered_endpoint_repair': 'clip exact1 to adjacent representable float; lower1e-10 preserved; nonsaturated source probabilities unchanged',
                'scope': 'frozen features, physical coords and superpixels; CPU verification is separate from GPU/WSI/paper scores'}


def smmile_epoch(model, bundle, groups, optimizer, device, config):
    losses = []
    for rows in sample_loader(groups, model.variant, model.weighted_sampling):
        optimizer.zero_grad()
        result = model.raw_forward(bundle, rows[0], device, True)
        result['loss'].backward()
        finite = [torch.isfinite(p.grad).all() for p in model.network.parameters() if p.grad is not None]
        if not finite or not torch.stack(finite).all():
            raise FloatingPointError('Nonfinite or absent SMMILe gradients; Adam update refused')
        optimizer.step(); model.optimizer_steps += 1
        extras = result['source_result'][4]
        losses.append({'loss': float(result['loss'].detach()),
            **{key: float(extras[key].detach()) if isinstance(extras.get(key), torch.Tensor) else float(extras.get(key, 0))
               for key in ['instance_loss', 'mrf_loss', 'consist_loss']}})
    return losses

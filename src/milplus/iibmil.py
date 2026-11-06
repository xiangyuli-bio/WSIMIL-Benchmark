"""Complete IIB-MIL reference training, retaining all published core operators.

The author supplies no training epoch. Loss weights, warm-up/update order and
validation selection are explicit reconstruction choices, not recovered code.
"""
import copy
import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .iibmil_data import Bags, Inputs
from .iibmil_source import COMMIT, author_components, build_original, source_config

CONDITIONS = {'iibmil_luad_gm_complete_reference': 'luad_gm',
              'iibmil_tcga_nsclc_complete_reference': 'tcga_nsclc',
              'iibmil_tcga_rcc_complete_reference': 'tcga_rcc'}


def check_inputs(bundle):
    if bundle.task_type != 'classification':
        raise ValueError('Original IIB-MIL defines classification; select an explicitly supported task')
    if bundle.config.get('training_unit', 'slide') != 'slide':
        raise ValueError('IIB-MIL trains individual WSI bags; patient evaluation can aggregate their probabilities')
    return Inputs(bundle)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class IIBMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True
    minimum_selection_epoch = 10

    def __init__(self, bundle, profile, bag_coefficient=1., instance_coefficient=1.):
        super().__init__()
        if profile not in CONDITIONS:
            raise ValueError('Select a complete named IIB-MIL reference configuration')
        for value in [bag_coefficient, instance_coefficient]:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Both IIB-MIL bag and instance supervision must remain active')
        self.inputs = check_inputs(bundle)
        self.profile, self.condition = profile, CONDITIONS[profile]
        self.config = source_config(self.condition)
        self.feature_dim, self.classes = bundle.dim, len(bundle.classes)
        network, self.NestedTensor, partial = author_components()
        model = self.config['MODEL_IIBMIL']
        if self.classes == model['NUM_CLASSES']:
            self.native = build_original(self.condition, feature_dimension=bundle.dim)
        else:
            # Explicit output-class interface port; every original transformer,
            # residual projection, decoder and five queries remain native.
            self.native = network.IIBMIL(num_classes=self.classes, patch_dim=bundle.dim,
                dim=model['HIDDEN_DIM'], depth=model['DEPTH'], num_queries=5)
        confidence = self.inputs.initial_confidence()
        self.register_buffer('confidence', confidence)
        self.register_buffer('initial_confidence', confidence.clone())
        self.register_buffer('confidence_ema', torch.tensor(.99, dtype=torch.float64))
        self.partial = partial(self.confidence, {key:i for i,key in enumerate(self.inputs.bank_keys)}, conf_ema_m=.99)
        self.bag_coefficient, self.instance_coefficient = float(bag_coefficient), float(instance_coefficient)
        self.current_epoch, self.updates, self.confidence_updates = 0, 0, 0
        self.scheduler, self.pending_scheduler = None, None
        self.draw_history, self.reference_history = [], []
        self.training_cfg = None

    def _apply(self, function):
        result = super()._apply(function)
        if hasattr(self, 'partial'):
            # Original PartialLoss stores plain tensor attributes, not buffers.
            # Rebind both aliases after actual CPU/CUDA buffer movement.
            self.partial.confidence = self.confidence
            self.partial.init_confidence = self.confidence
        return result

    def configure_training(self, cfg):
        if type(cfg['epochs']) is not int or not 11 <= cfg['epochs'] <= 300:
            raise ValueError('Complete IIB-MIL includes its10epoch confidence warm-up and subsequent disambiguation; reference supports11..300epochs')
        if cfg.get('max_instances') is not None:
            raise ValueError('IIB-MIL keeps the original fixed500-row sampler; no second hidden truncation')
        if cfg.get('grad_clip') is None or cfg['grad_clip'] <= 0:
            raise ValueError('IIB-MIL reference retains active source gradient clipping (native0.01)')
        self.training_cfg = copy.deepcopy(cfg)

    def make_optimizer(self, lr, weight_decay):
        # Exact published test.py optimizer/scheduler definitions, also encoded
        # in the three YAML recipes. The inferred parameter groups are unused
        # by its actual AdamW branch and are not silently activated here.
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.1)
        if self.pending_scheduler is not None:
            self.scheduler.load_state_dict(self.pending_scheduler)
            self.pending_scheduler = None
        return optimizer

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        self.partial.set_conf_ema_m(epoch)  # original200 denominator, even for YAML300
        self.confidence_ema.fill_(self.partial.conf_ema_m)
        self.partial.confidence = self.confidence
        self.partial.init_confidence = self.confidence

    def finish_epoch(self, bundle, device, cfg):
        self.scheduler.step()

    def get_extra_state(self):
        return {'schema_version':1, 'input_identity':self.inputs.identity, 'profile':self.profile,
                'current_epoch':self.current_epoch, 'updates':self.updates,
                'confidence_updates':self.confidence_updates,
                'conf_ema_m':self.partial.conf_ema_m,
                'scheduler':copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else self.pending_scheduler,
                'draw_history':copy.deepcopy(self.draw_history),
                'reference_history':copy.deepcopy(self.reference_history)}

    def set_extra_state(self, state):
        if (state.get('schema_version') != 1 or state.get('input_identity') != self.inputs.identity
                or state.get('profile') != self.profile):
            raise ValueError('IIB-MIL confidence bank checkpoint belongs to different actual TRAIN patch identities or condition')
        self.current_epoch, self.updates = state['current_epoch'], state['updates']
        self.confidence_updates = state['confidence_updates']
        self.partial.conf_ema_m = state['conf_ema_m']
        self.partial.confidence = self.confidence
        self.partial.init_confidence = self.confidence
        self.draw_history, self.reference_history = copy.deepcopy(state['draw_history']), copy.deepcopy(state['reference_history'])
        if self.scheduler is None:
            self.pending_scheduler = copy.deepcopy(state['scheduler'])
        elif state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])

    def _forward_batch(self, value, mask):
        if value.ndim != 3 or value.shape[1:] != (500, self.feature_dim) or mask.shape != value.shape[:2]:
            raise ValueError('IIB-MIL requires complete native500-row bags and aligned valid-instance masks')
        if mask.dtype != torch.bool or not mask.any(1).all() or not torch.isfinite(value).all():
            raise ValueError('IIB-MIL has empty, invalid or nonfinite actual bags')
        return self.native(self.NestedTensor(value, mask))

    def forward(self, features, coords=None, label=None):
        from .iibmil_data import select_rows
        if self.training:
            raise ValueError('Use the complete IIB-MIL batched training recipe and TRAIN confidence bank')
        value, mask, _ = select_rows(features.detach().cpu().numpy(), False)
        outputs = self._forward_batch(value.unsqueeze(0).to(features.device), mask.unsqueeze(0).to(features.device))
        return {'logits':outputs[3], 'patch_logits':outputs[0]}

    def train_batch(self, batch, optimizer, device):
        value, mask = batch['features'].to(device), batch['mask'].to(device)
        target = batch['target'].to(device)
        indices = batch['bank_indices'].to(device)[mask]
        if indices.min() < 0 or indices.max() >= len(self.confidence) or len(torch.unique(indices)) != len(indices):
            raise ValueError('Only unique actual TRAIN instances may update IIB-MIL confidence; padded IDs must be excluded')
        self.partial.confidence = self.confidence
        outputs = self._forward_batch(value, mask)
        patch_logits, features, prototype_scores, bag_logits = outputs
        bag_loss = F.cross_entropy(bag_logits, target)
        instance_loss = self.partial(patch_logits[mask], indices)
        loss = self.bag_coefficient*bag_loss + self.instance_coefficient*instance_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite IIB-MIL integrated supervision loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.training_cfg['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        self.updates += 1
        # Explicit reference order: loss uses the existing confidence bank;
        # detached pre-step embeddings update prototypes, then the original
        # pre-update prototype scores update confidence after10warm-up epochs.
        # No-gradient EMA prevents a graph growing through an epoch's buffers.
        with torch.no_grad():
            labels = target.detach().cpu().numpy()
            for c in range(self.classes):
                selected = np.argwhere(labels == c)
                if len(selected):
                    self.native.update_prototypes(c, selected, patch_logits.detach(), features.detach(), mask,
                                                  proto_m=self.config['TRAIN']['EMA'])
            if self.current_epoch >= 10:
                scores = prototype_scores.reshape(value.shape[0],500,self.classes)[mask]
                self.partial.confidence_update(scores, indices, torch.ones_like(scores))
                self.confidence_updates += len(indices)
        if not torch.isfinite(self.native.prototypes).all() or not torch.isfinite(self.confidence).all():
            raise FloatingPointError('IIB-MIL native prototype or confidence update became nonfinite')
        return {'loss':float(loss.detach()), 'bag_ce':float(bag_loss.detach()),
                'instance_partial_ce':float(instance_loss.detach()), 'valid_instances':float(mask.sum()),
                'confidence_update_active':float(self.current_epoch >= 10)}

    def epoch(self, bundle, optimizer, device):
        self.train()
        dataset = Bags(self.inputs,'train')
        batch_size = self.config['TRAIN']['BATCH_SIZE']
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False)
        terms = []
        sampled = hashlib.sha256()
        for batch in loader:
            for sid,rows,mask in zip(batch['slide_id'],batch['original_rows'],batch['mask']):
                self.draw_history.append({'epoch':self.current_epoch,'slide_id':sid,
                                          'original_rows':rows.clone(),'valid':mask.clone()})
                sampled.update(sid.encode());sampled.update(rows.numpy().tobytes());sampled.update(mask.numpy().tobytes())
            terms.append(self.train_batch(batch,optimizer,device))
        self.reference_history.append({'epoch':self.current_epoch,'updates':self.updates,
            'confidence_updates':self.confidence_updates,'confidence_ema':self.partial.conf_ema_m,
            'confidence_sha256':tensor_hash(self.confidence),'prototype_sha256':tensor_hash(self.native.prototypes),
            'ordered_actual_patch_draws_sha256':sampled.hexdigest(),
            'learning_rate':optimizer.param_groups[0]['lr']})
        return terms

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        self.eval()
        before_confidence, before_prototypes = tensor_hash(self.confidence), tensor_hash(self.native.prototypes)
        loader = DataLoader(Bags(self.inputs,subset),batch_size=self.config['TRAIN']['BATCH_SIZE'],
                            shuffle=False,num_workers=0,drop_last=False)
        records = []
        for batch in loader:
            outputs = self._forward_batch(batch['features'].to(device),batch['mask'].to(device))
            probability = outputs[3].softmax(-1)
            for i,sid in enumerate(batch['slide_id']):
                row = self.inputs.rows[sid]
                record = {'slide_id':sid,'patient_id':row['patient_id'],'global_patient_id':row['global_patient_id'],
                          'subset':subset,'source_slides':sid,'target':int(row['target'])}
                record.update({f'p_{c}':float(v) for c,v in enumerate(probability[i])})
                records.append(record)
        if tensor_hash(self.confidence) != before_confidence or tensor_hash(self.native.prototypes) != before_prototypes:
            raise ValueError('Validation/test must never update IIB-MIL prototypes or confidence')
        return pd.DataFrame(records)

    def protocol(self):
        model, training = self.config['MODEL_IIBMIL'], self.config['TRAIN']
        return {'method':'IIB-MIL','profile':self.profile,'source_commit':COMMIT,
            'source_training_epoch_available':False,'reference_reconstruction':True,
            'original_config':self.config,'actual_feature_dimension':self.feature_dim,
            'native_feature_dimension':1280,'FM_input_port':self.feature_dim != 1280,
            'original_class_count':model['NUM_CLASSES'],'actual_classes':self.classes,
            'output_class_port':self.classes != model['NUM_CLASSES'],
            'native_hidden':model['HIDDEN_DIM'],'native_depth':model['DEPTH'],
            'native_encoder_heads':2,'native_encoder_FFN':2048,'native_decoder_heads':4,
            'native_decoder_depth':2,'native_queries':5,'native_tokens_per_slide':500,
            'source_defined_unused_parameters_retained':True,
            'source_padding_attention_mask':'originally_ignored_not_silently_repaired',
            'padding_instance_loss_and_confidence_policy':'valid_original_instances_only_explicit_reference',
            'original500_row_policy':'50percent_train_permutation_then_prefix_else_prefix_zero_pad_and_repeat_first_ID',
            'patch_identity_provenance':dict(self.inputs.provenance),'actual_TRAIN_bank_rows':len(self.confidence),
            'TRAIN_bank_identity_sha256':self.inputs.identity,'bank_initialization_epsilon':1e-6,
            'native_top_fraction':.1,'prototype_momentum':training['EMA'],
            'prototype_momentum_class_scaling':'1-(1-m)*batch_size/class_matched_bag_count',
            'reference_loss_weights':{'bag_CE_raw_logits':self.bag_coefficient,'instance_PartialLoss':self.instance_coefficient},
            'confidence_warmup_epochs':10,'confidence_EMA':'literal_source0.95+(0.8-0.95)*epoch/200_even_at_YAML300',
            'reference_update_order':'existing_confidence_loss_backward_AdamW_detached_prototypeEMA_then_lagged_prototype_score_confidenceEMA',
            'native_epochs':300,'native_batch_size':training['BATCH_SIZE'],'native_lr':training['LR'],
            'native_weight_decay':training['WEIGHT_DECAY'],'native_clip_norm':.01,
            'optimizer':'source_AdamW_all_original_parameters','scheduler':'source_StepLR100_gamma0.1',
            'reference_loader_workers':0,'reference_validation':'macro_AUROC_after_warmup_no_early_stopping',
            'actual_requested_training_config':copy.deepcopy(self.training_cfg),
            'reference_choices_not_recovered_from_author_trainer':['loss_coefficients_and_raw_logit_CE',
                'confidence_warmup_interpretation','per_batch_update_order','valid_instance_loss_and_bank_update_mask',
                'no_early_stopping_and_post_warmup_validation_selection','single_cached_FM_view_and_zero_worker_loader'],
            'current_epoch':self.current_epoch,'actual_optimizer_updates':self.updates,
            'actual_confidence_instance_updates':self.confidence_updates,
            'formal_admission':False,'real_WSI_validation':'not_run','paper_score_reproduction':'not_run'}

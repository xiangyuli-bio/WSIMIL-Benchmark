"""Complete named INS frozen-feature recipe; source/paper repairs are explicit."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .ins_data import BagDataset, InstanceDataset, TrainPatchStore
from .ins_network import COMMIT, INSFrozenReference
from .io import digest_object

PROFILE = 'ins_frozen_source_repaired'


class INSFrozenMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile=PROFILE, batch_size=64, queue_size=8192, projection=128,
                 warmup_epochs=20, momentum=.999, prototype_momentum=.9,
                 confidence_start=.95, confidence_end=.8, contrastive_weight=.5,
                 bag_weight=1., instance_weight=1., sgd_momentum=.9, cosine=False,
                 lr_decay_epochs=(700, 800, 900), lr_decay_rate=.1, sampler_seed=0,
                 eval_chunk=128, cache_bytes=128*1024**2, max_open_files=16,
                 max_dense_bytes=32*1024**3):
        super().__init__()
        if profile != PROFILE or bundle.task_type != 'classification' or len(bundle.classes) != 2:
            raise ValueError('INS source-repaired frozen route requires its actual binary MIL setting')
        if bundle.raw_images is not None or bundle.raw_regions is not None or bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('This INS input condition requires complete frozen feature slide bags, not RGB or patient concatenation')
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError('Current INS controller is the verified single-rank source condition; do not duplicate the full dataset under external DDP')
        if any(type(v) is not int or v < 1 for v in [batch_size, eval_chunk, max_dense_bytes]):
            raise ValueError('INS batch, eval chunk and memory budgets must be positive integers')
        if type(sampler_seed) is not int or type(cosine) is not bool or not 0 <= sgd_momentum < 1 or not 0 < lr_decay_rate < 1:
            raise ValueError('INS requires explicit sampler/cosine and valid native SGD/schedule coefficients')
        if not isinstance(lr_decay_epochs, (list, tuple)) or any(type(v) is not int or v < 0 for v in lr_decay_epochs):
            raise ValueError('INS learning-rate milestones must be nonnegative integer epochs')
        self.bundle, self.profile = bundle, profile
        self.store = TrainPatchStore(bundle, cache_bytes, max_open_files)
        self.core = INSFrozenReference(bundle.dim, self.store.labels, queue_size=queue_size, projection=projection,
            warmup_epochs=warmup_epochs, momentum=momentum, prototype_momentum=prototype_momentum,
            confidence_start=confidence_start, confidence_end=confidence_end,
            instance_weight=instance_weight, contrastive_weight=contrastive_weight, bag_weight=bag_weight)
        if queue_size % batch_size:
            raise ValueError('Original INS queue requires full minibatch divisibility')
        if int((self.store.labels == 1).sum()) < batch_size:
            raise ValueError('INS warmup needs enough actual positive-bag TRAIN instances to fill its minibatch')
        self.settings = {'batch_size': batch_size, 'queue_size': queue_size, 'projection': projection,
            'warmup_epochs': warmup_epochs, 'momentum': momentum, 'prototype_momentum': prototype_momentum,
            'confidence_start': confidence_start, 'confidence_end': confidence_end,
            'contrastive_weight': contrastive_weight, 'bag_weight': bag_weight, 'instance_weight': instance_weight,
            'sgd_momentum': sgd_momentum, 'cosine': cosine, 'lr_decay_epochs': list(lr_decay_epochs),
            'lr_decay_rate': lr_decay_rate, 'sampler_seed': sampler_seed, 'eval_chunk': eval_chunk,
            'cache_bytes': cache_bytes, 'max_open_files': max_open_files, 'max_dense_bytes': max_dense_bytes}
        self.contract = digest_object({'source': COMMIT, 'patch_inventory': self.store.contract, 'settings': self.settings,
            'input_condition': 'actual source full feature pseudo-image, author native ResNet18 from scratch',
            'source_repairs': 'Cway mean-feature bag head; bag-only candidate/negative identities; actual valid TRAIN queue; complete resume'})
        self.current_epoch = -1
        self.completed_epochs = 0
        self.optimizer = None
        self.base_lr = self.total_epochs = None

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('INS source route uses complete bags and no gradient clipping')
        if cfg['epochs'] <= self.core.warmup_epochs:
            raise ValueError('INS must include both contrastive warmup and prototype/confidence refinement')
        if cfg['selection_metric'] != 'macro_auc':
            raise ValueError('INS controller uses explicit validation macro AUROC selection')
        self.total_epochs = cfg['epochs']
        self.minimum_selection_epoch = self.core.warmup_epochs
        self.stop_after_epoch = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        self.base_lr = lr
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,
            momentum=self.settings['sgd_momentum'], weight_decay=weight_decay)
        return self.optimizer

    def reserve(self, count, training=True):
        parameters = sum(p.numel() for p in self.parameters())
        # Explicit conservative native-CNN bound before repeated pseudo-images
        # or whole-bag activation graphs are allocated. No silent train chunks.
        active = count if training else min(count, self.settings['eval_chunk'])
        estimated = 20*parameters+4096*active*self.bundle.dim
        if estimated > self.settings['max_dense_bytes']:
            raise ValueError('INS full native feature-CNN exceeds max_dense_bytes before allocation')

    def prepare(self, bundle, seed, device):
        self.reserve(self.settings['batch_size'])
        self.reserve(int(max(self.store.counts)))

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        if self.settings['cosine']:
            eta = self.base_lr*self.settings['lr_decay_rate']**3
            lr = eta+(self.base_lr-eta)*(1+np.cos(np.pi*epoch/self.total_epochs))/2
        else:
            steps = sum(epoch > milestone for milestone in self.settings['lr_decay_epochs'])
            lr = self.base_lr*self.settings['lr_decay_rate']**steps
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def forward(self, x, coords=None, label=None):
        self.reserve(len(x), training=self.training)
        if self.training:
            logits = self.core(x)
        else:
            total = torch.zeros(512, dtype=torch.float64, device=x.device)
            for chunk in x.split(self.settings['eval_chunk']):
                encoded = self.core.network.encoder_q.encoder(self.core.image(chunk))
                total += encoded.double().sum(0)
            mean = (total/len(x)).float()[None]
            logits = self.core.network.bag_classifier(mean)
        return {'logits': logits}

    def epoch(self, bundle, optimizer, device):
        training = InstanceDataset(self.store, positive_only=self.current_epoch < self.core.warmup_epochs)
        sampler = DistributedSampler(training, num_replicas=1, rank=0, shuffle=True, seed=self.settings['sampler_seed'])
        sampler.set_epoch(self.current_epoch)
        loader = DataLoader(training, batch_size=self.settings['batch_size'], sampler=sampler,
                            drop_last=True, num_workers=0)
        losses = []
        for x, index in loader:
            x = x.to(device)
            result = self.core.instance_objective(x, x, index.to(device), self.current_epoch)
            optimizer.zero_grad()
            result['loss'].backward()
            optimizer.step()
            losses.append({'loss': float(result['loss'].detach()), 'instance_loss': float(result['classification'].detach()),
                'contrastive_loss': float(result['contrastive'].detach()), 'prior_valid_keys': result['prior_valid_keys'],
                'instance_update': 1., 'prototype_refinement': float(self.current_epoch >= self.core.warmup_epochs)})
        if not losses:
            raise ValueError('INS no complete actual TRAIN instance minibatch; no duplicate padding')
        # Source phase order: the complete instance pass precedes all TRAIN bags.
        bags = DataLoader(BagDataset(self.store), batch_size=1, shuffle=False, num_workers=0)
        for x, y, _slide in bags:
            x, y = x[0].to(device), y.to(device)
            logits = self(x)['logits']
            loss = F.cross_entropy(logits, y)*self.core.bag_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.core.bag_steps += 1
            losses.append({'loss': float(loss.detach()), 'bag_loss': float(loss.detach()), 'bag_update': 1.})
        self.core.finish_epoch(self.current_epoch, self.total_epochs)
        self.completed_epochs = self.current_epoch+1
        return losses

    def get_extra_state(self):
        return {'contract': self.contract, 'epoch': self.current_epoch, 'completed_epochs': self.completed_epochs}

    def set_extra_state(self, state):
        if state['contract'] != self.contract:
            raise ValueError('INS source, complete feature index, weak labels or recipe changed')
        self.current_epoch, self.completed_epochs = state['epoch'], state['completed_epochs']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        valid = state_dict.get('core.queue_valid')
        indices = state_dict.get('core.queue_indices')
        if valid is None or indices is None or valid.shape != indices.shape or valid.dtype != torch.bool:
            raise ValueError('INS complete queue identity/validity state is missing')
        if (indices[valid] < 0).any() or (indices[valid] >= len(self.store)).any() or (indices[~valid] != -1).any():
            raise ValueError('INS queue contains non-TRAIN instance identities')
        labels, confidence = state_dict.get('core.training_bag_labels'), state_dict.get('core.confidence')
        if labels is None or not torch.equal(labels.cpu(), self.store.labels):
            raise ValueError('INS checkpoint weak bag labels differ from the actual TRAIN instance map')
        if (confidence is None or confidence.shape != (len(self.store), 2) or not torch.isfinite(confidence).all() or
                (confidence < 0).any() or (confidence > 1).any() or
                not torch.allclose(confidence.sum(1), torch.ones(len(self.store), device=confidence.device), rtol=0, atol=2e-6)):
            raise ValueError('INS checkpoint contains invalid instance confidence state')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'settings': self.settings,
            'feature_dimension': self.bundle.dim, 'patch_inventory': self.store.contract,
            'input_condition': 'Frozen FM features stay frozen; actual author feature route reshapes full D as32x(D/32), repeats3channels and trains native ResNet18. Not actual RGB tissue or a generic attention head.',
            'initialization': 'Actual author ResNet18 from scratch, matching paper no-pretraining condition; source model.py hardcoded pretrained=True explicitly changed. No implicit download.',
            'objectives': 'Source partial-label CE +.5 canonical-source SupCon/MoCo, followed by full mean-feature512to2 bag CE; native weights configurable and recorded. Original Eq5 printed non-family-only denominator differs from source standard all-pool denominator; this profile explicitly uses source formula.',
            'weak_labels': 'Negative/reference class is classes[0]; only negative BAGS provide known negative patches. Positive-bag patches remain ambiguous. No true patch labels or positive-patch-ratio filtering.',
            'phases': 'Native20epoch positive-bag-only MoCo warmup, then all TRAIN patches with prototype pseudo-label/confidence EMA and supervised contrast. Every epoch ends with all training bag updates.',
            'prototype_update_timing': 'Literal source forward updates prototypes during warmup too; proto scores use the previous prototype snapshot. Only confidence updating and supervised-family masks wait for the warmup boundary.',
            'queues': 'Original8192key momentum/prototype update kernels. Ignore unfilled random slots; retain actual TRAIN patch IDs, valid mask, current/query/key/source order. Key params EMA. No VAL/TEST state updates.',
            'sampling': 'Actual world1 DistributedSampler(seed0+epoch), source full64instance batches/drop_last, DataLoader workers0; full ordered one-bag pass. No instance duplication to fill tails.',
            'bag_head_repair': 'Original bag path returns softmax over512latent channels and never calls declared512to2 head. Explicitly call that original head, train CE on logits. Its input is mean backbone features, the source condition; paper Eq10 q notation can instead denote128projected embeddings and is not silently conflated.',
            'unused_source_heads': 'Two teacher heads and auxiliary mlp remain in actual source state but are not executed by this routed model. No false teacher-training claim.',
            'evaluation': 'VAL-only first strict best macroAUROC after warmup; default complete150epochs, no early stop. TEST explicit. Inference bounded chunks, fixed eval BN and float64 accumulation of complete512features before class head; no patch removal.',
            'evaluation_unit': self.bundle.config.get('evaluation_unit', 'slide'),
            'evaluation_unit_condition': 'Source comparison is slide-level; a declared patient evaluation unit is an explicit outer evaluation extension using the common patient-fusion protocol.',
            'completed_epochs': self.completed_epochs, 'instance_updates': self.core.steps, 'bag_updates': self.core.bag_steps,
            'valid_queue_keys': int(self.core.queue_valid.sum()), 'contract': self.contract,
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

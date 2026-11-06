"""Original end-to-end center-embedding MIL, with explicit source typo repair."""
import copy
import math
import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .center_embeddings_loaders import CandidateTiles, SelectedBags
from .center_embeddings_source import COMMIT, PROFILES, build_model, definitions, staircase
from .io import digest_object


class CenterEmbeddingMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='center_embeddings_bag_classifier_repaired', train_batch_size=4,
                 center_lr=.0001):
        super().__init__()
        if bundle.center_embeddings is None or profile not in PROFILES:
            raise ValueError('Center Embeddings requires its original measured RGB assets and explicit source route')
        if type(train_batch_size) is not int or train_batch_size < 1 or not 0 < center_lr < 1:
            raise ValueError('Center Embeddings bag batch and center optimizer learning rate are invalid')
        self.bundle, self.assets, self.profile = bundle, bundle.center_embeddings, profile
        self.network, self.center_loss = build_model(self.assets.assets)
        self.train_batch_size, self.center_lr = train_batch_size, center_lr
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'bundle': bundle.fingerprint,
            'assets': self.assets.contract, 'batch_size': train_batch_size, 'center_lr': center_lr})
        self.recipe = None
        self.completed_epochs, self.optimizer_updates, self.epoch_index = 0, 0, -1
        self.train_selections = []

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'center_embeddings_val_balanced_accuracy'):
            raise ValueError('Center Embeddings requires exhaustive source candidate scans, no generic cap/clipping and VAL balanced ACC')
        self.recipe = {k: cfg[k] for k in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}

    def make_optimizers(self, lr, weight_decay):
        features = list(self.network.features.parameters())
        new = [p for name, p in self.network.named_parameters() if not name.startswith('features.')]
        self._optimizers = [torch.optim.Adam([{'params': features, 'lr': lr}, {'params': new, 'lr': .01}],
                weight_decay=weight_decay),
            torch.optim.Adam(self.center_loss.parameters(), lr=self.center_lr)]
        return self._optimizers

    def validate_optimizer_state(self):
        groups = self._optimizers[0].param_groups
        center = self._optimizers[1].param_groups
        if (len(groups) != 2 or len(center) != 1
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.features.parameters()}
                or {id(p) for p in groups[1]['params']} != {id(p) for n, p in self.network.named_parameters() if not n.startswith('features.')}
                or {id(p) for p in center[0]['params']} != {id(p) for p in self.center_loss.parameters()}
                or any(g['weight_decay'] != self.recipe['weight_decay'] for g in groups)
                or center[0]['weight_decay'] != 0 or center[0]['lr'] != self.center_lr
                or any(g['betas'] != (.9, .999) or g['eps'] != 1e-8 for g in [*groups, *center])):
            raise ValueError('Center Embeddings lost a genuine source optimizer group or hyperparameter')
        expected = [self.recipe['lr'], .01]
        for step in [10, 15, 17]:
            if self.completed_epochs >= step:
                expected = [base*.1**([10, 15, 17].index(step)+1) for base in [.001, .01]]
        if [g['lr'] for g in groups] != expected:
            raise ValueError('Center Embeddings source staircase phase changed')
        if self.optimizer_updates:
            for optimizer in self._optimizers:
                expected_params = {p for g in optimizer.param_groups for p in g['params']}
                if set(optimizer.state) != expected_params or any(int(s['step']) != self.optimizer_updates for s in optimizer.state.values()):
                    raise ValueError('Center Embeddings full backbone/FPN/heads or center Adam moments lost actual updates')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('Center Embeddings resumes at complete original epoch boundaries only')
        self.epoch_index = epoch

    @torch.no_grad()
    def select(self, subset, device):
        self.network.eval(); self.network.mode = 0
        candidates = CandidateTiles(self.assets, self.bundle.records(subset), training=subset == 'train')
        batch = self.train_batch_size if subset == 'train' else 1
        loader = DataLoader(candidates, batch_size=batch, num_workers=0, shuffle=False, pin_memory=True)
        probs = torch.empty(len(candidates)); cursor = 0
        try:
            for images, _ in loader:
                logits, _ = self.network(images.to(device))
                values = self.network.pooling.probabilities(logits).detach()[:, 1].cpu()
                probs[cursor:cursor+len(values)] = values; cursor += len(values)
        finally:
            candidates.close()
        if cursor != len(candidates) or not torch.isfinite(probs).all():
            raise ValueError('Center Embeddings must score every original candidate patch exactly once')
        selected = definitions()[3].group_argtopk(candidates.groups.copy(), probs.numpy(), 32)
        bags = SelectedBags(candidates, selected)
        if any(len(ids) != min(32, candidates.counts[i]) for i, ids in bags.selected.items()):
            raise ValueError('Center Embeddings source top32 candidate selection changed')
        return bags

    def epoch(self, bundle, optimizers, device):
        if bundle.fingerprint != self.bundle.fingerprint or len(optimizers) != 2 or any(a is not b for a, b in zip(optimizers, self._optimizers)):
            raise ValueError('Center Embeddings original inputs or dual optimizer identity changed')
        bags = self.select('train', device)
        # Source shuffledata uses random.sample, not random.shuffle.
        bags.order = random.sample(bags.order, len(bags.order))
        self.center_loss.train(); self.network.train(); self.network.mode = 1
        staircase(optimizers[0].param_groups, self.epoch_index)
        loader = DataLoader(bags, batch_size=self.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        losses = []
        try:
            for images, labels, all_labels in loader:
                images, labels = images.to(device), labels.to(device)
                all_labels = all_labels.reshape(-1).to(device).long()
                for optimizer in optimizers:
                    optimizer.zero_grad()
                instance_logits, _, bag_embed, bag_logits = self.network(images)
                instance_loss = self.network.pooling.loss(instance_logits, all_labels)
                bag_loss = self.network.pooling.loss(bag_logits, labels)
                # Public source returns bag_embed but references undefined
                # bag_feat. This named repair binds that exact normalized
                # third output; no objective, detach or weight is removed.
                center_loss = self.center_loss(bag_embed, labels)
                loss = instance_loss+center_loss+bag_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Center Embeddings complete three-loss source objective is nonfinite')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters()):
                    raise FloatingPointError('Center Embeddings lost an original network/center gradient')
                optimizers[0].step()
                for parameter in self.center_loss.parameters():
                    parameter.grad.data *= 1./1.  # Source center-loss gradient compensation.
                optimizers[1].step(); self.optimizer_updates += 1
                losses.append({'loss': float(loss.detach()), 'instance_ce': float(instance_loss.detach()),
                    'bag_ce': float(bag_loss.detach()), 'center_loss': float(center_loss.detach()),
                    'instance_accuracy': float((instance_logits.argmax(1) == all_labels).float().mean()),
                    'bag_accuracy': float((bag_logits.argmax(1) == labels).float().mean())})
        finally:
            bags.candidates.close()
        self.completed_epochs = self.epoch_index+1
        rows = bags.candidates.rows
        self.train_selections.append({'selection': {rows[i]['slide_id']: ids.copy() for i, ids in bags.selected.items()},
            'bag_order': [rows[i]['slide_id'] for i in bags.order],
            'scanned': {r['slide_id']: bags.candidates.counts[i] for i, r in enumerate(rows)}})
        self.validate_optimizer_state()
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle.fingerprint != self.bundle.fingerprint or subset not in {'val', 'test'}:
            raise ValueError('Center Embeddings predictions need its registered independent VAL/TEST partition')
        bags = self.select(subset, device); self.network.eval(); self.center_loss.eval(); self.network.mode = 1
        loader = DataLoader(bags, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        records = []
        try:
            for index, (images, _, _) in enumerate(loader):
                _, _, embedding, logits = self.network(images.to(device))
                if self.profile == 'center_embeddings_soft_assignment_repaired':
                    probability = self.center_loss.get_assignment(embedding)[0]
                else:
                    probability = self.network.pooling.probabilities(logits)[0]
                if not torch.isfinite(probability).all() or not torch.allclose(probability.sum(), probability.new_tensor(1.)):
                    raise FloatingPointError('Center Embeddings source classifier/Student-t probability is invalid')
                slide = bags.order[index]; row = bags.candidates.rows[slide]
                record = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'target']}
                record.update(subset=subset, source_slides=row['slide_id'], prediction=int(probability[1] >= .5),
                    center_candidate_count=bags.candidates.counts[slide], center_selected_count=len(bags.selected[slide]),
                    center_selected_indices_sha256=digest_object(bags.selected[slide]),
                    p_0=float(probability[0]), p_1=float(probability[1]))
                records.append(record)
        finally:
            bags.candidates.close()
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        positive = (frame.p_1.to_numpy() >= .5).astype(int)
        _, fpr, fnr = definitions()[3].calc_err(positive, frame.target.to_numpy())
        return {'center_embeddings_val_balanced_accuracy': float(1-(fpr+fnr)/2)}

    def selection_improved(self, objective, best):
        return objective >= max(0., best)

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'train_selections': self.train_selections})

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'contract', 'recipe', 'completed_epochs', 'optimizer_updates', 'train_selections'}
                or state.get('contract') != self.contract):
            raise ValueError('Center Embeddings source/asset/model/recipe identity changed')
        epochs, updates = state['completed_epochs'], state['optimizer_updates']
        rows = {r['slide_id']: r for r in self.bundle.records('train')}
        if (type(epochs) is not int or epochs < 0 or type(updates) is not int
                or updates != epochs*math.ceil(len(rows)/self.train_batch_size)
                or len(state['train_selections']) != epochs
                or (self.recipe is not None and self.recipe != state['recipe'])):
            raise ValueError('Center Embeddings complete epoch or actual dual optimizer step count changed')
        for selection in state['train_selections']:
            if (set(selection) != {'selection', 'bag_order', 'scanned'} or set(selection['selection']) != set(rows)
                    or len(selection['bag_order']) != len(rows) or set(selection['bag_order']) != set(rows)
                    or selection['scanned'] != {slide: self.assets.count(row) for slide, row in rows.items()}):
                raise ValueError('Center Embeddings original exhaustive TRAIN population or bag order changed')
            for slide, indices in selection['selection'].items():
                count = self.assets.count(rows[slide])
                if len(indices) != 32 or len(set(indices)) != 32 or any(type(i) is not int or not 0 <= i < count for i in indices):
                    raise ValueError('Center Embeddings checkpoint lost original selected32 patch identities')
        for key, value in state.items():
            if key != 'contract':
                setattr(self, key, copy.deepcopy(value))
        self.epoch_index = epochs-1

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile,
            'native_network': 'Full pretrained ImageNet ResNet34v2, p2/p3/p4 FPN128, fixed32-channel spatial bag CNN, shared512embedding, instance/bag heads, trainable512D class centers',
            'source_typo_repair': 'Undefined bag_feat -> the exact normalized bag_embed third output; CenterLoss device constant follows actual centers',
            'training': 'Every epoch scores all TRAIN candidates with original TRAIN RandomHorizontalFlip, selects positive-score top32 in source score order, Python random.sample bag order; joint instance CE + center loss + bag CE, coefficients1/1/1',
            'input_pixels': 256, 'selected_k': 32, 'train_batch_size': self.train_batch_size,
            'source_default_recipe': '20epochs; TRAIN bag batch4, VAL/TEST bag batch1; source-current worker0 recoverable I/O port; no clipping/earlystop; two Adam optimizers',
            'recipe': self.recipe, 'center_lr': self.center_lr,
            'network_optimizer': 'features cfg.lr(default1e-4), new modules.01, wd1e-4; literal ep10/15/17 reset bases.001/.01 and factors.1/.01/.001',
            'center_optimizer': 'Separate Adam glr(default1e-4), no weight decay; gradient compensation divided by center coefficient1',
            'preprocessing': 'Actual torchvision0.7 Resize256, TRAIN RandomHorizontalFlip.5, ToTensor, Normalize(mean.5/std.1); source RGB/coordinate read unchanged or exact cache',
            'selection': 'VAL binary balanced accuracy, best starts0 and last >= tie wins; positive p>=.5 including ties',
            'inference': 'Original commented Student-t center soft assignment' if 'soft_assignment' in self.profile else 'Original default bag classifier',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'resume': 'Full ResNet/FPN/shared heads/BN buffers and center tensors, both Adam states, literal LR phase, every selected patch/bag order and all global RNG',
            'raw_input_contract': self.assets.contract, 'synthetic': bool(self.bundle.config.get('synthetic', False)),
            'real_wsi_validation': 'not_run', 'paper_scores': 'not_reproduced'}

"""Complete SAM-MIL attention-route training, source control and named repair."""
import copy
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .io import digest_object
from .sammil_source import COMMIT, PROFILES, build_model, definitions, full_step


def weights_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SlideDataset(Dataset):
    def __init__(self, assets, rows):
        self.assets, self.rows = assets, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        value = self.assets.load(self.rows[index])
        return (value['features'], self.rows[index]['target'], value['is_group_feat'],
                value['relative_area'], index)


class SAMMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile, recipe='tcga', num_workers=0):
        super().__init__()
        if profile not in PROFILES or bundle.sammil is None or recipe not in {'tcga', 'camelyon16'}:
            raise ValueError('SAM-MIL needs its registered source profile, measured inputs and explicit recipe')
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError('SAM-MIL num_workers must be a nonnegative integer')
        self.bundle, self.profile, self.assets = bundle, profile, bundle.sammil
        self.repaired = profile == 'sammil_attn_objective_repaired'
        self.student = build_model(bundle.dim, len(bundle.classes), self.repaired)
        _, self._source_utility = definitions(self.repaired)
        # Literal SAM branch in main.py: random initialized deepcopy, not a
        # pretrained MHIM teacher and not an EMA update. The repaired objective
        # uses the student's attention; it does not invent a new EMA mechanism.
        self.teacher = copy.deepcopy(self.student)
        self.initial_teacher_sha256 = weights_digest(self.teacher)
        if self.repaired:
            self.teacher.requires_grad_(False)
        self.recipe_name, self.num_workers = recipe, num_workers
        self.stop_epoch = 70 if recipe == 'tcga' else 130
        self.contract = digest_object({'source': COMMIT, 'profile': profile,
            'input': self.assets.contract, 'bundle': bundle.fingerprint,
            'recipe': recipe, 'num_workers': num_workers})
        self.recipe = None
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders, self.selection_history, self.lr_history = [], [], []
        self.early_best_auc, self.early_bad, self.training_done = None, 0, False

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'sammil_val_auc'):
            raise ValueError('SAM-MIL requires complete bags, all objectives, native unclipped Adam and validation AUC')
        self.recipe = {key: cfg[key] for key in ['epochs', 'lr', 'weight_decay', 'patience', 'selection_metric']}
        self.stop_epoch = 70 if self.recipe_name == 'tcga' else 130

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self.recipe['epochs'], 0.)
        return self._optimizer

    def validate_optimizer_state(self):
        group = self._optimizer.param_groups
        if (len(group) != 1 or {id(p) for p in group[0]['params']} != {id(p) for p in self.student.parameters()}
                or group[0]['betas'] != (.9, .999) or group[0]['eps'] != 1e-8
                or group[0]['weight_decay'] != self.recipe['weight_decay']
                or group[0]['lr'] != self._scheduler.get_last_lr()[0]
                or self._scheduler.T_max != self.recipe['epochs']
                or self._scheduler.last_epoch != self.completed_epochs
                or weights_digest(self.teacher) != self.initial_teacher_sha256):
            raise ValueError('SAM-MIL actual Adam, complete cosine schedule or fixed source teacher changed')

    def _loader(self, subset):
        rows = self.bundle.records(subset); dataset = SlideDataset(self.assets, rows)
        sampler = RandomSampler(dataset) if subset == 'train' else SequentialSampler(dataset)
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=self.num_workers), rows

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('SAM-MIL needs the complete restored source epoch state')
        self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('SAM-MIL input or actual Adam identity changed')
        self.train(); loader, rows = self._loader('train')
        self.lr_history.append(optimizer.param_groups[0]['lr'])
        losses, order = [], []
        for index, batch in enumerate(loader):
            bag, target, flags, areas = [value.to(device) for value in batch[:4]]
            optimizer.zero_grad()
            result = full_step(self.student, self.teacher, bag, target, flags, areas,
                repaired=self.repaired, step=self.epoch_index*len(loader)+index,
                utility=self._source_utility)
            if not torch.isfinite(result['loss']):
                raise FloatingPointError('SAM-MIL complete source objective became nonfinite')
            result['loss'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise FloatingPointError('SAM-MIL complete source gradient became nonfinite')
            optimizer.step(); self.optimizer_updates += 1
            order.append(rows[int(batch[4].item())]['slide_id'])
            losses.append({key: float(result[key].detach()) if isinstance(result[key], torch.Tensor) else float(result[key])
                for key in ['loss', 'global_ce', 'global_forward_ce', 'distillation', 'group_ce', 'consistency', 'patch_count', 'kept']})
        self._scheduler.step()
        self.completed_epochs = self.epoch_index+1; self.train_orders.append(order)
        self.validate_optimizer_state()
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('SAM-MIL prediction needs the registered validation/test slide population')
        self.eval(); loader, rows = self._loader(subset); records = []
        for batch in loader:
            logits = self.student.forward_test(batch[0].to(device))
            row = rows[int(batch[4].item())]
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'target']}
            record.update(subset=subset, source_slides=row['slide_id'],
                          sammil_val_ce=float(F.cross_entropy(logits, batch[1].to(device))))
            record.update({f'p_{i}': float(value) for i, value in enumerate(logits.softmax(-1)[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        from sklearn.metrics import roc_auc_score
        probability = frame[[f'p_{i}' for i in range(len(self.bundle.classes))]].to_numpy()
        auc = (roc_auc_score(frame.target, probability[:, 1]) if probability.shape[1] == 2
               else roc_auc_score(frame.target, probability, multi_class='ovr', average='macro'))
        return {'sammil_val_auc': float(auc)}

    def after_validation(self, metrics):
        auc = metrics['sammil_val_auc']
        if self.early_best_auc is None or auc >= self.early_best_auc:
            self.early_best_auc, self.early_bad = auc, 0
        else:
            self.early_bad += 1
        self.training_done = self.early_bad >= self.recipe['patience'] and self.epoch_index > self.stop_epoch
        self.selection_history.append({'epoch': self.epoch_index, 'auc': auc,
            'best_auc': self.early_best_auc, 'bad': self.early_bad, 'done': self.training_done})

    def should_stop(self, bad, cfg):
        return self.training_done

    def get_extra_state(self):
        return {'contract': self.contract, 'recipe': copy.deepcopy(self.recipe),
            'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
            'optimizer_updates': self.optimizer_updates, 'train_orders': copy.deepcopy(self.train_orders),
            'selection_history': copy.deepcopy(self.selection_history), 'lr_history': self.lr_history[:],
            'scheduler': copy.deepcopy(self._scheduler.state_dict()) if hasattr(self, '_scheduler')
                else copy.deepcopy(getattr(self, '_loaded_scheduler_state', None)),
            'initial_teacher_sha256': self.initial_teacher_sha256,
            'teacher_gradients': {name: value.grad.detach().cpu().clone()
                for name, value in self.teacher.named_parameters() if value.grad is not None},
            'early_best_auc': self.early_best_auc, 'early_bad': self.early_bad, 'training_done': self.training_done}

    def set_extra_state(self, state):
        expected = set(self.get_extra_state())
        recipe = state.get('recipe')
        if (not isinstance(recipe, dict) or set(recipe) != {'epochs', 'lr', 'weight_decay', 'patience', 'selection_metric'}
                or type(recipe['epochs']) is not int or recipe['epochs'] < 1
                or type(recipe['patience']) is not int or recipe['patience'] < 1
                or not np.isfinite(recipe['lr']) or recipe['lr'] <= 0
                or not np.isfinite(recipe['weight_decay']) or recipe['weight_decay'] < 0
                or recipe['selection_metric'] != 'sammil_val_auc'):
            raise ValueError('SAM-MIL complete saved recipe is invalid')
        if self.recipe is None:  # Independent evaluate() rebuilds without an optimizer.
            self.recipe = copy.deepcopy(recipe)
        if (set(state) != expected or state['contract'] != self.contract
                or state['initial_teacher_sha256'] != self.initial_teacher_sha256
                or state['recipe'] != self.recipe
                or any(type(state[key]) is not int for key in ['completed_epochs', 'epoch_index', 'optimizer_updates', 'early_bad'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or state['completed_epochs'] > self.recipe['epochs']
                or len(state['selection_history']) != state['completed_epochs']
                or len(state['lr_history']) != state['completed_epochs']
                or len(state['train_orders']) != state['completed_epochs']):
            raise ValueError('SAM-MIL complete input, recipe or training state changed')
        train = {row['slide_id'] for row in self.bundle.records('train')}
        if (state['optimizer_updates'] != len(train)*state['completed_epochs']
                or any(len(order) != len(train) or set(order) != train for order in state['train_orders'])):
            raise ValueError('SAM-MIL omitted source bags or Adam updates')
        best, bad, done = None, 0, False
        for epoch, record in enumerate(state['selection_history']):
            auc = record['auc']
            if type(auc) not in {int, float} or not np.isfinite(auc) or not 0 <= auc <= 1:
                raise ValueError('SAM-MIL validation AUC history changed')
            if best is None or auc >= best:
                best, bad = auc, 0
            else:
                bad += 1
            done = bad >= self.recipe['patience'] and epoch > self.stop_epoch
            if record != {'epoch': epoch, 'auc': auc, 'best_auc': best, 'bad': bad, 'done': done}:
                raise ValueError('SAM-MIL latest-tie stopping history changed')
        if (best, bad, done) != (state['early_best_auc'], state['early_bad'], state['training_done']):
            raise ValueError('SAM-MIL stopping counters disagree with the actual validation history')
        if self.repaired and state['teacher_gradients']:
            raise ValueError('Repaired SAM-MIL teacher must remain fixed and gradient-free')
        parameters = dict(self.teacher.named_parameters())
        if set(state['teacher_gradients'])-set(parameters):
            raise ValueError('SAM-MIL source teacher gradient bank has unknown tensors')
        for name, parameter in parameters.items():
            gradient = state['teacher_gradients'].get(name)
            if gradient is not None and (gradient.shape != parameter.shape or not torch.isfinite(gradient).all()):
                raise ValueError('SAM-MIL source accumulated teacher gradient state changed')
            parameter.grad = None if gradient is None else gradient.to(parameter).clone()
        if state['scheduler'] is None or state['scheduler']['T_max'] != self.recipe['epochs']:
            raise ValueError('SAM-MIL complete cosine state missing')
        reference_optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(()))],
            lr=self.recipe['lr'], weight_decay=self.recipe['weight_decay'])
        reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reference_optimizer, self.recipe['epochs'], 0.)
        rates = []
        for _ in range(state['completed_epochs']):
            rates.append(reference_optimizer.param_groups[0]['lr'])
            reference_optimizer.step(); reference_scheduler.step()
        if state['lr_history'] != rates or state['scheduler'] != reference_scheduler.state_dict():
            raise ValueError('SAM-MIL cosine progression or actual learning-rate history changed')
        if hasattr(self, '_scheduler'):
            self._scheduler.load_state_dict(state['scheduler'])
        else:
            self._loaded_scheduler_state = copy.deepcopy(state['scheduler'])
        for key in ['epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders',
                    'selection_history', 'lr_history', 'early_best_auc', 'early_bad', 'training_done']:
            setattr(self, key, copy.deepcopy(state[key]))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if weights_digest(self.teacher) != self.initial_teacher_sha256:
            raise ValueError('SAM-MIL released teacher was changed despite having no optimizer or EMA updates')
        return result

    def protocol(self):
        return {'method': self.profile, 'source_commit': COMMIT,
            'architecture': 'author full512 projection/128 attention/512 bag embedding and explicit C-way head',
            'inputs': 'actual SAM-vit-H memberships, mask areas, original patch coordinates; source appended arithmetic-mean group tokens',
            'mask': 'source area-sensitive sigmoid(.0005*(area-5000))*.9, random_ratio.001; preserve all group tokens; no fabricated coordinates',
            'objective': 'global CE +0*feature KD +.5*mean-five-pseudobag CE +1000*original consistency kernel',
            'source_control': 'main CE and KD are overwritten by last pseudo-bag; consistency only accumulates unoptimized teacher gradients; erroneous batch-index mask preserved' if not self.repaired else None,
            'repairs': 'correct group/patch indexing and relative-area compact mapping; main CE and KD use global branch; consistency uses student attention with original detached projection and original kernel; explicit C-way validation' if self.repaired else None,
            'teacher': 'literal same-initialization deepcopy; never EMA-updated in released SAM branch; train-mode dropout retained for masking calls',
            'recipe': self.recipe, 'dataset_recipe': self.recipe_name,
            'native_recipe': 'Adam2e-4/wd1e-5,cosine200,batch1,RandomSampler,noAMP,no clipping; five random pseudobags retain remainder',
            'selection': 'first strict best VAL AUC checkpoint; early-stop counter resets on exact ties, stops after epoch>70 TCGA or>130 C16; disjoint fixed validation, no TEST guidance',
            'optimizer_updates': self.optimizer_updates, 'completed_epochs': self.completed_epochs,
            'input_contract': self.assets.contract, 'synthetic': bool(self.bundle.config.get('synthetic', False)),
            'real_wsi_validation': 'not_run', 'paper_scores': 'not_reproduced'}

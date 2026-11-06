"""Complete original Libra-MIL SOT, learned prototypes and source recipe."""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import digest_object
from .libramil_source import COMMIT, build_original
from .libramil_training import SourceSelection, original_training, source_optimizer


class Bags(Dataset):
    def __init__(self, owner, subset):
        self.owner, self.rows, self.order = owner, owner.bundle.records(subset), []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; self.order.append(index)
        return self.owner.inputs.load(row), int(row['target'])


class LibraMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='libramil_source_p4'):
        super().__init__()
        profiles = {'libramil_source_p4': 4, 'libramil_readme_p6': 6}
        if profile not in profiles or getattr(bundle, 'libramil', None) is None:
            raise ValueError('Libra-MIL requires bound full original language/feature inputs')
        self.bundle, self.inputs, self.profile = bundle, bundle.libramil, profile
        self.prototypes = profiles[profile]
        self.network = build_original(self.inputs.text['structural'], self.inputs.text['bags'], self.prototypes)
        self.contract = digest_object({'source': COMMIT, 'profile': profile,
            'inputs': self.inputs.contract, 'classes': bundle.classes,
            'selection': 'literal source stale best_auc branch; standard slide accuracy'})
        self.recipe = None
        self.selection = SourceSelection()
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders = []
        self.scheduler = self._optimizer = None
        self._saved_scheduler = None
        self._source_metrics = None
        self._selected = False

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'libramil_val_accuracy'
                or cfg['weight_decay'] != .01 or cfg['lr'] != .0001
                or type(cfg['epochs']) is not int or cfg['epochs'] < 1):
            raise ValueError('Libra-MIL requires original full-bag CE, AdamW1e-4/default.01decay, no clipping, source accuracy/AUC selection')
        self.recipe = {k: cfg[k] for k in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        if self.recipe is None or lr != self.recipe['lr'] or weight_decay != self.recipe['weight_decay']:
            raise ValueError('Configure full original Libra-MIL recipe before optimization')
        self._optimizer, self.scheduler = source_optimizer(self.network, len(self.bundle.records('train')),
                                                         epochs=self.recipe['epochs'], lr=lr)
        return self._optimizer

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('Libra-MIL epoch does not continue the complete source history')
        self.epoch_index = epoch

    def loader(self, subset):
        dataset = Bags(self, subset)
        return dataset, DataLoader(dataset, batch_size=1, shuffle=subset == 'train', num_workers=0)

    def epoch(self, bundle, optimizer, device):
        if bundle is not self.bundle or optimizer is not self._optimizer or self.epoch_index != self.completed_epochs:
            raise ValueError('Libra-MIL bound source epoch/optimizer mismatch')
        dataset, loader = self.loader('train')
        losses = []
        def capture(_module, _args, result):
            losses.append(float(result['loss'].detach()))
        hook = self.network.register_forward_hook(capture)
        try:
            original_training().train(self.network, loader, device, self.epoch_index,
                                      optimizer, self.scheduler, len(bundle.classes))
        finally:
            hook.remove()
        if sorted(dataset.order) != list(range(len(dataset))) or not np.isfinite(losses).all():
            raise ValueError('Libra-MIL full original sampler/loss history is invalid')
        self.train_orders.append([dataset.rows[i]['slide_id'] for i in dataset.order])
        self.optimizer_updates += len(dataset.order); self.completed_epochs += 1
        return [{'loss': x} for x in losses]

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle is not self.bundle or subset not in {'val', 'test'}:
            raise ValueError('Libra-MIL prediction needs its bound validation/test set')
        dataset, loader = self.loader(subset)
        probabilities = []
        def capture(_module, _args, result):
            probabilities.append(result['logits'].softmax(1).detach().cpu().numpy()[0])
        hook = self.network.register_forward_hook(capture)
        try:
            metrics = original_training().evaluate(self.network, loader, device,
                                                  max(0, self.epoch_index), len(bundle.classes))
        finally:
            hook.remove()
        records = []
        for index, probability in zip(dataset.order, probabilities):
            row = dataset.rows[index]
            result = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'target']}
            result.update(source_slides=row['slide_id'], **{f'p_{i}': float(v) for i, v in enumerate(probability)})
            records.append(result)
        if len(records) != len(dataset):
            raise ValueError('Libra-MIL missing complete subset predictions')
        self._source_metrics = tuple(map(float, metrics))
        if subset == 'val':
            # Preserve the unused sequential TEST loader's CPU base-seed RNG
            # draw without examining TEST during model selection. Source model
            # evaluation has no stochastic layers; its TEST size is irrelevant.
            torch.empty((), dtype=torch.int64).random_()
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        if self._source_metrics is None:
            raise ValueError('Libra-MIL source validation metrics missing')
        acc, auc, f1 = self._source_metrics
        return {'libramil_val_accuracy': acc, 'libramil_val_auc': auc, 'libramil_val_f1': f1}

    def selection_objective(self, metrics, epoch, current):
        self._selected = self.selection.observe(epoch, metrics['libramil_val_accuracy'], metrics['libramil_val_auc'])
        return float(epoch+1)

    def selection_improved(self, objective, best):
        return self._selected

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
            'optimizer_updates': self.optimizer_updates, 'train_orders': self.train_orders,
            'selection': self.selection.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self._saved_scheduler})

    def set_extra_state(self, state):
        keys = {'contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates',
                'train_orders', 'selection', 'scheduler'}
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.contract
                or (self.recipe is not None and state['recipe'] != self.recipe)
                or any(type(state[k]) is not int for k in ['epoch_index', 'completed_epochs', 'optimizer_updates'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1):
            raise ValueError('Libra-MIL checkpoint source/recipe/input/epoch binding changed')
        ids = {r['slide_id'] for r in self.bundle.records('train')}
        if (len(state['train_orders']) != state['completed_epochs']
                or any(len(order) != len(ids) or set(order) != ids for order in state['train_orders'])
                or state['optimizer_updates'] != len(ids)*state['completed_epochs']):
            raise ValueError('Libra-MIL complete TRAIN order or optimizer history changed')
        selection = SourceSelection(); selection.load_state_dict(state['selection'])
        if len(selection.history) != state['completed_epochs']:
            raise ValueError('Libra-MIL source selection history lost an epoch')
        scheduler = state['scheduler']
        if (state['completed_epochs'] and (not isinstance(scheduler, dict)
                or scheduler.get('last_epoch') != state['optimizer_updates'])):
            raise ValueError('Libra-MIL complete original scheduler state lost')
        self.selection = selection
        for k in ['recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates', 'train_orders']:
            setattr(self, k, copy.deepcopy(state[k]))
        self._saved_scheduler = copy.deepcopy(scheduler)
        if self.scheduler is not None and scheduler is not None:
            self.scheduler.load_state_dict(scheduler)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'inputs': self.inputs.contract,
            'recipe': self.recipe, 'native_dimension': 512, 'visual_prototypes': self.prototypes,
            'structural_descriptions': len(self.inputs.text['structural']), 'classes': len(self.bundle.classes),
            'network': 'Full author SOT20iterations/epsilon.05, visual and structural learned prototypes, learned bag prompts, full512projections/8headattention and CE',
            'text': 'Actual original complete pretrained CONCH text computed once, no_grad; learned copies of complete original embeddings trained',
            'unused_parameters': 'Original temp_struct/temp_vis retained unchanged; pooling_type/loss_weights/ablation flags unused by original forward',
            'optimization': 'Actual original AdamW default decay.01, lr1e-4; per-update100warmup/cosine;20epochs, no earlystop/clipping/instance sampling',
            'selection': 'Literal standard accuracy/AUC branch, including stale best_auc on accuracy increase; source behavior explicitly preserved',
            'test_lock': 'No TEST access until explicit evaluation. One original unused sequential loader base-seed draw preserved per VAL.',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'formal_admission': False}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in ['T_struct_llm', 'T_bag_llm']:
            key = 'network.'+name
            if key not in state_dict or not torch.equal(state_dict[key].cpu(), getattr(self.network, name).cpu()):
                raise ValueError('Libra-MIL checkpoint original frozen language buffers changed')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

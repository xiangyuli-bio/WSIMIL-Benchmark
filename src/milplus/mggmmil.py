"""Complete original MGGM-MIL Adam/CE, validation selection and exact recovery."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .mggmmil_data import GraphInputs
from .mggmmil_source import original_model, COMMIT
from .upstream import load_definitions

PROFILE = 'mggmmil_uaf_original_graph_patient_reference'


class Bags(Dataset):
    def __init__(self, owner, subset):
        self.owner, self.rows, self.order = owner, owner.bundle.records(subset), []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.order.append(index)
        row = self.rows[index]
        x, edges = self.owner.assets.load(row)
        return x, row['target'], row['slide_id'], [], edges


def original_training():
    return load_definitions('mggmmil', 'train_mil.py', ['MIL'],
        {'torch': torch, 'np': np, 'F': F, 'tqdm': tqdm, 'logging': __import__('logging')}).MIL


class SourceSelection:
    def __init__(self):
        self.epoch = 0
        self.best = float('inf')
        self.counter = 0
        self.improved = False
        self.done = False
        self.selected_epoch = None

    def observe(self, epoch, loss):
        if type(epoch) is not int or epoch != self.epoch+1 or not np.isfinite(loss):
            raise ValueError('MGGM-MIL source validation epochs/loss must be consecutive and finite')
        self.epoch = epoch
        self.improved = epoch >= 50 and loss < self.best
        if self.improved:
            self.best = float(loss)
            self.counter = 0
            self.selected_epoch = epoch
        elif epoch >= 50:
            self.counter += 1
        self.done = self.counter >= 20

    def load(self, state):
        if (set(state) != set(vars(self)) or type(state['epoch']) is not int or state['epoch'] < 0
                or type(state['counter']) is not int or not 0 <= state['counter'] <= 20
                or state['done'] != (state['counter'] >= 20)
                or type(state['improved']) is not bool
                or state['selected_epoch'] is not None and not 50 <= state['selected_epoch'] <= state['epoch']
                or state['selected_epoch'] is None and state['best'] != float('inf')
                or state['selected_epoch'] is not None and not np.isfinite(state['best'])):
            raise ValueError('Original MGGM-MIL selected epoch/patience state changed')
        self.__dict__.update(copy.deepcopy(state))


class MGGMMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile=PROFILE, backend='torch', scan_chunk=64):
        super().__init__()
        if profile != PROFILE:
            raise ValueError('Choose the complete original MGGM-MIL graph protocol')
        self.bundle, self.profile = bundle, profile
        self.assets = GraphInputs(bundle)
        self.network, self.identity = original_model(bundle.dim, backend, scan_chunk)
        self.selection = SourceSelection()
        self.completed_epochs = 0
        self.epoch_index = -1
        self.optimizer_updates = 0
        self.gradient_updates = {k: 0 for k, _ in self.network.named_parameters()}
        self.train_orders, self.val_orders = [], []
        self.recipe, self._optimizer, self._val_loss = None, None, None

    def configure_training(self, cfg):
        if (cfg['epochs'] < 50 or cfg['lr'] != 3e-5 or cfg['weight_decay'] != 1e-5
                or cfg['patience'] != 20 or cfg['grad_clip'] is not None
                or cfg['max_instances'] is not None or cfg['selection_metric'] != 'mggmmil_val_negative_ce'):
            raise ValueError('Native MGGM-MIL: full bags, Adam3e-5/wd1e-5, no clipping, first selection epoch50, patience20')
        self.recipe = {k: cfg[k] for k in ['epochs', 'lr', 'weight_decay', 'patience', 'threads', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        if self.recipe is None or (lr, weight_decay) != (3e-5, 1e-5):
            raise ValueError('Configure the original MGGM-MIL Adam recipe first')
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        def count(opt, args, kwargs):
            self.optimizer_updates += 1
            for k, p in self.network.named_parameters():
                self.gradient_updates[k] += int(p.grad is not None)
        optimizer.register_step_pre_hook(count)
        self._optimizer = optimizer
        return optimizer

    def start_epoch(self, epoch):
        if epoch != self.completed_epochs:
            raise ValueError('MGGM-MIL epoch must follow the exact previous checkpoint')
        self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        if bundle is not self.bundle or optimizer is not self._optimizer:
            raise ValueError('MGGM-MIL changed bound bundle/optimizer')
        self.assets.verify()
        data = Bags(self, 'train')
        loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        owner = SimpleNamespace(model=self.network, loss=criterion)
        train_fn = original_training().train_inference
        losses = []
        self.network.train()
        for fea, label, path, adj, edges in loader:
            fea, label = fea.to(device), label.to(device)
            optimizer.zero_grad()
            loss = train_fn(owner, fea, label, 'MGGMMIL', [], [e.to(device) for e in edges])
            if not torch.isfinite(loss):
                raise FloatingPointError('Original MGGM-MIL CE became nonfinite')
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        if sorted(data.order) != list(range(len(data))):
            raise ValueError('MGGM-MIL omitted or duplicated actual TRAIN slide bags')
        self.train_orders.append([data.rows[i]['slide_id'] for i in data.order])
        self.completed_epochs += 1
        return [{'loss': float(np.mean(losses))}]

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle is not self.bundle or subset not in {'val', 'test'}:
            raise ValueError('Use the bound MGGM-MIL held-out split')
        data = Bags(self, subset)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        owner = SimpleNamespace(model=self.network, loss=nn.CrossEntropyLoss())
        test_fn = original_training().test_inference
        self.network.eval()
        result, losses = [], []
        for fea, label, ids, adj, edges in loader:
            loss, probability = test_fn(owner, fea.to(device), label.to(device), 'MGGMMIL', [], [e.to(device) for e in edges])
            if not torch.isfinite(loss) or not torch.isfinite(probability).all():
                raise FloatingPointError('MGGM-MIL held-out prediction became nonfinite')
            row = data.rows[data.order[-1]]
            result.append({**{k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'target']},
                           'source_slides': row['slide_id'], 'p_0': float(probability[0, 0]), 'p_1': float(probability[0, 1])})
            losses.append(float(loss))
        self._prediction_loss = (subset, float(np.mean(losses)))
        if subset == 'val':
            self._val_loss = float(np.mean(losses))
            if len(self.val_orders) < self.completed_epochs:
                self.val_orders.append([data.rows[i]['slide_id'] for i in data.order])
        return pd.DataFrame(result)

    def validation_metrics(self, frame):
        bound = getattr(self, '_prediction_loss', None)
        if bound is None or set(frame.subset) != {bound[0]}:
            raise ValueError('MGGM-MIL metrics must bind actual held-out predictions')
        return {'mggmmil_'+bound[0]+'_negative_ce': -bound[1]}

    def selection_objective(self, metrics, epoch, current):
        loss = -metrics['mggmmil_val_negative_ce']
        self.selection.observe(epoch+1, loss)
        return -loss

    def selection_improved(self, objective, best):
        return self.selection.improved

    @property
    def training_done(self):
        return self.selection.done

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.assets.contract, 'identity': self.identity, 'recipe': self.recipe,
            'completed_epochs': self.completed_epochs, 'epoch_index': self.epoch_index,
            'optimizer_updates': self.optimizer_updates, 'gradient_updates': self.gradient_updates,
            'train_orders': self.train_orders, 'val_orders': self.val_orders, 'selection': vars(self.selection)})

    def set_extra_state(self, state):
        keys = set(self.get_extra_state())
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.assets.contract
                or state['identity'] != self.identity
                or self.recipe is not None and state['recipe'] != self.recipe):
            raise ValueError('MGGM-MIL source/graph/backend/recipe checkpoint identity changed')
        n = state['completed_epochs']
        train = {r['slide_id'] for r in self.bundle.records('train')}
        val = {r['slide_id'] for r in self.bundle.records('val')}
        if (type(n) is not int or n < 0 or state['epoch_index'] != n-1
                or state['optimizer_updates'] != n*len(train)
                or state['selection']['epoch'] != n
                or len(state['train_orders']) != n or len(state['val_orders']) != n
                or any(len(x) != len(train) or set(x) != train for x in state['train_orders'])
                or any(len(x) != len(val) or set(x) != val for x in state['val_orders'])
                or set(state['gradient_updates']) != set(self.gradient_updates)
                or any(type(x) is not int or x < 0 or x > state['optimizer_updates'] for x in state['gradient_updates'].values())):
            raise ValueError('MGGM-MIL checkpoint lost full slide/epoch/update history')
        self.selection.load(state['selection'])
        for key in keys-{'contract', 'identity', 'selection'}:
            setattr(self, key, copy.deepcopy(state[key]))

    def validate_optimizer_state(self):
        for k, p in self.network.named_parameters():
            n, state = self.gradient_updates[k], self._optimizer.state.get(p)
            if (n and (state is None or int(state['step']) != n)) or (not n and state is not None):
                raise ValueError('MGGM-MIL restored Adam moments disagree with actual source updates: '+k)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'graph_contract': self.assets.contract,
                'source_identity': self.identity, 'recipe': self.recipe,
                'architecture': '512to384;4fullGENConv/SRMamba16state4conv1expand;K4+8+12;GABMIL256',
                'optimizer': 'Original Adam3e-5wd1e-5, constant LR, one whole slide/update',
                'selection': 'strict minimum VAL CE beginning at1-basedepoch50;patience20;maximum200',
                'graph_origin': self.assets.raw['graph_provenance'],
                'graph_builder': 'not published; preserve supplied original edge lists, no inferred graph substitute',
                'test_boundary': 'No TEST access during training or model selection',
                'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
                'formal_admission': False, 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run'}

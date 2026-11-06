"""MG-Trans released two-scale graph, semantic loss and full sliding MI recipe."""
import copy

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .io import digest_object
from .mgtrans_source import COMMIT, build_model, source_definitions


class SlideDataset(Dataset):
    def __init__(self, assets, rows):
        self.assets, self.rows = assets, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (*self.assets.load(row), row['target'], index)


def collate(batch):
    if len(batch) != 1:
        raise ValueError('MG-Trans released training uses batch size one')
    x, c, y, d, mask, label, index = batch[0]
    return x, c, y, d, torch.tensor([label]), mask, index


def bandwidth(x):
    """Literal source first-ten-columns distance heuristic, not a new estimator."""
    distances = squareform(pdist(x.detach().cpu().numpy(), 'euclidean'))
    sigma = float(np.mean(np.mean(np.sort(distances[:, :10], 1))))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('MG-Trans source MI bandwidth is zero or nonfinite; input history is degenerate')
    return sigma**2


class MGTrans(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='mgtrans_source_repaired', num_workers=None):
        super().__init__()
        if (profile != 'mgtrans_source_repaired' or bundle.task_type != 'classification'
                or bundle.config.get('training_unit', 'slide') != 'slide' or bundle.mgtrans is None):
            raise ValueError('MG-Trans requires registered measured dual-scale slide-classification inputs')
        if num_workers is not None and (type(num_workers) is not int or num_workers < 0):
            raise ValueError('MG-Trans num_workers must be null or a nonnegative integer')
        self.bundle, self.assets, self.profile = bundle, bundle.mgtrans, profile
        self.network = build_model(bundle.dim, len(bundle.classes))
        _, self.source_util = source_definitions()
        self.num_workers = num_workers
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'assets': self.assets.contract,
                                       'bundle': bundle.fingerprint, 'num_workers': num_workers})
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.recipe, self.best_error = None, None
        self.early_bad, self.training_done = 0, False
        self.train_orders, self.mi_updates, self.last_window, self.selection_history = [], [], [], []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'mgtrans_val_error':
            raise ValueError('MG-Trans requires its full source anchor/graph/MI route and validation-error selection')
        self.recipe = {key: cfg[key] for key in ['epochs', 'patience', 'lr', 'weight_decay', 'selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay,
                                           betas=(.9, .999), eps=1e-8)
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 1 or groups[0]['lr'] != self.recipe['lr']
                or groups[0]['weight_decay'] != self.recipe['weight_decay']
                or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('MG-Trans actual Adam state or parameter ownership changed')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('MG-Trans epochs must advance from the complete restored state')
        self.epoch_index = epoch

    def _loader(self, subset, device):
        rows = self.bundle.records(subset)
        dataset = SlideDataset(self.assets, rows)
        sampler = RandomSampler(dataset) if subset == 'train' else SequentialSampler(dataset)
        readers = (4 if torch.device(device).type == 'cuda' else 0) if self.num_workers is None else self.num_workers
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=readers, collate_fn=collate), rows

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('MG-Trans actual input or optimizer changed')
        self.train(); loader, rows = self._loader('train', device)
        # The original history starts with a dummy zero row and resets each epoch.
        window = [torch.zeros((1, 192), device=device) for _ in range(4)]
        losses, order, mi_steps = [], [], 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            inputs = tuple(x.to(device) for x in [*batch[:4], batch[5], batch[4]])
            result = self.network(*inputs)
            loss, current = result[4], result[5:9]
            if step < 31:
                window = [torch.cat((old, value.detach()), dim=0) for old, value in zip(window, current)]
                mi = loss.new_zeros(())
            else:
                window = [torch.cat((old[1:].detach(), value), dim=0) for old, value in zip(window, current)]
                sigmas = [bandwidth(x) for x in window]
                mi = (.0001*self.source_util.calculate_MI(window[0], window[2], sigmas[0], sigmas[2])
                      + .0001*self.source_util.calculate_MI(window[1], window[3], sigmas[1], sigmas[3]))
                loss = loss+mi
                mi_steps += 1
            if not torch.isfinite(loss):
                raise FloatingPointError('MG-Trans complete source objective is nonfinite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError('MG-Trans source backward produced nonfinite gradients')
            optimizer.step(); optimizer.zero_grad()
            self.optimizer_updates += 1
            window = [x.detach() for x in window]
            order.append(rows[batch[6]]['slide_id'])
            losses.append({'loss': float(loss.detach()), 'source_semantic_and_ce': float(result[4].detach()),
                           'source_mib': float(mi.detach())})
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(order); self.mi_updates.append(mi_steps)
        self.last_window = [x.detach().cpu() for x in window]
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('MG-Trans prediction requires registered validation/test inputs')
        self.eval(); loader, rows = self._loader(subset, device); records = []
        for batch in loader:
            result = self.network(*(x.to(device) for x in [*batch[:4], batch[5], batch[4]]))
            row = rows[batch[6]]
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], target=row['target'],
                          prediction=int(result[2]), mgtrans_source_loss=float(result[4]))
            record.update({f'p_{i}': float(v) for i, v in enumerate(result[1][0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        if not len(frame) or not frame.slide_id.is_unique or not np.isfinite(frame.mgtrans_source_loss).all():
            raise ValueError('MG-Trans validation needs finite source loss for every actual slide')
        return {'mgtrans_val_error': float((frame.prediction != frame.target).mean())}

    def after_validation(self, metrics):
        error = metrics['mgtrans_val_error']
        if self.best_error is None or error <= self.best_error:
            self.best_error, self.early_bad = error, 0
        else:
            self.early_bad += 1
        self.training_done = self.early_bad >= self.recipe['patience'] and self.epoch_index > 50
        self.selection_history.append({'epoch': self.epoch_index, 'error': error,
            'best_error': self.best_error, 'bad': self.early_bad, 'done': self.training_done})

    def selection_objective(self, metrics, epoch, current):
        return -metrics['mgtrans_val_error']

    def selection_improved(self, objective, best):
        return objective >= best  # Source replaces the checkpoint on exact validation-error ties.

    def get_extra_state(self):
        return copy.deepcopy({key: getattr(self, key) for key in
            ['contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates',
             'best_error', 'early_bad', 'training_done', 'train_orders', 'mi_updates', 'last_window', 'selection_history']})

    def set_extra_state(self, state):
        keys = {'contract', 'recipe', 'epoch_index', 'completed_epochs', 'optimizer_updates',
                'best_error', 'early_bad', 'training_done', 'train_orders', 'mi_updates', 'last_window', 'selection_history'}
        if (not isinstance(state, dict) or set(state) != keys or state['contract'] != self.contract
                or any(type(state[k]) is not int for k in ['epoch_index', 'completed_epochs', 'optimizer_updates', 'early_bad'])
                or state['completed_epochs'] < 0 or state['epoch_index'] != state['completed_epochs']-1
                or len(state['train_orders']) != state['completed_epochs']
                or len(state['mi_updates']) != state['completed_epochs']
                or len(state['selection_history']) != state['completed_epochs']
                or state['early_bad'] < 0 or type(state['training_done']) is not bool):
            raise ValueError('MG-Trans complete source/input/counter state changed')
        train = {row['slide_id'] for row in self.bundle.records('train')}
        if (state['optimizer_updates'] != len(train)*state['completed_epochs']
                or any(len(order) != len(train) or set(order) != train for order in state['train_orders'])
                or any(value != max(0, len(train)-31) for value in state['mi_updates'])):
            raise ValueError('MG-Trans checkpoint omitted source samples, Adam updates or the sliding MI stage')
        if self.recipe is not None and state['recipe'] != self.recipe:
            raise ValueError('MG-Trans checkpoint changed the complete declared training recipe')
        best, bad, done = None, 0, False
        for epoch, record in enumerate(state['selection_history']):
            error = record.get('error')
            if type(error) not in {int, float} or not np.isfinite(error) or not 0 <= error <= 1:
                raise ValueError('MG-Trans validation-error history is invalid')
            if best is None or error <= best:
                best, bad = error, 0
            else:
                bad += 1
            done = bad >= state['recipe']['patience'] and epoch > 50
            if record != {'epoch': epoch, 'error': error, 'best_error': best, 'bad': bad, 'done': done}:
                raise ValueError('MG-Trans source latest-tie selection or early-stop history changed')
        if (best, bad, done) != (state['best_error'], state['early_bad'], state['training_done']):
            raise ValueError('MG-Trans selected optimum or stopping counters changed')
        if state['completed_epochs']:
            expected = min(32, len(train)+1)
            if (len(state['last_window']) != 4 or any(not isinstance(x, torch.Tensor)
                    or x.shape != (expected, 192) or not torch.isfinite(x).all() for x in state['last_window'])
                    or not isinstance(state['recipe'], dict)
                    or state['recipe']['epochs'] < state['completed_epochs']):
                raise ValueError('MG-Trans saved source representation window or training budget is incomplete')
        for key in keys-{'contract'}:
            setattr(self, key, copy.deepcopy(state[key]))

    def protocol(self):
        return {'method': self.profile, 'source': COMMIT, 'input_dimension': self.bundle.dim,
            'architecture': 'released192width/3heads/768MLP,512PAM anchors, shared two-scale Transformer and actual3layer GENConv/DeepGCN spatial branch',
            'graph': 'source HNSW M16/efConstruction400/ef90,k9,2048level0-distance cut; feature detach preserved; indexThreadQty1 and a fresh native thread reset native TLS RNG; exact geometric distance ties use node-ID order and the released self-first assumption is enforced, without replacing HNSW search or its budget',
            'source_repairs': 'CPU/device-neutral graph placement; latent graph queries hidden features rather than incompatible2D coords; legacy LinearWithBias and upper-triangle symeig compatibility',
            'short_bags': 'literal released repeat-to512 of actual features/coordinates/coverage rows, never invented physical grid coordinates',
            'objective': 'CE +1e-4 negativeCE of L1(high-low logits), plus1e-4 each-scale Renyi1.01 MI from actual32-slide rolling window after31 warmup slides per epoch',
            'history': 'source zero initialization, detached previous examples, reset each epoch; literal first-ten-columns Euclidean bandwidth',
            'recipe': self.recipe, 'native_recipe': '200maxepochs, Adam1e-4/wd1e-5,batch1,RandomSampler; no clipping/AMP; declared Plateau scheduler never stepped in source',
            'selection': 'VAL slide error, latest exact tie, patience20 and source zero-based epoch>50; never selects on TEST',
            'source_scope': 'released complete cached-feature MG-Trans; no claim that unused config.num_layers12 creates12blocks or that original graph detach is end-to-end graph input differentiation',
            'native_readers': 'source4 on CUDA,0 on CPU by default; explicit num_workers override recorded',
            'completed_epochs': self.completed_epochs, 'actual_optimizer_updates': self.optimizer_updates,
            'actual_mib_updates': sum(self.mi_updates), 'train_orders_sha256': digest_object(self.train_orders),
            'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

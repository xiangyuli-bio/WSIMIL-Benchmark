"""Complete source GAT-Mamba survival training, validation selection and recovery."""
import copy
import inspect
import math

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from .gatmamba_data import COMMIT
from .gatmamba_source import network_class, source_trainer
from .io import digest_object
from .upstream import load_definitions

PROFILES = {'gatmamba_source', 'gatmamba_edge_conditioned'}


def patients(bundle, subset):
    rows = bundle.records(subset)
    groups = {}
    for row in rows:
        groups.setdefault(row['global_patient_id'], []).append(row)
    return [groups[k] for k in natsorted(groups, key=lambda gid: groups[gid][0]['patient_id'])]


class PatientGraphs(Dataset):
    def __init__(self, assets, groups):
        self.assets, self.groups = assets, groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.assets.patient(self.groups[i])


def native_c_index(times, events, risks):
    tree = load_definitions('lifelines_metric', 'lifelines/utils/btree.py', ['_BTree'], {'np': np})._BTree
    metric = load_definitions('lifelines_metric', 'lifelines/utils/concordance.py',
        ['concordance_index', '_preprocess_scoring_data', '_concordance_summary_statistics',
         '_handle_pairs', '_concordance_ratio'], {'np': np, '_BTree': tree}).concordance_index
    try:
        return metric(np.asarray(times), -np.asarray(risks), np.asarray(events))
    except ZeroDivisionError:
        return None


class GATMamba(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='gatmamba_source', backend='torch_reference', batch_size=16,
                 uni_hidden=64, positional_embedding_size=16, gnn_dropout=.1, mlp_dropout=.3,
                 num_heads=1, num_model_layers=1, weight_regu=1e-5):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'survival' or bundle.gatmamba is None:
            raise ValueError('GAT-Mamba requires its survival profile and physical/subtype graph manifest')
        if not isinstance(bundle.config.get('time_unit'), str) or not bundle.config['time_unit'].strip():
            raise ValueError('GAT-Mamba requires actual clinical time units')
        if (any(type(v) is not int or v < 1 for v in [batch_size, uni_hidden, positional_embedding_size,
                                                     num_heads, num_model_layers]) or
                positional_embedding_size % 4 or num_heads != 1):
            raise ValueError('GAT-Mamba requires positive dimensions, PE divisible by4 and original single-head channel wiring')
        if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or
                not 0 <= v < 1 for v in [gnn_dropout, mlp_dropout]) or
                isinstance(weight_regu, bool) or not isinstance(weight_regu, (float, int)) or
                not math.isfinite(weight_regu) or weight_regu < 0):
            raise ValueError('Invalid GAT-Mamba dropout or all-parameter L1 coefficient')
        edge_conditioned = profile == 'gatmamba_edge_conditioned'
        if not edge_conditioned and any(e['nodes'] < 10 for e in bundle.gatmamba.meta['slides'].values()):
            raise ValueError('Original GAT-Mamba explicitly filters slides with fewer than10tiles; filter the task explicitly or select the named small-graph correction')
        self.bundle, self.assets, self.profile = bundle, bundle.gatmamba, profile
        self.batch_size, self.weight_regu = batch_size, float(weight_regu)
        factory, self.backend = network_class(bundle.dim, edge_conditioned, backend)
        self.network = factory(uni_hidden, positional_embedding_size, gnn_dropout,
                               mlp_dropout, num_heads, num_model_layers)
        if edge_conditioned:
            # PyG's declared batch-normalization option uses stored statistics
            # for a singleton; no fake node/duplicate graph is added.
            from torch_geometric.nn import BatchNorm
            for module in self.network.modules():
                if isinstance(module, BatchNorm):
                    module.allow_single_element = True
        self.author = source_trainer()
        self.settings = {'profile': profile, 'feature_dimension': bundle.dim, 'backend': copy.deepcopy(self.backend),
            'batch_size': batch_size, 'uni_hidden': uni_hidden, 'positional_embedding_size': positional_embedding_size,
            'gnn_dropout': gnn_dropout, 'mlp_dropout': mlp_dropout, 'num_heads': num_heads,
            'num_model_layers': num_model_layers, 'weight_regu': weight_regu}
        self.contract = digest_object({'source': COMMIT, 'bundle': bundle.fingerprint,
                                       'assets': self.assets.contract, 'settings': self.settings})
        self.info = {row['global_patient_id']: row for row in bundle.rows.to_dict('records')}
        self.completed_epochs = self.optimizer_steps = self.early_wait = 0
        self.early_best, self.training_done = None, False
        self.maximum_epochs, self.patience = 201, 5
        self.scheduler, self.pending_scheduler = None, None

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'gatmamba_val_loss':
            raise ValueError('GAT-Mamba preserves complete frozen patient graphs, has no source gradient clip and selects native validation batch loss')
        self.maximum_epochs, self.patience = cfg['epochs'], cfg['patience']

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        options = {'patience': self.patience}
        if 'verbose' in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
            options['verbose'] = True
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
        if self.pending_scheduler is not None:
            self.scheduler.load_state_dict(self.pending_scheduler)
            self.pending_scheduler = None
        return optimizer

    def forward(self, graphs):
        if graphs.x.ndim != 2 or graphs.x.shape[1] != self.bundle.dim+2 or not torch.isfinite(graphs.x).all():
            raise ValueError('GAT-Mamba requires real xy followed by the full registered FM features')
        risk, embedding = self.network(graphs)
        if not torch.isfinite(risk).all() or not torch.isfinite(embedding).all():
            raise FloatingPointError('Nonfinite GAT-Mamba risk/embedding')
        return risk, embedding

    def objective(self, graphs, risk):
        # Original source mean across patients, including censored terms; no
        # replacement event-only normalization or invented Cox minibatches.
        cox = self.author.CoxLoss(None, graphs.days, graphs.event, risk, risk.device)
        l1 = self.author.regularize_weights(None, self.network)
        return cox+self.weight_regu*l1

    def epoch(self, bundle, optimizer, device='cpu'):
        self.train()
        loader = DataLoader(PatientGraphs(self.assets, patients(bundle, 'train')),
                            batch_size=self.batch_size, shuffle=True, num_workers=0)
        losses = []
        for graph in loader:
            graph = graph.to(device)
            optimizer.zero_grad()
            risk, _ = self(graph)
            loss = self.objective(graph, risk.squeeze(1))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite original GAT-Mamba Cox plus all-parameter L1')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise FloatingPointError('Nonfinite GAT-Mamba gradient')
            optimizer.step();self.optimizer_steps += 1
            losses.append({'loss': float(loss.detach()), 'source_graph_weighted_loss': float(loss.detach())*graph.num_graphs})
        if not losses:
            raise ValueError('GAT-Mamba has no TRAIN patient graph batches')
        return losses

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        self.eval()
        # The original validation loader shuffles, changing batch-local risk
        # sets and the selected loss. TEST stays in frozen natural ID order.
        loader = DataLoader(PatientGraphs(self.assets, patients(bundle, subset)),
            batch_size=self.batch_size, shuffle=subset == 'val', num_workers=0)
        results, loss_sum, batches = [], 0., 0
        for graph in loader:
            graph = graph.to(device)
            risk, _ = self(graph)
            loss = self.objective(graph, risk.squeeze(1))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite GAT-Mamba validation objective')
            loss_sum += float(loss)*graph.num_graphs;batches += 1
            for i, gid in enumerate(graph.global_patient_id):
                row = self.info[gid]
                results.append({'slide_id': graph.slide_id[i], 'patient_id': row['patient_id'],
                    'global_patient_id': gid, 'source_slides': graph.source_slides[i], 'subset': subset,
                    'time': row['time'], 'event': row['event'], 'native_time': float(graph.days[i]),
                    'risk': float(risk[i, 0])})
        if not batches:
            raise ValueError('GAT-Mamba evaluation subset is empty')
        frame = pd.DataFrame(results)
        frame.attrs['gatmamba_native_batch_loss'] = loss_sum/batches
        return frame

    def validation_metrics(self, frame):
        value = frame.attrs['gatmamba_native_batch_loss']
        ci = native_c_index(frame.native_time, frame.event, frame.risk)
        result = {'gatmamba_native_batch_loss': value, 'gatmamba_native_c_index': ci}
        if set(frame.subset) == {'val'}:
            if ci is None:
                raise ValueError('Original GAT-Mamba validation C-index is undefined; provide a valid clinical split')
            result['gatmamba_val_loss'] = value
        return result

    def after_validation(self, metrics):
        value = metrics['gatmamba_val_loss']
        if not math.isfinite(value):
            raise ValueError('GAT-Mamba validation loss must be finite')
        self.scheduler.step(value)
        if self.early_best is None or value <= self.early_best:
            self.early_best, self.early_wait = value, 0
        else:
            self.early_wait += 1
        self.training_done = self.early_wait > self.patience

    def selection_objective(self, metrics, epoch, default):
        return -metrics['gatmamba_val_loss']

    def selection_improved(self, objective, best):
        return objective >= best

    def get_extra_state(self):
        return {'contract': self.contract, 'completed_epochs': self.completed_epochs,
            'optimizer_steps': self.optimizer_steps, 'early_best': self.early_best,
            'early_wait': self.early_wait, 'training_done': self.training_done,
            'maximum_epochs': self.maximum_epochs, 'patience': self.patience,
            'scheduler': self.scheduler.state_dict() if self.scheduler else self.pending_scheduler}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self.get_extra_state()) or state['contract'] != self.contract:
            raise ValueError('GAT-Mamba checkpoint source/assets/backend/configuration contract differs')
        if (any(type(state[k]) is not int or state[k] < 0 for k in
                ['completed_epochs', 'optimizer_steps', 'early_wait']) or
                any(type(state[k]) is not int or state[k] < 1 for k in ['maximum_epochs', 'patience']) or
                state['completed_epochs'] > state['maximum_epochs'] or type(state['training_done']) is not bool or
                (state['early_best'] is not None and (isinstance(state['early_best'], bool) or
                 not isinstance(state['early_best'], (float, int)) or not math.isfinite(state['early_best'])))):
            raise ValueError('GAT-Mamba checkpoint counters/validation state are invalid')
        scheduler = state['scheduler']
        if scheduler is None:
            if state['completed_epochs']:
                raise ValueError('GAT-Mamba trained epochs require their plateau scheduler state')
        elif (not isinstance(scheduler, dict) or scheduler.get('mode') != 'min' or scheduler.get('factor') != .1 or
                scheduler.get('patience') != state['patience'] or scheduler.get('last_epoch') != state['completed_epochs'] or
                type(scheduler.get('num_bad_epochs')) is not int or scheduler['num_bad_epochs'] < 0 or
                not isinstance(scheduler.get('_last_lr'), list) or len(scheduler['_last_lr']) != 1 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                    for v in scheduler['_last_lr'])):
            raise ValueError('GAT-Mamba plateau scheduler does not match the completed source recipe')
        for key in state:
            if key not in {'contract', 'scheduler'}:
                setattr(self, key, state[key])
        if self.scheduler is None:
            self.pending_scheduler = copy.deepcopy(scheduler)
        else:
            self.scheduler.load_state_dict(scheduler)

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'settings': self.settings,
            'architecture': 'complete original coordinate PE, GAT plus actual same-version Mamba, residual MLP/BatchNorm, global mean pooling and scalar-risk MLP',
            'input_port': 'source1024 UNI constants parameterized only in input projection/slice to use all registered FM channels; unchanged hidden/state architecture',
            'extra_information': self.assets.meta['schema'], 'graph_preparation': self.assets.meta['graph_builder'],
            'edge_conditioning': 'explicit edge_dim16 wiring and singleton normalization correction' if self.profile.endswith('edge_conditioned') else 'literal source GATConv with edge_dim=None; semantic/continuous edge encoders have no data gradients under declared PyG, only all-parameter L1',
            'patient_bags': 'naturally sorted patient IDs; slides sorted by source ID; real y/x row-major per WSI, disjoint spatial edges and original global within-patient Mamba sequence',
            'clinical_times': 'original float32 Torch times for native risk-set loss/C-index; original clinical table precision retained for standardized reporting',
            'recipe': 'Adam5e-5/wd1e-4, batch16 including short tails, mean source Cox plus1e-5 all-parameterL1, no clip; TRAIN and VAL shuffled; source ReduceLROnPlateau min/patience5',
            'selection': 'graph-weighted batch loss divided by number of VAL batches; last <= tie saved, >patience early stop, TEST sealed',
            'maximum_epochs': self.maximum_epochs, 'source_inclusive_epoch_argument': self.maximum_epochs-1,
            'completed_epochs': self.completed_epochs, 'optimizer_steps': self.optimizer_steps,
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

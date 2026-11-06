"""Full C²MIL: adaptive bias removal, causal graph views and five joint losses."""
import contextlib
import copy
import io
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .c2mil_source import COMMIT, modules
from .io import digest_object
from .upstream import load_definitions


def native_c_index(times, events, risks):
    tree = load_definitions('lifelines_metric', 'lifelines/utils/btree.py', ['_BTree'], {'np': np})._BTree
    source = load_definitions('lifelines_metric', 'lifelines/utils/concordance.py',
        ['concordance_index', '_preprocess_scoring_data', '_concordance_summary_statistics',
         '_handle_pairs', '_concordance_ratio'], {'np': np, '_BTree': tree})
    try:
        return source.concordance_index(np.asarray(times), -np.asarray(risks), np.asarray(events))
    except ZeroDivisionError:
        return None


def source_batch_boundaries(n, batch_size):
    """Literal author rule: merge the last full batch and any remainder."""
    return [i+1 for i in range(n) if ((i+1) % batch_size == 0 and
            (i+1)//batch_size != n//batch_size) or i == n-1]


class C2MIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True
    minimum_selection_epoch = 2

    def __init__(self, bundle, profile='c2mil_source', batch_size=16, hidden=128, dropout=.25):
        super().__init__()
        if profile != 'c2mil_source' or bundle.task_type != 'survival' or bundle.c2mil is None:
            raise ValueError('C²MIL requires its source survival profile and graph/thumbnail/TRAIN reference assets')
        if not isinstance(bundle.config.get('time_unit'), str) or not bundle.config['time_unit'].strip():
            raise ValueError('C²MIL requires the actual clinical time unit')
        if (any(type(v) is not int or v < 1 for v in [batch_size, hidden]) or
                isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or
                not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError('Invalid C²MIL batch size, graph hidden dimension or dropout')
        self.bundle, self.assets, self.profile = bundle, bundle.c2mil, profile
        self.batch_size, self.hidden, self.dropout = batch_size, hidden, float(dropout)
        if len(self.assets.order['train']) < max(6, batch_size):
            raise ValueError('C²MIL requires at least one full TRAIN batch and six adaptive-clustering references')
        thumb_all, patch_all = self.assets.reference_population()
        author, self.objectives = modules()
        # The author clustering constructor resets global NumPy/Torch seeds to
        #988 after creating its logits. Preserve this actual initialization.
        self.network = author.GraphCausal2_Surv(thumb_all, patch_all, K=0,
            num_features=bundle.dim, hidden_dim=hidden, dropout=dropout, n_classes=1)
        bias = self.network.biasremoval
        for name in ['thumb_all', 'patch_all']:
            value = getattr(bias, name)
            delattr(bias, name)
            # Assets are input-bound constants, moved with the model; not copied
            # into every checkpoint. All trainable/dynamic states are retained.
            bias.register_buffer(name, value, persistent=False)
        self.training_order = copy.deepcopy(self.assets.order['train'])
        self.completed_epochs = self.optimizer_steps = 0
        self.initial_lr, self.current_lr = None, None
        self.settings = {'profile': profile, 'patch_dimension': bundle.dim, 'thumbnail_dimension': bundle.dim,
            'hidden': hidden, 'gcn_layers': 4, 'thumbnail_hidden': 512, 'cluster_dimension': 1024,
            'batch_size': batch_size, 'dropout': self.dropout, 'cluster_maximum': 6,
            'cluster_max_iterations': 400, 'cluster_seed': 988, 'reference_patches': 1500}
        self.contract = digest_object({'source': COMMIT, 'bundle': bundle.fingerprint,
            'assets': self.assets.contract, 'settings': self.settings})

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None or
                cfg['selection_metric'] != 'c2mil_val_c_index' or cfg['epochs'] < 3):
            raise ValueError('C²MIL preserves full graphs, five losses, no source gradient clip and VAL C-index selection from epoch3')

    def make_optimizer(self, lr, weight_decay):
        self.initial_lr, self.current_lr = float(lr), float(lr)
        return torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, graph, thumbnail):
        if (graph.x.ndim != 2 or graph.x.shape[1] != self.bundle.dim or
                thumbnail.shape != (1, self.bundle.dim) or not torch.isfinite(graph.x).all() or
                not torch.isfinite(thumbnail).all()):
            raise ValueError('C²MIL requires actual full-dimensional graph and whole-slide thumbnail features')
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.network(graph, thumbnail)
        if len(result) != 8 or any(not torch.isfinite(v).all() for v in result):
            raise FloatingPointError('Nonfinite original C²MIL cluster, graph or bounded risk output')
        if result[0].shape != (1, 1) or result[2].shape != (1, 1):
            raise ValueError('C²MIL source outputs exactly one bounded full/causal risk per patient')
        return result

    def joint_objective(self, outputs, times, events):
        hazards, causal = [torch.cat([o[i].squeeze(0) for o in outputs]) for i in [0, 2]]
        full_h, causal_h, remaining_h = [torch.stack([o[i] for o in outputs]) for i in [1, 3, 4]]
        with contextlib.redirect_stdout(io.StringIO()):
            full_loss = self.objectives.CoxSurvLoss(hazards, times, events)
            causal_loss = self.objectives.CoxSurvLoss(causal, times, events)
            contrastive = self.objectives.causal_contrastive_loss(full_h, causal_h, remaining_h)
        # Even a merged final batch uses the nominal source batch_size divisor.
        ratio = sum(o[5]**2 for o in outputs)/self.batch_size
        alignment = F.mse_loss(torch.cat([o[7] for o in outputs]), torch.cat([o[6] for o in outputs]))
        loss = full_loss+causal_loss+.1*contrastive+.1*ratio+.1*alignment
        return loss, {'loss': loss, 'full_cox': full_loss, 'causal_cox': causal_loss,
            'causal_contrastive': contrastive, 'causal_ratio': ratio, 'old_new_alignment': alignment}

    def epoch(self, bundle, optimizer, device='cpu'):
        self.train()
        self.objectives.adjust_learning_rate(optimizer, self.initial_lr, self.completed_epochs, 40, .5)
        self.current_lr = optimizer.param_groups[0]['lr']
        # Retain and checkpoint the mutated list, not canonicalize every epoch.
        random.shuffle(self.training_order)
        boundaries = set(source_batch_boundaries(len(self.training_order), self.batch_size))
        outputs, times, events, records = [], [], [], []
        for i, gid in enumerate(self.training_order):
            row = self.assets.by_patient[gid]
            graph, thumb, _ = self.assets.load(row)
            outputs.append(self(graph.to(device), thumb.to(device)))
            times.append(row['time']);events.append(row['event'])
            if i+1 in boundaries:
                optimizer.zero_grad()
                loss, terms = self.joint_objective(outputs, times, events)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite original C²MIL joint objective')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                    raise FloatingPointError('Nonfinite original C²MIL gradient')
                optimizer.step();self.optimizer_steps += 1
                records.append({k: float(v.detach()) for k, v in terms.items()})
                outputs, times, events = [], [], []
        return records

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        self.eval()
        rows, full_risk = [], []
        for gid in self.assets.order[subset]:
            row = self.assets.by_patient[gid]
            graph, thumbnail, _ = self.assets.load(row)
            out = self(graph.to(device), thumbnail.to(device))
            full_risk.append(out[0].squeeze(0))
            rows.append({'slide_id': row['slide_id'], 'patient_id': row['patient_id'],
                'global_patient_id': gid, 'source_slides': row['slide_id'], 'subset': subset,
                'time': row['time'], 'event': row['event'], 'risk': float(out[2][0, 0]),
                'full_risk': float(out[0][0, 0])})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError('C²MIL prediction subset is empty')
        loss = self.objectives.CoxSurvLoss(torch.cat(full_risk), frame.time.tolist(), frame.event.tolist())
        frame.attrs['c2mil_native_full_cox'] = float(loss)
        return frame

    def validation_metrics(self, frame):
        ci = native_c_index(frame.time, frame.event, frame.risk)
        result = {'c2mil_native_full_cox': frame.attrs['c2mil_native_full_cox'], 'c2mil_native_c_index': ci}
        if set(frame.subset) == {'val'}:
            if ci is None:
                raise ValueError('Original C²MIL VAL C-index is undefined for this clinical split')
            result['c2mil_val_c_index'] = ci
        return result

    def selection_improved(self, objective, best):
        return objective >= max(0., best)

    def get_extra_state(self):
        bias = self.network.biasremoval
        centers = bias.cluster.saved_cluster_centers
        biases = getattr(bias, 'biases', None)
        return {'contract': self.contract, 'completed_epochs': self.completed_epochs,
            'optimizer_steps': self.optimizer_steps, 'training_order': copy.deepcopy(self.training_order),
            'initial_lr': self.initial_lr, 'current_lr': self.current_lr,
            'centers': centers.detach().clone() if centers is not None else None,
            'biases': biases.detach().clone() if biases is not None else None,
            'best_k': getattr(bias.cluster, 'best_k', None)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self.get_extra_state()) or state.get('contract') != self.contract:
            raise ValueError('C²MIL checkpoint source/inputs/configuration contract differs')
        if (any(type(state[k]) is not int or state[k] < 0 for k in ['completed_epochs', 'optimizer_steps']) or
                not isinstance(state['training_order'], list) or
                len(state['training_order']) != len(self.assets.order['train']) or
                set(state['training_order']) != set(self.assets.order['train'])):
            raise ValueError('C²MIL checkpoint counters or persistent TRAIN order are malformed')
        for k in ['initial_lr', 'current_lr']:
            if state[k] is not None and (isinstance(state[k], bool) or not isinstance(state[k], (float, int)) or
                    not math.isfinite(state[k]) or state[k] <= 0):
                raise ValueError('C²MIL checkpoint learning rates are malformed')
        centers, biases, k = state['centers'], state['biases'], state['best_k']
        if centers is None:
            if biases is not None or k is not None or state['optimizer_steps']:
                raise ValueError('C²MIL trained checkpoint lost its dynamic cluster/bias state')
        elif (type(k) is not int or not 1 <= k <= 6 or not isinstance(centers, torch.Tensor) or
                not isinstance(biases, torch.Tensor) or centers.shape != (k, 1024) or
                biases.shape != (k, self.bundle.dim) or not centers.is_floating_point() or
                not biases.is_floating_point() or not torch.isfinite(centers).all() or
                not torch.isfinite(biases).all()):
            raise ValueError('C²MIL dynamic cluster centers or FM-space biases are malformed')
        self.completed_epochs, self.optimizer_steps = state['completed_epochs'], state['optimizer_steps']
        self.training_order = copy.deepcopy(state['training_order'])
        self.initial_lr, self.current_lr = state['initial_lr'], state['current_lr']
        bias, device = self.network.biasremoval, next(self.network.parameters()).device
        bias.cluster.saved_cluster_centers = centers.detach().clone().to(device) if centers is not None else None
        if centers is not None:
            bias.cluster_centers = bias.cluster.saved_cluster_centers
            bias.biases, bias.K, bias.cluster.best_k = biases.detach().clone().to(device), k, k
        else:
            for owner, name in [(bias, 'cluster_centers'), (bias, 'biases'), (bias.cluster, 'best_k')]:
                if hasattr(owner, name):delattr(owner, name)

    def protocol(self):
        cache = self.network.biasremoval.cluster.saved_cluster_centers
        return {'source_commit': COMMIT, 'settings': self.settings, 'assets_sha256': self.assets.contract,
            'completed_epochs': self.completed_epochs, 'optimizer_steps': self.optimizer_steps,
            'current_lr': self.current_lr, 'dynamic_clusters': None if cache is None else len(cache),
            'TRAIN_reference_order': self.assets.order['train'], 'current_TRAIN_order': self.training_order,
            'selection': 'last-tie VAL causal-risk lifelines C-index from third epoch, initial best0',
            'schedule': 'Adam; initial lr1e-4, weight_decay5e-4; lr halved before epochs41/81;100epochs, no early stop',
            'objective': 'full Cox+causal Cox+.1*cosine contrastive+.1*sum(ratio²)/nominal batch+.1*MSE(z_new,z_old)',
            'source_behaviors': ['sigmoid-bounded risk passed to Cox', 'TRAIN Bernoulli straight-through causal masks',
                'VAL positive sigmoid masks select all nodes for causal edges; remaining features still weighted',
                'adaptive cluster constructor resets NumPy/Torch seed988', 'max400 differentiable clustering iterations',
                'native query/einsum and unused Subgraph initialization retained', 'final full TRAIN batch merged with remainder',
                'cluster/bias cache uses last TRAIN forward before last Adam update, not recomputed for VAL'],
            'integration_changes': ['private import namespace; unused scatter/timm/rotary/plot imports omitted',
                'Cox tensors use hazard device instead of unconditional CUDA/global device',
                'explicit selected diagnostic WSI and actual whole-slide thumbnail per patient',
                'all FM dimensions retained, no fixed1024 truncation',
                'input-bound reference constants are nonpersistent buffers moved with model',
                'complete dynamic cache, mutated patient list, optimizer and RNG saved for exact epoch resume',
                'final source weight-only overwrite avoided; TEST sealed until explicit evaluate'],
            'reference_sampling': self.assets.meta['sampling'], 'synthetic': bool(self.bundle.config.get('synthetic', False))}

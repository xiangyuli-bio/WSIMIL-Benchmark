"""Pinned ProtoSurv network, three objectives and complete source survival recipe."""
import contextlib
import copy
import io
import math
import random
import sys
import types

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch

from .io import digest_object
from .protosurv_data import COMMIT
from .upstream import load_definitions, package, pinned_text, verified_source

PROFILES = {'protosurv_source', 'protosurv_repaired'}


def source_modules(repaired=False):
    root, sha = verified_source('protosurv')
    if sha != COMMIT:
        raise ValueError('ProtoSurv source differs from the reviewed version')
    name = '_milplus_protosurv_'+sha+('_repaired' if repaired else '_source')
    if name+'.network' in sys.modules:
        return sys.modules[name+'.network'], sys.modules[name+'.loss'], sys.modules[name+'.data']
    package(name, root)
    for key, filename in [('model_utils', 'models/utils/model_utils.py'),
                          ('proto_utils', 'models/utils/proto_utils_v1.py'),
                          ('network', 'models/model_protosurv_v1.py'), ('loss', 'utils/utils_loss.py')]:
        text = pinned_text(root, sha, filename)
        text = text.replace('from models.utils.model_utils import *', 'from '+name+'.model_utils import *')
        text = text.replace('from models.utils.proto_utils_v1 import ProtoFusion, ProtoGNN',
                            'from '+name+'.proto_utils import ProtoFusion, ProtoGNN')
        if repaired and key == 'model_utils':
            old = '        if seed is not None:\n            torch.manual_seed(seed + i)'
            new = '        generator = torch.Generator().manual_seed(seed + i) if seed is not None else None'
            if text.count(old) != 1 or text.count('random_indices = torch.randperm(num_node)') != 1:
                raise ValueError('ProtoSurv local-generator repair no longer matches the source')
            text = text.replace(old, new).replace('random_indices = torch.randperm(num_node)',
                                                'random_indices = torch.randperm(num_node, generator=generator)')
        if repaired and key == 'loss':
            if text.count('F.one_hot(labels)') != 1:
                raise ValueError('ProtoSurv missing-class repair no longer matches')
            text = text.replace('F.one_hot(labels)', 'F.one_hot(labels, num_classes=num_classes)')
        module = types.ModuleType(name+'.'+key)
        module.__file__, module.__package__ = str(root/filename), name
        sys.modules[module.__name__] = module
        exec(compile(text, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- pinned Git objects, explicit import/repair boundary
    data = load_definitions('protosurv', 'datasets/wsi_dataset.py', ['data_transform'],
                            {'torch': torch, 'random': random, 'np': np})
    sampler = load_definitions('protosurv', 'utils/utils.py', ['Sampler_custom'],
                               {'torch': torch, 'np': np, 'random': random, 'copy': copy, 'Sampler': Sampler})
    data.Sampler_custom = sampler.Sampler_custom
    sys.modules[name+'.data'] = data
    return sys.modules[name+'.network'], sys.modules[name+'.loss'], data


def source_c_index(times, events, risks):
    """Exact1-minus-lifelines pair semantics, including observed/censored time ties."""
    time, event, risk = [np.asarray(v) for v in [times, events, risks]]
    if (time.ndim != 1 or event.shape != time.shape or risk.shape != time.shape or
            not all(np.isfinite(v).all() for v in [time, event, risk]) or not set(event) <= {0, 1}):
        raise ValueError('ProtoSurv C-index inputs must be finite aligned vectors and binary events')
    correct = tied = pairs = 0
    for i in range(len(time)):
        eligible = ((time > time[i]) | ((time == time[i]) & (event == 0))) & bool(event[i])
        pairs += int(eligible.sum())
        correct += int((risk[i] < risk[eligible]).sum())
        tied += int((risk[i] == risk[eligible]).sum())
    return None if not pairs else 1-(correct+.5*tied)/pairs


class GraphRows(Dataset):
    def __init__(self, assets, rows, augmentation=None):
        self.assets, self.rows, self.augmentation = assets, rows, augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        graph = self.assets.load(self.rows[i])
        if self.augmentation is not None:
            graph = self.augmentation(graph)
            # Source graphs also contain an unused latent graph. Its source
            # augmentation consumes one Python draw and one NumPy subset draw.
            # Replay precisely those draws using its original edge count,
            # without inventing latent edges or running the defective builder.
            rate = random.uniform(.1, .3)
            m = len(graph.x)*min(4, len(graph.x)-1)
            np.random.choice(m, m-int(rate*m), replace=False)
        return graph


def collate_graphs(rows):
    return rows


class ProtoSurv(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='protosurv_repaired', batch_size=6, accumulation_steps=2,
                 custom_sample=True, comp_alpha=.01, ortho_beta=.1, alpha_surv=0.,
                 pre_proto=False, hidden=128, gcn_layers=4, prototype_dimension=768,
                 prototype_hidden=512, prototypes_per_class=8, prototype_iterations=2,
                 prototype_heads=3, fusion_layers=4, fusion_heads=3, max_nodes=25000):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'survival' or bundle.protosurv is None:
            raise ValueError('ProtoSurv requires its registered survival profile and physical graph/category manifest')
        if bundle.rows.groupby('global_patient_id').size().max() != 1:
            raise ValueError('Native ProtoSurv uses one declared WSI per patient; select the source representative explicitly')
        if not isinstance(bundle.config.get('time_unit'), str) or not bundle.config['time_unit'].strip():
            raise ValueError('ProtoSurv needs the actual clinical time unit')
        self.repaired = profile.endswith('_repaired')
        if not self.repaired and not np.equal(bundle.rows.time, np.floor(bundle.rows.time)).all():
            raise ValueError('Source ProtoSurv casts clinical times to integer; fractional times require the repaired route')
        integers = [batch_size, accumulation_steps, hidden, gcn_layers, prototype_dimension,
                    prototype_hidden, prototypes_per_class, prototype_iterations, prototype_heads,
                    fusion_layers, fusion_heads, max_nodes]
        if (any(type(v) is not int or v < 1 for v in integers) or gcn_layers < 2 or
                prototype_dimension % prototype_heads or prototype_dimension % fusion_heads):
            raise ValueError('Invalid ProtoSurv native network/batch/node budgets')
        if type(custom_sample) is not bool or type(pre_proto) is not bool:
            raise ValueError('ProtoSurv sampling and prototype choices must be explicit booleans')
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in [comp_alpha, ortho_beta, alpha_surv]) or alpha_surv > 1 or
                (self.repaired and alpha_surv != 0)):
            raise ValueError('ProtoSurv objective weights must be finite/nonnegative; repaired Breslow uses event-only Cox')
        self.bundle, self.profile, self.assets = bundle, profile, bundle.protosurv
        self.batch_size, self.gc, self.custom_sample = batch_size, accumulation_steps, custom_sample
        self.comp_alpha, self.ortho_beta, self.alpha_surv = comp_alpha, ortho_beta, alpha_surv
        if self.assets.mode == 'unknown_fixed_prototype' and (not pre_proto or comp_alpha != 0):
            raise ValueError('Unknown tissue sentinels require fixed prototypes and explicitly comp_alpha=0; no fake categories')
        author, self.losses, self.data_source = source_modules(self.repaired)
        kwargs = {'input_dim': bundle.dim, 'num_layers': gcn_layers, 'hidden_dim': hidden,
                  'nr_types': len(self.assets.names), 'num_proto': prototypes_per_class,
                  'proto_iters': prototype_iterations, 'proto_n_head': prototype_heads,
                  'fusion_layer_to_p': fusion_layers, 'fusion_n_head': fusion_heads,
                  'dim_proto': prototype_dimension, 'proto_fc_hidden_dim': prototype_hidden,
                  'pre_proto': pre_proto, 'pre_proto_pth': None}
        with contextlib.redirect_stdout(io.StringIO()):
            self.network = author.LINKX_PROTO_oldv(**kwargs)
        self.network.max_input_len = max_nodes
        prototypes = self.assets.prototype()
        if pre_proto and prototypes is None:
            raise ValueError('Fixed ProtoSurv mode requires actual feature-space-matched prototype assets')
        delattr(self.network.ProtoGNN, 'pre_prototypes')
        self.network.ProtoGNN.register_buffer('pre_prototypes', prototypes)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cox = self.losses.CoxPHLoss(alpha=alpha_surv)
        settings = {'network': {k: v for k, v in kwargs.items() if k != 'pre_proto_pth'},
                    'profile': profile, 'batch_size': batch_size, 'gc': self.gc,
                    'custom_sample': custom_sample, 'comp_alpha': comp_alpha,
                    'ortho_beta': ortho_beta, 'alpha_surv': alpha_surv, 'max_nodes': max_nodes}
        self.contract = digest_object({'source': COMMIT, 'bundle': bundle.fingerprint,
                                       'assets': self.assets.contract, 'settings': settings})
        self.settings = settings
        self.completed_epochs = self.optimizer_steps = self.skipped_batches = 0
        self.early_best, self.early_wait, self.training_done = None, 0, False
        self.patience, self.scheduler, self.pending_scheduler = 15, None, None
        self.maximum_epochs = 100

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'protosurv_val_c_index':
            raise ValueError('ProtoSurv owns graph sampling, has no gradient clip and selects source-native patient C-index')
        self.patience = cfg['patience']
        self.maximum_epochs = cfg['epochs']
        if len(self.bundle.records('train')) < self.batch_size:
            raise ValueError('ProtoSurv requires at least one complete source patient batch')

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        if self.pending_scheduler is not None:
            self.scheduler.load_state_dict(self.pending_scheduler)
            self.pending_scheduler = None
        return optimizer

    def forward_graphs(self, graphs, device):
        if not graphs:
            raise ValueError('ProtoSurv cannot forward an empty patient graph batch')
        if not self.repaired and any(len(g.x) < 5 for g in graphs):
            raise ValueError('The original ProtoSurv four-neighbor graph construction requires at least5nodes')
        batched = Batch.from_data_list(graphs).to(device)
        result = self.network(batched)
        if not torch.isfinite(result[1]).all() or not torch.isfinite(result[4]).all():
            raise FloatingPointError('Nonfinite ProtoSurv risk or prototypes')
        return result

    def objective(self, graphs, result, device):
        if not self.repaired and self.comp_alpha and any(
                int(g.patch_classify_type.max()) != len(self.assets.names)-1 for g in graphs):
            raise ValueError('Released ProtoSurv compatibility one_hot is undefined without the highest tissue category')
        risk, prototypes = result[1], result[4]
        times = torch.tensor([g.event_time for g in graphs], device=device)
        c = torch.tensor([g.c for g in graphs], device=device)
        if self.repaired:
            from .recipes import cox_breslow_loss
            base = cox_breslow_loss(risk.flatten(), times, 1-c)
        else:
            base = self.cox(S=risk, c=c, event_time=times)
        comp = self.losses.get_compatibility_loss_batch(graphs, prototypes, len(self.assets.names)) if self.comp_alpha else base.new_zeros(())
        ortho = self.losses.get_orthogonal_regularization_loss(prototypes) if self.ortho_beta else base.new_zeros(())
        return base+self.comp_alpha*comp+self.ortho_beta*ortho

    def step_optimizer(self, optimizer):
        active = [p for p in self.parameters() if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in active):
            raise FloatingPointError('Nonfinite ProtoSurv gradient')
        optimizer.step()
        if active:
            self.optimizer_steps += 1
        optimizer.zero_grad()

    def epoch(self, bundle, optimizer, device='cpu'):
        self.train()
        rows = bundle.records('train')
        dataset = GraphRows(self.assets, rows, self.data_source.data_transform)
        if self.custom_sample:
            with contextlib.redirect_stdout(io.StringIO()):
                sampler = self.data_source.Sampler_custom([1-r['event'] for r in rows], self.batch_size)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_graphs)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                drop_last=True, num_workers=0, collate_fn=collate_graphs)
        losses, pending = [], 0
        for batch_idx, graphs in enumerate(loader):
            if all(g.c == 1 for g in graphs):
                self.skipped_batches += 1
                continue
            result = self.forward_graphs(graphs, device)
            loss = self.objective(graphs, result, device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite ProtoSurv source joint objective')
            (loss/self.gc).backward()
            value = float(loss.detach())
            losses.append({'loss': value, 'source_joint_loss_per_patient': value/len(graphs),
                           'source_gc_scaled_joint_loss_per_patient': value/(self.gc*len(graphs))})
            pending += 1
            if (pending if self.repaired else batch_idx+1) % self.gc == 0:
                self.step_optimizer(optimizer)
                pending = 0
        if not losses:
            raise ValueError('ProtoSurv source sampler produced no complete observed-event training batch')
        if self.repaired:
            # The source uses a fixed1/gc scale even on a partial tail. The
            # corrected route gives the actual tail its own average weight.
            if pending:
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.mul_(self.gc/pending)
                self.step_optimizer(optimizer)
        else:
            # Literal source unconditional tail call, including its no-op after
            # a complete accumulation under Torch2.5's set_to_none default.
            self.step_optimizer(optimizer)
        return losses

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1
        self.scheduler.step()

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        self.eval()
        rows = bundle.records(subset)
        loader = DataLoader(GraphRows(self.assets, rows), batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_graphs)
        result = []
        for row, graphs in zip(rows, loader):
            outputs = self.forward_graphs(graphs, device)
            result.append({'slide_id': row['slide_id'], 'patient_id': row['patient_id'],
                'global_patient_id': row['global_patient_id'], 'subset': subset,
                'source_slides': row['slide_id'], 'time': row['time'], 'event': row['event'],
                'risk': float(outputs[1][0, 0])})
        return pd.DataFrame(result)

    def validation_metrics(self, frame):
        return {'protosurv_val_c_index': source_c_index(frame.time, frame.event, frame.risk),
                'protosurv_native_c_index_all': source_c_index(frame.time, np.ones(len(frame)), frame.risk)}

    def after_validation(self, metrics):
        value = metrics['protosurv_val_c_index']
        if value is None or not math.isfinite(value):
            raise ValueError('ProtoSurv source-native validation C-index is undefined')
        if self.early_best is None or value > self.early_best:
            self.early_best, self.early_wait = value, 0
        else:
            self.early_wait += 1
        self.training_done = self.early_wait >= self.patience and self.completed_epochs > 20

    def get_extra_state(self):
        return {'contract': self.contract, 'completed_epochs': self.completed_epochs,
                'optimizer_steps': self.optimizer_steps, 'skipped_batches': self.skipped_batches,
                'early_best': self.early_best, 'early_wait': self.early_wait,
                'training_done': self.training_done, 'maximum_epochs': self.maximum_epochs,
                'patience': self.patience,
                'scheduler': self.scheduler.state_dict() if self.scheduler else self.pending_scheduler}

    def set_extra_state(self, state):
        if set(state) != set(self.get_extra_state()) or state['contract'] != self.contract:
            raise ValueError('ProtoSurv source/assets/configuration checkpoint contract differs')
        if (any(type(state[k]) is not int or state[k] < 0
                for k in ['completed_epochs', 'optimizer_steps', 'skipped_batches', 'early_wait']) or
                any(type(state[k]) is not int or state[k] < 1 for k in ['maximum_epochs', 'patience']) or
                state['completed_epochs'] > state['maximum_epochs'] or
                type(state['training_done']) is not bool or
                (state['early_best'] is not None and (isinstance(state['early_best'], bool) or
                 not isinstance(state['early_best'], (float, int)) or not 0 <= state['early_best'] <= 1))):
            raise ValueError('ProtoSurv checkpoint counters or selection state invalid')
        scheduler = state['scheduler']
        if scheduler is None:
            if state['completed_epochs']:
                raise ValueError('ProtoSurv completed epochs require a scheduler checkpoint')
        else:
            if (not isinstance(scheduler, dict) or scheduler.get('T_max') != 5 or
                    scheduler.get('eta_min') != 1e-6 or scheduler.get('last_epoch') != state['completed_epochs'] or
                    not isinstance(scheduler.get('base_lrs'), list) or len(scheduler['base_lrs']) != 1 or
                    not isinstance(scheduler.get('_last_lr'), list) or len(scheduler['_last_lr']) != 1 or
                    any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
                        for v in [*scheduler['base_lrs'], *scheduler['_last_lr']]) or
                    scheduler['base_lrs'][0] <= 0):
                raise ValueError('ProtoSurv scheduler checkpoint differs from the source recipe/epoch')
        for key, value in state.items():
            if key not in {'contract', 'scheduler'}:
                setattr(self, key, value)
        if self.scheduler is None:
            self.pending_scheduler = copy.deepcopy(state['scheduler'])
        else:
            self.scheduler.load_state_dict(state['scheduler'])

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'settings': self.settings,
            'architecture': 'complete original spatial GENConv/DeepGCN plus tissue multi-prototype attention, cross-view fusion and scalar risk head',
            'objectives': 'source sorted epsilon Cox or explicit tied-time Breslow; original global positive-distance/logsumexp compatibility plus original input-space orthogonal penalty',
            'augmentation': 'original noise, zero-node and spatial-edge dropout; original unused latent-edge augmentation RNG draws replayed without fabricating that graph',
            'recipe': 'Adam5e-5/wd1e-5, batch6 source custom event/censor sampling, gc2,100epochs, no clipping, epoch cosine T_max5/eta_min1e-6; explicit0-worker FP32 port',
            'selection': 'first strictly improved1-minus-lifelines censored-aware C-index; warmup0, patience15, stop after one-based epoch20; outerTEST sealed',
            'repairs': 'fixed one_hot vocabulary, local generator instead of global pad_batch reseeding, Breslow ties and actual partial-tail weighting' if self.repaired else 'source global manual_seed and fixed-gc/unconditional tail preserved; valid integer-time/highest-category domain',
            'preparation': self.assets.meta['graph_builder'], 'tissue_schema': self.assets.meta['schema'],
            'prototypes': self.assets.meta.get('prototype'),
            'optimizer_steps': self.optimizer_steps, 'completed_epochs': self.completed_epochs,
            'actual_maximum_epochs': self.maximum_epochs, 'actual_patience': self.patience,
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

"""Complete four-cluster AMIML training and validation-only cluster selection.

Public dataloader/working trainer are missing. Paper and released schedules have
separate names; neither profile claims historical scores. No omics are inputs.
"""
import copy
import math

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from .amiml_data import check_cluster_capacity, cluster_bags, patient_bags, train_kmeans
from .amiml_network import COMMIT, PAPER, SOURCE, forward_bags, make_network
from .io import digest_object
from .loss_attention import check_update_finite


class AMIML(nn.Module):
    owns_sampling = True
    handles_early_stopping = True
    minimum_selection_epoch = 5

    def __init__(self, bundle, profile=PAPER, R=100, alpha=.1, batch_size=64,
                 cluster_seed=2026, max_cluster_bytes=8*1024**3):
        super().__init__()
        if (profile not in {PAPER, SOURCE} or bundle.task_type != 'classification'
                or bundle.config.get('training_unit') != 'patient'):
            raise ValueError('AMIML requires patient-bag classification and a named reconstructed recipe')
        if profile == SOURCE and len(bundle.classes) != 2:
            raise ValueError('AMIML released CE/weights/AUC route is binary; use named paper C-way extension')
        if (any(type(v) is not int or v < 1 for v in [R, batch_size, max_cluster_bytes])
                or type(cluster_seed) is not int or not 0 <= cluster_seed < 2**32):
            raise ValueError('AMIML R/batch/budget must be positive integers; cluster_seed is uint32')
        self.bundle, self.profile = bundle, profile
        self.settings = {'R': R, 'alpha': alpha, 'batch_size': batch_size, 'cluster_seed': cluster_seed,
            'max_cluster_bytes': max_cluster_bytes, 'morphology_clusters': 4, 'latent': 8,
            'value_output': len(bundle.classes), 'input_dimension': bundle.dim}
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'settings': self.settings,
            'bundle': bundle.fingerprint, 'recipe': 'four_TRAIN_only_clusters_patient_bags_VAL_only_selection_v1'})
        groups = patient_bags(bundle, 'train')
        counts = torch.bincount(torch.tensor([r[0]['target'] for r in groups]), minlength=len(bundle.classes))
        if (counts == 0).any():
            raise ValueError('AMIML needs every ordered class present among TRAIN patients')
        # Original binary [positive_fraction, negative_fraction] weights are
        # proportional to inverse counts. Same normalized extension for C-way.
        weights = counts.float().reciprocal(); weights /= weights.sum()
        self.register_buffer('class_weights', weights)
        self.register_buffer('centers', torch.zeros(4, bundle.dim))
        self.register_buffer('prepared', torch.tensor(False))
        from .engine import rng_state
        # The author resets the same run seed for each independent cluster.
        # One identical initialization and four private RNG streams preserve
        # that property while scheduling the complete trainers in one process.
        initial = make_network(bundle.dim, len(bundle.classes), R, alpha)
        self.networks = nn.ModuleList([initial]+[copy.deepcopy(initial) for _ in range(3)])
        self.branch_rng = [copy.deepcopy(rng_state()) for _ in range(4)]
        self.best_networks = copy.deepcopy(self.networks)
        for p in self.best_networks.parameters(): p.requires_grad_(False)
        self.best_meta = [None]*4
        self.early_best = [None]*4
        self.bad = [0]*4; self.stopped = [False]*4
        self.cluster_fit = None
        self.completed_epochs = 0; self.optimizer_steps = [0]*4; self.train_orders = []
        self.optimizers, self.schedulers = [], []
        self.selected_cluster = None; self.total_epochs = 0; self.patience = 30

    def configure_training(self, cfg):
        if (cfg['epochs'] < 6 or cfg['grad_clip'] is not None or cfg['max_instances'] is not None
                or cfg['weight_decay'] != 0 or cfg['selection_metric'] != 'amiml_validation_generation'):
            raise ValueError('AMIML requires >=6 epochs, full bags, no source clipping/decay, and its two-level VAL selection')
        if type(cfg['patience']) is not int or cfg['patience'] < 6:
            raise ValueError('AMIML patience must allow source selection beginning at zero-based epoch5')
        self.total_epochs, self.patience = cfg['epochs'], cfg['patience']

    def prepare(self, bundle, seed, device):
        check_cluster_capacity(bundle, self.settings['R'])
        centers, meta = train_kmeans(bundle, self.settings['cluster_seed'], self.settings['max_cluster_bytes'])
        self.centers.copy_(centers.to(device)); self.cluster_fit = meta
        # Fail on any invalid declared input before any optimizer step. No
        # held-out labels or features enter fitted centers/weights.
        for subset in ['train', 'val', 'test']:
            groups = patient_bags(bundle, subset)
            if subset != 'test' and len({r[0]['target'] for r in groups}) != len(bundle.classes):
                raise ValueError('AMIML validation AUC protocol needs each class in each declared subset')
            cluster_bags(bundle, groups, centers, self.settings['R'], 0)
        self.prepared.fill_(True)

    def make_optimizers(self, lr, weight_decay):
        self.optimizers = [torch.optim.Adam(n.parameters(), lr=lr, weight_decay=weight_decay) for n in self.networks]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(o, mode='min', factor=.5,
            patience=10 if self.profile == PAPER else 5) for o in self.optimizers]
        return self.optimizers

    def objective(self, output, target):
        value = output['probabilities'] if self.profile == SOURCE else output['logits']
        return F.cross_entropy(value, target, weight=self.class_weights)

    def epoch(self, bundle, optimizers, device):
        from .engine import restore_rng, rng_state
        if not self.prepared:
            raise ValueError('AMIML morphology centers have not been fitted on TRAIN')
        groups = patient_bags(bundle, 'train'); records = []; order = []
        outer_rng = rng_state()
        for c, (network, optimizer) in enumerate(zip(self.networks, optimizers)):
            current_order = []
            if not self.stopped[c]:
                restore_rng(self.branch_rng[c])
                network.train()
                loader = DataLoader(groups, batch_size=self.settings['batch_size'], sampler=RandomSampler(groups),
                                    num_workers=0, collate_fn=list, drop_last=False)
                for batch in loader:
                    bags, _ = cluster_bags(bundle, batch, self.centers, self.settings['R'], c)
                    target = torch.tensor([g[0]['target'] for g in batch], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    result = forward_bags(network, [x.to(device) for x in bags])
                    loss = self.objective(result, target)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite AMIML weighted objective')
                    loss.backward(); check_update_finite(network, optimizer)
                    optimizer.step(); check_update_finite(network, optimizer)
                    self.optimizer_steps[c] += 1
                    current_order.extend(g[0]['global_patient_id'] for g in batch)
                    records.append({'loss': float(loss.detach()), 'cluster': c, 'lr': optimizer.param_groups[0]['lr']})
                self.branch_rng[c] = copy.deepcopy(rng_state())
            order.append(current_order)
        restore_rng(outer_rng)
        self.train_orders.append(order)
        return records

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1

    @torch.no_grad()
    def _predict_cluster(self, subset, cluster, network, device):
        network.eval(); groups = patient_bags(self.bundle, subset)
        records, losses = [], []
        for start in range(0, len(groups), self.settings['batch_size']):
            batch = groups[start:start+self.settings['batch_size']]
            bags, indices = cluster_bags(self.bundle, batch, self.centers, self.settings['R'], cluster)
            output = forward_bags(network, [x.to(device) for x in bags])
            target = torch.tensor([g[0]['target'] for g in batch], dtype=torch.long, device=device)
            losses.append(float(self.objective(output, target)))
            for rows, prob, ids in zip(batch, output['probabilities'], indices):
                row = rows[0]
                item = {k: row[k] for k in ['patient_id', 'global_patient_id', 'subset', 'target']}
                item.update(slide_id='patient:'+row['global_patient_id'], source_slides=';'.join(r['slide_id'] for r in rows),
                    amiml_cluster=cluster, amiml_cluster_patches=len(ids),
                    amiml_cluster_ids_sha256=digest_object(ids.tolist()))
                item.update({f'p_{i}': float(p) for i, p in enumerate(prob)})
                records.append(item)
        frame = pd.DataFrame(records)
        frame.attrs['amiml_mean_batch_loss'] = sum(losses)/len(losses)
        return frame

    @staticmethod
    def auc(frame, classes):
        if classes == 2: return float(roc_auc_score(frame.target, frame.p_1))
        return float(roc_auc_score(frame.target, frame[[f'p_{i}' for i in range(classes)]],
                                  multi_class='ovr', average='macro', labels=list(range(classes))))

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        self.eval()
        if subset == 'val' and self.schedulers:
            frames = [self._predict_cluster(subset, c, n, device) for c, n in enumerate(self.networks)]
            frame = frames[0].copy(); frame.attrs['amiml_all_cluster_frames'] = frames
            return frame
        if self.selected_cluster is None or self.best_meta[self.selected_cluster] is None:
            raise ValueError('AMIML TEST cannot choose its own cluster; complete VAL-only selection first')
        return self._predict_cluster(subset, self.selected_cluster, self.best_networks[self.selected_cluster], device)

    def validation_metrics(self, frame):
        if set(frame.subset) != {'val'}: return {}
        frames = frame.attrs.get('amiml_all_cluster_frames')
        if not self.schedulers:
            return {'amiml_selected_cluster': self.selected_cluster}
        if not isinstance(frames, list) or len(frames) != 4:
            raise ValueError('AMIML VAL must cover all four independently trained clusters')
        epoch = self.completed_epochs-1
        for c, current in enumerate(frames):
            if self.stopped[c]: continue
            loss = current.attrs['amiml_mean_batch_loss']
            if not math.isfinite(loss): raise ValueError('AMIML VAL loss is nonfinite')
            self.schedulers[c].step(loss)
            if self.early_best[c] is None or loss <= self.early_best[c]:
                self.early_best[c], self.bad[c] = loss, 0
            else: self.bad[c] += 1
            if epoch >= 5 and (self.best_meta[c] is None or loss <= self.best_meta[c]['loss']):
                self.best_networks[c].load_state_dict(self.networks[c].state_dict())
                self.best_meta[c] = {'epoch': epoch, 'loss': loss, 'auc': self.auc(current, len(self.bundle.classes)),
                                     'predictions': current.to_dict('records')}
            self.stopped[c] = self.bad[c] >= self.patience
        eligible = [c for c, meta in enumerate(self.best_meta) if meta is not None]
        if eligible:
            # Earliest cluster wins equal AUC; all cluster/timestep choices are
            # visible and depend exclusively on validation patient predictions.
            self.selected_cluster = max(eligible, key=lambda c: (self.best_meta[c]['auc'], -c))
            chosen = pd.DataFrame(self.best_meta[self.selected_cluster]['predictions'])
            for key in chosen: frame[key] = chosen[key].values
        from .metrics import classification_metrics
        return {**classification_metrics(frame, len(self.bundle.classes), 'patient'),
            'amiml_validation_generation': self.completed_epochs,
            'amiml_selected_cluster': self.selected_cluster,
            'amiml_best_cluster_val_auc': self.best_meta[self.selected_cluster]['auc'] if eligible else None,
            **{f'amiml_cluster_{c}_val_loss': f.attrs['amiml_mean_batch_loss'] for c, f in enumerate(frames)}}

    @property
    def training_done(self): return all(self.stopped)

    def get_extra_state(self):
        return {'contract': self.contract, 'cluster_fit': self.cluster_fit, 'best_meta': copy.deepcopy(self.best_meta),
            'early_best': self.early_best.copy(), 'bad': self.bad.copy(), 'stopped': self.stopped.copy(),
            'completed_epochs': self.completed_epochs, 'optimizer_steps': self.optimizer_steps.copy(),
            'train_orders': copy.deepcopy(self.train_orders), 'selected_cluster': self.selected_cluster,
            'total_epochs': self.total_epochs, 'patience': self.patience,
            'branch_rng': copy.deepcopy(self.branch_rng),
            'schedulers': [s.state_dict() for s in self.schedulers] if self.schedulers else
                copy.deepcopy(getattr(self, '_evaluation_schedulers', []))}

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) != set(self.get_extra_state())
                or state['contract'] != self.contract):
            raise ValueError('AMIML checkpoint source/inputs/reference recipe differ')
        if self.total_epochs and (self.total_epochs != state['total_epochs'] or self.patience != state['patience']):
            raise ValueError('AMIML training horizon/patience changed during resume')
        complete = state['completed_epochs']
        if (type(complete) is not int or complete < 0 or complete > state['total_epochs']
                or len(state['train_orders']) != complete or len(state['schedulers']) != 4):
            raise ValueError('AMIML checkpoint training counters or schedules invalid')
        allowed = {r[0]['global_patient_id'] for r in patient_bags(self.bundle, 'train')}
        steps = [0]*4
        for orders in state['train_orders']:
            if not isinstance(orders, list) or len(orders) != 4:
                raise ValueError('AMIML checkpoint per-cluster TRAIN order missing')
            for c, order in enumerate(orders):
                if order and (len(order) != len(allowed) or set(order) != allowed):
                    raise ValueError('AMIML checkpoint TRAIN patients differ')
                steps[c] += math.ceil(len(order)/self.settings['batch_size'])
        if steps != state['optimizer_steps']:
            raise ValueError('AMIML checkpoint optimizer counts inconsistent')
        for key in ['best_meta', 'early_best', 'bad', 'stopped', 'branch_rng']:
            if not isinstance(state[key], list) or len(state[key]) != 4:
                raise ValueError('AMIML checkpoint four-cluster state invalid')
        selected = state['selected_cluster']
        eligible = [c for c, m in enumerate(state['best_meta']) if m is not None]
        if eligible and selected != max(eligible, key=lambda c: (state['best_meta'][c]['auc'], -c)):
            raise ValueError('AMIML checkpoint selected cluster does not follow VAL-only rule')
        if not eligible and selected is not None: raise ValueError('AMIML checkpoint selects an untrained cluster')
        val_groups = patient_bags(self.bundle, 'val')
        val_identity = [(g[0]['global_patient_id'], g[0]['target'], ';'.join(r['slide_id'] for r in g)) for g in val_groups]
        for c, meta in enumerate(state['best_meta']):
            bad, stopped = state['bad'][c], state['stopped'][c]
            if type(bad) is not int or bad < 0 or type(stopped) is not bool or stopped != (bad >= state['patience']):
                raise ValueError('AMIML checkpoint stopping counters inconsistent')
            updates = sum(bool(orders[c]) for orders in state['train_orders'])
            scheduler = state['schedulers'][c]
            if (scheduler.get('last_epoch') != updates or scheduler.get('factor') != .5
                    or scheduler.get('patience') != (10 if self.profile == PAPER else 5)):
                raise ValueError('AMIML checkpoint plateau schedule disagrees with actual cluster epochs')
            if meta is not None:
                if (set(meta) != {'epoch', 'loss', 'auc', 'predictions'} or type(meta['epoch']) is not int
                        or not 5 <= meta['epoch'] < complete or not math.isfinite(meta['loss'])
                        or not math.isfinite(meta['auc']) or not 0 <= meta['auc'] <= 1):
                    raise ValueError('AMIML checkpoint best-cluster metadata invalid')
                frame = pd.DataFrame(meta['predictions'])
                identity = list(zip(frame.global_patient_id, frame.target, frame.source_slides))
                if identity != val_identity or set(frame.subset) != {'val'} or set(frame.amiml_cluster) != {c}:
                    raise ValueError('AMIML checkpoint selection predictions are not the fixed VAL patients')
                if not math.isclose(self.auc(frame, len(self.bundle.classes)), meta['auc'], rel_tol=1e-14):
                    raise ValueError('AMIML checkpoint selection AUC disagrees with saved VAL predictions')
        for key in ['cluster_fit', 'best_meta', 'early_best', 'bad', 'stopped', 'branch_rng', 'completed_epochs',
                    'optimizer_steps', 'train_orders', 'selected_cluster', 'total_epochs', 'patience']:
            setattr(self, key, copy.deepcopy(state[key]))
        if len(self.schedulers) != 4:
            # Evaluation builds no optimizer. Retain scheduler identity/state so
            # get_extra_state roundtrips without inventing optimizer objects.
            self._evaluation_schedulers = copy.deepcopy(state['schedulers'])
        else:
            for scheduler, saved in zip(self.schedulers, state['schedulers']): scheduler.load_state_dict(saved)

    def protocol(self):
        return {'source': 'Boomwwe/AMIML@'+COMMIT, 'profile': self.profile, 'settings': self.settings,
            'prepared': bool(self.prepared), 'cluster_fit': self.cluster_fit,
            'architecture': 'full 128/64/32/8 Conv1d+BN, top/bottom R, ConvTranspose1d decoder, alpha residual, Q/K8, V8->C, source pre-softmax attention dropout and output dropout, mean then SoftMax',
            'output_repair': 'paper says 2-D attended instance vectors; reconstruct V8->C while retaining encoder8. Public V8->8/binary CE is incompatible; no unpublished head recovered',
            'input_extension': 'full FM D->128 replaces source hardcoded256; paper actually uses fine-tuned Xception256, not PCA',
            'clustering': 'four KMeans on complete TRAIN features only, frozen held-out assignment; all four models trained independently; no TEST cluster selection',
            'ragged_minibatches': 'BN across only actual patch cells, no zero padding/repeated patches; public loader missing. Equal-length batches use exact source layout',
            'short_bags': 'reject any patient with <=10 or <R patches in any cluster; no silent cohort filtering',
            'training': 'Adam, weighted CE on logits (paper reference)' if self.profile == PAPER else 'Adam, source weighted CE on SoftMax probabilities preserved as explicit control',
            'class_weights': 'TRAIN patient inverse counts, binary matches source fractions; whole-cohort leakage removed',
            'randomness': 'same run-seed initialization per cluster and independent saved Python/NumPy/Torch/CUDA streams; interleaving does not share dropout or sampler RNG',
            'plateau_patience': 10 if self.profile == PAPER else 5, 'plateau_factor': .5,
            'early_stopping': 'source per-cluster equal-loss resets patience, default30; evaluated independently from epoch0',
            'selection': 'each cluster last-tied minimum VAL weighted mean-batch CE from epoch5; choose cluster by those checkpoints VAL macro-AUC, lowest-index tie. Container best.pt is latest complete selection generation, not pooled TEST oracle',
            'selected_cluster': self.selected_cluster, 'best_meta': [{k: v for k, v in m.items() if k != 'predictions'} if m else None for m in self.best_meta],
            'completed_epochs': self.completed_epochs, 'optimizer_steps': self.optimizer_steps,
            'scope': 'complete named paper/source-repaired reference recipes; original loader/training imports broken; no naturalWSI/CUDA/historical scores claimed'}

"""Original PMIL Shapley ranking and complete progressive, multi-round training."""
import copy
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import atomic_torch_save, digest_object, write_json
from .upstream import load_definitions

PROFILES = {f'pmil_shapley_{head}_{mode}': (head, mode)
            for head in ['abmil', 'clam_sb', 'clam_mb'] for mode in ['source', 'repaired']}
EXTRA_FIELDS = ['round_id', 'round_epoch', 'global_epoch', 'split_index', 'partitions', 'train_lengths', 'events',
                'completed_rounds', 'best_network', 'best_sha', 'best_epoch', 'best_loss', 'best_metric', 'counter',
                'early_stop', 'pending', 'training_done', 'improved', 'optimizer_steps', 'reset_count', 'current_lr']


def author_model(dim, head, classes, hidden=512, attention_dim=None, k_sample=8):
    attention_dim = (128 if head == 'abmil' else 256) if attention_dim is None else attention_dim
    patches = [('nn.Linear(1024, 512)', f'nn.Linear({dim}, {hidden})'),
               ('self.L = 512', f'self.L = {hidden}'), ('self.D = 128', f'self.D = {attention_dim}'),
               ('[1024, 512, 256]', f'[{dim}, {hidden}, {attention_dim}]'),
               ('[1024, 512, 384]', f'[{dim}, {hidden}, {attention_dim}]')]
    module = load_definitions('pmil_shapley', 'models/MIL_models.py',
        ['initialize_weights', 'ABMIL', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'CLAM_MB'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np}, patches)
    cls = {'abmil': module.ABMIL, 'clam_sb': module.CLAM_SB, 'clam_mb': module.CLAM_MB}[head]
    args = {'n_classes': classes}
    if head != 'abmil':
        args.update(k_sample=k_sample, instance_loss_fn=nn.CrossEntropyLoss())
    return cls(**args)


def author_loader(groups, shuffle):
    # Preserve the DataLoader base seed and RandomSampler draws, not only randperm.
    return DataLoader(groups, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=lambda batch: batch[0])


def tensor_tree_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()+b'\0')
        if isinstance(value, (list, tuple)):
            h.update(str(len(value)).encode()+b'\0')
            tensors = value
        else:
            tensors = [value]
        for tensor in tensors:
            tensor = tensor.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode()+b'\0')
            h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def source_metrics(logits, target, classes):
    target = np.asarray(target)
    p = logits.detach().cpu().softmax(1).numpy()
    pred = logits.detach().cpu().argmax(1).numpy()
    score = 0.
    if len(np.unique(target)) == classes:
        if classes == 2:
            fpr, tpr, _ = roc_curve(target, p[:, 1], pos_label=1)
        else:
            encoded = label_binarize(target, classes=list(range(classes)))
            fpr, tpr, _ = roc_curve(encoded.ravel(), p.ravel())
        score = float(auc(fpr, tpr))
    return {'acc': float(accuracy_score(target, pred)), 'f1': float(f1_score(target, pred, average='macro')),
            'auc': score}


@torch.no_grad()
def shapley_scores(network, head, indices, x, target, subsets=3, shuffles=2, repaired=False):
    """Author marginal-logit estimator, including its coalition order and reduction."""
    network.eval()
    indices = np.asarray(indices)
    remaining = [i for i in range(x.shape[1]) if i not in indices]
    random.shuffle(remaining)
    left = x[:, remaining]
    pieces = [left[:, i::subsets] for i in range(subsets)]
    minimum = 1 if head == 'abmil' else 2
    if not len(indices) or any(part.shape[1] < minimum for part in pieces):
        raise ValueError('PMIL Shapley requires candidates and nonempty valid residual coalitions')
    base = [network(part)[0].cpu() for part in pieces]
    contributions = torch.zeros(x.shape[1], base[-1].shape[-1])
    for index in indices:
        for j, part in enumerate(pieces):
            joined = torch.cat((x[:, index, :].unsqueeze(0), part), axis=1)
            for _ in range(shuffles):
                joined = joined[:, torch.randperm(joined.shape[1])]
                logits = network(joined)[0]
                contributions[index] = contributions[index]+logits.cpu()-base[j]
    contributions = contributions/shuffles  # Author sums residual coalitions, not their mean.
    result = contributions[indices, int(target)]
    span = result.max()-result.min()
    if span == 0:
        if not repaired:
            raise ValueError('PMIL source Shapley has zero score range; use the named repaired route')
        return torch.zeros_like(result)
    result = (result-result.min())/span
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite PMIL Shapley estimate')
    return result


class PMILShapley(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile='pmil_shapley_abmil_repaired', hidden=512, attention_dim=None,
                 rounds=10, round_epochs=200, max_pseudo=10, pseudo_step=2, search_rate=10,
                 sample_rate=3, shuffles=2, min_lr=1e-4, lr_patience=7, patience=20,
                 warmup_epoch=20, metric='f1_auc', k_sample=8, max_partition_instances=20_000_000):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification':
            raise ValueError('PMIL Shapley requires a registered classification route')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('PMIL pseudo bags preserve individual slide identities')
        self.profile, (self.head, self.variant) = profile, PROFILES[profile]
        positive = [hidden, rounds, round_epochs, max_pseudo, pseudo_step, search_rate,
                    sample_rate, shuffles, lr_patience, patience, k_sample, max_partition_instances]
        if any(type(v) is not int or v < 1 for v in positive) or (attention_dim is not None and (type(attention_dim) is not int or attention_dim < 1)):
            raise ValueError('PMIL dimensions, budgets and schedules must be positive integers')
        if rounds < 2 or max_pseudo < 2:
            raise ValueError('PMIL requires actual multi-round training and pseudo-bag augmentation')
        if type(warmup_epoch) is not int or warmup_epoch < 0 or not isinstance(min_lr, (int, float)) or isinstance(min_lr, bool) or not math.isfinite(min_lr) or min_lr <= 0:
            raise ValueError('PMIL warmup must be nonnegative and min_lr finite/positive')
        if metric not in {'acc', 'f1', 'auc', 'acc_auc', 'f1_auc', 'loss'}:
            raise ValueError('Unknown PMIL validation selection metric')
        self.bundle, self.dim, self.classes = bundle, bundle.dim, len(bundle.classes)
        self.hidden, self.attention_dim, self.k_sample = hidden, attention_dim, k_sample
        self.network = self.new_network()
        self.rounds, self.round_epochs = rounds, round_epochs
        self.schedule = ([max_pseudo] if pseudo_step >= max_pseudo else list(range(1, max_pseudo+1, pseudo_step)))
        if self.schedule[-1] != max_pseudo:
            self.schedule.append(max_pseudo)
        self.max_pseudo, self.search_rate, self.sample_rate, self.shuffles = max_pseudo, search_rate, sample_rate, shuffles
        self.min_lr, self.lr_patience, self.patience, self.warmup_epoch = min_lr, lr_patience, patience, warmup_epoch
        self.metric, self.max_partition_instances = metric, max_partition_instances
        self.round_id, self.round_epoch, self.global_epoch, self.split_index = 0, -1, -1, 0
        self.partitions, self.train_lengths, self.events, self.completed_rounds = None, {}, [], []
        self.best_network, self.best_sha, self.best_epoch = None, None, None
        self.best_loss, self.best_metric, self.counter, self.early_stop = None, None, 0, False
        self.pending, self.training_done, self.improved = None, False, False
        self.optimizer_steps, self.reset_count = 0, 0
        self.optimizer, self.output, self.current_lr = None, None, None
        self.contract = digest_object({'bundle': bundle.fingerprint, 'profile': profile,
            'configuration': [hidden, attention_dim, rounds, round_epochs, max_pseudo, pseudo_step, search_rate,
                sample_rate, shuffles, min_lr, lr_patience, patience, warmup_epoch, metric, k_sample]})

    def new_network(self):
        return author_model(self.dim, self.head, self.classes, self.hidden, self.attention_dim, self.k_sample)

    def bind_output(self, path):
        self.output = Path(path)

    def configure_training(self, cfg):
        if cfg['epochs'] != self.rounds*self.round_epochs or cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('PMIL requires complete rounds*round_epochs budget, full bags and unclipped Adam')
        if cfg['selection_metric'] != 'pmil_selection' or cfg['lr'] < self.min_lr:
            raise ValueError('PMIL uses native validation selection and max_lr >= min_lr')
        if cfg.get('patience', self.patience) != self.patience:
            raise ValueError('PMIL patience is per round; declare it in model.patience')
        if self.head != 'abmil' and cfg.get('bag_weight', .7) != .7:
            raise ValueError('PMIL author CLAM retains its .7 bag/.3 instance objective')
        self.max_lr, self.weight_decay, self.current_lr = cfg['lr'], cfg['weight_decay'], cfg['lr']
        self.stop_after_epoch = cfg['epochs']  # The method owns native, per-round early stopping.

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return self.optimizer

    def minimum_bag(self):
        return max(self.max_pseudo*(1 if self.head == 'abmil' else self.k_sample),
                   2*self.sample_rate*(1 if self.head == 'abmil' else 2))

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        total = 0
        for row in bundle.records('train'):
            x, _ = bundle.load(row)
            if len(x) < self.minimum_bag():
                raise ValueError('PMIL TRAIN slide is too small for native pseudo bags/CLAM/residual coalitions')
            self.train_lengths[row['slide_id']] = len(x)
            total += len(x)
            if total > self.max_partition_instances:
                raise ValueError('PMIL TRAIN partition budget exceeded; no hidden patch cap')
        if self.schedule[0] > 1:
            self.divide(None)  # Native missing prior checkpoint takes its random initialization path.
        self.network.reset()
        self.reset_count += 1

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) < (1 if self.head == 'abmil' else 2) or not torch.isfinite(x).all():
            raise ValueError('PMIL requires finite nonempty [instances,registered FM width], nonsingleton for CLAM')
        if label is not None and (label.shape != (1,) or label.dtype != torch.long or not 0 <= label.item() < self.classes):
            raise ValueError('PMIL label must be one in-range class index')
        if self.head == 'abmil':
            logits, attention = self.network(x[None])
            result = {'logits': logits, 'attention': attention}
            if label is not None:
                result['total_loss'] = F.cross_entropy(logits, label)
        else:
            if label is not None and len(x) < self.k_sample:
                raise ValueError('PMIL CLAM requires k_sample real instances in each pseudo bag')
            logits, attention, extra = self.network(x[None], label, instance_eval=label is not None)
            result = {'logits': logits, 'attention': attention}
            if label is not None:
                result['instance_loss'] = extra['instance_loss']
                result['total_loss'] = .7*F.cross_entropy(logits, label)+.3*extra['instance_loss']
        if not torch.isfinite(logits).all() or ('total_loss' in result and not torch.isfinite(result['total_loss'])):
            raise ValueError('Nonfinite PMIL logits or complete objective')
        return result

    @torch.no_grad()
    def divide(self, snapshot):
        from .engine import units
        if snapshot is not None:
            self.network.load_state_dict(snapshot)
        self.eval()
        partitions, target_count = {}, self.schedule[self.split_index]
        device = next(self.parameters()).device
        for rows in author_loader(units(self.bundle, 'train'), True):
            row = rows[0]
            x, _ = self.bundle.load(row)
            if len(x) != self.train_lengths[row['slide_id']]:
                raise ValueError('PMIL partition feature length changed')
            x = x[None].to(device)
            if snapshot is None:
                order = np.arange(x.shape[1]); np.random.shuffle(order)
            else:
                previous = None if self.partitions is None else self.partitions[row['slide_id']]
                branch = row['target'] if self.variant == 'repaired' and self.head == 'clam_mb' else 0
                if previous is None:
                    attention = self.network(x)[1]
                    scores = attention[branch]
                else:
                    attention = np.zeros((1, x.shape[1]))
                    for index in previous:
                        # Source shuffles the old index arrays in-place; persist the same order.
                        shuffled = index.numpy(); np.random.shuffle(shuffled)
                        kwargs = {'label': torch.tensor([row['target']], device=device), 'instance_eval': True} if self.head != 'abmil' else {}
                        sub_attention = self.network(x[:, shuffled], **kwargs)[1]
                        attention[:, shuffled] = sub_attention[branch]
                    scores = attention[0]
                order = np.argsort(-scores)
                search = int(min(target_count*self.search_rate, len(scores)/2))
                candidates = order[:search]
                values = shapley_scores(self.network, self.head, candidates, x, row['target'],
                    self.sample_rate, self.shuffles, self.variant == 'repaired')
                order = np.asarray(candidates[np.argsort(-values.numpy())].tolist()+order[search:].tolist())
            partitions[row['slide_id']] = [torch.tensor(np.asarray(order[i::target_count]).copy(), dtype=torch.long)
                                           for i in range(target_count)]
        self.partitions = partitions
        self.validate_partitions()
        self.events.append({'round': self.round_id, 'epoch': self.global_epoch, 'pseudo_count': target_count,
            'estimator': 'random_no_checkpoint' if snapshot is None else 'shap',
            'selector_sha256': None if snapshot is None else tensor_tree_sha(snapshot),
            'partition_sha256': tensor_tree_sha(partitions), 'train_slides': len(partitions)})

    def validate_partitions(self):
        if self.partitions is None:
            return
        if set(self.partitions) != set(self.train_lengths):
            raise ValueError('PMIL partitions must cover exactly TRAIN slides')
        count = self.events[-1]['pseudo_count'] if self.pending == 'divide' and self.events else self.schedule[self.split_index]
        for slide, parts in self.partitions.items():
            if len(parts) != count or any(p.dtype != torch.long or p.ndim != 1 or not len(p) for p in parts):
                raise ValueError('Invalid PMIL pseudo-bag count, shape or empty bag')
            combined = torch.cat(parts).cpu()
            if not torch.equal(combined.sort().values, torch.arange(self.train_lengths[slide])):
                raise ValueError('PMIL partitions duplicate, omit or exceed feature rows')

    def start_epoch(self, epoch):
        self.global_epoch = epoch
        if self.pending is not None:
            action, self.pending = self.pending, None
            if action == 'round':
                snapshot = self.best_network
                fresh = self.new_network()
                self.network.load_state_dict(fresh.state_dict())
                self.optimizer.state.clear()
                self.current_lr = self.max_lr
                self.optimizer.param_groups[0]['lr'] = self.current_lr
                self.round_id += 1
                self.round_epoch = -1
                self.split_index = len(self.schedule)-1
                self.divide(snapshot)
                self.network.reset(); self.reset_count += 1
                self.best_network, self.best_sha, self.best_epoch = None, None, None
                self.best_loss, self.best_metric, self.counter, self.early_stop = None, None, 0, False
            elif action == 'divide':
                self.divide(self.best_network)
            else:
                raise ValueError('Invalid PMIL pending transition')
        self.round_epoch += 1
        self.train()

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        import pandas as pd

        from .engine import units
        rows_out = []
        self.eval()
        for rows in author_loader(units(bundle, subset), False):
            row = rows[0]
            x, _ = bundle.load(row)
            result = self(x.to(device))
            p = result['logits'].softmax(1)[0]
            rows_out.append({**{k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'target']},
                'subset': subset, **{f'p_{i}': float(v) for i, v in enumerate(p)},
                **{f'pmil_logit_{i}': float(v) for i, v in enumerate(result['logits'][0])},
                'pmil_bag_ce': float(F.cross_entropy(result['logits'], torch.tensor([row['target']], device=device))),
                'pmil_round': self.round_id})
        return pd.DataFrame(rows_out)

    def validation_metrics(self, frame):
        logits = torch.tensor(frame[[f'pmil_logit_{i}' for i in range(self.classes)]].to_numpy(), dtype=torch.float32)
        metrics = source_metrics(logits, frame.target.to_numpy(), self.classes)
        loss = float(frame.pmil_bag_ce.mean())
        values = {**metrics, 'acc_auc': (metrics['acc']+metrics['auc'])/2,
                  'f1_auc': (metrics['f1']+metrics['auc'])/2, 'loss': -loss}
        return {'pmil_selection': values[self.metric], 'pmil_loss': loss,
                **{'pmil_'+key: value for key, value in metrics.items()}}

    def after_validation(self, metrics):
        loss, metric = metrics['pmil_loss'], metrics['pmil_selection']
        self.improved = self.best_loss is None or loss < self.best_loss
        if self.improved:
            self.best_loss = loss
        metric_improved = self.metric != 'loss' and (self.best_metric is None or metric >= self.best_metric)
        if metric_improved:
            self.best_metric = metric
        self.improved = self.improved or metric_improved
        if self.improved:
            self.counter = 0
            self.best_network = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            self.best_sha, self.best_epoch = tensor_tree_sha(self.best_network), self.round_epoch
        else:
            self.counter += 1
            if self.counter >= self.patience and self.round_epoch >= self.warmup_epoch:
                self.early_stop = True
        ended = (self.early_stop and self.split_index == len(self.schedule)-1) or self.round_epoch+1 == self.round_epochs
        if ended:
            self.completed_rounds.append({'round': self.round_id, 'epochs': self.round_epoch+1,
                'best_epoch': self.best_epoch, 'best_sha256': self.best_sha, 'pseudo_count': self.schedule[self.split_index]})
            if self.output is not None:
                directory = self.output/'rounds'; directory.mkdir(exist_ok=True)
                atomic_torch_save({'network': self.best_network, 'summary': self.completed_rounds[-1]}, directory/f'round_{self.round_id}.pt')
                write_json(directory/'summary.json', self.completed_rounds)
            # Native main reloads and revalidates the round best, including the last
            # round. Keep its DataLoader/RNG behavior while last.pt retains the last
            # optimized network; best.pt and round_*.pt retain selected networks.
            from .engine import predict
            current = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
            self.network.load_state_dict(self.best_network)
            self.validation_metrics(predict(self, self.bundle, self.profile, 'val', next(self.parameters()).device, {}))
            self.network.load_state_dict(current)
            self.training_done = self.round_id+1 == self.rounds
            self.pending = None if self.training_done else 'round'
            return
        counter = self.counter  # Native local counter is not rewritten by EarlyStopping.reset().
        if counter == self.lr_patience and self.current_lr == self.min_lr:
            if self.split_index < len(self.schedule)-1:
                self.counter = 0
                self.split_index += 1
            if self.schedule[self.split_index] > 1:
                self.pending = 'divide'
        if counter > 0 and counter % self.lr_patience == 0 and self.current_lr > self.min_lr:
            self.counter = 0
            self.current_lr = max(self.current_lr/10, self.min_lr)
            self.optimizer.param_groups[0]['lr'] = self.current_lr

    def selection_improved(self, objective, best):
        return self.improved  # Dual native validation criteria; final round replaces earlier-round best.

    def get_extra_state(self):
        return {'profile': self.profile, 'contract': self.contract,
                **{key: copy.deepcopy(getattr(self, key)) for key in EXTRA_FIELDS}}

    def set_extra_state(self, state):
        if set(state) != set(EXTRA_FIELDS)|{'profile', 'contract'} or state['profile'] != self.profile or state['contract'] != self.contract:
            raise ValueError('PMIL checkpoint profile/configuration/schema differs')
        if any(type(state[k]) is not int for k in ['round_id', 'round_epoch', 'global_epoch', 'split_index']) or not 0 <= state['round_id'] < self.rounds or not -1 <= state['round_epoch'] < self.round_epochs or not 0 <= state['split_index'] < len(self.schedule):
            raise ValueError('Invalid PMIL round/epoch/pseudo stage')
        if state['pending'] not in {None, 'round', 'divide'} or any(type(state[k]) is not int or state[k] < 0 for k in ['optimizer_steps', 'reset_count', 'counter']):
            raise ValueError('Invalid PMIL transition or update counters')
        expected = {r['slide_id'] for r in self.bundle.records('train')}
        if state['train_lengths'] and (set(state['train_lengths']) != expected or any(type(v) is not int or v < self.minimum_bag() for v in state['train_lengths'].values())):
            raise ValueError('PMIL TRAIN length manifest differs')
        if state['best_network'] is not None and tensor_tree_sha(state['best_network']) != state['best_sha']:
            raise ValueError('PMIL best selector snapshot hash differs')
        if state['partitions'] is not None and (not state['events'] or state['events'][-1]['partition_sha256'] != tensor_tree_sha(state['partitions'])):
            raise ValueError('PMIL partition hash differs')
        for key, value in state.items():
            if key not in {'profile', 'contract'}:
                setattr(self, key, copy.deepcopy(value))
        self.validate_partitions()

    def protocol(self):
        return {'profile': self.profile, 'source': 'RenaoYan/PMIL fixed native network and complete progressive recipe',
            'head': self.head, 'variant': self.variant, 'rounds': self.rounds, 'round_epochs': self.round_epochs,
            'round': self.round_id, 'round_epoch': self.round_epoch, 'schedule': self.schedule,
            'pseudo_count': self.schedule[self.split_index], 'search_rate': self.search_rate,
            'sample_rate': self.sample_rate, 'shuffles': self.shuffles, 'optimizer_steps': self.optimizer_steps,
            'resets': self.reset_count, 'events': self.events, 'completed_rounds': self.completed_rounds,
            'selector': 'lower validation bag CE OR >= validation chosen metric; last round best for final inference',
            'metric': self.metric, 'source_auc': 'binary ROC or flattened one-hot micro ROC, reported separately from benchmark macro AUC',
            'gradient': 'source stale-gradient carry between pseudo bags' if self.variant == 'source' else 'explicit zero_grad before each pseudo-bag update',
            'ranking': 'TRAIN Shapley marginal raw logits over random residual coalitions; previous-index shuffle retained',
            'training_meter': 'mean of optimized pseudo-bag objectives; source extra division by pseudo count in its printed meter is not used',
            'mb_ranking': 'target branch' if self.variant == 'repaired' else 'original branch0',
            'degeneracy': 'repaired constant scores become zeros' if self.variant == 'repaired' else 'zero-range source scores fail explicitly',
            'source_differences': ['sealed TEST evaluation; round-end VAL-only reload, no TEST logging during training',
                'generic FM width and explicit optional reduced model widths',
                'code defaults lr1e-3/step2/max10 differ from paper lr3e-4/step4/dataset-specific maxima',
                'complete checkpoint includes pending transitions, partitions, best selector, Adam and RNG',
                'FP32, no silent sampling or clipping'],
            'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run'}


def pmil_epoch(model, bundle, groups, optimizer, device, config):
    losses = []
    for rows in author_loader(groups, True):
        row = rows[0]
        x, _ = bundle.load(row)
        x = x.to(device)
        target = torch.tensor([row['target']], device=device)
        parts = [torch.arange(len(x))] if model.partitions is None else model.partitions[row['slide_id']]
        optimizer.zero_grad()
        for index in parts:
            if model.variant == 'repaired':
                optimizer.zero_grad()
            output = model(x[index.to(device)], label=target)
            loss = output['total_loss']
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite PMIL training gradients')
            optimizer.step()
            model.optimizer_steps += 1
            losses.append({'loss': float(loss.detach()), 'pseudo_count': len(parts), 'round': model.round_id})
    return losses

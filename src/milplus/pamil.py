"""Pinned PAMIL networks and complete clustering/warm/joint/push/last recipes.

Source-compatible CE-on-softmax and explicitly repaired CE-on-logits are
separate execution IDs. All routes retain extra TRAIN instance supervision.
"""
import math

import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .data import load_array, resolve
from .io import digest_file, digest_object
from .smmile_data import positive_int
from .upstream import load_definitions

PROFILES = {'pamil_prototype_source': 'source', 'pamil_prototype_repaired': 'repaired',
            'pamil_prototype_multilabel': 'multilabel'}


def author_network(dim, classes, prototypes, hidden=512, attention_dim=256, dropout=False,
                   variant='source', proto_weight=.5, k_sample=8):
    branch = 'PAMIL_multi_label' if variant == 'multilabel' else 'PAMIL_multi_class'
    replacements = [
        ("gate = True, size_arg = 'small'", "input_dim=1024, hidden_dim=512, attention_dim=256, gate = True, size_arg = 'small'"),
        ('size = self.size_dict[size_arg]', 'size = [input_dim, hidden_dim, attention_dim]'),
        ('if proto_path:', 'if True:'),
        ('proto_init = torch.from_numpy(np.load(proto_path))', 'proto_init = torch.zeros(n_protos, input_dim)'),
        ('proto_score = proto_score / torch.max(proto_score)',
         'proto_score = proto_score / torch.where(proto_score.max() == 0, proto_score.new_tensor(1.), proto_score.max())'),
        ('loss_inst = torch.tensor(0.)', 'loss_inst = h.new_zeros(())'),
    ]
    if variant == 'multilabel':
        replacements.append(('.cuda()', '.to(device)'))
    if variant != 'source':
        replacements.extend([
            ('torch.log(probs1)', 'F.logsigmoid(logits1)'),
            ('self.inst_eval(A, h, instance_classifier)', 'self.inst_eval(A[c:c+1], h, instance_classifier)'),
            ('self.inst_eval_out(A, h, instance_classifier)', 'self.inst_eval_out(A[c:c+1], h, instance_classifier)')])
    module = load_definitions('pamil_prototype', branch+'/models/model_protoattention.py',
        ['Attn_Net_Gated', 'KL_Divergence_Loss', 'abmil', 'PAMIL'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np}, replacements=replacements)
    return module.PAMIL(input_dim=dim, hidden_dim=hidden, attention_dim=attention_dim,
        n_protos=prototypes, n_classes=classes, dropout=dropout, proto_pred=True,
        proto_weight=proto_weight, k_sample=k_sample)


def target_vector(row, classes, variant, device):
    if variant == 'multilabel':
        return torch.tensor(row['targets'], dtype=torch.float32, device=device).unsqueeze(0)
    return F.one_hot(torch.tensor([row['target']], device=device), classes).float()


def loader(groups, weighted=False, multilabel=False):
    groups = sorted(groups, key=lambda g: g[0]['slide_id'])
    if any(len(g) != 1 for g in groups):
        raise ValueError('PAMIL requires individual slide bags')
    if weighted:
        keys = [tuple(g[0]['targets']) if multilabel else g[0]['target'] for g in groups]
        counts = {key: keys.count(key) for key in set(keys)}
        sampler = WeightedRandomSampler(torch.tensor([len(keys)/counts[k] for k in keys], dtype=torch.double), len(groups))
    else:
        sampler = RandomSampler(groups)
    return DataLoader(groups, batch_size=None, sampler=sampler, num_workers=0, collate_fn=lambda g: g)


class PAMIL(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'class attention logits, attention normalized over instances, unnormalized prototype similarities'

    def __init__(self, bundle, profile='pamil_prototype_repaired', hidden=512, attention_dim=256,
                 prototypes=10, local_clusters=8, warmup_epochs=80, push_epochs=None,
                 last_epochs=30, dropout=False, proto_weight=.5, w_er=.5, w_clst=-.5,
                 w_proto_clst=-.2, w_inst=.5, inst_pred=False, k_sample=8,
                 weighted_sampling=False, maximum_centers=100000, validation_selection=False):
        super().__init__()
        if profile not in PROFILES:
            raise ValueError('Unknown PAMIL profile')
        self.profile, self.variant = profile, PROFILES[profile]
        if bundle.task_type != ('multilabel' if self.variant == 'multilabel' else 'classification'):
            raise ValueError('PAMIL profile and task type differ')
        if bundle.config.get('training_unit', 'slide') != 'slide' or bundle.pamil is None:
            raise ValueError('PAMIL requires slide training and frozen TRAIN instance annotations')
        for name, value in [('hidden', hidden), ('attention_dim', attention_dim), ('prototypes', prototypes),
                            ('local_clusters', local_clusters), ('warmup_epochs', warmup_epochs),
                            ('last_epochs', last_epochs), ('k_sample', k_sample), ('maximum_centers', maximum_centers)]:
            positive_int(value, name)
        if prototypes < 2:
            raise ValueError('PAMIL requires at least two prototypes')
        if any(type(value) is not bool for value in [dropout, inst_pred, weighted_sampling, validation_selection]):
            raise ValueError('PAMIL dropout/instance/sampling flags must be booleans')
        if self.variant == 'source' and inst_pred:
            raise ValueError('Original multiclass instance path has scalar-target and class-index bugs; use the repaired profile')
        for name, value in [('proto_weight', proto_weight), ('w_er', w_er), ('w_clst', w_clst),
                            ('w_proto_clst', w_proto_clst), ('w_inst', w_inst)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name+' must be finite')
        if not 0 < proto_weight < 1 or w_er <= 0 or w_clst >= 0 or w_proto_clst >= 0 or w_inst <= 0:
            raise ValueError('PAMIL requires both prediction branches and active source cluster/consistency terms')
        self.push_epochs = ([320] if self.variant == 'multilabel' else [330]) if push_epochs is None else push_epochs
        if not isinstance(self.push_epochs, list) or not self.push_epochs:
            raise ValueError('PAMIL requires at least one push epoch')
        for value in self.push_epochs:
            positive_int(value, 'push epoch')
        if self.push_epochs != sorted(set(self.push_epochs)) or self.push_epochs[0] <= warmup_epochs:
            raise ValueError('PAMIL pushes must be ordered, unique and follow warm AND joint training')
        if any(b-a <= last_epochs for a, b in zip(self.push_epochs, self.push_epochs[1:])):
            raise ValueError('PAMIL push schedule must allow a joint epoch between last-layer phases')
        self.bundle, self.classes, self.dim = bundle, len(bundle.classes), bundle.dim
        self.network = author_network(bundle.dim, self.classes, prototypes, hidden, attention_dim,
                                      dropout, self.variant, proto_weight, k_sample)
        self.local_clusters, self.warmup_epochs, self.last_epochs = local_clusters, warmup_epochs, last_epochs
        self.w_er, self.w_clst, self.w_proto_clst, self.w_inst = w_er, w_clst, w_proto_clst, w_inst
        self.inst_pred, self.weighted_sampling, self.maximum_centers = inst_pred, weighted_sampling, maximum_centers
        self.validation_selection = validation_selection
        self.epoch, self.mode, self.last_steps, self.fc_stays_frozen = -1, 'warm', 0, False
        self.initialization, self.pushes, self.optimizer_steps = None, [], 0
        self.register_buffer('prepared', torch.tensor(False))
        self.stop_after_epoch = self.push_epochs[-1]  # no early stop before all required pushes

    def configure_training(self, cfg):
        if cfg['epochs'] <= self.push_epochs[-1]:
            raise ValueError('PAMIL budget must include clustering, warm, joint, push AND last training')
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('PAMIL uses complete bags and source unclipped Adam updates')
        if cfg['selection_metric'] != 'pamil_loss':
            raise ValueError('PAMIL selects on its native validation objective')
        # Released loops ignore the stopping flag; they always finish the fixed budget.
        self.stop_after_epoch = cfg['epochs']

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        from .engine import restore_rng, rng_state, seed_everything
        state = rng_state()
        try:
            # The author clustering.py runs in a separate seeded process. It must
            # not advance the model-training random stream in an integrated run.
            seed_everything(seed)
            self.initialize_prototypes(bundle, seed, device)
        finally:
            restore_rng(state)

    def initialize_prototypes(self, bundle, seed, device):
        centers, slides, local_counts = [], [], []
        groups = [[r] for r in bundle.records('train')]
        for rows in loader(groups, self.weighted_sampling, self.variant == 'multilabel'):
            row = rows[0]
            bundle.pamil.load(row)
            features, _ = bundle.load(row, require_coords=True)
            # Source collate wraps cors in a batch-size-one list, so len(cors)==1.
            count = 1 if self.variant == 'source' else min(len(features), self.local_clusters)
            if sum(local_counts)+count > self.maximum_centers:
                raise ValueError('PAMIL initialization exceeds maximum_centers before concatenation')
            fitted = KMeans(n_clusters=count, n_init=10).fit(features.numpy())
            if len(set(fitted.labels_)) != count:
                raise ValueError('PAMIL local clustering has empty clusters')
            centers.append(fitted.cluster_centers_)
            slides.append(row['slide_id']); local_counts.append(count)
        values = np.concatenate(centers)
        if len(values) < self.network.n_protos:
            raise ValueError('PAMIL has fewer local centers than global prototypes')
        result = KMeans(n_clusters=self.network.n_protos, n_init=10).fit(values)
        if len(set(result.labels_)) != self.network.n_protos:
            raise ValueError('PAMIL global clustering has empty clusters')
        self.network.proto.copy_(torch.from_numpy(result.cluster_centers_).to(device))
        self.prepared.fill_(True)
        self.initialization = {'slide_ids': slides, 'local_counts': local_counts, 'seed': seed,
            'centers_sha256': digest_object(result.cluster_centers_.tolist()),
            'algorithm': 'two-level sklearn KMeans n_init=10, TRAIN only',
            'source_collate_len_one': self.variant == 'source'}

    def make_optimizer(self, lr, weight_decay):
        # Source constructs Adam before stage freezes; keep all Parameter objects.
        return torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)

    def set_mode(self):
        self.network.proto.requires_grad_(False)  # Both released warm_only AND joint freeze it.
        for name in ['fc', 'attention_net', 'classifiers', 'proto_pred_classifier', 'instance_classifiers']:
            active = name != 'fc' or (self.mode != 'last' and not self.fc_stays_frozen)
            for parameter in getattr(self.network, name).parameters():
                parameter.requires_grad_(active)

    def start_epoch(self, epoch):
        if not self.prepared:
            raise ValueError('PAMIL train-only clustering has not run')
        if epoch >= self.warmup_epochs and self.mode == 'warm':
            self.mode = 'joint'
        elif self.mode == 'last' and self.last_steps >= self.last_epochs:
            self.mode = 'joint'
            # Original loop forgets joint(model) after LAST and never unfreezes fc.
            self.fc_stays_frozen = self.variant == 'source'
        self.epoch = epoch
        if epoch in self.push_epochs:
            if self.mode != 'joint':
                raise ValueError('PAMIL push encountered outside the joint stage')
            self.push()
            self.mode, self.last_steps = 'last', 0
        self.set_mode()

    def finish_epoch(self, bundle, device, cfg):
        if self.mode == 'last':
            self.last_steps += 1

    @torch.no_grad()
    def push(self):
        """Class-filtered, per-bag nonduplicate greedy projection; atomic commit."""
        net = self.network
        active = net.training
        net.eval()
        device = net.proto.device
        proto_class = net.attention_net.attention_c.weight.detach().argmax(0)
        scores = [-float('inf')]*net.n_protos
        winners = [None]*net.n_protos
        projected = torch.empty_like(net.proto)
        try:
            for rows in loader([[r] for r in self.bundle.records('train')], self.weighted_sampling, self.variant == 'multilabel'):
                row = rows[0]
                labels = self.bundle.pamil.load(row)
                x, _ = self.bundle.load(row, require_coords=True)
                x = x.to(device)
                coords = load_array(resolve(self.bundle.root, row.get('coords_path') or row['path']), 'coords')
                _, _, _, extra = net(x)
                similarity = extra['proto_A']
                if not torch.isfinite(similarity).all():
                    raise ValueError('PAMIL push similarity is nonfinite')
                present = target_vector(row, self.classes, self.variant, device)[0]
                used = set()
                for index in range(net.n_protos):
                    cls = int(proto_class[index])
                    if not present[cls]:
                        continue
                    eligible = np.flatnonzero(labels == cls)
                    if not len(eligible):
                        continue
                    available = [int(i) for i in eligible if int(i) not in used]
                    if not available:
                        raise ValueError('PAMIL push exhausted unique annotated patches; original loop would not terminate')
                    values = similarity[available, index]
                    patch = available[int(values.argmax())]
                    used.add(patch)
                    value = float(similarity[patch, index])
                    if value > scores[index]:
                        scores[index] = value
                        projected[index].copy_(x[patch])
                        winners[index] = {'slide_id': row['slide_id'], 'patch_index': patch,
                            'coordinate': coords[patch].tolist(), 'class_index': cls, 'similarity': value,
                            'features_sha256': digest_file(resolve(self.bundle.root, row['path']))}
            if any(w is None for w in winners):
                raise ValueError('PAMIL push has prototypes without eligible TRAIN annotations; refusing zero replacement')
            # Source replaces nn.Parameter, invalidating optimizer ownership. Copy in place.
            net.proto.copy_(projected)
            self.pushes.append({'epoch': self.epoch, 'winners': winners,
                                'prototype_sha256': digest_object(projected.cpu().tolist())})
        finally:
            net.train(active)

    def loss(self, result, target, validation=False):
        if self.variant == 'multilabel':
            classification = F.binary_cross_entropy(result['probabilities'][0], target[0])
        elif self.variant == 'source':
            classification = F.cross_entropy(result['probabilities'][0], target[0])
        else:
            classification = F.cross_entropy(result['logits'][0], target[0])
        total = classification+self.w_er*result['loss_er']+self.w_clst*result['loss_clst']
        terms = {'classification': classification, 'consistency': result['loss_er'],
                 'instance_cluster': result['loss_clst']}
        if not validation:
            if self.inst_pred:
                total = total+self.w_inst*result['loss_inst']
                terms['instance_prediction'] = result['loss_inst']
            # Source multi-label uses elif, so prototype clustering is absent when inst_pred=True.
            if not self.inst_pred or self.variant != 'multilabel':
                total = total+self.w_proto_clst*result['loss_proto_clst']
                terms['prototype_cluster'] = result['loss_proto_clst']
        return total, terms

    def forward(self, x, coords=None, label=None):
        logits, probability, _, extra = self.network(x, inst_pred=self.training and self.inst_pred,
                                                    label=label)
        result = {'logits': logits, 'probabilities': probability, **extra}
        if not all(torch.isfinite(result[key]).all() for key in ['logits', 'probabilities', 'loss_er', 'proto_A']):
            raise ValueError('PAMIL network produced nonfinite outputs')
        if coords is not None:
            result['coords'] = coords
        return result

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1 or config['max_instances'] is not None:
            raise ValueError('PAMIL requires a complete single-slide bag')
        row = rows[0]
        x, coords = bundle.load(row)
        return self(x.to(device), coords=coords, label=None)

    def prediction_fields(self, result, row):
        target = target_vector(row, self.classes, self.variant, result['logits'].device)
        value, _ = self.loss(result, target, validation=True)
        return {'pamil_loss': float(value)}

    def start_prediction(self, subset):
        # Actual source SequentialSampler DataLoader consumes one iterator seed.
        torch.empty((), dtype=torch.int64).random_()

    def validation_metrics(self, frame):
        return {'pamil_loss': float(frame.pamil_loss.mean())}

    def selection_improved(self, objective, best):
        return objective >= best  # Original EarlyStopping saves ties.

    def selection_objective(self, metrics, epoch, objective):
        return objective if self.validation_selection else float(epoch)

    def get_extra_state(self):
        return {name: getattr(self, name) for name in ['epoch', 'mode', 'last_steps', 'fc_stays_frozen',
                                                      'initialization', 'pushes', 'optimizer_steps']}

    def set_extra_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self.set_mode()

    def protocol(self):
        return {'method': self.profile, 'family': 'pamil_prototype', 'variant': self.variant,
            'extra_supervision': self.bundle.pamil.manifest['scope'],
            'annotation_provenance': self.bundle.pamil.manifest['provenance'],
            'prototype_training': 'frozen in warm/joint/last as in released source; projected by TRAIN annotations',
            'warmup_epochs': self.warmup_epochs, 'push_epochs': self.push_epochs, 'last_epochs': self.last_epochs,
            'initialization': self.initialization, 'pushes': self.pushes, 'epoch': self.epoch, 'mode': self.mode,
            'optimizer_steps': self.optimizer_steps, 'sampling': 'source DataLoader random/weighted replacement',
            'selection': 'native validation objective, ties saved; best can precede push' if self.validation_selection else 'last epoch as in source early_stopping=False',
            'classification_objective': 'CE-on-softmax' if self.variant == 'source' else 'BCE-on-sigmoid' if self.variant == 'multilabel' else 'CE-on-logits',
            'clustering': 'one local center per slide: source collate len(cors)' if self.variant == 'source' else 'min(N,local_clusters) per slide: explicit repair',
            'repairs': ['device-neutral multilabel/index tensors', 'finite-zero prototype normalization',
                        'unwrap collated instance labels and compare integer class indices',
                        'atomic in-place push; fail on missing/exhausted annotations',
                        *([] if self.variant == 'source' else ['stable log-sigmoid ER', 'class-specific instance mining', 'restore fc after last phase'])],
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

    def export_instances(self, result, row, directory):
        from pathlib import Path
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object({'slide_id': row['slide_id']})[:24]+'.h5')
        # Read original coordinate precision; no labels or second stochastic forward.
        cp = resolve(self.bundle.root, row.get('coords_path') or row['path'])
        coords = None
        if row.get('coords_path'):
            coords = load_array(cp, 'coords')
        elif cp.suffix in {'.h5', '.hdf5'}:
            with h5py.File(cp, 'r') as h:
                if 'coords' in h:
                    coords = h['coords'][:]
        with h5py.File(path, 'w') as h:
            for key, value in {'attention_logits': result['A_raw'].T,
                               'attention': result['A_raw'].softmax(-1).T,
                               'prototype_similarity': result['proto_A']}.items():
                h[key] = value.detach().cpu().numpy()
            if coords is not None:
                h['coords'] = coords
            h.attrs['method'] = self.profile
            h.attrs['score_semantics'] = 'class attention is softmax over instances; prototype similarity is unnormalized'
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path),
                'instances': len(result['proto_A']), 'class_count': self.classes,
                'prototype_count': self.network.n_protos, 'coordinates': coords is not None}


def pamil_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    for rows in loader(groups, model.weighted_sampling, model.variant == 'multilabel'):
        row = rows[0]
        x, _ = bundle.load(row)
        target = target_vector(row, model.classes, model.variant, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(x.to(device), label=target)
        total, terms = model.loss(result, target)
        if not torch.isfinite(total):
            raise ValueError('PAMIL objective is nonfinite')
        total.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('PAMIL gradient is nonfinite')
        optimizer.step()
        model.optimizer_steps += 1
        losses.append({'loss': float(total.detach()), **{k: float(v.detach()) for k, v in terms.items()}})
    return losses

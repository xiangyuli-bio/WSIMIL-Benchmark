"""Complete end-to-end C2C: original ResNet18, per-epoch FAISS and all losses."""
import copy
import random

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .c2c_source import COMMIT, PROFILES, ToTensor, build_model, objective
from .io import digest_object


def tensor_hash(value):
    import hashlib
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class Images(Dataset):
    def __init__(self, pixels, transform, patch_ids):
        self.pixels, self.transform, self.patch_ids = pixels, transform, patch_ids

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        return self.transform(self.pixels[index]), self.patch_ids[index]


class ClusterBags(Dataset):
    def __init__(self, model, rows):
        self.model, self.rows = model, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        model, row = self.model, self.rows[index]
        assignment = model.clusters[row['slide_id']]['labels'].numpy()
        selected, labels = [], []
        # Literal WSIClusterDataloader: loop all eight cluster IDs, Python
        # random.shuffle on each NumPy index array, then retain up to eight.
        for cluster in range(8):
            indices = np.where(assignment == cluster)[0]
            if not len(indices):
                continue
            random.shuffle(indices)
            for count, patch in enumerate(indices):
                if count >= 8:
                    break
                selected.append(int(patch)); labels.append(cluster)
        with model.assets.open(row) as pixels:
            images = [model.transform(pixels[patch]) for patch in selected]
        # Original source decodes all selected images before applying their
        # transforms; decoding has no RNG and the transform consumes one draw
        # per image. Source sampling is fully completed before either operation.
        return torch.stack(images), row['target'], labels, index, torch.tensor(selected, dtype=torch.long)


class C2C(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='c2c_cluster_source'):
        super().__init__()
        if profile not in PROFILES or bundle.c2c is None:
            raise ValueError('C2C needs its registered original RGB bags and complete cluster route')
        self.bundle, self.assets, self.profile = bundle, bundle.c2c, profile
        self.network, self.source, self.criterion, self.cluster_function, self.predict_function = build_model(
            self.assets.assets, len(bundle.classes), profile)
        self.transform = ToTensor(self.assets.assets)
        self.repaired = profile == 'c2c_cluster_multiclass_repaired'
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'bundle': bundle.fingerprint,
            'input': self.assets.contract, 'faiss_version': faiss.__version__})
        self.recipe = None
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders, self.cluster_history, self.samples_history = [], [], []
        self.clusters = {}

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'c2c_val_accuracy':
            raise ValueError('Complete C2C needs all-image reclustering, full three-loss training and source VAL accuracy')
        self.recipe = {key: cfg[key] for key in ['epochs', 'lr', 'weight_decay', 'threads', 'selection_metric']}
        faiss.omp_set_num_threads(cfg['threads'])

    def make_optimizer(self, lr, weight_decay):
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        return self._optimizer

    def validate_optimizer_state(self):
        groups = self._optimizer.param_groups
        if (len(groups) != 1 or groups[0]['lr'] != self.recipe['lr']
                or groups[0]['weight_decay'] != self.recipe['weight_decay']
                or groups[0]['betas'] != (.9, .999) or groups[0]['eps'] != 1e-8
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.network.parameters()}):
            raise ValueError('C2C complete source Adam or parameter ownership changed')
        if self.optimizer_updates and (len(self._optimizer.state) != len(list(self.network.parameters()))
                or any(int(state['step']) != self.optimizer_updates for state in self._optimizer.state.values())):
            raise ValueError('C2C source Adam lost actual complete parameter updates')

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('C2C epochs must resume from the complete original recipe state')
        self.epoch_index = epoch

    @torch.no_grad()
    def _recluster(self, device):
        # Source Enc shares the actual backbone/tail, so eval-mode legacy BN
        # updates during this stage are observable model buffers, not a throwaway
        # feature cache. Training then explicitly restores model.train().
        encoder = self.source.Enc(self.network).eval().to(device)
        current = {}
        for row in self.bundle.records('train'):
            features, patch_ids = None, []
            ids = self.assets.meta['slides'][row['slide_id']]['patch_ids']
            with self.assets.open(row) as pixels:
                loader = DataLoader(Images(pixels, self.transform, ids), batch_size=128,
                                    shuffle=False, num_workers=0)
                for images, names in loader:
                    value = encoder(images.to(device)).detach().cpu().numpy()
                    features = value if features is None else np.concatenate((features, value))
                    patch_ids += list(names)
            labels, distances, actual_ids, centers = self.cluster_function((features, patch_ids), num_cluster=8)
            if actual_ids != ids or not np.isfinite(centers).all() or not np.isfinite(distances).all():
                raise ValueError('C2C complete source FAISS returned nonfinite state or changed original patch order')
            current[row['slide_id']] = {'labels': torch.from_numpy(np.asarray(labels, dtype=np.int64)),
                'distances': torch.from_numpy(np.asarray(distances, dtype=np.float32)),
                'centers': torch.from_numpy(np.asarray(centers, dtype=np.float32)),
                'patch_ids_sha256': digest_object(patch_ids)}
        nmi = None if not self.clusters else float(np.mean([
            normalized_mutual_info_score(current[key]['labels'].numpy(), self.clusters[key]['labels'].numpy())
            for key in current]))
        self.clusters = current
        self.cluster_history.append({'epoch': self.epoch_index, 'nmi': nmi,
            'labels': {key: tensor_hash(value['labels']) for key, value in current.items()},
            'centers': {key: tensor_hash(value['centers']) for key, value in current.items()}})

    def epoch(self, bundle, optimizer, device):
        if bundle.fingerprint != self.bundle.fingerprint or optimizer is not self._optimizer:
            raise ValueError('C2C actual images, labels or Adam identity changed')
        self._recluster(device); self.network.train()
        rows = self.bundle.records('train')
        loader = DataLoader(ClusterBags(self, rows), batch_size=1, shuffle=True, num_workers=0)
        losses, order, sampled = [], [], {}
        # Retain the original extra epoch-level zero_grad before the per-bag
        # zero_grad calls. Neither source path accumulates gradients across WSI.
        optimizer.zero_grad(set_to_none=False)
        for images, target, clusters, row_index, indices in loader:
            optimizer.zero_grad(set_to_none=False)
            images, target = images.to(device), target.to(device)
            logits, patch_logits, attention = self.network(images)
            values = objective(logits, patch_logits, attention, target, np.array(clusters), self.criterion, self.repaired)
            if not torch.isfinite(values['loss']):
                raise FloatingPointError('C2C complete bag/patch/cluster objective became nonfinite')
            values['loss'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError('C2C full original ResNet18 backward became nonfinite')
            optimizer.step(); self.optimizer_updates += 1
            row = rows[int(row_index.item())]; order.append(row['slide_id'])
            sampled[row['slide_id']] = indices[0].tolist()
            record = {key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                      for key, value in values.items()}
            record['selected_patches'] = len(patch_logits)
            record['kl_active_groups'] = sum(np.count_nonzero(np.asarray(clusters) == cluster) > 4
                                            for cluster in np.unique(np.asarray(clusters)))
            losses.append(record)
        self.completed_epochs = self.epoch_index+1
        self.train_orders.append(order); self.samples_history.append(sampled)
        self.validate_optimizer_state()
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if subset not in {'val', 'test'} or bundle.fingerprint != self.bundle.fingerprint:
            raise ValueError('C2C prediction requires the registered complete validation/test image bags')
        self.network.eval(); records = []
        for row in bundle.records(subset):
            ids = self.assets.meta['slides'][row['slide_id']]['patch_ids']
            with self.assets.open(row) as pixels:
                loader = DataLoader(Images(pixels, self.transform, ids), batch_size=128,
                                    shuffle=False, num_workers=0)
                logits, _ = self.predict_function(loader, self.network)
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'target']}
            record.update(subset=subset, source_slides=row['slide_id'], prediction=int(logits.argmax(1)))
            record.update({f'p_{index}': float(value) for index, value in enumerate(logits.softmax(-1)[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        return {'c2c_val_accuracy': float(np.mean(frame.target.to_numpy() == frame.prediction.to_numpy()))}

    def selection_improved(self, objective, best):
        return objective >= best  # Literal source replaces on validation ties.

    def get_extra_state(self):
        return copy.deepcopy({'contract': self.contract, 'recipe': self.recipe,
            'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
            'optimizer_updates': self.optimizer_updates, 'train_orders': self.train_orders,
            'cluster_history': self.cluster_history, 'samples_history': self.samples_history,
            'clusters': self.clusters})

    def set_extra_state(self, state):
        keys = set(self.get_extra_state())
        recipe = state.get('recipe')
        if (set(state) != keys or state['contract'] != self.contract or not isinstance(recipe, dict)
                or set(recipe) != {'epochs', 'lr', 'weight_decay', 'threads', 'selection_metric'}
                or type(recipe['epochs']) is not int or recipe['epochs'] < 1
                or type(recipe['threads']) is not int or recipe['threads'] < 1
                or not np.isfinite(recipe['lr']) or recipe['lr'] <= 0
                or not np.isfinite(recipe['weight_decay']) or recipe['weight_decay'] < 0
                or recipe['selection_metric'] != 'c2c_val_accuracy'):
            raise ValueError('C2C actual inputs, source identity or complete training recipe changed')
        if self.recipe is not None and self.recipe != recipe:
            raise ValueError('C2C checkpoint changed the declared source recipe')
        self.recipe = copy.deepcopy(recipe); faiss.omp_set_num_threads(recipe['threads'])
        epochs = state['completed_epochs']
        if (any(type(state[key]) is not int for key in ['epoch_index', 'completed_epochs', 'optimizer_updates'])
                or not 0 <= epochs <= recipe['epochs'] or state['epoch_index'] != epochs-1
                or any(len(state[key]) != epochs for key in ['train_orders', 'cluster_history', 'samples_history'])):
            raise ValueError('C2C complete epoch, clustering or sampling history changed')
        rows = {row['slide_id']: row for row in self.bundle.records('train')}
        if state['optimizer_updates'] != epochs*len(rows):
            raise ValueError('C2C checkpoint omitted complete source Adam updates')
        for epoch, (order, sampled, clusters) in enumerate(zip(state['train_orders'], state['samples_history'], state['cluster_history'])):
            if (len(order) != len(rows) or set(order) != set(rows) or set(sampled) != set(rows)
                    or set(clusters) != {'epoch', 'nmi', 'labels', 'centers'} or clusters['epoch'] != epoch
                    or set(clusters['labels']) != set(rows) or set(clusters['centers']) != set(rows)
                    or (epoch == 0 and clusters['nmi'] is not None)
                    or (epoch > 0 and (not isinstance(clusters['nmi'], (int, float))
                        or not np.isfinite(clusters['nmi']) or not 0 <= clusters['nmi'] <= 1))):
                raise ValueError('C2C original per-WSI clustering/sampling identity changed')
            for slide, indices in sampled.items():
                if (not 1 <= len(indices) <= 64 or len(indices) != len(set(indices))
                        or any(type(index) is not int or not 0 <= index < self.assets.count(rows[slide]) for index in indices)):
                    raise ValueError('C2C original source cluster sample is incomplete or fabricated')
        if set(state['clusters']) != (set(rows) if epochs else set()):
            raise ValueError('C2C checkpoint dropped actual TRAIN clustering state')
        for slide, value in state['clusters'].items():
            count = self.assets.count(rows[slide]); k = min(8, count)
            if (set(value) != {'labels', 'distances', 'centers', 'patch_ids_sha256'}
                    or value['labels'].shape != (count,) or value['labels'].dtype != torch.int64
                    or value['distances'].shape != (count,) or value['distances'].dtype != torch.float32
                    or value['centers'].shape != (k, 64) or value['centers'].dtype != torch.float32
                    or not torch.isfinite(value['distances']).all() or not torch.isfinite(value['centers']).all()
                    or (value['labels'] < 0).any() or (value['labels'] >= k).any()
                    or value['patch_ids_sha256'] != self.assets.meta['slides'][slide]['patch_ids_sha256']
                    or tensor_hash(value['labels']) != state['cluster_history'][-1]['labels'][slide]
                    or tensor_hash(value['centers']) != state['cluster_history'][-1]['centers'][slide]):
                raise ValueError('C2C actual FAISS labels, distances or centroids were lost or changed')
            selected = state['samples_history'][-1][slide]
            assignment = value['labels'].cpu().numpy(); selected_labels = assignment[selected]
            if (len(selected) != sum(min(8, int(np.count_nonzero(assignment == cluster))) for cluster in range(8))
                    or any(np.count_nonzero(selected_labels == cluster) != min(8, np.count_nonzero(assignment == cluster)) for cluster in range(8))
                    or not np.array_equal(selected_labels, np.sort(selected_labels))):
                raise ValueError('C2C eight-cluster/eight-sample source sampling contract changed')
        for key in keys-{'contract', 'recipe'}:
            setattr(self, key, copy.deepcopy(state[key]))
        # map_location=cuda also relocates tensors inside _extra_state. These
        # original FAISS/NumPy caches are host state, independently of the CNN.
        self.clusters = {slide: {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                                for key, value in record.items()}
                         for slide, record in self.clusters.items()}

    def protocol(self):
        return {'method': self.profile, 'source_commit': COMMIT,
            'architecture': 'complete original ImageNet ResNet18/512 ->256->64, attention64->32->1; separate64->C bag and patch heads',
            'weights': self.assets.meta['assets']['resnet18-5c106cde.pth'],
            'pixels': 'original full RGB patches; exact Albumentations0.4.6 ToTensor only, including Compose Python RNG draw; no resize/ImageNet normalization',
            'batchnorm': 'exact PyTorch1.5.1 _BatchNorm.forward, including buffer updates and train/eval flag semantics; source notebook disables tracking after initialization' if self.profile != 'c2c_cluster_default_bn_source' else 'exact PyTorch1.5.1 forward with original constructor-default reset stats/affine and momentum.9; explicit alternate source condition',
            'clustering': 'every epoch all TRAIN patches; real source normalized64D embeddings/CPU FAISS Kmeans defaults, K=min(8,N); all eight IDs sampled up to8each via original Python random.shuffle',
            'faiss_version': faiss.__version__, 'faiss_threads': None if self.recipe is None else self.recipe['threads'],
            'objective': '1*bagCE+.01*weakPatchCE+.01*original KL(uniform || softmax(unnormalized attention)), skips groups with<=4 samples exactly as source',
            'multiclass_repair': 'explicit true bag-class weak labels instead of source binary-if1-else0 branch' if self.repaired else None,
            'recipe': self.recipe, 'native_recipe': 'train_model default25epochs, Adam1e-4/wd0,batch1; no scheduler/early stop/clipping; Notebook1epoch/topkTrue is an ablation, not used',
            'selection': 'VAL all-patch streaming128/chunk, original deep-copy WSI batch normalization; accuracy latest ties; no TEST use during training',
            'completed_epochs': self.completed_epochs, 'optimizer_updates': self.optimizer_updates,
            'resume': 'full source BN buffers, Adam, all per-epoch cluster/sampling histories, last labels/distances/centroids, original patch identity and global RNG',
            'synthetic': bool(self.bundle.config.get('synthetic', False)), 'real_wsi_validation': 'not_run',
            'paper_scores': 'not_reproduced', 'source_ports': 'device-neutral calls; export original prediction logits/FAISS centroids without changing source computation; original dependencies executed with explicit modern CPU FAISS package identity'}

"""Full hard-MIL ResNet34 selector followed by the author's recurrent aggregator."""
import copy
import hashlib
import importlib.metadata
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import resolve
from .io import digest_object
from .loss_attention import check_update_finite, finite
from .raw_images import positive_integer
from .raw_regions import grid_sha256
from .upstream import load_definitions


def author_components():
    selector = load_definitions('campanella_rnn', 'MIL_train.py', ['group_argtopk', 'group_max'], {'np': np})
    recurrent = load_definitions('campanella_rnn', 'RNN_train.py', ['rnn_single'], {'torch': torch, 'nn': nn})
    return selector, recurrent.rnn_single


def state_sha256(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class FlatTiles(Dataset):
    def __init__(self, model, rows, selected=None):
        self.model, self.rows = model, rows
        self.counts = [len(model.reader.grid(row)) for row in rows]
        self.offsets = np.cumsum([0]+self.counts)
        self.pairs = None if selected is None else list(selected)
        self.labeled = selected is not None
        if self.pairs is None:
            model.reserve_meta(int(self.offsets[-1]))

    def __len__(self):
        return int(self.offsets[-1]) if self.pairs is None else len(self.pairs)

    def pair(self, index):
        if self.pairs is not None:
            return self.pairs[index]
        slide = int(np.searchsorted(self.offsets, index, side='right')-1)
        return slide, int(index-self.offsets[slide])

    def group_ids(self):
        return np.repeat(np.arange(len(self.rows)), self.counts) if self.pairs is None else np.array([s for s, _ in self.pairs])

    def __getitem__(self, index):
        slide, tile = self.pair(index)
        row = self.rows[slide]
        image = self.model.reader.read(row, int(tile), 'cnn')
        return (image, row['target']) if self.labeled else image


class RecurrentBags(Dataset):
    def __init__(self, model, rows, shuffle_sequence=False):
        self.model, self.rows, self.shuffle_sequence = model, rows, shuffle_sequence

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        selected = self.model.top_cache[row['slide_id']]['indices']
        if self.shuffle_sequence:
            selected = random.sample(selected, len(selected))
        return [self.model.reader.read(row, int(i), 'rnn') for i in selected], row['target']


class CampanellaMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, cnn_epochs=100, rnn_epochs=100, cnn_batch_size=512, rnn_batch_size=128,
                 test_every=10, cnn_k=1, steps=10, ndims=128, positive_weight=.5, shuffle_sequence=False,
                 rnn_lr=.1, max_dense_bytes=32*1024**3):
        super().__init__()
        if bundle.raw_regions is None or bundle.task_type != 'classification' or len(bundle.classes) != 2:
            raise ValueError('Campanella requires raw WSI regions and explicit binary source0/1 labels')
        if bundle.config.get('training_unit', 'slide') != 'slide' or bundle.config.get('evaluation_unit', 'slide') != 'slide':
            raise ValueError('Original Campanella recipe trains and evaluates individual WSI bags; register patient fusion separately')
        if importlib.metadata.version('torchvision').split('+')[0] != '0.20.1':
            raise ValueError('Campanella native ResNet34 route requires locked torchvision0.20.1')
        for key, value in [('cnn_epochs', cnn_epochs), ('rnn_epochs', rnn_epochs), ('cnn_batch_size', cnn_batch_size),
                           ('rnn_batch_size', rnn_batch_size), ('test_every', test_every), ('cnn_k', cnn_k),
                           ('steps', steps), ('ndims', ndims), ('max_dense_bytes', max_dense_bytes)]:
            positive_integer(value, key)
            setattr(self, key, value)
        if cnn_epochs % test_every:
            raise ValueError('CNN stage must end on a real source validation boundary; no unvalidated selector fallback')
        if type(shuffle_sequence) is not bool or not 0 < finite(positive_weight, 'positive_weight') < 1 or finite(rnn_lr, 'rnn_lr') <= 0:
            raise ValueError('Invalid Campanella source sequence/class weighting/learning rate')
        self.bundle, self.reader = bundle, bundle.raw_regions
        self.positive_weight, self.shuffle_sequence, self.rnn_lr = positive_weight, shuffle_sequence, rnn_lr
        for row in bundle.rows.to_dict('records'):
            if len(self.reader.grid(row)) < max(steps, cnn_k):
                raise ValueError('Campanella requires enough real tiles for the fixed native sequence/top-k, without padding or reducing steps')
        self.reserve(cnn_batch_size, 'cnn')
        self.reserve(rnn_batch_size, 'rnn')
        from torchvision import models
        self.cnn = models.resnet34(weights=None)
        if self.reader.weights is not None:
            self.cnn.load_state_dict(torch.load(self.reader.weights, map_location='cpu', weights_only=True), strict=True)
        self.cnn.fc = nn.Linear(512, 2)
        self.selector, recurrent = author_components()
        # A shape-stable placeholder keeps optimizer ownership and checkpoint loading simple.
        # Its random draws are undone; the real stage2 initialization happens after CNN selection.
        saved = torch.get_rng_state()
        self.rnn = recurrent(ndims)
        torch.set_rng_state(saved)
        self.phase, self.epoch = 'cnn', -1
        self.rnn_initialized = False
        self.cnn_best_state, self.cnn_best_sha256, self.cnn_best_epoch = None, None, None
        self.cnn_best_score = -float('inf')
        self.cnn_updates, self.rnn_updates = 0, 0
        self.top_cache, self.selected_training_rows = {}, []
        self.apply_phase()

    def reserve(self, batch, phase):
        pixels = batch*3*224**2*4
        estimate = 24_000_000*20 + pixels*(64 if phase == 'cnn' else self.steps+8)
        if estimate > self.max_dense_bytes:
            raise ValueError('Campanella native CNN/sequence batch exceeds max_dense_bytes before allocation')

    def reserve_meta(self, tiles):
        if 64*tiles+24_000_000*20 > self.max_dense_bytes:
            raise ValueError('Campanella full-grid source ranking exceeds max_dense_bytes before allocation')

    def apply_phase(self):
        for p in self.cnn.parameters():
            p.requires_grad_(self.phase == 'cnn')
            if self.phase == 'rnn':
                p.grad = None
        for p in self.rnn.parameters():
            p.requires_grad_(self.phase == 'rnn')
        if self.phase == 'rnn':
            self.cnn.eval()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'phase', None) == 'rnn':
            self.cnn.eval()
        return self

    def configure_training(self, cfg):
        if cfg['epochs'] != self.cnn_epochs+self.rnn_epochs or cfg['grad_clip'] is not None or cfg['max_instances'] is not None or cfg['selection_metric'] != 'campanella_balanced_accuracy':
            raise ValueError('Campanella requires complete two-phase budgets, own full-grid sampling, no clipping and native balanced-error selection')
        self.stop_after_epoch = cfg['epochs']

    def make_optimizers(self, lr, weight_decay):
        self.optimizers = [torch.optim.Adam(self.cnn.parameters(), lr=lr, weight_decay=weight_decay),
                           torch.optim.SGD(self.rnn.parameters(), self.rnn_lr, momentum=.9, dampening=0, weight_decay=1e-4, nesterov=True)]
        return self.optimizers

    def criterion(self, device):
        weight = None if self.positive_weight == .5 else torch.tensor([1-self.positive_weight, self.positive_weight], device=device)
        return nn.CrossEntropyLoss(weight=weight)

    def start_epoch(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < self.cnn_epochs+self.rnn_epochs:
            raise ValueError('Campanella epoch is outside the registered two-phase budget')
        self.epoch = epoch
        if epoch >= self.cnn_epochs and self.phase == 'cnn':
            if self.cnn_best_state is None:
                raise ValueError('Stage2 cannot start without a validated best CNN selector')
            from torchvision import models
            # Preserve the original stage2 ResNetEncoder construction/random draw sequence.
            fresh = models.resnet34(weights=None)
            fresh.fc = nn.Linear(512, 2)
            fresh.load_state_dict(self.cnn_best_state, strict=True)
            self.cnn.load_state_dict(fresh.state_dict(), strict=True)
            del fresh
            self.phase = 'rnn'
            self.apply_phase()
            _, recurrent = author_components()
            fresh_rnn = recurrent(self.ndims)
            self.rnn.load_state_dict(fresh_rnn.state_dict(), strict=True)
            self.rnn_initialized = True
            self.optimizers[1].state.clear()
            self.prepare_top(self.bundle.records('train'), next(self.parameters()).device)
        self.apply_phase()

    def should_validate(self, epoch):
        return self.phase == 'rnn' or (epoch+1) % self.test_every == 0

    @torch.no_grad()
    def infer_tiles(self, rows, device):
        self.cnn.eval()
        ds = FlatTiles(self, rows)
        loader = DataLoader(ds, batch_size=self.cnn_batch_size, shuffle=False, num_workers=0)
        probs = torch.empty(len(ds), dtype=torch.float32)
        offset = 0
        for image in loader:
            self.reserve(len(image), 'cnn')
            scores = self.cnn(image.to(device)).softmax(1)[:, 1].detach().cpu()
            if not torch.isfinite(scores).all():
                raise ValueError('Campanella CNN selection probabilities are nonfinite')
            probs[offset:offset+len(scores)] = scores
            offset += len(scores)
        return ds, probs.numpy()

    @torch.no_grad()
    def prepare_top(self, rows, device):
        if self.phase != 'rnn' or self.cnn_best_sha256 is None:
            raise ValueError('Recurrent tile selection needs a frozen validated CNN')
        missing = []
        for row in rows:
            p = resolve(self.bundle.root, row['path'])
            identity = digest_object({'grid': grid_sha256(self.reader.grid(row)), 'selector': self.cnn_best_sha256,
                                      'image_stamp': [p.stat().st_size, p.stat().st_mtime_ns]})
            cached = self.top_cache.get(row['slide_id'])
            if cached is None or cached['identity'] != identity:
                missing.append((row, identity))
        if not missing:
            return
        ds, scores = self.infer_tiles([r for r, _ in missing], device)
        groups = ds.group_ids()
        selected = self.selector.group_argtopk(groups, scores, self.steps)
        for slide, (row, identity) in enumerate(missing):
            flat = [int(i) for i in selected if groups[i] == slide]
            flat.sort(key=lambda i: (-float(scores[i]), ds.pair(i)[1]))
            indices = [int(ds.pair(i)[1]) for i in flat]
            if len(indices) != self.steps:
                raise ValueError('Native recurrent selection did not produce the registered number of real tiles')
            self.top_cache[row['slide_id']] = {'identity': identity, 'indices': indices,
                'probabilities': [float(scores[i]) for i in flat], 'selector_sha256': self.cnn_best_sha256,
                'locations_sha256': grid_sha256(self.reader.grid(row)[indices])}

    def embeddings(self, images):
        x = images
        for layer in list(self.cnn.children())[:-1]:
            x = layer(x)
        x = x.view(len(x), -1)
        return self.cnn.fc(x), x

    def recurrent(self, inputs, device):
        if len(inputs) != self.steps or self.phase != 'rnn' or not self.rnn_initialized:
            raise ValueError('Native recurrence requires the complete registered sequence and initialized stage2')
        self.cnn.eval()
        state = self.rnn.init_hidden(len(inputs[0])).to(device)
        for image in inputs:
            with torch.no_grad():
                _, embedding = self.embeddings(image.to(device))
            output, state = self.rnn(embedding, state)
        return output

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        self.eval()
        rows = bundle.records(subset)
        values = []
        if self.phase == 'cnn':
            ds, scores = self.infer_tiles(rows, device)
            groups = ds.group_ids()
            probabilities = self.selector.group_max(groups, scores, len(rows))
            values = [(float(p), int(p >= .5), None) for p in probabilities]
        else:
            self.prepare_top(rows, device)
            loader = DataLoader(RecurrentBags(self, rows), batch_size=self.rnn_batch_size, shuffle=False, num_workers=0)
            for inputs, target in loader:
                logits = self.recurrent(inputs, device)
                loss = self.criterion(device)(logits, target.to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Campanella RNN validation is nonfinite')
                p = logits.softmax(1)[:, 1].cpu().tolist()
                selection = logits.argmax(1).cpu().tolist()  # train source errors(), distinct from exported >=.5 at exact ties
                values.extend((float(a), int(b), float(loss)) for a, b in zip(p, selection))
        records = []
        for row, (probability, selection, loss) in zip(rows, values):
            record = {key: row[key] for key in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'target']}
            record.update(source_slides=row['slide_id'], p_0=1-probability, p_1=probability,
                          prediction=int(probability >= .5), decision_rule='binary_positive_at_half',
                          campanella_selection_prediction=selection, campanella_phase=self.phase,
                          campanella_ce_batch=loss, raw_level=self.reader.level,
                          raw_grid_sha256=grid_sha256(self.reader.grid(row)))
            if self.phase == 'rnn':
                selected = self.top_cache[row['slide_id']]
                record.update(raw_selector_sha256=self.cnn_best_sha256, raw_top_rows=json.dumps(selected['indices']),
                              raw_top_locations_sha256=selected['locations_sha256'])
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        y, pred = frame.target.to_numpy(), frame.campanella_selection_prediction.to_numpy()
        fpr, fnr = float(np.mean(pred[y == 0] == 1)), float(np.mean(pred[y == 1] == 0))
        metrics = {'campanella_balanced_accuracy': 1-(fpr+fnr)/2, 'campanella_fpr': fpr, 'campanella_fnr': fnr}
        if self.phase == 'rnn':
            metrics['campanella_ce'] = float(frame.campanella_ce_batch.mean())
        return metrics

    def after_validation(self, metrics):
        value = metrics['campanella_balanced_accuracy']
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Campanella selection requires a finite native balanced accuracy')
        if self.phase == 'cnn' and value >= self.cnn_best_score:
            self.cnn_best_score, self.cnn_best_epoch = value, self.epoch
            self.cnn_best_state = {k: v.detach().cpu().clone() for k, v in self.cnn.state_dict().items()}
            self.cnn_best_sha256 = state_sha256(self.cnn_best_state)

    def selection_improved(self, objective, best):
        return self.phase == 'rnn' and objective >= best  # latest validation tie, as in the author launcher

    def get_extra_state(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in ['phase', 'epoch', 'rnn_initialized', 'cnn_best_state',
                'cnn_best_sha256', 'cnn_best_epoch', 'cnn_best_score', 'cnn_updates', 'rnn_updates', 'top_cache', 'selected_training_rows']}

    def set_extra_state(self, state):
        expected = set(self.get_extra_state())
        if set(state) != expected or state['phase'] not in {'cnn', 'rnn'} or type(state['rnn_initialized']) is not bool or state['rnn_initialized'] != (state['phase'] == 'rnn'):
            raise ValueError('Invalid Campanella two-phase checkpoint state')
        if type(state['epoch']) is not int or not -1 <= state['epoch'] < self.cnn_epochs+self.rnn_epochs or (state['phase'] == 'rnn' and state['epoch'] < self.cnn_epochs):
            raise ValueError('Invalid Campanella checkpoint phase/epoch')
        for name in ['cnn_updates', 'rnn_updates']:
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError('Invalid Campanella optimizer update count')
        if state['cnn_best_state'] is not None:
            state = {**state, 'cnn_best_state': {k: v.detach().cpu().clone() for k, v in state['cnn_best_state'].items()}}
            if state_sha256(state['cnn_best_state']) != state['cnn_best_sha256']:
                raise ValueError('Campanella validated selector snapshot hash mismatch')
        if state['phase'] == 'rnn' and (not state['rnn_initialized'] or state['cnn_best_state'] is None):
            raise ValueError('Campanella stage2 checkpoint lacks its validated CNN/recurrence initialization')
        for name, value in state.items():
            setattr(self, name, copy.deepcopy(value))
        self.apply_phase()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.phase == 'rnn' and state_sha256(self.cnn.state_dict()) != self.cnn_best_sha256:
            raise ValueError('Campanella frozen encoder differs from its validated selector snapshot')
        return result

    def protocol(self):
        result = {'method': 'campanella_rnn', 'backbone': 'full native ResNet34, trained by hard-MIL then frozen',
                'initialization': self.reader.backbone, 'cnn_epochs': self.cnn_epochs, 'rnn_epochs': self.rnn_epochs,
                'cnn_batch_size': self.cnn_batch_size, 'rnn_batch_size': self.rnn_batch_size, 'cnn_k': self.cnn_k,
                'test_every': self.test_every, 'steps': self.steps, 'ndims': self.ndims, 'positive_weight': self.positive_weight,
                'shuffle_sequence': self.shuffle_sequence, 'phase': self.phase, 'epoch': self.epoch,
                'cnn_updates': self.cnn_updates, 'rnn_updates': self.rnn_updates,
                'cnn_best_epoch': self.cnn_best_epoch, 'cnn_best_score': None if self.cnn_best_state is None else self.cnn_best_score,
                'cnn_best_sha256': self.cnn_best_sha256, 'cached_top_grids': len(self.top_cache),
                'stage1': 'infer ALL TRAIN tiles every epoch; source lexsort/group_argtopk; Python random.sample; Adam1e-4/decay1e-4; CE; full native CNN and BN training',
                'stage2': f'BEST validated CNN; frozen512-dim features; native fc1(512,ndims)+fc2(state),ReLU,fc3; final-step-only CE; SGDNesterov lr{self.rnn_lr}/momentum.9/decay1e-4',
                'selection': 'CNN every test_every epochs, RNN every epoch; native balanced error, latest ties; RNN selection uses source topk/argmax, exported CSV uses positive probability>=.5',
                'grid_bridge': 'score real TRAIN/VAL/query grid using validated CNN, no labels; source top-k then descending probability, equal-score ties in original row order',
                'normalization': 'native streamed OpenSlide RGB->ToTensor->Normalize(mean.5,std.1); no augmentation; source CNN round(224*mult) and RNN int(224*mult) retained',
                'random_stream': 'one synchronous worker and unified checkpointed Python/NumPy/Torch RNG; no claim to recreate separate historic CLI process seeds',
                'repairs': ['coherent best/last, both optimizers, frozen best selector, top-grid identities and RNG',
                            'require enough real tiles rather than silent ragged sequence padding/truncation; explicit validation boundary at CNN stage end'],
                'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}
        if self.reader.cache is not None:
            result['region_storage'] = {'kind': self.reader.config['kind'],
                'manifest_sha256': self.reader.cache.manifest_sha256,
                'source_read_level': self.reader.level, 'source_mult': self.reader.mult,
                'native_cnn_read_size': self.reader.size('cnn'), 'native_rnn_read_size': self.reader.size('rnn'),
                'io_port': 'Full original measured read_region RGB pixels from a lossless cache; source resize and normalization unchanged, no whole-WSI claim'}
        return result


def campanella_epoch(model, bundle, groups, optimizers, device, cfg):
    if any(len(g) != 1 for g in groups):
        raise ValueError('Campanella requires independent slide bags with registered physical ROI coordinates')
    rows = [g[0] for g in groups]
    criterion = model.criterion(device)
    losses = []
    if model.phase == 'cnn':
        ds, scores = model.infer_tiles(rows, device)
        selected = model.selector.group_argtopk(ds.group_ids(), scores, model.cnn_k)
        pairs = random.sample([ds.pair(int(i)) for i in selected], len(selected))
        model.selected_training_rows = [(rows[s]['slide_id'], int(i)) for s, i in pairs]
        loader = DataLoader(FlatTiles(model, rows, pairs), batch_size=model.cnn_batch_size, shuffle=False, num_workers=0)
        model.cnn.train()
        for image, target in loader:
            logits = model.cnn(image.to(device))
            loss = criterion(logits, target.to(device))
            optimizer, network = optimizers[0], model.cnn
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise ValueError('Campanella hard-MIL CNN loss is nonfinite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                raise ValueError('Campanella hard-MIL CNN gradient is nonfinite')
            optimizer.step()
            check_update_finite(network, optimizer)
            model.cnn_updates += 1
            losses.append({'loss': float(loss.detach()), 'batch_size': len(target)})
    else:
        model.prepare_top(rows, device)
        loader = DataLoader(RecurrentBags(model, rows, model.shuffle_sequence), batch_size=model.rnn_batch_size, shuffle=True, num_workers=0)
        model.cnn.eval()
        model.rnn.train()
        for inputs, target in loader:
            model.rnn.zero_grad()
            output = model.recurrent(inputs, device)
            loss = criterion(output, target.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Campanella recurrent final-step loss is nonfinite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.rnn.parameters()):
                raise ValueError('Campanella recurrent gradient is nonfinite')
            optimizers[1].step()
            check_update_finite(model.rnn, optimizers[1])
            model.rnn_updates += 1
            pred = output.detach().argmax(1).cpu()
            losses.append({'loss': float(loss.detach()), 'batch_size': len(target),
                           'fp': int(((pred == 1) & (target == 0)).sum()),
                           'fn': int(((pred == 0) & (target == 1)).sum())})
    if not losses:
        raise ValueError('Campanella has no TRAIN batches')
    summary = {'loss': sum(v['loss']*v['batch_size'] for v in losses)/sum(v['batch_size'] for v in losses),
               'optimizer_updates': len(losses)}
    if model.phase == 'rnn':
        summary.update(fpr=sum(v['fp'] for v in losses)/sum(r['target'] == 0 for r in rows),
                       fnr=sum(v['fn'] for v in losses)/sum(r['target'] == 1 for r in rows))
    return [summary]

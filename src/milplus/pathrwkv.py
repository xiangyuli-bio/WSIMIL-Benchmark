"""Complete PathRWKV source recipe and explicitly repaired heterogeneous training."""
import contextlib
import math
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import digest_object
from .pathrwkv_backend import COMMIT, HEAD_SIZE, NATIVE_SEQUENCE_LIMIT, Backend
from .pathrwkv_data import PathInputs, TaskSet, TrainingSlides
from .pathrwkv_source import NativeNetwork, carry_token_shifts
from .recipes import cox_breslow_loss

PROFILES = {'pathrwkv_source', 'pathrwkv_repaired'}


def corrected_loss(predictions, targets, observed, specs):
    values = []
    for spec in specs:
        name = spec['name']
        valid = observed[name]
        if not valid.any():
            continue
        pred, target = predictions[name][valid], targets[name][valid]
        if spec['type'] == 'classification':
            values.append(F.cross_entropy(pred, target))
        elif spec['type'] == 'regression':
            values.append(F.smooth_l1_loss(pred.flatten(), target.flatten(), beta=1.))
        elif not target[:, 1].bool().any():
            # Preserve the source's zero contribution without manufacturing a
            # model gradient or applying decay to an unobserved survival head.
            values.append(torch.zeros((), device=pred.device, requires_grad=True))
        else:
            values.append(cox_breslow_loss(pred.float().flatten(), target[:, 0], target[:, 1]))
    if not values:
        raise ValueError('A PathRWKV training/evaluation sample needs an observed target')
    return sum(values)


class PathRWKVMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile='pathrwkv_repaired', backend='native',
                 hidden=768, depth=2, grids=1200, max_tiles=2000, batch_size=4,
                 inference_chunk=2000, precision='bf16-mixed', lrf=.1,
                 min_delta=1e-5, max_model_bytes=1024**3):
        super().__init__()
        if profile not in PROFILES:
            raise ValueError('Unknown PathRWKV profile')
        integers = [hidden, depth, grids, max_tiles, batch_size, inference_chunk, max_model_bytes]
        if any(type(x) is not int or x < 1 for x in integers) or depth < 2 or hidden % HEAD_SIZE:
            raise ValueError('PathRWKV requires positive integer budgets, depth>=2 and64-channel heads')
        if max_tiles > NATIVE_SEQUENCE_LIMIT or inference_chunk > NATIVE_SEQUENCE_LIMIT:
            raise ValueError('Pinned PathRWKV training/inference kernel limit is2000 tiles')
        if precision not in {'32', 'bf16-mixed', '16-mixed'}:
            raise ValueError('PathRWKV precision must be32, bf16-mixed or16-mixed')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in [lrf, min_delta]) or not 0 <= lrf <= 1 or min_delta < 0:
            raise ValueError('PathRWKV requires finite lrf in[0,1] and nonnegative min_delta')
        if torch.get_default_dtype() != torch.float32:
            raise ValueError('PathRWKV source initialization and kernels require float32 default dtype')
        if bundle.config.get('training_unit', 'slide') == 'patient' and (bundle.rows.groupby('global_patient_id').size() > 1).any():
            raise ValueError('PathRWKV trains individual slides; do not merge patient coordinate frames')
        self.bundle, self.profile = bundle, profile
        self.tasks = TaskSet(bundle)
        estimate = 4*((8*depth+len(self.tasks.specs))*hidden**2+bundle.dim*hidden+grids*hidden//2)
        if estimate > max_model_bytes:
            raise ValueError('PathRWKV estimated parameter/position budget exceeds max_model_bytes before allocation')
        self.inputs = PathInputs(bundle, grids)
        self.backend = Backend(backend)
        self.hidden, self.max_tiles, self.batch_size = hidden, max_tiles, batch_size
        self.inference_chunk, self.precision = inference_chunk, precision
        self.min_delta = min_delta
        self.network = NativeNetwork(SimpleNamespace(input_dim=bundle.dim,
            tasks=self.tasks.source_tasks, lr=1e-4, epochs=100, lrf=lrf), self.backend,
            hidden=hidden, depth=depth, grids=grids, inference_chunk=inference_chunk)
        self.settings = dict(zip(['hidden', 'depth', 'grids', 'max_tiles', 'batch_size',
                                 'inference_chunk', 'max_model_bytes'], integers))
        self.settings.update(backend=backend, precision=precision, lrf=lrf, min_delta=min_delta)
        self.contract = digest_object({'source': COMMIT, 'profile': profile,
            'bundle': bundle.fingerprint, 'tasks': self.tasks.contract,
            'geometry': self.inputs.contract, 'settings': self.settings})
        self.scheduler = None
        self.pending_scheduler = self.pending_scaler = None
        self.configured = False
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.completed_epochs = self.optimizer_steps = self.skipped_steps = self.noop_batches = 0
        self.batch_calls = 0
        self.early_best, self.early_wait, self.training_done = float('inf'), 0, False
        self.epochs, self.patience = 100, 10

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric'] != 'pathrwkv_val_loss':
            raise ValueError('PathRWKV owns tile sampling, uses no gradient clipping and selects distinct VAL joint loss')
        if len(self.bundle.records('train')) < self.batch_size:
            raise ValueError('PathRWKV source drop_last=True requires at least one full TRAIN batch')
        device = next(self.parameters()).device
        if self.precision == '16-mixed' and device.type != 'cuda':
            raise ValueError('PathRWKV16-mixed requires CUDA GradScaler')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.precision == '16-mixed')
        self.epochs, self.patience = cfg['epochs'], cfg['patience']
        self.network.args.epochs, self.network.args.lr = cfg['epochs'], cfg['lr']
        self.configured = True

    def make_optimizer(self, lr, weight_decay):
        self.network.args.lr = lr
        recipe = self.network.optimizer()
        optimizer = recipe['optimizer']
        for group in optimizer.param_groups:
            group['weight_decay'] = weight_decay
        self.scheduler = recipe['lr_scheduler']['scheduler']
        if self.pending_scheduler is not None:
            self.scheduler.load_state_dict(self.pending_scheduler)
            self.pending_scheduler = None
        if self.pending_scaler is not None:
            self.scaler.load_state_dict(self.pending_scaler)
            self.pending_scaler = None
        return optimizer

    def autocast(self, device):
        if self.precision == '32':
            return contextlib.nullcontext()
        return torch.autocast(device_type=torch.device(device).type,
            dtype=torch.bfloat16 if self.precision == 'bf16-mixed' else torch.float16)

    def objective(self, predictions, targets, observed):
        if self.profile == 'pathrwkv_source':
            value = self.network.source_loss(predictions, targets)
            if not isinstance(value, torch.Tensor):
                raise ValueError('Source -1 sentinel removed every target; use explicit corrected masks for valid -1 values')
            return value
        return corrected_loss(predictions, targets, observed, self.tasks.specs)

    def epoch(self, bundle, optimizer, device='cpu'):
        self.train()
        dataset = TrainingSlides(self.inputs, self.tasks, bundle.records('train'), self.max_tiles)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = {k: v.to(device) for k, v in batch['task_labels'].items()}
            observed = {k: v.to(device) for k, v in batch['observed'].items()}
            with self.autocast(device):
                outputs = self.network(batch['image_features'].to(device), batch['coords_yx'].to(device))
                loss = self.objective(outputs, targets, observed)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PathRWKV joint loss')
            self.scaler.scale(loss).backward()
            active = [p for p in self.parameters() if p.grad is not None]
            self.batch_calls += 1
            if not active:
                self.noop_batches += 1
            else:
                if self.precision != '16-mixed' and any(not torch.isfinite(p.grad).all() for p in active):
                    raise FloatingPointError('Nonfinite PathRWKV gradient')
                old = self.scaler.get_scale()
                self.scaler.step(optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < old:
                    self.skipped_steps += 1
                else:
                    self.optimizer_steps += 1
            losses.append({'loss': float(loss.detach()),
                           **{f'observed_{name}': float(mask.sum()) for name, mask in observed.items()}})
        return losses

    def finish_epoch(self, bundle, device, cfg):
        self.scheduler.step()
        self.completed_epochs += 1

    def selection_objective(self, metrics, epoch, objective):
        return -metrics['pathrwkv_val_loss']

    def after_validation(self, metrics):
        value = metrics['pathrwkv_val_loss']
        if self.early_best-value > self.min_delta:
            self.early_best, self.early_wait = value, 0
        else:
            self.early_wait += 1
        self.training_done = self.early_wait >= self.patience

    @torch.no_grad()
    def stream(self, chunks, device):
        backbone = self.network.backbone
        states = [torch.zeros(1, self.hidden//HEAD_SIZE, HEAD_SIZE, HEAD_SIZE,
                              dtype=torch.float32, device=device) for _ in backbone.blocks]
        maximum = torch.full((1, len(self.tasks.specs), self.hidden), -torch.inf, device=device)
        count = 0
        with carry_token_shifts(backbone):
            for x, coords in chunks:
                if len(x) < 1 or len(x) > self.inference_chunk:
                    raise ValueError('PathRWKV streaming chunk outside its declared inference budget')
                features = self.network.input_projector(x.to(device)[None])
                features = backbone.pos_embed(features, coords.to(device)[None])
                for i, block in enumerate(backbone.blocks):
                    features, states[i] = block(features, states[i])
                projected = backbone.MTL_module(features).view(1, len(x), len(self.tasks.specs), self.hidden)
                maximum = torch.maximum(maximum, projected.amax(1))
                count += len(x)
        if not count:
            raise ValueError('PathRWKV inference requires a nonempty slide')
        return self.network.head(maximum)

    def forward(self, x, coords=None):
        if x.ndim != 2 or x.shape[1] != self.bundle.dim or len(x) < 1 or coords is None or coords.shape != (len(x), 2):
            raise ValueError('PathRWKV requires one finite feature bag and real native224-grid yx coordinates')
        if not torch.isfinite(x).all() or not torch.isfinite(coords).all():
            raise ValueError('Nonfinite PathRWKV input')
        if not self.training and self.profile == 'pathrwkv_repaired':
            chunks = ((x[i:i+self.inference_chunk], coords[i:i+self.inference_chunk]) for i in range(0, len(x), self.inference_chunk))
            outputs = self.stream(chunks, x.device)
        else:
            outputs = self.network(x[None], coords[None])
        return {'outputs': outputs, 'logits': outputs[self.tasks.specs[0]['name']]}

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        self.eval()
        records = []
        # The source's sequential VAL/TEST DataLoader still consumes a global
        # base seed. Retain that draw, because it changes next-epoch sampling.
        loader = DataLoader(bundle.records(subset), batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=lambda batch: batch[0])
        for row in loader:
            with self.autocast(device):
                if self.profile == 'pathrwkv_repaired':
                    outputs = self.stream(self.inputs.chunks(row, self.inference_chunk), device)
                else:
                    x, coords = self.inputs.full(row)
                    outputs = self.network(x.to(device)[None], coords.to(device)[None])
                targets, observed = self.tasks.tensors([row], device)
                loss = self.objective(outputs, targets, observed)
            if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in outputs.values()):
                raise FloatingPointError('Nonfinite PathRWKV evaluation output/loss')
            record = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id']}
            record.update(subset=subset, source_slides=row['slide_id'], _pathrwkv_loss=float(loss.detach()))
            for i, spec in enumerate(self.tasks.specs):
                name, prefix = spec['name'], f't{i}_'
                known = bool(observed[name][0])
                prediction = outputs[name].float().flatten()
                record[prefix+'observed'] = known
                if spec['type'] == 'classification':
                    record[prefix+'target'] = int(targets[name][0]) if known else None
                    probabilities = prediction.softmax(0)
                    record.update({prefix+f'p{j}': float(v) for j, v in enumerate(probabilities)})
                elif spec['type'] == 'survival':
                    record.update({prefix+'time': float(targets[name][0, 0]) if known else None,
                                   prefix+'event': int(targets[name][0, 1]) if known else None,
                                   prefix+'risk': float(prediction[0])})
                else:
                    record.update({prefix+'target': float(targets[name][0]) if known else None,
                                   prefix+'prediction': float(prediction[0])})
            if bundle.task_type in {'classification', 'ordinal'}:
                record['target'] = row['target']
                record.update({f'p_{i}': record[f't0_p{i}'] for i in range(len(bundle.classes))})
            elif bundle.task_type == 'survival':
                record.update(time=row['time'], event=row['event'], risk=record['t0_risk'])
            elif bundle.task_type == 'regression':
                record['project'] = bundle.regression_targets.projects.loc[row['global_patient_id']]
                for i in range(len(self.tasks.specs)):
                    record.update({f'target_{i}': record[f't{i}_target'],
                                   f'raw_{i}': record[f't{i}_prediction'],
                                   f'prediction_{i}': record[f't{i}_prediction']})
            records.append(record)
        frame = pd.DataFrame(records)
        if bundle.task_type == 'survival':
            # Source training remains slide based; standard survival reporting
            # uses one mean risk per patient, preserving the original VAL loss.
            loss = float(frame._pathrwkv_loss.mean())
            group = frame.groupby('global_patient_id', sort=True)
            frame = group[['patient_id', 'time', 'event', 'subset']].first().join(group.risk.mean()).reset_index()
            frame['slide_id'] = 'patient:'+frame.global_patient_id
            frame['source_slides'] = group.slide_id.apply(';'.join).to_numpy()
            frame['_pathrwkv_loss'] = loss
        return frame

    def validation_metrics(self, frame):
        return {'pathrwkv_val_loss': float(frame._pathrwkv_loss.mean())}

    def protocol(self):
        return {'source_commit': COMMIT, 'profile': self.profile, 'settings': self.settings,
                'tasks': self.tasks.specs, 'completed_epochs': self.completed_epochs,
                'optimizer_steps': self.optimizer_steps, 'skipped_steps': self.skipped_steps,
                'noop_batches': self.noop_batches, 'batch_calls': self.batch_calls,
                'epochs': self.epochs, 'patience': self.patience, 'early_wait': self.early_wait,
                'early_best': self.early_best if math.isfinite(self.early_best) else None,
                'sampling': 'source randperm/truncate/zero-pad, shuffled full batches, drop_last=True, explicit num_workers0; source sequential VAL/TEST loader base-seed draw retained',
                'optimizer': 'source AdamW/LambdaLR cosine with configurable lr/wd; native defaults1e-4/.01/lrf.1',
                'selection': 'first strict best distinct VAL joint loss; independent early stop min_delta1e-5/patience10',
                'precision': self.precision, 'cuda_backend_receipt': self.backend.receipt,
                'inference': 'source whole-bag projection and reset shifts' if self.profile == 'pathrwkv_source' else 'chunked projection/MTL max plus both normalized token-shift states and WKV matrices',
                'loss': 'actual source -1 masks and Cox[B,1] broadcasting' if self.profile == 'pathrwkv_source' else 'explicit observed masks including valid -1 regression, flattened Cox with Breslow ties',
                'coordinates': self.inputs.spatial,
                'position_cache': 'exact factorized CPU-native1D tables, original yx orientation; invalid per-axis positions rejected',
                'probabilities': 'standard float32 softmax of emitted logits; not BF16-quantized TorchMetrics probabilities',
                'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

    def get_extra_state(self):
        return {'contract': self.contract, 'completed_epochs': self.completed_epochs,
                'epochs': self.epochs, 'patience': self.patience,
                'optimizer_steps': self.optimizer_steps, 'skipped_steps': self.skipped_steps,
                'noop_batches': self.noop_batches, 'batch_calls': self.batch_calls,
                'early_best': self.early_best, 'early_wait': self.early_wait,
                'training_done': self.training_done,
                'scheduler': self.pending_scheduler if self.scheduler is None else self.scheduler.state_dict(),
                'scaler': self.pending_scaler if self.pending_scaler is not None else self.scaler.state_dict()}

    def set_extra_state(self, state):
        if state['contract'] != self.contract:
            raise ValueError('PathRWKV source/task/geometry/configuration changed')
        if any(type(state[k]) is not int or state[k] < 1 for k in ['epochs', 'patience']):
            raise ValueError('Malformed PathRWKV training budget')
        if self.configured and (state['epochs'] != self.epochs or state['patience'] != self.patience):
            raise ValueError('PathRWKV training budget changed during resume')
        self.epochs, self.patience = state['epochs'], state['patience']
        for key in ['completed_epochs', 'optimizer_steps', 'skipped_steps', 'noop_batches', 'batch_calls', 'early_wait']:
            value = state[key]
            if type(value) is not int or value < 0:
                raise ValueError('Malformed PathRWKV checkpoint counters')
            setattr(self, key, value)
        if self.optimizer_steps+self.skipped_steps+self.noop_batches != self.batch_calls:
            raise ValueError('PathRWKV optimizer counter inconsistency')
        if self.completed_epochs > self.epochs or self.batch_calls != self.completed_epochs*(len(self.bundle.records('train'))//self.batch_size):
            raise ValueError('PathRWKV checkpoint must contain complete epoch boundaries')
        self.early_best, self.training_done = state['early_best'], state['training_done']
        if (type(self.training_done) is not bool or not isinstance(self.early_best, (float, int))
                or math.isnan(self.early_best) or (self.completed_epochs > 0 and not math.isfinite(self.early_best))):
            raise ValueError('Malformed PathRWKV selection state')
        if state['scheduler'] is not None:
            if self.scheduler is None:
                self.pending_scheduler = state['scheduler']
            else:
                self.scheduler.load_state_dict(state['scheduler'])
        if self.configured:
            self.scaler.load_state_dict(state['scaler'])
        else:
            self.pending_scaler = state['scaler']

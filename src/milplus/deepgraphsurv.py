"""Full2018 adaptive graph survival network and author epoch through a TF boundary."""
import contextlib
import copy
import io
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from .deepgraphsurv_network import NativeAGCN, TF1Adam, source_cox
from .deepgraphsurv_source import COMMIT, training_functions
from .io import digest_object


class DeepGraphSurv(nn.Module):
    owns_sampling = True
    handles_early_stopping = True
    minimum_selection_epoch = 5

    def __init__(self, bundle, profile='deepgraphsurv_agcn', batch_size=64, evaluation_seed=2018):
        super().__init__()
        if profile != 'deepgraphsurv_agcn' or bundle.task_type != 'survival' or bundle.deepgraphsurv is None:
            raise ValueError('DeepGraphSurv requires its original AGCN survival profile and frozen feature graph assets')
        if (type(batch_size) is not int or batch_size < 2 or type(evaluation_seed) is not int or
                not 0 <= evaluation_seed < 2**32):
            raise ValueError('Original AGCN requires batch_size>=2 and an explicit valid inference random seed')
        if not isinstance(bundle.config.get('time_unit'), str) or not bundle.config['time_unit'].strip():
            raise ValueError('DeepGraphSurv requires actual clinical time units')
        self.bundle, self.assets, self.profile = bundle, bundle.deepgraphsurv, profile
        self.batch_size, self.evaluation_seed = batch_size, evaluation_seed
        if len(self.assets.order['train']) < batch_size:
            raise ValueError('Original DeepGraphSurv needs at least one complete TRAIN graph batch; no fake patients')
        self.network = NativeAGCN(bundle.dim)
        self.author = training_functions(self, batch_size)
        self.completed_epochs = self.optimizer_steps = 0
        self.maximum_epochs, self.initial_lr = 100, .0002
        self.training_done = False
        self.optimizer = None
        self.last_trace = []
        self.last_native_train_ci = None
        self.ops = {key: key for key in ['pointclouds_pl', 'status_pl', 'laplacian_pl', 'size_x_pl',
            'is_training_pl', 'final_lr', 'learning_rate_pl', 'pred', 'graph_feature', 'loss', 'train_op', 'step']}
        self.ops['saver_op'] = None
        self.settings = {'profile': profile, 'dimension': bundle.dim, 'batch_size': batch_size,
            'max_nodes': self.assets.maximum, 'backbone_channels': [256, 256, 128],
            'attention_channels': [32, 1], 'Chebyshev_K': 3, 'dense_channels': [64, 1],
            'evaluation_seed': evaluation_seed, 'evaluation_random_policy': 'Dropout remains enabled; reproducible per subset and completed epoch, isolated from TRAIN random stream',
            'representation': 'All registered frozen feature channels; original VGG16/PCA preprocessing is a separate explicitly supplied encoder condition'}
        self.contract = digest_object({'source': COMMIT, 'bundle': bundle.fingerprint,
                                       'assets': self.assets.contract, 'settings': self.settings})

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None or
                cfg['selection_metric'] != 'deepgraphsurv_val_patient_ci' or cfg['epochs'] < 6):
            raise ValueError('DeepGraphSurv preserves source graph sampling, unclipped sum-Cox and validation every5epochs after epoch0; at least6epochs required')
        self.maximum_epochs = cfg['epochs']

    def make_optimizer(self, lr, weight_decay):
        if weight_decay != 0:
            raise ValueError('Original AGCN optimizes only the literal Cox sum; collected layer/graph penalties are not in its loss')
        self.initial_lr = float(lr)
        self.optimizer = TF1Adam(self.network.parameters(), lr=lr)
        return self.optimizer

    def prepare_train_data(self, *args):
        # Explicit manifest provider corrects the original five-return/four-
        # unpack mismatch without changing the author's epoch control flow.
        return self.assets.arrays('train')

    def retrieve_survival_labels_train(self, *args):
        return None

    def find_labels(self, slides, *args, **kwargs):
        rows = [self.assets.rows[slide] for slide in slides.tolist()]
        return (np.asarray([r['event'] for r in rows], np.float32),
                np.asarray([r['time'] for r in rows], np.float32),
                np.asarray([r['global_patient_id'] for r in rows]))

    def forward(self, x, laplacian, sizes, bn_decay=.5, dropout_masks=None):
        if (x.ndim != 3 or x.shape[0] < 2 or x.shape[-1] != self.bundle.dim or
                laplacian.shape != (x.shape[0], x.shape[1], x.shape[1]) or
                sizes.shape != (x.shape[0],) or sizes.dtype not in [torch.int32, torch.int64] or
                sizes.min() < 4 or sizes.max() > x.shape[1] or
                not torch.isfinite(x).all() or not torch.isfinite(laplacian).all()):
            raise ValueError('Original AGCN needs complete real padded feature graphs and actual node counts')
        result = self.network(x, laplacian, sizes, bn_decay=bn_decay, dropout_masks=dropout_masks)
        if not torch.isfinite(result['output']).all() or not torch.isfinite(result['graph_feature']).all():
            raise FloatingPointError('Nonfinite original adaptive graph risk or embedding')
        return result

    def run(self, fetches, feed_dict):
        """Explicit session I/O bridge; the actual author epoch owns all windows."""
        training = bool(feed_dict['is_training_pl'])
        expected = ['step', 'train_op', 'final_lr', 'loss', 'pred', 'graph_feature'] if training else ['pred', 'graph_feature']
        if fetches != expected:
            raise ValueError('Unsupported DeepGraphSurv native session fetch boundary')
        self.network.train(training)
        device = next(self.network.parameters()).device
        x = torch.from_numpy(feed_dict['pointclouds_pl']).to(device)
        lap = torch.from_numpy(feed_dict['laplacian_pl']).to(device)
        sizes = torch.from_numpy(feed_dict['size_x_pl']).to(device)
        schedule = (self.optimizer_steps*self.batch_size)//2000
        decay = min(.99, 1-.5*.5**schedule)
        lr = max(float(feed_dict.get('learning_rate_pl', self.initial_lr))*.8**schedule, 1e-8)
        if training:
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            self.optimizer.zero_grad()
            output = self(x, lap, sizes, bn_decay=decay)
            loss = source_cox(output['output'], torch.from_numpy(feed_dict['status_pl']).to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native AGCN sum-Cox objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError('DeepGraphSurv needs finite derivatives for every native parameter')
            predictions = output['output'].detach().cpu().numpy().copy()
            features = output['graph_feature'].detach().cpu().numpy().copy()
            value, step = float(loss.detach()), self.optimizer_steps
            self.optimizer.step()
            self.optimizer_steps += 1
            self.last_trace.append({'step': step, 'lr': lr, 'bn_decay': decay,
                                    'loss': value, 'events': feed_dict['status_pl'].tolist()})
            return [step, None, lr, value, predictions, features]
        with torch.no_grad():
            output = self(x, lap, sizes, bn_decay=decay)
        return [output['output'].cpu().numpy().copy(), output['graph_feature'].cpu().numpy().copy()]

    def epoch(self, bundle, optimizer, device='cpu'):
        if optimizer is not self.optimizer:
            raise ValueError('DeepGraphSurv optimizer differs from the session boundary')
        self.train()
        self.last_trace = []
        with contextlib.redirect_stdout(io.StringIO()):
            result, loss = self.author.train_one_epoch(self.completed_epochs, self, self.ops, self.initial_lr)
        if not self.last_trace or not math.isfinite(loss):
            raise ValueError('Original AGCN produced no finite mixed-event complete training batch')
        self.last_native_train_ci = float(result['train_ci']) if np.isfinite(result['train_ci']) else None
        record = {'loss': float(loss), 'source_train_ci_defined': self.last_native_train_ci is not None,
                  'optimizer_updates': len(self.last_trace)}
        if self.last_native_train_ci is not None:
            record['source_train_ci'] = self.last_native_train_ci
        return [record]

    def finish_epoch(self, bundle, device, cfg):
        self.completed_epochs += 1
        self.training_done = self.completed_epochs >= self.maximum_epochs

    def should_validate(self, epoch):
        return epoch > 0 and epoch % 5 == 0

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, config):
        self.eval()
        data, lap, sizes, slides = self.assets.arrays(subset)
        events, times, pids = self.find_labels(slides)
        gpu = next(self.network.parameters()).device
        devices = [gpu.index or 0] if gpu.type == 'cuda' else []
        # Keep original always-on inference dropout, while making selected-
        # checkpoint evaluation repeatable without perturbing TRAIN RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.evaluation_seed+2*self.completed_epochs+int(subset == 'test'))
            _, prognosis, average, oneshot, patient_ci, _ = self.author.retrieve_feature(
                self, self.ops, data, lap, sizes, times, events, pids, is_training=False)
        # The author averages -exp(logrisk) over each patient's WSIs. Export
        # the sign-reversed positive risk, rather than averaging log risks.
        predictions = []
        for gid in np.unique(pids):
            positions = np.flatnonzero(pids == gid)
            original = self.assets.rows[slides[positions[0]]]
            predictions.append({'slide_id': str(gid), 'patient_id': original['patient_id'],
                'global_patient_id': str(gid), 'source_slides': '|'.join(slides[positions].tolist()),
                'subset': subset, 'time': original['time'], 'event': original['event'],
                'native_time': float(times[positions[0]]), 'risk': float(-np.mean(prognosis[positions]))})
        frame = pd.DataFrame(predictions)
        frame.attrs['native_metrics'] = {key: float(value) if np.isfinite(value) else None for key, value in
            [('deepgraphsurv_patient_c_index', patient_ci), ('deepgraphsurv_wsi_c_index', oneshot),
             ('deepgraphsurv_batch_average_c_index', average)]}
        return frame

    def validation_metrics(self, frame):
        result = copy.deepcopy(frame.attrs['native_metrics'])
        if set(frame.subset) == {'val'}:
            result['deepgraphsurv_val_patient_ci'] = result['deepgraphsurv_patient_c_index']
        return result

    def get_extra_state(self):
        return {'contract': self.contract, 'completed_epochs': self.completed_epochs,
                'optimizer_steps': self.optimizer_steps, 'maximum_epochs': self.maximum_epochs,
                'initial_lr': self.initial_lr, 'training_done': self.training_done,
                'last_native_train_ci': self.last_native_train_ci}

    def set_extra_state(self, state):
        if state['contract'] != self.contract:
            raise ValueError('DeepGraphSurv checkpoint graph, full feature or source contract differs')
        for key in ['completed_epochs', 'optimizer_steps', 'maximum_epochs', 'initial_lr', 'training_done', 'last_native_train_ci']:
            setattr(self, key, state[key])

    def protocol(self):
        return {'source_commit': COMMIT, 'settings': copy.deepcopy(self.settings),
            'architecture': 'Literal TF1 AGCN attention32/1 and native256/256/128backbone, K3adaptive feature Laplacians, population BN/EMA and always-on20percent dropout, tanh64/tanh1risk',
            'training': 'Actual author train_one_epoch; mixed observed/censored windows shift by1, descending float32times, literal summed Cox, TF1Adam, source update-count LR and BN schedules; no unrequested penalties',
            'selection': 'Every5epochs after epoch0, first best patient CI on VAL; TEST sealed',
            'last_epoch_native_train_ci': self.last_native_train_ci,
            'integration_changes': ['Explicit actual slide/patient IDs replace filename-prefix clinical lookup',
                'Original prepare_data five-return/four-unpack and small-evaluation p_feature defects repaired at I/O boundaries',
                'AGCN branch explicitly selected; author top-level script currently hardcodes Chebyshev',
                'All input FM channels retained; original ImageNetVGG16/train-only128PCA is a separate supplied encoder condition',
                'Full attention/BN/optimizer/RNG checkpoint replaces incomplete backbone-only original saver',
                'Undefined native batch-CI diagnostics remain null instead of crashing valid Cox updates or singleton-tail evaluation; global patient validation CI must still be defined',
                'Inference dropout retained with explicit frozen per-subset/epoch random stream'],
            'synthetic': bool(self.bundle.config.get('synthetic', False))}

"""Complete PAMIL dynamic-policy reference trainer, independent main/PPO state."""
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .io import digest_object
from .pamil_policy_network import COMMIT, PAPER, PAMILPolicyReference


class SlideRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def one_row(items):
    return items[0]


class PAMILPolicyMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile=PAPER, bag_size=512, scheme='liis',
                 input_projection='auto_full_dimension', policy_lr=1e-5, policy_gamma=.1,
                 policy_epochs=3, action_std=.5, eps_clip=.2, layers=2, reward=1.,
                 sia_initial=.5, sia_final=0., stl_initial=0., stl_final=.1,
                 threshold_final=.9, threshold_schedule='train_bag_iteration', max_dense_bytes=32*1024**3):
        super().__init__()
        if bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('PAMIL dynamic policy requires individual slide classification bags')
        provenance = bundle.encoder.get('row_order_provenance')
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError('PAMIL interpolation requires actual encoder.row_order_provenance; never invent a physical grid')
        if input_projection == 'auto_full_dimension':
            projection = 'source_dimensions' if bundle.dim in {512, 1024} else 'learned_full_dim_extension'
        else:
            projection = input_projection
        for name, value in {'reward': reward, 'sia_initial': sia_initial, 'sia_final': sia_final,
                            'stl_initial': stl_initial, 'stl_final': stl_final, 'threshold_final': threshold_final}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError('PAMIL requires finite nonnegative '+name)
        if threshold_schedule not in {'train_bag_iteration', 'epoch_reference'}:
            raise ValueError('PAMIL threshold_schedule must explicitly name train_bag_iteration or epoch_reference')
        if reward <= 0 or stl_initial > .1 or stl_final > .1 or threshold_final > 1:
            raise ValueError('PAMIL requires positive reward, paper STL<=0.1 and threshold<=1')
        if type(max_dense_bytes) is not int or max_dense_bytes < 1:
            raise ValueError('PAMIL requires positive max_dense_bytes')
        self.bundle, self.profile = bundle, profile
        self.train_size = len(bundle.records('train'))
        self.core = PAMILPolicyReference(bundle.dim, len(bundle.classes), profile, bag_size, scheme,
            projection, policy_lr, policy_gamma, policy_epochs, action_std, eps_clip, layers)
        self.settings = {'bag_size': bag_size, 'scheme': scheme, 'input_projection': projection,
            'requested_input_projection': input_projection, 'policy_lr': policy_lr, 'policy_gamma': policy_gamma,
            'policy_epochs': policy_epochs, 'action_std': action_std, 'eps_clip': eps_clip, 'layers': layers,
            'reward': reward, 'sia_initial': sia_initial, 'sia_final': sia_final,
            'stl_initial': stl_initial, 'stl_final': stl_final, 'threshold_final': threshold_final,
            'threshold_schedule': threshold_schedule, 'max_dense_bytes': max_dense_bytes}
        self.contract = digest_object({'source': COMMIT, 'profile': profile, 'settings': self.settings,
            'feature_dimension': bundle.dim, 'classes': bundle.classes, 'row_order_provenance': provenance,
            'recipe': 'complete explicitly reconstructed paper WSL/STL/stopgradSIA/actual reward/PPO/AdaMax'})
        self.total_epochs, self.epoch_index, self.completed_epochs = 0, -1, 0
        self.main_updates, self.policy_updates, self.total_rollout_steps = 0, 0, 0
        self.train_orders, self.rollout_digests, self.last_epoch_audits = [], [], []

    def configure_training(self, cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None:
            raise ValueError('PAMIL uses complete registered bags and its own policy, without generic clipping/truncation')
        if cfg['selection_metric'] not in {'macro_auc', 'log_loss'}:
            raise ValueError('PAMIL requires an explicit benchmark VAL macro_auc or log_loss selection criterion')
        self.total_epochs = cfg['epochs']

    def make_optimizers(self, lr, weight_decay):
        main = [*self.core.base.parameters(), *self.core.fusion.parameters(), *self.core.head.parameters()]
        # Construct against the actual parameters after outer model.to(device),
        # rather than retaining an optimizer created by a CPU-only factory.
        self.core.ppo.optimizer = torch.optim.Adam(self.core.policy.parameters(),
            lr=self.settings['policy_lr'], betas=(.9, .999))
        return [torch.optim.Adamax(main, lr=lr, weight_decay=weight_decay), self.core.ppo.optimizer]

    def reserve(self, count):
        width = min(count, self.settings['bag_size']) if self.profile == PAPER else self.settings['bag_size']
        steps = math.ceil(count/self.settings['bag_size'])
        estimate = 64*self.bundle.dim*count+128*width*width*steps*8*self.settings['layers']
        if estimate > self.settings['max_dense_bytes']:
            raise ValueError('PAMIL full native attention/policy rollout exceeds max_dense_bytes before graph allocation')

    def prepare(self, bundle, seed, device):
        for row in bundle.rows.to_dict('records'):
            value, _ = bundle.load(row)
            self.reserve(len(value))

    def start_epoch(self, epoch):
        self.epoch_index = epoch

    def coefficients(self):
        if self.total_epochs < 1:
            raise ValueError('Configure complete PAMIL training schedule before use')
        epoch = max(0, self.epoch_index)
        alpha = .5*(1+np.cos(np.pi*epoch/self.total_epochs))
        sia = self.settings['sia_final']+(self.settings['sia_initial']-self.settings['sia_final'])*alpha
        stl = self.settings['stl_final']+(self.settings['stl_initial']-self.settings['stl_final'])*alpha
        progress = (epoch/max(1, self.total_epochs-1) if self.settings['threshold_schedule'] == 'epoch_reference'
                    else self.main_updates/max(1, self.total_epochs*self.train_size-1))
        threshold = self.settings['threshold_final']*min(1., progress)
        return float(sia), float(stl), float(threshold)

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1 or config.get('max_instances') is not None:
            raise ValueError('PAMIL requires one complete ordered bag per step')
        row = rows[0]
        value, _ = bundle.load(row); self.reserve(len(value))
        sia, stl, threshold = self.coefficients()
        label = torch.tensor([row['target']], dtype=torch.long, device=device) if train else None
        result = self.core(value.to(device), label, sia, stl, self.settings['reward'], threshold,
                           update_policy=train)
        if train:
            flattened = [i for group in result['selected_row_ids'] for i in group]
            if any(type(i) is not int or i < 0 or i >= len(value) for i in flattened):
                raise ValueError('PAMIL rollout contains row identities outside its actual TRAIN bag')
            unique = len(set(flattened))
            if self.profile == PAPER and (unique != len(value) or len(flattened) != len(value)):
                raise ValueError('PAMIL paper masking must visit every original patch exactly once')
            audit = {'slide_id': row['slide_id'], 'original_rows': len(value), 'sampled_rows': len(flattened),
                'unique_rows': unique, 'steps': result['steps'],
                'selection_sha256': digest_object(result['selected_row_ids'])}
            result['audit'] = audit
        return result

    def epoch(self, bundle, optimizer, device):
        loader = DataLoader(SlideRows(bundle.records('train')), batch_size=1,
            sampler=RandomSampler(bundle.records('train')), num_workers=0, collate_fn=one_row)
        losses, order, digests, audits = [], [], [], []
        for row in loader:
            # PPO uses detached current tokens and detached TRAIN reward; its
            # independent optimizer can run before the main objective backward.
            # Public training order was not released; this order is explicit.
            result = self.forward_unit(bundle, [row], device, {}, train=True)
            optimizer.zero_grad()
            result['loss'].backward(); optimizer.step()
            self.main_updates += 1
            self.policy_updates += self.settings['policy_epochs']
            self.total_rollout_steps += result['steps']
            order.append(row['slide_id']); digests.append(result['audit']['selection_sha256'])
            audits.append(result['audit'])
            losses.append({k: float(result[k].detach()) for k in ['loss', 'wsl', 'stl', 'sia']})
        self.train_orders.append(order); self.rollout_digests.append(digests)
        self.last_epoch_audits = audits
        self.completed_epochs = self.epoch_index+1
        return losses

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        self.eval()
        loader = DataLoader(SlideRows(bundle.records(subset)), batch_size=1,
            sampler=SequentialSampler(bundle.records(subset)), num_workers=0, collate_fn=one_row)
        frame = []
        for row in loader:
            result = self.forward_unit(bundle, [row], device, cfg, train=False)
            record = {k: row[k] for k in ['slide_id', 'patient_id', 'global_patient_id', 'subset', 'target']}
            record.update(source_slides=row['slide_id'], pamil_steps=result['steps'],
                          pamil_selection_sha256=digest_object(result['selected_row_ids']))
            record.update({f'p_{i}': float(p) for i, p in enumerate(result['probabilities'][0])})
            frame.append(record)
        return pd.DataFrame(frame)

    def get_extra_state(self):
        return {'contract': self.contract, 'total_epochs': self.total_epochs,
            'epoch_index': self.epoch_index, 'completed_epochs': self.completed_epochs,
            'main_updates': self.main_updates, 'policy_updates': self.policy_updates,
            'total_rollout_steps': self.total_rollout_steps, 'train_orders': self.train_orders,
            'rollout_digests': self.rollout_digests, 'last_epoch_audits': self.last_epoch_audits}

    def set_extra_state(self, state):
        expected = set(self.get_extra_state())
        if not isinstance(state, dict) or set(state) != expected or state.get('contract') != self.contract:
            raise ValueError('PAMIL full policy/recipe/ordered input contract changed')
        if self.total_epochs and state['total_epochs'] != self.total_epochs:
            raise ValueError('PAMIL complete cosine schedule horizon changed')
        numeric = ['total_epochs', 'completed_epochs', 'main_updates', 'policy_updates', 'total_rollout_steps']
        if any(type(state[k]) is not int or state[k] < 0 for k in numeric):
            raise ValueError('PAMIL checkpoint training counters invalid')
        if state['completed_epochs'] > state['total_epochs']:
            raise ValueError('PAMIL checkpoint exceeds its complete training horizon')
        if type(state['epoch_index']) is not int or state['epoch_index'] != state['completed_epochs']-1:
            raise ValueError('PAMIL checkpoint epoch state inconsistent')
        allowed = {r['slide_id'] for r in self.bundle.records('train')}
        if (not isinstance(state['train_orders'], list) or len(state['train_orders']) != state['completed_epochs']
                or any(not isinstance(order, list) or len(order) != len(allowed) or set(order) != allowed for order in state['train_orders'])):
            raise ValueError('PAMIL checkpoint must contain each actual TRAIN bag once per epoch')
        if (state['main_updates'] != len(allowed)*state['completed_epochs']
                or state['policy_updates'] != state['main_updates']*self.settings['policy_epochs']):
            raise ValueError('PAMIL main and independent PPO update counts differ')
        if (not isinstance(state['rollout_digests'], list) or len(state['rollout_digests']) != state['completed_epochs']
                or any(not isinstance(x, list) or len(x) != len(allowed) or
                       any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in x)
                       for x in state['rollout_digests'])):
            raise ValueError('PAMIL ordered TRAIN rollout digests invalid')
        audits = state['last_epoch_audits']
        if (not isinstance(audits, list) or state['completed_epochs'] and
                (len(audits) != len(allowed) or {a.get('slide_id') for a in audits} != allowed)):
            raise ValueError('PAMIL last epoch rollout identity audit invalid')
        steps = {}
        sizes = {}
        for row in self.bundle.records('train'):
            value, _ = self.bundle.load(row)
            sizes[row['slide_id']] = len(value)
            steps[row['slide_id']] = (math.ceil(len(value)/self.settings['bag_size']) if self.profile == PAPER
                                     else max(1, len(value)//self.settings['bag_size']))
        if state['total_rollout_steps'] != state['completed_epochs']*sum(steps.values()):
            raise ValueError('PAMIL complete TRAIN rollout-step count differs from registered bag sizes')
        if state['completed_epochs']:
            if [a.get('slide_id') for a in audits] != state['train_orders'][-1]:
                raise ValueError('PAMIL audit order differs from actual TRAIN sampler order')
            for audit, digest in zip(audits, state['rollout_digests'][-1]):
                sid = audit['slide_id']
                required = {'slide_id', 'original_rows', 'sampled_rows', 'unique_rows', 'steps', 'selection_sha256'}
                if (set(audit) != required or audit['original_rows'] != sizes[sid]
                        or audit['steps'] != steps[sid] or audit['selection_sha256'] != digest
                        or any(type(audit[k]) is not int for k in ['original_rows', 'sampled_rows', 'unique_rows', 'steps'])
                        or not 0 < audit['unique_rows'] <= audit['original_rows']):
                    raise ValueError('PAMIL registered row identity/count/digest audit differs')
                if self.profile == PAPER:
                    if audit['sampled_rows'] != sizes[sid] or audit['unique_rows'] != sizes[sid]:
                        raise ValueError('PAMIL paper masking audit lost or duplicated original rows')
                elif audit['sampled_rows'] != steps[sid]*self.settings['bag_size']:
                    raise ValueError('PAMIL released replacement sampling audit has wrong group size')
        for key in expected-{'contract'}:
            setattr(self, key, state[key])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in self.core.policy.state_dict():
            current, old = 'core.policy.'+name, 'core.policy_old.'+name
            if (current not in state_dict or old not in state_dict
                    or not torch.equal(state_dict[current].cpu(), state_dict[old].cpu())):
                raise ValueError('PAMIL policy_old must be the actual synchronized source policy at an epoch checkpoint')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def protocol(self):
        return {'source': COMMIT, 'profile': self.profile, 'settings': self.settings,
            'native_hidden': 512, 'policy_hidden': 512, 'native_attention_heads': 8,
            'feature_dimension': self.bundle.dim, 'classes': self.bundle.classes,
            'row_order_provenance': self.bundle.encoder['row_order_provenance'],
            'full_recipe': 'Explicit complete paper-reference training, public source has no complete trainer',
            'objective': 'WSL CE + cosine-weighted STL CE + symmetric previous/current predictor negative-cosine stop-gradient SIA; missing first predecessor contributes0 and sum dividedbyT',
            'reward': 'Actual TRAIN bag correctness times disclosed positive reward (1CAMELYON/2Lung) minus actual CLS/fused-token cosine, detached before PPO',
            'optimization': 'Paper AdaMax1e-4 wd1e-5,300epochs; independent original Gaussian-clipped GRU actor/critic PPO Adam1e-5 gamma0.1 K3 clip0.2,0.5valueMSE,0.01entropy. PPO step before independent main backward declared; actions/state at all T including terminal draw as released inference/PPO buffers.',
            'schedules': 'Default xi advances at every TRAIN-bag iteration, first0/last0.9, using saved main-update count; optional epoch_reference names the earlier coarser ablation. Public training cadence not released. Source cosine function, explicit SIA0.5to0 andSTL0to0.1 reference endpoints; full historical endpoints not disclosed. Literal GMSS P>xi withxi0to0.9; supplement exploration narrative does not match this threshold direction.',
            'sampling': ('Paper LIIS actual preceding row-ID interpolation/topM or explicit cosine-GMSS/GHSS far-rank2M; random first group, unique complete masking, short final group, no fabricated physical coordinates'
                         if self.profile == PAPER else 'Actual released normalized-index interpolation/weighted replacement, repeat/pad/floorT, retain first pool then removeunique sampled IDs; explicit empty-padding/allzero-score numeric fixes'),
            'history': ('Paper-reference fusion reads previous fused tokens and TCM consumes fused sequence; named changes to released raw-token routing'
                        if self.profile == PAPER else 'Actual released SFFR/TCM raw-token history, distinct from paper fused-history equation'),
            'projection': 'Source512 bypass/1024fc1; otherfullD or forcedfullD via explicitly recorded learned projection port; original inactive declared parameters retained',
            'inference': 'Source label-independent TCM probability only, not printed true-label-conditional Eq5. Policy remains stochastic as released; complete allRNG restores repeated full-run predictions. TEST labels never choose tokens or output route.',
            'selection': 'Benchmark first best VAL macro_auc or explicit log_loss, not an author-released rule. Full configured epochs, no inferred early stop or test selection.',
            'device_shape_ports': 'Registered action-scale buffer, actual tensor-device zeros, no hardcoded.cuda, scalar action reshape, finiteT1 normalization; original512/T>1 numerical kernel retained',
            'completed_epochs': self.completed_epochs, 'main_updates': self.main_updates,
            'policy_updates': self.policy_updates, 'total_rollout_steps': self.total_rollout_steps,
            'last_epoch_audits': self.last_epoch_audits,
            'checkpoint': 'Complete base/fusion/TCM/actual policy/policy_old/Gaussian scale, both optimizers, all epoch/train-order/selection digests/counters and allRNG; episode memory resets at each bag, checkpoints at epoch boundaries',
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}

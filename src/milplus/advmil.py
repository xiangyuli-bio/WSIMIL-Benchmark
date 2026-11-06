"""Pinned AdvMIL continuous survival with alternating adversarial optimization.

The source and corrected discriminator BCE are distinct execution profiles.
Graph/cluster backbones and naturally unlabeled cohorts are not implied by
the ESAT/ABMIL and explicit label-masking experiments implemented here.
"""
import hashlib
import math
from types import SimpleNamespace

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F

from .region_data import patient_regions
from .upstream import load_definitions


def author_components():
    scope = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'List': list,
             'SimpleNamespace': SimpleNamespace, 'dist': torch.distributions}
    files = [
        ('utils/func.py', ['generate_noise'], []),
        ('model/backbone_utils.py', ['Attn_Net_Gated', 'GAPool', 'sequence2square', 'square2sequence',
            'make_embedding_layer', 'make_transformer_layer', 'AVGPoolPatchEmbedding'],
            [('x.transpose(2, 1).view(size[0]//L, -1, size[1])',
              'x.transpose(2, 1).reshape(size[0]//L, -1, size[1])')]),
        ('model/model_utils.py', ['init_weights', 'get_hop_dims', 'make_noise_mlp_layer',
            'make_efficient_mlp_layer', 'make_mlp_layer', 'make_embedding_y_layer', 'EmbedXLayer'], []),
        ('model/backbone.py', ['ABMIL', 'DualTrans_HS', 'load_backbone_param'], []),
        ('model/GANSurv.py', ['Generator', 'PrjDiscriminator'], []),
        ('loss/utils.py', ['recon_loss', 'loss_reg_l1', 'real_fake_loss', 'fake_generator_loss'], []),
        ('optim/optim_factory.py', ['add_weight_decay'], []),
    ]
    for path, names, replacements in files:
        module = load_definitions('advmil', path, names, scope, replacements)
        scope.update({name: getattr(module, name) for name in names})
    return SimpleNamespace(**scope)


def corrected_bce(real, fake):
    loss = F.softplus(fake).mean()
    return loss if real is None else loss + F.softplus(-real).mean()


class AdversarialSurvival(nn.Module):
    def __init__(self, dim, time_unit, profile='advmil_esat', hidden=384, discriminator_dim=128,
                 dropout=.6, discriminator_dropout=.25, noise_distribution='uniform',
                 batch_size=16, generator_updates=1, gan_weight=.004, l1_weight=1e-5,
                 validation_samples=1, test_samples=30, inference_seed=2026, selection_warmup=5):
        super().__init__()
        if profile not in {'advmil_esat', 'advmil_abmil', 'advmil_esat_source', 'advmil_abmil_source'}:
            raise ValueError('Unknown AdvMIL profile')
        if any(type(v) is not int or v < 1 for v in (dim, hidden, discriminator_dim, batch_size,
                generator_updates, validation_samples, test_samples)) or hidden < 4 or discriminator_dim < 2:
            raise ValueError('AdvMIL dimensions and budgets must be positive integers; hidden>=4 and D>=2')
        if 'esat' in profile and hidden % 8:
            raise ValueError('AdvMIL ESAT hidden dimension must be divisible by 8')
        if time_unit not in {'months', 'days'} or noise_distribution not in {'uniform', 'gaussian'}:
            raise ValueError('Declare time_unit months/days and uniform/gaussian noise')
        if not all(math.isfinite(v) and 0 <= v < 1 for v in (dropout, discriminator_dropout)):
            raise ValueError('Invalid AdvMIL dropout')
        if not math.isfinite(gan_weight) or gan_weight <= 0 or not math.isfinite(l1_weight) or l1_weight < 0:
            raise ValueError('AdvMIL requires positive GAN weight and nonnegative L1')
        if type(selection_warmup) is not int or selection_warmup < 0 or type(inference_seed) is not int:
            raise ValueError('Invalid selection warmup or inference seed')
        self.author = author_components()
        self.profile, self.time_unit = profile, time_unit
        self.dim, self.hidden, self.discriminator_dim = dim, hidden, discriminator_dim
        mode = 'patch' if 'esat' in profile else 'abmil'
        args, kwargs = self.author.load_backbone_param(mode, [dim, hidden, hidden])
        backbone = (self.author.DualTrans_HS if mode == 'patch' else self.author.ABMIL)(*args, **kwargs)
        noise = SimpleNamespace(noise=[0, 1], hops=1, noise_dist=noise_distribution)
        self.generator = self.author.Generator(hidden, 1, backbone, noise, norm=False, dropout=dropout, out_scale='sigmoid')
        self.generator.apply(self.author.init_weights)
        self.discriminator = self.author.PrjDiscriminator(
            SimpleNamespace(in_dim=dim, out_dim=discriminator_dim, ksize=1, backbone='avgpool', dropout=discriminator_dropout),
            SimpleNamespace(in_dim=1, hid_dims=[discriminator_dim//2, discriminator_dim], norm=False, dropout=0.),
            prj_path='x', inner_product='instance')
        self.batch_size, self.generator_updates = batch_size, generator_updates
        self.gan_weight, self.l1_weight = gan_weight, l1_weight
        self.validation_samples, self.test_samples = validation_samples, test_samples
        self.inference_seed, self.minimum_selection_epoch = inference_seed, selection_warmup
        self.register_buffer('time_scale', torch.tensor(0.))
        self.plan = {'mode': 'supervised', 'visible': [], 'folds': []}
        self.scheduler, self.epoch, self.grad_clip = None, 0, None

    def configure_training(self, cfg):
        if cfg['epochs'] <= self.minimum_selection_epoch:
            raise ValueError('AdvMIL epochs must exceed selection_warmup; set warmup explicitly for short smoke runs')
        if cfg['max_instances'] is not None and (type(cfg['max_instances']) is not int or cfg['max_instances'] < 16):
            raise ValueError('AdvMIL max_instances must retain at least 16 patches')
        if cfg['grad_clip'] is not None and (not math.isfinite(cfg['grad_clip']) or cfg['grad_clip'] <= 0):
            raise ValueError('grad_clip must be positive finite or null')
        self.grad_clip = cfg['grad_clip']

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        table = bundle.rows[bundle.rows.subset == 'train'].drop_duplicates('global_patient_id').set_index('global_patient_id')
        config = bundle.config.get('adversarial_training', {})
        if not isinstance(config, dict):
            raise ValueError('adversarial_training must be a mapping')
        mode = config.get('mode', 'supervised')
        if mode not in {'supervised', 'UD+LD', 'UD', 'LD'}:
            raise ValueError('Unknown AdvMIL label visibility mode')
        allowed = {'mode'} if mode == 'supervised' else {'mode', 'visible_training_patients', 'unlabeled_folds', 'label_provenance'}
        if set(config) - allowed or (mode != 'supervised' and set(config) != allowed):
            raise ValueError('Semi-supervised masking requires explicit train IDs, folds and label_provenance')
        visible = list(table.index) if mode == 'supervised' else config['visible_training_patients']
        if (not isinstance(visible, list) or not visible or any(not isinstance(p, str) for p in visible) or
                len(visible) != len(set(visible)) or not set(visible) <= set(table.index)):
            raise ValueError('Visible IDs must be unique nonempty global training-patient IDs')
        hidden = sorted(set(table.index) - set(visible))
        folds = []
        if mode != 'supervised':
            n = config['unlabeled_folds']
            if (not hidden or type(n) is not int or not 1 <= n <= len(hidden) or
                    not isinstance(config['label_provenance'], str) or not config['label_provenance'].strip()):
                raise ValueError('Invalid unlabeled folds or label provenance')
            folds = [hidden] if n == 1 else [[hidden[i] for i in idx] for _, idx in KFold(n, shuffle=True, random_state=seed).split(hidden)]
        self.plan = {'mode': mode, 'visible': sorted(visible), 'folds': folds,
                     'label_provenance': config.get('label_provenance', bundle.config.get('label_provenance'))}
        self.time_scale.fill_(float(table.loc[visible].time.max()))

    def start_epoch(self, epoch):
        self.epoch = epoch

    def active_groups(self, groups):
        mode = self.plan['mode']
        if mode == 'supervised':
            return groups
        active = set(self.plan['visible']) if mode in {'UD+LD', 'LD'} else set()
        if mode == 'UD+LD':
            active.update(self.plan['folds'][self.epoch % len(self.plan['folds'])])
        elif mode == 'UD':
            # Author UD uses the entire unlabeled loader; only UD+LD cycles folds.
            active.update(patient for fold in self.plan['folds'] for patient in fold)
        return [rows for rows in groups if rows[0]['global_patient_id'] in active]

    def make_optimizers(self, lr, weight_decay):
        generator = torch.optim.Adam(self.author.add_weight_decay(self.generator, weight_decay), lr=lr)
        discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=0.)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generator, mode='min', factor=.5, patience=10)
        return [generator, discriminator]

    def get_extra_state(self):
        return {'plan': self.plan, 'scheduler': None if self.scheduler is None else self.scheduler.state_dict()}

    def set_extra_state(self, state):
        self.plan = state['plan']
        if self.scheduler is not None and state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])

    def after_validation(self, metrics):
        self.scheduler.step(metrics['survival_loss'])

    def d_loss(self, real, fake):
        return self.author.real_fake_loss(real, fake, which='bce') if self.profile.endswith('_source') else corrected_bce(real, fake)

    def reconstruction(self, prediction, time, event):
        return self.author.recon_loss(prediction, time, event, alpha=0., gamma=0., norm='l1')

    def _step(self, loss, network, optimizer):
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Nonfinite AdvMIL objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), self.grad_clip or float('inf'), error_if_nonfinite=True)
        optimizer.step()

    def discriminator_step(self, xs, targets, visible, optimizer):
        self.generator.eval()
        self.discriminator.train()
        real, fake = [], []
        for x, y, observed in zip(xs, targets, visible):
            if observed and y[0, 1] == 1:
                real.append(self.discriminator(x, y[:, [0]]).view(-1))
            pred = self.generator(x, None)
            fake.append(self.discriminator(x, pred.detach()).view(-1))
        loss = self.d_loss(torch.cat(real) if real else None, torch.cat(fake))
        self._step(loss, self.discriminator, optimizer)
        return float(loss.detach())

    def generator_step(self, xs, targets, visible, optimizer):
        self.generator.train()
        self.discriminator.eval()
        predictions, fake = [], []
        for x in xs:
            pred = self.generator(x, None)
            predictions.append(pred)
            fake.append(self.discriminator(x, pred).view(-1))
        adversarial = self.author.fake_generator_loss(torch.cat(fake))
        reconstruction = adversarial.new_zeros(())
        if any(visible):
            prediction = torch.cat([p for p, v in zip(predictions, visible) if v])
            y = torch.cat([t for t, v in zip(targets, visible) if v])
            reconstruction = self.reconstruction(prediction, y[:, [0]], y[:, [1]])
        regularization = self.author.loss_reg_l1(self.l1_weight)(self.generator.parameters())
        loss = reconstruction + self.gan_weight * adversarial + regularization
        self._step(loss, self.generator, optimizer)
        return {'loss': float(loss.detach()), 'reconstruction': float(reconstruction.detach()),
                'generator_adversarial': float(adversarial.detach()),
                'generator_l1': float(regularization.detach()) if isinstance(regularization, torch.Tensor) else regularization}

    def forward_unit(self, bundle, rows, device, config, train=False):
        if train:
            raise ValueError('AdvMIL uses its alternating trainer')
        if self.time_scale <= 0:
            raise RuntimeError('AdvMIL needs train-only fitted time scaling')
        x = patient_regions(bundle, rows, device, config['max_instances'])
        key = f"{self.inference_seed}:{rows[0]['global_patient_id']}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)
        count = self.validation_samples if rows[0]['subset'] == 'val' else self.test_samples
        # Author noise is sampled on CPU. Do not alter training or CUDA RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            single = self.generator(x, None)
            draws = single.unsqueeze(0) if count == 1 else torch.stack([self.generator(x, None) for _ in range(count)])
        normalized = draws.median(0).values
        prediction = normalized * self.time_scale
        return {'logits': -prediction, 'prediction_time': prediction, 'single_time': single*self.time_scale,
                'time_std': draws.std(0, unbiased=False)*self.time_scale, 'normalized_time': normalized}

    def prediction_fields(self, result, row):
        t = self.time_scale.new_tensor([[row['time']]]) / self.time_scale
        e = self.time_scale.new_tensor([[row['event']]])
        return {'predicted_time': float(result['prediction_time'].squeeze()),
                'single_draw_time': float(result['single_time'].squeeze()),
                'predicted_time_std': float(result['time_std'].squeeze()),
                'survival_loss': float(self.reconstruction(result['normalized_time'], t, e))}

    def validation_metrics(self, predictions):
        return {'survival_loss': float(predictions.survival_loss.mean())}

    def protocol(self):
        return {'profile': self.profile, 'backbone': 'ESAT' if 'esat' in self.profile else 'author gated ABMIL',
                'discriminator': 'author region-level inner-product projection (RLIP), x projection',
                'discriminator_loss': 'literal released fake log(sigmoid)-1' if self.profile.endswith('_source') else 'explicit repaired real/fake BCE via softplus',
                'generator_loss': 'censored L1 reconstruction + GAN * (-mean raw discriminator score) + all-generator L1',
                'gan_weight': self.gan_weight, 'l1_weight': self.l1_weight, 'batch_size': self.batch_size,
                'generator_updates': self.generator_updates, 'tail_policy': 'retain incomplete final patient batch (source drops it)',
                'time_unit': self.time_unit, 'time_scale': float(self.time_scale),
                'scaling': 'maximum visible TRAIN time only; no target clipping; sigmoid prediction bounded by training maximum',
                'label_plan': self.plan, 'semi_supervision_scope': 'known training outcomes explicitly masked; not naturally missing outcomes',
                'regions': 'explicit IDs, complete physical 4x4 grids, row-major ordering; budget samples whole regions',
                'inference': 'isolated per-patient CPU RNG; one initial draw, then lower-median of M fresh draws if M>1',
                'validation_samples': self.validation_samples, 'test_samples': self.test_samples,
                'optimizer': 'two Adam optimizers; G bias/1D decay exclusion; D no weight decay',
                'scheduler': 'G ReduceLROnPlateau on validation reconstruction, factor .5 patience 10',
                'selection_warmup': self.minimum_selection_epoch,
                'scope': 'continuous ESAT/ABMIL; not graph/cluster/discrete variants or paper numerical reproduction'}


def adversarial_epoch(model, bundle, groups, optimizers, device, cfg):
    groups = model.active_groups(groups)
    history = []
    for offset in range(0, len(groups), model.batch_size):
        batch = groups[offset:offset + model.batch_size]
        xs = [patient_regions(bundle, rows, device, cfg['max_instances'], train=True) for rows in batch]
        visible = [rows[0]['global_patient_id'] in model.plan['visible'] for rows in batch]
        targets = [model.time_scale.new_tensor([[rows[0]['time']/float(model.time_scale), rows[0]['event']]])
                   if v else model.time_scale.new_zeros((1, 2)) for rows, v in zip(batch, visible)]
        discriminator = model.discriminator_step(xs, targets, visible, optimizers[1])
        for _ in range(model.generator_updates):
            result = model.generator_step(xs, targets, visible, optimizers[0])
        history.append({**result, 'discriminator_loss': discriminator, 'patients': len(batch), 'visible_patients': sum(visible)})
    if not history:
        raise ValueError('AdvMIL label plan selected no training patients')
    return history

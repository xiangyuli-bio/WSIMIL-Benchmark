"""Private original AugDiff/LDM definitions and explicit feature-layout repair.

The released image-layout YAML cannot consume the released feature tensors.
``singleton_1d_reconstruction`` is an explicitly named reconstruction, NOT an
assertion about the authors' missing custom UNet or unpublished checkpoints.
All published UNet widths, levels, residual/attention blocks and the complete
1000-step epsilon objective are retained. No tiny denoiser is substituted.
This module alone is not a complete or admitted AugDiff training profile.
"""
import ast
import contextlib
import copy
import math
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache, partial
from inspect import isfunction
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .upstream import load_definitions, pinned_text, verified_source

COMMIT = 'abfdfe25c6373019d71bf3323cf59be799203563'
LDM_COMMIT = 'a506df5756472e2ebaf9078affdde2c4f1502cd4'
LAYOUT = 'singleton_1d_reconstruction'


def fixed():
    for key, expected in [('augdiff', COMMIT), ('augdiff_ldm', LDM_COMMIT)]:
        _, sha = verified_source(key)
        if sha != expected:
            raise ValueError('AugDiff source/dependency differs from its reviewed fixed commit')


def definitions(source, relative, names=None, namespace=None, replacements=()):
    root, sha = verified_source(source)
    if names is None:
        names = [node.name for node in ast.parse(pinned_text(root, sha, relative)).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    scope = {'torch': torch, 'th': torch, 'nn': nn, 'F': F, 'np': np,
             'math': math, 'partial': partial, 'rearrange': rearrange,
             'repeat': repeat, 'tqdm': tqdm, 'isfunction': isfunction,
             'einsum': torch.einsum, 'abstractmethod': abstractmethod,
             'Iterable': Iterable, 'contextmanager': contextlib.contextmanager}
    scope.update(namespace or {})
    return load_definitions(source, relative, names, scope, replacements)


class _TrainingHost(nn.Module):
    """Only framework logging/device hooks; every numerical method is source code."""
    current_epoch = 0
    global_step = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def log(self, *args, **kwargs):
        pass

    def log_dict(self, *args, **kwargs):
        pass


@lru_cache(maxsize=1)
def classes():
    fixed()
    # These are actual source classes, loaded privately to avoid importing
    # unused image codecs/Taming or replacing an installed ``ldm`` package.
    util = definitions('augdiff_ldm', 'ldm/modules/diffusionmodules/util.py', replacements=[
        ('ctx.input_params = list(args[length:])',
         ('ctx.input_params = list(args[length:])\n'
         '        ctx.autocast_device = ctx.input_tensors[0].device.type\n'
         '        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.autocast_device)\n'
         '        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.autocast_device)\n'
         '        ctx.autocast_cache = torch.is_autocast_cache_enabled()')),
        ('with torch.enable_grad():',
         ('with torch.enable_grad(), torch.autocast(ctx.autocast_device, enabled=ctx.autocast_enabled, '
         'dtype=ctx.autocast_dtype, cache_enabled=ctx.autocast_cache):'))])
    util_scope = {key: value for key, value in vars(util).items() if not key.startswith('__')}
    attention = definitions('augdiff_ldm', 'ldm/modules/attention.py', namespace=util_scope)
    # A feature is [D,1], as in the actual released sampler. Keeping length1
    # through upsampling is necessary for same-length skip concatenation.
    # This repairs the missing feature-layout implementation, not a disclosed
    # historical architecture. All channels/levels/blocks remain full sized.
    network = definitions('augdiff_ldm', 'ldm/modules/diffusionmodules/openaimodel.py',
        namespace={**util_scope, 'SpatialTransformer': attention.SpatialTransformer},
        replacements=[('x = F.interpolate(x, scale_factor=2, mode="nearest")',
            'x = x if self.dims == 1 and x.shape[-1] == 1 else F.interpolate(x, scale_factor=2, mode="nearest")')])
    identity = definitions('augdiff_ldm', 'ldm/models/autoencoder.py', ['IdentityFirstStage']).IdentityFirstStage

    def instantiate(config):
        # Only the author's recommended unconditional, identity-first-stage
        # route is supported. No fake VQ codec/conditioning model is inserted.
        if not isinstance(config, dict) or set(config) - {'target', 'params'}:
            raise ValueError('Invalid fixed AugDiff dependency configuration')
        target = config['target']
        if target == 'ldm.modules.diffusionmodules.openaimodel.UNetModel':
            return network.UNetModel(**config.get('params', {}))
        if target == 'ldm.models.autoencoder.IdentityFirstStage':
            return identity(**config.get('params', {}))
        raise ValueError('AugDiff feature reconstruction does not support a different codec/conditioner')

    utility = definitions('augdiff_ldm', 'ldm/util.py',
        ['exists', 'default', 'ismap', 'isimage', 'mean_flat', 'count_params'])
    distribution = definitions('augdiff_ldm', 'ldm/modules/distributions/distributions.py')
    ema = definitions('augdiff_ldm', 'ldm/modules/ema.py', ['LitEma'])
    sampler = sampler_class(repaired=True, utility=util)
    root, sha = verified_source('augdiff_ldm')
    source = pinned_text(root, sha, 'ldm/models/diffusion/ddpm.py')
    vq_check = 'isinstance(self.first_stage_model, VQModelInterface)'
    if source.count(vq_check) != 6:
        raise ValueError('Actual pinned LDM identity-codec branch assumptions changed')
    ddpm = definitions('augdiff_ldm', 'ldm/models/diffusion/ddpm.py',
        ['disabled_train', 'uniform_on_device', 'DDPM', 'LatentDiffusion', 'DiffusionWrapper'],
        namespace={**util_scope, **{k: v for k, v in vars(utility).items() if not k.startswith('__')},
            'pl': SimpleNamespace(LightningModule=_TrainingHost),
            'instantiate_from_config': instantiate, 'LitEma': ema.LitEma,
            'normal_kl': distribution.normal_kl,
            'DiagonalGaussianDistribution': distribution.DiagonalGaussianDistribution,
            'IdentityFirstStage': identity, 'DDIMSampler': sampler,
            'rank_zero_only': lambda function: function,
            '__conditioning_keys__': {'concat': 'c_concat', 'crossattn': 'c_crossattn', 'adm': 'y'}},
        replacements=[(vq_check, 'False'),
            ('.mean(dim=[1, 2, 3])', '.mean(dim=tuple(range(1, model_out.ndim)))'),
            ('.mean([1, 2, 3])', '.mean(tuple(range(1, model_output.ndim)))'),
            ('.mean(dim=(1, 2, 3))', '.mean(dim=tuple(range(1, model_output.ndim)))')])
    return SimpleNamespace(network=network, diffusion=ddpm, utility=util, ema=ema)


@lru_cache(maxsize=4, typed=True)
def sampler_class(repaired=True, utility=None):
    if type(repaired) is not bool:
        raise ValueError('AugDiff sampler repair must be an explicit boolean')
    util = utility or definitions('augdiff_ldm', 'ldm/modules/diffusionmodules/util.py')
    patches = [('attr = attr.to(torch.device("cuda"))', 'attr = attr.to(self.model.betas.device)')]
    if repaired:
        patches.append(('index = total_steps - i - 1',
                        'index = total_steps - i - 1\n            ts = torch.full((b,), step, device=device, dtype=torch.long)'))
    return definitions('augdiff', 'LatentDiffusion/ddim.py', ['DDIMSampler'],
        {'make_ddim_sampling_parameters': util.make_ddim_sampling_parameters,
         'make_ddim_timesteps': util.make_ddim_timesteps, 'noise_like': util.noise_like}, patches).DDIMSampler


def build_diffusion(dimension, layout=LAYOUT, use_ema=True):
    """Complete YAML-width DAE; unsupported literal image layout fails explicitly."""
    fixed()
    if type(dimension) is not int or dimension < 1 or layout != LAYOUT or type(use_ema) is not bool:
        raise ValueError('AugDiff requires an explicit singleton_1d_reconstruction and complete input dimension')
    # Read, validate and retain the published YAML, rather than allowing a
    # test-only smaller UNet to silently become an implementation profile.
    import yaml
    root, sha = verified_source('augdiff')
    cfg = yaml.safe_load(pinned_text(root, sha, 'LatentDiffusion/unitopatho_uncondition.yaml'))
    params = cfg['model']['params']
    unet = dict(params['unet_config']['params'])
    expected = {'image_size': 32, 'in_channels': 512, 'out_channels': 512,
        'model_channels': 256, 'attention_resolutions': [4, 2, 1],
        'num_res_blocks': 2, 'channel_mult': [1, 2, 4], 'num_head_channels': 32}
    if unet != expected or params['timesteps'] != 1000 or params['first_stage_config'] != {
            'target': 'ldm.models.autoencoder.IdentityFirstStage'} or params['cond_stage_config'] != '__is_unconditional__':
        raise ValueError('AugDiff published native architecture/configuration changed')
    unet.update(in_channels=dimension, out_channels=dimension, dims=1)
    args = dict(params)
    args['unet_config'] = {'target': params['unet_config']['target'], 'params': unet}
    args.update(use_ema=use_ema)
    model = classes().diffusion.LatentDiffusion(**args)
    # Fixed source logvar was a plain CPU tensor. Register the exact fixed
    # value so device movement and strict checkpoints include it.
    if not model.learn_logvar:
        value = model.logvar
        del model.logvar
        model.register_buffer('logvar', value)
    model.feature_dimension = dimension
    model.feature_layout = layout
    # Source main.py scales base_lr by batch * devices * accumulate_grad.
    # The actual trainer must record its explicit scale; this default is the
    # published batch1200 on one device, not the historical 3-GPU claim.
    model.learning_rate = float(cfg['model']['base_learning_rate']) * 1200
    return model


def epsilon_loss(model, features, timesteps=None, noise=None):
    if (features.ndim != 2 or not len(features) or features.shape[1] != model.feature_dimension
            or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError('AugDiff DAE requires complete finite patch feature rows [B,D]')
    x = features[..., None]
    if timesteps is None:
        timesteps = torch.randint(0, model.num_timesteps, (len(features),), device=x.device)
    if timesteps.shape != (len(features),) or timesteps.dtype != torch.int64 or not bool(
            ((timesteps >= 0) & (timesteps < model.num_timesteps)).all()):
        raise ValueError('AugDiff diffusion timesteps must be one native0..999 integer per patch')
    if noise is not None:
        if noise.shape != features.shape or not torch.isfinite(noise).all():
            raise ValueError('AugDiff epsilon noise must match the full [B,D] feature tensor')
        noise = noise[..., None]
    return model.p_losses(x, None, timesteps, noise=noise)


@torch.no_grad()
def augment(model, features, steps=20, nu=.4, repaired=True):
    if (features.ndim != 2 or not len(features) or features.shape[1] != model.feature_dimension
            or not features.is_floating_point() or not torch.isfinite(features).all()
            or type(steps) is not int or steps not in {20, 30}
            or isinstance(nu, bool) or nu not in {.2, .4}):
        raise ValueError('AugDiff native augmentation requires full [N,D], T20/30 and nu0.2/0.4')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze the complete trained DAE before MIL feature augmentation')
    sampler = sampler_class(repaired=repaired)(model)
    result, _ = sampler.sample(S=steps, batch_size=len(features),
        shape=[model.feature_dimension, 1], nu=nu, x_T=features,
        eta=0., verbose=False)
    result = model.decode_first_stage(result)
    if result.shape != (len(features), model.feature_dimension, 1) or not torch.isfinite(result).all():
        raise ValueError('AugDiff denoising failed to preserve all feature rows/dimensions')
    return result.squeeze(-1)


def build_classifier(backbone, dimension, classes_count):
    """Actual three published MIL heads, without unused coordinate columns."""
    fixed()
    if (backbone not in {'amil', 'lossattn', 'dsmil'} or type(dimension) is not int
            or dimension < 1 or type(classes_count) is not int or classes_count < 2):
        raise ValueError('AugDiff requires an actual AMIL/LossAttn/DSMIL classifier')
    patches = [("kwargs['data'].float()[..., 2:]", "kwargs['data'].float()")]
    if backbone in {'amil', 'lossattn'}:
        filename, name = ('AMIL.py', 'AMIL') if backbone == 'amil' else ('LossAttn.py', 'LossAttn')
        patches.append(('h = h.squeeze()', 'h = h.squeeze(0)'))
    else:
        filename, name = 'DSMIL.py', 'DSMIL'
        patches += [('x = x.squeeze()', 'x = x.squeeze(0)'), ('.cuda()', '')]
    module = definitions('augdiff', 'MIL/models/'+filename, replacements=patches)
    return getattr(module, name)(feats_size=dimension, n_classes=classes_count)


def classifier_result(model, features, target=None):
    if (features.ndim != 2 or not len(features) or not features.is_floating_point()
            or not torch.isfinite(features).all()):
        raise ValueError('AugDiff classifier requires all finite patch rows [N,D]')
    result = model(data=features[None])
    logits = result['logits']
    if target is not None:
        if target.shape != (1,) or target.dtype != torch.int64:
            raise ValueError('AugDiff classifier requires one int64 slide target')
        result['loss'] = F.cross_entropy(logits, target)
    return result


@lru_cache(maxsize=1)
def optimizer_class():
    fixed()
    source = definitions('augdiff', 'MIL/MyOptimizer/lookahead.py', ['Lookahead'],
        {'Optimizer': torch.optim.Optimizer, 'defaultdict': defaultdict})

    class PortableLookahead(source.Lookahead):
        """Original numerical update, modern hooks and index-based slow state.

        Original Lookahead stored Python object ids as slow-state keys, which
        do not remap to newly constructed parameters on resume. This fixes the
        serialization, retaining the actual late-created slow-buffer update.
        """
        def __init__(self, base):
            super().__init__(base, alpha=.5, k=6)
            torch.optim.Optimizer.__init__(self, base.param_groups, self.defaults)
            self.param_groups = base.param_groups

        def state_dict(self):
            packed = self.base_optimizer.state_dict()
            slow = {}
            for live_group, saved_group in zip(self.param_groups, packed['param_groups']):
                for parameter, index in zip(live_group['params'], saved_group['params']):
                    if parameter in self.state:
                        slow[index] = self.state[parameter]
            return {**packed, 'slow_state': slow,
                    'radam_buffer': copy.deepcopy(self.base_optimizer.buffer)}

        def load_state_dict(self, state):
            if set(state) != {'state', 'slow_state', 'param_groups', 'radam_buffer'}:
                raise ValueError('AugDiff requires complete fast/slow/RAdam optimizer state')
            self.base_optimizer.load_state_dict({k: state[k] for k in ['state', 'param_groups']})
            self.param_groups = self.base_optimizer.param_groups
            mapping = {index: parameter for group, saved in zip(self.param_groups, state['param_groups'])
                       for parameter, index in zip(group['params'], saved['params'])}
            if set(state['slow_state']) - set(mapping):
                raise ValueError('AugDiff slow optimizer state has unknown parameter indices')
            self.state = defaultdict(dict)
            for index, values in state['slow_state'].items():
                parameter = mapping[index]
                if set(values) != {'slow_buffer'} or values['slow_buffer'].shape != parameter.shape:
                    raise ValueError('AugDiff Lookahead slow-buffer shape/state changed')
                self.state[parameter] = {'slow_buffer': values['slow_buffer'].detach().clone().to(parameter)}
            self.base_optimizer.buffer = copy.deepcopy(state['radam_buffer'])

    return PortableLookahead


def classifier_optimizer(model, lr=2e-4, weight_decay=1e-5):
    fixed()
    module = definitions('augdiff', 'MIL/MyOptimizer/radam.py', ['RAdam'],
        {'Optimizer': torch.optim.Optimizer})
    group = definitions('augdiff', 'MIL/MyOptimizer/optim_factory.py', ['add_weight_decay']).add_weight_decay
    base = module.RAdam(group(model, weight_decay), lr=lr, weight_decay=0.)
    return optimizer_class()(base)

"""MamMIL's pinned Mamba2 topology scans, GENConv, source/repaired loops and full recipes."""
import importlib
import importlib.metadata
import math
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import DeepGCNLayer, GENConv

from .discrete_survival import DiscreteSurvival
from .io import digest_file, digest_object
from .mammil_graph import graph_asset, prepare_graphs
from .ssd_scan import gated_rms, split_conv_scan
from .upstream import load_definitions, load_single

PROFILES = {'mammil', 'mammil_source', 'mammil_patient_union'}


class NoHubMixin:
    """Only numerical author definitions are loaded; no HuggingFace Hub serialization API."""


def backend_namespace(backend='torch', checkpoint_chunk=64):
    if backend == 'torch':
        def kernel(*args, **kwargs):
            return split_conv_scan(*args, **kwargs, checkpoint_chunk=checkpoint_chunk)
        norm = gated_rms
        identity = {'name': 'torch', 'torch': torch.__version__, 'checkpoint_chunk': checkpoint_chunk,
                    'scan': 'exact head-shared SSD recurrence with differentiable cross-chunk state'}
    elif backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('MamMIL CUDA backend requires a GPU; no automatic fallback')
        try:
            versions = {p: importlib.metadata.version(p) for p in ('mamba-ssm', 'causal-conv1d', 'triton')}
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Missing MamMIL fused dependencies: build the pinned mamba2.2.2 source and causal-conv1d1.4.0') from error
        if versions['mamba-ssm'] != '2.2.2' or versions['causal-conv1d'] != '1.4.0':
            raise ValueError('MamMIL fused backend requires source-compatible mamba-ssm2.2.2/causal-conv1d1.4.0')
        kernel = load_single('mammil', 'mamba/mamba_ssm/ops/triton/ssd_combined.py').mamba_split_conv1d_scan_combined
        norm = load_single('mammil', 'mamba/mamba_ssm/ops/triton/layernorm_gated.py').rmsnorm_fn
        package = Path(importlib.import_module('mamba_ssm').__file__).parent
        identity = {'name': 'cuda', 'versions': versions, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'package_python_sha256': digest_object({str(p.relative_to(package)): digest_file(p) for p in sorted(package.rglob('*.py'))}),
                    'causal_binary_sha256': digest_file(importlib.import_module('causal_conv1d_cuda').__file__)}
    else:
        raise ValueError('MamMIL backend must be explicitly torch or cuda')
    norm_cls = load_definitions('mammil', 'mamba/mamba_ssm/ops/triton/layernorm_gated.py',
        ['RMSNorm'], {'torch': torch, 'rmsnorm_fn': norm}).RMSNorm
    ns = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'rearrange': rearrange, 'repeat': repeat,
          'RMSNormGated': norm_cls, 'PyTorchModelHubMixin': NoHubMixin,
          'mamba_split_conv1d_scan_combined': kernel}
    direct = load_definitions('mammil', 'mamba/mamba_ssm/modules/direct_mamba.py', ['DirectMamba'], ns).DirectMamba
    return direct, identity


def author_network(profile, backend='torch', checkpoint_chunk=64):
    direct, identity = backend_namespace(backend, checkpoint_chunk)
    init = load_definitions('timm_mfemil', 'timm/layers/weight_init.py',
        ['_trunc_normal_', 'trunc_normal_', 'trunc_normal_tf_', 'variance_scaling_', 'lecun_normal_'],
        {'torch': torch, 'math': math, 'warnings': warnings,
         '_calculate_fan_in_and_fan_out': nn.init._calculate_fan_in_and_fan_out})
    loss = load_definitions('mammil', 'utils/survival_utils.py', ['nll_loss', 'NLLSurvLoss'],
                            {'torch': torch, 'nn': nn}).NLLSurvLoss
    patches = [('from utils.survival_utils import NLLSurvLoss', '')]
    if profile != 'mammil_source':
        patches.append(('for idx, (layer, ppeg) in enumerate(zip(self.layers, self.ppeg)):',
                        'for idx, layer in enumerate(self.layers):\n            ppeg = self.ppeg[idx] if idx < len(self.ppeg) else None'))
    cls = load_definitions('mammil', 'models/MamMIL2.py', ['_init_weights', 'segm_init_weights', 'MamMIL2'],
        {'torch': torch, 'nn': nn, 'np': np, 'F': F, 'math': math, 'partial': partial,
         'GENConv': GENConv, 'DeepGCNLayer': DeepGCNLayer, 'LayerNorm': nn.LayerNorm, 'ReLU': nn.ReLU,
         'DirectMamba': direct, 'trunc_normal_': init.trunc_normal_, 'lecun_normal_': init.lecun_normal_,
         'NLLSurvLoss': loss}, patches).MamMIL2
    return cls, identity


class MamMIL(DiscreteSurvival):
    def __init__(self, bundle, profile='mammil', hidden=512, layers=2, state_size=128,
                 dropout=0., backend='torch', scan_chunk=64, neighbors=8, graph_seed=2024,
                 graph_cache=None, bins=4, alpha=0.):
        super().__init__(bins, alpha, accumulation_steps=1)
        if profile not in PROFILES or bundle.task_type not in {'classification', 'survival'}:
            raise ValueError('MamMIL needs a registered classification/survival profile')
        if any(type(v) is not int or v < 1 for v in (hidden, layers, state_size, scan_chunk, neighbors)):
            raise ValueError('MamMIL widths, layers, state and chunk/neighbors must be positive integers')
        if hidden % 32 or layers < 2 or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('MamMIL requires hidden multiple of32, layers>=2 and finite dropout in[0,1)')
        if type(graph_seed) is not int or graph_seed < 0:
            raise ValueError('MamMIL graph_seed must be a nonnegative integer')
        self.profile, self.survival, self.dim = profile, bundle.task_type == 'survival', bundle.dim
        if profile == 'mammil_patient_union' and not self.survival:
            raise ValueError('MamMIL patient_union is an explicitly registered survival extension')
        if not self.survival and bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('MamMIL classification requires individual slide bags')
        if profile != 'mammil_patient_union' and self.survival and (bundle.rows.groupby('global_patient_id').size() > 1).any():
            raise ValueError('Source MamMIL survival requires one WSI per patient; choose explicit mammil_patient_union for disjoint patient graphs')
        self.hidden, self.layers, self.state_size = hidden, layers, state_size
        self.neighbors, self.graph_seed = neighbors, graph_seed
        self.graph_cache = Path(graph_cache or bundle.root/'.milplus-cache/mammil').resolve()
        self.graph_manifest, self.optimizer_steps = {}, 0
        cls, self.backend_identity = author_network(profile, backend, scan_chunk)
        self.network = cls(input_dim=bundle.dim, embed_dim=hidden, layer_n=layers, d_state=state_size,
                           dropout=dropout, num_classes=bins if self.survival else len(bundle.classes), survival=self.survival)
        if self.survival:
            self.network.loss_fn.alpha = alpha

    def configure_training(self, config):
        if config['max_instances'] is not None:
            raise ValueError('MamMIL rejects max_instances truncation; it would invalidate topology and traversals')

    def prepare(self, bundle, seed, device):
        if self.survival:
            super().prepare(bundle, seed, device)
        _, self.graph_manifest = prepare_graphs(bundle, self.graph_cache, self.neighbors, self.graph_seed)

    def make_optimizer(self, lr, weight_decay):
        namespace = {'torch': torch, 'math': math, 'Optimizer': torch.optim.Optimizer}
        optimizer = load_definitions('mammil', 'MyOptimizer/radam.py', ['RAdam'], namespace).RAdam
        groups = load_definitions('mammil', 'MyOptimizer/optim_factory.py', ['add_weight_decay'], {})
        return optimizer(groups.add_weight_decay(self.network, weight_decay), lr=lr, weight_decay=0.)

    def get_extra_state(self):
        return {'backend': self.backend_identity, 'graphs': self.graph_manifest, 'optimizer_steps': self.optimizer_steps}

    def set_extra_state(self, state):
        if state['backend'] != self.backend_identity:
            raise ValueError('MamMIL numerical backend differs from its checkpoint')
        self.graph_manifest, self.optimizer_steps = state['graphs'], state['optimizer_steps']

    def graph(self, bundle, rows, device):
        if len(rows) > 1 and self.profile != 'mammil_patient_union':
            raise ValueError('Only explicit MamMIL patient_union can merge multiple slide graphs')
        features, edges, orders = [], [], {name: [] for name in ('dfs_preorder_index', 'dfs_postorder_index', 'bfs_levelorder_index')}
        offset = 0
        for row in sorted(rows, key=lambda r: r['slide_id']):
            x, coords = bundle.load(row, require_coords=True)
            expected = self.graph_manifest.get(row['slide_id'])
            if expected is None:
                raise RuntimeError('MamMIL topology assets were not prepared')
            arrays, _ = graph_asset(x.numpy(), coords.numpy(), self.graph_cache,
                self.neighbors, self.graph_seed, expected=expected)
            features.append(x); edges.append(arrays['edge_index']+offset)
            for name, parts in orders.items():
                parts.append(arrays[name]+offset)
            offset += len(x)
        graph = Data(x=torch.cat(features), edge_index=torch.cat(edges, 1),
                     **{name: torch.cat(parts) for name, parts in orders.items()})
        return graph.to(device)

    def forward(self, graph, label=None, censor=None):
        if graph.x.ndim != 2 or not len(graph.x) or graph.x.shape[1] != self.dim or not torch.isfinite(graph.x).all():
            raise ValueError('MamMIL expects finite nonempty graph features with the registered FM width')
        label = torch.zeros(1, dtype=torch.long, device=graph.x.device) if label is None else label
        kwargs = {'data': graph, 'label': label}
        if self.survival:
            kwargs['c'] = graph.x.new_zeros(1) if censor is None else censor
        result = self.network(**kwargs)
        if not torch.isfinite(result['logits']).all() or not torch.isfinite(result['loss']):
            raise ValueError('Nonfinite MamMIL output/objective')
        if self.survival:
            return {'logits': -result['S'].sum(1, keepdim=True), 'hazard_logits': result['logits'],
                    'hazards': result['hazards'], 'survival': result['S'], 'task_loss': result['loss']}
        return {**result, 'task_loss': result['loss']}

    def forward_unit(self, bundle, rows, device, config, train=False):
        graph = self.graph(bundle, rows, device)
        label = self.time_label([rows[0]['time']]) if train and self.survival else torch.tensor([rows[0]['target']], device=device) if train else None
        censor = graph.x.new_tensor([1-rows[0]['event']]) if train and self.survival else None
        return self(graph, label, censor)

    def protocol(self):
        return {**(super().protocol() if self.survival else {}), 'profile': self.profile,
                'hidden': self.hidden, 'layers': self.layers, 'state_size': self.state_size,
                'active_scans': self.layers-1 if self.profile == 'mammil_source' else self.layers,
                'backend': self.backend_identity, 'graph_manifest': self.graph_manifest,
                'graph_cache': str(self.graph_cache), 'neighbors': self.neighbors, 'graph_seed': self.graph_seed,
                'graph_recipe': 'author HNSW spatial/latent edges, cosine-weighted MST forest and DFS/BFS traversals; explicit serial index build and frozen checked assets',
                'repairs': 'tiny-bag k capped at N; train-only survival bins; named loop correction unless mammil_source',
                'patient_extension': self.profile == 'mammil_patient_union', 'cross_slide_edges': False,
                'objective': 'author censored NLL' if self.survival else 'author CE',
                'optimizer': 'author RAdam; bias/1D no decay; native lr1e-4 wd.05',
                'optimizer_steps': self.optimizer_steps, 'hub_serialization': 'not included',
                'gpu_validation': 'not_run', 'paper_reproduction': 'not_run'}


def mammil_epoch(model, bundle, groups, optimizer, device, config):
    model.train(); losses = []
    for rows in groups:
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_unit(bundle, rows, device, config, train=True)
        loss = result['task_loss']; loss.backward()
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite MamMIL gradients')
        optimizer.step(); model.optimizer_steps += 1
        losses.append({'loss': float(loss.detach())})
    return losses

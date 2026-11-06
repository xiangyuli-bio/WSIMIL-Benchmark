"""Full native TTIS heads, optimizer recipe and TEST-only multiview selection."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import digest_file, digest_object
from .ttis_source import HEADS, load_ttis, load_ttis_optimizer
from .upstream import pinned_text, verified_source

PROFILES = {f'ttis_{head}_source': head for head in HEADS}
PROFILES['ttis_clam_full_loss'] = 'clam'


class FrozenSemantic:
    def __init__(self, scores):
        self.scores = scores
    def zeroshot(self, h):
        return self.scores, self.scores.softmax(-1)


class TTISMIL(nn.Module):
    owns_sampling = True
    supports_instance_outputs = True
    instance_score_semantics = 'original TTIS selected row indices per view, not attention heatmaps or causal explanations'

    def __init__(self, bundle, profile='ttis_abmil_source', hidden=512, attention_dim=384,
                 sampling_ratio=.3, scheduler_epochs=50, eta_min=1e-7):
        super().__init__()
        if profile not in PROFILES or bundle.task_type != 'classification' or bundle.ttis is None:
            raise ValueError('TTIS requires a registered classifier and prepared spatial/semantic assets')
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('TTIS uses individual slide bags for training')
        if any(type(v) is not int or v < 1 for v in [hidden, attention_dim, scheduler_epochs]) or hidden % 8:
            raise ValueError('TTIS hidden width must be a positive multiple of eight; budgets positive')
        if isinstance(sampling_ratio, bool) or not isinstance(sampling_ratio, (int, float)) or not math.isfinite(sampling_ratio) or not 0 < sampling_ratio < 1:
            raise ValueError('TTIS source sampling ratio must lie strictly between zero and one')
        if isinstance(eta_min, bool) or not isinstance(eta_min, (int, float)) or not math.isfinite(eta_min) or eta_min < 0:
            raise ValueError('TTIS cosine minimum must be finite nonnegative')
        self.bundle, self.assets, self.profile = bundle, bundle.ttis, profile
        self.kind, self.dim, self.hidden = PROFILES[profile], bundle.dim, hidden
        self.ratio, self.scheduler_epochs, self.eta_min = sampling_ratio, scheduler_epochs, eta_min
        root, sha = verified_source('ttis')
        import yaml
        folder = {'clam': 'clamsb', 'rrt': 'rrtmil', 's4': 's4mil'}.get(self.kind, self.kind)
        cfg = yaml.safe_load(pinned_text(root, sha, f'configs/models/{folder}/uni_v1/brca.yaml'))['Model']
        import inspect
        cls = load_ttis(self.kind)
        cfg.update(in_dim=self.dim, num_classes=len(bundle.classes), embed_dim=hidden)
        if self.kind == 'rrt':
            cfg['mlp_dim'] = hidden
            if hidden != 512:
                # The source RRT projection is explicitly 512-wide; no silent incoherent width override.
                raise ValueError('Native TTIS RRT uses 512 hidden; reduced RRT requires separate source verification')
        if self.kind == 'abmil':cfg['attn_dim'] = attention_dim
        if self.kind == 'clam':cfg['attention_dim'] = attention_dim
        cfg = {k: v for k, v in cfg.items() if k in inspect.signature(cls).parameters}
        self.network = cls(**cfg)
        self.network.sampling.sampling_ratio = sampling_ratio
        self.register_buffer('semantic_text', self.assets.text.clone())
        self.register_buffer('semantic_scale', torch.tensor(float(self.assets.semantic['logit_scale'])))
        self.optimizer_steps, self.completed_epochs = 0, 0
        self.scheduler, self.saved_scheduler = None, None
        self.trace = None
        original = self.network.sampling.run
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            self.trace = {k: v.detach().cpu().clone() for k, v in result.items()}
            return result
        self.network.sampling.run = capture
        self.contract = digest_object({'profile': profile, 'assets': digest_file(self.assets.path),
            'dimension': self.dim, 'hidden': hidden, 'attention_dim': attention_dim, 'ratio': sampling_ratio,
            'scheduler_epochs': scheduler_epochs, 'eta_min': eta_min})

    def configure_training(self, cfg):
        if cfg['grad_clip'] is not None or cfg['max_instances'] is not None:
            raise ValueError('TTIS source train/VAL use complete unclipped bags')
        if cfg['selection_metric'] != 'ttis_val_loss' or cfg['lr'] <= self.eta_min:
            raise ValueError('TTIS selects validation bag CE with initial lr > cosine minimum')
        if self.kind == 'clam' and cfg.get('bag_weight', .7) != .7:
            raise ValueError('TTIS native CLAM fixes its combined bag weight at .7')

    def make_optimizer(self, lr, weight_decay):
        self.optimizer = load_ttis_optimizer()(SimpleNamespace(opt='lookahead_adamw', lr=lr,
            weight_decay=weight_decay, opt_eps=1e-10, opt_betas=None), self.network)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_epochs, eta_min=self.eta_min)
        if self.saved_scheduler is not None:self.scheduler.load_state_dict(self.saved_scheduler)
        return self.optimizer

    def finish_epoch(self, bundle, device, cfg):
        self.scheduler.step(); self.completed_epochs += 1

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('TTIS requires finite nonempty registered patch features')
        if self.kind in {'clam', 'wikg'} and len(x) < (8 if self.kind == 'clam' else 6):
            raise ValueError('Native TTIS head needs enough real instances; no synthetic padding')
        target = label
        if self.kind == 'clam' and target is None:
            target = torch.zeros(1, dtype=torch.long, device=x.device)  # Only unused source auxiliary loss; never actual TEST labels.
        loss_fn = nn.CrossEntropyLoss() if target is not None else None
        result, _ = self.network(x[None], label=target, loss_fn=loss_fn, use_ttis=False)
        out = {'logits': result['logits']}
        if label is not None:
            out['total_loss'] = result['loss'] if self.profile == 'ttis_clam_full_loss' else F.cross_entropy(result['logits'], label)
        if not torch.isfinite(out['logits']).all():raise ValueError('TTIS head produced nonfinite logits')
        return out

    def forward_unit(self, bundle, rows, device, config, train=False):
        if len(rows) != 1:raise ValueError('TTIS preserves individual slide identities')
        row = rows[0];x, _ = bundle.load(row);x = x.to(device)
        self.trace = None
        if train:return self(x, label=torch.tensor([row['target']], device=device))
        if row['subset'] != 'test':return self(x)
        clusters, guidance = self.assets.load(row)
        selected = sum(max(1, int(len(v)*self.ratio)) for v in clusters.values())
        if self.kind in {'clam', 'wikg'} and selected < (8 if self.kind == 'clam' else 6):
            raise ValueError('TTIS retained subsets are too small for the native head')
        # Same literal MUSK score operation. The declared asset policy controls which visual stream is scored.
        scores = self.semantic_scale * F.normalize(guidance.to(device), dim=-1) @ self.semantic_text.T
        options = {'label': torch.zeros(1, dtype=torch.long, device=device), 'loss_fn': nn.CrossEntropyLoss()} if self.kind == 'clam' else {}
        result, _ = self.network(x[None], use_ttis=True, clusters=clusters,
                                 zeroshot_model=FrozenSemantic(scores[None]), **options)
        if not torch.isfinite(result['logits']).all():raise ValueError('Nonfinite TTIS multiview logits')
        return {'logits': result['logits'], 'selected_indices': self.trace}

    def prediction_fields(self, result, row):
        target = torch.tensor([row['target']], device=result['logits'].device)
        return {'ttis_bag_ce': float(F.cross_entropy(result['logits'], target)),
                'ttis_applied': row['subset'] == 'test'}

    def validation_metrics(self, frame):
        return {'ttis_val_loss': float(frame.ttis_bag_ce.mean())}

    def export_instances(self, result, row, directory):
        directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
        path = directory/(digest_object(row['slide_id'])[:24]+'.h5')
        from .ttis_data import spatial_inputs
        x, coords = spatial_inputs(self.bundle, row)
        views = result.get('selected_indices') or {'full': torch.arange(len(x))}
        with h5py.File(path, 'w') as h:
            h['coords'] = coords
            for name, indices in views.items():h['indices/'+name] = indices.numpy()
            h.attrs['method'] = self.profile;h.attrs['score_semantics'] = self.instance_score_semantics
        return {'slide_id': row['slide_id'], 'path': str(path), 'sha256': digest_file(path),
                'instances': len(x), 'selected_per_view': {k: len(v) for k, v in views.items()}}

    def get_extra_state(self):
        return {'contract': self.contract, 'optimizer_steps': self.optimizer_steps, 'completed_epochs': self.completed_epochs,
                'scheduler': copy.deepcopy(self.scheduler.state_dict() if self.scheduler is not None else self.saved_scheduler)}

    def set_extra_state(self, state):
        if set(state) != {'contract', 'optimizer_steps', 'completed_epochs', 'scheduler'} or state['contract'] != self.contract or any(type(state[k]) is not int or state[k] < 0 for k in ['optimizer_steps', 'completed_epochs']):
            raise ValueError('TTIS checkpoint contract or counters differ')
        self.optimizer_steps, self.completed_epochs = state['optimizer_steps'], state['completed_epochs']
        self.saved_scheduler = copy.deepcopy(state['scheduler'])
        if self.scheduler is not None:self.scheduler.load_state_dict(self.saved_scheduler)

    def protocol(self):
        return {'profile': self.profile, 'head': self.kind, 'hidden': self.hidden,
            'recipe': 'whole-bag source LookaheadAdamW, alpha.5/k6/slow initialization at first sync, eps1e-10, native no-decay groups, epoch cosine',
            'loss': 'native CLAM bag/instance combined objective, explicit repaired training route' if self.profile == 'ttis_clam_full_loss' else 'original TTIS ModelInterface recomputed bag CE; CLAM returned instance loss is not optimized',
            'selection': 'strict improvement of validation slide-bag CE; TRAIN and VAL never use TTIS',
            'sampling': 'native per-cluster centroid/attention/semantic FPS; original duplicate-index behavior retained',
            'fusion': 'softmax L2-norm feature weights then original classifier, no probability-ensemble substitution',
            'semantic_policy': self.assets.policy, 'semantic_model': self.assets.semantic['model'],
            'semantic_embedding_space': self.assets.semantic['embedding_space'],
            'clustering': self.assets.meta['clustering'], 'optimizer_steps': self.optimizer_steps,
            'completed_epochs': self.completed_epochs, 'precision': 'explicit FP32 CPU/GPU path; source paper BF16 hardware setting not claimed',
            'source_repairs': ['private namespace without unrelated Lightning imports', 'unused opt_einsum import omitted; equivalent DropPath import',
                               'fixed dummy CLAM evaluation label only for unused auxiliary loss; predictions never consume target labels',
                               'complete optimizer/scheduler/RNG checkpoint and input-bound asset hashes'],
            'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}


def ttis_epoch(model, bundle, groups, optimizer, device, cfg):
    losses = []
    for rows in DataLoader(groups, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda batch: batch[0]):
        optimizer.zero_grad()
        result = model.forward_unit(bundle, rows, device, cfg, train=True)
        loss = result['total_loss']
        if not torch.isfinite(loss):raise ValueError('TTIS objective is nonfinite')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite TTIS gradients')
        optimizer.step();model.optimizer_steps += 1
        losses.append({'loss': float(loss.detach())})
    return losses

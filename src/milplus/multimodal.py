"""Author SurvPath, MCAT and PORPOISE with patient-level omics and discrete survival."""
import torch

from .discrete_survival import DiscreteSurvival
from .upstream import load_survival

MULTIMODAL_METHODS = {'survpath', 'mcat', 'porpoise'}


class MolecularSurvival(DiscreteSurvival):
    def __init__(self, method, dim, omics, bins=4, alpha=0., accumulation_steps=1,
                 l1_weight=0.):
        super().__init__(bins, alpha, accumulation_steps, l1_weight)
        self.method = method
        self.group_indices = omics.group_indices
        self.columns = omics.columns
        self.group_names = [g['name'] for g in omics.groups]
        self.normalization = omics.normalization
        if method in {'survpath', 'mcat', 'cmta', 'motcat', 'motcat_micro'} and not self.group_indices:
            raise ValueError(f'{method} needs explicit molecular pathway/signature definitions')
        if method in {'mcat', 'cmta', 'motcat', 'motcat_micro'} and len(self.group_indices) != 6:
            raise ValueError(f'{method} requires exactly six molecular signatures')
        self.register_buffer('omics_mean', torch.zeros(len(self.columns)))
        self.register_buffer('omics_scale', torch.ones(len(self.columns)))

    @torch.no_grad()
    def prepare(self, bundle, seed, device):
        super().prepare(bundle, seed, device)
        mean, scale = bundle.omics.training_statistics(bundle)
        self.omics_mean.copy_(mean)
        self.omics_scale.copy_(scale)

    def normalize_omics(self, omics):
        if not self.prepared:
            raise RuntimeError('Fit training survival intervals and molecular normalization first')
        if omics.shape != self.omics_mean.shape or not torch.isfinite(omics).all():
            raise ValueError('Molecular vector dimension or finiteness mismatch')
        return (omics-self.omics_mean)/self.omics_scale

    def forward_unit(self, bundle, rows, device, config, train=False):
        from .engine import load_unit
        x, _ = load_unit(bundle, rows, device, config['max_instances'], train)
        molecular = bundle.omics.tensor(rows[0]['global_patient_id']).to(device)
        return self(x, molecular)

    def protocol(self):
        return {**super().protocol(), 'omics_columns': self.columns,
                'pathway_names': self.group_names, 'pathway_indices': self.group_indices,
                'normalization': self.normalization, 'normalization_fit': 'unique training patients',
                'omics_mean': self.omics_mean.tolist(), 'omics_scale': self.omics_scale.tolist(),
                'recipe_scope': 'author architecture; PULS patient splits, optimizer and training budget'}


class MultimodalSurvival(MolecularSurvival):
    def __init__(self, method, dim, omics, bins=4, alpha=0., accumulation_steps=1,
                 l1_weight=0., dropout=None, fusion=None):
        super().__init__(method, dim, omics, bins, alpha, accumulation_steps, l1_weight)
        author = load_survival(method)
        sizes = [len(indices) for indices in self.group_indices]
        if method == 'survpath':
            if fusion is not None:
                raise ValueError('SurvPath uses pathway/histology token attention; fusion option is not applicable')
            self.network = author.SurvPath(omic_sizes=sizes, wsi_embedding_dim=dim, num_classes=bins,
                                           dropout=.1 if dropout is None else dropout)
        elif method == 'mcat':
            if fusion not in {None, 'concat', 'bilinear'}:
                raise ValueError('MCAT fusion must be concat or bilinear')
            self.network = author.MCAT_Surv(path_input_dim=dim, omic_sizes=sizes, n_classes=bins,
                                            fusion=fusion or 'concat', dropout=.25 if dropout is None else dropout)
        elif method == 'porpoise':
            if fusion not in {None, 'concat', 'bilinear'}:
                raise ValueError('PORPOISE fusion must be concat or bilinear')
            self.network = author.PorpoiseMMF(omic_input_dim=len(self.columns), path_input_dim=dim,
                                             n_classes=bins, fusion=fusion or 'bilinear',
                                             dropout=.25 if dropout is None else dropout)
        else:
            raise ValueError(f'Unknown multimodal method {method}')

    def forward(self, x, omics, coords=None):
        molecular = self.normalize_omics(omics)
        if self.method == 'porpoise':
            logits = self.network(x_path=x, x_omic=molecular.unsqueeze(0))
        else:
            arguments = {f'x_omic{i+1}': molecular[indices] for i, indices in enumerate(self.group_indices)}
            arguments['x_path'] = x.unsqueeze(0) if self.method == 'survpath' else x
            if self.method == 'survpath':
                logits = self.network(**arguments, return_attn=False)
            else:
                logits = self.network(**arguments)[0]
        return self.result(logits)

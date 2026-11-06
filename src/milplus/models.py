"""Runnable architectures. Recipes and provenance are separate registry fields."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, hidden=128, heads=1, gated=False, dropout=0.):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden) if gated else None
        self.w = nn.Linear(hidden, heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.v(x).tanh())
        if self.u is not None:
            h = h * self.dropout(self.u(x).sigmoid())
        return self.w(h).T.softmax(-1)


class PoolMIL(nn.Module):
    def __init__(self, dim, classes, mode, hidden=512, dropout=0.0):
        super().__init__()
        self.mode = mode
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, classes))

    def forward(self, x, coords=None, label=None):
        scores = self.head(x)
        logits = scores.mean(0) if self.mode == "mean" else scores.max(0).values
        return {"logits": logits.unsqueeze(0)}


class ABMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, attention_dim=128, gated=False, dropout=0.0):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention = Attention(hidden, attention_dim, gated=gated)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x, coords=None, label=None):
        h = self.project(x)
        a = self.attention(h)
        return {"logits": self.classifier(a @ h), "attention": a, "embedding": a @ h}


class CLAM(nn.Module):
    def __init__(self, dim, classes, multibranch=False, k_sample=8, dropout=0.0,
                 subtyping=False):
        super().__init__()
        from .vendor.clam.model_clam import CLAM_MB, CLAM_SB
        self.model = (CLAM_MB if multibranch else CLAM_SB)(embed_dim=dim, n_classes=classes,
                         k_sample=k_sample, dropout=dropout, subtyping=subtyping)
        self.k_sample = k_sample

    def forward(self, x, coords=None, label=None):
        if self.training and len(x) < 2 * self.k_sample:
            raise ValueError(f"CLAM requires at least {2 * self.k_sample} instances for disjoint top/bottom selection")
        logits, _prob, _, attn, extra = self.model(x, label=label,
                                       instance_eval=self.training and label is not None, return_features=True)
        return {"logits": logits, "attention": attn.softmax(-1), "embedding": extra["features"],
                **({"instance_loss": extra["instance_loss"]} if "instance_loss" in extra else {})}


class DSMIL(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.instance_head = nn.Linear(dim, classes)
        self.query = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        self.bag_head = nn.Conv1d(classes, classes, kernel_size=dim)

    def forward(self, x, coords=None, label=None):
        ins = self.instance_head(x)
        q = self.query(x)
        critical = x[ins.argmax(0)]
        a = (q @ self.query(critical).T / math.sqrt(q.shape[-1])).softmax(0)
        bag = a.T @ x
        return {"logits": self.bag_head(bag.unsqueeze(0)).squeeze(-1),
                "max_logits": ins.max(0).values.unsqueeze(0), "attention": a.T}


class TransMIL(nn.Module):
    def __init__(self, dim, classes, hidden=512, dropout=0.1):
        super().__init__()
        from .vendor.transmil.trans_mil import TRANS_MIL
        if hidden % 8:
            raise ValueError("TransMIL hidden dimension must be divisible by 8")
        self.model = TRANS_MIL(in_dim=dim, num_classes=classes, hidden_dim=hidden, dropout=dropout)

    def forward(self, x, coords=None, label=None):
        # Original TransMIL's sequence-to-square PPEG padding, not physical coordinates.
        return self.model(x.unsqueeze(0))


class DTFD(nn.Module):
    def __init__(self, dim, classes, hidden=512, groups=4, distill="AFS",
                 instances_per_group=1, dropout=0.0):
        super().__init__()
        if distill not in {"AFS", "MaxS", "MaxMinS"}:
            raise ValueError("DTFD distill must be AFS, MaxS or MaxMinS")
        if distill != "AFS" and classes != 2:
            raise ValueError("MaxS/MaxMinS use the original binary positive-class ordering; use AFS for multiclass")
        self.reduce = nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.ReLU())
        self.attention = Attention(hidden, gated=True)
        self.tier1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, classes))
        self.tier2_attention = Attention(hidden, gated=True)
        self.tier2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, classes))
        self.groups, self.distill, self.instances_per_group = groups, distill, instances_per_group

    def first_tier(self, x):
        if len(x) < self.groups * (2 if self.distill == "MaxMinS" else 1) * self.instances_per_group:
            raise ValueError("DTFD bag too small for group/distillation configuration")
        indices = torch.randperm(len(x), device=x.device) if self.training else torch.arange(len(x), device=x.device)
        pseudo, logits = [], []
        for idx in indices.tensor_split(self.groups):
            h = self.reduce(x[idx])
            weighted = self.attention(h).T * h
            pooled = weighted.sum(0, keepdim=True)
            logits.append(self.tier1(pooled))
            if self.distill == "AFS":
                pseudo.append(pooled)
            else:
                # CAM excludes classifier bias, matching original get_cam_1d.
                cam = F.linear(weighted, self.tier1[-1].weight).softmax(-1)[:, -1]
                order = cam.argsort(descending=True)
                selected = order[:self.instances_per_group]
                if self.distill == "MaxMinS":
                    selected = torch.cat([selected, order[-self.instances_per_group:]])
                pseudo.append(h[selected])
        return torch.cat(logits), torch.cat(pseudo)

    def second_tier(self, pseudo):
        a = self.tier2_attention(pseudo)
        return self.tier2(a @ pseudo)

    def forward(self, x, coords=None, label=None):
        group_logits, pseudo = self.first_tier(x)
        return {"logits": self.second_tier(pseudo), "group_logits": group_logits}


def build_bundle_model(method, bundle, config=None):
    if method.startswith("mexd_"):
        from .mexd import MExD
        return MExD(bundle,profile=method,**(config or {}))
    if method.startswith("lpgmil_"):
        from .lpgmil import LPGMIL
        return LPGMIL(bundle,profile=method,**(config or {}))
    if method.startswith('promptmil_'):
        from .promptmil import PromptMIL
        return PromptMIL(bundle,profile=method,**(config or {}))
    if method.startswith('iibmil_'):
        from .iibmil import IIBMIL
        return IIBMIL(bundle, profile=method, **(config or {}))
    if method.startswith('longmil_'):
        from .longmil import LongMIL
        return LongMIL(bundle, profile=method, **(config or {}))
    if method.startswith('mggmmil_'):
        from .mggmmil import MGGMMIL
        return MGGMMIL(bundle, profile=method, **(config or {}))
    if method.startswith('nciemil_'):
        from .nciemil import NcIEMIL
        return NcIEMIL(bundle,profile=method,**(config or {}))
    if method.startswith('taxomil_'):
        from .taxomil import TaxoMIL
        return TaxoMIL(bundle, profile=method, **(config or {}))
    if method.startswith('gmmamba_'):
        from .gmmamba import GMMambaMIL
        return GMMambaMIL(bundle, profile=method, **(config or {}))
    if method.startswith('muse_'):
        from .muse import MUSEMIL
        return MUSEMIL(bundle, profile=method, **(config or {}))
    if method.startswith('libramil_'):
        from .libramil import LibraMIL
        return LibraMIL(bundle, profile=method, **(config or {}))
    if method.startswith('m4_'):
        from .m4 import M4MIL
        return M4MIL(bundle, profile=method, **(config or {}))
    if method.startswith('convl_'):
        from .convl import ConVLMIL
        return ConVLMIL(bundle, profile=method, **(config or {}))
    if method.startswith('center_embeddings_'):
        from .center_embeddings import CenterEmbeddingMIL
        return CenterEmbeddingMIL(bundle, profile=method, **(config or {}))
    if method.startswith('miladapter_'):
        from .miladapter import MILAdapter
        return MILAdapter(bundle, profile=method, **(config or {}))
    if method.startswith('conmil_'):
        from .conmil import ConMIL
        return ConMIL(bundle, profile=method, **(config or {}))
    if method.startswith('sparseconvmil_raw_'):
        from .sparseconvmil import SparseConvMIL
        return SparseConvMIL(bundle, profile=method, **(config or {}))
    if method.startswith('c2c_cluster_'):
        from .c2c import C2C
        return C2C(bundle, profile=method, **(config or {}))
    if method.startswith('sammil_'):
        from .sammil import SAMMIL
        return SAMMIL(bundle, profile=method, **(config or {}))
    if method.startswith('lrmil_'):
        from .lrmil import LRMIL
        return LRMIL(bundle, profile=method, **(config or {}))
    if method.startswith('kmgmamba_'):
        from .kmgmamba import KMGMamba
        return KMGMamba(bundle, profile=method, **(config or {}))
    if method.startswith('ahdmil_'):
        from .ahdmil import AHDMIL
        return AHDMIL(bundle, profile=method, **(config or {}))
    if method.startswith('hdmil_'):
        from .hdmil import HDMIL
        return HDMIL(bundle, profile=method, **(config or {}))
    if method.startswith('mgtrans_'):
        from .mgtrans import MGTrans
        return MGTrans(bundle, profile=method, **(config or {}))
    if method.startswith('zoommil_'):
        from .zoommil import ZoomMIL
        return ZoomMIL(bundle, profile=method, **(config or {}))
    if method.startswith('augdiff_'):
        from .augdiff import AugDiffMIL
        return AugDiffMIL(bundle, profile=method, **(config or {}))
    if method.startswith('vlsa_'):
        from .vlsa import VLSAMIL
        return VLSAMIL(bundle,profile=method,**(config or {}))
    if method.startswith('slip_'):
        from .slip import SLIPMIL
        return SLIPMIL(bundle,profile=method,**(config or {}))
    if method.startswith('dispro_'):
        from .dispro import DisPro
        return DisPro(bundle,profile=method,**(config or {}))
    if method.startswith('amiml_'):
        from .amiml import AMIML
        return AMIML(bundle, profile=method, **(config or {}))
    if method.startswith('dtmil_'):
        from .dtmil import DTMIL
        return DTMIL(bundle, profile=method, **(config or {}))
    if method.startswith('pamil_policy_'):
        from .pamil_policy import PAMILPolicyMIL
        return PAMILPolicyMIL(bundle, profile=method, **(config or {}))
    if method.startswith('focus_'):
        from .focus import FOCUSMIL
        return FOCUSMIL(bundle, profile=method, **(config or {}))
    if method.startswith('codmil_'):
        from .codmil import CoDMIL
        return CoDMIL(bundle, profile=method, **(config or {}))
    if method.startswith('ins_frozen_'):
        from .ins import INSFrozenMIL
        return INSFrozenMIL(bundle, profile=method, **(config or {}))
    if method.startswith('sclwc_'):
        from .sclwc import SCLWC
        return SCLWC(bundle, profile=method, **(config or {}))
    if method.startswith('deepgraphsurv_'):
        from .deepgraphsurv import DeepGraphSurv
        return DeepGraphSurv(bundle, profile=method, **(config or {}))
    if method.startswith('triagemil_'):
        from .triagemil import TRIAGEMIL
        return TRIAGEMIL(bundle, profile=method, **(config or {}))
    if method.startswith('c2mil_'):
        from .c2mil import C2MIL
        return C2MIL(bundle, profile=method, **(config or {}))
    if method.startswith('gatmamba_'):
        from .gatmamba import GATMamba
        return GATMamba(bundle, profile=method, **(config or {}))
    if method.startswith('protosurv_'):
        from .protosurv import ProtoSurv
        return ProtoSurv(bundle, profile=method, **(config or {}))
    if method.startswith('pathrwkv_'):
        from .pathrwkv import PathRWKVMIL
        return PathRWKVMIL(bundle, profile=method, **(config or {}))
    if method.startswith('c2aug_'):
        from .c2aug import C2AugMIL
        return C2AugMIL(bundle, profile=method, **(config or {}))
    if method.startswith('agdrmil_'):
        from .agdrmil import AGDRMIL
        return AGDRMIL(bundle,profile=method,**(config or {}))
    if method.startswith('h2mil_'):
        from .h2mil import H2MIL
        return H2MIL(bundle,profile=method,**(config or {}))
    if method.startswith('hipt_lgp_'):
        from .hipt import HIPTMIL
        return HIPTMIL(bundle, profile=method, **(config or {}))
    if method.startswith('simil_'):
        from .simil import SIMIL
        return SIMIL(bundle, profile=method, **(config or {}))
    if method.startswith('mamba2dmil_'):
        from .mamba2d import Mamba2DMIL
        return Mamba2DMIL(bundle, profile=method, **(config or {}))
    if method.startswith('ttis_'):
        from .ttis import TTISMIL
        return TTISMIL(bundle, profile=method, **(config or {}))
    if method.startswith('msattention_'):
        from .msattention import MultiAttentionMIL
        return MultiAttentionMIL(bundle, profile=method, **(config or {}))
    if method.startswith('pmil_shapley_'):
        from .pmil_shapley import PMILShapley
        return PMILShapley(bundle, profile=method, **(config or {}))
    if method.startswith('mammoth_'):
        from .mammoth import MammothMIL
        return MammothMIL(bundle, profile=method, **(config or {}))
    if method == 'campanella_rnn':
        from .campanella import CampanellaMIL
        return CampanellaMIL(bundle, **(config or {}))
    if method.startswith('monai_'):
        from .monai_mil import MONAIMIL
        return MONAIMIL(bundle, profile=method, **(config or {}))
    if method.startswith('stablemil_'):
        from .stablemil import StableMIL
        return StableMIL(bundle, profile=method, **(config or {}))
    if method in {'setmil_source', 'setmil_repaired'}:
        from .setmil import SETMIL
        return SETMIL(bundle, profile=method, **(config or {}))
    if method in {'gtp_source', 'gtp_repaired'}:
        from .gtp import GTP
        return GTP(bundle, profile=method, **(config or {}))
    if method.startswith('probsa_'):
        from .probsa import ProbSA
        return ProbSA(bundle, profile=method, **(config or {}))
    if method == 'micromil_frozen':
        from .micromil import MicroMIL
        return MicroMIL(bundle, **(config or {}))
    if method.startswith('loss_attention_'):
        from .loss_attention import LossAttention
        return LossAttention(bundle, profile=method, **(config or {}))
    if method.startswith('pamil_prototype_'):
        from .pamil import PAMIL
        return PAMIL(bundle, profile=method, **(config or {}))
    if method in {'smmile', 'smmile_binary', 'smmile_multilabel'}:
        from .smmile import SMMILe
        return SMMILe(bundle, profile=method, **(config or {}))
    if method in {"momil", "momil_survival_extension"}:
        from .momil import MoMIL
        return MoMIL(bundle, profile=method, **(config or {}))
    if method.startswith('recomil_'):
        from .recomil import ReCoMIL
        return ReCoMIL(bundle, profile=method, **(config or {}))
    if method in {'mammil', 'mammil_source', 'mammil_patient_union'}:
        from .mammil import MamMIL
        return MamMIL(bundle, profile=method, **(config or {}))
    if method.startswith('mfemil_'):
        from .mfemil import MFEMIL
        return MFEMIL(bundle, profile=method, **(config or {}))
    if method == 'msmmil':
        from .msmmil import MSMMIL
        return MSMMIL(bundle, **(config or {}))
    if method.startswith('mambamil_'):
        from .mamba_mil import MambaMIL
        return MambaMIL(bundle, profile=method, **(config or {}))
    if method in {'dgmil', 'dgmil_source_extremes'}:
        from .dgmil import DGMIL
        return DGMIL(bundle, profile=method, **(config or {}))
    if method.startswith('pamoe_'):
        from .pamoe import PAMoE
        return PAMoE(bundle, profile=method, **(config or {}))
    if method.startswith('pathomics_'):
        from .pathomics import PathOmicsSurvival
        return PathOmicsSurvival(bundle.dim, bundle.omics, profile=method, **(config or {}))
    if method.startswith('mmp_'):
        from .mmp import MultimodalPrototyping
        return MultimodalPrototyping(bundle.dim, bundle.omics, profile=method, **(config or {}))
    if method == 'pibd_source':
        from .pibd import PIBDSurvival
        return PIBDSurvival(bundle.dim, bundle.omics, **(config or {}))
    if method.startswith('advmil_'):
        from .advmil import AdversarialSurvival
        return AdversarialSurvival(bundle.dim, bundle.config.get('time_unit'), profile=method, **(config or {}))
    if method == 'hmil_source_repaired':
        from .hmil import HierarchicalMIL
        return HierarchicalMIL(bundle.dim, len(bundle.classes), bundle.hierarchy, **(config or {}))
    if method == 'cate':
        from .concept_cate import ConceptCATE
        return ConceptCATE(bundle.dim, len(bundle.classes), bundle.vision_language, **(config or {}))
    if method in {'vilamil', 'vilamil_source'}:
        from .vila import ViLaMIL
        return ViLaMIL(bundle.dim, len(bundle.classes), bundle.vision_language, profile=method, **(config or {}))
    if method == 'he2rna':
        from .he2rna import HE2RNARegression
        return HE2RNARegression(bundle.dim, bundle.regression_targets, **(config or {}))
    if method == 'healnet':
        from .healnet import HEALNetSurvival
        return HEALNetSurvival(bundle.dim, bundle.omics, **(config or {}))
    if method == 'scmil':
        from .continuous_survival import SparseContextSurvival
        if 'time_unit' in (config or {}):
            raise ValueError('Set time_unit in task.yaml, not model options')
        return SparseContextSurvival(bundle.dim, time_unit=bundle.config.get('time_unit'), **(config or {}))
    from .bayesian import BAYES_METHODS, BayesMIL, check_bayes_bundle
    if method in BAYES_METHODS:
        options = dict(config or {})
        if method == 'bayesmil_apcrf':
            if 'patch_size' in options:
                raise ValueError('APCRF patch_size belongs in encoder.spatial metadata')
            options['patch_size'] = check_bayes_bundle(bundle)
        return BayesMIL(bundle.dim, len(bundle.classes), profile=method, **options)
    from .spatial_attention import SM_METHODS, ProbabilisticSpatialMIL, SmoothMIL, spatial_metadata
    if method in SM_METHODS | {'psamil', 'slidegraph'}:
        meta = spatial_metadata(bundle)
        options = dict(config or {})
        if 'patch_size' in options:
            raise ValueError('Set spatial patch_size in the encoder metadata, not model overrides')
        if method == 'slidegraph':
            from .slidegraph import SlideGraphMIL
            if meta['unit'] != 'level0_pixels' and not {'lambda_d', 'edge_distance'} <= options.keys():
                raise ValueError('SlideGraph non-pixel units require explicit lambda_d and edge_distance')
            return SlideGraphMIL(bundle.dim, len(bundle.classes), **options)
        if method in SM_METHODS:
            return SmoothMIL(bundle.dim, len(bundle.classes), meta['patch_size'], profile=method, **options)
        return ProbabilisticSpatialMIL(bundle.dim, len(bundle.classes), meta['patch_size'], **options)
    from .crossmodal import CROSSMODAL_METHODS, CrossModalSurvival
    if method in CROSSMODAL_METHODS:
        if bundle.omics is None:
            raise ValueError(f'{method} requires patient-aligned omics in task.yaml')
        return CrossModalSurvival(method, bundle.dim, bundle.omics, **(config or {}))
    from .multimodal import MULTIMODAL_METHODS, MultimodalSurvival
    if method in MULTIMODAL_METHODS:
        if bundle.omics is None:
            raise ValueError(f'{method} requires patient-aligned omics in task.yaml')
        return MultimodalSurvival(method, bundle.dim, bundle.omics, **(config or {}))
    return build_model(method, bundle.dim, 1 if bundle.task_type == 'survival' else len(bundle.classes), config)


def build_model(method, dim, classes, config=None):
    if method == 'tpmil':
        from .tpmil import TPMIL
        return TPMIL(dim, classes, **(config or {}))
    if method == 'cdpmil_source':
        from .cdpmil import CDPMILSource
        return CDPMILSource(dim, classes, **(config or {}))
    if method.startswith('rankmix_'):
        from .rankmix import RankMix
        return RankMix(dim, classes, backbone=method.split('_')[1],
                       literal=method.endswith('_source'), **(config or {}))
    if method == 'camil_channel':
        from .channel_camil import ChannelCAMIL
        return ChannelCAMIL(dim, classes, **(config or {}))
    if method == 'camil_context':
        from .context_camil import ContextCAMIL
        return ContextCAMIL(dim, classes, **(config or {}))
    if method == 'dyhg':
        from .dynamic_hypergraph import DynamicHypergraphMIL
        return DynamicHypergraphMIL(dim, classes, **(config or {}))
    if method.startswith('mfcmil_'):
        from .frequency_causal import FrequencyCausalMIL
        return FrequencyCausalMIL(dim, classes, backbone=method.removeprefix('mfcmil_'), **(config or {}))
    if method.startswith('sdfp_'):
        from .slide_prompt import SlidePromptMIL
        return SlidePromptMIL(dim, classes, backbone=method.removeprefix('sdfp_'), **(config or {}))
    if method.startswith('mil_dropout_'):
        from .mil_dropout import MILDropout
        return MILDropout(dim, classes, backbone=method.removeprefix('mil_dropout_'), **(config or {}))
    if method.startswith('packmil_'):
        from .packing import PackedMIL
        return PackedMIL(dim, classes, survival=method == 'packmil_survival', **(config or {}))
    c = dict(config or {})
    if method == 'deepattnmisl':
        from .cluster_survival import DeepAttentionSurvival
        return DeepAttentionSurvival(dim, **c)
    if method in {'mico', 'mico_survival'}:
        from .context_clustering import MiCoClassification, MiCoSurvival
        return MiCoSurvival(dim, **c) if method == 'mico_survival' else MiCoClassification(dim, classes, **c)
    from .efficient_sequence import EFFICIENT_METHODS, EfficientSequenceMIL
    if method in EFFICIENT_METHODS:
        return EfficientSequenceMIL(dim, classes, profile=method, **c)
    from .context_variance import CAPR_METHODS, ContextPatchMIL, VarianceMIL
    if method in CAPR_METHODS:
        return ContextPatchMIL(dim, classes, profile=method, **c)
    if method == 'varmil':
        return VarianceMIL(dim, classes, **c)
    if method in {'csmil', 'hagmil_ce'}:
        from .multiscale import CrossScaleMIL, HierarchicalAttentionMIL
        return {'csmil': CrossScaleMIL, 'hagmil_ce': HierarchicalAttentionMIL}[method](dim, classes, **c)
    if method == 'cbmil':
        from .cluster_contrastive import ClusterContrastiveMIL
        return ClusterContrastiveMIL(dim, classes, **c)
    if method == 'sgpmil':
        from .gaussian_process import SparseGPMIL
        return SparseGPMIL(dim, classes, **c)
    if method in {'mmil', 'mmil_coords'}:
        from .multilevel import MMIL
        if method == 'mmil_coords':
            if c.get('grouping', 'coords') != 'coords':
                raise ValueError('mmil_coords requires coordinate grouping')
            c['grouping'] = 'coords'
        elif c.get('grouping') == 'coords':
            raise ValueError('Use the explicit mmil_coords profile for spatial inputs')
        return MMIL(dim, classes, **c)
    if method == 'ptcmil':
        from .prompt_clustering import PTCMIL
        return PTCMIL(dim, classes, **c)
    if method == 'patchgcn':
        from .graph_survival import PatchGCNSurvival
        return PatchGCNSurvival(dim, **c)
    if method in {"psemix_abmil", "prodiv_abmil", "remix_abmil"}:
        from .augmentation import PseudoBagMIL, ReMix
        if method == "remix_abmil":
            return ReMix(dim, classes, **c)
        return PseudoBagMIL(dim, classes, mode=method.split("_")[0], **c)
    if method == "dgrmil":
        from .dgr import DGRMIL
        return DGRMIL(dim, classes, **c)
    if method in {"frmil", "focusmil"}:
        from .instance_models import FRMIL, FocusMIL
        return {"frmil": FRMIL, "focusmil": FocusMIL}[method](dim, classes, **c)
    from .expansion import EXPANSION_METHODS, build_expansion
    if method in EXPANSION_METHODS:
        return build_expansion(method, dim, classes, c)
    if method in {"h2t", "panther", "ibmil", "chowder", "varpool", "varpool_cox"}:
        from .prototype_models import IBMIL, Chowder, PrototypeMIL, VarPool
        if method in {"h2t", "panther"}:
            return PrototypeMIL(dim, classes, mode=method, **c)
        return {"ibmil": IBMIL, "chowder": Chowder, "varpool": VarPool,
                "varpool_cox": VarPool}[method](dim, classes, **c)
    if method == "mhim_abmil":
        from .teacher import MHIM
        return MHIM(dim, classes, **c)
    if method in {"aem_abmil", "acmil_ga", "ilra", "wikg"}:
        from .extended import ACMIL, AEM, ILRA, WiKG
        return {"aem_abmil": AEM, "acmil_ga": ACMIL, "ilra": ILRA, "wikg": WiKG}[method](dim, classes, **c)
    if method in {"mean_mil", "max_mil"}:
        return PoolMIL(dim, classes, method.split("_")[0], **c)
    if method in {"abmil", "gated_abmil", "abmil_cox", "gated_abmil_cox"}:
        return ABMIL(dim, classes, gated=method.startswith("gated"), **c)
    if method in {"clam_sb", "clam_mb"}:
        return CLAM(dim, classes, multibranch=method == "clam_mb", **c)
    mapping = {"dsmil": DSMIL, "transmil": TransMIL, "dtfd": DTFD}
    if method not in mapping:
        raise ValueError(f"No runnable adapter for {method}; see `milplus inspect`")
    return mapping[method](dim, classes, **c)

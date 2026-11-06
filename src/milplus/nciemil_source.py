"""Actual NcIEMIL aggregator components; full filtering/pretraining is not yet admitted.

The author repository supplies the aggregator and its SGD loop, but omits the
task-trained Swin filtering stage. This module is deliberately not a public
runnable method profile until that whole mechanism has an executable contract.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_

from .io import digest_file
from .upstream import load_definitions,verified_source

COMMIT='780d8b4c04ef6266e552bfd4c47b76470f7510fd'
VARIANTS={'released_interleaved_kv','paper_independent_kv_repaired'}


def checked_source():
    root,sha=verified_source('nciemil')
    if sha!=COMMIT:raise ValueError('NcIEMIL original source revision changed')
    return root,sha


def author_classes(variant='released_interleaved_kv'):
    checked_source()
    if variant not in VARIANTS:raise ValueError('NcIEMIL must name its source or independently projectedKV condition')
    replacements=[]
    if variant=='paper_independent_kv_repaired':
        replacements=[('self._kv = nn.Linear(latent_dim, latent_dim, bias=qkv_bias)',
                       'self._kv = nn.Linear(latent_dim, 2 * latent_dim, bias=qkv_bias)')]
    return load_definitions('nciemil','model/model.py',['NcIEMIL','ChannelAttention','SpatialAttention'],
        {'torch':torch,'nn':nn,'F':F,'math':math,'trunc_normal_':trunc_normal_},replacements)


def discriminative_selection(features,positive_probabilities,k):
    """Paper Eq4: highK followed by lowK, retaining descending score order.

    Only a mathematical component. A complete method must bind these actual
    probabilities to the trained filtering network and original patch IDs.
    """
    if (type(k) is not int or k<1 or features.ndim!=2 or len(features)<2*k
            or positive_probabilities.shape!=(len(features),)
            or not torch.isfinite(features).all() or not torch.isfinite(positive_probabilities).all()
            or (positive_probabilities<0).any() or (positive_probabilities>1).any()):
        raise ValueError('NcIEMIL selection requires full aligned features,actual finite positive probabilities and at least2K instances')
    indices=torch.argsort(positive_probabilities,descending=True,stable=True)
    selected=torch.cat([indices[:k],indices[-k:]])
    if len(selected.unique())!=2*k:raise ValueError('NcIEMIL high/low selections must not duplicate instances')
    return features.index_select(0,selected),selected


def max_instance_bce(instance_logits,weak_targets):
    """Paper Eq3, exactly one maximum-logit instance per binary-supervised bag.

    This is the defined objective component, not a replacement for training
    the missing full Swin-Tiny extraction/filtering network. Ties use the first
    original patch index explicitly; no averaging or patch-label propagation.
    """
    if (not isinstance(instance_logits,(list,tuple)) or not instance_logits
            or weak_targets.shape!=(len(instance_logits),)
            or not torch.isfinite(weak_targets).all() or not ((weak_targets==0)|(weak_targets==1)).all()):
        raise ValueError('NcIEMIL Eq3 requires complete variable-length bags and explicit binary weak targets')
    if any(x.ndim!=1 or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all()
            or x.device!=weak_targets.device for x in instance_logits):
        raise ValueError('NcIEMIL Eq3 requires finite complete per-instance logits on the declared device')
    maxima=[torch.max(x,dim=0) for x in instance_logits]
    values=torch.stack([x.values for x in maxima]);indices=torch.stack([x.indices for x in maxima])
    loss=F.binary_cross_entropy_with_logits(values,weak_targets.to(values.dtype))
    return {'loss':loss,'winning_original_indices':indices,'max_logits':values}


class NcIEAggregator(nn.Module):
    """Full author1024D/four-head graph for native256/1024 selected instances."""
    def __init__(self,dimension=768,classes=4,k=128,variant='released_interleaved_kv',dropout=.2):
        super().__init__()
        if (type(dimension) is not int or dimension<4 or dimension%4 or type(classes) is not int or classes<2
                or k not in {128,512} or isinstance(dropout,bool) or not isinstance(dropout,(int,float)) or not 0<=dropout<1):
            raise ValueError('NcIEMIL needs declared input width/classes,full nativeK128/512 and finite dropout')
        self.dimension,self.classes,self.k,self.variant=dimension,classes,k,variant
        self.network=author_classes(variant).NcIEMIL(in_dim=dimension,in_chans=2*k,latent_dim=1024,
            n_classes=classes,num_heads=4,ratio=32,attn_drop=dropout,proj_drop=dropout,conv_drop=dropout)

    def forward(self,x):
        if (x.ndim!=3 or x.shape[1:]!=(2*self.k,self.dimension) or not len(x) or not torch.isfinite(x).all()):
            raise ValueError('NcIEMIL aggregator requires complete ordered[B,2K,input_dim] selected bags')
        if self.training and len(x)<2:
            raise ValueError('Original channel squeeze BatchNorm requires at least2 bags; no hidden singleton normalization repair')
        return self.network(x)

    def protocol(self):
        root,_=checked_source()
        return {'commit':COMMIT,'model_sha256':digest_file(root/'model/model.py'),'variant':self.variant,
            'input_dimension':self.dimension,'latent_dimension':1024,'heads':4,'k':self.k,'instances':2*self.k,
            'channel_grid':[int(math.sqrt(2*self.k))]*2,'channel_ratio':32,
            'actual_KV_output_dimension':self.network._kv.out_features,
            'source_initialization':'Preserve default Linear/Conv initialization; unused author _init_weights is not applied; original CLS randn then actual timm trunc_normal .02',
            'source_KV_discrepancy':'Released Linear1024to1024 reshapes neighboring token pairs into512or128 keys/values; paper Eq7 calls for independent K/V projections. Explicit repaired condition is Linear1024to2048; never silently relabel.',
            'spatial_semantics':'Source grid is highK/lowK probability-rank order, not measured physical coordinates',
            'full_method_admission':False,'missing':'Task-trained full Swin-Tiny filtering/extraction and provenance-bound selection pipeline; complete training/recovery/publicCLI still required',
            'paper_reproduction':'not_run','real_wsi_validation':'not_run'}

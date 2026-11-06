"""Full native TOD-MIL and original SP-LNPL operators; not a full method adapter.

The external weak ResNet18/LPA stage and its actual pseudo scores are mandatory
parts of LNPL-MIL. No feature-only TOD head is registered as the full method.
"""
from functools import lru_cache, partial
from itertools import chain
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from .upstream import load_definitions, verified_source

COMMIT = '1dc4e703a2b86dd0950e0b75aa7d45337d2127cb'

def checked_source():
    root, commit = verified_source('lnplmil')
    if commit != COMMIT:
        raise ValueError('Use the pinned LNPL-MIL source and its actual LPA/weak-score pipeline')
    return root

@lru_cache(maxsize=1)
def definitions():
    checked_source()
    return load_definitions('lnplmil','models/TODMIL.py',
        ['PreNormResidual','FeedForward','positionalencoding2d','FixedPositionalEncoding_2d',
         'PreNorm','FeedForward_','Attention','Transformer','AttentionLayer','TODMIL'],
        {'np':np,'torch':torch,'nn':nn,'einsum':einsum,'F':F,'math':math,'partial':partial,
         'rearrange':rearrange,'reduce':reduce,'Rearrange':Rearrange,'Reduce':Reduce})

def model():
    source = definitions()
    # Exact released 200-patch/512D/4-layer configuration, not the smaller
    # two-layer __main__ FLOP demo and not a replacement attention pool.
    return source.TODMIL(num_layers=4, dim=256, num_heads=8, ff_dim=1024,
        dropout=0., input_feature=512, n_classes=2, num_patches=200,
        expansion_factor=2., expansion_factor_token=.5)

def paper_model():
    """Paper Eq.2 residual convolution; released code drops this residual.

    All author layer constructors, dimensions, IOA-MLP and BG-attention remain.
    This explicit reference condition is not labelled as the released forward.
    """
    source=definitions()
    class PaperTOD(source.TODMIL):
        def forward(self, **kwargs):
            original=kwargs['data'].float()
            if original.shape!=(1,200,515):
                raise ValueError('Native paper TOD reference keeps one200-patch512D scored bag')
            score,coords=original[...,0],original[...,1:3]
            x=self.position_layer(self._fc1(original[...,3:]),coords)
            for convolution,attention,feedforward in self.C_Trans.layers:
                x=convolution(x.transpose(1,2)).transpose(1,2)+x
                x=attention(x)+x
                x=feedforward(x)+x
            x=self.IOAMLP(x)
            context,_,_=self.BGAttn(x.squeeze(0),self._fc2.weight,self._fc2.bias,1)
            bag=context.mean(0,keepdim=True)
            logits=self._fc2(bag)
            return {'logits':logits,'Y_prob':F.softmax(logits,dim=1),
                    'Y_hat':logits.argmax(dim=1),'Score':score,'logits_ratio':self._fc3(bag)}
    return PaperTOD(num_layers=4,dim=256,num_heads=8,ff_dim=1024,dropout=0.,
        input_feature=512,n_classes=2,num_patches=200,expansion_factor=2.,expansion_factor_token=.5)

def validate_scored_features(value, minimum=200):
    if (not torch.is_tensor(value) or value.ndim!=2 or value.shape[1]!=515
            or len(value)<minimum or value.dtype!=torch.float32 or not torch.isfinite(value).all()
            or value.requires_grad):
        raise ValueError('LNPL needs actual frozen512D patch features, positive scores and original coordinates, N>=200')
    scores, coords = value[:,0], value[:,1:3]
    if ((scores<0)|(scores>1)).any() or (coords<0).any() or (coords>=240000).any():
        raise ValueError('Actual weak positive probabilities or original1200-stride positional grid are invalid')
    return value

def select_top200(scored_features):
    value = validate_scored_features(scored_features)
    # Keep the source torch.topk tie/order behavior, rather than reorder tokens.
    indices = torch.topk(value[:,0],200).indices
    return value[indices], indices

@lru_cache(maxsize=1)
def superpatch_definitions():
    checked_source()
    import nmslib
    return load_definitions('lnplmil','knn_feature.py',['Hnsw','pt2graph'],
        {'np':np,'torch':torch,'nmslib':nmslib,'chain':chain})

@lru_cache(maxsize=1)
def canonical_superpatch_definitions():
    """Original HNSW equations with explicit reproducible construction policy.

    Source default parallel index construction changes selected patch identities
    between processes on tied/repeated features. Do not silently rename this
    policy as literal default-thread author execution. M16/post0/ef400/query90,
    approximate search, ROI and TopK equations remain unchanged.
    """
    source = superpatch_definitions()
    class SingleBuildThreadHnsw(source.Hnsw):
        def __init__(self, space='cosinesimil'):
            super().__init__(space=space, index_params={
                'M':16,'post':0,'efConstruction':400,'indexThreadQty':1})
    body = load_definitions('lnplmil','knn_feature.py',['pt2graph'],
        {'np':np,'torch':torch,'chain':chain,'Hnsw':SingleBuildThreadHnsw})
    return SimpleNamespace(Hnsw=SingleBuildThreadHnsw,pt2graph=body.pt2graph)


def filter_superpatches(scored_features, radius=50, ratio=.4, min_size=200, canonical=False):
    value = validate_scored_features(scored_features)
    if (value.device.type!='cpu' or type(radius) is not int or not 1<radius<=len(value)
            or type(min_size) is not int or min_size<200
            or type(ratio) not in {int,float} or not 0<=ratio<=1):
        raise ValueError('Declare the actual CPU HNSW radius/ratio/minimum for SP-LNPL')
    # Original HNSW: l2, M16/post0/efConstruction400/queryef90. Its full
    # reflect padding, median threshold and minimum-size return rule execute.
    try:
        source = canonical_superpatch_definitions() if canonical else superpatch_definitions()
        result = source.pt2graph(value,radius=radius,ratio=ratio,min_size=min_size)
    except ValueError as error:
        if str(error) == 'need at least one array to stack':
            raise ValueError('Released SP-LNPL selected no eligible superpatches; the original source fails before its minimum-size fallback') from error
        raise
    return validate_scored_features(result.to(dtype=torch.float32))

def paper_disjoint_superpatches(scored_features, radius=50, ratio=.4, canonical=False):
    """Paper Algorithm1: iterative feature-nearest clusters, retaining all members.

    Original source HNSW remains the backend. The paper's strict adaptive
    median and actual deletion are used; source one-shot representatives and
    its minimum-size input-return rule are not silently substituted.
    Reflect padding follows the released preprocessing. Insufficient selected
    patches fail explicitly, since the paper supplies no fallback rule.
    """
    value=validate_scored_features(scored_features)
    if value.device.type!='cpu' or type(radius) is not int or not 1<radius<=len(value):
        raise ValueError('Use the actual CPU feature HNSW and a valid superpatch size')
    if type(ratio) not in {int,float} or not 0<=ratio<=1:
        raise ValueError('Declare a valid paper ROI-positive ratio')
    rows=value.numpy()
    extra=(len(rows)//radius+1)*radius-len(rows)
    padded=np.pad(rows,((extra//2,extra-extra//2),(0,0)),mode='reflect')
    roi=padded[:,0]>np.median(padded[:,0])
    remaining=np.arange(len(padded))
    selected=[];clusters=[]
    Source=canonical_superpatch_definitions() if canonical else superpatch_definitions()
    while len(remaining):
        assert len(remaining)>=radius
        features=padded[remaining,3:]
        search=Source.Hnsw(space='l2').fit(features)
        local=np.asarray(search.query(features[0],topn=radius))
        if len(local)!=radius or len(np.unique(local))!=radius or (local<0).any() or (local>=len(remaining)).any():
            raise ValueError('Actual source HNSW returned invalid disjoint superpatch members')
        members=remaining[local]
        fraction=float(roi[members].sum()/radius)
        keep=fraction>ratio
        if keep:selected.extend(members.tolist())
        clusters.append({'padded_indices':members.tolist(),'positive_ratio':fraction,'retained':keep})
        remaining=np.delete(remaining,local)
    if len(selected)<200:
        raise ValueError('Paper SP-LNPL produced fewer than200 eligible patches; no undocumented input fallback')
    output=torch.from_numpy(padded[np.asarray(selected)].copy()).to(dtype=torch.float32)
    return validate_scored_features(output),{'clusters':clusters,'padded_patch_count':len(padded),
        'original_patch_count':len(value),'paper_strict_median':float(np.median(padded[:,0])),
        'padding':'released_reflect_full_next_window','minimum_fallback':'none_fail_explicitly'}

def objective(net, selected, target, variant='paper_auxiliary', ratio_weight=.001):
    value = validate_scored_features(selected)
    if len(value)!=200 or target.dtype!=torch.long or target.shape!=(1,) or int(target.item()) not in {0,1}:
        raise ValueError('Native TOD-MIL requires exactly200 scored patches and one binary slide label')
    if variant not in {'paper_auxiliary','released_overwritten_ce'}:
        raise ValueError('Explicitly distinguish the paper auxiliary objective from the released CE overwrite')
    if type(ratio_weight) not in {float,int} or not math.isfinite(ratio_weight) or ratio_weight<0:
        raise ValueError('Declare a finite nonnegative distribution-loss weight')
    output = net(data=value.unsqueeze(0))
    fraction = (output['Score']>.5).sum().to(dtype=torch.float32)/200
    interval = torch.bucketize(fraction, fraction.new_tensor([.25,.5,.75]), right=True).reshape(1).long()
    slide_ce = F.cross_entropy(output['logits'],target)
    ratio_ce = F.cross_entropy(output['logits_ratio'],interval)
    loss = slide_ce+ratio_weight*ratio_ce if variant=='paper_auxiliary' else slide_ce
    if not torch.isfinite(loss):
        raise FloatingPointError('Native LNPL slide/distribution objective became nonfinite')
    return {**output,'loss':loss,'slide_ce':slide_ce,'ratio_ce':ratio_ce,'ratio_label':interval,
            'ratio_fraction':fraction,'objective_variant':variant}

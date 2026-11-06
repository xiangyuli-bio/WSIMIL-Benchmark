"""Complete GMMamba graph with an explicit pinned BiMamba dependency port.

The public GMMamba tree imports `mamba.mamba_ssm.BiMamba` but does not
include that directory. This condition binds the actual MambaMIL BiMamba,
not a plain-Mamba or small surrogate. Its separate identity remains visible.
"""
import math
from types import SimpleNamespace
from typing import Optional

from einops import rearrange,repeat
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import Tensor,nn
from torch.nn import functional as F

from .io import digest_file
from .mamba_mil import backend_functions
from .upstream import load_definitions,verified_source

COMMIT='70576f1aaeef0d7041d6860799ade1ac905b85aa'
BIMAMBA_COMMIT='9aea39efc13fdcd3703b5b54c28a64887b4acbd7'
PROFILES={'gmmamba_mambamil_bimamba_source','gmmamba_fm_projection_repaired'}


def checked_source():
    root,sha=verified_source('gmmamba')
    if sha!=COMMIT:raise ValueError('GMMamba original source revision changed')
    return root,sha


def dependency(backend='torch',scan_chunk=64):
    root,sha=verified_source('mambamil')
    if sha!=BIMAMBA_COMMIT:raise ValueError('GMMamba explicit BiMamba dependency revision changed')
    no_out,with_out,identity=backend_functions(backend,scan_chunk)
    def forbidden(*args,**kwargs):
        raise RuntimeError('GMMamba requires its declared complete bidirectional fast graph; no slow/plain-Mamba fallback')
    cls=load_definitions('mambamil','mamba/mamba_ssm/modules/bimamba.py',
        ['TransposeTokenReEmbedding','BiMamba'],
        {'math':math,'torch':torch,'nn':nn,'F':F,'Tensor':Tensor,'Optional':Optional,
         'rearrange':rearrange,'repeat':repeat,'mamba_inner_fn_no_out_proj':no_out,
         'mamba_inner_fn':with_out,'selective_scan_fn':forbidden}).BiMamba
    return cls,{'source':'mambamil','commit':sha,
        'file_sha256':digest_file(root/'mamba/mamba_ssm/modules/bimamba.py'),'backend':identity,
        'limitation':'Explicit dependency reconstruction; original GMMamba repository omits its imported mamba directory'}


def author_classes(backend='torch',scan_chunk=64):
    checked_source();bimamba,identity=dependency(backend,scan_chunk)
    namespace={'torch':torch,'nn':nn,'F':F,'rearrange':rearrange}
    imm=load_definitions('gmmamba','models/IMM.py',['initialize_weights','IMM_block'],
        {**namespace,'BiMamba':bimamba},
        [('mask = torch.ones(len_ins, dtype=torch.bool)',
          'mask = torch.ones(len_ins, dtype=torch.bool, device=h_orgin.device)')]).IMM_block
    css=load_definitions('gmmamba','models/CSS.py',['Memory','CSS_block'],namespace)
    classifier=load_definitions('gmmamba','models/classmodel.py',
        ['Attention','AttenLayer','GroupsSuperAttenLayer','ClassMultiMILTransformer'],namespace)
    return imm,css,classifier,identity


def author_grouping():
    checked_source()
    return load_definitions('gmmamba','datasets/pre_input_data.py',['grouping'],
        {'torch':torch,'np':np,'KMeans':KMeans}).grouping


def validated_groups(features,coords,groups=10):
    if (type(groups) is not int or groups<2 or features.ndim!=2 or coords.ndim!=2
            or coords.shape!=(len(features),2) or len(features)<2*groups
            or not torch.isfinite(features).all() or not torch.isfinite(coords).all()
            or len(torch.unique(coords,dim=0))<groups):
        raise ValueError('GMMamba needs full aligned finite features/real coordinates and at least two instances per original spatial group')
    result=author_grouping()(groups).coords_grouping(coords,features)
    # The released `.squeeze()` produces a scalar index for a singleton group,
    # and masking removes at least one instance. Never invent/pad instances or
    # silently reduce group count to make this invalid source input run.
    if len(result)!=groups or any(x.ndim!=3 or x.shape[0]!=1 or x.shape[1]<2 or x.shape[2]!=features.shape[1] for x in result):
        raise ValueError('GMMamba source grouping contains an empty/singleton cluster; input does not satisfy the original IMM')
    if sum(x.shape[1] for x in result)!=len(features):
        raise ValueError('GMMamba grouping lost original feature rows')
    return result


class ProjectedIMM(nn.Module):
    """Explicit FM-input repair using the actual otherwise-unused source FC1."""
    def __init__(self,source):
        super().__init__();self.source=source

    def forward(self,x,first=False):
        if x.ndim==2:x=x.unsqueeze(0)
        return self.source(self.source._fc1(x.float()),first=first)


class GMMambaNetwork(nn.Module):
    def __init__(self,dimension=512,classes=2,profile='gmmamba_mambamil_bimamba_source',
                 mask_ratio=.3,groups=10,backend='torch',scan_chunk=64):
        super().__init__()
        if (profile not in PROFILES or type(dimension) is not int or dimension<1 or classes not in {2,3}
                or type(groups) is not int or groups<2 or isinstance(mask_ratio,bool)
                or not isinstance(mask_ratio,(float,int)) or not 0<=mask_ratio<1):
            raise ValueError('Invalid complete GMMamba source condition')
        if profile=='gmmamba_mambamil_bimamba_source' and dimension!=512:
            raise ValueError('Released GMMamba IMM bypasses its input projection and requires512D; select explicit FM-projection repair for other FM widths')
        imm,css,classifier,self.dependency_identity=author_classes(backend,scan_chunk)
        original=imm(in_dim=dimension,n_classes=classes,dropout=.25,act='gelu',layer=1,mask_ratio=mask_ratio)
        self.imm=ProjectedIMM(original) if profile=='gmmamba_fm_projection_repaired' else original
        # Literal create_model order: IMM, CSS, then classifier. Keep all
        # initialization draws and native512/8head/16state/4conv/2expand widths.
        self.css=css.CSS_block(dim=512,n_classes=classes,n_iter=1)
        self.classifier=classifier.ClassMultiMILTransformer(SimpleNamespace(embed_dim=512,n_classes=classes,num_layers=1))
        self.dimension,self.classes,self.groups,self.profile=dimension,classes,groups,profile
        self.mask_ratio=mask_ratio;self.memory=css.Memory()

    def forward(self,features,coords):
        if features.shape[1]!=self.dimension:raise ValueError('GMMamba registered FM input width changed')
        if self.memory.msg_states:raise ValueError('GMMamba memory must be cleared between original slide bags')
        grouped=validated_groups(features,coords,self.groups)
        try:
            for group in grouped:self.memory.msg_states.append(self.imm(group,first=True))
            x=torch.stack(self.memory.msg_states,dim=1).view(1,-1,1,512)
            return self.classifier(self.css.stoken_forward(x))
        finally:self.memory.clear_memory()

    def protocol(self):
        root,_=checked_source()
        return {'source_commit':COMMIT,'profile':self.profile,'dependency':self.dependency_identity,
            'dimension':self.dimension,'classes':self.classes,'groups':self.groups,'mask_ratio':self.mask_ratio,
            'architecture':'Two full512D bidirectional16state/4conv/expand2 scans,128D hard-mask/group attention,512D8head CSS,one8head class-token layer',
            'source_files_sha256':{name:digest_file(root/name) for name in
                ['models/IMM.py','models/CSS.py','models/classmodel.py','datasets/pre_input_data.py','utils/core.py','main.py']},
            'mask':'Original lowest-score floor(mask_ratio*N),at least1 removed,retained rows keep original order; hard indices do not backpropagate to mask_attention',
            'grouping':'Original coordinate KMeans(random_state=0,n_init=auto),same original index order; complete bags,no cached detached feature substitution',
            'FM_port':'Source-unused FC1 remains unused' if self.profile=='gmmamba_mambamil_bimamba_source' else 'Explicit source FC1/GELU/dropout.25 activated before unchanged original512D IMM',
            'device_port':'Only allocate bool hard mask on input device; equations/selection/widths unchanged',
            'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'}

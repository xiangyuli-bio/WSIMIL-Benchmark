"""Pinned MambaMIL fast-path graphs, explicit scan backends and full training recipes."""
import importlib
import importlib.metadata
import math
from collections import Counter
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .discrete_survival import DiscreteSurvival
from .io import digest_file
from .upstream import load_definitions, load_single

PROFILES = {'mambamil_srmamba': 'SRMamba', 'mambamil_bimamba': 'BiMamba', 'mambamil_unidirectional': 'Mamba'}


def _scan_chunk(state, u, delta, A, B, C):
    values=[]
    transitions=torch.exp(torch.einsum('bdl,dn->bdln',delta,A))
    updates=torch.einsum('bdl,bnl,bdl->bdln',delta,B,u)
    for t in range(u.shape[-1]):
        state = transitions[:,:,t]*state + updates[:,:,t]
        values.append(torch.einsum('bdn,bn->bd',state,C[:,:,t]))
    return torch.stack(values,-1),state


def selective_scan_torch(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                         delta_softplus=True, chunk_size=64):
    """Exact real-valued variable-B/C recurrence; checkpoint chunks, never truncate the state."""
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError('scan chunk_size must be a positive integer')
    if A.is_complex() or B.ndim!=3 or C.ndim!=3 or u.ndim!=3 or not u.shape[-1]:
        raise ValueError('MambaMIL scan requires real A and variable B,C on a nonempty sequence')
    dtype=u.dtype;u=u.float();delta=delta.float();B=B.float();C=C.float()
    if delta_bias is not None:delta=delta+delta_bias.float()[None,:,None]
    if delta_softplus:delta=F.softplus(delta)
    state=A.new_zeros((u.shape[0],A.shape[0],A.shape[1]));out=[]
    for start in range(0,u.shape[-1],chunk_size):
        sl=slice(start,start+chunk_size)
        args=(state,u[:,:,sl],delta[:,:,sl],A,B[:,:,sl],C[:,:,sl])
        if torch.is_grad_enabled() and any(x.requires_grad for x in args):
            y,state=checkpoint(_scan_chunk,*args,use_reentrant=False,preserve_rng_state=False)
        else:y,state=_scan_chunk(*args)
        out.append(y)
    y=torch.cat(out,-1)
    if D is not None:y=y+u*D[None,:,None]
    if z is not None:y=y*F.silu(z)
    return y.to(dtype)


def inner_torch(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, *, chunk_size=64):
    """Same equations as the author's MambaInnerFnNoOutProj; no generic-Mamba substitution."""
    x,z=xz.chunk(2,dim=1);length=x.shape[-1]
    x=F.silu(F.conv1d(x,conv1d_weight,conv1d_bias,padding=conv1d_weight.shape[-1]-1,groups=x.shape[1])[...,:length])
    dbl=F.linear(rearrange(x,'b d l -> (b l) d'),x_proj_weight)
    rank,state=delta_proj_weight.shape[1],A.shape[1]
    delta=rearrange(delta_proj_weight@dbl[:,:rank].t(),'d (b l) -> b d l',l=length)
    if B is None:
        B=dbl[:,rank:rank+state]
        if B_proj_bias is not None:B=B+B_proj_bias.to(B.dtype)
        B=rearrange(B,'(b l) n -> b n l',l=length).contiguous()
    if C is None:
        C=dbl[:,-state:]
        if C_proj_bias is not None:C=C+C_proj_bias.to(C.dtype)
        C=rearrange(C,'(b l) n -> b n l',l=length).contiguous()
    return selective_scan_torch(x,delta,A,B,C,D,z,delta_bias,delta_softplus,chunk_size)


def backend_functions(backend, chunk_size, source='mambamil'):
    if backend=='torch':
        def no_out(*args,**kw):return inner_torch(*args,**kw,chunk_size=chunk_size)
        def with_out(xz,cw,cb,xw,dw,ow,ob,A,*args,**kw):
            return F.linear(rearrange(no_out(xz,cw,cb,xw,dw,A,*args,**kw),'b d l -> b l d'),ow,ob)
        return no_out,with_out,{'name':'torch','scan':'chunk-checkpointed exact recurrence','chunk_size':chunk_size,'torch':torch.__version__}
    if backend!='cuda':raise ValueError('MambaMIL backend must be explicitly torch or cuda')
    if not torch.cuda.is_available():raise RuntimeError('MambaMIL cuda backend requires an accessible CUDA GPU; no automatic fallback')
    try:
        versions={p:importlib.metadata.version(p) for p in ('causal-conv1d','mamba-ssm')}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('Missing pinned CUDA dependencies: build author mamba-ssm==1.1.2 and causal-conv1d==1.1.1; no automatic fallback') from error
    if versions!={'causal-conv1d':'1.1.1','mamba-ssm':'1.1.2'}:
        raise ValueError('Build pinned MambaMIL/mamba (1.1.2) and causal-conv1d==1.1.1 in a separate compatible CUDA environment')
    binaries={name:digest_file(importlib.import_module(name).__file__) for name in ('selective_scan_cuda','causal_conv1d_cuda')}
    ops=load_single(source,'mamba/mamba_ssm/ops/selective_scan_interface.py')
    return ops.mamba_inner_fn_no_out_proj,ops.mamba_inner_fn,{'name':'cuda','versions':versions,'binary_sha256':binaries,'torch':torch.__version__,'cuda':torch.version.cuda}


def author_network(hidden=512, backend='torch', chunk_size=64):
    no_out,with_out,identity=backend_functions(backend,chunk_size)
    def forbidden(*args,**kw):raise RuntimeError('MambaMIL must execute the pinned fast-path graph; slow-path substitution is disabled')
    ns={'torch':torch,'nn':nn,'F':F,'Tensor':Tensor,'Optional':Optional,'math':math,
        'rearrange':rearrange,'repeat':repeat,'mamba_inner_fn_no_out_proj':no_out,
        'mamba_inner_fn':with_out,'selective_scan_fn':forbidden}
    root='mamba/mamba_ssm/modules/'
    for name,file in [('SRMamba','srmamba'),('BiMamba','bimamba'),('Mamba','mamba_simple')]:
        names=[name]+(['TransposeTokenReEmbedding'] if name!='Mamba' else [])
        ns[name]=getattr(load_definitions('mambamil',root+file+'.py',names,ns),name)
    model=load_definitions('mambamil','models/MambaMIL.py',['initialize_weights','MambaMIL'],ns,
        replacements=[('512',str(hidden))]).MambaMIL
    return model,identity


class MambaMIL(DiscreteSurvival):
    def __init__(self,bundle,profile='mambamil_srmamba',hidden=512,layers=2,rate=5,
                 dropout=None,backend='torch',scan_chunk=64,bins=4,alpha=0.,
                 accumulation_steps=None,weighted_sampling=None,stop_epoch=None):
        survival=bundle.task_type=='survival'
        if profile not in PROFILES or bundle.task_type not in {'classification','survival'}:
            raise ValueError('MambaMIL requires a classification or survival task and a registered variant')
        if any(type(v) is not int or v<1 for v in (hidden,layers,rate,scan_chunk)):
            raise ValueError('MambaMIL dimensions, layers, rate and scan_chunk must be positive integers')
        if not survival and bundle.config.get('training_unit','slide')!='slide':
            raise ValueError('Classification MambaMIL uses one slide bag; patient classification adaptation is not registered')
        dropout=(.25 if survival else 0.) if dropout is None else dropout
        if not 0<=dropout<1:raise ValueError('Invalid MambaMIL dropout')
        steps=(32 if survival else 1) if accumulation_steps is None else accumulation_steps
        if type(steps) is not int or steps<1:raise ValueError('accumulation_steps must be a positive integer')
        if not survival and steps!=1:raise ValueError('Classification MambaMIL updates once per bag')
        if weighted_sampling is not None and type(weighted_sampling) is not bool:raise ValueError('weighted_sampling must be boolean')
        super().__init__(bins,alpha,steps)
        self.survival=survival;self.profile=profile;self.dim=bundle.dim
        self.hidden,self.layers,self.rate=hidden,layers,rate
        self.weighted_sampling=survival if weighted_sampling is None else weighted_sampling
        self.stop_after_epoch=(41 if survival else 51) if stop_epoch is None else stop_epoch
        if type(self.stop_after_epoch) is not int or self.stop_after_epoch<0:raise ValueError('Invalid stop_epoch')
        net,self.backend_identity=author_network(hidden,backend,scan_chunk)
        self.network=net(bundle.dim,bins if survival else len(bundle.classes),dropout,'gelu',survival=False,
                         layer=layers,rate=rate,type=PROFILES[profile])
        self.author_nll=load_definitions('mambamil','utils/survival_utils.py',['nll_loss'],{'torch':torch}).nll_loss
        self.optimizer_steps=0

    def configure_training(self,config):
        if config['max_instances'] is not None:
            raise ValueError('MambaMIL preserves the full registered instance sequence; max_instances truncation is not supported')

    def prepare(self,bundle,seed,device):
        if self.survival:super().prepare(bundle,seed,device)

    def get_extra_state(self):return {'backend':self.backend_identity,'optimizer_steps':self.optimizer_steps}

    def set_extra_state(self,state):
        if state['backend']!=self.backend_identity:raise ValueError('MambaMIL backend identity changed since checkpoint creation')
        self.optimizer_steps=state['optimizer_steps']

    def make_optimizer(self,lr,weight_decay):
        # Author get_optim ignores _no_weight_decay tags, and model initialization resets dt biases.
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)

    def forward(self,x,coords=None):
        if x.ndim!=2 or x.shape[1]!=self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('MambaMIL expects finite nonempty [instances, registered FM dimension] input')
        logits=self.network(x)[0]
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite MambaMIL logits')
        if not self.survival:return {'logits':logits}
        hazards=logits.sigmoid();survival=torch.cumprod(1-hazards,dim=1)
        return {'logits':-survival.sum(1,keepdim=True),'hazard_logits':logits,'hazards':hazards,'survival':survival}

    def objective(self,result,time,event):
        return self.author_nll(result['hazards'],result['survival'],self.time_label([time]),
                               self.bin_edges.new_tensor([1-event]),alpha=self.alpha)

    def forward_unit(self,bundle,rows,device,config,train=False):
        from .engine import load_unit
        # Stable slide-ID concatenation for multi-slide patient bags, preserving each stored patch order.
        rows=sorted(rows,key=lambda r:r['slide_id'])
        x,_=load_unit(bundle,rows,device)
        return self(x)

    def protocol(self):
        return {**(super().protocol() if self.survival else {}),'profile':self.profile,
                'variant':PROFILES[self.profile],'backend':self.backend_identity,'hidden':self.hidden,
                'layers':self.layers,'rate':self.rate,'optimizer':'Adam all parameters; source weight decay tags not applied',
                'initialization':'literal author initializer, including zeroing marked dt_proj.bias',
                'order':'stored patch order; patient slides sorted by slide_id; no permutation or hidden sampling',
                'objective':'source eps-clamped discrete NLL' if self.survival else 'CE',
                'selection':'validation c-index for survival; validation log-loss for classification',
                'weighted_sampling':self.weighted_sampling,'optimizer_steps':self.optimizer_steps,
                'survival_repairs':'training-only event quantiles; flush and average final short accumulation window',
                'fast_graph':'SR transpose-pad/inverse, Bi reverse/inverse, uni original; no author slow-path fallback',
                'gpu_validation':'not_run','paper_reproduction':'not_run'}


def sampled_groups(model,groups):
    if not model.weighted_sampling:return groups
    labels=[(int(model.time_label([g[0]['time']]).item()),1-g[0]['event']) if model.survival else g[0]['target'] for g in groups]
    counts=Counter(labels)
    weights=torch.tensor([len(labels)/counts[x] for x in labels],dtype=torch.double)
    indices=list(torch.utils.data.WeightedRandomSampler(weights,len(groups),replacement=True))
    return [groups[i] for i in indices]


def mamba_epoch(model,bundle,groups,optimizer,device,config):
    groups=sampled_groups(model,groups);losses=[];model.train()
    for start in range(0,len(groups),model.accumulation_steps):
        window=groups[start:start+model.accumulation_steps];optimizer.zero_grad(set_to_none=True)
        for rows in window:
            output=model.forward_unit(bundle,rows,device,config,train=True)
            loss=model.objective(output,rows[0]['time'],rows[0]['event']) if model.survival else F.cross_entropy(output['logits'],torch.tensor([rows[0]['target']],device=device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite MambaMIL loss')
            (loss/len(window)).backward();losses.append({'loss':float(loss.detach())})
        if config.get('grad_clip') is not None:nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite MambaMIL gradients')
        optimizer.step();model.optimizer_steps+=1
    return losses

"""Complete pinned StableMIL networks and explicitly repaired classification/survival wiring."""
import collections.abc
import math
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from .discrete_survival import DiscreteSurvival
from .loss_attention import check_update_finite, finite
from .smmile_data import load_spatial_inputs, positive_int
from .spatial_attention import spatial_metadata
from .stable_geometry import NATIVE_RADIUS, GeometryCache
from .upstream import load_definitions

PROFILES={'stablemil_classification_source','stablemil_classification_repaired','stablemil_survival_source','stablemil_survival_repaired'}


def author_model(repaired=False):
    helpers=load_definitions('timm_stablemil','timm/layers/helpers.py',['_ntuple'],{'collections':collections,'repeat':repeat})
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'partial':partial,'Optional':Optional,'Type':type,'Union':Union,
        'Tuple':tuple,'to_2tuple':helpers._ntuple(2),'autocast':autocast}
    ns['DropPath']=load_definitions('timm_stablemil','timm/layers/drop.py',['drop_path','DropPath'],ns).DropPath
    ns['Mlp']=load_definitions('timm_stablemil','timm/layers/mlp.py',['Mlp'],ns).Mlp
    ns['LayerScale']=load_definitions('timm_stablemil','timm/layers/layer_scale.py',['LayerScale'],ns).LayerScale
    rpe=load_definitions('stablemil','models/d2d_rope.py',['compute_axial_cis','apply_rotary_emb','rotary_','project_','random_project'],ns)
    ns.update({name:getattr(rpe,name) for name in ['compute_axial_cis','apply_rotary_emb','rotary_','project_','random_project']})
    patches=[('torch.zeros([B, self.num_heads , pile_k.shape[2],pile_k.shape[3] - self.k_neighbors - 1])',
        'pile_k.new_zeros([B,self.num_heads,pile_k.shape[2],pile_k.shape[3] - self.k_neighbors - 1])'),
        ('torch.zeros(pile_k.shape[:-1])','pile_k.new_zeros(pile_k.shape[:-1])')]
    if repaired:patches.append(('pile_k.shape[3] - self.k_neighbors - 1','pile_k.shape[3] - k_neighbors - 1'))
    attention=load_definitions('stablemil','models/attn_module.py',['flex_doucment_Attention','pile_attention'],ns,patches)
    ns.update(flex_doucment_Attention=attention.flex_doucment_Attention,pile_attention=attention.pile_attention)
    return load_definitions('stablemil','models/model.py',['initialize_weights','attn_block','sequentialV2','stableMIL'],ns,
        [('return hazards, S, Y_hat,','return logits, hazards, S, Y_hat')]).stableMIL


def author_nll():
    return load_definitions('stablemil','survival/utils/pro_utils.py',['nll_loss','NLLSurvLoss'],{'torch':torch}).NLLSurvLoss


class StableMIL(DiscreteSurvival):
    owns_sampling=True

    def __init__(self,bundle,profile='stablemil_classification_repaired',hidden=512,depth=2,heads=8,
                 aggregate_num=256,k_neighbors=8,ratio=2,reference_size=None,geometry_radius=NATIVE_RADIUS,
                 dropout=0.,drop_path=0.,learnable_mapping=False,accumulation_steps=4,bins=4,alpha=.15,
                 max_dense_bytes=4*1024**3,geometry_cache_bytes=128*1024**2,geometry_cache_dir=None):
        survival=bundle.task_type=='survival'
        if profile not in PROFILES or bundle.task_type not in {'classification','survival'} or ('survival' in profile)!=survival:
            raise ValueError('StableMIL requires its explicit classification or survival profile')
        if (not survival and bundle.config.get('training_unit','slide')!='slide') or (survival and (bundle.rows.groupby('global_patient_id').size()>1).any()):
            raise ValueError('StableMIL requires one independent slide frame per classification bag or survival patient; freeze a representative-slide rule instead of silently discarding slides')
        for name,value in [('hidden',hidden),('depth',depth),('heads',heads),('aggregate_num',aggregate_num),('k_neighbors',k_neighbors),
            ('ratio',ratio),('accumulation_steps',accumulation_steps),('max_dense_bytes',max_dense_bytes),('geometry_cache_bytes',geometry_cache_bytes)]:positive_int(value,name)
        if hidden%(4*heads) or type(learnable_mapping) is not bool:raise ValueError('StableMIL head dimension must be divisible by four and mapping mode boolean')
        for name,value in [('dropout',dropout),('drop_path',drop_path)]:
            if not 0<=finite(value,name)<1:raise ValueError('StableMIL dropout rates must be in [0,1)')
        super().__init__(bins=bins,alpha=alpha,accumulation_steps=accumulation_steps)
        self.bundle,self.dim,self.profile,self.survival=bundle,bundle.dim,profile,survival
        self.hidden,self.depth,self.heads=hidden,depth,heads
        self.aggregate_num,self.k_neighbors,self.ratio=aggregate_num,k_neighbors,ratio
        metadata=spatial_metadata(bundle)
        self.reference_size=finite(metadata['patch_size'] if reference_size is None else reference_size,'reference_size')
        self.radius=finite(geometry_radius,'geometry_radius')
        if min(self.reference_size,self.radius)<=0:raise ValueError('StableMIL physical reference and radius must be positive')
        self.max_dense_bytes,self.dropout,self.drop_path=max_dense_bytes,dropout,drop_path
        self.learnable_mapping=learnable_mapping;self.repaired=profile.endswith('repaired')
        self.reserve(1,1,1)
        self.network=author_model(self.repaired)(dim=bundle.dim,hidden_dim=hidden,depth=depth,num_heads=heads,
            aggregate_num=aggregate_num,k_neighbors=k_neighbors,ratio=ratio,n_classes=bins if survival else len(bundle.classes),
            task='survival' if survival else 'subtype',drop_rate=dropout,drop_path_rate=drop_path,learnable_mapping=learnable_mapping)
        self.nll=author_nll()(alpha=alpha)
        self.geometry=GeometryCache(geometry_cache_bytes);self.seen={}
        self.geometry_cache_dir=None if geometry_cache_dir is None else Path(geometry_cache_dir).expanduser().resolve()
        self.optimizer_steps,self.pending_count,self.epoch=0,0,-1
        self.stop_after_epoch=51

    def bind_output(self,output):
        self.geometry.directory=self.geometry_cache_dir or Path(output)/'stable_geometry'

    def reserve(self,n,regions,extent):
        params=self.dim*self.hidden*(self.ratio**2 if self.learnable_mapping else 1)+(12*self.depth+5)*self.hidden**2
        bytes_needed=20*params+16*self.dim*n*self.ratio**2+16*self.hidden*n*(min(n,self.k_neighbors+1)+regions)+32*self.heads*n*regions+32*self.heads*(extent+2)*2
        if bytes_needed>self.max_dense_bytes:raise ValueError('StableMIL attention/projection exceeds max_dense_bytes before allocation')

    def configure_training(self,cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric']!='stablemil_loss':
            raise ValueError('StableMIL retains its physical fusion, unclipped Adam and validation bag/patient loss')
        if not self.survival and not self.repaired and len(self.bundle.records('train'))<self.accumulation_steps:
            raise ValueError('StableMIL source classification TRAIN cannot form one accumulation update')

    def prepare(self,bundle,seed,device):
        if self.survival:super().prepare(bundle,seed,device)

    def make_optimizer(self,lr,weight_decay):
        if not self.survival and weight_decay!=0:raise ValueError('StableMIL source classification Adam has no weight decay')
        return torch.optim.Adam(self.network.parameters(),lr=lr,weight_decay=weight_decay)

    def start_epoch(self,epoch):self.epoch=epoch

    def structure(self,coords):
        arrays,meta=self.geometry.get(coords,reference_size=self.reference_size,ratio=self.ratio,aggregate_num=self.aggregate_num,
            k_neighbors=self.k_neighbors,radius=self.radius,repaired=self.repaired,max_bytes=self.max_dense_bytes)
        n=meta['fused_instances']
        if not self.repaired and n<=self.k_neighbors:raise ValueError('StableMIL source attention requires fused_instances > k_neighbors; named repair handles small bags')
        extent=int(np.ptp(arrays['fuse_coords'],axis=0).max())+2
        self.reserve(n,meta['semantic_regions'],extent)
        self.seen[meta['coordinate_sha256']]={k:v for k,v in meta.items() if k!='arrays'}
        return arrays,meta

    def forward(self,x,coords=None,label=None):
        if x.ndim!=2 or len(x)==0 or x.shape[1]!=self.dim or not x.is_floating_point() or not torch.isfinite(x).all() or coords is None or len(coords)!=len(x):
            raise ValueError('StableMIL requires finite [N,D] features and aligned real coordinates')
        arrays,meta=self.structure(coords)
        names=['fuse_coords','fuse_labels','fuse_sorted_idx','region_indices','region_sorted_index','attention_mask_1','attention_mask_2']
        values=[torch.from_numpy(arrays[name].copy()).to(x.device) for name in names]
        values[0]=values[0].unsqueeze(0)
        raw=self.network(x.unsqueeze(0),*values)
        if self.survival:
            logits,hazards,survival,_=raw
            result={'hazard_logits':logits,'hazards':hazards,'survival':survival,'logits':-survival.sum(1,keepdim=True)}
        else:result={'logits':raw[0]}
        if not all(torch.isfinite(v).all() for v in result.values()):raise ValueError('StableMIL produced nonfinite predictions')
        result['geometry']=meta
        return result

    def forward_unit(self,bundle,rows,device,config,train=False):
        if len(rows)!=1 or config['max_instances'] is not None:raise ValueError('StableMIL cannot concatenate independent slide coordinate frames or sample away patches')
        x,coords=load_spatial_inputs(bundle,rows[0]);return self(x.to(device),coords)

    def objective(self,result,time,event,alpha=None):
        y=self.time_label([time]).to(result['hazards'].device)
        c=result['hazards'].new_tensor([1-event])
        return self.nll(result['hazards'],result['survival'],y,c,alpha=alpha)

    def prediction_fields(self,result,row):
        if self.survival:loss=self.objective(result,row['time'],row['event'],alpha=0.)
        else:loss=F.cross_entropy(result['logits'],torch.tensor([row['target']],device=result['logits'].device))
        m=result['geometry']
        return {'stable_loss':float(loss.detach()),'stable_geometry_sha256':m['geometry_sha256'],
            'stable_raw_instances':m['raw_instances'],'stable_fused_instances':m['fused_instances'],
            'stable_retained_instances':m['retained_instances'],'stable_semantic_regions':m['semantic_regions']}

    def validation_metrics(self,frame):return {'stablemil_loss':float(frame.stable_loss.mean())}

    def selection_objective(self,metrics,epoch,objective):return -metrics['stablemil_loss']

    def selection_improved(self,value,best):return value>=best  # native EarlyStopping saves tied validation-loss checkpoints

    def get_extra_state(self):
        return {'optimizer_steps':self.optimizer_steps,'pending_count':self.pending_count,'epoch':self.epoch,
            'gradients':{name:p.grad.detach().clone() for name,p in self.network.named_parameters() if p.grad is not None}}

    def set_extra_state(self,state):
        if set(state)!={'optimizer_steps','pending_count','epoch','gradients'} or any(type(state[k]) is not int or state[k]<0 for k in ['optimizer_steps','pending_count']):
            raise ValueError('Invalid StableMIL checkpoint extra state')
        self.optimizer_steps,self.pending_count,self.epoch=state['optimizer_steps'],state['pending_count'],state['epoch']
        params=dict(self.network.named_parameters())
        if set(state['gradients'])-set(params):raise ValueError('Unknown StableMIL pending gradient')
        for name,p in params.items():
            value=state['gradients'].get(name)
            if value is not None and (value.shape!=p.shape or not torch.isfinite(value).all()):raise ValueError('Invalid StableMIL pending gradient tensor')
            p.grad=None if value is None else value.detach().to(p.device).clone()

    def protocol(self):
        return {'method':self.profile,'family':'stablemil','hidden':self.hidden,'depth':self.depth,'heads':self.heads,
            'reference_size':self.reference_size,'aggregate_num':self.aggregate_num,'ratio':self.ratio,'k_neighbors':self.k_neighbors,
            'geometry_radius':self.radius,'geometry_radius_source':'actual preprocessing default6sqrt2; CLI3sqrt2 is not the generated neighbour mask',
            'learnable_mapping':self.learnable_mapping,'dropout':self.dropout,'drop_path':self.drop_path,'accumulation_steps':self.accumulation_steps,
            'input':'real physical frozen-FM features; each survival patient has one registered representative slide, no silent first-slide selection',
            'objective':'author clamped discrete NLL' if self.survival else 'CrossEntropyLoss','alpha':self.alpha if self.survival else None,
            'survival_defaults':'explicit NLLSurvLoss constructor alpha.15/bins4 and zero weight_decay; missing launcher options are completed, not claimed as original paper settings' if self.survival else None,
            'bin_fit':'unique uncensored TRAIN patients, tails unbounded' if self.survival else None,'interior_edges':self.bin_edges.tolist() if self.survival else None,
            'optimizer':'Adam continuous state; no scheduler or clipping','optimizer_steps':self.optimizer_steps,'pending_samples':self.pending_count,
            'accumulation':'flush every partial window with actual count' if self.repaired else 'classification: retain unstepped tail gradients across epochs; survival: flush final window divided by nominal gc',
            'selection':'latest tied validation slide CE/patient NLL(alpha0), patience20 after epoch50; held-out test only explicit evaluation',
            'common_repairs':['device-owned masks and nn.Module.to instead of broken relocate','source variable-name and survival preprocessing/model/loss wiring',
                'coherent best/last checkpoints; no mutable best-model reference','TRAIN-only survival quantiles, no held-out label fitting'],
            'named_repairs':['small/degenerate physical inputs and neighbour mask width','tail loss weighting and classification flush'] if self.repaired else [],
            'random_position':'original random_project p.8 with 90-degree rotations/sorted random coordinates during TRAIN; axial RoPE otherwise',
            'mask':'native local neighbours plus all semantic tokens; semantic queries attend globally; no entropy penalty invented',
            'geometries':self.seen,'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}


def stable_epoch(model,bundle,groups,optimizer,device,cfg):
    if any(len(g)!=1 for g in groups):raise ValueError('StableMIL requires one slide coordinate frame per training unit')
    rows=[g[0] for g in groups]
    loader=DataLoader(rows,batch_size=1,sampler=RandomSampler(rows),num_workers=0,collate_fn=lambda batch:batch[0])
    if model.repaired or model.survival:
        optimizer.zero_grad(set_to_none=True);model.pending_count=0
    losses=[];total=len(loader)
    for index,row in enumerate(loader):
        result=model.forward_unit(bundle,[row],device,cfg,train=True)
        loss=model.objective(result,row['time'],row['event']) if model.survival else F.cross_entropy(result['logits'],torch.tensor([row['target']],device=device))
        divisor=min(model.accumulation_steps,total-(index//model.accumulation_steps)*model.accumulation_steps) if model.repaired else model.accumulation_steps
        (loss/divisor).backward();model.pending_count+=1
        step=(index+1)%model.accumulation_steps==0 or ((model.survival or model.repaired) and index+1==total)
        if step:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('StableMIL nonfinite gradient')
            check_update_finite(model,optimizer);optimizer.step();check_update_finite(model,optimizer)
            optimizer.zero_grad(set_to_none=True);model.optimizer_steps+=1;model.pending_count=0
        losses.append({'loss':float(loss.detach()),'divisor':divisor})
    if not losses:raise ValueError('StableMIL has no TRAIN units')
    return losses

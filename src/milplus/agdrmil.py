"""Complete fixed-source AGDR-MIL dual-branch classification and native recipe."""
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import digest_file, digest_object
from .upstream import load_definitions, verified_source

COMMIT='8e830683c0a695e1e1983f01d35dc85471bf327c'
PRESETS={'agdrmil_yaml_source':(.95,.5,.2,.3),'agdrmil_tcga_source':(.9,.4,.3,.4)}


def author_model(classes,dimension,dropout,high,low,mask,high_pool):
    _,sha=verified_source('agdrmil')
    if sha!=COMMIT:raise ValueError('AGDR source differs from reviewed commit')
    scope={'torch':torch,'nn':nn,'F':F}
    helper=load_definitions('agdrmil','utils/variate.py',['get_act'],scope)
    scope['get_act']=helper.get_act
    original=load_definitions('agdrmil','model.py',['initialize_weights','att_split',
        'CosineSimilarityWeighted','RowMask','mil_aggregation','AGDR_MIL'],scope,
        replacements=[('        masked_x = x[retained_indices]',
            '        self.last_retained_indices = retained_indices\n        masked_x = x[retained_indices]')])
    return original.AGDR_MIL(num_classes=classes,in_dim=dimension,dropout=dropout,
        h_thr=high,l_thr=low,mask=mask,mil_1=high_pool,mil_2='attention')


class AGDRMIL(nn.Module):
    owns_sampling=True
    supports_instance_outputs=True
    instance_score_semantics='Released hard grader is untrained: boolean partition blocks all direct loss gradients. Overall attention is learned after high/mid/retained-low concatenation; zero on dropped rows. Neither score is a causal attribution.'
    def __init__(self,bundle,profile='agdrmil_yaml_source',dropout=.25,high_pool='max',stop_epoch=20):
        super().__init__()
        if profile not in PRESETS or high_pool not in {'max','mean','attention','gate'}:
            raise ValueError('AGDR requires an explicit source preset and valid high-subset pool')
        if isinstance(dropout,bool) or not isinstance(dropout,(int,float)) or not math.isfinite(dropout) or not 0<=dropout<1 or type(stop_epoch) is not int or stop_epoch<0:
            raise ValueError('AGDR dropout and early-stop epoch invalid')
        if bundle.task_type!='classification' or bundle.config.get('training_unit','slide')!='slide' or len(bundle.classes)<2:
            raise ValueError('AGDR source uses one complete slide bag and at least two classes')
        self.bundle=bundle;self.profile=profile;self.high,self.low,self.mask,self.loss_weight=PRESETS[profile]
        self.high_pool=high_pool
        self.network=author_model(len(bundle.classes),bundle.dim,dropout,self.high,self.low,self.mask,high_pool)
        self.stop_after_epoch=stop_epoch+1;self.epochs=200;self.optimizer_steps=0;self.completed_epochs=0
        self.generator=torch.Generator().manual_seed(2000)
        self.contract=digest_object({'source':COMMIT,'profile':profile,'dimension':bundle.dim,'classes':bundle.classes,
            'dropout':dropout,'high_pool':high_pool,'stop_epoch':stop_epoch,'thresholds':[self.high,self.low],
            'mask':self.mask,'loss_weight':self.loss_weight,'sampler_seed':2000})
        self.network.att.attention.register_forward_hook(self._capture_grader)
        self.network.aggre_2.register_forward_hook(self._capture_attention)

    def _capture_grader(self,module,inputs,output):
        span=output.max()-output.min()
        if not torch.isfinite(output).all() or not torch.isfinite(span) or span<=0:
            raise ValueError('Original AGDR min-max grader is undefined on constant/nonfinite scores; no fabricated partition fallback')
        self.grades=((output-output.min())/span).detach().flatten()

    def _capture_attention(self,module,inputs,output):self.pool_attention=output[1].detach().flatten()

    def configure_training(self,cfg):
        if cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['selection_metric']!='agdr_val_loss':
            raise ValueError('AGDR source preserves complete bags, unclipped gradients and weighted dual-branch validation CE')
        self.epochs=cfg['epochs']

    def make_optimizer(self,lr,weight_decay):return torch.optim.Adam(self.network.parameters(),lr=lr,weight_decay=weight_decay)
    def forward(self,x,coords=None):
        if x.ndim!=2 or x.shape[1]!=self.bundle.dim or len(x)<2 or not torch.isfinite(x).all():
            raise ValueError('AGDR requires at least two finite input feature rows in its declared encoder space')
        if torch.equal(x,x[0].expand_as(x)):
            raise ValueError('AGDR min-max grading is undefined when every input feature row is identical')
        high,all_bag=self.network(x.unsqueeze(0))
        if not torch.isfinite(high).all() or not torch.isfinite(all_bag).all():raise ValueError('Nonfinite source AGDR prediction')
        index=torch.arange(len(x),device=x.device)
        hi=index[self.grades>=self.high];mid=index[(self.grades<self.high)&(self.grades>=self.low)]
        low=index[self.grades<self.low];retained=low[self.network.mask.last_retained_indices.to(x.device)]
        order=torch.cat([hi,mid,retained]);attention=torch.zeros(len(x),device=x.device)
        attention[order]=self.pool_attention
        return {'logits':all_bag,'high_logits':high,'probabilities':all_bag.softmax(1),
            'grades':self.grades,'attention':attention,'retained_order':order,
            'partition':torch.where(self.grades>=self.high,2,torch.where(self.grades>=self.low,1,0)),
            'coords':coords}

    def forward_unit(self,bundle,rows,device,config,train=False):
        if len(rows)!=1:raise ValueError('AGDR source does not merge separate WSI bags')
        x,coords=bundle.load(rows[0]);return self(x.to(device),None if coords is None else coords.to(device))

    def objective(self,result,target):
        return (self.loss_weight*F.cross_entropy(result['high_logits'],target)+(1-self.loss_weight)*F.cross_entropy(result['logits'],target)).mean()

    def prediction_fields(self,result,row):
        fields={'agdr_source_loss':float(self.objective(result,torch.tensor([row['target']],device=result['logits'].device)))}
        for name,key in [('high','high_logits'),('full','logits')]:
            fields.update({f'agdr_{name}_sigmoid_{i}':float(value) for i,value in enumerate(result[key][0])})
        return fields
    def start_prediction(self,subset):
        # The native non-shuffled validation/test DataLoader still draws a base
        # seed from the global CPU generator; its subsequent RowMask uses it.
        torch.empty((),dtype=torch.int64).random_()
    def validation_metrics(self,frame):
        p=frame[[f'p_{i}' for i in range(len(self.bundle.classes))]].to_numpy()
        prediction=(p[:,1]>.5).astype(int) if len(self.bundle.classes)==2 else p.argmax(1)
        try:
            auc=float(roc_auc_score(frame.target,p[:,1]) if len(self.bundle.classes)==2 else
                roc_auc_score(frame.target,p,multi_class='ovr',average='weighted'))
            if not np.isfinite(auc):auc=None
        except ValueError:auc=None
        return {'agdr_val_loss':sum(map(float,frame.agdr_source_loss))/len(frame),
            'agdr_native_slide_accuracy':float((prediction==frame.target).mean()),
            'agdr_native_slide_auc':auc}
    def selection_objective(self,metrics,epoch,current):return -metrics['agdr_val_loss']
    def selection_improved(self,objective,best):return objective>=best

    def epoch(self,bundle,optimizer,device='cpu'):
        self.train();rows=bundle.records('train')
        loader=DataLoader(rows,batch_size=1,shuffle=True,generator=self.generator,collate_fn=lambda batch:batch[0])
        losses=[]
        for row in loader:
            optimizer.zero_grad();result=self.forward_unit(bundle,[row],device,{},train=True)
            loss=self.objective(result,torch.tensor([row['target']],device=device));loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise ValueError('Nonfinite source AGDR objective or gradient')
            optimizer.step();self.optimizer_steps+=1;losses.append({'loss':float(loss.detach())})
        return losses
    def finish_epoch(self,bundle,device,cfg):self.completed_epochs+=1
    def get_extra_state(self):
        return {'contract':self.contract,'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,
            'sampler_generator':self.generator.get_state()}
    def set_extra_state(self,state):
        if set(state)!={'contract','optimizer_steps','completed_epochs','sampler_generator'} or state['contract']!=self.contract or any(
            type(state[k]) is not int or state[k]<0 for k in ['optimizer_steps','completed_epochs']):
            raise ValueError('AGDR source/profile/sampler/training contract differs')
        generator_state=state['sampler_generator']
        if not torch.is_tensor(generator_state) or generator_state.dtype!=torch.uint8 or generator_state.shape!=self.generator.get_state().shape:
            raise ValueError('AGDR private sampler generator state invalid')
        candidate=torch.Generator()
        try:candidate.set_state(generator_state.cpu())
        except RuntimeError as error:raise ValueError('AGDR private sampler generator state invalid') from error
        self.generator=candidate
        self.optimizer_steps=state['optimizer_steps'];self.completed_epochs=state['completed_epochs']

    def export_instances(self,result,row,directory):
        if self.training:raise ValueError('AGDR instance export requires evaluation mode')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        path=directory/(digest_object(row['slide_id'])[:24]+'.h5')
        with h5py.File(path,'w') as h:
            for key in ['grades','attention','retained_order','partition']:
                h[key]=result[key].cpu().numpy()
            if result['coords'] is not None:h['coords']=result['coords'].cpu().numpy()
            h.attrs['semantics']=self.instance_score_semantics;h.attrs['profile']=self.profile
        return {'slide_id':row['slide_id'],'path':str(path),'sha256':digest_file(path),'instances':len(result['grades'])}

    def protocol(self):
        return {'profile':self.profile,'source_commit':COMMIT,'epochs':self.epochs,
            'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,
            'architecture':'actual512-wide source reduction, hard attention grading, high-guided cosine redistribution, random low-subset mask, dual classification branches',
            'high_subset_pool':self.high_pool,
            'objective':'released Sigmoid heads before weighted high/full CE; native Adam2e-4/weight_decay1e-5, no scheduler/clip, batch1 DataLoader private generator seed2000',
            'selection':'last tied best weighted dual-branch validation CE, source early-stop zero-based epoch>20 by default',
            'source_gradient_boundary':'hard boolean grader receives no direct gradient; no invented auxiliary grader objective',
            'evaluation_mask':'original RowMask remains stochastic even in eval; native sequential validation/test DataLoader base-seed draw is retained, unified evaluation seed and row order replay it. Checkpoint retains global RNG.',
            'metrics':'source slide accuracy and binary or weighted multiclass OVR AUC retained separately from standardized patient/slide macro reports. Undefined AUC is null, not a substituted zero.',
            'source_preset_boundary':'released YAML C16 values include high_pool=max despite README attention recommendation; TCGA preset uses explicit README values',
            'degenerate_input':'undefined constant min-max grader explicitly rejected; no surrogate membership or source gradient correction',
            'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}

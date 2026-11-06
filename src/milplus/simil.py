"""Complete pinned SI-MIL dual branches and coupled optimizer recipe."""
import copy
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file, digest_object
from .simil_data import PathExpertAssets
from .simil_normalization import PathExpertNormalization
from .simil_source import load_model, patch_dropout


class SIMIL(nn.Module):
    owns_sampling=True
    supports_instance_outputs=True
    instance_score_semantics='Conditional additive PathExpert terms and a bias per selected patch; not causal effects'

    def __init__(self,bundle,profile='simil_full_vjp',top_k=20,stop_gradient=False,temperature=3.,percentile=.75,
                 dropout_patch=.2,normalization_workspace_bytes=2*1024**3):
        super().__init__()
        if bundle.task_type!='classification' or len(bundle.classes)!=2 or bundle.config.get('training_unit','slide')!='slide':
            raise ValueError('Native SI-MIL requires binary individual slide bags')
        variants={'simil_source':'source','simil_full_vjp':'full_vjp'}
        if profile not in variants or type(top_k) is not int or top_k<2 or type(stop_gradient) is not bool:
            raise ValueError('SI-MIL requires a named gradient policy and native K>=2')
        variant=variants[profile]
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in [temperature,percentile]) or
                not math.isfinite(temperature) or temperature<=0 or not math.isfinite(percentile) or not 0<=percentile<=1):
            raise ValueError('SI-MIL feature gate settings invalid')
        if not isinstance(dropout_patch,(float,int)) or isinstance(dropout_patch,bool) or not math.isfinite(dropout_patch) or not 0<=dropout_patch<.5:
            raise ValueError('Source patch resampling requires p in [0,.5)')
        if type(normalization_workspace_bytes) is not int or normalization_workspace_bytes<1:
            raise ValueError('SI-MIL normalization workspace must be a positive byte budget')
        self.assets=PathExpertAssets(bundle)
        self.bundle=bundle;self.profile=profile;self.variant=variant;self.top_k=top_k
        source=load_model(variant)
        self.network=source.MILNet(source.BClassifier(self.assets.dimension,bundle.dim,2,
            stop_gradient='yes' if stop_gradient else 'no',top_k=top_k,temperature=temperature,percentile=percentile))
        self.normalization=None;self.prepared=False;self.order=[]
        self.optimizer_steps=0;self.completed_epochs=0
        self.epochs=39;self.dropout_patch=dropout_patch
        self.normalization_workspace_bytes=normalization_workspace_bytes;self.fit_plan=None
        self.scheduler=None;self.saved_scheduler=None
        self.trace=None
        self.contract=digest_object({'manifest':self.assets.contract,'variant':variant,'top_k':top_k,
            'stop_gradient':stop_gradient,'temperature':temperature,'percentile':percentile,'input':bundle.dim,
            'dropout_patch':dropout_patch,'normalization_workspace_bytes':normalization_workspace_bytes})

    def configure_training(self,cfg):
        if cfg.get('max_instances') is not None or cfg.get('grad_clip') is not None:
            raise ValueError('SI-MIL native recipe uses coupled patch resampling and unclipped gradients')
        self.epochs=cfg['epochs']

    def prepare(self,bundle,seed,device):
        self.normalization=self.assets.fit_training_normalization(self.normalization_workspace_bytes)
        self.fit_plan=copy.deepcopy(self.assets.fit_plan)
        self.order=[r['slide_id'] for r in bundle.records('train')]
        self.prepared=True

    def make_optimizer(self,lr,weight_decay):
        opt=torch.optim.Adam(self.network.parameters(),lr=lr,betas=(.5,.9),weight_decay=weight_decay)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,self.epochs+1,eta_min=.000005)
        if self.saved_scheduler is not None:self.scheduler.load_state_dict(self.saved_scheduler)
        return opt

    def forward_unit(self,bundle,rows,device,config,train=False):
        if not self.prepared or len(rows)!=1:
            raise ValueError('SI-MIL requires prepared PathExpert normalization and one aligned slide bag')
        x,raw,coords=self.assets.load(rows[0])
        aux=self.normalization.transform(raw)
        deep=x.numpy()
        if train:
            aux,deep=patch_dropout()(aux,deep,self.dropout_patch)
            if not len(aux):raise ValueError('SI-MIL source patch resampling produced an empty bag')
        result=self(torch.as_tensor(aux,dtype=torch.float32,device=device),
                    torch.as_tensor(deep,dtype=torch.float32,device=device),train=train)
        if not train:self.trace={'normalized':aux,'coords':coords}
        return result

    def forward(self,aux,x,train=None):
        train=self.training if train is None else train
        if (aux.ndim!=2 or aux.shape[1]!=self.assets.dimension or x.shape!=(len(aux),self.bundle.dim)
                or not len(aux) or not torch.isfinite(aux).all() or not torch.isfinite(x).all()):
            raise ValueError('SI-MIL requires finite, nonempty and row-aligned normalized descriptor/deep tensors')
        values=self.network(aux,x,training='yes' if train else 'no')
        c,deep,a,b,bdeep,gate=values
        if not all(torch.isfinite(v).all() for v in values):
            raise ValueError('Nonfinite source SI-MIL output')
        z=b.sum(0)
        return {'logits':torch.stack([torch.zeros_like(z),z],dim=1),
                'probabilities':torch.stack([1-c,c],dim=1),'main':c,'deep':deep,
                'attention':a,'selected_prediction_terms':b,'deep_prediction_terms':bdeep,'feature_gate':gate}

    def objective(self,result,target):
        y=target.to(result['main']).reshape(1)
        classification=F.binary_cross_entropy(result['main'],y)+F.binary_cross_entropy(result['deep'],y)
        distillation=20*F.mse_loss(result['main'],result['deep'].clone().detach())
        regularization=.05*torch.norm(result['attention'],p=2)
        return {'loss':classification+distillation+regularization,'classification':classification,
                'distillation':distillation,'attention_regularization':regularization}

    def epoch(self,bundle,optimizer,device='cpu'):
        self.train();rows={r['slide_id']:r for r in bundle.records('train')}
        if set(self.order)!=set(rows) or len(self.order)!=len(rows):
            raise ValueError('SI-MIL persistent training order differs from the fixed split')
        np.random.shuffle(self.order)
        losses=[]
        for sid in self.order:
            row=rows[sid];optimizer.zero_grad()
            result=self.forward_unit(bundle,[row],device,{},True)
            terms=self.objective(result,torch.tensor([row['target']],device=device))
            if not torch.isfinite(terms['loss']):raise ValueError('Nonfinite SI-MIL loss')
            terms['loss'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise ValueError('Nonfinite SI-MIL gradients')
            optimizer.step();self.optimizer_steps+=1
            losses.append({k:float(v.detach()) for k,v in terms.items()})
        return losses

    def finish_epoch(self,bundle,device,cfg):
        self.scheduler.step();self.completed_epochs+=1

    def validation_metrics(self,frame):
        from sklearn.metrics import accuracy_score, roc_auc_score
        return {'simil_joint':.5*(float(accuracy_score(frame.target,np.asarray(frame.p_1)>.5))+
                                  float(roc_auc_score(frame.target,frame.p_1)))}

    def selection_improved(self,objective,best):
        return objective>=best

    def prediction_fields(self,result,row):
        return {'deep_probability':float(result['deep'].item())}

    def get_extra_state(self):
        return {'contract':self.contract,'prepared':self.prepared,
                'normalization':None if self.normalization is None else copy.deepcopy(self.normalization.state()),
                'order':list(self.order),'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,
                'fit_plan':copy.deepcopy(self.fit_plan),
                'scheduler':copy.deepcopy(self.scheduler.state_dict() if self.scheduler is not None else self.saved_scheduler)}

    def set_extra_state(self,state):
        required={'contract','prepared','normalization','order','optimizer_steps','completed_epochs','fit_plan','scheduler'}
        if set(state)!=required or state['contract']!=self.contract or type(state['prepared']) is not bool or any(
                type(state[k]) is not int or state[k]<0 for k in ['optimizer_steps','completed_epochs']):
            raise ValueError('SI-MIL checkpoint descriptor/input/gradient contract differs')
        if state['prepared'] and (state['normalization'] is None or len(state['normalization']['edges'])!=self.assets.dimension or
                len(state['order'])!=len(set(state['order'])) or set(state['order'])!={r['slide_id'] for r in self.bundle.records('train')}):
            raise ValueError('SI-MIL checkpoint normalization or training order differs')
        self.prepared=state['prepared'];self.order=list(state['order'])
        self.normalization=None if state['normalization'] is None else PathExpertNormalization.restore(copy.deepcopy(state['normalization']))
        self.optimizer_steps=state['optimizer_steps'];self.completed_epochs=state['completed_epochs']
        self.fit_plan=copy.deepcopy(state['fit_plan'])
        self.saved_scheduler=copy.deepcopy(state['scheduler'])
        if self.scheduler is not None:self.scheduler.load_state_dict(self.saved_scheduler)

    def export_instances(self,result,row,directory):
        if self.trace is None or self.training:
            raise ValueError('SI-MIL instance explanations require deterministic inference')
        n=len(self.trace['normalized'])
        gate_indices=torch.sort(result['attention'].reshape(-1),descending=True).indices
        if n<self.top_k:gate_indices=gate_indices.repeat(self.top_k//n+1)
        gate_indices=gate_indices[:self.top_k]
        selected=gate_indices if n>self.top_k else torch.arange(n,device=result['attention'].device)
        hcf=torch.as_tensor(self.trace['normalized'],dtype=torch.float32,device=selected.device)[selected]
        head=self.network.b_classifier.classification_head[0]
        contribution=hcf*result['feature_gate']*head.weight
        bias=head.bias.reshape(1)
        reconstructed=(contribution.sum()+len(selected)*bias).sigmoid()
        torch.testing.assert_close(reconstructed,result['main'],rtol=1e-5,atol=1e-6)
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        path=directory/(digest_object(row['slide_id'])[:24]+'.h5')
        with h5py.File(path,'w') as h:
            h['coords']=self.trace['coords'];h['attention']=result['attention'].cpu().numpy()
            h['prediction_indices']=selected.cpu().numpy();h['gate_input_indices']=gate_indices.cpu().numpy()
            h['feature_gate']=result['feature_gate'].cpu().numpy();h['feature_contributions']=contribution.cpu().numpy()
            h['per_instance_bias']=bias.cpu().numpy()
            h.attrs['feature_names']=json.dumps(self.assets.names)
            h.attrs['semantics']='Conditional additive PathExpert feature terms plus a bias per selected instance; not causal effects'
        return {'slide_id':row['slide_id'],'path':str(path),'sha256':digest_file(path),
                'instances':n,'selected_instances':len(selected),'prediction':float(result['main'].item())}

    def protocol(self):
        return {'profile':self.profile,'source_commit':'3ffb218672ffe0e4a183924af50e5920b498483d',
            'input_contract':self.assets.contract,'feature_names':self.assets.names,'top_k':self.top_k,
            'gradient_policy':self.variant,'dropout_patch':self.dropout_patch,
            'normalization':self.fit_plan,'actual_training_epochs':self.epochs,
            'optimizer':'Adam betas(.5,.9), explicit configured weight decay; source shell .005/.01 vs parser .05',
            'scheduler':'CosineAnnealingLR T_max=actual epochs+1, eta_min=5e-6; source range(1,num_epochs)',
            'objective':'BCE(SI,y)+BCE(deep,y)+20*MSE(SI,stopgrad(deep))+.05*L2(patch attention)',
            'selection':'VAL slide mean(AUROC, accuracy at p>.5), last tied best; TEST sealed until evaluation',
            'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,
            'repairs':['TRAIN-only PathExpert qcut and population statistics; original preprocessing includes VAL',
                       'exact optimizer/scheduler/persistent order/all RNG restore',
                       'first valid checkpoint retained at zero metric'],
            'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}


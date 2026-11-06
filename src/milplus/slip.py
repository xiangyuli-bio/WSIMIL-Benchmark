"""Complete SLIP prompt-only SGD/10epochs/final-epoch source-repaired recipe."""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .focus import SlideRows, one_row
from .io import digest_object
from .slip_source import ASSETS, COMMIT, SLIPReference


class SLIPMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile,context_size=1,context_gain=.01,temperature=.01):
        super().__init__()
        expected={'slip_clip_vit_b16_source_repaired':'clip_vit_b16',
                  'slip_clip_rn50_source_repaired':'clip_rn50'}
        if (profile not in expected or bundle.slip is None
                or bundle.slip.architecture!=expected[profile]):
            raise ValueError('SLIP profile requires its matching native CLIP frozen features and ordered text manifest')
        self.bundle,self.inputs,self.profile=bundle,bundle.slip,profile
        meta=self.inputs.manifest
        self.core=SLIPReference(self.inputs.architecture,meta['templates'],meta['slide_classnames'],
            meta['tissue_classnames'],context_size=context_size,context_gain=context_gain,temperature=temperature)
        self.settings={'context_size':context_size,'context_gain':context_gain,'temperature':temperature}
        self.contract=digest_object({'source_commit':COMMIT,'inputs':self.inputs.contract,
            'profile':profile,'settings':self.settings,'bundle':bundle.fingerprint})
        self.epoch_index,self.completed_epochs,self.optimizer_updates=-1,0,0
        self.train_orders=[]

    def configure_training(self,cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['weight_decay']!=0 or cfg['selection_metric']!='slip_final_epoch'):
            raise ValueError('SLIP requires complete bags, source SGD with zero decay/no clipping and final-epoch selection')

    def make_optimizer(self,lr,weight_decay):
        if weight_decay!=0:
            raise ValueError('SLIP source SGD has zero weight decay')
        self._configured_lr=float(lr)
        self._optimizer=torch.optim.SGD(self.core.prompt_learner.parameters(),lr=lr)
        return self._optimizer

    def validate_optimizer_state(self):
        optimizer=self._optimizer
        if type(optimizer) is not torch.optim.SGD or len(optimizer.param_groups)!=1 or optimizer.state:
            raise ValueError('SLIP checkpoint requires the original prompt-only momentum-free SGD state')
        group=optimizer.param_groups[0]
        if (group['lr']!=self._configured_lr or group['weight_decay']!=0 or group['momentum']!=0
                or group['dampening']!=0 or group['nesterov'] or group.get('maximize',False)
                or [id(p) for p in group['params']]!=[id(self.core.prompt_learner.ctx)]):
            raise ValueError('SLIP checkpoint changed the original prompt-only SGD configuration')

    def _apply(self,fn,recurse=True):
        if hasattr(self,'_optimizer'):
            raise ValueError('Move SLIP to its target device before constructing the prompt optimizer')
        return super()._apply(fn,recurse)

    def start_epoch(self,epoch):
        self.inputs.check_unchanged();self.epoch_index=epoch

    def epoch(self,bundle,optimizer,device):
        dataset=SlideRows(bundle.records('train'))
        sampler=(SequentialSampler(dataset) if self.inputs.manifest['n_shots'] in ['1','2'] else RandomSampler(dataset))
        loader=DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=0,collate_fn=one_row)
        order,losses=[],[]
        for row in loader:
            self.inputs.check_unchanged()
            features,_=bundle.load(row)
            loss,_=self.core.loss(features.to(device),row['target'])
            optimizer.zero_grad();loss.backward();optimizer.step()
            self.optimizer_updates+=1;order.append(row['slide_id']);losses.append({'loss':float(loss.detach())})
        self.train_orders.append(order);self.completed_epochs=self.epoch_index+1
        return losses

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        self.eval();self.inputs.check_unchanged()
        loader=DataLoader(SlideRows(bundle.records(subset)),batch_size=1,num_workers=0,collate_fn=one_row)
        records=[]
        for row in loader:
            features,_=bundle.load(row)
            logits,_=self.core(features.to(device))
            probs=logits.softmax(0)
            result={k:row[k] for k in ['slide_id','patient_id','global_patient_id','subset','target']}
            result['source_slides']=row['slide_id']
            result.update({f'p_{i}':float(value) for i,value in enumerate(probs)})
            records.append(result)
        return pd.DataFrame(records)

    def validation_metrics(self,frame):
        return {'slip_final_epoch':float(self.completed_epochs)}

    def selection_objective(self,metrics,epoch,current):
        return float(epoch+1)

    def selection_improved(self,objective,best):
        return True

    def get_extra_state(self):
        return {'contract':self.contract,'epoch_index':self.epoch_index,
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'train_orders':copy.deepcopy(self.train_orders)}

    def set_extra_state(self,state):
        if (not isinstance(state,dict) or set(state)!={'contract','epoch_index','completed_epochs','optimizer_updates','train_orders'}
                or state['contract']!=self.contract
                or any(type(state[k]) is not int for k in ['epoch_index','completed_epochs','optimizer_updates'])
                or state['completed_epochs']<0 or state['epoch_index']!=state['completed_epochs']-1
                or not isinstance(state['train_orders'],list) or len(state['train_orders'])!=state['completed_epochs']):
            raise ValueError('SLIP checkpoint source/inputs/prompt/epoch contract differs')
        expected=[r['slide_id'] for r in self.bundle.records('train')]
        if (any(not isinstance(order,list) or len(order)!=len(expected) or set(order)!=set(expected) for order in state['train_orders'])
                or state['optimizer_updates']!=len(expected)*state['completed_epochs']):
            raise ValueError('SLIP checkpoint does not contain complete TRAIN-only sampler/update history')
        if self.inputs.manifest['n_shots'] in ['1','2'] and any(order!=expected for order in state['train_orders']):
            raise ValueError('SLIP one/two-shot source sampler must be sequential')
        self.epoch_index,self.completed_epochs,self.optimizer_updates=state['epoch_index'],state['completed_epochs'],state['optimizer_updates']
        self.train_orders=copy.deepcopy(state['train_orders'])

    def _foundation_names(self):
        ids={id(p) for p in self.core.native.parameters()}
        return {n for n,p in self.named_parameters(remove_duplicate=False) if id(p) in ids}

    def state_dict(self,*args,**kwargs):
        state=super().state_dict(*args,**kwargs)
        for name in self._foundation_names():state.pop(name,None)
        return state

    def load_state_dict(self,state_dict,strict=True,assign=False):
        expected=self.state_dict()
        if not strict or assign or not isinstance(state_dict,dict) or set(state_dict)!=set(expected):
            raise ValueError('SLIP checkpoint requires exact nonfoundation state and fixed original asset reconstruction')
        learned={'core.prompt_learner.ctx','_extra_state'}
        context=state_dict['core.prompt_learner.ctx']
        if (not torch.is_tensor(context) or context.shape!=expected['core.prompt_learner.ctx'].shape
                or context.dtype!=expected['core.prompt_learner.ctx'].dtype or not torch.isfinite(context).all()):
            raise ValueError('SLIP checkpoint learned prompt shape/dtype/finite values differ')
        for name,value in expected.items():
            if name not in learned and (not torch.is_tensor(state_dict[name]) or not torch.equal(value.cpu(),state_dict[name].cpu())):
                raise ValueError('SLIP checkpoint fixed prompt/token/tissue buffers differ: '+name)
        complete=super().state_dict();complete.update(state_dict)
        return super().load_state_dict(complete,strict=True,assign=False)

    def protocol(self):
        return {**self.core.protocol(),'profile':self.profile,'inputs_binding':self.inputs.contract,
            'settings':self.settings,'n_shots':self.inputs.manifest['n_shots'],
            'recipe':'Original prompt-only SGD lr0.002/no momentum/no decay/no scheduler, B1, full bags, default10epochs; no early stopping',
            'selection':'Final epoch always; VAL diagnostics never choose a preferred earlier epoch, TEST only after the completed run',
            'sampling':'Original sequential one/two-shot or shuffled remaining modes, workers0; caller supplies explicit TRAIN-only few-shot split',
            'checkpoint':'Actual learned prompt, fixed prompt/tissue buffers, SGD, full order/counter/RNG; frozen native CLIP reconstructed only from exact verified original checkpoint',
            'public_source_gaps':['datasets/__init__.py imports absent datasets/tcga.py; unified fixed-patient loader is an explicit reconstruction',
                'Public split text files are not shipped; few-shot subset must be prepared only from TRAIN with provenance'],
            'evaluation_ports':['VAL diagnostics added without changing final-epoch selection',
                'VAL DataLoader consumes the unified run RNG stream; no paper seed identity claim',
                'Multiclass and patient reporting use common benchmark evaluator; source metrics are binary-specific'],
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'}

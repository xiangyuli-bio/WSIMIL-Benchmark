"""Complete released NcIEMIL aggregation over task-trained, provenance-bound bags."""
import copy
import math
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,RandomSampler,SequentialSampler,Sampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc,confusion_matrix
from sklearn.preprocessing import label_binarize

from .io import digest_object
from .nciemil_source import COMMIT,NcIEAggregator
from .upstream import load_definitions

VARIANTS={'nciemil_full_reference_released_kv':'released_interleaved_kv',
          'nciemil_full_reference_independent_kv_repaired':'paper_independent_kv_repaired'}


def source_loop():
    return load_definitions('nciemil','main.py',['train_one_epoch','val_one_epoch','calculate_metrics'],
        {'torch':torch,'nn':nn,'F':torch.nn.functional,'sys':sys,'tqdm':tqdm,'np':np,
         'accuracy_score':accuracy_score,'f1_score':f1_score,'roc_curve':roc_curve,
         'auc':auc,'confusion_matrix':confusion_matrix,'label_binarize':label_binarize})


class Bags(Dataset):
    def __init__(self,owner,subset):self.owner=owner;self.rows=owner.bundle.records(subset)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        self.owner.inputs.check();row=self.rows[index];x,_=self.owner.bundle.load(row)
        if x.shape!=(2*self.owner.inputs.k,768):raise ValueError('NcIEMIL complete native selected bag changed')
        return x,row['target']


class Tracked(Sampler):
    def __init__(self,sampler):self.sampler=sampler;self.indices=[]
    def __iter__(self):
        self.indices=[]
        for i in self.sampler:self.indices.append(i);yield i
    def __len__(self):return len(self.sampler)


class NcIEMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True
    def __init__(self,bundle,profile='nciemil_full_reference_released_kv',num_workers=4):
        super().__init__()
        if profile not in VARIANTS or getattr(bundle,'nciemil',None) is None or type(num_workers) is not int or num_workers<0:
            raise ValueError('NcIEMIL needs complete task-trained raw-image filtering/selection cache and a declared KV condition')
        self.bundle,self.inputs,self.profile=bundle,bundle.nciemil,profile
        n=len(bundle.records('train'))
        if n<2 or n%4==1:raise ValueError('Original NcIEMIL batch4 BatchNorm cannot train a singleton tail; no dropped or duplicated patients')
        self.num_workers=num_workers
        self.core=NcIEAggregator(768,len(bundle.classes),self.inputs.k,VARIANTS[profile],.2)
        self.loop=source_loop();self.criterion=nn.CrossEntropyLoss()
        self.contract=digest_object({'source':COMMIT,'bundle':bundle.fingerprint,'cache':self.inputs.contract,
            'profile':profile,'readers':num_workers})
        self.recipe=None;self.completed_epochs=0;self.epoch_index=-1;self.optimizer_updates=0
        self.train_orders=[];self._optimizer=None
        self.gradient_updates={k:0 for k,_ in self.core.network.named_parameters()}

    def configure_training(self,cfg):
        if (cfg['lr']!=2e-4 or cfg['weight_decay']!=1e-5 or cfg['max_instances'] is not None
                or cfg['grad_clip'] is not None or cfg['selection_metric']!='nciemil_val_harmonic'
                or type(cfg['epochs']) is not int or cfg['epochs']<1):
            raise ValueError('NcIEMIL requires actual source constant SGD2e-4/wd1e-5, complete selected bags and validation harmonic selector')
        self.recipe={k:cfg[k] for k in ['epochs','lr','weight_decay','threads','selection_metric']}

    def make_optimizer(self,lr,weight_decay):
        if self.recipe is None or lr!=2e-4 or weight_decay!=1e-5:raise ValueError('Configure the complete NcIEMIL recipe first')
        self._optimizer=torch.optim.SGD(self.core.network.parameters(),lr=lr,weight_decay=weight_decay)
        def count(optimizer,args,kwargs):
            self.optimizer_updates+=1
            for name,p in self.core.network.named_parameters():self.gradient_updates[name]+=int(p.grad is not None)
        self._optimizer.register_step_pre_hook(count)
        return self._optimizer

    def validate_optimizer_state(self):
        groups=self._optimizer.param_groups
        if (len(groups)!=1 or groups[0]['lr']!=2e-4 or groups[0]['weight_decay']!=1e-5
                or groups[0]['momentum']!=0 or groups[0]['dampening']!=0 or groups[0]['nesterov']
                or self._optimizer.state or {id(p) for p in groups[0]['params']}!={id(p) for p in self.core.network.parameters()}):
            raise ValueError('NcIEMIL actual released no-momentum SGD state or parameter ownership changed')

    def loader(self,subset):
        ds=Bags(self,subset);sampler=Tracked(RandomSampler(ds) if subset=='train' else SequentialSampler(ds))
        return ds,sampler,DataLoader(ds,batch_size=4,sampler=sampler,num_workers=self.num_workers,drop_last=False,pin_memory=True)

    def start_epoch(self,epoch):
        if type(epoch) is not int or epoch!=self.completed_epochs:raise ValueError('NcIEMIL must restore consecutive complete epochs')
        self.epoch_index=epoch

    def epoch(self,bundle,optimizer,device):
        if bundle is not self.bundle or optimizer is not self._optimizer:raise ValueError('NcIEMIL bound inputs/optimizer changed')
        self.inputs.check();ds,sampler,loader=self.loader('train');before=self.optimizer_updates
        loss=self.loop.train_one_epoch(self.core.network,loader,self.criterion,optimizer,device,self.epoch_index+1)
        if (not math.isfinite(loss) or self.optimizer_updates-before!=math.ceil(len(ds)/4)
                or sorted(sampler.indices)!=list(range(len(ds)))):
            raise ValueError('NcIEMIL original complete shuffled TRAIN bags/SGD updates missing')
        self.completed_epochs+=1;self.train_orders.append([ds.rows[i]['slide_id'] for i in sampler.indices])
        self.validate_optimizer_state();return [{'loss':loss}]

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        if bundle is not self.bundle or subset not in {'val','test'}:raise ValueError('NcIEMIL prediction must use registered held-out bags')
        ds,sampler,loader=self.loader(subset)
        logits,targets=self.loop.val_one_epoch(self.core.network,loader,device)
        assert sampler.indices==list(range(len(ds)))
        rows=[]
        for row,score,target in zip(ds.rows,logits,targets):
            if int(target)!=row['target']:raise ValueError('NcIEMIL held-out target/order changed')
            rec={k:row[k] for k in ['slide_id','patient_id','global_patient_id','target']}
            rec.update(subset=subset,source_slides=row['slide_id'],prediction=int(score.argmax()))
            rec.update({f'p_{i}':float(p) for i,p in enumerate(score.softmax(0))})
            rec.update({f'nciemil_logit_{i}':float(p) for i,p in enumerate(score)})
            rows.append(rec)
        return pd.DataFrame(rows)

    def validation_metrics(self,frame):
        logits=torch.tensor(frame[[f'nciemil_logit_{i}' for i in range(len(self.bundle.classes))]].to_numpy(),dtype=torch.float32)
        labels=torch.tensor(frame.target.to_numpy(),dtype=torch.long)
        acc,f1,area=self.loop.calculate_metrics(logits,labels,len(self.bundle.classes))
        # Continuous finite-zero extension of the source reciprocal harmonic
        # expression; zero components otherwise crash the original runner.
        harmonic=0. if min(acc,f1,area)==0 else 3/(1/acc+1/area+1/f1)
        return {'nciemil_val_harmonic':harmonic,'nciemil_source_accuracy':acc,
            'nciemil_source_macro_f1':f1,'nciemil_source_auc':area}

    def selection_improved(self,objective,best):return objective>=best

    def get_extra_state(self):
        return copy.deepcopy({'contract':self.contract,'recipe':self.recipe,'completed_epochs':self.completed_epochs,
            'epoch_index':self.epoch_index,'optimizer_updates':self.optimizer_updates,'train_orders':self.train_orders,
            'gradient_updates':self.gradient_updates})

    def set_extra_state(self,state):
        if (set(state)!=set(self.get_extra_state()) or state['contract']!=self.contract
                or (self.recipe is not None and state['recipe']!=self.recipe)):
            raise ValueError('NcIEMIL source/cache/recipe checkpoint identity changed')
        epochs=state['completed_epochs'];rows=self.bundle.records('train');ids={r['slide_id'] for r in rows}
        if (type(epochs) is not int or epochs<0 or epochs>state['recipe']['epochs']
                or state['epoch_index']!=epochs-1 or len(state['train_orders'])!=epochs
                or state['optimizer_updates']!=epochs*math.ceil(len(rows)/4)
                or any(len(order)!=len(rows) or set(order)!=ids for order in state['train_orders'])
                or set(state['gradient_updates'])!=set(self.gradient_updates)
                or any(type(x) is not int or x<0 or x>state['optimizer_updates'] for x in state['gradient_updates'].values())):
            raise ValueError('NcIEMIL complete TRAIN order, native SGD counters or gradient history changed')
        for key,value in state.items():
            if key!='contract':setattr(self,key,copy.deepcopy(value))

    def protocol(self):
        return {**self.core.protocol(),'method':self.profile,'completed_epochs':self.completed_epochs,
            'optimizer_updates':self.optimizer_updates,'gradient_updates':self.gradient_updates,
            'filter_checkpoint_sha256':self.inputs.meta['filter_checkpoint_sha256'],
            'filter_stage':self.inputs.meta['filter_protocol'],'full_method_admission':False,
            'recipe':'Literal source train/VAL, batch4/4readers, SGD2e-4/wd1e-5/no momentum,100epochs/no early stopping; latest harmonic ACC/macroF1/sourceAUC ties',
            'explicit_zero_harmonic_repair':True,'multiclass_source_auc':'micro ROC over flattened one-hot classes; separately report standard macro metrics',
            'missing':'No author filtering trainer/checkpoint/complete hyperparameters published; explicit reference reconstruction disclosed'}

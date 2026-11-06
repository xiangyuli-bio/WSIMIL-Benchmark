"""Full released GMMamba coordinate-grouped trainer and recoverable state."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import joblib
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import label_binarize
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,RandomSampler,SequentialSampler
from tqdm import tqdm

from .gmmamba_source import COMMIT,GMMambaNetwork,author_grouping,validated_groups
from .io import digest_file,digest_object
from .spatial_attention import spatial_metadata
from .upstream import load_definitions


class TrackedSampler:
    def __init__(self,sampler):self.sampler,self.indices=sampler,[]
    def __len__(self):return len(self.sampler)
    def __iter__(self):
        self.indices=[]
        for index in self.sampler:
            self.indices.append(int(index));yield index


def load_bag(bundle,row):
    from .data import load_array,resolve
    features,_=bundle.load(row,require_coords=True)
    if row.get('coords_path'):raw=load_array(resolve(bundle.root,row['coords_path']),'coords')
    else:
        with h5py.File(resolve(bundle.root,row['path']),'r') as archive:raw=archive['coords'][:]
    # Original HDF5 Dataset preserves coordinate dtype. In particular integer
    # source coordinates become float64 inside sklearn; casting to float32
    # before KMeans can change group assignment and must not happen here.
    return features,torch.from_numpy(np.ascontiguousarray(raw))


def deferred_test_rng_port():
    """Original sequential TEST iterator consumes exactly one CPU base seed.

    Its two child-reader RNG streams do not modify the parent, and the source
    eval network/dataset has no random operators. Replay the parent draw only,
    deferring actual TEST access to explicit evaluation. An actual literal
    source TEST pass is the independent RNG oracle.
    """
    torch.empty((),dtype=torch.int64).random_()


def author_training():
    original_grouping=author_grouping()
    class CheckedGrouping(original_grouping):
        def coords_grouping(self,coords,features,c_norm=False):
            if c_norm:raise ValueError('Original GMMamba training does not normalize coordinates')
            return validated_groups(features.squeeze(0),coords.squeeze(0),self.groups_num)
    metrics=load_definitions('gmmamba','utils/utils.py',
        ['calculate_error','calculate_metrics_two','calculate_metrics_overtwo'],
        {'torch':torch,'np':np,'precision_score':precision_score,'recall_score':recall_score,
         'f1_score':f1_score,'roc_auc_score':roc_auc_score,'accuracy_score':accuracy_score,'label_binarize':label_binarize})
    return load_definitions('gmmamba','utils/core.py',['train_function','val_function'],
        {'torch':torch,'nn':nn,'np':np,'os':os,'joblib':joblib,'tqdm':tqdm,
         'grouping':CheckedGrouping,'calculate_error':metrics.calculate_error,
         'calculate_metrics_two':metrics.calculate_metrics_two,'calculate_metrics_overtwo':metrics.calculate_metrics_overtwo,
         # CSV timestamps do not affect model/RNG/selection; metrics are saved
         # by the public engine's history/prediction artifacts instead.
         'save_result':lambda *args,**kwargs:None})


class SourceSelection:
    def __init__(self):
        self.best=0.;self.counter=0;self.selected_epoch=None;self.epoch=-1
        self.improved=False;self.done=False

    def observe(self,epoch,auc,f1):
        if type(epoch) is not int or epoch!=self.epoch+1 or not np.isfinite([auc,f1]).all():
            raise ValueError('GMMamba source selection history is not consecutive/finite')
        self.epoch=epoch;score=float(auc+f1)
        self.improved=epoch>5 and score>self.best
        if self.improved:self.best=score;self.counter=0;self.selected_epoch=epoch
        else:self.counter+=1
        self.done=self.counter>=20 and epoch>50
        return score

    def state_dict(self):return copy.deepcopy(vars(self))

    def load_state_dict(self,state):
        if (not isinstance(state,dict) or set(state)!=set(vars(self)) or type(state['epoch']) is not int
                or type(state['counter']) is not int or state['counter']<0
                or type(state['improved']) is not bool or type(state['done']) is not bool
                or not np.isfinite(state['best']) or not 0<=state['best']<=2
                or (state['selected_epoch'] is not None and not 6<=state['selected_epoch']<=state['epoch'])
                or state['done']!=(state['counter']>=20 and state['epoch']>50)):
            raise ValueError('GMMamba exact source early-stopping state changed')
        self.__dict__.update(copy.deepcopy(state))


class Bags(Dataset):
    def __init__(self,owner,subset):
        self.owner,self.rows=owner,owner.bundle.records(subset)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        self.owner.check_inputs();row=self.rows[index]
        features,coords=load_bag(self.owner.bundle,row)
        if len(features)<2*self.owner.network.groups or coords.shape!=(len(features),2):
            raise ValueError('GMMamba needs complete aligned spatial bags')
        # Source grouping belongs in the parent training process, not inside
        # forked readers. The checked original grouping below performs one
        # actual KMeans per bag and validates all groups before IMM execution.
        # Explicitly compute source coordinate groups for this bound feature
        # version; never accept an unversioned detached `.pkl` cache.
        cache=self.owner.bundle.root/('.milplus_gmmamba_no_cache_'+self.owner.contract)/row['slide_id']
        if Path(str(cache)+'_'+str(self.owner.network.groups)+'.pkl').exists():
            raise ValueError('Unversioned GMMamba precomputed feature cache is not accepted')
        return coords,features,int(row['target']),str(cache)


class GMMambaMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile='gmmamba_mambamil_bimamba_source',groups=10,mask_ratio=.3,
                 backend='torch',scan_chunk=64):
        super().__init__()
        if (bundle.task_type!='classification' or len(bundle.classes) not in {2,3}
                or bundle.config.get('training_unit','slide')!='slide'
                or bundle.config.get('evaluation_unit','slide')!='slide'):
            raise ValueError('Original GMMamba uses two/three-class slide bags and slide-level source VAL metrics')
        self.bundle,self.profile=bundle,profile;self.spatial=spatial_metadata(bundle)
        self._rows=self.labels();self._files=self.input_files()
        self.network=GMMambaNetwork(bundle.dim,len(bundle.classes),profile,mask_ratio,groups,backend,scan_chunk)
        self.contract=digest_object({'source':COMMIT,'profile':profile,'bundle':bundle.fingerprint,
            'dimension':bundle.dim,'classes':bundle.classes,'spatial':self.spatial,'rows':self._rows,
            'files':self._files,'network':self.network.protocol()})
        self.recipe=None;self._optimizer=None;self.selection=SourceSelection()
        self.completed_epochs=0;self.epoch_index=-1;self.optimizer_updates=0
        self.train_orders=[];self.val_orders=[];self._source_metrics=None
        self.gradient_updates={name:0 for name,_ in self.network.named_parameters()}
        self.deferred_test_rng_epochs=[]

    def labels(self):
        return [{k:row.get(k) for k in ['slide_id','patient_id','global_patient_id','subset','target','path','coords_path']}
                for row in self.bundle.rows.to_dict('records')]

    def input_files(self):
        from .data import resolve
        result={}
        for row in self.bundle.rows.to_dict('records'):
            self.bundle.load(row,require_coords=True)
            for name in ['path','coords_path']:
                if row.get(name):
                    path=resolve(self.bundle.root,row[name]);result[str(path)]={
                        'sha256':digest_file(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        return result

    def check_inputs(self):
        if self.labels()!=self._rows or spatial_metadata(self.bundle)!=self.spatial:
            raise ValueError('GMMamba patient/split/label/coordinate declaration changed')
        for name,expected in self._files.items():
            path=Path(name)
            if not path.is_file() or path.stat().st_size!=expected['bytes'] or path.stat().st_mtime_ns!=expected['mtime_ns']:
                raise ValueError('GMMamba original feature/coordinate bytes changed')

    def configure_training(self,cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['lr']!=1e-4
                or cfg['weight_decay']!=1e-5 or cfg['selection_metric']!='gmmamba_val_auc_f1'
                or cfg['patience']!=20 or type(cfg['epochs']) is not int or cfg['epochs']<7):
            raise ValueError('GMMamba needs full bags/CE/AdamW1e-4wd1e-5,source first selection epoch6 and patience20/epoch>50')
        self.recipe={k:cfg[k] for k in ['epochs','lr','weight_decay','patience','threads','selection_metric']}

    def make_optimizer(self,lr,weight_decay):
        if self.recipe is None or lr!=self.recipe['lr'] or weight_decay!=self.recipe['weight_decay']:
            raise ValueError('GMMamba complete original recipe must be configured before optimization')
        # Literal author optimizer parameter order is IMM, classifier, CSS.
        parameters=list(self.network.imm.parameters())+list(self.network.classifier.parameters())+list(self.network.css.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=weight_decay)
        def count(_optimizer,_args,_kwargs):
            self.optimizer_updates+=1
            for name,p in self.network.named_parameters():self.gradient_updates[name]+=int(p.grad is not None)
        optimizer.register_step_pre_hook(count);self._optimizer=optimizer;return optimizer

    def loader(self,subset):
        dataset=Bags(self,subset)
        sampler=TrackedSampler(RandomSampler(dataset) if subset!='test' else SequentialSampler(dataset))
        # Preserve original two readers and shuffled TRAIN/VAL, sequential TEST.
        return dataset,sampler,DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True)

    def start_epoch(self,epoch):
        if type(epoch) is not int or epoch!=self.completed_epochs:
            raise ValueError('GMMamba epoch does not continue complete source history')
        self.epoch_index=epoch

    def epoch(self,bundle,optimizer,device):
        if bundle is not self.bundle or optimizer is not self._optimizer or self.epoch_index!=self.completed_epochs:
            raise ValueError('GMMamba source epoch/optimizer binding changed')
        dataset,sampler,loader=self.loader('train');losses=[];criterion=nn.CrossEntropyLoss()
        hook=criterion.register_forward_hook(lambda module,args,value:losses.append(float(value.detach())))
        before=self.optimizer_updates
        try:
            author_training().train_function(SimpleNamespace(n_classes=len(bundle.classes),num_subbags=self.network.groups),
                self.network.imm,self.network.css,self.network.classifier,self.network.memory,self.epoch_index,
                loader,device,criterion,optimizer)
        finally:hook.remove()
        if (len(losses)!=len(dataset) or self.optimizer_updates-before!=len(dataset)
                or sorted(sampler.indices)!=list(range(len(dataset))) or not np.isfinite(losses).all()
                or self.network.memory.msg_states):
            raise ValueError('GMMamba original full-bag updates or memory reset missing')
        self.train_orders.append([dataset.rows[i]['slide_id'] for i in sampler.indices]);self.completed_epochs+=1
        return [{'loss':sum(losses)/len(losses)}]

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        if bundle is not self.bundle or subset not in {'val','test'}:raise ValueError('GMMamba bound VAL/TEST subset required')
        dataset,sampler,loader=self.loader(subset);probabilities=[]
        def capture(module,args,value):probabilities.append(value['Y_prob'][0].detach().cpu().tolist())
        hook=self.network.classifier.register_forward_hook(capture)
        try:
            values=author_training().val_function(SimpleNamespace(n_classes=len(bundle.classes),num_subbags=self.network.groups),
                self.network.imm,self.network.css,self.network.classifier,self.network.memory,loader,device,
                self.epoch_index,currenttype=subset)
        finally:hook.remove()
        if sorted(sampler.indices)!=list(range(len(dataset))) or len(probabilities)!=len(dataset) or self.network.memory.msg_states:
            raise ValueError('GMMamba source prediction omitted/merged original slide bags')
        self._source_metrics=(subset,{k:float(v) for k,v in zip(['auc','accuracy','f1','precision','recall'],values)})
        if subset=='val' and len(self.val_orders)<self.completed_epochs:
            self.val_orders.append([dataset.rows[i]['slide_id'] for i in sampler.indices])
        records=[]
        for index,values in zip(sampler.indices,probabilities):
            row=dataset.rows[index];record={k:row[k] for k in ['slide_id','patient_id','global_patient_id','subset','target']}
            record.update(source_slides=row['slide_id'],**{f'p_{i}':float(v) for i,v in enumerate(values)});records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self,frame):
        if self._source_metrics is None or set(frame.subset)!={self._source_metrics[0]}:
            raise ValueError('GMMamba source metrics not bound to actual predictions')
        subset,values=self._source_metrics
        return {**{'gmmamba_'+subset+'_'+k:v for k,v in values.items()},
                'gmmamba_'+subset+'_auc_f1':values['auc']+values['f1']}

    def selection_objective(self,metrics,epoch,current):
        score=self.selection.observe(epoch,metrics['gmmamba_val_auc'],metrics['gmmamba_val_f1'])
        if self.selection.improved:
            deferred_test_rng_port();self.deferred_test_rng_epochs.append(epoch)
        return score

    def selection_improved(self,objective,best):return self.selection.improved

    @property
    def training_done(self):return self.selection.done

    def get_extra_state(self):
        if self.network.memory.msg_states:raise ValueError('GMMamba checkpoint must be at an actual cleared bag boundary')
        return copy.deepcopy({'contract':self.contract,'recipe':self.recipe,'epoch_index':self.epoch_index,
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'train_orders':self.train_orders,'val_orders':self.val_orders,'gradient_updates':self.gradient_updates,
            'selection':self.selection.state_dict(),'deferred_test_rng_epochs':self.deferred_test_rng_epochs})

    def set_extra_state(self,state):
        keys={'contract','recipe','epoch_index','completed_epochs','optimizer_updates','train_orders','val_orders','gradient_updates','selection','deferred_test_rng_epochs'}
        if (not isinstance(state,dict) or set(state)!=keys or state['contract']!=self.contract
                or self.recipe is not None and state['recipe']!=self.recipe):
            raise ValueError('GMMamba checkpoint source/input/recipe contract changed')
        epochs=state['completed_epochs'];ids={r['slide_id'] for r in self.bundle.records('train')}
        val={r['slide_id'] for r in self.bundle.records('val')}
        if (type(epochs) is not int or epochs<0 or state['epoch_index']!=epochs-1
                or state['optimizer_updates']!=epochs*len(ids) or len(state['train_orders'])!=epochs
                or len(state['val_orders'])!=epochs
                or any(len(order)!=len(ids) or set(order)!=ids for order in state['train_orders'])
                or any(len(order)!=len(val) or set(order)!=val for order in state['val_orders'])
                or set(state['gradient_updates'])!=set(self.gradient_updates)
                or any(type(n) is not int or n<0 or n>state['optimizer_updates'] for n in state['gradient_updates'].values())
                or state['selection']['epoch']!=epochs-1
                or state['deferred_test_rng_epochs']!=sorted(set(state['deferred_test_rng_epochs']))
                or any(type(n) is not int or n<6 or n>=epochs for n in state['deferred_test_rng_epochs'])
                or (state['deferred_test_rng_epochs'][-1] if state['deferred_test_rng_epochs'] else None)!=state['selection']['selected_epoch']):
            raise ValueError('GMMamba checkpoint omitted full original training/validation/update history')
        self.selection.load_state_dict(state['selection'])
        for name in keys-{'contract','selection'}:setattr(self,name,copy.deepcopy(state[name]))

    def validate_optimizer_state(self):
        if self._optimizer is None:raise ValueError('GMMamba optimizer must be constructed before moment restoration')
        for name,p in self.network.named_parameters():
            count=self.gradient_updates[name];state=self._optimizer.state.get(p)
            if (count and (state is None or int(state['step'])!=count)) or (not count and state is not None):
                raise ValueError('GMMamba AdamW state disagrees with actual original parameter updates: '+name)

    def protocol(self):
        return {**self.network.protocol(),'contract':self.contract,'recipe':self.recipe,
            'optimizer':'Original constant AdamW1e-4/1e-5,one complete slide per update,no clipping',
            'loader':'Original2readers/shuffledTRAINandVAL/sequentialTEST; complete original coordinates and row order',
            'selection':'First strict VAL AUC+F1 improvement after epoch_index5; initial best0,patience20 and epoch_index>50',
            'selection_state':self.selection.state_dict(),'completed_epochs':self.completed_epochs,
            'optimizer_updates':self.optimizer_updates,
            'logging_port':'Source diagnostic CSV routed to unified history; no RNG/model effects',
            'test_boundary':'Original per-improvement TEST access deferred; replay only its exact parent CPU iterator base-seed. Two child RNG streams and deterministic eval do not modify parent. Actual source TEST oracle required before formal admission.',
            'deferred_test_rng_epochs':self.deferred_test_rng_epochs,
            'formal_admission':False,'paper_reproduction':'not_run','real_wsi_validation':'not_run'}

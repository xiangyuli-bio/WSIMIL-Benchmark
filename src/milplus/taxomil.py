"""TaxoMIL complete released generative training recipe and exact-match outputs."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler

from .io import digest_file, digest_object
from .taxomil_data import TaxoMILInputs
from .taxomil_source import author_components, source_identity
from .upstream import load_definitions


class TrackedSampler(Sampler):
    def __init__(self,sampler):self.sampler=sampler;self.indices=[]
    def __iter__(self):
        self.indices=[]
        for index in self.sampler:self.indices.append(index);yield index
    def __len__(self):return len(self.sampler)


class Bags(Dataset):
    def __init__(self,owner,subset):self.owner=owner;self.rows=owner.bundle.records(subset)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];self.owner.check_inputs()
        x,_=self.owner.bundle.load(row)
        coarse=self.owner.bundle.hierarchy.coarse[self.owner.bundle.hierarchy.mapping[row['target']]]
        fine=self.owner.bundle.classes[row['target']]
        # Preserve original coarse normalization and PANDA's case-sensitive
        # fine labels. The original target tokenizer appends EOS then PAD.
        fine=fine.strip() if self.owner.inputs.dataset=='PANDA' else fine.strip().lower()
        return x,self.owner.encode(coarse.strip().lower(),5),self.owner.encode(fine,15)


def text_metrics(frame):
    from sklearn.metrics import cohen_kappa_score, f1_score
    if not len(frame):raise ValueError('TaxoMIL cannot score an empty generated output set')
    result={}
    for level in ['coarse','fine']:
        truth=frame['taxomil_'+level+'_target'].tolist();pred=frame['taxomil_'+level+'_text'].tolist()
        if not all(isinstance(x,str) for x in truth+pred):raise ValueError('TaxoMIL generated and target values must be literal text')
        labels=sorted(set(truth+pred));ids={x:i for i,x in enumerate(labels)}
        y=[ids[x] for x in truth];p=[ids[x] for x in pred]
        result['taxomil_'+level+'_accuracy']=sum(a==b for a,b in zip(truth,pred))/len(truth)*100.0
        result['taxomil_'+level+'_weighted_f1']=float(f1_score(y,p,average='weighted',zero_division=0))
        kappa=float(cohen_kappa_score(y,p))
        result['taxomil_'+level+'_kappa']=kappa if math.isfinite(kappa) else None
    result['taxomil_score']=(result['taxomil_coarse_accuracy']+result['taxomil_fine_accuracy'])/2
    result['taxomil_holistic_accuracy']=100*float((
        frame.taxomil_coarse_text.eq(frame.taxomil_coarse_target)&frame.taxomil_fine_text.eq(frame.taxomil_fine_target)).mean())
    target=(frame.taxomil_coarse_target+'|'+frame.taxomil_fine_target).tolist()
    pred=(frame.taxomil_coarse_text+'|'+frame.taxomil_fine_text).tolist()
    labels={v:i for i,v in enumerate(sorted(set(target+pred)))}
    y=[labels[v] for v in target];p=[labels[v] for v in pred]
    result['taxomil_holistic_weighted_f1']=float(f1_score(y,p,average='weighted',zero_division=0))
    value=float(cohen_kappa_score(y,p))
    result['taxomil_holistic_kappa']=value if math.isfinite(value) else None
    return {**result,'n':len(frame),'unit':'slide','prediction_type':'unconstrained_generated_text_exact_match'}


class TaxoMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile='taxomil_gpt2_all_losses_key_repaired',batch_size=64,num_workers=4):
        super().__init__()
        if profile not in {'taxomil_gpt2_all_losses_key_repaired','taxomil_gpt2_paper_lr_key_repaired'} or bundle.dim!=1024:
            raise ValueError('Original TaxoMIL aggregator outputs1024D; its source split adaptors require native1024D input. No silent FM projection')
        if type(batch_size) is not int or batch_size<1 or type(num_workers) is not int or num_workers<0:
            raise ValueError('TaxoMIL needs positive batch_size and nonnegative original reader count')
        self.bundle,self.profile=bundle,profile;self.inputs=TaxoMILInputs(bundle)
        self.batch_size,self.num_workers=batch_size,num_workers
        self.decoder_defs,self.loss_defs,self.scheduler_cls,self.utils,self.collate,self.trainer=author_components()
        self.network=self.decoder_defs.build_model(model_path=str(self.inputs.assets),device='cpu',
            dataset=self.inputs.dataset,emb_dim=1024,gated=True,output_len_coarse=5,output_len_fine=15,
            all_coarse_labels=bundle.hierarchy.coarse,all_fine_labels=bundle.classes)
        # Required original run_experiment patch: two additional real tokens
        # were added after constructing the original50257-class output heads.
        self.utils.patch_unembedding_to_tokenizer(self.network,self.network.tokenizer)
        self.label_maps={'coarse':{s:i for i,s in enumerate(bundle.hierarchy.coarse)},
                         'fine':{s:i for i,s in enumerate(bundle.classes)}}
        self.criteria={'gen':nn.CrossEntropyLoss(ignore_index=self.network.tokenizer.pad_token_id),
            'hier':self.loss_defs.TextHierarchicalLoss(bundle.classes,bundle.hierarchy.coarse,
                bundle.config['hierarchy']['fine_to_coarse'],tau=1.,margin=1.5),
            'c_ital':self.loss_defs.AlignmentLoss(.07),'f_ital':self.loss_defs.AlignmentLoss(.07),
            'c_cl':self.loss_defs.ContrastiveLoss(.07),'f_cl':self.loss_defs.ContrastiveLoss(.07)}
        self._rows=bundle.rows.to_dict('records');self._feature_identity={}
        from .data import resolve
        for row in self._rows:
            bundle.load(row);p=resolve(bundle.root,row['path'])
            self._feature_identity[str(p)]={'sha256':digest_file(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        self.contract=digest_object({'source':source_identity(),'profile':profile,'inputs':self.inputs.protocol(),
            'rows':self._rows,'features':self._feature_identity,'batch_size':batch_size,'num_workers':num_workers})
        self.recipe=None;self._optimizer=None;self.scheduler=None;self.epoch_index=-1;self.completed_epochs=0
        self.optimizer_updates=0;self.gradient_updates={name:0 for name,_ in self.network.named_parameters()}
        self.train_orders=[];self.val_orders=[];self._bank=None;self._last_weights=None
        self.best_score=-1e9;self.bad=0;self.selected_epoch=None;self._improved=False;self._metrics=None

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse)
        if hasattr(self,'network'):
            device=next(self.network.parameters()).device
            self.network.device=str(device);self.network.decoder.device=str(device)
            for criterion in self.criteria.values():criterion.to(device)
        return result

    def encode(self,text,length):
        tokenizer=self.network.tokenizer
        ids=tokenizer(text,return_tensors='pt',truncation=True,max_length=max(1,length-1),add_special_tokens=False)['input_ids'].squeeze(0)
        ids=torch.cat([ids,torch.tensor([tokenizer.eos_token_id])])[:length]
        if len(ids)<length:ids=torch.cat([ids,torch.full((length-len(ids),),tokenizer.pad_token_id,dtype=ids.dtype)])
        return ids

    def check_inputs(self):
        from pathlib import Path
        self.inputs.check_inputs()
        if self.bundle.rows.to_dict('records')!=self._rows:raise ValueError('TaxoMIL complete patient/slide/target/split inputs changed')
        for name,row in self._feature_identity.items():
            p=Path(name)
            if not p.is_file() or p.stat().st_size!=row['bytes'] or p.stat().st_mtime_ns!=row['mtime_ns']:
                raise ValueError('TaxoMIL complete feature bytes changed')

    def configure_training(self,cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None or cfg['lr']!=(1e-4 if self.profile=='taxomil_gpt2_paper_lr_key_repaired' else 1e-5)
                or cfg['weight_decay']!=.01 or cfg['patience']!=20 or cfg['selection_metric']!='taxomil_score'
                or type(cfg['epochs']) is not int or cfg['epochs']<1):
            raise ValueError('TaxoMIL requires full padded source bags,declared source1e-5 or paper-LR1e-4 AdamW/wd.01,CosineLR/eta1e-7,all four objectives and sourceVAL exact match/patience20')
        self.recipe={k:cfg[k] for k in ['epochs','lr','weight_decay','patience','threads','selection_metric']}

    def make_optimizer(self,lr,weight_decay):
        if self.recipe is None or lr!=self.recipe['lr'] or weight_decay!=.01:
            raise ValueError('TaxoMIL full original recipe must be configured before optimization')
        self._optimizer=self.utils.get_optimizer('adamw',self.network.parameters(),lr)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,T_max=self.recipe['epochs'],eta_min=1e-7)
        def count(optimizer,args,kwargs):
            self.optimizer_updates+=1
            for name,p in self.network.named_parameters():self.gradient_updates[name]+=int(p.grad is not None)
        self._optimizer.register_step_pre_hook(count)
        return self._optimizer

    def loader(self,subset):
        ds=Bags(self,subset);sampler=TrackedSampler(RandomSampler(ds) if subset=='train' else SequentialSampler(ds))
        return ds,sampler,DataLoader(ds,batch_size=self.batch_size,sampler=sampler,collate_fn=self.collate,
            drop_last=False,num_workers=self.num_workers)

    def start_epoch(self,epoch):
        if type(epoch) is not int or epoch!=self.completed_epochs:raise ValueError('TaxoMIL source epochs must continue consecutively')
        self.epoch_index=epoch

    def source_args(self):
        return SimpleNamespace(use_HIER=True,use_ITAL=True,use_CL=True,wandb=False)

    def weights(self):
        # Literal source run_experiment omits total_epochs when constructing
        # this scheduler. Retain its100-epoch normalization for short probes.
        return self.scheduler_cls(hier_max=.3,ital_max=.3,cl_max=.1,n_cycles=3).get_weights(self.epoch_index)

    def text_bank(self,device):
        return self.utils.build_text_bank_for_losses(self.network,self.network.tokenizer,device,
            self.bundle.hierarchy.coarse,self.bundle.classes)

    def epoch(self,bundle,optimizer,device):
        if bundle is not self.bundle or optimizer is not self._optimizer or self.epoch_index!=self.completed_epochs:
            raise ValueError('TaxoMIL source epoch/optimizer binding changed')
        self.check_inputs();self._bank=self.text_bank(device);self._last_weights=self.weights()
        ds,sampler,loader=self.loader('train');values={key:[] for key in ['gen','hier','c_ital','f_ital','c_cl','f_cl']};hooks=[]
        for key in values:
            def capture(module,args,value,key=key):
                values[key].append(float((value[0] if isinstance(value,tuple) else value).detach()))
            hooks.append(self.criteria[key].register_forward_hook(capture))
        before=self.optimizer_updates
        try:
            self.trainer.train_one_epoch(self.epoch_index+1,self.network,self.criteria,optimizer,loader,device,
                self.network.tokenizer,self.source_args(),self.label_maps,self._last_weights,
                self.bundle.hierarchy.coarse,self.bundle.classes)
        finally:
            for hook in hooks:hook.remove()
        steps=math.ceil(len(ds)/self.batch_size)
        if self.optimizer_updates-before!=steps or sorted(sampler.indices)!=list(range(len(ds))):
            raise ValueError('TaxoMIL original batch updates/complete bag visits missing')
        if (len(values['gen'])!=2*steps or any(len(values[k])!=steps for k in values if k!='gen')
                or not all(math.isfinite(v) for a in values.values() for v in a)):
            raise ValueError('TaxoMIL complete four-objective training path was not executed')
        losses={'ce':sum(values['gen'])/steps,'hier':sum(values['hier'])/steps,
            'ital':sum(values['c_ital']+values['f_ital'])/steps,'cl':sum(values['c_cl']+values['f_cl'])/steps}
        total=sum(self._last_weights[k]*losses[k] for k in losses)
        self.train_orders.append([ds.rows[i]['slide_id'] for i in sampler.indices]);self.completed_epochs+=1
        return [{'loss':total,**losses}]

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        if bundle is not self.bundle or subset not in {'val','test'}:raise ValueError('TaxoMIL requires bound originalVAL/TEST slides')
        ds,sampler,loader=self.loader(subset);self.network.eval();outputs=[]
        def capture(module,args,value):outputs.extend(zip(value[0],value[1]))
        hook=self.network.register_forward_hook(capture)
        try:
            if subset=='val':
                bank=self._bank if self._bank is not None else self.text_bank(device)
                weights=self._last_weights if self._last_weights is not None else self.weights()
                score,ce,acc_c,acc_f=self.trainer.validate(self.epoch_index+1,self.network,self.criteria,loader,
                    device,self.network.tokenizer,self.source_args(),self.label_maps,weights,bank)
                self._metrics={'taxomil_score':score,'taxomil_generation_ce':ce,
                    'taxomil_coarse_accuracy':acc_c,'taxomil_fine_accuracy':acc_f}
            else:
                test=load_definitions('taxomil','trainer/test.py',['test'],
                    {'torch':torch,'tqdm':lambda it,**kw:it}).test
                test(self.network,loader,device,self.network.tokenizer);self._metrics=None
        finally:hook.remove()
        if sampler.indices!=list(range(len(ds))) or len(outputs)!=len(ds):
            raise ValueError('TaxoMIL original generated predictions omitted/merged slide bags')
        records=[]
        for row,(coarse,fine) in zip(ds.rows,outputs):
            c=self.bundle.hierarchy.coarse[self.bundle.hierarchy.mapping[row['target']]]
            f=self.bundle.classes[row['target']]
            c=self.network.tokenizer.decode(self.encode(c.strip().lower(),5),skip_special_tokens=True).strip()
            f=self.network.tokenizer.decode(self.encode(f.strip() if self.inputs.dataset=='PANDA' else f.strip().lower(),15),skip_special_tokens=True).strip()
            record={k:row[k] for k in ['slide_id','patient_id','global_patient_id','subset','target']}
            record.update(source_slides=row['slide_id'],taxomil_coarse_text=str(coarse).strip(),
                taxomil_fine_text=str(fine).strip(),taxomil_coarse_target=c,taxomil_fine_target=f)
            records.append(record)
        frame=pd.DataFrame(records)
        if subset=='val':
            measured=text_metrics(frame)
            for key in ['taxomil_score','taxomil_coarse_accuracy','taxomil_fine_accuracy']:
                if measured[key]!=self._metrics[key]:raise ValueError('TaxoMIL exported exact matches differ from literal source validation')
            if len(self.val_orders)<self.completed_epochs:self.val_orders.append([r['slide_id'] for r in ds.rows])
        return frame

    def validation_metrics(self,frame):return {**text_metrics(frame),**(self._metrics or {})}

    def selection_objective(self,metrics,epoch,current):
        if epoch!=self.epoch_index or len(self.val_orders)!=self.completed_epochs:raise ValueError('TaxoMIL requires actual complete source validation before selection')
        score=metrics['taxomil_score'];self._improved=score>self.best_score
        if self._improved:self.best_score=score;self.bad=0;self.selected_epoch=epoch
        else:self.bad+=1
        # Literal source CosineAnnealingLR step follows validation and precedes
        # its checkpoint/early-stop decisions. Preserve exact resume position.
        self.scheduler.step();self._bank=None
        return score

    def selection_improved(self,objective,best):return self._improved
    @property
    def training_done(self):return self.bad>=20

    def get_extra_state(self):
        if self._bank is not None:raise ValueError('TaxoMIL checkpoint must follow complete source validation/selection')
        return copy.deepcopy({'contract':self.contract,'recipe':self.recipe,'epoch_index':self.epoch_index,
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'gradient_updates':self.gradient_updates,'train_orders':self.train_orders,'val_orders':self.val_orders,
            'best_score':self.best_score,'bad':self.bad,'selected_epoch':self.selected_epoch,'improved':self._improved,
            'last_weights':self._last_weights,'scheduler':getattr(self,'_loaded_scheduler',None) if self.scheduler is None else self.scheduler.state_dict()})

    def set_extra_state(self,state):
        fields={'contract','recipe','epoch_index','completed_epochs','optimizer_updates','gradient_updates',
            'train_orders','val_orders','best_score','bad','selected_epoch','improved','last_weights','scheduler'}
        if not isinstance(state,dict) or set(state)!=fields or state['contract']!=self.contract or self.recipe is not None and state['recipe']!=self.recipe:
            raise ValueError('TaxoMIL checkpoint source/input/taxonomy/recipe contract changed')
        n=state['completed_epochs'];train={r['slide_id'] for r in self.bundle.records('train')};val={r['slide_id'] for r in self.bundle.records('val')}
        if (type(n) is not int or n<0 or state['epoch_index']!=n-1
                or state['optimizer_updates']!=n*math.ceil(len(train)/self.batch_size)
                or len(state['train_orders'])!=n or len(state['val_orders'])!=n
                or any(len(x)!=len(train) or set(x)!=train for x in state['train_orders'])
                or any(len(x)!=len(val) or set(x)!=val for x in state['val_orders'])
                or set(state['gradient_updates'])!=set(self.gradient_updates)
                or any(type(v) is not int or not 0<=v<=state['optimizer_updates'] for v in state['gradient_updates'].values())
                or state['selected_epoch'] is not None and not 0<=state['selected_epoch']<n
                or not math.isfinite(state['best_score']) or type(state['bad']) is not int or not 0<=state['bad']<=n
                or type(state['improved']) is not bool or n and (state['scheduler'] is None or state['scheduler']['last_epoch']!=n)):
            raise ValueError('TaxoMIL checkpoint omitted complete source optimizer/scheduler/sampling/selection history')
        for key in fields-{'contract','scheduler','improved','last_weights'}:setattr(self,key,copy.deepcopy(state[key]))
        self._last_weights=copy.deepcopy(state['last_weights'])
        self._improved=state['improved']
        if state['scheduler'] is not None:
            if self.scheduler is None:
                # Independent evaluation builds model before configuring an
                # optimizer. Retain scheduler receipt without manufacturing one.
                self._loaded_scheduler=copy.deepcopy(state['scheduler'])
            else:self.scheduler.load_state_dict(state['scheduler'])

    def validate_optimizer_state(self):
        if self._optimizer is None:raise ValueError('TaxoMIL optimizer is required before moment restoration')
        for name,p in self.network.named_parameters():
            n=self.gradient_updates[name];state=self._optimizer.state.get(p)
            if n and (state is None or int(state['step'])!=n) or not n and state is not None:
                raise ValueError('TaxoMIL AdamW moments disagree with actual source parameter updates: '+name)
        if self.scheduler.last_epoch!=self.completed_epochs:raise ValueError('TaxoMIL full source cosine scheduler state missing')

    def protocol(self):
        return {'source':source_identity(),'profile':self.profile,'contract':self.contract,'inputs':self.inputs.protocol(),
            'architecture':'Full original pretrained GPT2 12layer/768D/12head plus two GPT2Block heads and independent50259vocab unembeddings; original1024D gatedABMIL and512/512 adaptors',
            'recipe':self.recipe,'batch_size':self.batch_size,'num_workers':self.num_workers,
            'objective':'Literal generation coarse+fine CE,TextHierarchicalLoss,dualITAL,dualSupCon; cos2/sin2 three-cycle weights .3/.3/.1; no substitute bag CE',
            'padding':'Literal source zero-padded unequal bags without attention masking; complete original row order/no truncation',
            'selection':'First strict average coarse/fine generated-string exact-match percentage; patience20; TEST only explicit after training',
            'explicit_source_repairs':['Alignment helper coarse_ital/fine_ital bound to actual c_ital/f_ital criteria; original all-loss source otherwise raises KeyError'],
            'source_paper_differences':['Released sibling margin uses raw detached token embeddings and factor.5, whereas paper Eq6 uses projected label embeddings; preserved explicitly',
                'Released label-text loss bank mean-pooling is no_grad; projection remains trainable',
                'Released epoch validation bank is computed before training; preserve snapshot for sourceVAL losses',
                'Released periodic scheduler defaults to100 epochs even for explicitly shorter software probes',
                'Released BRACS/PANDA CLI defaults lr1e-5 while paper AppendixD prescribes1e-4; paper_lr profile changes only the declared LR and still preserves/discloses source loss equations'],
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'paper_reproduction':'not_run','real_wsi_validation':'not_run','formal_admission':False}

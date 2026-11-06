"""Complete DT-MIL reference training; unpublished historic choices stay explicit."""
import math
import types

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .dtmil_data import SpatialInputs, encode_map
from .dtmil_network import COMMIT, REPAIRED, SOURCE, network
from .io import digest_object
from .loss_attention import check_update_finite, finite
from .smmile_data import positive_int


class DTMIL(nn.Module):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self, bundle, profile=REPAIRED, map_size=60, hidden=256, encoder_layers=6,
                 decoder_layers=6, heads=8, feedforward=1024, dropout=.1, queries=100,
                 points=4, batch_size=20, lr_backbone=2e-5, lr_projection_multiplier=.1,
                 lr_drop=40, max_dense_bytes=32*1024**3):
        super().__init__()
        if (profile not in {SOURCE,REPAIRED} or bundle.task_type!='classification'
                or bundle.config.get('training_unit','slide')!='slide' or len(bundle.classes)<2):
            raise ValueError('DT-MIL requires individual-slide classification with a named reference recipe')
        if profile==SOURCE and len(bundle.classes)!=2:
            raise ValueError('DT-MIL released dataset collapses labels to binary; select repaired route for C-way tasks')
        for name,value in {'map_size':map_size,'hidden':hidden,'encoder_layers':encoder_layers,
                'decoder_layers':decoder_layers,'heads':heads,'feedforward':feedforward,'queries':queries,
                'points':points,'batch_size':batch_size,'lr_drop':lr_drop,'max_dense_bytes':max_dense_bytes}.items():
            positive_int(value,name)
        if map_size<2 or hidden%32 or hidden%heads or hidden%2:
            raise ValueError('DT-MIL map_size>=2, hidden divisible by32 and heads required by original norms/attention')
        if not 0<=finite(dropout,'dropout')<1 or finite(lr_backbone,'lr_backbone')<=0 or finite(lr_projection_multiplier,'lr_projection_multiplier')<=0:
            raise ValueError('DT-MIL dropout/LR-group settings invalid')
        self.bundle,self.profile=bundle,profile
        self.settings={'map_size':map_size,'hidden':hidden,'encoder_layers':encoder_layers,
            'decoder_layers':decoder_layers,'heads':heads,'feedforward':feedforward,'dropout':dropout,
            'queries':queries,'points':points,'batch_size':batch_size,'lr_backbone':lr_backbone,
            'lr_projection_multiplier':lr_projection_multiplier,'lr_drop':lr_drop,
            'max_dense_bytes':max_dense_bytes,'operator_backend':'exact_author_torch_reference'}
        self.reserve(batch_size)
        self.assets=SpatialInputs(bundle,profile,map_size,max_dense_bytes)
        self.network,self.criterion=network(bundle.dim,len(bundle.classes),hidden,encoder_layers,
            decoder_layers,heads,feedforward,dropout,queries,points)
        self.contract=digest_object({'source':COMMIT,'profile':profile,'settings':self.settings,
            'assets':self.assets.contract,'recipe':'explicit50epoch_reference_AdamW_LRgroups_StepLR_clip_firstVAL'})
        self.optimizer,self.scheduler=None,None
        self.saved_scheduler=None
        self.total_epochs,self.epoch_index,self.completed_epochs=0,-1,0
        self.optimizer_steps,self.train_orders,self.last_epoch_audits=0,[],[]
        self.gradient_clip=.1

    def reserve(self,count):
        s=self.settings if hasattr(self,'settings') else None
        if s is None:
            raise ValueError('DT-MIL dimension settings must precede memory planning')
        n,h,q,l=s['map_size']**2,s['hidden'],s['queries'],s['encoder_layers']
        # Dense FM canvas, all retained attention/FFN graphs and parameters/Adam.
        estimate=(count*n*(self.bundle.dim*8+h*160*l)+count*q*n*s['heads']*16*s['decoder_layers']
                  +32*(self.bundle.dim*h+18*h*h*(l+s['decoder_layers'])))
        if estimate>s['max_dense_bytes']:
            raise ValueError('DT-MIL complete native model/map exceeds max_dense_bytes before allocation')

    def configure_training(self,cfg):
        if (cfg['max_instances'] is not None or cfg['selection_metric'] not in {'log_loss','macro_auc'}
                or cfg['grad_clip'] is None or finite(cfg['grad_clip'],'grad_clip')<=0):
            raise ValueError('DT-MIL needs complete physical inputs, positive clipping and explicit VAL log_loss/macro_auc')
        if cfg.get('patience',cfg['epochs'])<cfg['epochs']:
            raise ValueError('DT-MIL reference completes configured epochs; no unpublished early-stopping rule is inferred')
        self.total_epochs=cfg['epochs'];self.gradient_clip=cfg['grad_clip']

    def prepare(self,bundle,seed,device):
        for row in bundle.rows.to_dict('records'):
            self.assets.row(row,False)
            if row['subset']=='train' and self.assets.cached is not None:
                features,_=bundle.load(row)
                self.assets.read_views(row,len(features))

    def make_optimizer(self,lr,weight_decay):
        groups={'main':[],'backbone':[],'projection':[]}
        for name,p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            backbone='backbone.0' in name
            projection=any(k in name for k in ['reference_points','sampling_offsets'])
            if backbone and projection:
                raise ValueError('DT-MIL reference LR ownership overlaps')
            groups['backbone' if backbone else 'projection' if projection else 'main'].append(p)
        rates={'main':lr,'backbone':self.settings['lr_backbone'],
               'projection':lr*self.settings['lr_projection_multiplier']}
        self.optimizer=torch.optim.AdamW([{'params':values,'lr':rates[name],'group_name':name}
            for name,values in groups.items()],lr=lr,weight_decay=weight_decay)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.settings['lr_drop'],gamma=.1)
        return self.optimizer

    def start_epoch(self,epoch):
        self.epoch_index=epoch

    def finish_epoch(self,bundle,device,cfg):
        self.scheduler.step()
        self.completed_epochs=self.epoch_index+1

    def inputs(self,rows,device,train=False):
        self.reserve(len(rows))
        maps=[self.assets.row(row,train) for row in rows]
        sample=types.SimpleNamespace(tensors=torch.stack([x for x,_,_ in maps]).to(device),
                                    mask=torch.stack([mask for _,mask,_ in maps]).to(device))
        return sample,[meta for _,_,meta in maps]

    def batch_forward(self,sample,metadata=None,targets=None):
        x,mask=sample.tensors,sample.mask
        size=self.settings['map_size']
        if (x.ndim!=4 or x.shape[1:]!=(self.bundle.dim,size,size) or len(x)<1
                or not x.is_floating_point() or not torch.isfinite(x).all()
                or mask.shape!=(len(x),size,size) or mask.dtype!=torch.bool or mask.flatten(1).all(1).any()):
            raise ValueError('DT-MIL requires finite aligned full spatial maps and nonempty boolean padding masks')
        self.reserve(len(x))
        logits=self.network(sample)
        if logits.shape!=(len(x),len(self.bundle.classes)) or not torch.isfinite(logits).all():
            raise ValueError('DT-MIL returned invalid full model logits')
        result={'logits':logits,'probabilities':logits.softmax(-1),'maps':metadata}
        if targets is not None:
            if targets.shape!=(len(x),) or targets.dtype!=torch.long or (targets<0).any() or (targets>=len(self.bundle.classes)).any():
                raise ValueError('DT-MIL requires aligned in-range integer class targets')
            result['loss']=self.criterion(logits,targets)
        return result

    def forward(self,x,coords=None,label=None):
        value,mask,metadata=encode_map(x,coords,self.assets.stride,self.settings['map_size'],
            self.profile==REPAIRED,self.training,max_bytes=self.settings['max_dense_bytes'])
        sample=types.SimpleNamespace(tensors=value[None].to(x.device),mask=mask[None].to(x.device))
        return self.batch_forward(sample,[metadata],label)

    def forward_unit(self,bundle,rows,device,config,train=False):
        if len(rows)!=1 or config.get('max_instances') is not None:
            raise ValueError('DT-MIL cannot combine unrelated physical slide frames or generic patch truncation')
        sample,meta=self.inputs(rows,device,train)
        return self.batch_forward(sample,meta)

    def epoch(self,bundle,optimizer,device):
        rows=bundle.records('train')
        loader=DataLoader(rows,batch_size=self.settings['batch_size'],sampler=RandomSampler(rows),
                          num_workers=0,collate_fn=list,drop_last=False)
        losses,order,audits=[],[],[]
        for batch in loader:
            sample,metadata=self.inputs(batch,device,True)
            targets=torch.tensor([r['target'] for r in batch],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True)
            output=self.batch_forward(sample,metadata,targets)
            output['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.gradient_clip,error_if_nonfinite=True)
            check_update_finite(self.network,optimizer);optimizer.step();check_update_finite(self.network,optimizer)
            self.optimizer_steps+=1
            order.extend(r['slide_id'] for r in batch)
            audits.extend({'slide_id':row['slide_id'],**meta} for row,meta in zip(batch,metadata))
            losses.append({'loss':float(output['loss'].detach()),'gradient_norm':float(norm),
                           **{g['group_name']+'_lr':g['lr'] for g in optimizer.param_groups}})
        if not losses:
            raise ValueError('DT-MIL needs at least one full TRAIN minibatch')
        self.train_orders.append(order);self.last_epoch_audits=audits
        return losses

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        self.eval();rows=bundle.records(subset);records=[]
        loader=DataLoader(rows,batch_size=self.settings['batch_size'],sampler=SequentialSampler(rows),
                          num_workers=0,collate_fn=list,drop_last=False)
        for batch in loader:
            sample,metadata=self.inputs(batch,device,False)
            output=self.batch_forward(sample,metadata)
            for row,prob,meta in zip(batch,output['probabilities'],metadata):
                entry={k:row[k] for k in ['slide_id','patient_id','global_patient_id','subset','target']}
                entry.update(source_slides=row['slide_id'],dtmil_original_rows=meta['original_rows'],
                             dtmil_retained_rows=meta['retained_rows'],dtmil_retained_ids_sha256=meta['retained_ids_sha256'])
                entry.update({f'p_{c}':float(p) for c,p in enumerate(prob)})
                records.append(entry)
        return pd.DataFrame(records)

    def get_extra_state(self):
        return {'contract':self.contract,'total_epochs':self.total_epochs,'epoch_index':self.epoch_index,
            'completed_epochs':self.completed_epochs,'optimizer_steps':self.optimizer_steps,
            'train_orders':self.train_orders,'last_epoch_audits':self.last_epoch_audits,
            'scheduler':self.scheduler.state_dict() if self.scheduler is not None else self.saved_scheduler,
            'gradient_clip':self.gradient_clip}

    def set_extra_state(self,state):
        if not isinstance(state,dict) or set(state)!=set(self.get_extra_state()) or state.get('contract')!=self.contract:
            raise ValueError('DT-MIL physical input/model/reference recipe contract differs')
        if self.total_epochs and self.total_epochs!=state['total_epochs']:
            raise ValueError('DT-MIL complete LR schedule horizon differs')
        count=len(self.bundle.records('train'))
        for name in ['total_epochs','completed_epochs','optimizer_steps']:
            if type(state[name]) is not int or state[name]<0:
                raise ValueError('DT-MIL checkpoint counters invalid')
        complete=state['completed_epochs']
        if (complete>state['total_epochs'] or type(state['epoch_index']) is not int or state['epoch_index']!=complete-1
                or state['optimizer_steps']!=complete*math.ceil(count/self.settings['batch_size'])):
            raise ValueError('DT-MIL complete minibatch/epoch counts inconsistent')
        orders=state['train_orders'];allowed={r['slide_id'] for r in self.bundle.records('train')}
        if (not isinstance(orders,list) or len(orders)!=complete
                or any(not isinstance(o,list) or len(o)!=count or set(o)!=allowed for o in orders)):
            raise ValueError('DT-MIL checkpoint TRAIN identity/order differs')
        audits=state['last_epoch_audits']
        if (not isinstance(audits,list) or complete and
                (len(audits)!=count or [r.get('slide_id') for r in audits]!=orders[-1])):
            raise ValueError('DT-MIL checkpoint physical crop audit does not match TRAIN order')
        scheduler=state['scheduler']
        if complete and (not isinstance(scheduler,dict) or scheduler.get('last_epoch')!=complete
                or scheduler.get('step_size')!=self.settings['lr_drop'] or scheduler.get('gamma')!=.1):
            raise ValueError('DT-MIL checkpoint complete StepLR state inconsistent')
        clip=finite(state['gradient_clip'],'checkpoint gradient_clip')
        if clip<=0 or self.total_epochs and clip!=self.gradient_clip:
            raise ValueError('DT-MIL checkpoint gradient clipping differs')
        for name in ['total_epochs','epoch_index','completed_epochs','optimizer_steps','train_orders','last_epoch_audits','gradient_clip']:
            setattr(self,name,state[name])
        self.saved_scheduler=scheduler
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler)

    def protocol(self):
        return {'method':self.profile,'family':'dtmil','source_commit':COMMIT,'settings':self.settings,
            'complete_model':'Original BN/1x1/BN/LeakyReLU/1x1 full-D map backbone, sineposition, deformable encoder and ordinary self/cross-attention decoder, concatenated query head and released CE',
            'operator':'Exact pinned author ms_deform_attn_core_pytorch; all mathematical operations retained, CPU/CUDA Torch; compiled author extension not claimed',
            'padding_port':'Only zero valid ratios replaced by1 in single-level model; positive ratios unchanged; normalized reference grid otherwise identical. Prevents NaNs from source symmetric padding.',
            'crop':'original stale-height second crop' if self.profile==SOURCE else 'current-height second crop correction',
            'labels':'source binary label>0 on registered0/1' if self.profile==SOURCE else 'explicit C-way full label extension',
            'physical_geometry':'Actual aligned coordinates/stride, zero interior holes, center crop60 and symmetric padding; no arbitrary bag reshaping',
            'cached_views':self.assets.cached,'cached_view_policy':'Actual author TRAIN-only stochastic views; VAL/TEST registered val vectors',
            'augmentation':'Exact source NumPy flipH/V, scale/crop/pad, cutout androt90, TRAIN only',
            'unpublished_recipe':'Public main.py andengine.py are inference-only. Complete trainer reconstructed explicitly; no historic paper-training equivalence claimed.',
            'optimizer':'Reference AdamW: main2e-4, backbone2e-5, sampling_offsets/reference_points multiplier.1; weight_decay1e-4, globalclip.1; exact actual overrides recorded',
            'scheduler':'Reference StepLR after everyepoch, step_size40 gamma.1; full50epochs, no invented earlystop',
            'sampling':'Reference full TRAIN RandomSampler, actual batch20, keep final short batch; synchronous transforms, no claimed original train-prefetch order',
            'selection':'Benchmark first strict best VAL log_loss default or explicitly registered macro_auc; TEST only after selected checkpoint load',
            'unused_states':'Released transformer.reference_points linear is unused by ordinary decoder; preserve it, do not invent a loss',
            'input_contract':self.assets.contract,'completed_epochs':self.completed_epochs,'optimizer_steps':self.optimizer_steps,
            'train_orders_sha256':digest_object(self.train_orders),'checkpoint':'Full coherent model/BN/AdamW/groups/StepLR/TRAIN order/crop digests/counters/allRNG; source partial eval checkpoints not treated as resume',
            'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}

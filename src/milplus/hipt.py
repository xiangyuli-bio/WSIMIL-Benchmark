"""Complete original pretrained HIPT classifier and balanced native training recipe."""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .hipt_data import HIPTPyramid
from .hipt_source import COMMIT, PROFILES, TENSOR_SHA, author_definitions, original_optimizer
from .io import digest_file, digest_object


class HIPTMIL(nn.Module):
    owns_sampling=True
    supports_instance_outputs=True
    instance_score_semantics='Product of native global region pooling and mean-head local CLS-to-patch attention; local CLS self-attention is not renormalized away. Not a causal attribution or complete Transformer rollout.'

    def __init__(self,bundle,profile="hipt_lgp_frozen_source",early_stopping=False):
        super().__init__()
        if profile not in PROFILES or type(early_stopping) is not bool:
            raise ValueError('HIPT freezing and early stopping must be explicit booleans')
        freeze_4k=PROFILES[profile]
        self.bundle=bundle;self.profile=profile;self.assets=HIPTPyramid(bundle)
        self.freeze_4k=freeze_4k;self.early_stopping=early_stopping
        self.stop_after_epoch=51 if early_stopping else 0
        self.optimizer_steps=0;self.completed_epochs=0;self.epochs=20;self.trace=None
        factory,_=author_definitions()
        self.network=factory(n_classes=len(bundle.classes),freeze_4k=freeze_4k)
        self.network.local_vit.load_state_dict(self.assets.backbone,strict=True)
        self.network.local_vit.blocks[-1].attn.register_forward_hook(self._capture_local)
        self.network.global_attn_pool.register_forward_hook(self._capture_global)
        self.contract=digest_object({'pyramid':self.assets.contract,'teacher':TENSOR_SHA,
            'freeze_4k':freeze_4k,'early_stopping':early_stopping,'classes':bundle.classes})

    def _capture_local(self,module,inputs,output):
        # Copy only the CLS row: retaining its view would retain the entire R*H*257^2 storage.
        self.local_attention=output[1][:,:,0,:].detach().clone()

    def _capture_global(self,module,inputs,output):
        self.global_attention=output[0].detach().transpose(0,1).softmax(1)

    def configure_training(self,cfg):
        if cfg.get('max_instances') is not None or cfg.get('grad_clip') is not None:
            raise ValueError('Native HIPT requires complete physical regions and unclipped source gradients')
        self.epochs=cfg['epochs']

    def make_optimizer(self,lr,weight_decay):
        return original_optimizer(self.network,lr,weight_decay)

    def forward_unit(self,bundle,rows,device,config,train=False):
        if len(rows)!=1:raise ValueError('HIPT requires one native physical WSI pyramid')
        sequence,coords,origins,order=self.assets.load(rows[0])
        result=self(sequence.to(device))
        if not train:self.trace={'coords':coords,'region_origins':origins,'row_order':order}
        return result

    def forward(self,sequence):
        if sequence.ndim!=3 or sequence.shape[1:]!=(256,384) or not len(sequence) or sequence.dtype!=torch.float32 or not torch.isfinite(sequence).all():
            raise ValueError('HIPT requires finite [regions,256,384] native complete pyramid tokens')
        logits,probability,*_=self.network(sequence)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite native HIPT logits')
        return {'logits':logits,'probabilities':probability,
            'region_attention':self.global_attention,'local_cls_attention':self.local_attention}

    def prediction_fields(self,result,row):
        target=torch.tensor([row['target']],device=result['logits'].device)
        return {'hipt_source_loss':float(F.cross_entropy(result['logits'],target))}

    def validation_metrics(self,frame):
        return {'hipt_val_loss':sum(map(float,frame.hipt_source_loss))/len(frame)}

    def selection_objective(self,metrics,epoch,current):
        return -metrics['hipt_val_loss']

    def epoch(self,bundle,optimizer,device='cpu'):
        self.train();rows=bundle.records('train')
        counts=np.bincount([r['target'] for r in rows],minlength=len(bundle.classes))
        if (counts==0).any():raise ValueError('HIPT inverse-class sampling requires every TRAIN class')
        weights=torch.tensor([len(rows)/counts[r['target']] for r in rows],dtype=torch.float64)
        sampler=WeightedRandomSampler(weights,len(rows),replacement=True)
        loader=DataLoader(rows,batch_size=1,sampler=sampler,num_workers=0,collate_fn=lambda batch:batch[0])
        losses=[]
        for row in loader:
            output=self.forward_unit(bundle,[row],device,{},train=True)
            raw=F.cross_entropy(output['logits'],torch.tensor([row['target']],device=device));loss=raw/32
            if not torch.isfinite(loss):raise ValueError('Nonfinite HIPT objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise ValueError('Nonfinite HIPT gradient')
            optimizer.step();optimizer.zero_grad();self.optimizer_steps+=1
            losses.append({'loss':float(raw.detach()),'optimization_loss':float(loss.detach())})
        return losses

    def finish_epoch(self,bundle,device,cfg):self.completed_epochs+=1

    def selection_improved(self,objective,best):
        return objective>=best if self.early_stopping else True

    def get_extra_state(self):
        return {'contract':self.contract,'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs}

    def set_extra_state(self,state):
        if set(state)!={'contract','optimizer_steps','completed_epochs'} or state['contract']!=self.contract or any(
                type(state[k]) is not int or state[k]<0 for k in ['optimizer_steps','completed_epochs']):
            raise ValueError('HIPT checkpoint input/pretrained/freezing contract differs')
        self.optimizer_steps=state['optimizer_steps'];self.completed_epochs=state['completed_epochs']

    def export_instances(self,result,row,directory):
        if self.training or self.trace is None:
            raise ValueError('HIPT attention export requires deterministic inference of a registered physical pyramid')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        path=directory/(digest_object(row['slide_id'])[:24]+'.h5')
        order=self.trace['row_order'];local=result['local_cls_attention'].cpu().numpy()
        global_attention=result['region_attention'].cpu().numpy().reshape(-1)
        cls_mean=local.mean(1)
        product=(global_attention[:,None]*cls_mean[:,1:]).reshape(-1)
        original=np.empty(len(order)*256,dtype=product.dtype);original[order.reshape(-1)]=product
        with h5py.File(path,'w') as h:
            h['coords']=self.trace['coords'];h['region_origins']=self.trace['region_origins']
            h['canonical_region_rows']=order;h['region_attention']=global_attention
            h['local_cls_attention_all_heads']=local;h['hierarchical_attention_product']=original
            h.attrs['semantics']=self.instance_score_semantics
            h.attrs['initial_teacher_tensor_sha256']=TENSOR_SHA;h.attrs['profile']=self.profile
        return {'slide_id':row['slide_id'],'path':str(path),'sha256':digest_file(path),
            'instances':len(original),'regions':len(order),'patch_attention_mass':float(product.sum()),
            'cls_self_attention_mass':float((global_attention*cls_mean[:,0]).sum())}

    def protocol(self):
        return {'profile':self.profile,'source_commit':COMMIT,'pyramid_contract':self.assets.contract,
            'pretrained_teacher_tensor_sha256':TENSOR_SHA,'freeze_local_vit':self.freeze_4k,
            'patch_encoder':'native HIPT ViT256 DINO384, exact source embedding-space declaration required',
            'hierarchy':'complete physical 16x16 patch grid per region, native6-block192-wide local ViT and two-layer192 global Transformer',
            'objective':'original CE/32 with an Adam update on EVERY sampled bag; not gradient accumulation',
            'sampling':'source N/class_count inverse-frequency replacement sampler; native DataLoader base-seed consumption, CPU deterministic readers',
            'optimizer':'original source Adam helper, default lr2e-4/weight_decay1e-5; no scheduler or gradient clipping',
            'selection':'validation slide CE, last tied best and source stop-after-epoch51' if self.early_stopping else 'source default FINAL epoch; best.pt denotes the selected final checkpoint',
            'epochs':self.epochs,'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,
            'source_boundaries':['pretrained local weights loaded explicitly and all backbone keys checked instead of hardcoded relative path/strict=False',
                'safe weights-only conversion of actual public teacher; legacy metadata isolated from caller allowlists, no default random fallback',
                'TEST sealed until selected checkpoint, exact optimizer/sampler/dropout/RNG restore',
                'classification source available; original survival Python was not published at this commit and is not replaced by a generic head'],
            'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}


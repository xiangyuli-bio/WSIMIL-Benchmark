"""Full RGB prompt tuning with the author's three-pass retained-feature gradients."""
import copy
import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score

from .promptmil_source import COMMIT,WEIGHTS,build_original

RECIPES={}
for condition in ['brca','bright','crc']:
    for edition in ['source','paper_reference']:
        RECIPES[f'promptmil_{condition}_dino_dsmil_{edition}']={
            'condition':condition,'edition':edition,'epochs':40 if edition=='paper_reference' or condition=='crc'else 50,
            'lr':5e-4 if condition=='crc'and edition=='paper_reference'else 1e-4,
            'weight_decay':.05 if condition=='bright'else.01,'adam':condition=='bright',
            'loss':'CE'if condition=='bright'else'onehot_BCE',
            'native_classes':3 if condition=='bright'else 2,
            'train_chunk':256 if condition=='crc'else 512,'eval_augmentation_chunk':512 if condition=='crc'else 1024,
            'accumulation':8,'flush_tail':edition=='paper_reference',
            'selection':'promptmil_source_AUROC'if edition=='paper_reference'else'promptmil_source_completed_epoch'}


def check_inputs(bundle,profile=None):
    if (bundle.task_type!='classification'or bundle.config.get('training_unit','slide')!='slide'
            or getattr(bundle,'promptmil',None)is None):
        raise ValueError('Prompt-MIL requires prepared actual RGB224 and original pretrained DINO; frozen FM vectors cannot substitute')
    if profile is not None:
        recipe=RECIPES[profile]
        if bundle.promptmil.condition!=recipe['condition']:raise ValueError('Prompt-MIL source cohort backbone does not match the actual bound weights')
        if not recipe['flush_tail'] and len(bundle.records('train'))<8:
            raise ValueError('Original Prompt-MIL updates every8slide batches; fewer than8TRAIN slides would never step. Use its declared tail-flush reference')
    return bundle.promptmil


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Slides(Dataset):
    def __init__(self,model,subset):self.model=model;self.rows=model.bundle.records(subset)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        return self.model.assets.pixels(row),int(row['target']),row['slide_id']


class PromptMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile,deep_prompt=False):
        super().__init__()
        if profile not in RECIPES:raise ValueError('Select an explicit original Prompt-MIL cohort/recipe')
        self.assets=check_inputs(bundle,profile);self.bundle=bundle;self.profile=profile;self.recipe=copy.deepcopy(RECIPES[profile])
        self.deep_prompt=deep_prompt;self.classes=len(bundle.classes)
        self.backbone,self.classifier,functions=build_original(self.assets.weights,self.assets.condition,self.classes,deep_prompt)
        self.classifier_function=functions.dsmil_forward_ce if self.recipe['loss']=='CE'else functions.dsmil_forward
        self.loss_function=nn.CrossEntropyLoss()if self.recipe['loss']=='CE'else nn.BCEWithLogitsLoss()
        self.current_epoch=0;self.optimizer_updates=0;self.pending_bags=0;self.actual_bags=0
        self.scheduler=None;self.pending_scheduler=None;self.training_cfg=None
        self.draw_history=[];self.epoch_history=[]

    def configure_training(self,cfg):
        if type(cfg['epochs'])is not int or not 1<=cfg['epochs']<=self.recipe['epochs']:
            raise ValueError('Prompt-MIL needs a positive explicit budget within its complete native recipe')
        if cfg.get('max_instances')is not None or cfg.get('grad_clip')is not None:
            raise ValueError('Original Prompt-MIL retains every RGB patch and has no gradient clipping')
        self.training_cfg=copy.deepcopy(cfg)

    def make_optimizer(self,lr,weight_decay):
        groups=[{'params':[p for p in self.backbone.parameters()if p.requires_grad],'lr':lr},
                {'params':[p for p in self.classifier.parameters()if p.requires_grad]}]
        cls=torch.optim.Adam if self.recipe['adam']else torch.optim.AdamW
        options={'betas':(.5,.9)}if self.recipe['adam']else{}
        optimizer=cls(groups,lr=lr,weight_decay=weight_decay,**options)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.training_cfg['epochs'],eta_min=5e-6)
        if self.pending_scheduler is not None:
            self.scheduler.load_state_dict(self.pending_scheduler);self.pending_scheduler=None
        return optimizer

    def start_epoch(self,epoch):self.current_epoch=epoch

    def get_extra_state(self):
        return {'schema_version':1,'profile':self.profile,'input_contract':self.assets.contract,
            'epoch':self.current_epoch,'optimizer_updates':self.optimizer_updates,'pending_bags':self.pending_bags,
            'actual_bags':self.actual_bags,'scheduler':copy.deepcopy(self.scheduler.state_dict())if self.scheduler is not None else self.pending_scheduler,
            'pending_gradients':{n:None if p.grad is None else p.grad.detach().cpu().clone()
                                 for n,p in self.named_parameters()if p.requires_grad},
            'draw_history':copy.deepcopy(self.draw_history),'epoch_history':copy.deepcopy(self.epoch_history)}

    def set_extra_state(self,state):
        if state.get('schema_version')!=1 or state.get('profile')!=self.profile or state.get('input_contract')!=self.assets.contract:
            raise ValueError('Prompt-MIL pending gradients belong to a different actual RGB/teacher/task/recipe')
        self.current_epoch=state['epoch'];self.optimizer_updates=state['optimizer_updates'];self.pending_bags=state['pending_bags']
        self.actual_bags=state['actual_bags'];self.draw_history=copy.deepcopy(state['draw_history']);self.epoch_history=copy.deepcopy(state['epoch_history'])
        parameters={n:p for n,p in self.named_parameters()if p.requires_grad}
        if parameters.keys()!=state['pending_gradients'].keys():raise ValueError('Prompt-MIL exact pending gradient inventory changed')
        for name,p in parameters.items():
            value=state['pending_gradients'][name]
            if value is not None and (value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all()):
                raise ValueError('Invalid pending Prompt-MIL gradient: '+name)
            p.grad=None if value is None else value.to(p.device).clone()
        if self.scheduler is None:self.pending_scheduler=copy.deepcopy(state['scheduler'])
        elif state['scheduler']is not None:self.scheduler.load_state_dict(state['scheduler'])

    def split_tensor(self,value):
        # Literal author MilE2EModule.split_tensor: balanced torch.chunk, not
        # fixed limit/limit/tail slices. Evaluation uses the TRAIN chunk too.
        return torch.chunk(value,int(np.ceil(value.shape[0]/self.recipe['train_chunk'])),dim=0)

    def backbone_forward(self,pixels):
        with torch.no_grad():return torch.cat([self.backbone(x)for x in self.split_tensor(pixels)],dim=0)

    def train_bag(self,pixels,target,optimizer,local_batch_index):
        if pixels.ndim!=4 or pixels.shape[1:]!=(3,224,224)or not len(pixels):
            raise ValueError('Prompt-MIL needs every actual full224RGB patch of a nonempty WSI')
        # Three complete source passes. No pretrained feature cache substitutes
        # for either RGB forward or the prompt-gradient recomputation.
        features=self.backbone_forward(pixels);features.requires_grad=True
        logits,loss,_=self.classifier_function(features,self.classifier,self.loss_function,self.classes,label=target)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Prompt-MIL dual-stream loss')
        loss.backward()
        if features.grad is None or not torch.isfinite(features.grad).all():raise ValueError('Prompt-MIL retained feature gradients are missing/nonfinite')
        for x,g in zip(self.split_tensor(pixels),self.split_tensor(features.grad)):
            new_features=self.backbone(x)
            new_features.backward(g)
        if any(p.grad is not None and not torch.isfinite(p.grad).all()for p in self.parameters()):
            raise FloatingPointError('Prompt-MIL prompt/classifier gradients became nonfinite')
        if any(p.grad is not None for p in self.backbone.parameters()if not p.requires_grad):
            raise ValueError('Prompt-MIL must not fine-tune the original frozen backbone')
        self.pending_bags+=1;self.actual_bags+=1
        stepped=(local_batch_index+1)%8==0
        if stepped:
            optimizer.step();optimizer.zero_grad();self.optimizer_updates+=1;self.pending_bags=0
        return {'loss':float(loss.detach()),'actual_patches':float(len(pixels)),
                'gradient_cached_chunks':float(len(self.split_tensor(pixels))),'optimizer_step':float(stepped),
                'pending_unnormalized_slide_gradients':float(self.pending_bags)},tensor_hash(features),logits.detach()

    def epoch(self,bundle,optimizer,device):
        self.train();loader=DataLoader(Slides(self,'train'),batch_size=1,shuffle=True,num_workers=0,drop_last=False)
        terms=[]
        for i,(pixels,target,sid)in enumerate(loader):
            value,feature_sha,_=self.train_bag(pixels.squeeze(0).to(device),target.to(device),optimizer,i)
            terms.append(value);item=self.assets.rows[sid[0]]
            self.draw_history.append({'epoch':self.current_epoch,'local_batch_index':i,'slide_id':sid[0],
                'actual_pixels':item['shape'][0],'ordered_patch_ids_sha256':item['ordered_patch_ids_sha256'],
                'temporary_feature_sha256':feature_sha,'optimizer_updates':self.optimizer_updates,'pending_bags':self.pending_bags})
        return terms

    def finish_epoch(self,bundle,device,cfg):
        flushed=False
        if self.recipe['flush_tail']and self.pending_bags:
            self.scheduler.optimizer.step();self.scheduler.optimizer.zero_grad();self.optimizer_updates+=1;self.pending_bags=0;flushed=True
        self.scheduler.step()
        self.epoch_history.append({'epoch':self.current_epoch,'actual_bags':self.actual_bags,'optimizer_updates':self.optimizer_updates,
            'pending_bags':self.pending_bags,'tail_flushed':flushed,'learning_rates':self.scheduler.get_last_lr()})

    def forward(self,*args,**kwargs):
        raise ValueError('Prompt-MIL requires its actual raw-RGB three-pass recipe; frozen vector forward is unsupported')

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        self.eval();records=[]
        rows={row['slide_id']:row for row in bundle.records(subset)}
        loader=DataLoader(Slides(self,subset),batch_size=1,shuffle=False,num_workers=0,drop_last=False)
        for pixels,target,sid in loader:
            row=rows[sid[0]];pixels=pixels.squeeze(0).to(device);features=self.backbone_forward(pixels)
            target=target.to(device)
            logits,loss,scores=self.classifier_function(features,self.classifier,self.loss_function,self.classes,label=target)
            prob=logits.softmax(-1)[0]
            record={k:row[k]for k in ['slide_id','patient_id','global_patient_id','target']}
            record.update(subset=subset,source_slides=row['slide_id'],source_loss=float(loss))
            record.update({f'p_{i}':float(x)for i,x in enumerate(prob)})
            record.update({f'promptmil_source_score_{i}':float(x)for i,x in enumerate(scores[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self,frame):
        columns=[f'promptmil_source_score_{i}'for i in range(self.classes)]
        unit=self.bundle.config.get('evaluation_unit','slide')
        if unit=='patient':
            if (frame.groupby('global_patient_id').target.nunique()>1).any():raise ValueError('Inconsistent patient labels for Prompt-MIL fusion')
            frame=frame.groupby('global_patient_id',sort=True).agg({**{c:'mean'for c in columns},'target':'first','source_loss':'mean'}).reset_index()
        scores=frame[columns].to_numpy(dtype=float);targets=frame.target.to_numpy(dtype=int)
        if not np.isfinite(scores).all()or(scores<0).any()or(scores>1).any():raise ValueError('Invalid original Prompt-MIL Sigmoid scores')
        auc=float(np.mean([roc_auc_score(targets==c,scores[:,c])for c in range(self.classes)]))if len(np.unique(targets))==self.classes else None
        return {'promptmil_source_AUROC':auc,'promptmil_source_accuracy':float(np.mean(scores.argmax(1)==targets)),
            'promptmil_source_loss':float(frame.source_loss.mean()),'promptmil_source_completed_epoch':float(self.current_epoch),
            'promptmil_source_score_semantics':'original_independent_Sigmoid_columns_macro_OVR_AUC_no_row_normalization',
            'promptmil_standard_probability_semantics':'softmax_logits_for_shared_probability_metrics_separately_labelled'}

    def protocol(self):
        return {'method':'Prompt-MIL','source_commit':COMMIT,'profile':self.profile,'recipe':self.recipe,
            'input':'actual_all_RGB224_patches_uint8_div255_no_cached_FM_vector','original_weights_sha256':WEIGHTS[self.assets.condition]['sha256'],
            'native_embedding':192,'native_depth':12,'native_heads':3,'native_patch_size':16,'actual_image_pixels':224,
            'pooling':'author_DINO_global_average_of_nonprompt_tokens','prompt_tokens':1,'deep_prompt':self.deep_prompt,
            'prompt_dropout':0.,'feature_forward':'no_grad_full_RGB_balanced_chunks','retained_gradient':'exact_author_dual_DSMIL_bag_and_max_half_weight',
            'prompt_backward':'recompute_every_RGB_chunk_then_backward_with_retained_feature_gradient',
            'full_original_pretrained_backbone_frozen':True,'actual_trainable_parameters':[n for n,p in self.named_parameters()if p.requires_grad],
            'gradient_accumulation':8,'accumulated_loss_normalization':'source_unnormalized_sum_not_divided_by8',
            'source_epoch_tail':'unstepped_gradient_carried_to_next_epoch_and_ignored_if_final_epoch',
            'reference_epoch_tail':'flush_last_partial_minibatch'if self.recipe['flush_tail']else'source_tail_preserved',
            'source_prediction':'independent_Sigmoid_columns_even_for_CE','shared_probability_export':'separate_softmax_logits',
            'source_selection':'passed_model_final_epoch_not_best_checkpoint','reference_selection':'validation_source_macro_OVR_AUC'if self.recipe['edition']=='paper_reference'else'original_final_epoch',
            'source_eval_backbone_chunk':'train_chunk_field_used_in_original_backbone_forward_even_at_eval',
            'reference_choices':['patient_mean_score_fusion_if_requested','validation_selection_and_tail_flush_for_paper_reference',
                'zero_loader_workers_for_exact_resume','packed_actual_RGB_byte_contract',
                'omit_original_per_epoch_test_monitor_loader_to_preserve_locked_test_boundary'],
            'actual_requested_training_config':copy.deepcopy(self.training_cfg),'optimizer_updates':self.optimizer_updates,
            'actual_training_bags':self.actual_bags,'pending_unnormalized_slide_gradients':self.pending_bags,
            'formal_admission':False,'real_WSI_experiments':False,'paper_score_reproduction':False}

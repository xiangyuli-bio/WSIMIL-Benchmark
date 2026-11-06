"""Complete repaired LPGMIL: two native SSMs, six masked prototype branches,
source dual cross-entropy and all-parameter L1, with immutable TRAIN assets.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .lpgmil_source import COMMIT, native_network

PROFILES={'lpgmil_tls1024_reference','lpgmil_fm_width_reference'}


def check_inputs(bundle,profile):
    if profile not in PROFILES or bundle.task_type!='classification':
        raise ValueError('LPGMIL currently supports its explicitly named classification references')
    if bundle.config.get('training_unit','slide')!='slide':
        raise ValueError('LPGMIL trains one original WSI bag per step')
    if profile=='lpgmil_tls1024_reference' and (bundle.dim!=1024 or len(bundle.classes)!=2):
        raise ValueError('Native LPGMIL TLS reference needs1024D features and two classes; select the explicit FM/class port')
    inputs=getattr(bundle,'lpgmil',None)
    if inputs is None:raise ValueError('Run prepare-lpgmil with real TRAIN nucleus counts and detector provenance')
    inputs.verify()
    return inputs


class _Slides(Dataset):
    def __init__(self,bundle):self.bundle=bundle;self.rows=bundle.records('train')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];values,_=self.bundle.load(row)
        return values,int(row['target']),row['slide_id']


class LPGMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile,scan_backend='torch',scan_chunk=64,l1_coefficient=1e-5):
        super().__init__()
        if type(l1_coefficient)not in(int,float)or l1_coefficient!=1e-5:
            raise ValueError('Complete LPGMIL retains the released all-parameter L1 coefficient1e-5')
        self.inputs=check_inputs(bundle,profile);self.profile=profile
        self.dim,self.classes=bundle.dim,len(bundle.classes)
        network,self.l1,self.backend_identity=native_network(self.dim,scan_backend,scan_chunk)
        self.native=network(in_dim=self.dim,num_classes=self.classes,dropout=.25,activation='relu',
            num_layers=2,num_masked_patches=10,num_tokens=6,mask_drop_rate=.6,
            model_type='Mamba',use_topk=True,prototypes=self.inputs.prototypes.clone())
        self.l1_coefficient=float(l1_coefficient)
        self.current_epoch,self.updates=0,0;self.draw_history=[];self.training_cfg=None

    def configure_training(self,config):
        if config.get('max_instances')is not None:
            raise ValueError('LPGMIL keeps every input patch before native top1000 selection')
        if config.get('grad_clip')is not None:
            raise ValueError('Source LPGMIL has no gradient clipping; declare grad_clip:null')
        self.training_cfg=copy.deepcopy(config)

    def make_optimizer(self,lr,weight_decay):
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)

    def start_epoch(self,epoch):self.current_epoch=epoch

    def selection_objective(self,metrics,epoch,default):
        # Source default disables early stopping and tests the final epoch.
        return float(epoch)

    def get_extra_state(self):
        return {'schema_version':1,'profile':self.profile,'inputs':self.inputs.identity,
                'backend':self.backend_identity,'current_epoch':self.current_epoch,
                'updates':self.updates,'draw_history':copy.deepcopy(self.draw_history)}

    def set_extra_state(self,state):
        if state.get('schema_version')!=1 or state.get('profile')!=self.profile or state.get('inputs')!=self.inputs.identity:
            raise ValueError('LPGMIL checkpoint belongs to different TRAIN prototypes or condition')
        if state['backend']!=self.backend_identity:raise ValueError('LPGMIL numerical backend changed')
        self.current_epoch,self.updates=state['current_epoch'],state['updates']
        self.draw_history=copy.deepcopy(state['draw_history'])

    def forward(self,features,coords=None):
        if features.ndim!=2 or features.shape[1]!=self.dim or not len(features)or not torch.isfinite(features).all():
            raise ValueError('LPGMIL needs every actual finite ordered instance in its registered FM space')
        logits,prob,hat,attention,tokens,combined=self.native(features)
        if logits.shape!=(1,self.classes)or tokens.shape!=(6,1,self.classes):
            raise ValueError('Native LPGMIL bag or six-branch output contract changed')
        return {'logits':logits,'token_logits':tokens[:,0,:],'attention':attention,
                'source_probabilities':prob,'source_prediction':hat,'combined_attention':combined}

    def forward_unit(self,bundle,rows,device,config,train=False):
        if len(rows)!=1:raise ValueError('LPGMIL training is one source slide per bag')
        values,_=bundle.load(rows[0]);return self(values.to(device))

    def objective(self,result,target):
        bag=F.cross_entropy(result['logits'],target.reshape(1))
        tokens=F.cross_entropy(result['token_logits'],target.reshape(1).repeat_interleave(6))
        regularizer=self.l1(self.native)*self.l1_coefficient
        return bag+tokens+regularizer,{'bag_ce':bag,'six_branch_ce':tokens,'all_parameter_l1':regularizer}

    def train_bag(self,features,target,optimizer):
        result=self(features);loss,parts=self.objective(result,target)
        if not torch.isfinite(loss):raise FloatingPointError('LPGMIL source loss is nonfinite')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.updates+=1
        return {'loss':float(loss.detach()),**{k:float(v.detach())for k,v in parts.items()}}

    def epoch(self,bundle,optimizer,device):
        self.train();self.inputs.verify()
        # Same Torch random sampler and1-slide default as author;0workers makes
        # the complete draw/dropout/masking RNG state reproducible on resume.
        loader=DataLoader(_Slides(bundle),batch_size=1,shuffle=True,num_workers=0)
        losses=[]
        for value,target,sid in loader:
            self.draw_history.append({'epoch':self.current_epoch,'slide_id':sid[0],
                                      'actual_patches':int(value.shape[1])})
            losses.append(self.train_bag(value[0].to(device),target.to(device),optimizer))
        return losses

    def protocol(self):
        return {'method':'LPGMIL','profile':self.profile,'source_commit':COMMIT,
            'native_hidden':512,'native_SSM_layers':2,'native_state':16,'native_conv':4,'native_expand':2,
            'native_heads':8,'native_prototypes':6,'native_topk':1000,
            'masked_top':10,'masked_drop_fraction':.6,'attention_dropout':.1,'feature_dropout':.25,
            'actual_feature_dimension':self.dim,'FM_port':self.dim!=1024,'actual_classes':self.classes,
            'backend':self.backend_identity,'prototype_identity':self.inputs.identity,
            'prototype_reference_choices':self.inputs.manifest['reference_choices'],
            'repairs':['released_constructor_keyword_and_six_output_trainer_contract',
                'same_published_projection_applied_to_learnable_raw_prototype_queries',
                'frozen_selector_registered_as_buffer_for_device_state_and_checkpoint',
                'six_branch_stack_squeezed_only_at_single_bag_axis_for_CE'],
            'source_topk_head_policy':'first_head_mean_over_six_queries_preserved_not_silently_averaged_over_heads',
            'hard_topk_gradient_policy':'published_hard_selection_retained; selector-only_parameters_receive_L1_not_task_gradient',
            'initializer':'literal_author_Xavier_including_zeroing_Mamba_dt_bias',
            'loss':'one_bag_CE_plus_mean_six_branch_CE_plus1e-5_all_parameter_L1',
            'optimizer':'Adam_all_parameters_with_source_weight_decay1e-5',
            'scheduler':None,'native_epochs':50,'source_unused_accumulation_steps_4':'no_accumulation_in_actual_loop',
            'selection':'final_epoch_as_source_default_no_early_stopping; thresholds_never_fit_on_test',
            'loader_workers':0,'current_epoch':self.current_epoch,'updates':self.updates,
            'requested_training_config':copy.deepcopy(self.training_cfg),
            'reference_reconstruction':True,'formal_admission':False,'real_WSI_validation':'not_run',
            'paper_score_reproduction':'not_run'}

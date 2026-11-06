"""Complete100+200epoch MExD paper-reference lifecycle and full posterior.

This is an explicit reference reconstruction: the public pretrain and inference
MoE implementations disagree. No private checkpoint or absent trainer is invented.
"""
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from .mexd_data import Inputs
from .mexd_graph import COHORTS,DynamicExperts
from .mexd_source import COMMIT,components,complete_radam,denoiser_config
from .io import digest_file,write_json

PROFILES={'mexd_camelyon16_paper_reference':'camelyon16',
          'mexd_tcga_nsclc_paper_reference':'tcga_nsclc',
          'mexd_bracs_paper_reference':'bracs',
          'mexd_class_port_reference':'class_port'}


def check_inputs(bundle,profile):
    if profile not in PROFILES:raise ValueError('Select a complete MExD cohort/label reference')
    if PROFILES[profile]=='class_port':
        background=bundle.config.get('class_roles',{}).get('background')
        if background is not None and background!=bundle.classes[0]:
            raise ValueError('MExD explicit background class must be classes[0]; relabel the prepared task consistently')
        return Inputs(bundle)
    condition=COHORTS[PROFILES[profile]]
    if len(bundle.classes)!=condition['classes']:
        raise ValueError('MExD cohort reference has a different observed class/expert mapping')
    background=bundle.config.get('class_roles',{}).get('background')
    if condition['observed_background']==0 and background!=bundle.classes[0]:
        raise ValueError('MExD Camelyon reference requires explicit class_roles.background equal to classes[0]')
    if condition['observed_background']is None and background is not None:
        raise ValueError('This MExD cohort treats every observed class as a positive subtype; background mapping differs')
    return Inputs(bundle)


class _Slides(Dataset):
    def __init__(self,bundle):self.bundle=bundle;self.rows=bundle.records('train')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];features,_=self.bundle.load(row)
        return features,int(row['target']),row['slide_id']


class MExD(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile,moe_epochs=100,diffusion_epochs=200,diffusion_lr=1e-3,
                 sampling_seed=2026):
        super().__init__()
        if any(type(v)is not int or v<1 for v in [moe_epochs,diffusion_epochs]):
            raise ValueError('Complete MExD requires positive budgets for BOTH MoE and diffusion stages')
        if type(diffusion_lr)not in(int,float)or not math.isfinite(diffusion_lr)or diffusion_lr<=0:
            raise ValueError('MExD diffusion LR must be finite and positive')
        if type(sampling_seed)is not int or not 0<=sampling_seed<2**63:
            raise ValueError('Declare a nonnegative integral MExD sampling seed')
        self.inputs=check_inputs(bundle,profile);self.profile=profile;self.cohort=PROFILES[profile]
        self.dim,self.classes=bundle.dim,len(bundle.classes)
        self.moe_epochs,self.diffusion_epochs=moe_epochs,diffusion_epochs
        self.diffusion_lr=float(diffusion_lr);self.sampling_seed=sampling_seed
        self.minimum_selection_epoch=moe_epochs
        role=bundle.config.get('class_roles',{}).get('background')
        self.auxiliary=DynamicExperts(bundle.dim,self.cohort,classes=len(bundle.classes),
                                     observed_background=0 if role is not None else None)
        _,noise,self.diffusion,_=components()
        self.noise=noise(denoiser_config(self.classes,self.dim),guidance=True)
        betas=self.diffusion.make_beta_schedule('linear',500,1e-4,.02).float()
        alphas=1-betas;product=alphas.cumprod(0)
        self.register_buffer('betas',betas);self.register_buffer('alphas',alphas)
        self.register_buffer('alpha_bar_sqrt',product.sqrt())
        self.register_buffer('one_minus_alpha_bar_sqrt',(1-product).sqrt())
        self.current_epoch=0;self.phase='moe';self.moe_steps=0;self.diffusion_steps=0
        self.draw_history=[];self.epoch_history=[];self.training_cfg=None
        self.schedulers=None;self.pending_schedulers=None
        self.posterior_root=None

    def bind_output(self,path):
        self.posterior_root=Path(path)/'posterior_draws'

    def configure_training(self,config):
        if config['epochs']!=self.moe_epochs+self.diffusion_epochs:
            raise ValueError('MExD total epochs must equal declared complete MoE+diffusion budgets')
        if config.get('max_instances')is not None:
            raise ValueError('MExD receives every registered original instance before its learned routing')
        if config.get('grad_clip')!=1.:
            raise ValueError('MExD reference keeps active source denoiser clip_norm1; MoE remains unclipped')
        self.training_cfg=copy.deepcopy(config)

    def make_optimizers(self,lr,weight_decay):
        first=complete_radam(self.auxiliary.parameters(),lr,weight_decay)
        second=torch.optim.Adam(self.noise.parameters(),lr=self.diffusion_lr,
                                betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(first,self.moe_epochs,eta_min=0.),
                         torch.optim.lr_scheduler.CosineAnnealingLR(second,self.diffusion_epochs,eta_min=0.)]
        if self.pending_schedulers is not None:
            for scheduler,state in zip(self.schedulers,self.pending_schedulers):scheduler.load_state_dict(state)
            self.pending_schedulers=None
        return [first,second]

    def _set_phase(self,phase):
        if phase not in {'moe','diffusion'}:raise ValueError('Unknown MExD training stage')
        self.phase=phase
        self.auxiliary.requires_grad_(phase=='moe')
        self.noise.requires_grad_(phase=='diffusion')

    def start_epoch(self,epoch):
        self.current_epoch=epoch;self._set_phase('moe'if epoch<self.moe_epochs else'diffusion')

    def finish_epoch(self,bundle,device,config):
        self.schedulers[0 if self.phase=='moe'else 1].step()
        self.epoch_history.append({'epoch':self.current_epoch,'phase':self.phase,
            'moe_steps':self.moe_steps,'diffusion_steps':self.diffusion_steps,
            'next_learning_rates':[s.optimizer.param_groups[0]['lr']for s in self.schedulers]})

    def selection_objective(self,metrics,epoch,default):
        # The original source selects using TEST. This benchmark locks TEST and
        # retains a fixed final-epoch reference; no test-side tuning is retained.
        return float(epoch)

    def get_extra_state(self):
        return {'schema_version':1,'profile':self.profile,'inputs':self.inputs.identity,
            'budgets':[self.moe_epochs,self.diffusion_epochs],'sampling_seed':self.sampling_seed,
            'current_epoch':self.current_epoch,'phase':self.phase,'moe_steps':self.moe_steps,
            'diffusion_steps':self.diffusion_steps,'draw_history':copy.deepcopy(self.draw_history),
            'epoch_history':copy.deepcopy(self.epoch_history),
            'schedulers':[copy.deepcopy(s.state_dict())for s in self.schedulers]if self.schedulers is not None else self.pending_schedulers}

    def set_extra_state(self,state):
        if(state.get('schema_version')!=1 or state.get('profile')!=self.profile or
           state.get('inputs')!=self.inputs.identity or state.get('budgets')!=[self.moe_epochs,self.diffusion_epochs]
           or state.get('sampling_seed')!=self.sampling_seed):
            raise ValueError('MExD checkpoint has different actual features/patients, expert mapping, stages or posterior seed')
        self.current_epoch=state['current_epoch'];self._set_phase(state['phase'])
        self.moe_steps,self.diffusion_steps=state['moe_steps'],state['diffusion_steps']
        self.draw_history=copy.deepcopy(state['draw_history']);self.epoch_history=copy.deepcopy(state['epoch_history'])
        if self.schedulers is None:self.pending_schedulers=copy.deepcopy(state['schedulers'])
        elif state['schedulers']is not None:
            for scheduler,value in zip(self.schedulers,state['schedulers']):scheduler.load_state_dict(value)

    def train_moe(self,features,target,optimizer):
        if self.phase!='moe':raise ValueError('MExD MoE update requested outside its declared stage')
        self.auxiliary.train();self.noise.eval()
        result=self.auxiliary(features);loss,parts=self.auxiliary.loss(result,target)
        if not torch.isfinite(loss):raise FloatingPointError('MExD full MoE objective is nonfinite')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        optimizer.zero_grad(set_to_none=True);self.moe_steps+=1
        return {'loss':float(loss.detach()),**{k:float(v.detach())for k,v in parts.items()},
            'selected_actual_instances':float(result['observed_sparse_instances']),
            'empty_experts':float(result['empty_expert_sentinels'])}

    def train_diffusion(self,features,target,optimizer):
        if self.phase!='diffusion':raise ValueError('MExD diffusion update requires its declared second stage')
        self.auxiliary.eval();self.noise.train()
        with torch.no_grad():result=self.auxiliary(features)
        prior=result['probabilities'];condition=result['condition']
        # Literal public batch1 antithetic sampler, shifted q_sample and MSE.
        t=torch.randint(0,500,(1,),device=features.device)
        t=torch.cat([t,499-t],0)[:1]
        y=F.one_hot(target.reshape(1),num_classes=self.classes).float()
        epsilon=torch.randn_like(y)
        noisy=self.diffusion.q_sample(y,prior,self.alpha_bar_sqrt,self.one_minus_alpha_bar_sqrt,t,noise=epsilon)
        prediction=self.noise(condition,noisy,t,prior)
        loss=(epsilon-prediction).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('MExD full diffusion MSE is nonfinite')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(self.noise.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True);self.diffusion_steps+=1
        return {'loss':float(loss.detach()),'noise_mse':float(loss.detach()),'timestep':float(t.item()),
                'selected_actual_instances':float(result['observed_sparse_instances'])}

    def epoch(self,bundle,optimizers,device):
        loader=DataLoader(_Slides(bundle),batch_size=1,shuffle=True,num_workers=0)
        self.train();losses=[]
        if self.phase=='diffusion':self.auxiliary.eval()
        for features,target,ids in loader:
            features,target=features[0].to(device),target.to(device)
            self.draw_history.append({'epoch':self.current_epoch,'phase':self.phase,'slide_id':ids[0],
                                      'actual_instances':len(features),'ordered_original_features':True})
            function=self.train_moe if self.phase=='moe'else self.train_diffusion
            losses.append(function(features,target,optimizers[0 if self.phase=='moe'else 1]))
        return losses

    @torch.no_grad()
    def posterior(self,features,slide_id):
        self.auxiliary.eval();self.noise.eval()
        result=self.auxiliary(features);prior=result['probabilities']
        if self.phase=='moe':
            return {'probabilities':prior,'raw_draws':None,'sample_probabilities':None,
                    'source_sigmoid_scores':prior.sigmoid(),'stage':'moe'}
        seed=int.from_bytes(hashlib.sha256((str(self.sampling_seed)+'\0'+slide_id+'\0mexd500x100').encode()).digest()[:8],'big')%(2**63)
        devices=[features.device.index if features.device.index is not None else torch.cuda.current_device()]if features.is_cuda else[]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if features.is_cuda:torch.cuda.manual_seed(seed)
            # All100 independent posterior paths execute every500 reverse step.
            # The shared deterministic condition encoder is broadcast; no path
            # or denoising step is replaced by a prior/classifier shortcut.
            means=prior.expand(100,-1)
            draws=self.diffusion.p_sample_loop(self.noise,result['condition'],means,means,500,
                self.alphas,self.one_minus_alpha_bar_sqrt,only_last_sample=True)
        if draws.shape!=(100,self.classes)or not torch.isfinite(draws).all():
            raise FloatingPointError('MExD complete100x500 posterior produced invalid class signals')
        probabilities=draws.softmax(-1)
        return {'probabilities':probabilities.mean(0,keepdim=True),'raw_draws':draws,
                'sample_probabilities':probabilities,'source_sigmoid_scores':probabilities.sigmoid().mean(0,keepdim=True),
                'stage':'diffusion'}

    def forward(self,features,coords=None):
        if features.ndim!=2 or features.shape[1]!=self.dim or len(features)<2 or not torch.isfinite(features).all():
            raise ValueError('MExD requires a complete finite bag in its registered actual FM dimension')
        if self.training:raise RuntimeError('Use both complete MExD training stages, not the generic CE loop')
        result=self.posterior(features,'direct_forward')
        return {'logits':result['probabilities'].clamp_min(1e-30).log(),**result}

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,config):
        self.eval();records=[]
        for row in bundle.records(subset):
            features,_=bundle.load(row);result=self.posterior(features.to(device),row['slide_id'])
            probability=result['probabilities'][0].cpu().numpy()
            record={k:row[k]for k in ['slide_id','patient_id','global_patient_id','target']}
            record.update(subset=subset,source_slides=row['slide_id'],mexd_stage=result['stage'])
            record.update({f'p_{i}':float(v)for i,v in enumerate(probability)})
            record.update({f'mexd_source_sigmoid_softmax_{i}':float(v)for i,v in enumerate(result['source_sigmoid_scores'][0].cpu())})
            if result['raw_draws']is not None:
                samples=result['sample_probabilities'].cpu().numpy();top=np.argsort(-probability,kind='stable')[:2]
                difference=samples[:,top[0]]-samples[:,top[1]]
                if np.all(difference==difference[0]):pvalue=1. if difference[0]==0 else 0.
                else:pvalue=float(ttest_rel(samples[:,top[0]],samples[:,top[1]]).pvalue)
                if not math.isfinite(pvalue):raise FloatingPointError('MExD posterior uncertainty t-test is undefined')
                record.update(mexd_uncertainty_p=pvalue,mexd_certain=int(pvalue<.05),mexd_posterior_samples=100,
                    mexd_reverse_steps=500,mexd_sample_seed_identity=hashlib.sha256((str(self.sampling_seed)+'\0'+row['slide_id']).encode()).hexdigest())
                record.update({f'mexd_mean_raw_signal_{i}':float(v)for i,v in enumerate(result['raw_draws'].mean(0).cpu())})
                if self.posterior_root is not None:
                    folder=self.posterior_root/f'epoch_{self.current_epoch:04d}'/subset/hashlib.sha256(row['slide_id'].encode()).hexdigest()
                    folder.mkdir(parents=True,exist_ok=True)
                    for name,value in [('raw_signals',result['raw_draws'].cpu().numpy()),('probabilities',samples)]:
                        path=folder/(name+'.npy')
                        if path.exists():
                            if not np.array_equal(np.load(path,allow_pickle=False),value):
                                raise ValueError('Existing MExD posterior proof disagrees with fresh locked replay')
                        else:np.save(path,value)
                    metadata={'slide_id':row['slide_id'],'profile':self.profile,'epoch':self.current_epoch,
                        'subset':subset,'samples':100,'steps':500,'input_identity':self.inputs.identity,
                        'sampling_seed':self.sampling_seed,'raw_sha256':digest_file(folder/'raw_signals.npy'),
                        'probability_sha256':digest_file(folder/'probabilities.npy'),'uncertainty_p':pvalue}
                    path=folder/'metadata.json'
                    if path.exists():
                        import json
                        if json.loads(path.read_text())!=metadata:raise ValueError('MExD posterior identity changed')
                    else:write_json(path,metadata)
            records.append(record)
        return pd.DataFrame(records)

    def protocol(self):
        return {'method':'MExD','profile':self.profile,'source_commit':COMMIT,'reference_reconstruction':True,
            'native_adapter_width':512,'adapter_count':2,'layers_per_adapter':2,'attention_heads':8,
            'landmarks':256,'pinv_iterations':6,'PPEG_kernels':[7,5,3],
            'actual_FM_dimension':self.dim,'published_FM_dimensions':[384,768],'actual_classes':self.classes,
            'experts':self.auxiliary.spec['experts'],'negative_expert':0,
            'observed_background_class':self.auxiliary.spec['observed_background'],
            'output_class_port':self.cohort=='class_port',
            'routing':'independent_binary_hard_argmax_and_topk_negative0.25_positive0.5',
            'router_task_gradient':'none_as_literal_hard_Eq2; no_straight_through_surrogate',
            'source_adapter_padding':'retained_inside_TransMIL_but_duplicate_rows_excluded_from_observed_expert_instances',
            'empty_expert_policy':'source_style_zero_sentinel_excluded_from_patch_counts; skip_undefined_empty_expert_CE',
            'expert_condition':'unnormalized_sum_of_expert_mean_times_own_class_confidence',
            'phase1':'Eq9_priorCE_plus_negativeCE_plus_bag_matching_positiveCE',
            'phase1_optimizer':'literal_author_RAdam_no_unpublished_Lookahead_from_paper_description',
            'phase2':'frozen_complete_DynMoE_plus_shifted_q_sample_and_noise_MSE',
            'phase2_optimizer':'Adam_beta0.9_0.999_eps1e-8_wd0_clip1',
            'native_budgets':[100,200],'actual_requested_budgets':[self.moe_epochs,self.diffusion_epochs],
            'schedulers':'separate_complete_stage_cosine_to_zero_no_unpublished50epoch_warmup',
            'diffusion_steps':500,'beta_schedule':'linear1e-4_to0.02','posterior_samples':100,
            'denoiser':'unchanged_author512to384to384to512_condition_and_three512D_timestep_embedded_layers',
            'posterior':'unchanged_full_shifted_source_posterior; vectorized100paths_with_original_condition_broadcast',
            'normalized_output':'mean100_softmax_reconstructed_signals',
            'source_score_output':'mean100_sigmoid_softmax_preserved_separately',
            'uncertainty':'paired_top_two_posterior_probability_ttest_alpha0.05; slide_level_only',
            'sampling':'per_original_slide_hash_seed_with_preserved_global_CPU_CUDA_RNG',
            'selection':'fixed_final_epoch_no_early_stopping_and_no_test_side_selection',
            'initializer':'full_DynMoE_trained_from_scratch_on_actual_frozen_features; no_private_checkpoint_substitution',
            'reference_choices':['paper_equation_MoE_unifies_incompatible_released_graphs',
                'stable_topk_ties','empty_expert_convention','unpad_pseudo_grid_for_original_instance_identity',
                'class_to_positive_expert_mapping','random_trainable_aggregator_init','fixed_final_epoch_selection',
                'stored_patch_order_and_zero_loader_workers','posterior_mean_and_per_slide_sampling_seed'],
            'input_identity':self.inputs.identity,'current_epoch':self.current_epoch,'current_phase':self.phase,
            'moe_updates':self.moe_steps,'diffusion_updates':self.diffusion_steps,
            'formal_admission':False,'real_WSI_validation':'not_run','paper_score_reproduction':'not_run'}


def source_and_uncertainty_metrics(predictions):
    """Native scores and PAvPU are explicit slide-level additions, never patient p-values."""
    columns=sorted((c for c in predictions if c.startswith('mexd_source_sigmoid_softmax_')),
                   key=lambda c:int(c.rsplit('_',1)[1]))
    target=predictions.target.to_numpy(dtype=int)
    metrics={}
    if columns:
        scores=predictions[columns].to_numpy(dtype=float)
        metrics['mexd_source_sigmoid_softmax_macro_ovr_auc']=float(np.mean([
            roc_auc_score(target==c,scores[:,c])for c in range(len(columns))]))
    if 'mexd_certain'in predictions:
        probability=predictions[[f'p_{i}'for i in range(len(columns))]].to_numpy(dtype=float)
        correct=probability.argmax(1)==target;certain=predictions.mexd_certain.to_numpy(dtype=bool)
        metrics.update(mexd_slide_pavpu=float(np.mean(correct&certain|~correct&~certain)),
            mexd_slide_certain_fraction=float(certain.mean()),
            mexd_slide_correct_certain=int(np.sum(correct&certain)),
            mexd_slide_incorrect_uncertain=int(np.sum(~correct&~certain)))
    return metrics

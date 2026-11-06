"""Full three-stage DisPro controller over the actual fixed-source networks.

The source-repaired reference retains UniPro-P, UniPro-G and MultiPro, including
the missing-modality survival distillation. No runnable admission is claimed by
the existence of this development module. Full-budget package gates are separate.
"""
import copy
import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from .discrete_survival import DiscreteSurvival
from .dispro_objectives import (multimodal_objective, predict_multimodal,
                                predict_unimodal, unimodal_objective)
from .dispro_source import (ASSET_SHA, COMMIT, REVISION, backbone,
                           build_multimodal, build_unimodal)
from .io import digest_object

PROFILE = 'dispro_source_repaired'
PHASES = ('unipro_wsi', 'unipro_omics', 'multipro')


def concordance(time, event, risk, tolerance=1e-8):
    """Right-censored comparable pairs, including event/censor time ties.

    Independent pair enumeration of the public sksurv metric semantics used by
    DisPro: two events at the same time are incomparable; event/censor time ties
    are comparable; risk differences <=1e-8 count as half-concordant. Do not
    silently substitute the library's strict-time/no-tolerance summary metric.
    """
    t, e, r = (np.asarray(v) for v in (time, event, risk))
    if (any(v.ndim != 1 for v in (t,e,r)) or not len(t)==len(e)==len(r)
            or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(r).all()
            or not np.isin(e,[0,1]).all() or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('DisPro concordance needs finite patient time/event/risk vectors')
    points, count = 0., 0
    for i in np.flatnonzero(e):
        eligible = (t>t[i]) | ((t==t[i]) & (e==0))
        difference = r[i]-r[eligible]
        points += np.sum(difference>tolerance)+.5*np.sum(np.abs(difference)<=tolerance)
        count += int(eligible.sum())
    if not count:
        raise ValueError('DisPro validation has no comparable survival pairs')
    return float(points/count)


def _foundation_keys(module):
    ids = {id(p) for p in backbone().parameters()}
    ids.update(id(p) for p in backbone().buffers())
    names = {n for n,p in module.named_parameters(remove_duplicate=False) if id(p) in ids}
    names.update(n for n,p in module.named_buffers(remove_duplicate=False) if id(p) in ids)
    return names & set(nn.Module.state_dict(module))


def _portable_state(module, clone=False):
    constants = _foundation_keys(module)
    result = {n:v for n,v in nn.Module.state_dict(module).items() if n not in constants}
    if clone:
        result = {n:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                  for n,v in result.items()}
    return result


def _load_portable(module, state):
    current = nn.Module.state_dict(module)
    constants = _foundation_keys(module)
    if not isinstance(state,dict) or set(state) != set(current)-constants:
        raise ValueError('DisPro checkpoint has missing/extra non-foundation model states')
    current.update(state)
    return nn.Module.load_state_dict(module,current,strict=True)


def _weight_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not torch.is_tensor(value): raise ValueError('DisPro teacher weights must be tensors')
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _single_patient(batch):
    return batch[0]


class DisPro(DiscreteSurvival):
    owns_sampling = True
    handles_early_stopping = True

    def __init__(self,bundle,profile=PROFILE,stage_epochs=(30,50,50)):
        super().__init__(bins=4,alpha=0.)
        if (profile!=PROFILE or bundle.task_type!='survival' or bundle.dispro is None
                or bundle.config.get('training_unit')!='patient'
                or bundle.config.get('evaluation_unit')!='patient'):
            raise ValueError('DisPro requires explicit incomplete patient WSI/RNA survival inputs')
        if (not isinstance(stage_epochs,(tuple,list)) or len(stage_epochs)!=3
                or any(type(v) is not int or v<1 for v in stage_epochs)):
            raise ValueError('DisPro stage_epochs must explicitly budget all three stages')
        self.bundle,self.profile=bundle,profile
        self.stage_epochs=tuple(stage_epochs)
        sizes=[len(p['features']) for p in bundle.dispro.pathways]
        # Construct the actual full native modules before one device transfer.
        # All runtime aliases refer to these same learned modality encoders.
        wsi=build_unimodal('WSI',bundle.dim,sizes)
        params_wsi=[p for p in wsi.parameters() if p.requires_grad]
        omics=build_unimodal('Omics',bundle.dim,sizes)
        params_omics=[p for p in omics.parameters() if p.requires_grad]
        self.network=build_multimodal(wsi,omics,sizes,bundle.dim,repaired_mask=True)
        fusion_groups=[('holder_path',[self.network.holder_path],2e-4),
            ('holder_omic',[self.network.holder_omic],1e-5),
            ('cls_token',[self.network.cls_token],2e-4)]
        for name,rate in [('wsi_adapter',2e-4),('omics_adapter',1e-5),
                ('wsi_attn_head',2e-4),('omics_attn_head',1e-5),
                ('linear_head',2e-4),('classifier',2e-4)]:
            fusion_groups.append((name,list(getattr(self.network,name).parameters()),rate))
        self._fusion_groups=fusion_groups
        self._phase_params=[params_wsi,params_omics,[p for _,ps,_ in fusion_groups for p in ps]]
        if any(len({id(p) for p in ps})!=len(ps) for ps in self._phase_params):
            raise ValueError('DisPro optimizer parameters must not appear twice in one stage')
        names={id(p):n for n,p in self.network.named_parameters()}
        self.phase_parameters=[[names[id(p)] for p in ps] for ps in self._phase_params]
        self._fusion_parameter_names=[(name,[names[id(p)] for p in ps],rate)
                                      for name,ps,rate in fusion_groups]
        self.contract=digest_object({'source':COMMIT,'biobert_revision':REVISION,'biobert':ASSET_SHA,
            'bundle':bundle.fingerprint,'inputs':bundle.dispro.contract,'profile':profile,
            'stage_epochs':self.stage_epochs,'phase_parameters':self.phase_parameters,
            'reference':'actual_three_stages_VAL_only_outcome_free_actual_parameter_cosine_v1'})
        self.phase=0;self.completed_epochs=0;self.stage_progress=[0,0,0]
        self.optimizer_steps=[0,0,0];self.train_orders=[]
        self.teacher_best=[None,None];self.teacher_states=[None,None]
        self.bound_teachers=None;self.optimizers=[];self.schedulers=[]
        self._configured=False
        self._activate(0)

    @property
    def wsi(self):return self.network.uni_model_wsi

    @property
    def omics(self):return self.network.uni_model_omics

    def _apply(self,fn,recurse=True):
        if getattr(self,'optimizers',[]):
            raise ValueError('DisPro device transfer must precede stage optimizer construction')
        # PyTorch device/dtype conversion may replace Parameter objects. Keep
        # both raw source aliases (e.g. wsi_logit_scale) and stage-owned lists
        # attached to the ACTUAL converted modules, rather than stale CPU leaves.
        aliases={}
        for name,p in self.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(p),[]).append(name)
        result=super()._apply(fn,recurse=recurse)
        for names in aliases.values():
            parameter=self.get_parameter(names[0])
            for name in names[1:]:
                prefix,_,leaf=name.rpartition('.')
                self.get_submodule(prefix)._parameters[leaf]=parameter
        if hasattr(self,'phase_parameters'):
            self._phase_params=[[self.network.get_parameter(n) for n in group] for group in self.phase_parameters]
            self._fusion_groups=[(name,[self.network.get_parameter(n) for n in group],rate)
                                 for name,group,rate in self._fusion_parameter_names]
            self._activate(self.phase)
        return result

    def _activate(self,phase):
        self.network.requires_grad_(False)
        for p in self._phase_params[phase]:p.requires_grad_(True)
        self.phase=phase

    def configure_training(self,cfg):
        if (cfg['epochs']!=sum(self.stage_epochs) or cfg['grad_clip'] is not None
                or cfg['max_instances'] is not None or cfg['weight_decay']!=1e-5
                or cfg['lr']!=2e-4 or cfg['selection_metric']!='dispro_c_index'):
            raise ValueError('DisPro requires all declared stage budgets, full bags, native Adam rates/decay, no clipping and explicit censored concordance selection')
        self._configured=True

    def _groups(self,subset,phase=None):
        from .engine import units
        phase=self.phase if phase is None else phase
        groups=units(self.bundle,subset)
        if phase<2:
            modal=['WSI','Omics'][phase]
            groups=[g for g in groups if self.bundle.dispro.observed(g[0]['global_patient_id'],modal)]
        if not groups:raise ValueError('DisPro stage has no actually observed patient inputs')
        return groups

    def prepare(self,bundle,seed,device):
        super().prepare(bundle,seed,device)
        # Each stage's VAL population must actually permit its own selection.
        # Only TRAIN observed event times above fit the four interval edges.
        for phase in range(3):
            self._groups('train',phase)
            rows=[g[0] for g in self._groups('val',phase)]
            concordance([r['time'] for r in rows],[r['event'] for r in rows],[0.]*len(rows))

    def make_optimizers(self,lr,weight_decay):
        if not self._configured:raise ValueError('Configure the full DisPro stage recipe first')
        self.optimizers=[torch.optim.Adam(self._phase_params[0],lr=2e-4,weight_decay=1e-5),
            torch.optim.Adam(self._phase_params[1],lr=1e-5,weight_decay=1e-5),
            torch.optim.Adam([{'params':ps,'lr':rate,'name':name} for name,ps,rate in self._fusion_groups],
                             weight_decay=0.)]
        self.schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs,eta_min=0.)
                         for o,epochs in zip(self.optimizers,self.stage_epochs)]
        return self.optimizers

    def start_epoch(self,epoch):
        if type(epoch) is not int or epoch!=self.completed_epochs or not 0<=epoch<sum(self.stage_epochs):
            raise ValueError('DisPro stage progression cannot skip/repeat epochs')
        phase=int(np.searchsorted(np.cumsum(self.stage_epochs),epoch,side='right'))
        if phase!=self.phase:
            if phase!=self.phase+1 or self.stage_progress[self.phase]!=self.stage_epochs[self.phase]:
                raise ValueError('DisPro cannot skip an unfinished unimodal stage')
            previous=self.phase
            if self.teacher_best[previous] is None or self.teacher_states[previous] is None:
                raise ValueError('DisPro stage transition needs the actual selected VAL teacher')
            _load_portable([self.wsi,self.omics][previous],self.teacher_states[previous])
            if phase==2:
                # Reload BOTH independently selected teachers, rather than the
                # latest RNA epoch or a newly initialized surrogate.
                for i,modal in enumerate([self.wsi,self.omics]):
                    _load_portable(modal,self.teacher_states[i])
                self.bound_teachers=copy.deepcopy(self.teacher_best)
                self.teacher_states=[None,None]
            # A completed stage will never be optimized again. Its selected
            # weights/provenance survive; inactive Adam moments are released.
            self.optimizers[previous].state.clear()
            for p in self._phase_params[previous]:p.grad=None
        self._activate(phase)

    def validate_optimizer_state(self):
        if len(self.optimizers)!=3 or len(self.schedulers)!=3:
            raise ValueError('DisPro resume requires all three original stage optimizers/schedules')
        for i,(optimizer,scheduler) in enumerate(zip(self.optimizers,self.schedulers)):
            rates=[group['lr'] for group in optimizer.param_groups]
            if rates!=scheduler.get_last_lr() or scheduler.optimizer is not optimizer:
                raise ValueError('DisPro optimizer LR/state differs from its actual cosine schedule')
            allowed={id(p) for p in self._phase_params[i]}
            if (len(optimizer.param_groups)!=(9 if i==2 else 1)
                    or any(id(p) not in allowed for group in optimizer.param_groups for p in group['params'])
                    or len([p for group in optimizer.param_groups for p in group['params']])!=len(allowed)
                    or any(group['weight_decay']!=(0. if i==2 else 1e-5) for group in optimizer.param_groups)):
                raise ValueError('DisPro optimizer parameter ownership/decay changed')
            if i<self.phase and optimizer.state:
                raise ValueError('DisPro completed-stage optimizer must release inactive moments')
            if i==self.phase and self.optimizer_steps[i] and not optimizer.state:
                raise ValueError('DisPro active-stage Adam continuation moments are absent')
            for p,state in optimizer.state.items():
                step=state.get('step')
                if (id(p) not in allowed or not torch.is_tensor(step) or step.numel()!=1
                        or not torch.isfinite(step) or step.item()!=int(step.item())
                        or not 1<=step.item()<=self.optimizer_steps[i]
                        or any(k not in state or state[k].shape!=p.shape or not torch.isfinite(state[k]).all()
                               for k in ['exp_avg','exp_avg_sq'])):
                    raise ValueError('DisPro Adam moments/update counts are not a valid stage continuation')
        if self.phase==2 and self.optimizer_steps[2]:
            for name in ['cls_token','holder_path','holder_omic']:
                state=self.optimizers[2].state.get(getattr(self.network,name),{})
                if 'step' not in state or state['step'].item()!=self.optimizer_steps[2]:
                    raise ValueError('DisPro actual CLS/holder Adam steps were lost during resume')

    def epoch(self,bundle,optimizers,device):
        if not bool(self.prepared) or optimizers is not self.optimizers:
            raise ValueError('DisPro training must use its fitted bins and three actual bound optimizers')
        groups=self._groups('train');order=[];records=[]
        loader=DataLoader(groups,batch_size=1,sampler=SubsetRandomSampler(range(len(groups))),
                          num_workers=0,collate_fn=_single_patient)
        optimizer=optimizers[self.phase]
        for rows in loader:
            self.train();optimizer.zero_grad(set_to_none=True)
            inputs=bundle.dispro.inputs(rows,device,mode=['wsi','omics','available'][self.phase])
            event,interval=rows[0]['event'],int(self.time_label(rows[0]['time']))
            if self.phase<2:
                modal,key=[(self.wsi,'x_WSI'),(self.omics,'x_Omics')][self.phase]
                result=unimodal_objective(modal,inputs[key],event,interval)
            else:result=multimodal_objective(self.network,inputs,event,interval)
            loss=result['loss']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DisPro stage loss')
            loss.backward()
            for p in self._phase_params[self.phase]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite DisPro native gradient')
            optimizer.step();self.optimizer_steps[self.phase]+=1
            order.append(rows[0]['global_patient_id'])
            record={'loss':float(loss.detach()),'dispro_phase':self.phase}
            for key in ['fused_survival','missing_wsi_unipro_distillation','missing_omics_unipro_distillation']:
                if key in result:record[key]=float(result[key].detach())
            records.append(record)
        self.train_orders.append({'phase':self.phase,'patients':order})
        self.stage_progress[self.phase]+=1;self.completed_epochs+=1
        self.schedulers[self.phase].step()
        return records

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        if subset=='test' and (self.phase!=2 or self.bound_teachers is None):
            raise ValueError('DisPro TEST requires the complete three-stage fitted model')
        self.eval();records=[]
        for rows in self._groups(subset):
            row=rows[0];inputs=bundle.dispro.inputs(rows,device,mode=['wsi','omics','available'][self.phase])
            result=(predict_unimodal([self.wsi,self.omics][self.phase],inputs[['x_WSI','x_Omics'][self.phase]])
                    if self.phase<2 else predict_multimodal(self.network,inputs))
            records.append({'slide_id':'patient:'+row['global_patient_id'],'patient_id':row['patient_id'],
                'global_patient_id':row['global_patient_id'],'subset':subset,
                'source_slides':';'.join(r['slide_id'] for r in rows),'time':row['time'],'event':row['event'],
                'risk':float(result['risk'].item()),'dispro_phase':PHASES[self.phase],
                'wsi_observed':int(inputs['x_WSI'] is not None),
                'omics_observed':int(inputs['x_Omics'] is not None),
                **{f'{key}_{i}':float(v) for key in ['hazards','survival']
                   for i,v in enumerate(result[key][0])}})
        return pd.DataFrame(records)

    def validation_metrics(self,frame):
        return {'dispro_c_index':concordance(frame.time,frame.event,frame.risk),
                'dispro_phase':self.phase,'dispro_stage_epoch':self.stage_progress[self.phase]}

    def after_validation(self,metrics):
        if self.phase>=2:return
        value=metrics['dispro_c_index'];previous=self.teacher_best[self.phase]
        if previous is None or value>previous['dispro_c_index']:
            state=_portable_state([self.wsi,self.omics][self.phase],clone=True)
            self.teacher_best[self.phase]={'dispro_c_index':value,'epoch':self.stage_progress[self.phase]-1,
                'weights_sha256':_weight_digest(state),
                'val_patients':[g[0]['global_patient_id'] for g in self._groups('val')]}
            self.teacher_states[self.phase]=state

    def selection_improved(self,objective,best):
        return self.phase==2 and objective>best

    def selection_checkpoint(self,state):
        if self.phase!=2 or self.bound_teachers is None:
            raise ValueError('DisPro selected checkpoint must be a fitted MultiPro model')
        # Evaluation consumes weights/signature, whereas exact continuation
        # always consumes last.pt. Do not duplicate ~1.7GB of genomic Adam
        # moments in each validation-selected inference checkpoint.
        return {**{k:v for k,v in state.items() if k not in {'optimizers','rng'}},
                'checkpoint_role':'validation_selected_weights; resume_from_last.pt'}

    def state_dict(self,*args,**kwargs):
        state=super().state_dict(*args,**kwargs)
        for name in _foundation_keys(self.network):state.pop('network.'+name,None)
        return state

    def load_state_dict(self,state_dict,strict=True,assign=False):
        current=super().state_dict()
        constants={'network.'+n for n in _foundation_keys(self.network)}
        if not strict or assign or set(state_dict)!=set(current)-constants:
            raise ValueError('DisPro requires strict complete checkpoint states plus verified external BioBERT')
        current.update(state_dict)
        return super().load_state_dict(current,strict=True)

    def get_extra_state(self):
        return {'contract':self.contract,'biobert_sha256':ASSET_SHA,'phase':self.phase,
            'completed_epochs':self.completed_epochs,'stage_progress':self.stage_progress.copy(),
            'optimizer_steps':self.optimizer_steps.copy(),'train_orders':copy.deepcopy(self.train_orders),
            'teacher_best':copy.deepcopy(self.teacher_best),'teacher_states':self.teacher_states,
            'bound_teachers':copy.deepcopy(self.bound_teachers),
            'schedulers':[s.state_dict() for s in self.schedulers] if self.schedulers else
                copy.deepcopy(getattr(self,'_saved_schedulers',[]))}

    def set_extra_state(self,state):
        if (not isinstance(state,dict) or set(state)!=set(self.get_extra_state())
                or state['contract']!=self.contract or state['biobert_sha256']!=ASSET_SHA):
            raise ValueError('DisPro resume input/source/recipe/foundation identity changed')
        progress=state['stage_progress'];complete=state['completed_epochs'];phase=state['phase']
        if (type(phase) is not int or phase not in range(3) or type(complete) is not int
                or not isinstance(progress,list) or len(progress)!=3
                or any(type(n) is not int or not 0<=n<=b for n,b in zip(progress,self.stage_epochs))
                or sum(progress)!=complete or len(state['train_orders'])!=complete
                or any(progress[i]!=self.stage_epochs[i] for i in range(phase))
                or any(progress[i] for i in range(phase+1,3))):
            raise ValueError('DisPro checkpoint stage progression is inconsistent')
        expected_phases=[i for i,n in enumerate(progress) for _ in range(n)]
        updates=[0,0,0]
        for item,expected in zip(state['train_orders'],expected_phases):
            allowed={g[0]['global_patient_id'] for g in self._groups('train',expected)}
            if (set(item)!={'phase','patients'} or item['phase']!=expected
                    or len(item['patients'])!=len(allowed) or set(item['patients'])!=allowed):
                raise ValueError('DisPro saved sampler contains missing/duplicate/non-TRAIN patients')
            updates[expected]+=len(allowed)
        if updates!=state['optimizer_steps'] or len(state['schedulers'])!=3:
            raise ValueError('DisPro optimizer or scheduler counters disagree with real TRAIN updates')
        for i,saved in enumerate(state['schedulers']):
            if saved['last_epoch']!=progress[i] or saved['T_max']!=self.stage_epochs[i] or saved['eta_min']!=0:
                raise ValueError('DisPro cosine schedule does not match complete stage budget')
        if len(state['teacher_best'])!=2 or len(state['teacher_states'])!=2:
            raise ValueError('DisPro needs both independent unimodal selection records')
        for i,(meta,weights) in enumerate(zip(state['teacher_best'],state['teacher_states'])):
            if meta is None:
                if progress[i] or weights is not None:raise ValueError('DisPro selected teacher is absent')
                continue
            expected=[g[0]['global_patient_id'] for g in self._groups('val',i)]
            if (set(meta)!={'dispro_c_index','epoch','weights_sha256','val_patients'}
                    or meta['val_patients']!=expected or type(meta['epoch']) is not int
                    or not 0<=meta['epoch']<progress[i] or not math.isfinite(meta['dispro_c_index'])
                    or not 0<=meta['dispro_c_index']<=1):
                raise ValueError('DisPro teacher metadata is not an eligible VAL-only selection')
            if phase<2 and (weights is None or _weight_digest(weights)!=meta['weights_sha256']):
                raise ValueError('DisPro selected teacher snapshot differs from its checkpoint digest')
            if phase==2 and weights is not None:
                raise ValueError('DisPro completed pretraining must bind teachers into actual MultiPro')
        if phase==2 and state['bound_teachers']!=state['teacher_best']:
            raise ValueError('DisPro joint training is not bound to both selected teachers')
        if phase<2 and state['bound_teachers'] is not None:
            raise ValueError('DisPro teacher binding cannot precede both pretraining stages')
        for key in ['completed_epochs','stage_progress','optimizer_steps','train_orders','teacher_best',
                    'teacher_states','bound_teachers']:
            setattr(self,key,copy.deepcopy(state[key]) if key!='teacher_states' else state[key])
        self._activate(phase)
        if self.schedulers:
            for scheduler,saved in zip(self.schedulers,state['schedulers']):scheduler.load_state_dict(saved)
        else:self._saved_schedulers=copy.deepcopy(state['schedulers'])

    def protocol(self):
        return {**super().protocol(),'source':'Innse/DisPro@'+COMMIT,'profile':self.profile,
            'foundation':{'revision':REVISION,'files_sha256':ASSET_SHA,'native_hidden':768,'native_layers':12},
            'stages':[{'name':name,'epochs':budget,'completed_epochs':done,'optimizer_updates':steps}
                for name,budget,done,steps in zip(PHASES,self.stage_epochs,self.stage_progress,self.optimizer_steps)],
            'native_inputs':{'FM_dimension':self.bundle.dim,'ordered_pathways':len(self.bundle.dispro.pathways),
                             'full_pathway_sizes':[len(p['features']) for p in self.bundle.dispro.pathways]},
            'teacher_best':copy.deepcopy(self.teacher_best),'bound_teachers':copy.deepcopy(self.bound_teachers),
            'phase_parameters':self.phase_parameters,'completed_epochs':self.completed_epochs,
            'optimizer':'three independent Adam; UniPro rates2e-4/1e-5 decay1e-5; actual MultiPro parameter groups2e-4/1e-5 decay0; CLS and holders included',
            'schedule':'each cosine bound to its ACTUAL optimizer and own complete stage horizon; no early stopping',
            'checkpoint':'all learned weights and active Adam moments, all schedules/samplers/RNG/selected teachers; frozen public BioBERT verified externally and omitted from payload',
            'selection':'first strictly best VAL right-censored concordance per UniPro; MultiPro only enters final best.pt selection; TEST never selects',
            'inference':'UniPro fixed uncensored risk prompts, no outcomes; original MultiPro labelNone/all-eight scoring and constant auxiliary censor interface',
            'reference_ports':'source four shared context groups, fixed original native channels; actual Parameters/CLS, additive padding mask, explicit real presence flags, TRAIN bins, deterministic VAL order, one unified initialization/RNG stream and num_workers0 input port',
            'scope':'complete source-repaired three-stage development controller; full native package budgets and admission gates still required'}

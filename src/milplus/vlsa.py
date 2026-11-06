"""Full VLSA ordinal-survival recipe, including original patient-window loss."""
import copy
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .io import digest_object
from .vlsa_source import COMMIT,WEIGHTS_SHA,build_network,mean_times,objective,optimizer_groups,source_concordance


class Patients:
    def __init__(self,bundle,subset):
        self.rows=[g.to_dict('records') for _,g in
            bundle.rows.loc[bundle.rows.subset==subset].groupby('global_patient_id',sort=False)]

    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def one_patient(batch):return batch[0]


class VLSAMIL(nn.Module):
    owns_sampling=True
    handles_early_stopping=True

    def __init__(self,bundle,profile='vlsa_conch_source_repaired',update_patients=32):
        super().__init__()
        if (profile!='vlsa_conch_source_repaired' or bundle.vlsa is None
                or type(update_patients) is not int or update_patients<1):
            raise ValueError('VLSA requires native CONCH survival inputs and a positive explicit patient-window size')
        self.bundle,self.inputs,self.profile=bundle,bundle.vlsa,profile
        self.core=build_network(self.inputs.weights,self.inputs.rank_prompts,
            self.inputs.query_prompts,self.inputs.query_key,self.inputs.num_ranks)
        self.update_patients=update_patients
        self.contract=digest_object({'source':COMMIT,'inputs':self.inputs.contract,
            'bundle':bundle.fingerprint,'profile':profile,'update_patients':update_patients})
        self.register_buffer('time_coordinates',torch.tensor(self.inputs.coordinates,dtype=torch.float64))
        self.epoch_index,self.completed_epochs,self.optimizer_updates=-1,0,0
        self.train_orders=[];self.window_sizes=[]

    def configure_training(self,cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric']!='vlsa_final_epoch'):
            raise ValueError('VLSA source uses complete patient bags, no clipping and final-epoch selection')

    def make_optimizer(self,lr,weight_decay):
        self._configured_lr=float(lr);self._configured_decay=float(weight_decay)
        self._optimizer=torch.optim.Adam(optimizer_groups(self.core,weight_decay),lr=lr)
        return self._optimizer

    def validate_optimizer_state(self):
        optimizer=self._optimizer
        expected=optimizer_groups(self.core,self._configured_decay)
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups)!=len(expected):
            raise ValueError('VLSA requires original Adam with original bias/1D zero-decay grouping')
        for group,wanted in zip(optimizer.param_groups,expected):
            if (group['lr']!=self._configured_lr or group['weight_decay']!=wanted['weight_decay']
                    or group['betas']!=(.9,.999) or group['eps']!=1e-8 or group['amsgrad']
                    or group.get('maximize',False)
                    or [id(p) for p in group['params']]!=[id(p) for p in wanted['params']]):
                raise ValueError('VLSA saved optimizer changed its source update/parameter grouping')
            for p in group['params']:
                state=optimizer.state.get(p,{})
                if self.optimizer_updates==0 and not state:continue
                if (set(state)!={'step','exp_avg','exp_avg_sq'}
                        or float(state['step'])!=self.optimizer_updates
                        or any(state[k].shape!=p.shape or not torch.isfinite(state[k]).all()
                               for k in ['exp_avg','exp_avg_sq'])):
                    raise ValueError('VLSA Adam moments or actual patient-window update count differs')

    def _apply(self,fn,recurse=True):
        if hasattr(self,'_optimizer'):
            raise ValueError('Move VLSA before creating its original Adam optimizer')
        return super()._apply(fn,recurse)

    def start_epoch(self,epoch):
        if epoch!=self.completed_epochs:
            raise ValueError('VLSA epochs must continue the complete saved patient history')
        self.inputs.check_unchanged();self.epoch_index=epoch

    def epoch(self,bundle,optimizer,device):
        loader=DataLoader(Patients(bundle,'train'),batch_size=1,shuffle=True,
                          num_workers=0,collate_fn=one_patient)
        losses,order,sizes=[],[],[]
        pending=[]

        def update(rows_window):
            logits,times,events=[],[],[]
            for rows in rows_window:
                value,*_=self.core(self.inputs.patient(rows,device))
                t,e=self.inputs.label(rows,device)
                logits.append(value);times.append(t);events.append(e)
            # Original code forwards every patient in the window, then computes
            # IFMLE+EMD once over concatenated predictions, then a single step.
            # This is not an approximation using per-patient optimizer steps.
            optimizer.zero_grad()
            loss,parts=objective(torch.cat(logits),torch.cat(times),torch.cat(events),self.core.get_logit_scale())
            if not torch.isfinite(loss):raise ValueError('VLSA original complete window loss became nonfinite')
            loss.backward();optimizer.step();self.optimizer_updates+=1
            sizes.append(len(rows_window))
            losses.append({'loss':float(loss.detach()),'ifmle':float(parts['ifmle'].detach()),
                           'emd':float(parts['emd'].detach())})

        for rows in loader:
            order.append(rows[0]['global_patient_id']);pending.append(rows)
            if len(pending)==self.update_patients:
                update(pending);pending=[]
        if pending:update(pending)
        self.train_orders.append(order);self.window_sizes.append(sizes)
        self.completed_epochs=self.epoch_index+1
        return losses

    @torch.no_grad()
    def predict_subset(self,bundle,subset,device,cfg):
        self.eval();self.inputs.check_unchanged()
        loader=DataLoader(Patients(bundle,subset),batch_size=1,num_workers=0,collate_fn=one_patient)
        records=[]
        for rows in loader:
            logits,*_=self.core(self.inputs.patient(rows,device))
            probabilities=logits.softmax(-1).cpu()
            means,survival=mean_times(probabilities,self.time_coordinates.cpu().numpy())
            row=rows[0]
            record={k:row[k] for k in ['patient_id','global_patient_id','subset','time','event']}
            record.update(slide_id='patient:'+row['global_patient_id'],
                source_slides=';'.join(r['slide_id'] for r in rows),
                risk=-float(means[0]),vlsa_mean_survival_time=float(means[0]))
            record.update({f'incidence_{i}':float(v) for i,v in enumerate(probabilities[0])})
            record.update({f'survival_{i}':float(v) for i,v in enumerate(survival[0])})
            records.append(record)
        return pd.DataFrame(records)

    def validation_metrics(self,frame):
        try:concordance=source_concordance(frame.time,frame.event,frame.vlsa_mean_survival_time)
        except ValueError:concordance=None
        return {'vlsa_source_mean_time_c_index':concordance,'vlsa_final_epoch':float(self.completed_epochs)}

    def selection_objective(self,metrics,epoch,current):return float(epoch+1)
    def selection_improved(self,objective,best):return True

    def get_extra_state(self):
        return {'contract':self.contract,'epoch_index':self.epoch_index,'completed_epochs':self.completed_epochs,
            'optimizer_updates':self.optimizer_updates,'train_orders':copy.deepcopy(self.train_orders),
            'window_sizes':copy.deepcopy(self.window_sizes)}

    def set_extra_state(self,state):
        if (not isinstance(state,dict) or set(state)!={'contract','epoch_index','completed_epochs',
                'optimizer_updates','train_orders','window_sizes'} or state['contract']!=self.contract
                or any(type(state[k]) is not int for k in ['epoch_index','completed_epochs','optimizer_updates'])
                or state['completed_epochs']<0 or state['epoch_index']!=state['completed_epochs']-1
                or any(not isinstance(state[k],list) or len(state[k])!=state['completed_epochs']
                       for k in ['train_orders','window_sizes'])):
            raise ValueError('VLSA checkpoint changed the exact source/input/window/epoch contract')
        expected=Patients(self.bundle,'train')
        ids=[rows[0]['global_patient_id'] for rows in expected.rows]
        sizes=[self.update_patients]*(len(ids)//self.update_patients)
        if len(ids)%self.update_patients:sizes.append(len(ids)%self.update_patients)
        if (any(not isinstance(order,list) or len(order)!=len(ids) or set(order)!=set(ids)
                for order in state['train_orders'])
                or any(window!=sizes for window in state['window_sizes'])
                or state['optimizer_updates']!=len(sizes)*state['completed_epochs']):
            raise ValueError('VLSA checkpoint lost complete TRAIN-only patient/window/update history')
        for name in ['epoch_index','completed_epochs','optimizer_updates','train_orders','window_sizes']:
            setattr(self,name,copy.deepcopy(state[name]))

    def _foundation_names(self):
        ids={id(p) for p in self.core.prompt_encoder.parameters()}
        return {n for n,p in self.named_parameters(remove_duplicate=False) if id(p) in ids}

    def state_dict(self,*args,**kwargs):
        state=super().state_dict(*args,**kwargs)
        for name in self._foundation_names():state.pop(name,None)
        return state

    def load_state_dict(self,state_dict,strict=True,assign=False):
        expected=self.state_dict()
        if not strict or assign or not isinstance(state_dict,dict) or set(state_dict)!=set(expected):
            raise ValueError('VLSA requires exact compact nonfoundation state and actual frozen checkpoint reconstruction')
        learned={n for n,p in self.named_parameters() if p.requires_grad}
        for name,value in expected.items():
            if name=='_extra_state':continue
            saved=state_dict[name]
            if (not torch.is_tensor(saved) or saved.shape!=value.shape or saved.dtype!=value.dtype
                    or (name in learned and not torch.isfinite(saved).all())
                    or (name not in learned and not torch.equal(saved.cpu(),value.cpu()))):
                raise ValueError('VLSA checkpoint learned/fixed state differs: '+name)
        complete=super().state_dict();complete.update(state_dict)
        return super().load_state_dict(complete,strict=True,assign=False)

    def protocol(self):
        return {'source_commit':COMMIT,'pretrained_weights_sha256':WEIGHTS_SHA,
            'profile':self.profile,'input_contract':self.inputs.contract,'feature_dimension':512,
            'text_architecture':'Original CONCH width768/12layers/12heads/128positions/32007vocab, causal+CLS; frozen',
            'language':'Original4-base-rank linear interpolation, learned context/rank tokens; text initialization infers actual4 context tokens',
            'vision':'Original VLFAN native512 linear adapter, actual prognostic-text queries, TaskRes ratio0.5, fixed cross-attention temperature100, mean query pooling',
            'num_queries':self.inputs.num_queries,'num_ranks':self.inputs.num_ranks,
            'time_coordinates':self.inputs.coordinates.tolist(),'terminal_bin_edge':'+infinity',
            'time_bins':'Original TRAIN-event interval edges, ceil(sqrt(TRAIN event count)) unless explicit; infinite terminal edge replaces full-table max+EPS',
            'tokenizer_compatibility':'Explicit original JSON add_prefix_space=True; preserve author transformers4.41.2/tokenizers0.19.1 token IDs under the unified modern dependency environment',
            'objective':'Literal original softmax incidence IFMLE(alpha0,eps1e-7)+SurvEMD(p2,raw squared); original second softmax and detached target temperature retained',
            'recipe':'Original Adam lr0.0002, decay0.00001 with bias/1D zero decay, B1,32-patient concatenated loss per update,10epochs, final checkpoint; no scheduler/clipping/early stop',
            'update_patients':self.update_patients,'native_window':self.update_patients==32,
            'actual_optimizer':{'name':'Adam','lr':getattr(self,'_configured_lr',None),
                'weight_decay':getattr(self,'_configured_decay',None),
                'bias_and_1d_weight_decay':0.,'betas':[.9,.999],'eps':1e-8},
            'source_c_index':'Original Mean survival time, Comparable pairs, ties All',
            'common_c_index':'Common patient Harrell separately; risk is negative source mean survival time',
            'ports':['Original full-table last time edge replaced by infinity, finite TRAIN interior and left-edge coordinates unchanged',
                'Stable slide-ID concat for every registered slide of a true patient',
                'DataLoader workers0 instead of4; added VAL diagnostics never change final selection; no paper RNG identity claim',
                'Only unused frozen CONCH visual tower allocation omitted; actual150 text tensors and learned temperature preserved',
                'Source provides legacy dependency versions; benchmark binds the CONCH asset SHA and explicitly preserves original ByteLevel prefix behavior in its modern dependency environment'],
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'formal_admission':False,'paper_reproduction':'not_run','natural_wsi_training':'not_run',
            'author_cached_example_notebook':'Numeric mismatch observed under pinned CPU environment; unresolved, not passed'}

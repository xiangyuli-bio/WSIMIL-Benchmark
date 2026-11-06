"""Full continual GDAT/BPPL/OWLoRA, original optimizer and complete task recovery."""
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score

from . import comel_source as source
from .comel_data import CoMELInputs
from .data import Bundle
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import save,load,exists,checkpoint_path,cpu_state,Interrupted
from .io import read_json,write_json,digest_file
from .provenance import software_identity


class Bags(Dataset):
    def __init__(self,assets,task):self.assets=assets;self.rows=assets.records(task,'train');self.observed=[]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];self.observed.append(row['slide_id']);x=self.assets.features(row)
        # The source CE branch never uses patch ground truth. An explicit -1
        # missing marker preserves its tensor interface without extra supervision.
        return x,torch.tensor(row['target']),torch.full((len(x),1),-1,dtype=torch.long)


def _device(device):
    d=torch.device(device)
    if d.type not in {'cpu','cuda'} or (d.type=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Choose an available actual CPU or CUDA runtime; no device fallback')
    return d


def _inputs(data,fm=None,split=None):
    bundle=Bundle(data,fm,split);assets=CoMELInputs(bundle);return bundle,assets


def _layout(network):return {'architecture':source.architecture(network),'model':cpu_state(network),
    'training_modes':{n:m.training for n,m in network.named_modules()}}


def _restore(dimension,state,device):
    network=source.restore_model(dimension,state['architecture'],state['model'],device)
    modules=dict(network.named_modules())
    if set(modules)!=set(state['training_modes']):raise ValueError('CoMEL module mode graph changed')
    for name,module in modules.items():module.training=state['training_modes'][name]
    return network


@torch.no_grad()
def predictions(network,assets,task,subset,device):
    network.eval();rows=[];patch=[]
    for row in assets.records(task,subset):
        x=assets.features(row).unsqueeze(0).to(device)
        logits,attention=network(x,return_attn=True,no_norm=False)
        if logits.shape!=(1,2*len(network.classifier)) or not torch.isfinite(logits).all() or not torch.isfinite(attention).all():
            raise FloatingPointError('CoMEL full seen-head logits/attention are nonfinite or incomplete')
        p=logits.softmax(1)[0].cpu().numpy();a=attention[0].cpu().numpy()
        probability=(a-a.min())/(a.max()+1e-10) # literal native source score, not max-min
        rows.append({**{k:row[k] for k in ['slide_id','patient_id','global_patient_id','target']},
            'subset':subset,'source_slides':row['slide_id'],'task':task,'prediction':int(p.argmax()),
            **{'p_'+str(k):float(v) for k,v in enumerate(p)}})
        labels=assets.annotations(row)
        if labels is not None:
            patch.append({'slide_id':row['slide_id'],'positive_bag':row['target']%2==1,
                'labels':labels,'scores':probability})
    return rows,patch


def metrics(rows,patch,task,threshold=None,select_threshold=False):
    y=np.asarray([r['target'] for r in rows])
    # Global prediction competes against every seen head; task-specific AUROC
    # uses the original global softmax positive column, with no task mask.
    result={'accuracy':float(accuracy_score(y,[r['prediction'] for r in rows])),
        'positive_auroc':float(roc_auc_score(y%2,[r['p_'+str(2*task+1)] for r in rows])),
        'macro_f1':float(f1_score(y,[r['prediction'] for r in rows],average='macro',zero_division=0)),
        'prediction_scope':'all_seen_classes_without_task_id_mask','slides':len(rows)}
    positive=[r for r in patch if r['positive_bag']]
    if not positive:
        result['localization']={'available':False,'reason':'No held-out positive-bag patch ground truth; not reported as zero'}
        return result,.5 if threshold is None else threshold
    labels=np.concatenate([r['labels'] for r in positive]);scores=np.concatenate([r['scores'] for r in positive])
    if select_threshold and len(np.unique(labels))==2:
        fpr,tpr,ths=roc_curve(labels,scores);valid=np.isfinite(ths)
        threshold=float(ths[np.flatnonzero(valid)[np.argmax((tpr-fpr)[valid])]])
    if threshold is None:threshold=.5
    counts=[]
    for row in positive:
        truth=row['labels'].astype(bool);pred=row['scores']>=threshold
        tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
        counts.append({'slide_id':row['slide_id'],'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'foreground_iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'foreground_dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'mean_iou':.5*((tp/(tp+fp+fn) if tp+fp+fn else 1.)+(tn/(tn+fp+fn) if tn+fp+fn else 1.))})
    result['localization']={'available':True,'scope':'annotated positive bags; per-slide mean',
        'threshold':threshold,'threshold_source':'patient-disjoint VAL only; fixed after task completion',
        'patch_auroc':float(roc_auc_score(labels,scores)) if len(np.unique(labels))==2 else None,
        **{key:float(np.mean([r[key] for r in counts])) for key in ['foreground_iou','foreground_dice','mean_iou']},
        'slides':counts}
    return result,threshold


def _configuration(config):
    cfg=dict(config or {})
    if set(cfg)-{'epochs','threads'}:raise ValueError('CoMEL preserves native512D/two-block/rank16 GDAT and BPPL/OWLoRA optimizer; only full epoch budget/threads vary')
    value={'epochs':cfg.get('epochs',100),'threads':cfg.get('threads',2)}
    if any(type(v) is not int or v<1 for v in value.values()):raise ValueError('CoMEL epoch budget and threads must be positive integers')
    return value


def train(data,method,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None):
    from .engine import seed_everything,rng_state,restore_rng
    if method not in source.PROFILES:raise ValueError('Choose an explicit complete CoMEL source or SVD-basis-repaired profile')
    if type(seed) is not int or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):
        raise ValueError('CoMEL requires integer seed and positive optional actual-epoch interruption')
    cfg=_configuration(config);bundle,assets=_inputs(data,fm,split);dev=_device(device)
    out=Path(output).expanduser().resolve()
    request={'backend':'comel_complete_continual_reference','method':method,'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,
        'seed':seed,'device':str(dev),'config':cfg,'software':software_identity(),'bundle_sha256':bundle.fingerprint,
        'contract':assets.contract,'task_sequence':assets.tasks,'synthetic':bool(bundle.config.get('synthetic',False)),
        'test_used_for_training_or_selection':False,'training_instance_labels_used':False,
        'selection':'native final fixed-epoch snapshot before after_task; no test/VAL early stopping',
        'reference_repairs':'patient-disjoint VAL; complete task count; stable available-class/global-head metrics; explicit VAL-only localization thresholds',
        'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('CoMEL resume changed actual source, inputs, task order, objective or budget')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume)
    try:
        seed_everything(seed);torch.set_num_threads(cfg['threads']);last=out/'last.pt'
        if exists(last):
            state=load(last)
            if state['request']!=request:raise ValueError('CoMEL checkpoint request differs from actual run')
            network=_restore(bundle.dim,state,dev);source.restore_gradients(network,state['gradients'])
            task=state['next_task'];start=state['next_epoch'];history=state['history'];matrix=state['validation_matrix'];thresholds=state['thresholds']
            owner_data=state['owner'];snapshots=state['snapshots'];updates=state['updates'];resume_rng=state['rng']
        else:
            network=source.model(bundle.dim).to(dev);task=0;start=0;history=[];matrix=[];thresholds={};snapshots=[];updates=0;owner_data=None;resume_rng=None
        done_here=0
        while task<len(assets.tasks):
            dataset=Bags(assets,task);loader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
            # A between-task checkpoint contains the previous task's exact
            # post-expansion graph. Only then create the new fresh classifier.
            if owner_data is None:
                if resume_rng is not None:restore_rng(resume_rng);resume_rng=None
                if task>0:source.add_head(network)
                owner=source.context(network,task,cfg['epochs'],dev);source.optimizer(owner,loader)
            else:
                owner=source.context(network,task,cfg['epochs'],dev);source.optimizer(owner,loader)
                source.restore_owner(owner,owner_data);owner_data=None
                restore_rng(resume_rng);resume_rng=None
            for epoch in range(start,cfg['epochs']):
                dataset.observed=[];value=source.epoch(owner,loader,epoch);updates+=len(dataset)
                row={'task':task,'epoch':epoch+1,'updates':updates,'train_order':list(dataset.observed),**value}
                finished=epoch+1==cfg['epochs'];validation=None
                if (epoch+1)%50==0 or finished:
                    validation=[]
                    for seen in range(task+1):
                        predicted,annotated=predictions(network,assets,seen,'val',dev)
                        metric,threshold=metrics(predicted,annotated,seen,thresholds.get(str(seen)),select_threshold=seen==task)
                        if seen==task:thresholds[str(seen)]=threshold
                        validation.append({'task':seen,**metric})
                    row['validation']=validation
                history.append(row)
                if finished:
                    snapshot=save({**_layout(network),'task':task,'epochs':cfg['epochs'],'updates':updates,
                        'thresholds':copy.deepcopy(thresholds),'contract':assets.contract,'method':method},out/f'task_{task}_final.pt')
                    snapshots.append({'task':task,'path':str(snapshot),'sha256':digest_file(snapshot)})
                    matrix.append(validation);source.transition(owner,method)
                    next_task=task+1;next_epoch=0;next_owner=None
                else:next_task=task;next_epoch=epoch+1;next_owner=source.owner_state(owner)
                state={**_layout(network),'request':request,'next_task':next_task,'next_epoch':next_epoch,
                    'gradients':source.gradients(network),'owner':next_owner,'history':history,'validation_matrix':matrix,
                    'thresholds':thresholds,'snapshots':snapshots,'updates':updates,'rng':rng_state()}
                save(state,last);write_json(out/'history.json',history)
                write_json(out/'status.json',{'state':'running','completed_tasks':next_task,'next_epoch':next_epoch,
                    'updates':updates,'epochs_completed':len(history),'last_task':task,'last_epoch':epoch+1,'formal_admission':False})
                done_here+=1
                if stop_after is not None and done_here>=stop_after:raise Interrupted()
            task+=1;start=0;owner_data=None
        assets.verify_bytes()
        write_json(out/'selected.json',{'snapshots':snapshots,'thresholds':thresholds,'selection':request['selection']})
        write_json(out/'validation_continual.json',_continual(matrix))
        write_json(out/'status.json',{'state':'completed','completed_tasks':len(assets.tasks),'epochs_completed':len(history),
            'updates':updates,'test_evaluated':False,'formal_admission':False})
        return out
    except Interrupted:
        write_json(out/'status.json',{**read_json(out/'status.json'),'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


def _continual(matrix):
    count=len(matrix)
    if not count:return {'matrix':[],'average_final_accuracy':None,'backward_transfer':None}
    diagonal=[matrix[t][t]['accuracy'] for t in range(count)];final=[r['accuracy'] for r in matrix[-1]]
    return {'matrix':matrix,'average_final_accuracy':float(np.mean(final)),
        'backward_transfer':float(np.mean([final[t]-diagonal[t] for t in range(count-1)])) if count>1 else None,
        'forgetting':float(np.mean([max(matrix[s][t]['accuracy'] for s in range(t,count))-final[t] for t in range(count-1)])) if count>1 else None,
        'independent_patients_not_tasks_or_seeds':True}


def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything
    out=Path(run).resolve();req=read_json(out/'request.json')
    if subset not in {'val','test'} or req['backend']!='comel_complete_continual_reference' or read_json(out/'status.json')['state']!='completed':
        raise ValueError('Complete every actual CoMEL task before explicit held-out evaluation')
    if req['software']!=software_identity():raise ValueError('CoMEL evaluation software changed')
    bundle,assets=_inputs(req['data'],req['fm'],req['split'])
    if bundle.fingerprint!=req['bundle_sha256'] or assets.contract!=req['contract']:raise ValueError('CoMEL evaluation input/task protocol changed')
    seed_everything(req['seed']);torch.set_num_threads(req['config']['threads']);dev=_device(device)
    selected=read_json(out/'selected.json');matrix=[];final=[]
    if len(selected['snapshots'])!=len(assets.tasks):raise ValueError('CoMEL selected snapshots omit continual tasks')
    for item in selected['snapshots']:
        if digest_file(item['path'])!=item['sha256']:raise ValueError('CoMEL learned task snapshot changed')
        state=load(item['path']);task=item['task']
        if state['contract']!=assets.contract or state['task']!=task or state['method']!=req['method']:raise ValueError('CoMEL task checkpoint provenance changed')
        network=_restore(bundle.dim,state,dev);row=[]
        for seen in range(task+1):
            predicted,patch=predictions(network,assets,seen,subset,dev)
            metric,_=metrics(predicted,patch,seen,selected['thresholds'].get(str(seen)),select_threshold=False)
            row.append({'task':seen,**metric})
            if task==len(assets.tasks)-1:final.extend(predicted)
        matrix.append(row)
    assets.verify_bytes();result=_continual(matrix);result.update(subset=subset,threshold_selection='VAL_only',test_selection=False)
    pd.DataFrame(final).to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),result)
    return result

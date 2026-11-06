"""Full QPMIL-VL continual recipe implementation work; admission gates separate."""
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

from . import qpmilvl_source as source
from .qpmilvl_data import QPMILInputs
from .data import Bundle
from .io import read_json,write_json,digest_file
from .provenance import software_identity
from .akdpmp_driver import _device
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import save,load,exists,checkpoint_path,Interrupted


PROFILES={f'qpmilvl_conch_{pooling}_query_patient_reference':{'pooling':pooling} for pooling in ['max','mean']}
DEFAULT_PROFILE='qpmilvl_conch_max_query_patient_reference'


class Bags(Dataset):
    def __init__(self,assets,task,subset):self.assets=assets;self.rows=assets.records(task,subset);self.order=[]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];self.order.append(r['slide_id'])
        return torch.tensor([index],dtype=torch.int),self.assets.features(r),torch.tensor([r['target']],dtype=torch.long)


def loaders(assets,seed):
    result={};generators={}
    for t in range(len(assets.tasks)):
        for subset in ['train','val']:
            key=f'{t}:{subset}';g=torch.Generator().manual_seed(seed)
            result[key]=DataLoader(Bags(assets,t,subset),batch_size=1,shuffle=subset=='train',generator=g,num_workers=0)
            generators[key]=g
    return result,generators


def recipe(assets,config,method=DEFAULT_PROFILE):
    if method not in PROFILES:raise ValueError('Choose an explicit original QPMIL max/mean query profile')
    cfg=source.config();cfg['pooling']=PROFILES[method]['pooling'];n=len(assets.tasks)
    if set(config)-{'epochs','threads'}:raise ValueError('Preserve original QPMIL20/24/5 pool,16bag updates, objectives,Adam,plateauLR and final-model selection')
    budgets=config.get('epochs',[12]*n);budgets=[budgets]*n if type(budgets) is int else budgets
    if not isinstance(budgets,list) or len(budgets)!=n or any(type(e) is not int or e<1 for e in budgets):raise ValueError('Declare every actual QPMIL task epoch budget')
    cfg.update(epochs=budgets,dataset_names=[t['id'] for t in assets.tasks],dataset_subtype_num=[len(t['classes']) for t in assets.tasks],
        dataset_label_shift=[sum(len(p['classes']) for p in assets.tasks[:i]) for i in range(n)],
        lrs_patience=[3]*n,es_patience=[5]*n,task_num=0)
    # Local reporting only; no external telemetry or tSNE RNG side effects.
    cfg.update(check_feature=False,check_key_matching=False,print_mb_key_matching=False)
    for task in range(n):
        if len(assets.records(task,'train'))<16:raise ValueError('Original QPMIL16bag minibatch needs at least16actualTRAIN slides per task; no smaller substitute')
    return cfg


def pool_state(pools):return [{k:v.detach().cpu().clone() for k,v in p.state_dict().items()} for p in pools]

def restore_pools(pools,state):
    for p,v in zip(pools,state):p.load_state_dict(v,strict=True)


def gradients(pools):return [[None if p.grad is None else p.grad.detach().cpu().clone() for p in group] for group in pools]

def restore_gradients(pools,state):
    for group,values in zip(pools,state):
        if len(group)!=len(values):raise ValueError('Original QPMIL parameter ownership changed')
        for p,g in zip(group,values):p.grad=None if g is None else g.to(p.device).clone()


def validation(model,loader,device):
    model.eval();pred=[];labels=[];frequency=torch.zeros(model.cfg['pool_size'],dtype=torch.int)
    # Original eval_model keeps grad enabled but detaches predictions. It never
    # backpropagates, changes a parameter or fits a TEST threshold.
    for _,x,y in loader:
        value,indices=model([x.to(device)],eval=True);pred.append(value.detach().cpu());labels.append(y.squeeze(1))
        for index in indices.detach().cpu().flatten():frequency[index]+=1
    logits=torch.cat(pred);target=torch.cat(labels)
    loss=float(torch.nn.functional.cross_entropy(logits,target))
    predicted=(logits.softmax(1)[:,1]>.5).long() if logits.shape[1]==2 else logits.argmax(1)
    accuracy=float(np.mean((predicted==target).numpy()))
    return {'loss':loss,'accuracy':accuracy,'key_frequencies':frequency}


@source.reproducible_execution()
def train(data,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None,method=DEFAULT_PROFILE):
    from .engine import seed_everything,rng_state,restore_rng
    config=dict(config or {});threads=config.get('threads',2)
    if type(seed) is not int or type(threads) is not int or threads<1 or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):
        raise ValueError('Declare valid QPMIL seed, threads and actual epoch interruption point')
    bundle=Bundle(data,fm,split);assets=bundle.qpmilvl
    if assets is None:raise ValueError('Prepare actual QPMIL patient/task/features/text/CONCH bindings')
    cfg=recipe(assets,config,method);dev=_device(device);out=Path(output).resolve()
    request={'backend':'qpmilvl_complete_continual_reference','method':method,'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,
        'seed':seed,'device':str(dev),'threads':threads,'recipe':cfg,'contract':assets.contract,
        'bundle_sha256':bundle.fingerprint,'software':software_identity(),'runtime':source.runtime_identity(),'conch':assets.conch,
        'selection':'original last task epoch; native early stopping and combined VAL monitor; no best-model reload',
        'port':'single-process original feature I/O; local logging; no TSNE or external telemetry; TEST only at explicit evaluate',
        'original_training_minibatch':16,'synthetic':bool(bundle.config.get('synthetic',False)),
        'test_used_for_training_or_selection':False,'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('QPMIL resume changed original inputs, pretrained weights, complete recipe or source')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume)
    try:
        seed_everything(seed);torch.set_num_threads(threads)
        base=source.load_base(assets.conch['path'],assets.conch['sha256'],dev)
        data_loaders,generators=loaders(assets,seed);pools=source.parameter_pools(cfg,base)
        last=out/'last.pt';state=load(last) if exists(last) else None
        frequency={t['id']:[] for t in assets.tasks};val_frequency=copy.deepcopy(frequency);history=[];snapshots=[];task=0;start=0;updates=0;resume_rng=None
        if state:
            if state['request']!=request:raise ValueError('Actual QPMIL checkpoint identity changed')
            restore_pools(pools,state['pools']);restore_gradients(pools,state['gradients'])
            frequency=state['frequency'];val_frequency=state['val_frequency'];history=state['history'];snapshots=state['snapshots'];updates=state['updates']
            task=state['next_task'];start=state['next_epoch'];resume_rng=state['rng']
            for k,g in generators.items():g.set_state(state['generators'][k])
        done_here=0
        while task<len(assets.tasks):
            current=assets.tasks[task]['id'];task_cfg=copy.deepcopy(cfg);task_cfg['task_num']=task+1
            model=source.model(task_cfg,base,pools,assets.ensemble(task),frequency).eval()
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['adam_lr'],eps=cfg['adam_eps'],weight_decay=cfg['adam_weight_decay'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode=cfg['lrs_mode'],factor=cfg['lrs_factor'],patience=cfg['lrs_patience'][task],threshold=cfg['lrs_threshold'],threshold_mode=cfg['lrs_threshold_mode'])
            early=source.definitions().utilities.EarlyStopping(warmup=cfg['es_warmup'],patience=cfg['es_patience'][task],verbose=False,threshold=cfg['es_threshold'])
            if state and start:
                optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);early.__dict__.update(state['early'])
                restore_gradients(pools,state['gradients'])
            if resume_rng is not None:restore_rng(resume_rng);resume_rng=None
            for epoch in range(start,cfg['epochs'][task]):
                forgetting=[validation(model,data_loaders[f'{t}:val'],dev) for t in range(task)]
                loader=data_loaders[f'{task}:train'];loader.dataset.order=[];bags=[];labels=[];logs=[];counts=torch.zeros(20,dtype=torch.int)
                for _,x,y in loader:
                    bags.append(x.to(dev));labels.append(y.to(dev))
                    if len(bags)==16:
                        (_,indices,_),losses=source.original_update(model,optimizer,task_cfg,bags,labels);updates+=1;logs.append(losses)
                        if not all(np.isfinite(v) for v in losses.values()):raise FloatingPointError('Original QPMIL objective became nonfinite')
                        for index in indices.detach().cpu().flatten():counts[index]+=1
                        bags=[];labels=[]
                # Literal source drops an incomplete tail; it is disclosed in
                # history rather than silently flushing a smaller minibatch.
                if not logs:raise ValueError('Original full16bag QPMIL training produced no actual update')
                frequency[current].append(counts)
                val=validation(model,data_loaders[f'{task}:val'],dev);val_frequency[current].append(val['key_frequencies'])
                monitor=(val['loss']+(1-val['accuracy']))/2
                if epoch+1>cfg['lrs_warmup']:scheduler.step(monitor)
                early(epoch+1,monitor)
                row={'task':task,'epoch':epoch+1,'updates':updates,'loss_terms':logs,'validation_loss':val['loss'],
                    'validation_accuracy':val['accuracy'],'monitor_metric':monitor,'train_order':list(loader.dataset.order),
                    'dropped_native_incomplete_tail_slides':len(bags),'learning_rate':optimizer.param_groups[0]['lr'],
                    'previous_task_validation':[{'loss':r['loss'],'accuracy':r['accuracy']} for r in forgetting]};history.append(row)
                finished=early.stop() or epoch+1==cfg['epochs'][task]
                if early.save_ckpt():save({'pools':pool_state(pools),'task':task,'epoch':epoch+1,'monitor':monitor,'contract':assets.contract},out/f'task_{task}_best.pt')
                if finished:
                    selected=out/f'task_{task}_final.pt'
                    save({'pools':pool_state(pools),'task':task,'epoch':epoch+1,'frequency':frequency,'cfg':task_cfg,'contract':assets.contract},selected)
                    actual=checkpoint_path(selected);snapshots.append({'task':task,'path':str(actual),'sha256':digest_file(actual),
                        'actual_epochs':epoch+1,'maximum_epochs':cfg['epochs'][task]})
                state={'request':request,'pools':pool_state(pools),'gradients':gradients(pools),'next_task':task+1 if finished else task,
                    'next_epoch':0 if finished else epoch+1,'optimizer':optimizer.state_dict() if not finished else None,
                    'scheduler':scheduler.state_dict() if not finished else None,'early':copy.deepcopy(early.__dict__) if not finished else None,
                    'frequency':copy.deepcopy(frequency),'val_frequency':copy.deepcopy(val_frequency),'history':history,'snapshots':snapshots,
                    'updates':updates,'generators':{k:g.get_state() for k,g in generators.items()},'rng':rng_state()}
                save(state,last);write_json(out/'history.json',history);write_json(out/'status.json',{'state':'running','completed_tasks':state['next_task'],'epochs_completed':len(history),'updates':updates,'formal_admission':False})
                done_here+=1
                if stop_after is not None and done_here>=stop_after:raise Interrupted()
                if finished:break
            task+=1;start=0;state=None
        assets.verify_bytes();write_json(out/'selected.json',{'snapshots':snapshots,'selection':request['selection']})
        write_json(out/'status.json',{'state':'completed','completed_tasks':len(assets.tasks),'epochs_completed':len(history),'updates':updates,'test_evaluated':False,'formal_admission':False});return out
    except Interrupted:
        write_json(out/'status.json',{**read_json(out/'status.json'),'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


@source.reproducible_execution()
def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything
    from .comel_driver import _continual
    from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
    out=Path(run).resolve();req=read_json(out/'request.json')
    if subset not in {'val','test'} or req['backend']!='qpmilvl_complete_continual_reference' or read_json(out/'status.json')['state']!='completed':
        raise ValueError('Complete every original QPMIL task before explicit held-out evaluation')
    if req['software']!=software_identity() or req['runtime']!=source.runtime_identity():raise ValueError('QPMIL evaluation source/runtime changed')
    bundle=Bundle(req['data'],req['fm'],req['split']);assets=bundle.qpmilvl
    if assets is None:raise ValueError('Actual QPMIL manifest is missing')
    if assets.contract!=req['contract'] or bundle.fingerprint!=req['bundle_sha256']:raise ValueError('Actual QPMIL inputs or patient/label splits changed')
    dev=_device(device);seed_everything(req['seed']);torch.set_num_threads(req['threads'])
    base=source.load_base(assets.conch['path'],assets.conch['sha256'],dev);pools=source.parameter_pools(req['recipe'],base)
    chosen=read_json(out/'selected.json')['snapshots']
    if [r['task'] for r in chosen]!=list(range(len(assets.tasks))):raise ValueError('QPMIL final task models are missing or reordered')
    matrix=[];final=[]
    for item in chosen:
        if digest_file(item['path'])!=item['sha256']:raise ValueError('Actual QPMIL final task checkpoint bytes changed')
        state=load(item['path']);task=item['task']
        if state['task']!=task or state['contract']!=assets.contract:raise ValueError('Actual QPMIL final task checkpoint identity changed')
        restore_pools(pools,state['pools']);model=source.model(state['cfg'],base,pools,assets.ensemble(task),state['frequency']).eval();row=[]
        for seen_task in range(task+1):
            records=[];active=assets.seen(task);task_classes=assets.classes(seen_task);masked=[]
            for r in assets.records(seen_task,subset):
                logits,_=model([assets.features(r).unsqueeze(0).to(dev)],eval=True)
                if not torch.isfinite(logits).all():raise FloatingPointError('Actual QPMIL inference produced nonfinite logits')
                p=logits.detach().cpu().softmax(1)[0].numpy();restricted=logits.detach().cpu()[0,task_classes].softmax(0).numpy()
                record={**{k:r[k] for k in ['slide_id','patient_id','global_patient_id','target']},'subset':subset,
                    'source_slides':r['slide_id'],'task':seen_task,'prediction':int(p.argmax()),
                    **{'p_'+str(k):float(p[k]) if k<len(active) else 0. for k in range(len(bundle.classes))}}
                records.append(record);masked.append(task_classes[int(restricted.argmax())])
            truth=[r['target'] for r in records];auc=[]
            for cls in task_classes:
                y=[int(v==cls) for v in truth]
                if len(set(y))==2:auc.append(float(roc_auc_score(y,[r['p_'+str(cls)] for r in records])))
            row.append({'task':seen_task,'accuracy':float(accuracy_score(truth,[r['prediction'] for r in records])),
                'macro_f1':float(f1_score(truth,[r['prediction'] for r in records],average='macro',zero_division=0)),
                'macro_auroc':float(np.mean(auc)) if auc else None,'task_masked_accuracy':float(accuracy_score(truth,masked)),
                'slides':len(records),'prediction_scope':'primary all-seen global classes; separate task-masked diagnostic'})
            if task==len(assets.tasks)-1:final.extend(records)
        matrix.append(row)
    result={**_continual(matrix),'subset':subset,'test_selection':False,'selection':req['selection']};assets.verify_bytes()
    pd.DataFrame(final).to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),result);return result

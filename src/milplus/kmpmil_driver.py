"""KMP-MIL: literal complete epochs, selected rehearsal-free banks and exact resume."""
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

from . import kmpmil_source as source
from .data import Bundle
from .io import read_json,write_json,digest_file
from .provenance import software_identity
from .akdpmp_driver import _device
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import save,load,exists,checkpoint_path,Interrupted

PROFILES={f'kmpmil_conch_{pooling}_query_patient_reference':{'pooling':pooling} for pooling in ['max','mean']}
DEFAULT_PROFILE='kmpmil_conch_max_query_patient_reference'


class Bags(Dataset):
    def __init__(self,assets,task,subset):
        self.assets=assets;self.task=task;self.rows=assets.records(task,subset);self.order=[]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];self.order.append(r['slide_id'])
        # The literal source epoch adds the current task's global class shift.
        label=r['target']-self.assets.classes(self.task)[0]
        return torch.tensor([index],dtype=torch.int),self.assets.features(r),torch.tensor([label],dtype=torch.long)


def loaders(assets,seed):
    result={};generators={}
    for task in range(len(assets.tasks)):
        for subset in ['train','val']:
            key=f'{task}:{subset}';g=torch.Generator().manual_seed(seed)
            result[key]=DataLoader(Bags(assets,task,subset),batch_size=1,shuffle=subset=='train',generator=g,num_workers=0)
            generators[key]=g
    return result,generators


def recipe(assets,config,method=DEFAULT_PROFILE):
    if method not in PROFILES:raise ValueError('Choose an explicit original KMP max/mean retrieval profile')
    if set(config)-{'epochs','threads'}:raise ValueError('Preserve original semantic keys/24token/5matches/16bag/PFC/TC/routing and selected-bank recipe')
    n=len(assets.tasks);budgets=config.get('epochs',[20]*n);budgets=[budgets]*n if type(budgets) is int else budgets
    if not isinstance(budgets,list) or len(budgets)!=n or any(type(e) is not int or e<1 for e in budgets):raise ValueError('Declare each actual task epoch budget')
    cfg=source.config();cfg.update(pooling=PROFILES[method]['pooling'],epochs=budgets,dataset_names=[t['id'] for t in assets.tasks],
        dataset_subtype_num=[len(t['classes']) for t in assets.tasks],dataset_label_shift=[assets.classes(i)[0] for i in range(n)],
        num_classes=len(assets.bundle.classes),lrs_patience=[3]*n,es_patience=[10]*n,task_num=0)
    return cfg


def tc_state(tc):return [p.detach().cpu().clone() for p in tc]


def restore_tc(tc,state):
    if len(tc)!=len(state):raise ValueError('Actual TC task structure changed')
    with torch.no_grad():
        for p,v in zip(tc,state):p.copy_(v.to(p.device))


def learned_state(model):
    # This is the exact original Manager._get_state_dict key-selection rule.
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if any(part in k for part in ['key','prompt','tc_residuals','pfc'])}


def restore_model(model,state):
    actual=learned_state(model)
    if set(actual)!=set(state):raise ValueError('Actual original learned parameter/alias structure changed')
    with torch.no_grad():
        for k,v in model.state_dict().items():
            if k in state:
                if v.shape!=state[k].shape or v.dtype!=state[k].dtype:raise ValueError('Original learned tensor shape/dtype changed')
                v.copy_(state[k].to(v.device))


def gradients(model):return {k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in model.named_parameters()}


def restore_gradients(model,state):
    if set(state)!=dict(model.named_parameters()).keys():raise ValueError('Original parameter/gradient ownership changed')
    for k,p in model.named_parameters():p.grad=None if state[k] is None else state[k].to(p.device).clone()


def _context(model,cfg,dev,optimizer=None,loader=None):
    context=SimpleNamespace(model=model,cfg=cfg,device=dev,optimizer=optimizer,loss_function=torch.nn.CrossEntropyLoss())
    context._compute_metrics=source.definitions().metrics.MetricsMixin._compute_metrics.__get__(context)
    if loader is not None:
        context.current_dataset=cfg['dataset_names'][cfg['task_num']-1];context.data_loader={context.current_dataset:{'train':loader}}
    return context


def validation(model,loader,cfg,dev,task):
    definitions=source.definitions();ctx=_context(model,cfg,dev)
    collected=definitions.evaluation.EvaluationMixin.eval_model(ctx,model,loader)['pred']
    classes=cfg['dataset_subtype_num'][task];shift=cfg['dataset_label_shift'][task]
    collected['y']+=shift;collected['y_hat']=collected['y_hat'][:,shift:shift+classes]
    # Literal original metric path preserves >=0.5 binary tie, remapping and CE.
    return definitions.evaluation.EvaluationMixin._eval_and_print(ctx,collected,name='val/'+cfg['dataset_names'][task],if_binary=classes==2)


def train_epoch(model,optimizer,cfg,loader,dev):
    ctx=_context(model,cfg,dev,optimizer,loader);logs=[];batches=[];retrieval=[]
    def update(i_batch,xs,ys):
        prediction,indices,terms=source.definitions().observed_update(ctx,i_batch,xs,ys)
        if not all(np.isfinite(v) for v in terms.values()):raise FloatingPointError('Original KMP objective became nonfinite')
        logs.append(terms);batches.append(len(xs));retrieval.append(indices.detach().cpu().tolist());return prediction,indices
    ctx._update_network=update
    collected=source.definitions().training.TrainingMixin._train_each_epoch(ctx)['pred']
    task=cfg['task_num']-1;shift=cfg['dataset_label_shift'][task];classes=cfg['dataset_subtype_num'][task]
    collected['y_hat']=collected['y_hat'][:,shift:shift+classes]
    metrics=source.definitions().evaluation.EvaluationMixin._eval_and_print(ctx,collected,name='train/'+cfg['dataset_names'][task],if_binary=classes==2)
    return logs,batches,retrieval,metrics


def append_bank(banks,model):
    with torch.no_grad():
        banks['keys'].append(torch.cat([p.detach().cpu() for p in model.key],dim=0).clone())
        banks['prompts'].append(torch.cat([p.detach().cpu() for p in model.prompt],dim=0).clone())
        banks['gamma'].append(torch.cat([p.detach().cpu() for p in model.pfc.gamma_delta_pool],dim=0).clone())
        banks['beta'].append(torch.cat([p.detach().cpu() for p in model.pfc.beta_pool],dim=0).clone())


def universal(cfg,base,dev,banks,tc,ensemble):
    return source.definitions().builders.build_universal_kmp_mil(cfg,base,dev,base.dtype,banks['keys'],banks['prompts'],tc,ensemble,banks['gamma'],banks['beta'])


def task_model(cfg,base,dev,banks,tc,ensemble,task):
    return source.definitions().builders.build_task_kmp_mil(cfg,base,dev,base.dtype,banks['keys'][task],banks['prompts'][task],tc,ensemble,banks['gamma'][task],banks['beta'][task])


def train(data,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None,method=DEFAULT_PROFILE):
    from .engine import seed_everything,rng_state,restore_rng
    config=dict(config or {});threads=config.get('threads',2)
    if type(seed) is not int or type(threads) is not int or threads<1 or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):
        raise ValueError('Declare valid KMP seed, threads and actual epoch interruption point')
    bundle=Bundle(data,fm,split);assets=bundle.kmpmil
    if assets is None:raise ValueError('Prepare actual KMP patients/tasks/features/CONCH/class descriptions/semantic descriptors')
    cfg=recipe(assets,config,method);dev=_device(device);cfg['cuda_id']=dev.index or 0;out=Path(output).resolve()
    request={'backend':'kmpmil_complete_continual_reference','method':method,'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,
        'seed':seed,'device':str(dev),'threads':threads,'recipe':cfg,'contract':assets.contract,'bundle_sha256':bundle.fingerprint,
        'software':software_identity(),'runtime':source.runtime_identity(),'conch':assets.conch,
        'selection':'original current-task combined VAL monitor; reload best before appending learned KMP/PFC bank',
        'port':'single-process original feature I/O; local reporting; TEST deferred to explicit evaluation',
        'original_training_minibatch':16,'flush_incomplete_tail':True,'old_WSI_replay':False,
        'synthetic':bool(bundle.config.get('synthetic',False)),'test_used_for_training_or_selection':False,'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('KMP resume changed actual inputs, weights, complete recipe or source')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume)
    try:
        seed_everything(seed);torch.set_num_threads(threads)
        base=source.load_base(assets.conch['path'],assets.conch['sha256'],dev);data_loaders,generators=loaders(assets,seed)
        tc=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(len(t['classes']),512,dtype=base.dtype,device=dev),requires_grad=False) for t in assets.tasks])
        last=out/'last.pt';state=load(last) if exists(last) else None
        banks={'keys':[],'prompts':[],'gamma':[],'beta':[]};history=[];snapshots=[];task=0;start=0;updates=0;resume_rng=None
        if state:
            if state['request']!=request:raise ValueError('Actual KMP checkpoint identity changed')
            restore_tc(tc,state['tc']);banks=state['banks'];history=state['history'];snapshots=state['snapshots'];updates=state['updates']
            task=state['next_task'];start=state['next_epoch'];resume_rng=state['rng']
            for k,g in generators.items():g.set_state(state['generators'][k])
        done_here=0
        while task<len(assets.tasks):
            current=assets.tasks[task]['id'];task_cfg=copy.deepcopy(cfg);task_cfg['task_num']=task+1
            source.definitions().utilities.activate_current_tc_residual(tc,task+1)
            # At a task boundary the NEXT task's randomly initialized prompts
            # must be drawn after restoring saved RNG; midtask reconstruction
            # consumes throwaway initialization and restores RNG afterwards.
            if resume_rng is not None and start==0:restore_rng(resume_rng);resume_rng=None
            pools=source.task_pools(task_cfg,base,len(assets.descriptors(task)))
            model=source.definitions().model.KMPMIL(task_cfg,base,dev,pools[0],pools[1],tc,assets.ensemble(task),
                pfc_gamma_delta_pool=pools[2],pfc_beta_pool=pools[3],past_keys=list(banks['keys']),current_descriptors=assets.descriptors(task)).eval()
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['adam_lr'],eps=cfg['adam_eps'],weight_decay=cfg['adam_weight_decay'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode=cfg['lrs_mode'],factor=cfg['lrs_factor'],patience=cfg['lrs_patience'][task],threshold=cfg['lrs_threshold'],threshold_mode=cfg['lrs_threshold_mode'])
            early=source.definitions().utilities.EarlyStopping(warmup=cfg['es_warmup'],patience=cfg['es_patience'][task],verbose=False,threshold=cfg['es_threshold'])
            if state and start:
                restore_model(model,state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);early.__dict__.update(state['early']);restore_gradients(model,state['gradients'])
            if resume_rng is not None:restore_rng(resume_rng);resume_rng=None
            for epoch in range(start,cfg['epochs'][task]):
                loader=data_loaders[f'{task}:train'];loader.dataset.order=[]
                losses,batches,retrieval,train_metrics=train_epoch(model,optimizer,task_cfg,loader,dev);updates+=len(losses)
                val=validation(model,data_loaders[f'{task}:val'],task_cfg,dev,task)
                monitor=val['loss'] if cfg['only_val_loss'] else (val['loss']+(1-val['acc']))/2
                if not np.isfinite(monitor):raise FloatingPointError('Original KMP VAL monitor became nonfinite')
                if epoch+1>cfg['lrs_warmup']:scheduler.step(monitor)
                early(epoch+1,monitor)
                history.append({'task':task,'epoch':epoch+1,'updates':updates,'loss_terms':losses,'native_minibatch_sizes':batches,
                    'retrieval_indices':retrieval,'train_metrics':train_metrics,'validation':val,'monitor_metric':monitor,
                    'train_order':list(loader.dataset.order),'learning_rate':optimizer.param_groups[0]['lr']})
                best_path=out/f'task_{task}_best.pt'
                if early.save_ckpt():save({'model':learned_state(model),'task':task,'epoch':epoch+1,'monitor':monitor,'contract':assets.contract},best_path)
                finished=early.stop() or epoch+1==cfg['epochs'][task]
                if finished:
                    # Preserve last epoch evidence before literal best-model reload.
                    save({'model':learned_state(model),'task':task,'epoch':epoch+1,'contract':assets.contract},out/f'task_{task}_last.pt')
                    chosen=load(best_path)
                    if chosen['task']!=task or chosen['contract']!=assets.contract:raise ValueError('Actual selected KMP task identity changed')
                    restore_model(model,chosen['model']);append_bank(banks,model)
                    # Original Manager constructs the universal model here. Held-
                    # out TEST decoding is deferred, not used for future updates.
                    selected_model=universal(task_cfg,base,dev,banks,tc,assets.ensemble(task));del selected_model
                    selected=out/f'task_{task}_selected.pt'
                    save({'banks':copy.deepcopy(banks),'tc':tc_state(tc),'model':learned_state(model),'task':task,'selected_epoch':chosen['epoch'],
                        'actual_epochs':epoch+1,'cfg':task_cfg,'contract':assets.contract},selected)
                    actual=checkpoint_path(selected);snapshots.append({'task':task,'path':str(actual),'sha256':digest_file(actual),
                        'actual_epochs':epoch+1,'selected_epoch':chosen['epoch'],'maximum_epochs':cfg['epochs'][task]})
                state={'request':request,'model':learned_state(model),'gradients':gradients(model),'tc':tc_state(tc),'banks':copy.deepcopy(banks),
                    'next_task':task+1 if finished else task,'next_epoch':0 if finished else epoch+1,
                    'optimizer':optimizer.state_dict() if not finished else None,'scheduler':scheduler.state_dict() if not finished else None,
                    'early':copy.deepcopy(early.__dict__) if not finished else None,'history':history,'snapshots':snapshots,'updates':updates,
                    'generators':{k:g.get_state() for k,g in generators.items()},'rng':rng_state()}
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


@torch.no_grad()
def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything
    from .comel_driver import _continual
    from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
    out=Path(run).resolve();req=read_json(out/'request.json')
    if subset not in {'val','test'} or req['backend']!='kmpmil_complete_continual_reference' or read_json(out/'status.json')['state']!='completed':raise ValueError('Complete every original KMP task before explicit held-out evaluation')
    if req['software']!=software_identity() or req['runtime']!=source.runtime_identity():raise ValueError('KMP evaluation source/runtime changed')
    bundle=Bundle(req['data'],req['fm'],req['split']);assets=bundle.kmpmil
    if assets is None or assets.contract!=req['contract'] or bundle.fingerprint!=req['bundle_sha256']:raise ValueError('Actual KMP patient/task/features/text inputs changed')
    dev=_device(device);seed_everything(req['seed']);torch.set_num_threads(req['threads'])
    base=source.load_base(assets.conch['path'],assets.conch['sha256'],dev)
    tc=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(len(t['classes']),512,dtype=base.dtype,device=dev),requires_grad=False) for t in assets.tasks])
    chosen=read_json(out/'selected.json')['snapshots']
    if [r['task'] for r in chosen]!=list(range(len(assets.tasks))):raise ValueError('KMP selected tasks missing or reordered')
    matrix=[];final=[];task_specific=[]
    for item in chosen:
        if digest_file(item['path'])!=item['sha256']:raise ValueError('Actual KMP selected checkpoint bytes changed')
        state=load(item['path']);task=item['task'];active=assets.seen(task)
        if state['task']!=task or state['contract']!=assets.contract:raise ValueError('Actual KMP selected checkpoint identity changed')
        restore_tc(tc,state['tc']);model=universal(state['cfg'],base,dev,state['banks'],tc,assets.ensemble(task));row=[]
        for seen_task in range(task+1):
            task_classes=assets.classes(seen_task);records=[];masked=[]
            for r in assets.records(seen_task,subset):
                logits,_=model([assets.features(r).unsqueeze(0).to(dev)],eval=True)
                if not torch.isfinite(logits).all():raise FloatingPointError('Actual KMP inference became nonfinite')
                p=logits.cpu().softmax(1)[0].numpy();restricted=logits.cpu()[0,task_classes].softmax(0).numpy()
                record={**{k:r[k] for k in ['slide_id','patient_id','global_patient_id','target']},'subset':subset,'source_slides':r['slide_id'],
                    'task':seen_task,'prediction':int(p.argmax()),**{'p_'+str(k):float(p[k]) if k<len(active) else 0. for k in range(len(bundle.classes))}}
                records.append(record);masked.append(task_classes[int(restricted[1]>=.5)] if len(task_classes)==2 else task_classes[int(restricted.argmax())])
            truth=[r['target'] for r in records];auc=[]
            for cls in task_classes:
                y=[int(v==cls) for v in truth]
                if len(set(y))==2:auc.append(float(roc_auc_score(y,[r['p_'+str(cls)] for r in records])))
            row.append({'task':seen_task,'accuracy':float(accuracy_score(truth,[r['prediction'] for r in records])),
                'macro_f1':float(f1_score(truth,[r['prediction'] for r in records],average='macro',zero_division=0)),
                'macro_auroc':float(np.mean(auc)) if auc else None,'task_masked_accuracy':float(accuracy_score(truth,masked)),
                'slides':len(records),'prediction_scope':'all-seen global classes; separate task-masked diagnostic'})
            if task==len(assets.tasks)-1:
                final.extend(records)
                local_model=task_model(state['cfg'],base,dev,state['banks'],tc,assets.ensemble(task),seen_task)
                for r in assets.records(seen_task,subset):
                    logits,_=local_model([assets.features(r).unsqueeze(0).to(dev)],eval=True);p=logits.cpu()[0,task_classes].softmax(0).numpy()
                    pred=task_classes[int(p[1]>=.5)] if len(task_classes)==2 else task_classes[int(p.argmax())]
                    task_specific.append({**{k:r[k] for k in ['slide_id','patient_id','global_patient_id','target']},'subset':subset,'task':seen_task,
                        'prediction':pred,**{'p_'+str(k):float(p[task_classes.index(k)]) if k in task_classes else 0. for k in range(len(bundle.classes))}})
                del local_model
        matrix.append(row);del model
    result={**_continual(matrix),'subset':subset,'test_selection':False,'selection':req['selection'],
        'task_CL_scope':'final per-task selected memory bank and class mask; explicit labels do not update parameters'}
    assets.verify_bytes();pd.DataFrame(final).to_csv(out/(subset+'_predictions.csv'),index=False)
    pd.DataFrame(task_specific).to_csv(out/(subset+'_task_cl_predictions.csv'),index=False)
    write_json(out/(subset+'_metrics.json'),result);return result

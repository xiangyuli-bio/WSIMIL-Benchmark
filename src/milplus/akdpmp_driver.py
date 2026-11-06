"""Complete original AKD+PMP: selected teachers, native replay and bound memory."""
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score

from . import akdpmp_source as source
from .akdpmp_data import AKDInputs
from .data import Bundle
from .io import read_json,write_json,digest_file
from .provenance import software_identity
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import save,load,exists,checkpoint_path,cpu_state,Interrupted

PROFILES={f'akdpmp_{kind}_{preset}_patient_reference':{'kind':kind,'preset':preset}
    for kind in ['clam','transmil'] for preset in ['tcga','skin']}


class Bags(Dataset):
    def __init__(self,assets,task,subset):self.assets=assets;self.rows=assets.records(task,subset);self.observed=[]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];self.observed.append(r['slide_id'])
        return {'features':self.assets.features(r),'label':torch.tensor(r['target']),
            'slide_id':r['slide_id'],'global_patient_id':r['global_patient_id']}


def collate(batch):
    # Native mil_collate_fn, batch1: feature rows stay in original order.
    row=batch[0]
    return {k:[r[k] for r in batch] if isinstance(v,str) else
        (torch.LongTensor([r[k] for r in batch]) if v.ndim==0 else torch.cat([r[k] for r in batch])) for k,v in row.items()}


def loader(dataset,args,training):
    generator=torch.Generator().manual_seed(1)
    common={'batch_size':1,'num_workers':0,'generator':generator,'collate_fn':collate}
    if training and args.weighted_sample:
        labels=[r['target'] for r in dataset.rows];counts={y:labels.count(y) for y in set(labels)}
        weights=torch.DoubleTensor([len(labels)/counts[y] for y in labels])
        value=DataLoader(dataset,sampler=WeightedRandomSampler(weights,len(weights)),**common)
    else:value=DataLoader(dataset,shuffle=training,**common)
    return value,generator


def buffer_state(buffer):
    # Native buffers contain only tensors, arrays and primitive Python state.
    return copy.deepcopy(buffer.__dict__)


def _buffer(args,state=None):
    cls=source.definitions().utilities.ClsssIncrementalBuffer if args.buffer_balance else source.definitions().utilities.SimpleBuffer
    result=cls(args.buffer_size,device='cpu')
    if state is not None:
        if state['buffer_size']!=args.buffer_size:raise ValueError('AKD-PMP replay capacity changed')
        result.__dict__.update(state)
    return result


def _samples(buffer):
    if hasattr(buffer,'buffers'):return [r for values in buffer.buffers.values() for r in values]
    return list(buffer.buffer)


def verify_memory(buffer,assets,completed_task):
    for sample in _samples(buffer):
        sid=sample['slide_id'][0];row=assets.rows[sid];positions=sample['row_positions'];x=sample['features']
        if row['subset']!='train' or row['task']>completed_task or sample['global_patient_id'][0]!=row['global_patient_id'] or int(sample['label'])!=row['target']:
            raise ValueError('AKD-PMP replay contains held-out/future/incorrectly identified data')
        if sample['parent_sha256']!=row['sha256'] or sample['ordered_ids_sha256']!=row['ordered_ids_sha256']:
            raise ValueError('AKD-PMP pseudo-bag parent identity changed')
        if positions.ndim!=1 or positions.dtype!=torch.int64 or len(positions)!=len(x) or (positions<0).any() or (positions>=row['patches']).any():
            raise ValueError('AKD-PMP original pseudo-bag indices changed')
        if x.shape!=(len(positions),assets.bundle.dim) or not torch.isfinite(x).all():raise ValueError('AKD-PMP stored pseudo-bag features are invalid')
    if len(buffer)!=len(_samples(buffer)) or buffer.n_patches_total!=sum(len(r['features']) for r in _samples(buffer)):
        raise ValueError('AKD-PMP native replay accounting differs from actual stored features')


def refill(args,model,buffer,trainloader,assets,seen,task,device):
    native=source.definitions();last_indices=None;original_add=buffer.add
    def select(slide,attn,size,method):
        nonlocal last_indices
        last_indices=native.selection_indices(slide,attn,size,method)
        return slide[last_indices]
    def add(sample,index):
        row=assets.rows[sample['slide_id'][0]];sample=dict(sample)
        sample['row_positions']=(last_indices.detach().cpu().clone() if args.buffer_slide_size>0 else torch.arange(row['patches']))
        sample.update(parent_sha256=row['sha256'],ordered_ids_sha256=row['ordered_ids_sha256'],
            selection_method=args.distill_method if args.buffer_slide_size>0 else 'original_full_bag')
        original_add(sample,index)
    # Observe original selected indices and parent identity; all actual source
    # reservoir/quota/top-k/random sampling and stored feature values are intact.
    buffer.add=add
    try:native.refill(args,model,buffer,trainloader,seen,np.asarray(assets.classes(task),dtype=int),source.Fabric(device),select)
    finally:del buffer.add


def _device(value):
    device=torch.device(value)
    if device.type not in {'cpu','cuda'} or (device.type=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Use an available actual CPU or CUDA device; no runtime fallback')
    return device


def _gradients(model):return {n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()}


def _restore_gradients(model,state):
    parameters=dict(model.named_parameters())
    if parameters.keys()!=state.keys():raise ValueError('AKD-PMP parameter/gradient ownership changed')
    for name,p in parameters.items():p.grad=None if state[name] is None else state[name].to(p.device).clone()


def _modes(model):return {n:m.training for n,m in model.named_modules()}


def _restore_modes(model,modes):
    modules=dict(model.named_modules())
    if modules.keys()!=modes.keys():raise ValueError('AKD-PMP module identity changed')
    for n,m in modules.items():m.training=modes[n]


def inputs(data,fm=None,split=None):
    bundle=Bundle(data,fm,split)
    if getattr(bundle,'akdpmp',None) is None:raise ValueError('Prepare actual ordered AKD-PMP tasks and native patch identities')
    return bundle,bundle.akdpmp


def train(data,method,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None):
    from .engine import seed_everything,rng_state,restore_rng
    if method not in PROFILES:raise ValueError('Choose an explicit full AKD-PMP CLAM/TransMIL and TCGA/skin recipe')
    cfg=dict(config or {});threads=cfg.get('threads',2)
    if type(seed) is not int or type(threads) is not int or threads<1 or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):
        raise ValueError('Declare valid AKD-PMP seed, threads and actual-epoch interruption point')
    bundle,assets=inputs(data,fm,split);spec=PROFILES[method];args=source.arguments(spec['kind'],spec['preset'],len(bundle.classes),cfg,len(assets.tasks));args.seed=seed
    divisor=4 if args.distill_method in {'maxminrand'} else (2 if args.distill_method in {'maxmin','maxrand'} else 1)
    if args.buffer_slide_size>0 and any(int(args.buffer_slide_size*r['patches'])//divisor<(2 if divisor==1 else 1) for r in assets.rows.values() if r['subset']=='train'):
        raise ValueError('Actual TRAIN bags are too short for original maxrand pseudo-bags; no fake padding or changed sampling')
    out=Path(output).expanduser().resolve();dev=_device(device)
    request={'backend':'akdpmp_complete_continual_reference','method':method,'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,
        'seed':seed,'device':str(dev),'threads':threads,'recipe':vars(args),'software':software_identity(),
        'bundle_sha256':bundle.fingerprint,'contract':assets.contract,'task_sequence':assets.tasks,
        'synthetic':bool(bundle.config.get('synthetic',False)),'test_used_for_training_or_selection':False,
        'selection':'native validation loss with patience10 and stop_epoch=floor(task_budget/2); reload selected task model before memory refill',
        'reference_repairs':'patient-disjoint VAL; single-process deterministic feature I/O; no TEST decoding before explicit evaluate',
        'formal_admission':False}
    from .akdpmp_runtime import numerical_policy
    request['cpu_numerical_policy']=numerical_policy(threads) if dev.type=='cpu' else None
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('AKD-PMP resume changed inputs, source, original objectives, task sequence or complete budgets')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume)
    try:
        seed_everything(seed);torch.set_num_threads(threads);network=source.model(spec['kind'],bundle.dim,len(bundle.classes)).to(dev)
        last=out/'last.pt';history=[];snapshots=[];task=0;start=0;updates=0;state=None;resume_rng=None;memory=_buffer(args)
        if exists(last):
            state=load(last)
            if state['request']!=request:raise ValueError('AKD-PMP checkpoint request changed')
            network.load_state_dict(state['model'],strict=True);_restore_modes(network,state['modes']);_restore_gradients(network,state['gradients'])
            task=state['next_task'];start=state['next_epoch'];history=state['history'];snapshots=state['snapshots'];updates=state['updates'];memory=_buffer(args,state['memory']);resume_rng=state['rng']
            verify_memory(memory,assets,task-1 if start else task-1)
        done_here=0
        while task<len(assets.tasks):
            trainset=Bags(assets,task,'train');valset=Bags(assets,task,'val');trainloader,traingen=loader(trainset,args,True);valloader,valgen=loader(valset,args,False)
            budget=args.epochs[task];seen=assets.seen(task);previous=assets.seen(task-1) if task else [];unseen=list(set(range(len(bundle.classes)))-set(seen))
            optimizer=torch.optim.Adam(network.parameters(),lr=args.lr[task],weight_decay=args.weight_decay)
            early=source.definitions().utilities.EarlyStopping(patience=10,stop_epoch=budget//2)
            teacher=None
            if task:
                teacher=copy.deepcopy(network)
                if state is not None and start:
                    teacher.load_state_dict(state['teacher'],strict=True)
                    _restore_gradients(teacher,state['teacher_gradients'])
                for p in teacher.parameters():p.requires_grad=False
                teacher.eval()
            if state is not None and start:
                optimizer.load_state_dict(state['optimizer']);early.__dict__.update(state['early'])
                traingen.set_state(state['train_generator']);valgen.set_state(state['val_generator'])
            if resume_rng is not None:restore_rng(resume_rng);resume_rng=None
            for epoch in range(start,budget):
                metric=source.Mean();logger=source.Logger();fabric=source.Fabric(dev);trainset.observed=[]
                source.definitions().epoch(args,network,optimizer,trainloader,memory,teacher,np.asarray(seen,dtype=int),
                    np.asarray(previous,dtype=int),unseen,task,epoch,fabric,metric,logger)
                updates+=fabric.updates
                if any(not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()) for p in network.parameters()):
                    raise FloatingPointError('Original full AKD-PMP learned parameters/gradients became nonfinite')
                val_loss=source.Mean();val_acc=source.Accuracy()
                source.definitions().validation(args,network,valloader,np.asarray(seen,dtype=int),unseen,source.Fabric(dev),val_loss,val_acc)
                value=val_loss.mean()
                if not np.isfinite(value):raise FloatingPointError('Original AKD-PMP held-out selection loss became nonfinite')
                early(epoch,value);selected=out/f'task_{task}_selected.pt'
                if early.val_loss_min==value:save({'model':cpu_state(network),'task':task,'epoch':epoch+1,'val_loss':value,
                    'seen_classes':seen,'contract':assets.contract,'method':method},selected)
                row={'task':task,'epoch':epoch+1,'updates':updates,'train_loss':metric.mean(),'val_loss':value,
                    'val_accuracy':float(torch.tensor(val_acc.correct,dtype=torch.float32)/val_acc.count),'train_order':list(trainset.observed),
                    'optimizer_updates_this_epoch':fabric.updates,'native_objective_terms':logger.rows,
                    'early_stop':copy.deepcopy(early.__dict__)};history.append(row)
                finished=early.early_stop or epoch+1==budget
                if finished:
                    learned=load(selected);network.load_state_dict(learned['model'],strict=True);network.eval()
                    refill(args,network,memory,trainloader,assets,np.asarray(seen,dtype=int),task,dev);verify_memory(memory,assets,task)
                    chosen=checkpoint_path(selected);snapshots.append({'task':task,'path':str(chosen),'sha256':digest_file(chosen),
                        'selected_epoch':learned['epoch'],'actual_epochs':epoch+1,'maximum_epochs':budget,
                        'memory_samples':len(memory),'memory_patches':memory.n_patches_total})
                    next_task=task+1;next_epoch=0
                else:next_task=task;next_epoch=epoch+1
                state={'request':request,'model':cpu_state(network),'modes':_modes(network),'gradients':_gradients(network),
                    'next_task':next_task,'next_epoch':next_epoch,'optimizer':optimizer.state_dict() if not finished else None,
                    'teacher':cpu_state(teacher) if teacher is not None and not finished else None,
                    'teacher_gradients':_gradients(teacher) if teacher is not None and not finished else None,
                    'early':copy.deepcopy(early.__dict__) if not finished else None,'memory':buffer_state(memory),
                    'train_generator':traingen.get_state(),'val_generator':valgen.get_state(),'history':history,
                    'snapshots':snapshots,'updates':updates,'rng':rng_state()}
                save(state,last);write_json(out/'history.json',history);write_json(out/'status.json',{
                    'state':'running','completed_tasks':next_task,'next_epoch':next_epoch,'updates':updates,'epochs_completed':len(history),
                    'memory_samples':len(memory),'memory_patches':memory.n_patches_total,'formal_admission':False})
                done_here+=1
                if stop_after is not None and done_here>=stop_after:raise Interrupted()
                if finished:break
            task+=1;start=0;state=None
        assets.verify_bytes();write_json(out/'selected.json',{'snapshots':snapshots,'selection':request['selection']})
        write_json(out/'status.json',{'state':'completed','completed_tasks':len(assets.tasks),'epochs_completed':len(history),
            'updates':updates,'memory_samples':len(memory),'memory_patches':memory.n_patches_total,'test_evaluated':False,'formal_admission':False})
        return out
    except Interrupted:
        write_json(out/'status.json',{**read_json(out/'status.json'),'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything
    out=Path(run).resolve();req=read_json(out/'request.json')
    if subset not in {'val','test'} or read_json(out/'status.json')['state']!='completed' or req['backend']!='akdpmp_complete_continual_reference':
        raise ValueError('Complete all selected-model/memory transitions before held-out evaluation')
    if req['software']!=software_identity():raise ValueError('AKD-PMP evaluation source/runtime changed')
    bundle,assets=inputs(req['data'],req['fm'],req['split'])
    if bundle.fingerprint!=req['bundle_sha256'] or assets.contract!=req['contract']:raise ValueError('AKD-PMP actual evaluation inputs changed')
    spec=PROFILES[req['method']];dev=_device(device);seed_everything(req['seed']);torch.set_num_threads(req['threads'])
    chosen=read_json(out/'selected.json')['snapshots']
    if [r['task'] for r in chosen]!=list(range(len(assets.tasks))):raise ValueError('AKD-PMP selected tasks are incomplete or reordered')
    matrix=[];final=[]
    for item in chosen:
        if digest_file(item['path'])!=item['sha256']:raise ValueError('AKD-PMP selected model bytes changed')
        state=load(item['path']);task=item['task']
        if state['method']!=req['method'] or state['contract']!=assets.contract or state['task']!=task:raise ValueError('AKD-PMP selected checkpoint provenance changed')
        network=source.model(spec['kind'],bundle.dim,len(bundle.classes)).to(dev);network.load_state_dict(state['model'],strict=True);network.eval();row=[]
        with torch.no_grad():
            for seen_task in range(task+1):
                records=[];active=assets.seen(task);unseen=list(set(range(len(bundle.classes)))-set(active))
                for record in assets.records(seen_task,subset):
                    x=assets.features(record).to(dev)
                    value=network(x,seen_classes=np.asarray(active,dtype=int)) if spec['kind']=='clam' else network(x)
                    logits=value['logits'];logits[:,unseen]=-100
                    if not torch.isfinite(logits).all():raise FloatingPointError('Original AKD-PMP evaluation produced nonfinite logits')
                    p=logits.softmax(1)[0].cpu().numpy();records.append({**{k:record[k] for k in ['slide_id','patient_id','global_patient_id','target']},
                        'subset':subset,'source_slides':record['slide_id'],'task':seen_task,'prediction':int(p.argmax()),**{'p_'+str(k):float(v) for k,v in enumerate(p)}})
                metric={'task':seen_task,'accuracy':float(accuracy_score([r['target'] for r in records],[r['prediction'] for r in records])),
                    'macro_f1':float(f1_score([r['target'] for r in records],[r['prediction'] for r in records],average='macro',zero_division=0)),
                    'slides':len(records),'prediction_scope':'all_seen_global_classes_without_task_mask'}
                auc=[]
                for cls in assets.classes(seen_task):
                    labels=[int(r['target']==cls) for r in records]
                    if len(set(labels))==2:auc.append(float(roc_auc_score(labels,[r['p_'+str(cls)] for r in records])))
                metric['macro_auroc']=float(np.mean(auc)) if auc else None
                row.append(metric)
                if task==len(assets.tasks)-1:final.extend(records)
        matrix.append(row)
    from .comel_driver import _continual
    result={**_continual(matrix),'subset':subset,'test_selection':False};assets.verify_bytes()
    pd.DataFrame(final).to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),result);return result

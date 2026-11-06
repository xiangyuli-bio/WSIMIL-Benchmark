"""Full raw max-MIL and original dual-loss PMIL stages with recoverable state.

The source's validation-backprop call and overwritten best-model export are
explicitly repaired here; this is a patient-disjoint reference recipe, not a
claim of bitwise equivalence to those unsafe source main() behaviors.
"""
import copy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from .icmil_coupled_training import save,load,exists,checkpoint_path,cpu_state
from .io import digest_file,write_json,atomic_torch_save
from .pmil_prototypical_source import encoder,patch_tensor,select_indices,bag_model,bag_step,bag_objective


class Patches(Dataset):
    def __init__(self,assets,rows,references=None):
        self.assets=assets;self.rows={r['slide_id']:r for r in rows}
        self.references=references if references is not None else [(r['slide_id'],i) for r in rows for i in range(assets.count(r))]

    def __len__(self):return len(self.references)

    def __getitem__(self,index):
        sid,patch=self.references[index];row=self.rows[sid]
        with self.assets.open(row) as pixels:value=patch_tensor(pixels[patch])
        return value,torch.tensor(row['target'],dtype=torch.long)


@torch.no_grad()
def infer(model,dataset,device,batch_size=256):
    model.eval();scores=[];features=[]
    for x,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
        logits,value=model(x.to(device))
        scores.append(logits.softmax(1)[:,1].cpu());features.append(value.cpu())
    score=torch.cat(scores).numpy();feature=torch.cat(features).numpy()
    if not np.isfinite(score).all() or not np.isfinite(feature).all():raise FloatingPointError('PMIL full encoder inference became nonfinite')
    return score,feature


def selected_references(dataset,scores,k):
    if len(dataset)!=len(scores):raise ValueError('PMIL patch scores lost actual input row alignment')
    result=[];begin=0
    for sid,row in dataset.rows.items():
        count=dataset.assets.count(row)
        result.extend(dataset.references[begin+i] for i in select_indices(scores[begin:begin+count],k))
        begin+=count
    if begin!=len(dataset):raise ValueError('PMIL expected complete original per-slide patch ordering')
    return result


@torch.no_grad()
def patch_validation(model,assets,rows,device):
    full=Patches(assets,rows);scores,_=infer(model,full,device)
    chosen=Patches(assets,rows,selected_references(full,scores,1));losses=[]
    for x,y in DataLoader(chosen,batch_size=128,shuffle=True,num_workers=0):
        logits,_=model(x.to(device));losses.append((float(F.cross_entropy(logits,y.to(device))),len(x)))
    loss=sum(a*n for a,n in losses)/len(chosen)
    maximum=[];begin=0
    for row in rows:
        count=assets.count(row);maximum.append(float(scores[begin:begin+count].max()));begin+=count
    accuracy=float(np.mean([(p>=.5)==bool(r['target']) for p,r in zip(maximum,rows)]))
    return loss,accuracy


def encoder_phase(bundle,assets,output,epochs,device,progress):
    from .engine import rng_state,restore_rng
    out=Path(output);out.mkdir(parents=True,exist_ok=True);last=out/'last.pt';selected=out/'selected.pt'
    state=load(last) if exists(last) else None
    if state is not None and state['finished']:
        restore_rng(state['rng']);return checkpoint_path(selected)
    model=encoder(assets.weights).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=.1)
    train_rows=bundle.records('train');val_rows=bundle.records('val')
    full=Patches(assets,train_rows);history=[];start=0;updates=0;best=None;bad=0;orders=[]
    if state is not None:
        if state['epochs']!=epochs:raise ValueError('PMIL max-MIL full epoch budget changed')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        history=state['history'];start=state['epoch'];updates=state['updates'];best=state['best'];bad=state['bad'];orders=state['orders'];restore_rng(state['rng'])
    for epoch in range(start,epochs):
        probabilities,_=infer(model,full,device)
        references=selected_references(full,probabilities,1)
        # Source does both Python random.sample and the shuffled Torch loader.
        references=random.sample(references,len(references));orders.append(copy.deepcopy(references))
        selected_dataset=Patches(assets,train_rows,references);loss_sum=0.;seen=0;model.train()
        for x,y in DataLoader(selected_dataset,batch_size=128,shuffle=True,num_workers=0):
            logits,_=model(x.to(device));loss=F.cross_entropy(logits,y.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Original PMIL max-MIL patch loss is nonfinite')
            optimizer.zero_grad();loss.backward();optimizer.step();updates+=1
            loss_sum+=float(loss.detach())*len(x);seen+=len(x)
        scheduler.step();validation=None
        if (epoch+1)%5==0 or epoch+1==epochs:
            val_loss,accuracy=patch_validation(model,assets,val_rows,device)
            validation={'patch_ce':val_loss,'slide_accuracy':accuracy}
            # Literal EarlyStopping tie policy, but only no-grad VAL patients;
            # the public main's subsequent unconditional overwrite is removed.
            if best is None or val_loss<=best:
                best=val_loss;bad=0;save({'model':cpu_state(model),'selected_epoch':epoch,'val_patch_ce':val_loss},selected)
            else:bad+=1
        finished=epoch+1==epochs or bad>=3
        row={'epoch':epoch,'train_loss':loss_sum/seen,'validation':validation,'updates':updates,
            'selected_train_patches':len(references),'learning_rate':optimizer.param_groups[0]['lr']};history.append(row)
        state={'epochs':epochs,'epoch':epoch+1,'model':cpu_state(model),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'history':history,'updates':updates,'orders':orders,
            'best':best,'bad':bad,'rng':rng_state(),'finished':finished,'validation_backprop':False}
        save(state,last);write_json(out/'history.json',history);progress.record(out.name,epoch+1,history,finished)
        if finished:break
    return checkpoint_path(selected)


def export_encoder(selected,output):
    out=Path(output);value=load(selected)['model']
    if out.exists():
        existing=torch.load(out,map_location='cpu',weights_only=True)
        if set(existing)!={'state_dict'} or set(existing['state_dict'])!=set(value) or any(not torch.equal(existing['state_dict'][k],v) for k,v in value.items()):
            raise ValueError('PMIL selected encoder export changed')
    else:atomic_torch_save({'state_dict':value},out)
    return out


class FeatureBags(Dataset):
    def __init__(self,records):self.records=records
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index];path=Path(row['path'])
        if digest_file(path)!=row['sha256']:raise ValueError('PMIL actual learned feature bytes changed')
        x=np.load(path,allow_pickle=False)
        if x.dtype!=np.float32 or x.shape!=(row['patches'],512) or not np.isfinite(x).all():raise ValueError('PMIL original learned feature shape/order changed')
        return torch.from_numpy(x.copy()),torch.tensor(row['target'],dtype=torch.long)


@torch.no_grad()
def bag_validation(model,records,prototypes,device):
    model.eval();losses=[];correct=[]
    for x,y in DataLoader(FeatureBags(records),batch_size=1,shuffle=False,num_workers=0):
        values=bag_objective(model,x[0].to(device),prototypes,y.to(device))
        losses.append(float(values['loss']));correct.append(bool((values['bag_logits'].softmax(1)[0,1]>=.5)==y.to(device)[0]))
    return float(np.mean(losses)),float(np.mean(correct))


def bag_phase(assets,selected_encoder,feature_index,prototype_features,output,epochs,device,progress):
    from .engine import rng_state,restore_rng
    out=Path(output);out.mkdir(parents=True,exist_ok=True);last=out/'last.pt';selected=out/'selected.pt'
    state=load(last) if exists(last) else None
    if state is not None and state['finished']:
        restore_rng(state['rng']);return checkpoint_path(selected)
    prototypes=torch.from_numpy(prototype_features.copy()).to(device)
    model=bag_model(assets.weights,selected_encoder,prototypes).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=.1)
    train=[r for r in feature_index if r['subset']=='train'];val=[r for r in feature_index if r['subset']=='val']
    history=[];start=0;updates=0;best=0.
    if state is not None:
        if state['epochs']!=epochs:raise ValueError('PMIL complete30epoch dual-loss budget changed')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];history=state['history'];best=state['best'];updates=state['updates'];restore_rng(state['rng'])
    for epoch in range(start,epochs):
        terms=[]
        for x,y in DataLoader(FeatureBags(train),batch_size=1,shuffle=True,num_workers=0):
            terms.append(bag_step(model,x[0].to(device),prototypes,y.to(device),optimizer));updates+=1
        scheduler.step();val_loss,accuracy=bag_validation(model,val,prototypes,device)
        row={'epoch':epoch,**{k:float(np.mean([t[k] for t in terms])) for k in ['loss','bag_ce','max_ce']},
            'validation_loss':val_loss,'validation_accuracy':accuracy,'updates':updates,'learning_rate':optimizer.param_groups[0]['lr']}
        history.append(row)
        if accuracy>=best:
            best=accuracy;save({'model':cpu_state(model),'selected_epoch':epoch,'val_accuracy':accuracy},selected)
        state={'epochs':epochs,'epoch':epoch+1,'model':cpu_state(model),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'updates':updates,'history':history,'best':best,'rng':rng_state(),
            'finished':epoch+1==epochs,'source_train_eval_mode':not model.training}
        save(state,last);write_json(out/'history.json',history);progress.record(out.name,epoch+1,history,state['finished'])
    return checkpoint_path(selected)

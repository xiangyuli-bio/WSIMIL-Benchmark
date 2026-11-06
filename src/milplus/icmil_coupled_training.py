"""Original fixed-epoch MIL/Adam and three-KL encoder phases with full state."""
import copy
import hashlib
from pathlib import Path
import uuid

import numpy as np
import torch

from .icmil_coupled_source import BagModel,Distiller,bag_step
from .icmil_coupled_stages import CachedBags,PatchViews,bag_loader,patch_loader,bag_validation,distillation_validation
from .io import atomic_torch_save,digest_file,digest_object,read_json,write_json


def exists(path):return Path(path).with_suffix('.checkpoint.json').is_file()


def checkpoint_path(path):
    path=Path(path)
    if path.is_file():return path
    record=read_json(path.with_suffix('.checkpoint.json'));name=record['payload']
    if Path(name).name!=name or not name.startswith(path.stem+'.'):
        raise ValueError('ICMIL checkpoint generation escapes its owned stage')
    actual=path.parent/name
    if actual.is_symlink() or digest_file(actual)!=record['sha256']:
        raise ValueError('ICMIL checkpoint generation or pointer checksum changed')
    return actual


def load(path):
    path=checkpoint_path(path)
    if read_json(path.with_suffix(path.suffix+'.sha256.json'))['sha256']!=digest_file(path):
        raise ValueError('Owned ICMIL learned checkpoint bytes changed')
    return torch.load(path,map_location='cpu',weights_only=False)


def save(value,path):
    path=Path(path);pointer=path.with_suffix('.checkpoint.json')
    previous=read_json(pointer) if pointer.exists() else None
    actual=path.with_name(path.stem+'.'+uuid.uuid4().hex+path.suffix)
    atomic_torch_save(value,actual);sha=digest_file(actual)
    write_json(actual.with_suffix(actual.suffix+'.sha256.json'),{'sha256':sha})
    # The immutable payload and its checksum are complete before committing a
    # single atomic pointer. A kill between these writes retains the previous
    # valid checkpoint instead of pairing new weights with an old checksum.
    write_json(pointer,{'payload':actual.name,'sha256':sha})
    if previous is not None:
        old=path.parent/previous['payload']
        if old.name!=previous['payload'] or not old.name.startswith(path.stem+'.') or old.is_symlink():
            raise ValueError('Previous ICMIL checkpoint generation is not owned by this stage')
        # Ordinary last/best checkpoint replacement, after the new generation
        # is durable. Failed-run artifacts and unrelated snapshots are never
        # searched for or removed; interrupted orphan generations are retained.
        old.unlink();old.with_suffix(old.suffix+'.sha256.json').unlink()
    return actual


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


class Interrupted(Exception):pass


class Progress:
    def __init__(self,output,stop_after):self.root=Path(output);self.stop_after=stop_after;self.completed=0

    def record(self,phase,epoch,history,finished):
        self.completed+=1
        write_json(self.root/'status.json',{'state':'running','phase':phase,'phase_completed_epochs':epoch,
            'phase_finished':finished,'epochs_this_invocation':self.completed,'last_epoch':history[-1]})
        if self.stop_after is not None and self.completed>=self.stop_after:raise Interrupted()


def bag_phase(kind,feature_index,output,epochs,device,progress,initial_state=None):
    from .engine import rng_state,restore_rng
    out=Path(output);out.mkdir(parents=True,exist_ok=True);last=out/'last.pt';selected=out/'selected.pt'
    initial_sha=None if initial_state is None else digest_object({k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()} for k,v in initial_state.items()})
    state=load(last) if exists(last) else None
    if state is not None and state['initial_state_sha256']!=initial_sha:
        raise ValueError('ICMIL branch-specific previous-MIL initialization changed')
    if state is not None and state['finished']:
        restore_rng(state['rng']);return checkpoint_path(selected)
    model=BagModel(kind).to(device);optimizers=model.optimizers()
    if initial_state is not None:model.load_state_dict(initial_state,strict=True)
    if not exists(out/'initial_model.pt'):
        save({'model':cpu_state(model),'initial_state_sha256':initial_sha,
            'initialization':'fresh_source_constructor' if initial_state is None else 'previous_selected_MIL_checkpoint'},out/'initial_model.pt')
    train=CachedBags(feature_index,'train',None if state is None else state['orders']['train'])
    val=CachedBags(feature_index,'val',None if state is None else state['orders']['val'])
    trainloader=bag_loader(train);valloader=bag_loader(val)
    start=0;history=[];best=None;updates=0
    if state is not None:
        if state['kind']!=kind or state['epochs']!=epochs:raise ValueError('ICMIL bag architecture or complete epoch budget changed')
        model.load_state_dict(state['model'],strict=True)
        for opt,s in zip(optimizers,state['optimizers']):opt.load_state_dict(s)
        start=state['epoch'];history=state['history'];best=state['best'];updates=state['updates'];restore_rng(state['rng'])
    for epoch in range(start,epochs):
        model.train();losses=[]
        for x,y in trainloader:
            values=bag_step(model,x.squeeze(0).to(device),y.to(device),optimizers,legacy_adam14=kind=='dtfd')
            losses.append(values['loss']);updates+=values['optimizer_updates']
        auc=bag_validation(model,valloader,device)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'val_auc':auc,'updates':updates,'bags':len(train)};history.append(row)
        if best is None or auc>best:
            best=auc;save({'kind':kind,'model':cpu_state(model),'selected_epoch':epoch,'val_auc':auc},selected)
        state={'kind':kind,'epochs':epochs,'epoch':epoch+1,'model':cpu_state(model),'initial_state_sha256':initial_sha,
            'optimizers':[o.state_dict() for o in optimizers],'history':history,'best':best,'updates':updates,
            'orders':{'train':copy.deepcopy(train.orders),'val':copy.deepcopy(val.orders)},
            'rng':rng_state(),'finished':epoch+1==epochs}
        save(state,last);write_json(out/'history.json',history)
        progress.record(out.name,epoch+1,history,state['finished'])
    return checkpoint_path(selected)


def distillation_phase(bundle,assets,encoder_state,bag,output,epochs,device,progress):
    from .engine import rng_state,restore_rng
    out=Path(output);out.mkdir(parents=True,exist_ok=True);last=out/'last.pt';selected=out/'selected_encoder.pt'
    state=load(last) if exists(last) else None
    if state is not None and state['finished']:
        restore_rng(state['rng']);return checkpoint_path(selected)
    model=Distiller(assets.weights,encoder_state,bag).to(device);optimizer=model.optimizer()
    train=PatchViews(bundle,assets,'train');val=PatchViews(bundle,assets,'val')
    trainloader=patch_loader(train);valloader=patch_loader(val)
    start=0;history=[];best=None;updates=0
    if state is not None:
        if state['epochs']!=epochs:raise ValueError('ICMIL encoder complete epoch budget changed')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        for name,module in model.named_modules():module.training=state['training_modes'][name]
        start=state['epoch'];history=state['history'];best=state['best'];updates=state['updates'];restore_rng(state['rng'])
    for epoch in range(start,epochs):
        model.student_encoder.train();terms=[]
        for original,view1,unused_view2 in trainloader:
            value=model.step(original.to(device),view1.to(device),optimizer);updates+=1;terms.append(value)
        mae=distillation_validation(model,valloader,device)
        row={'epoch':epoch,'val_teacher_probability_mae':mae,'updates':updates,'patches':len(train),
            **{k:float(np.mean([v[k] for v in terms])) for k in terms[0]}}
        history.append(row)
        if best is None or mae<best:
            best=mae;save({'encoder':cpu_state(model.student_encoder),'selected_epoch':epoch,'val_mae':mae,
                'student_reduction':cpu_state(model.student_reduction),'student_classifier':cpu_state(model.student_classifier)},selected)
        state={'epochs':epochs,'epoch':epoch+1,'model':cpu_state(model),'optimizer':optimizer.state_dict(),
            'history':history,'best':best,'updates':updates,'rng':rng_state(),'finished':epoch+1==epochs,
            'training_modes':{n:m.training for n,m in model.named_modules()}}
        save(state,last);write_json(out/'history.json',history)
        progress.record(out.name,epoch+1,history,state['finished'])
    return checkpoint_path(selected)

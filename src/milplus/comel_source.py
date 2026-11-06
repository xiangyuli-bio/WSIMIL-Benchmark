"""Full pinned GDAT/BPPL/OWLoRA mechanisms, with explicit SVD-basis variants."""
from contextlib import suppress
from functools import lru_cache,partial
import copy
import time
from types import SimpleNamespace
from typing import Optional,Tuple,Union,List
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions,verified_source

COMMIT='9fd667994eb57e3960e36970a9509a8217d84a22'
PROFILES={'comel_owlora_source_patient_reference','comel_owlora_svd_basis_repaired'}


class AverageMeter:
    """The timm meter arithmetic used by the source; logging only."""
    def __init__(self):self.val=0.;self.sum=0.;self.count=0;self.avg=0.
    def update(self,value,n=1):
        self.val=value;self.sum+=value*n;self.count+=n;self.avg=self.sum/self.count


@lru_cache(maxsize=1)
def definitions():
    _,commit=verified_source('comel')
    if commit!=COMMIT:raise ValueError('CoMEL requires its pinned author GDAT/BPPL/OWLoRA source')
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'partial':partial,
        'Optional':Optional,'Tuple':Tuple,'Union':Union,'List':List}
    lora=load_definitions('comel','continual/modules/cl_modules/lora.py',
        ['LoRALinearLayer','LoRAWeightedLinearLayer','LoRACompatibleLinear','LoRACompatibleIncrementalLinear'],ns)
    attention=load_definitions('comel','continual/modules/cl_modules/datten.py',
        ['Attention','AttentionGated','DAttention'],ns)
    from .layers import DropPath
    network=load_definitions('comel','continual/modules/cl_modules/cdatmil_owlora.py',
        ['initialize_weights','Mlp','CDATAttention','CDATBlock','CDATMIL'],
        {**ns,'DropPath':DropPath,'trunc_normal_':nn.init.trunc_normal_,
         'DAttention':attention.DAttention,'LoRAWeightedLinearLayer':lora.LoRAWeightedLinearLayer,
         'LoRACompatibleIncrementalLinear':lora.LoRACompatibleIncrementalLinear})
    # Load the actual author optimizer, epoch, penalty and after-task methods
    # without unrelated imports, CLI launchers, W&B or external I/O. Its parent
    # constructor is never used; the complete driver supplies audited inputs.
    trainer=load_definitions('comel','continual/main/continual_bag/cdatmil_ppl_owlora_trainer.py',
        ['CDATMIL_PPL_OWLoRA_Trainer'],{**ns,'CDATMIL_PPL_FT_Trainer':object,
            'AverageMeter':AverageMeter,'time':time,'OrderedDict':OrderedDict})
    return SimpleNamespace(network=network,lora=lora,attention=attention,
        trainer=trainer.CDATMIL_PPL_OWLoRA_Trainer)


def model(dimension):
    if type(dimension) is not int or dimension<1:raise ValueError('CoMEL needs the actual declared FM feature dimension')
    # These are actual launcher values, including qkv_bias=False from argparse.
    # rank=8 is ignored in the released constructor: its effective rank is16.
    return definitions().network.CDATMIL(input_dim=dimension,mlp_dim=512,n_classes=2,
        dropout=.25,n_layers=2,n_heads=8,da_act='tanh',qkv_bias=False,
        trans_dropout=.1,drop_path=0.)


def context(network,current_task,epochs,device):
    source=definitions();owner=object.__new__(source.trainer)
    owner.model=network;owner.current_task=current_task;owner.device=torch.device(device)
    owner.T=1.;owner.tau1=.95;owner.tau=.5;owner.tau2=.5;owner.lambda3=1.
    # Source's parent never honors its early-stop flag and saves the final
    # fixed100-epoch model. The unused callback is omitted; all actual source
    # Adam/cosine/CE/BPPL/orthogonality mathematics run unchanged below.
    owner.args=SimpleNamespace(datasets='tcga',opt='adam',lr=1e-4,weight_decay=1e-5,
        lr_sche='cosine',num_epoch=epochs,lr_supi=False,early_stopping=False,
        patch_shuffle=False,group_shuffle=False,model='cdatmil',loss='ce',
        cls_alpha=1.,aux_alpha=1.,train_patch_epoch=0,seg_coef=1.,
        accumulation_steps=1,clip_grad=0.,log_iter=100,wandb=False)
    return owner


def optimizer(owner,train_loader):
    owner.train_loader=train_loader
    definitions().trainer.set_optimizer(owner,owner.args)
    return owner.optimizer


def epoch(owner,loader,index):
    if type(index) is not int or index<0:raise ValueError('CoMEL epoch index must be nonnegative')
    value,_,_=definitions().trainer.train_loop(owner,owner.args,owner.model,loader,owner.optimizer,
        owner.device,suppress,nn.CrossEntropyLoss(),None,owner.scheduler,0,index,None)
    if not np.isfinite(value):raise FloatingPointError('Original full CoMEL training loss became nonfinite')
    if any(not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()) for p in owner.model.parameters()):
        raise FloatingPointError('Original CoMEL parameters or active/inactive accumulated gradients became nonfinite')
    return {'loss':float(value),'tau2':float(owner.tau2),'learning_rate':owner.optimizer.param_groups[0]['lr']}


def repaired_initial_basis(network):
    """Source construction/RNG/truncation, with the absent U/V/S assignment added."""
    cls=definitions().lora.LoRAWeightedLinearLayer
    for name,module in network.update_modules.items():
        weight=module.weight.data
        u,s,vh=torch.linalg.svd(weight,full_matrices=False)
        dim_max=min(u.size()+vh.size());energy=(s**2).sum()
        if not torch.isfinite(energy) or energy<=0:raise ValueError('CoMEL source SVD energy normalization is undefined')
        for d in range(dim_max):
            if (s[:d]**2).sum()/energy>network.eps_singular:break
        d=max(d,1);u,s,vh=u[:,:d],s[:d],vh[:d]
        layer=cls(in_features=module.in_features,out_features=module.out_features,rank=d).to(module.weight.device)
        module.add_lora(layer)
        with torch.no_grad():
            layer.up.weight.copy_(u);layer.down.weight.copy_(vh);layer.scales.copy_(s)
        module.weight.data=(u*s)@vh


def transition(owner,profile):
    if profile not in PROFILES:raise ValueError('Choose an explicit CoMEL source or SVD-basis repair')
    if profile=='comel_owlora_source_patient_reference':
        definitions().trainer.after_task(owner)
    else:
        if owner.current_task==0:repaired_initial_basis(owner.model)
        owner.model.expand_lora();owner.current_task+=1


def add_head(network):
    # Source before_task.update_model(): do not copy the previous head weights.
    source=network.classifier[-1].weight
    network.classifier.append(nn.Linear(source.shape[1],2).to(source.device))
    network.classifier[-2].weight.requires_grad=False
    network.classifier[-2].bias.requires_grad=False


def architecture(network):
    return {'heads':len(network.classifier),'num_expanded':network.num_expanded,
        'modules':{n:{'ranks':[l.rank for l in m.lora_layers],'num_expanded':m.num_expanded} for n,m in network.update_modules.items()},
        'requires_grad':{n:p.requires_grad for n,p in network.named_parameters()}}


def restore_model(dimension,layout,state,device):
    network=model(dimension)
    for _ in range(1,layout['heads']):add_head(network)
    source=definitions()
    if set(layout['modules'])!=set(network.update_modules):raise ValueError('CoMEL native LoRA attachment graph changed')
    for name,module in network.update_modules.items():
        description=layout['modules'][name]
        for rank in description['ranks']:
            if type(rank) is not int or rank<1:raise ValueError('Invalid CoMEL learned LoRA rank')
            module.add_lora(source.lora.LoRAWeightedLinearLayer(module.in_features,module.out_features,rank))
        if module.num_expanded!=description['num_expanded']:raise ValueError('CoMEL source module expansion counter changed')
    network.num_expanded=layout['num_expanded']
    parameters=dict(network.named_parameters())
    if set(parameters)!=set(layout['requires_grad']):raise ValueError('CoMEL parameter and freeze identity changed')
    for name,p in parameters.items():p.requires_grad=layout['requires_grad'][name]
    network.load_state_dict(state,strict=True)
    return network.to(device)


def gradients(network):
    # Source leaves non-optimized base/previous-LoRA gradients accumulated.
    # They are preserved too, rather than silently declaring them frozen.
    return {n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in network.named_parameters()}


def restore_gradients(network,state):
    parameters=dict(network.named_parameters())
    if set(parameters)!=set(state):raise ValueError('CoMEL checkpoint omitted active/inactive gradients')
    for name,p in parameters.items():p.grad=None if state[name] is None else state[name].to(p.device).clone()


def owner_state(owner):
    return {'task':owner.current_task,'T':owner.T,'tau1':owner.tau1,'tau':owner.tau,'tau2':float(owner.tau2),
        'tau2_tensor':isinstance(owner.tau2,torch.Tensor),
        'lambda3':owner.lambda3,'optimizer':copy.deepcopy(owner.optimizer.state_dict()),
        'scheduler':copy.deepcopy(owner.scheduler.state_dict()),'epochs':owner.args.num_epoch}


def restore_owner(owner,state):
    if (state['task']!=owner.current_task or state['epochs']!=owner.args.num_epoch
            or state['T']!=1. or state['tau1']!=.95 or state['tau']!=.5 or state['lambda3']!=1.):
        raise ValueError('CoMEL task, original objective or schedule changed')
    owner.tau2=torch.tensor([state['tau2']],device=owner.device) if state['tau2_tensor'] else state['tau2']
    owner.optimizer.load_state_dict(state['optimizer']);owner.scheduler.load_state_dict(state['scheduler'])

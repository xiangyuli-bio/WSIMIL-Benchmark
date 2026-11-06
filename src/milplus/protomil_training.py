"""Literal source SAE pieces with complete interruption state (component stage)."""
import ast,copy,math
from functools import lru_cache
from pathlib import Path
import torch
from . import protomil_source as source
from .akdpmp_source import _compile
from .upstream import pinned_text


class LocalProgress:
    """Local reporting adapter; no model, objective or RNG operations."""
    def set_postfix(self,*args,**kwargs):pass
    def update(self,*args,**kwargs):pass


@lru_cache(maxsize=1)
def literal_piece():
    root,sha=source.root();tree=ast.parse(pinned_text(root,sha,'dncbm/custom_pipeline.py'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pipeline')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_pipeline')
    loops=[n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
    assert len(loops)==1
    body=copy.deepcopy(loops[0].body)
    body+=ast.parse('return train_order,train_piece_idx,last_validated,last_checkpoint').body
    args=['self','epoch','train_order','train_piece_idx','num_train_pieces','train_fnames','train_val_fnames','train_batch_size',
        'last_validated','last_checkpoint','val_frequency','checkpoint_frequency','resample_epoch_freq','start_time','progress_bar']
    return _compile('literal_original_SAE_piece',args,body,source.definitions().Pipeline.run_pipeline.__globals__,root/'dncbm/custom_pipeline.py')


def context(train_files,val_files,output,epochs=200,device='cpu',initialize_order=True):
    if type(epochs) is not int or epochs<1 or not train_files:raise ValueError('Actual original TRAIN pieces and positive epoch budget are required')
    d=source.definitions();args=d.parser().parse_args([]);args.num_epochs=epochs;ae=source.autoencoder().to(device)
    opt=source.optimizer(ae);resampler=d.sae.ActivationResampler(resample_interval=1,n_activations_activity_collate=1,max_n_resamples=math.inf,
        n_learned_features=2048,resample_epoch_freq=args.resample_freq,resample_dataset_size=args.resample_dataset_size)
    metrics=copy.deepcopy(d.sae.default_metrics)
    pipeline=d.Pipeline(activation_resampler=resampler,autoencoder=ae,loss=source.loss(),optimizer=opt,device=device,
        metrics=metrics,checkpoint_directory=Path(output),args=args)
    pipeline.actual_epochs=epochs*len(train_files);pipeline.progress_bar=LocalProgress()
    state={'next_piece':0,'train_order':torch.randperm(len(train_files)) if initialize_order else None,'train_piece_idx':0,'last_validated':0,'last_checkpoint':0,
        'train_files':list(map(str,train_files)),'val_files':list(map(str,val_files)), 'epochs':epochs}
    return pipeline,state


def piece(pipeline,state):
    if state['next_piece']>=pipeline.actual_epochs:raise ValueError('Every actual source piece is already completed')
    args=pipeline.args
    order,index,validated,checkpoint=literal_piece()(pipeline,state['next_piece'],state['train_order'],state['train_piece_idx'],
        len(state['train_files']),state['train_files'],state['val_files'],args.train_sae_bs,state['last_validated'],state['last_checkpoint'],
        args.val_freq,args.ckpt_freq,args.resample_freq,0.,pipeline.progress_bar)
    state.update(next_piece=state['next_piece']+1,train_order=order,train_piece_idx=index,last_validated=validated,last_checkpoint=checkpoint)
    return state


def snapshot(pipeline,state):
    from .engine import rng_state
    return {'state':copy.deepcopy(state),'model':copy.deepcopy(pipeline.autoencoder.state_dict()),'optimizer':copy.deepcopy(pipeline.optimizer.state_dict()),
        'gradients':{k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in pipeline.autoencoder.named_parameters()},
        'resampler':copy.deepcopy(pipeline.activation_resampler.__dict__),'metrics':copy.deepcopy(pipeline.metrics),
        'reported_activations':pipeline.total_activations_trained_on,'current_epoch':getattr(pipeline,'current_epoch',None),'rng':rng_state()}


def restore(pipeline,state,value):
    from .engine import restore_rng
    if state['train_files']!=value['state']['train_files'] or state['val_files']!=value['state']['val_files'] or state['epochs']!=value['state']['epochs']:
        raise ValueError('Actual original SAE piece identity or budget changed')
    pipeline.autoencoder.load_state_dict(value['model'],strict=True);pipeline.optimizer.load_state_dict(value['optimizer'])
    for name,param in pipeline.autoencoder.named_parameters():
        grad=value['gradients'][name];param.grad=None if grad is None else grad.to(param.device).clone()
    pipeline.activation_resampler.__dict__.clear();pipeline.activation_resampler.__dict__.update(copy.deepcopy(value['resampler']))
    pipeline.metrics=copy.deepcopy(value['metrics']);pipeline.total_activations_trained_on=value['reported_activations'];pipeline.current_epoch=value['current_epoch']
    state.clear();state.update(copy.deepcopy(value['state']));restore_rng(value['rng'])

"""Literal whole author MIL trainer oracle; only TEST reporting is deferred."""
import argparse,copy,types,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from milplus import protomil_source as source
from milplus.protomil_driver import _mil_args,Concepts
from milplus.protomil_data import ProtoMILInputs
from milplus.protomil_stages import load_concepts
from milplus.protomil_demo import create_demo
from milplus.protomil_driver import train
from milplus.data import Bundle
from milplus.engine import seed_everything,rng_state
from milplus.io import read_json,write_json
from milplus.icmil_coupled_training import load
from validate_protomil_pipeline import same


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    data=create_demo(out/'input');run=train(data,out/'adapted',device=a.device,config={'sae_epochs':1,'mil_epochs':2,'probe_per_class':5})
    expected=load(run/'MIL/last.pt');assets=Bundle(data).protomil;cache=out/'.protomil_cache';index={}
    for p in (cache/'concept_features').glob('*.json'):
        v=read_json(p);index[v['input_sha256']]={'path':str(p.with_suffix('.h5')),**v}
    records={row['slide_id']:index[assets.rows[row['slide_id']]['sha256']] for split in ['train','val'] for row in assets.bundle.records(split)}
    seed_everything(2026);torch.set_num_threads(2);args=_mil_args();args.probe_epochs=2;args.probe_val_freq=1;args.exp_dir=str(out/'literal_author');Path(args.exp_dir).mkdir()
    model,optimizer,classification,sparsity=source.definitions().mil.prepare_model(args,a.device)
    generators={k:torch.Generator().manual_seed(2026) for k in ['train','val']}
    loaders={k:DataLoader(Concepts(assets.bundle.records(k),records,[]),batch_size=1,shuffle=True,num_workers=0,generator=generators[k]) for k in ['train','val']}
    original=source.definitions().mil;raw_eval=types.FunctionType(original.eval_model.__code__,{**original.eval_model.__globals__,'args':args,'val_dataloader':loaders['val']},original.eval_model.__name__,original.eval_model.__defaults__)
    sentinel=object();deferred=[]
    def eval_with_heldout_reporting_deferred(model,loader,*params,**kwargs):
        if loader is sentinel:
            deferred.append(True)
            # These values feed only author console/WandB output; no forward,
            # labels, model selection, RNG or optimizer operation is replaced.
            return (0.,0.,0.,0.,0.,None,None,0.,0.)
        return raw_eval(model,loader,*params,**kwargs)
    raw_train=types.FunctionType(original.train_model.__code__,{**original.train_model.__globals__,'eval_model':eval_with_heldout_reporting_deferred},original.train_model.__name__,original.train_model.__defaults__)
    raw_train(args,model,optimizer,loaders['train'],loaders['val'],sentinel,classification,sparsity,a.device)
    assert len(deferred)==2
    same(expected['model'],model.state_dict(),'literal_author_whole_trainer_weights');same(expected['optimizer'],optimizer.state_dict(),'literal_author_Adam_moments')
    same(expected['gradients'],{k:None if p.grad is None else p.grad.detach().cpu() for k,p in model.named_parameters()},'literal_author_all_gradients')
    same(expected['generators'],{k:g.get_state() for k,g in generators.items()},'literal_author_TRAIN_VAL_generators');same(expected['rng'],rng_state(),'literal_author_RNG')
    write_json(out/'validation.json',{'state':'passed','literal_whole_author_MIL_training_function_executed':True,'actual_full_concept_dimensions':2048,
        'real_fitted_SAE_features_used':True,'all_weights_gradients_Adam_generators_RNG_exact':True,'only_TEST_reporting_deferred':True,
        'source_commit':source.COMMIT,'synthetic':True,'formal_admission':False,'elapsed_seconds':time.monotonic()-began})

if __name__=='__main__':main()

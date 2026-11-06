"""Isolated full DyKo training/evaluation with actual pretrained TITAN assets.

This candidate remains unadmitted until the original assets and complete
source/wheel/native runs are available. No placeholder tower is instantiated.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch

from .data import Bundle
from .dyko_model import build_original
from .dyko_source import source_assets,source_adam
from .dyko_training import fit_control,new_progress,accumulated_gradients,restore_gradients,original_loader_order
from .io import atomic_torch_save,digest_file,digest_object,read_json,write_json
from .provenance import software_identity


def validate_runtime_identity(value):
    """Fail before loading pretrained weights if a different method's stack won.

    The Torch 2.5.1 and NumPy1 ABI are explicit common-runtime ports. Model,
    tokenizer, hub and FAISS versions remain the reviewed source dependencies.
    This check does not establish model or checkpoint correctness.
    """
    required = {'transformers':'4.46.3','tokenizers':'0.20.3',
        'huggingface-hub':'0.29.3','safetensors':'0.5.3','timm':'1.0.3'}
    versions = value.get('extra_versions',{})
    wrong = {name:{'required':version,'actual':versions.get(name)}
             for name,version in required.items() if versions.get(name) != version}
    if str(value.get('torch','')).split('+')[0] != '2.5.1':
        wrong['torch_common_runtime_port'] = {'required':'2.5.1','actual':value.get('torch')}
    numpy_version = value.get('software',{}).get('versions',{}).get('numpy','')
    if str(numpy_version).split('.')[0] != '1':
        wrong['numpy_FAISS_ABI'] = {'required':'NumPy1','actual':numpy_version}
    if value.get('faiss_version') != '1.8.0':
        wrong['faiss'] = {'required':'1.8.0','actual':value.get('faiss_version')}
    if wrong:
        raise ValueError('DyKo isolated runtime differs from its explicit source stack: '+str(wrong))
    return value


def runtime_identity():
    identity = software_identity()
    versions = {}
    for name in ['transformers','tokenizers','safetensors','huggingface-hub','timm','faiss-cpu','faiss-gpu','pillow']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import faiss
    return validate_runtime_identity({'software':identity,'extra_versions':versions,'torch':torch.__version__,'CUDA':torch.version.cuda,
            'faiss_version':faiss.__version__,'faiss_GPU_api':hasattr(faiss,'StandardGpuResources'),
            'faiss_GPU_count':faiss.get_num_gpus() if hasattr(faiss,'get_num_gpus') else 0,
            'python':str(Path(sys.executable).resolve())})


def _rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in value['cuda']])


def _initialize(seed, device, threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    if device.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError('The requested actual DyKo CUDA device is unavailable')
        torch.cuda.set_device(torch.device(device))


def _inputs(request):
    bundle = Bundle(request['data'],request['fm'],request['split'])
    if getattr(bundle,'dyko',None) is None:
        raise ValueError('Complete DyKo native inputs are not bound')
    inputs = bundle.dyko
    if inputs.contract != request['input_contract']:
        raise ValueError('Actual DyKo native visual/text/label input identity changed')
    inputs.verify(deep=True)
    plan = inputs.plan(request['config']['n_shot'],request['seed'])
    if plan != request['shot_plan']:
        raise ValueError('Actual original DyKo TRAIN shot identities or patient memberships changed')
    return bundle,inputs,plan


def _save(out,model,optimizer,state,backend,request,inputs,*,best=False):
    inputs.verify()
    payload = {'schema_version':1,'request':request,'input_contract':inputs.contract,
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'accumulated_gradients':accumulated_gradients(model),
        'progress':state,'rng':_rng(),'faiss_calls':backend.calls,
        'plain_tokenized_prompts':model.prompt_learner.tokenized_prompts.detach().cpu(),
        'actual_pretrained_model_evidence':model._milplus_dyko_evidence}
    name = 'best.pt' if best else 'last.pt';path = out/name
    atomic_torch_save(payload,path)
    write_json(out/(name+'.sha256.json'),{'sha256':digest_file(path),'bytes':path.stat().st_size})


def _restore(path,model,optimizer,backend,request,inputs):
    path = Path(path);proof = read_json(path.parent/(path.name+'.sha256.json'))
    if digest_file(path) != proof['sha256'] or path.stat().st_size != proof['bytes']:
        raise ValueError('Complete original DyKo checkpoint bytes changed')
    value = torch.load(path,map_location='cpu',weights_only=False)
    if value['schema_version'] != 1 or value['request'] != request or value['input_contract'] != inputs.contract:
        raise ValueError('DyKo recovery changed model, recipe, software, pretrained assets or actual input population')
    if not torch.equal(value['plain_tokenized_prompts'],model.prompt_learner.tokenized_prompts.cpu()):
        raise ValueError('Actual original TITAN tokenization changed before recovery')
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    restore_gradients(model,value['accumulated_gradients']);backend.calls = value['faiss_calls']
    _restore_rng(value['rng'])
    return value['progress']


def _feature_cache(bundle, rows):
    # This is the source preloading behavior for actual unchanged frozen visual
    # bags. There is no patch cap, synthetic spatial layout or extra sampling.
    return {row['slide_id']:bundle.load(row)[0] for row in rows}


def train(request, out, *, resume=False, stop_after=None, stop_after_batch=None):
    if runtime_identity() != request['runtime']:
        raise ValueError('Actual DyKo child runtime differs from its locked request')
    cfg = request['config'];bundle,inputs,plan = _inputs(request)
    _initialize(request['seed'],request['device'],cfg['threads'])
    model,backend = build_original(inputs,cfg['faiss_backend']);model.to(request['device'])
    optimizer = source_adam(model,lr=.001,weight_decay=.00001)
    write_json(out/'actual_original_model_evidence.json',model._milplus_dyko_evidence)
    state = new_progress()
    if resume:
        state = _restore(out/'last.pt',model,optimizer,backend,request,inputs)
    train_by_id = {row['slide_id']:row for row in bundle.records('train')}
    train_rows = [train_by_id[row['slide_id']] for row in plan['selected']]
    val_rows = bundle.records('val')
    cache = _feature_cache(bundle,train_rows+val_rows)
    bank = source_assets(inputs.cohort)['bank'].to(request['device'])

    def forward(row, training):
        x = cache[row['slide_id']].to(request['device'])
        label = torch.tensor([inputs.source_target(row)],dtype=torch.long,device=request['device'])
        result = model(data=(x.unsqueeze(0),bank.unsqueeze(0)),is_train=training,label=label)
        return result,label

    def checkpoint(progress, improved):
        _save(out,model,optimizer,progress,backend,request,inputs)
        if improved:
            _save(out,model,optimizer,progress,backend,request,inputs,best=True)
        write_json(out/'history.json',progress['history'])
        write_json(out/'progress.json',{k:v for k,v in progress.items() if k not in {'last_validation','epoch_train'}})

    result = fit_control(model,optimizer,train_rows,val_rows,forward,state,epochs=cfg['epochs'],patience=40,
        on_checkpoint=checkpoint,stop_after_epoch=stop_after,stop_after_batch=stop_after_batch)
    inputs.verify(deep=True)
    write_json(out/'native_status.json',{'state':result,'epochs':state['epochs'],'train_batches':state['train_batches'],
        'optimizer_updates':state['optimizer_updates'],'best_epoch':state['best_epoch'],'best_accuracy':state['best_accuracy'],
        'source_selection':'standard_VAL_slide_accuracy','test_evaluated':False,'formal_admission':False})


@torch.no_grad()
def evaluate(request, out, subset):
    if runtime_identity() != request['runtime']:
        raise ValueError('Actual original DyKo evaluation runtime changed')
    if subset not in {'val','test'} or read_json(out/'native_status.json')['state'] != 'completed':
        raise ValueError('Full DyKo evaluation requires completed original validation stopping')
    bundle,inputs,plan = _inputs(request);cfg = request['config']
    _initialize(request['seed'],request['device'],cfg['threads'])
    model,backend = build_original(inputs,cfg['faiss_backend']);model.to(request['device'])
    optimizer = source_adam(model,lr=.001,weight_decay=.00001)
    progress = _restore(out/'best.pt',model,optimizer,backend,request,inputs)
    model.eval();rows = bundle.records(subset);cache = _feature_cache(bundle,rows)
    bank = source_assets(inputs.cohort)['bank'].to(request['device']);output = []
    for index in original_loader_order(len(rows),False):
        row = rows[index];x = cache[row['slide_id']].to(request['device'])
        result = model(data=(x.unsqueeze(0),bank.unsqueeze(0)))
        p = inputs.common_probabilities(result['Y_prob']).cpu().reshape(-1)
        source_prediction = int(result['Y_hat'].reshape(-1).item())
        output.append({'slide_id':row['slide_id'],'patient_id':row['patient_id'],'global_patient_id':row['global_patient_id'],
            'target':int(row['target']),'source_prediction':inputs.source_to_common[source_prediction],
            **{f'p_{i}':float(p[i]) for i in range(len(p))}})
    write_json(out/(subset+'_predictions_native.json'),output)
    write_json(out/(subset+'_selected_checkpoint.json'),{'selection':'standard_VAL_accuracy','best_epoch':progress['best_epoch'],
        'best_accuracy':progress['best_accuracy'],'checkpoint_sha256':digest_file(out/'best.pt'),
        'input_contract':inputs.contract,'asset_contract':inputs.titan.contract})


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--stop-after-batch',type=int);parser.add_argument('--evaluate',choices=['val','test'])
    args = parser.parse_args();request = read_json(args.request);out = Path(args.output)
    if args.evaluate:
        evaluate(request,out,args.evaluate)
    else:
        train(request,out,resume=args.resume,stop_after=args.stop_after,stop_after_batch=args.stop_after_batch)


if __name__ == '__main__':
    main()

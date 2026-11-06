"""Complete frozen-pretrained context fitting and evaluation through the native stack."""
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd

from .icmil_context_data import ContextInputs
from .icmil_context_source import ARCH,WEIGHTS,checkpoint
from .io import digest_file,read_json,write_json
from .provenance import software_identity

PROFILES={'icmil_context_'+s+'_source':s for s in WEIGHTS}


def environment(config):
    default='/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/development/icmil_context_v91/native_env_r1/bin/python'
    python=config.get('native_python') or os.environ.get('MILPLUS_ICMIL_PYTHON') or default
    if not Path(python).is_file():raise FileNotFoundError('Set MILPLUS_ICMIL_PYTHON to the original Python3.11/Torch2.5.1 environment')
    root=Path(__file__).resolve().parent.parent
    env={**os.environ,'PYTHONPATH':str(root),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
        'OMP_NUM_THREADS':str(config['threads']),'MKL_NUM_THREADS':str(config['threads'])}
    return str(python),env


def runtime(config):
    import json
    python,env=environment(config)
    r=subprocess.run([python,'-c','import json;from milplus.icmil_context_native import identity;print(json.dumps(identity()))'],
        env=env,capture_output=True,text=True,check=True)
    identity=json.loads(r.stdout)
    if not identity['python'].startswith('3.11.') or identity['source_sha256']!=software_identity()['source_sha256']:
        raise ValueError('ICMIL requires matching adapter source and its actual original Python3.11 runtime')
    return identity


def _predict(out,inputs,request,subset,device):
    from .engine import score
    folder=out/('prediction_'+subset+'_'+uuid.uuid4().hex);folder.mkdir()
    features,targets,rows=inputs.dense(subset);np.save(folder/'queries.npy',features,allow_pickle=False)
    cfg=request['config'];native=runtime(cfg)
    if native!=request['native_runtime']:
        raise ValueError('Actual original ICMIL runtime changed after context fitting')
    call={'native_runtime':native,'weights':str(out/('icmil-'+request['model_seed']+'.pt')),
        'weights_sha256':WEIGHTS[request['model_seed']],'support':str(out/'support.npz'),
        'support_sha256':request['support_sha256'],'queries':str(folder/'queries.npy'),
        'queries_sha256':digest_file(folder/'queries.npy'),'threads':cfg['threads'],'device':str(device),
        'model_seed':request['model_seed'],'output':str(folder/'logits.npy')}
    write_json(folder/'native_call.json',call);python,env=environment(cfg)
    cmd=[python,'-m','milplus.icmil_context_native','--call',str(folder/'native_call.json')]
    with (folder/'native.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(folder/'process.json',{'pid':child.pid,'command':cmd})
        try:code=child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    if code:raise RuntimeError('Actual native context prediction failed; inspect '+str(folder/'native.log'))
    receipt=read_json(folder/'logits.receipt.json')
    if receipt['state']!='passed' or receipt['query_bags']!=len(rows) or receipt['downstream_optimizer_updates']!=0:
        raise ValueError('Original ICMIL did not complete the entire fixed context/query pass')
    probs=np.load(folder/'logits.probabilities.npy',allow_pickle=False)
    if digest_file(folder/'logits.probabilities.npy')!=receipt['probabilities_sha256'] or probs.shape!=(len(rows),2):
        raise ValueError('Native ICMIL probability bytes/row mapping changed')
    frame=pd.DataFrame([{**{k:r[k] for k in ['slide_id','patient_id','global_patient_id','target']},
        'subset':subset,'source_slides':r['slide_id'],'prediction':int(p.argmax()),'p_0':float(p[0]),'p_1':float(p[1])}
        for r,p in zip(rows,probs)])
    inputs.check();metrics=score(inputs.bundle,frame)
    frame.to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),metrics)
    return metrics


def train(data,method,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None):
    from .registry import inspect_method
    profile=method if method in PROFILES else inspect_method(method)['id']
    if profile not in PROFILES:raise ValueError('Choose one of the actual three pretrained in-context ICMIL conditions')
    supplied=config or {}
    if set(supplied)-{'threads','model'}:raise ValueError('Pretrained in-context ICMIL performs zero downstream gradient steps; epochs/learning-rate/loss replacement are not its algorithm')
    model=supplied.get('model',{})
    if not isinstance(model,dict) or set(model)-{'native_python'}:raise ValueError('Original ICMIL widths, pretrained seeds and full context cannot be silently replaced')
    cfg={'threads':supplied.get('threads',2),'native_python':model.get('native_python')}
    if type(cfg['threads']) is not int or cfg['threads']<1 or type(seed) is not int:raise ValueError('Declare positive threads and an integer run seed')
    if stop_after not in {None,1}:raise ValueError('In-context ICMIL only supports a context-binding phase interruption, not invented optimizer epochs')
    inputs=ContextInputs(data,fm,split);pretrained_seed=PROFILES[profile];weights=checkpoint(pretrained_seed)
    out=Path(output).expanduser().resolve();native=runtime(cfg)
    request={'backend':'icmil_context_original_pretrained','method':profile,'data':str(inputs.bundle.root),
        'fm':inputs.bundle.fm,'split':inputs.bundle.split,'seed':seed,'model_seed':pretrained_seed,'device':str(device),
        'config':cfg,'software':software_identity(),'native_runtime':native,'input_contract':inputs.contract,
        'architecture':ARCH,'pretrained_weights_sha256':WEIGHTS[pretrained_seed],
        'downstream_optimizer_updates':0,'selection':'fixed public pretrained checkpoint; no VAL/TEST model selection',
        'synthetic':bool(inputs.bundle.config.get('synthetic',False)),'formal_admission':False}
    if resume:
        previous=read_json(out/'request.json');expected={k:v for k,v in previous.items() if k!='support_sha256'}
        if expected!=request or read_json(out/'status.json')['state']!='interrupted':
            raise ValueError('ICMIL resume must retain actual fixed support, task, pretrained model and runtime')
        request=previous
        if digest_file(out/'support.npz')!=request['support_sha256']:raise ValueError('Original saved TRAIN context changed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        x,y,rows=inputs.dense('train');np.savez(out/'support.npz',features=x,targets=y)
        request['support_sha256']=digest_file(out/'support.npz')
        shutil.copy2(weights,out/weights.name);write_json(out/'request.json',request)
        write_json(out/'support_ids.json',[{k:r[k] for k in ['slide_id','patient_id','global_patient_id','target']} for r in rows])
    lock=out/'RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    try:
        write_json(out/'status.json',{'state':'running','downstream_optimizer_updates':0})
        if stop_after==1:
            write_json(out/'status.json',{'state':'interrupted','phase':'actual_full_TRAIN_context_bound','downstream_optimizer_updates':0});return out
        _predict(out,inputs,request,'val',device)
        write_json(out/'status.json',{'state':'completed','downstream_optimizer_updates':0,
            'test_evaluated':False,'pretraining_epochs_of_original_checkpoint':8000,'formal_admission':False})
        return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


def evaluate(run,subset='test',device='cpu'):
    out=Path(run).resolve();request=read_json(out/'request.json')
    if subset not in {'val','test'} or read_json(out/'status.json')['state']!='completed':
        raise ValueError('ICMIL evaluation requires its completed frozen-context fitting phase')
    inputs=ContextInputs(request['data'],request['fm'],request['split'])
    if request['software']!=software_identity() or inputs.contract!=request['input_contract']:
        raise ValueError('ICMIL evaluation data/features/preparation/software differ from the recorded run')
    return _predict(out,inputs,request,subset,device)

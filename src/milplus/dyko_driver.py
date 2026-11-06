"""Unified-entry bridge for full original DyKo in its explicit native runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .data import Bundle
from .io import digest_file,read_json,write_json
from .provenance import software_identity
from .dyko_spec import EXECUTION_CONDITIONS as PROFILES


def environment(config, out):
    executable = config.get('runtime_python') or os.environ.get('MILPLUS_DYKO_PYTHON') or sys.executable
    if not Path(executable).is_file():
        raise FileNotFoundError('Declare the actual DyKo Torch/transformers4.46.3/FAISS1.8 runtime executable')
    root = Path(__file__).resolve().parent.parent
    paths = [str(root)]
    extra = config.get('runtime_dependencies') or os.environ.get('MILPLUS_DYKO_DEPENDENCIES')
    if extra:
        extra = Path(extra).resolve()
        if not extra.is_dir():
            raise FileNotFoundError('Declared DyKo dependency prefix is missing')
        paths.append(str(extra))
    cache = Path(out)/'private_runtime_cache'
    env = {**os.environ,'PYTHONPATH':os.pathsep.join(paths),'PYTHONNOUSERSITE':'1',
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
        'OMP_NUM_THREADS':str(config['threads']),'MKL_NUM_THREADS':str(config['threads']),
        'CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_DATASETS_OFFLINE':'1','HF_HUB_CACHE':str(cache/'hub'),'HF_MODULES_CACHE':str(cache/'modules')}
    return str(executable),env


def runtime(executable, env):
    result = subprocess.run([executable,'-s','-c',
        'import json;from milplus.dyko_runtime import runtime_identity;print(json.dumps(runtime_identity()))'],
        capture_output=True,text=True,env=env)
    if result.returncode:
        raise RuntimeError('DyKo runtime preflight failed; use a separate .[dyko] environment '
                           'and MILPLUS_DYKO_PYTHON.\n' + result.stderr[-6000:])
    return json.loads(result.stdout)


def train(data, method, output, *, fm=None, split=None, seed=2026, device='cpu',
          config=None, resume=False, stop_after=None):
    if method not in PROFILES:
        from .registry import inspect_method
        method = inspect_method(method)['id']
    if method not in PROFILES:
        raise ValueError('Choose an explicit complete native DyKo cohort and FAISS backend')
    bundle = Bundle(data,fm,split);inputs = getattr(bundle,'dyko',None)
    if inputs is None or inputs.cohort != PROFILES[method]['cohort']:
        raise ValueError('The actual native DyKo cohort, source concepts and visual/text input contract are missing')
    supplied = config or {}
    if not isinstance(supplied,dict) or set(supplied)-{'epochs','threads','model'}:
        raise ValueError('DyKo preserves original loss/Adam/accumulation/stopping; declare epochs, threads and explicit runtime/shot settings')
    settings = supplied.get('model',{})
    if not isinstance(settings,dict) or set(settings)-{'n_shot','runtime_python','runtime_dependencies','stop_after_train_batches'}:
        raise ValueError('DyKo full pretrained tower,768D graph,16prompts,10clusters/10concepts and8head attentions cannot be reduced')
    cfg = {**PROFILES[method],'epochs':supplied.get('epochs',1000),'threads':supplied.get('threads',2),
        'n_shot':settings.get('n_shot',16),'runtime_python':settings.get('runtime_python'),
        'runtime_dependencies':settings.get('runtime_dependencies'),'patience':40,'lr':.001,
        'weight_decay':.00001,'accumulation':4,'precision':'float32','selection':'standard_VAL_accuracy'}
    if (type(seed) is not int or not 0 <= seed < 2**32
            or any(type(cfg[k]) is not int or cfg[k] < 1 for k in ['epochs','threads','n_shot'])):
        raise ValueError('DyKo requires positive integer budgets and the original32bit pandas/model seed')
    if device != 'cpu' and (not device.startswith('cuda:') or not device[5:].isdigit()):
        raise ValueError('Choose an explicit cpu or cuda:N DyKo device')
    if cfg['faiss_backend'] == 'cuda_faiss' and device == 'cpu':
        raise ValueError('The requested CUDA FAISS DyKo condition needs an actual CUDA model device')
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after < cfg['epochs']):
        raise ValueError('DyKo epoch interruption must be inside its same declared max-epoch budget')
    stop_batch = settings.get('stop_after_train_batches')
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError('DyKo accumulated-gradient interruption needs a positive actual TRAIN-forward count')
    if stop_batch is not None and stop_after is not None:
        raise ValueError('Request one explicit DyKo interruption boundary per invocation')
    inputs.verify(deep=True)
    request = {'schema_version':1,'backend':'dyko_full_original_TITAN','method':method,
        'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,'seed':seed,'device':device,'config':cfg,
        'input_contract':inputs.contract,'shot_plan':inputs.plan(cfg['n_shot'],seed),'software':software_identity(),
        'synthetic':bool(bundle.config.get('synthetic',False)),'formal_admission':False,
        'full_TITAN_method_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}
    out = Path(output).resolve();executable,env = environment(cfg,out)
    request['runtime'] = runtime(executable,env)
    child = request['runtime']['software']
    if child['source_sha256'] != request['software']['source_sha256'] or child['sources_lock_sha256'] != request['software']['sources_lock_sha256']:
        raise ValueError('Actual isolated DyKo model/recipe code differs from the common entry point')
    out.mkdir(parents=True,exist_ok=resume)
    if resume:
        if read_json(out/'request.json') != request:
            raise ValueError('DyKo recovery changed actual pretrained assets, patients, source concepts, shot identities or full recipe')
        if read_json(out/'status.json')['state'] == 'completed':
            raise ValueError('DyKo run already completed; use a new output')
    else:
        write_json(out/'request.json',request);write_json(out/'actual_TRAIN_shot_plan.json',request['shot_plan'])
    lock = out/'RUNNING.lock'
    try:
        fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        raise ValueError('Inspect the actual DyKo controller/child before recovering a locked run') from None
    with os.fdopen(fd,'w') as stream:
        stream.write(str(os.getpid()))
    child_process = None
    try:
        inputs.titan.cache_view(env['HF_HUB_CACHE'])
        command = [executable,'-s','-u','-m','milplus.dyko_runtime','--request',str(out/'request.json'),'--output',str(out)]
        if resume:
            command.append('--resume')
        if stop_after is not None:
            command += ['--stop-after',str(stop_after)]
        if stop_batch is not None:
            command += ['--stop-after-batch',str(stop_batch)]
        write_json(out/'status.json',{'state':'running','backend':request['backend'],'command':command})
        with (out/('resume_native.log' if resume else 'native.log')).open('a') as log:
            child_process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(out/'native_process.json',{'pid':child_process.pid,'command':command});code = child_process.wait()
        if code:
            raise RuntimeError('The actual full original DyKo process failed; inspect its native log')
        status = read_json(out/'native_status.json')
        if status['state'] not in {'completed','interrupted'}:
            raise ValueError('Full original DyKo returned an invalid invocation state')
        write_json(out/'status.json',{**status,'backend':request['backend']})
        return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error),'backend':request['backend']})
        raise
    finally:
        if child_process is None or child_process.poll() is not None:
            lock.unlink(missing_ok=True)


def evaluate(run, subset='test', device='cpu'):
    from .engine import score
    out = Path(run).resolve();request = read_json(out/'request.json')
    if (subset not in {'val','test'} or read_json(out/'status.json')['state'] != 'completed'
            or device != request['device'] or request['software'] != software_identity()):
        raise ValueError('Full DyKo evaluation requires completed training and its locked actual code/device')
    bundle = Bundle(request['data'],request['fm'],request['split']);inputs = bundle.dyko
    if inputs.contract != request['input_contract']:
        raise ValueError('Original DyKo evaluation population or native inputs changed')
    inputs.verify(deep=True)
    executable,env = environment(request['config'],out)
    if runtime(executable,env) != request['runtime']:
        raise ValueError('Actual original DyKo evaluation stack differs from the recorded training runtime')
    inputs.titan.cache_view(env['HF_HUB_CACHE'])
    command = [executable,'-s','-u','-m','milplus.dyko_runtime','--request',str(out/'request.json'),
        '--output',str(out),'--evaluate',subset]
    with (out/(subset+'_evaluation.log')).open('a') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    frame = pd.DataFrame(read_json(out/(subset+'_predictions_native.json')));rows = bundle.records(subset)
    if frame.slide_id.tolist() != [r['slide_id'] for r in rows] or frame.target.tolist() != [int(r['target']) for r in rows]:
        raise ValueError('Actual original DyKo prediction IDs or benchmark class order changed')
    metrics = score(bundle,frame);frame.to_csv(out/(subset+'_predictions.csv'),index=False)
    write_json(out/(subset+'_metrics.json'),metrics)
    return metrics

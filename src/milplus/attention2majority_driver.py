"""Uniform CLI bridge to the method's isolated original TensorFlow training stack."""
import os
from pathlib import Path
import subprocess

import pandas as pd

from .attention2majority_data import AttentionInputs
from .io import digest_object,read_json,write_json
from .provenance import software_identity

PROFILES={
    'attention2majority_camelyon_source_repaired':{'classes':2,'preprocessing':'source_signed_int8_caffe_then_rescale',
        'discriminator_lr':.001,'discriminator_early_stopping':False,'zero_tail_repaired':False},
    'attention2majority_camelyon_paper_reference':{'classes':2,'preprocessing':'paper_uint8_caffe',
        'discriminator_lr':.0002,'discriminator_early_stopping':True,'zero_tail_repaired':True},
    'attention2majority_kidney_paper_reference':{'classes':3,'preprocessing':'paper_uint8_caffe',
        'discriminator_lr':.0002,'discriminator_early_stopping':True,'zero_tail_repaired':True},
}


def environment(config,device):
    default='/root/MIL/validation/milplus_validation_envs/camil_tf/bin/python'
    executable=config.get('tensorflow_python') or os.environ.get('MILPLUS_TF_PYTHON') or default
    if not Path(executable).is_file():raise FileNotFoundError('Attention2majority requires an isolated TensorFlow2.15.1 environment; set MILPLUS_TF_PYTHON')
    root=Path(__file__).resolve().parent.parent
    env={**os.environ,'PYTHONPATH':str(root),'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
        'TF_ENABLE_ONEDNN_OPTS':'0','TF_NUM_INTRAOP_THREADS':str(config['threads']),'TF_NUM_INTEROP_THREADS':'1'}
    if device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    elif not device.startswith('cuda:') or not device[5:].isdigit():raise ValueError('Choose explicit cpu or cuda:N for Attention2majority')
    if device.startswith('cuda:'):
        # Resolve libraries belonging to this isolated tensorflow[and-cuda]
        # environment. Never replace the host driver or another method stack.
        prefix=Path(executable).absolute().parent.parent
        nvidia=list(prefix.glob('lib/python*/site-packages/nvidia'))
        if nvidia:
            native=nvidia[0];libraries=sorted(str(p) for p in native.glob('*/lib') if p.is_dir())
            env['LD_LIBRARY_PATH']=':'.join(libraries+[env.get('LD_LIBRARY_PATH','')])
            compiler=native/'cuda_nvcc'
            if compiler.is_dir():
                env['PATH']=str(compiler/'bin')+':'+env.get('PATH','')
                env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_gpu_cuda_data_dir='+str(compiler)
        env.update(TF_DETERMINISTIC_OPS='1',TF_CUDNN_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return str(executable),env


def train(data,method,output,seed=2026,device='cpu',config=None,resume=False,stop_after=None,fm=None,split=None):
    from .registry import inspect_method
    profile=method if method in PROFILES else inspect_method(method)['id']
    if profile not in PROFILES:raise ValueError('Choose a complete registered Attention2majority source/reference condition')
    supplied=config or {};allowed={'epochs','threads','model'}
    if set(supplied)-allowed:raise ValueError('Attention2majority preserves native original stage optimizers; supported overrides are epochs,threads,model')
    model_cfg=supplied.get('model',{})
    if not isinstance(model_cfg,dict) or set(model_cfg)-{'discriminator_epochs','tensorflow_python','stop_discriminator_after'}:
        raise ValueError('Attention2majority does not allow hidden width/input reduction or replacement scores')
    inputs=AttentionInputs(data,fm,split);expected=PROFILES[profile]
    if inputs.classes!=expected['classes']:raise ValueError('Attention2majority source condition does not match native binary/three-grade task')
    cfg={**expected,'epochs':supplied.get('epochs',150),'discriminator_epochs':model_cfg.get('discriminator_epochs',150),
        'threads':supplied.get('threads',2),'tensorflow_python':model_cfg.get('tensorflow_python')}
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in ['epochs','discriminator_epochs','threads']) or type(seed) is not int:
        raise ValueError('Attention2majority native stage budgets, thread count and seed must be declared positive integers')
    if stop_after is not None and (type(stop_after) is not int or not 0<stop_after<cfg['epochs']):
        raise ValueError('MIL interruption must be a positive epoch strictly inside the complete declared budget')
    ds=model_cfg.get('stop_discriminator_after')
    if ds is not None and (type(ds) is not int or not 0<ds<cfg['discriminator_epochs']):
        raise ValueError('Discriminator interruption must be a positive epoch strictly inside the complete declared budget')
    request={'schema_version':1,'backend':'attention2majority_native_tensorflow','method':profile,'data':str(inputs.bundle.root),
        'fm':inputs.bundle.fm,'split':inputs.bundle.split,'seed':seed,'device':device,'config':cfg,
        'software':software_identity(),'input_contract':inputs.contract,'synthetic':bool(inputs.bundle.config.get('synthetic',False)),
        'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'}
    executable,env=environment(cfg,device)
    probe=subprocess.run([executable,'-c',
        'import json;from milplus.attention2majority_tf import runtime_identity;print(json.dumps(runtime_identity()))'],
        env=env,capture_output=True,text=True,check=True)
    runtime=__import__('json').loads(probe.stdout)
    if runtime['runtime_source_sha256']!=request['software']['source_sha256']:
        raise ValueError('Isolated TensorFlow adapter code differs from the common CLI code')
    request['tensorflow_runtime']=runtime
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=resume)
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('Attention2majority resume changed raw data/complete recipe/software/device')
        if read_json(out/'status.json')['state']=='completed':raise ValueError('Attention2majority run already completed; use a new output')
    else:
        write_json(out/'request.json',request);write_json(out/'input_binding.json',inputs.export())
    cmd=[executable,'-m','milplus.attention2majority_tf','--binding',str(out/'input_binding.json'),
        '--request',str(out/'request.json'),'--output',str(out)]
    if resume:cmd.append('--resume')
    if stop_after is not None:cmd+=['--stop-stage','MIL','--stop-after',str(stop_after)]
    if model_cfg.get('stop_discriminator_after') is not None:
        if stop_after is not None:raise ValueError('Only one explicit stage interruption can be requested')
        cmd+=['--stop-stage','discriminator','--stop-after',str(model_cfg['stop_discriminator_after'])]
    lock=out/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError('Attention2majority run is already locked; inspect its actual process before recovery') from None
    with os.fdopen(fd,'w') as stream:stream.write(str(os.getpid()))
    try:
        write_json(out/'status.json',{'state':'running','backend':request['backend'],'command':cmd})
        with (out/('resume_native.log' if resume else 'native.log')).open('a') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(out/'native_process.json',{'pid':child.pid,'command':cmd});code=child.wait()
        if code:raise RuntimeError('Native TensorFlow Attention2majority failed; see actual native.log/resume_native.log')
        state=read_json(out/'native_status.json')
        if state['state'] not in {'completed','interrupted'}:raise ValueError('Actual complete native training state is invalid')
        write_json(out/'status.json',{**state,'backend':request['backend'],'test_evaluated':False})
        return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error),'backend':request['backend']});raise
    finally:lock.unlink(missing_ok=True)


def evaluate(run,subset='test',device='cpu'):
    from .engine import score
    out=Path(run).resolve();request=read_json(out/'request.json')
    if subset not in {'val','test'} or read_json(out/'status.json')['state']!='completed':
        raise ValueError('Attention2majority evaluation requires actual completed native source stopping budget')
    if request['software']!=software_identity():raise ValueError('Attention2majority evaluation software differs from the recorded run')
    inputs=AttentionInputs(request['data'],request['fm'],request['split'])
    if inputs.contract!=request['input_contract']:raise ValueError('Actual original raw image/task/pool identities changed')
    executable,env=environment(request['config'],device)
    probe=subprocess.run([executable,'-c',
        'import json;from milplus.attention2majority_tf import runtime_identity;print(json.dumps(runtime_identity()))'],
        env=env,capture_output=True,text=True,check=True)
    runtime=__import__('json').loads(probe.stdout)
    if runtime['runtime_source_sha256']!=request['software']['source_sha256']:
        raise ValueError('Independent evaluation adapter differs from the locked training source')
    # Evaluation may use a different explicit device/runtime. Preserve the
    # original immutable training request and record this separate invocation.
    evaluation={**request,'device':device,'tensorflow_runtime':runtime}
    eval_request=out/(subset+'_evaluation_request_'+digest_object(evaluation)+'.json')
    if eval_request.exists():
        if read_json(eval_request)!=evaluation:raise ValueError('Evaluation binding collision')
    else:write_json(eval_request,evaluation)
    cmd=[executable,'-m','milplus.attention2majority_tf','--binding',str(out/'input_binding.json'),
        '--request',str(eval_request),'--output',str(out),'--evaluate',subset]
    with (out/(subset+'_evaluation.log')).open('a') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    frame=pd.DataFrame(read_json(out/(subset+'_predictions_native.json')))
    expected=inputs.bundle.records(subset)
    if frame.slide_id.tolist()!=[r['slide_id'] for r in expected] or frame.target.tolist()!=[r['target'] for r in expected]:
        raise ValueError('Native Attention2majority prediction IDs/order/labels are misaligned')
    metrics=score(inputs.bundle,frame);frame.to_csv(out/(subset+'_predictions.csv'),index=False)
    write_json(out/(subset+'_metrics.json'),metrics);return metrics

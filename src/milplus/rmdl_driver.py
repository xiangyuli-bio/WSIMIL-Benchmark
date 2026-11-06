"""Uniform data+method entry for the complete, isolated Keras RMDL pipeline."""
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from .data import Bundle
from .io import read_json, write_json
from .provenance import software_identity

PROFILES = {
    'rmdl_source_sum_full_reference': {'pooling':'sum'},
    'rmdl_paper_mean_full_reference': {'pooling':'mean'},
}


def environment(config, device):
    # Reuse only the general TF2.15 runtime/library resolution. No other
    # method's model, sampling, losses or trainer is used for RMDL.
    from .attention2majority_driver import environment as tf_environment
    executable,env = tf_environment(config,device)
    overlay = config.get('tensorflow_pixel_dependencies') or os.environ.get('MILPLUS_RMDL_PIXEL_DEPENDENCIES')
    if overlay:
        overlay = Path(overlay).resolve()
        if not overlay.is_dir():
            raise FileNotFoundError('RMDL declared isolated Pillow/OpenSlide dependency prefix is missing')
        env['PYTHONPATH'] += os.pathsep+str(overlay)
    env['PYTHONNOUSERSITE'] = '1'
    env['TF_DETERMINISTIC_OPS'] = '1'
    return executable,env


def _runtime(executable, env):
    probe = subprocess.run([executable,'-s','-c',
        'import json;from milplus.rmdl_tf import runtime_identity;print(json.dumps(runtime_identity()))'],
        env=env,capture_output=True,text=True,check=True)
    return json.loads(probe.stdout)


def train(data, method, output, seed=2026, device='cpu', config=None, resume=False,
          stop_after=None, fm=None, split=None):
    if method not in PROFILES:
        from .registry import inspect_method
        method = inspect_method(method)['id']
    if method not in PROFILES:
        raise ValueError('Choose the complete original-SUM or explicitly named paper-MEAN RMDL reference')
    supplied = config or {}
    if not isinstance(supplied,dict) or set(supplied)-{'epochs','threads','model'}:
        raise ValueError('RMDL accepts declared MIL iteration budget, threads and explicit localization/reference runtime settings')
    settings = supplied.get('model',{})
    allowed = {'localizer_iterations','hard_negative_after','normals_per_slide','tensorflow_python',
               'tensorflow_pixel_dependencies','stop_localizer_after'}
    if not isinstance(settings,dict) or set(settings)-allowed:
        raise ValueError('RMDL full widths, input geometry, batches80/256 and300 selected instances cannot be reduced or replaced')
    bundle = Bundle(data,fm,split)
    if bundle.rmdl is None:
        raise ValueError('RMDL needs its full actual WSI/TRAIN pixel annotation contract')
    inputs = bundle.rmdl;binding = inputs.export()
    cfg = {**PROFILES[method],'MIL_iterations':supplied.get('epochs',7000),'threads':supplied.get('threads',2),
           'localizer_iterations':settings.get('localizer_iterations',10000),
           'hard_negative_after':settings.get('hard_negative_after',5000),'normals_per_slide':settings.get('normals_per_slide',1024),
           'tensorflow_python':settings.get('tensorflow_python'),'tensorflow_pixel_dependencies':settings.get('tensorflow_pixel_dependencies'),
           'localizer_budget_provenance':'explicit_10000_reference_default_original_total_not_published',
           'MIL_step_unit':'optimizer_iteration','full_native_batches':[80,256],'native_selected_instances':300}
    if type(seed) is not int or seed < 0 or any(type(cfg[k]) is not int or cfg[k] < 1 for k in
            ['MIL_iterations','threads','localizer_iterations','hard_negative_after','normals_per_slide']):
        raise ValueError('Declare positive RMDL complete stage budgets and a nonnegative integer seed')
    if cfg['hard_negative_after'] >= cfg['localizer_iterations']:
        raise ValueError('The full localization budget must include actual training both before and after hard-negative mining')
    stop_localizer = settings.get('stop_localizer_after')
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after < cfg['MIL_iterations']):
        raise ValueError('RMDL MIL interruption must be a positive update inside the same declared final budget')
    if stop_localizer is not None and (type(stop_localizer) is not int or not 0 < stop_localizer < cfg['localizer_iterations']):
        raise ValueError('RMDL localization interruption must be a positive update inside the same declared final budget')
    if stop_after is not None and stop_localizer is not None:
        raise ValueError('Interrupt exactly one explicitly selected RMDL stage per invocation')
    request = {'schema_version':1,'backend':'rmdl_complete_native_tensorflow','method':method,
        'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,'seed':seed,'device':device,
        'config':cfg,'software':software_identity(),'input_contract':inputs.contract,
        'synthetic':binding['synthetic'],'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'}
    executable,env = environment(cfg,device);request['tensorflow_runtime'] = _runtime(executable,env)
    if request['tensorflow_runtime']['runtime_source_sha256'] != request['software']['source_sha256']:
        raise ValueError('The common entry point and isolated full RMDL trainer have different source identities')
    out = Path(output).resolve();out.mkdir(parents=True,exist_ok=resume)
    if resume:
        if read_json(out/'request.json') != request or read_json(out/'input_binding.json') != binding:
            raise ValueError('RMDL recovery changed actual patients, pixel supervision, recipe, graph, runtime or device')
        if read_json(out/'status.json')['state'] == 'completed':
            raise ValueError('Complete RMDL run already finished; use a separate output')
    else:
        write_json(out/'request.json',request);write_json(out/'input_binding.json',binding)
    command = [executable,'-s','-u','-m','milplus.rmdl_tf','--binding',str(out/'input_binding.json'),
               '--request',str(out/'request.json'),'--output',str(out)]
    if stop_after is not None:
        command += ['--stop-stage','MIL','--stop-after',str(stop_after)]
    if stop_localizer is not None:
        command += ['--stop-stage','localizer','--stop-after',str(stop_localizer)]
    lock = out/'RUNNING.lock'
    try:
        descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        raise ValueError('Inspect the actual RMDL controller/child process before recovering a locked output') from None
    with os.fdopen(descriptor,'w') as stream:
        stream.write(str(os.getpid()))
    try:
        write_json(out/'status.json',{'state':'running','backend':request['backend'],'command':command})
        with (out/('resume_native.log' if resume else 'native.log')).open('a') as stream:
            child = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            write_json(out/'native_process.json',{'pid':child.pid,'command':command});code = child.wait()
        if code:
            raise RuntimeError('Full native RMDL failed; inspect its actual native log')
        state = read_json(out/'native_status.json')
        if state['state'] not in {'interrupted','completed'}:
            raise ValueError('Actual RMDL full trainer did not return a valid terminal invocation state')
        write_json(out/'status.json',{**state,'backend':request['backend']})
        return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','backend':request['backend'],'error':repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def evaluate(run, subset='test', device='cpu'):
    from .engine import score
    out = Path(run).resolve();request = read_json(out/'request.json')
    if subset not in {'val','test'} or read_json(out/'status.json')['state'] != 'completed':
        raise ValueError('RMDL evaluation requires completed localization/mining and the full declared MIL budget')
    if device != request['device']:
        raise ValueError('RMDL exact-state evaluation currently requires the recorded explicit device/runtime')
    if request['software'] != software_identity():
        raise ValueError('RMDL evaluation code differs from the locked fitted method')
    bundle = Bundle(request['data'],request['fm'],request['split']);inputs = bundle.rmdl
    if inputs is None or inputs.export() != read_json(out/'input_binding.json'):
        raise ValueError('Actual RMDL evaluation patients, labels, masks or pixel geometry changed')
    executable,env = environment(request['config'],device)
    if _runtime(executable,env) != request['tensorflow_runtime']:
        raise ValueError('RMDL evaluation runtime differs from the recorded full training stack')
    command = [executable,'-s','-u','-m','milplus.rmdl_tf','--binding',str(out/'input_binding.json'),
        '--request',str(out/'request.json'),'--output',str(out),'--evaluate',subset]
    with (out/(subset+'_evaluation.log')).open('a') as stream:
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    frame = pd.DataFrame(read_json(out/(subset+'_predictions_native.json')))
    rows = bundle.records(subset)
    if frame.slide_id.tolist() != [r['slide_id'] for r in rows] or frame.target.tolist() != [int(r['target']) for r in rows]:
        raise ValueError('RMDL predictions are not aligned with actual fixed patient/slide labels')
    metrics = score(bundle,frame);frame.to_csv(out/(subset+'_predictions.csv'),index=False)
    write_json(out/(subset+'_metrics.json'),metrics)
    return metrics

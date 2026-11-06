"""Public original-pretrained FEATHER entry; full research validation is separate."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .data import Bundle
from .feather_training import reference_recipe
from .io import read_json, write_json
from .provenance import software_identity
from .feather_spec import EXECUTION_CONDITIONS as PROFILES


def environment(config, out):
    executable = config.get('runtime_python') or os.environ.get('MILPLUS_FEATHER_PYTHON') or sys.executable
    if not Path(executable).is_file():
        raise FileNotFoundError('Declare the actual FEATHER isolated runtime executable')
    paths = [str(Path(__file__).resolve().parent.parent)]
    extra = config.get('runtime_dependencies') or os.environ.get('MILPLUS_FEATHER_DEPENDENCIES')
    if extra:
        path = Path(extra).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError('FEATHER native dependency prefix is missing')
        paths.append(str(path))
    cache = Path(out) / 'private_runtime_cache'
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(paths), 'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': str(config['threads']), 'MKL_NUM_THREADS': str(config['threads']),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1', 'HF_HUB_CACHE': str(cache / 'hub'),
        'HF_MODULES_CACHE': str(cache / 'modules')}
    return str(executable), env


def runtime(executable, env):
    result = subprocess.run([executable, '-s', '-c',
        'import json;from milplus.feather_runtime import runtime_identity;print(json.dumps(runtime_identity()))'],
        capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError('FEATHER runtime preflight failed; use a separate .[feather] environment '
                           'and MILPLUS_FEATHER_PYTHON.\n' + result.stderr[-6000:])
    return json.loads(result.stdout)


def train(data, method, output, *, fm=None, split=None, seed=2026, device='cpu',
          config=None, resume=False, stop_after=None):
    if method not in PROFILES:
        raise ValueError('Choose an explicit FEATHER24K native encoder and complete downstream reference')
    bundle = Bundle(data, fm, split)
    inputs = getattr(bundle, 'feather', None)
    if inputs is None or inputs.encoder != PROFILES[method]['encoder']:
        raise ValueError('Bind actual original FEATHER pretrained weights to this native visual encoder first')
    supplied = config or {}
    if not isinstance(supplied, dict) or set(supplied) - {'threads', 'model'}:
        raise ValueError('FEATHER preserves its complete20/10epoch recipe; declare only threads and runtime/interruption settings')
    settings = supplied.get('model', {})
    if not isinstance(settings, dict) or set(settings) - {'runtime_python', 'runtime_dependencies', 'stop_after_train_batches'}:
        raise ValueError('FEATHER native512/384 architecture, original pretrained initialization, KNN20 and recipe cannot be reduced')
    cfg = {'encoder': PROFILES[method]['encoder'], 'recipe': reference_recipe(PROFILES[method]['mode']),
        'threads': supplied.get('threads', 2), 'runtime_python': settings.get('runtime_python'),
        'runtime_dependencies': settings.get('runtime_dependencies')}
    if type(seed) is not int or not 0 <= seed < 2**32 or type(cfg['threads']) is not int or cfg['threads'] < 1:
        raise ValueError('FEATHER requires a32bit seed and positive integer thread count')
    if device != 'cpu' and (not device.startswith('cuda:') or not device[5:].isdigit()):
        raise ValueError('Declare an actual cpu or cuda:N FEATHER device')
    if stop_after is not None:
        raise ValueError('FEATHER resumes at actual weighted-sampling minibatch boundaries; declare stop_after_train_batches')
    stop_batch = settings.get('stop_after_train_batches')
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError('FEATHER interruption requires a positive actual TRAIN minibatch count')
    if cfg['recipe']['mode'] == 'frozen_knn20' and (resume or stop_batch is not None or len(bundle.records('train')) < 20):
        raise ValueError('Frozen FEATHER KNN20 needs at least20 TRAIN slides and no optimizer-resume settings')
    inputs.verify(deep=True)
    request = {'schema_version': 1, 'backend': 'feather24k_full_pretrained_reference',
        'method': method, 'data': str(bundle.root), 'fm': bundle.fm, 'split': bundle.split,
        'seed': seed, 'device': device, 'config': cfg, 'input_contract': inputs.contract,
        'software': software_identity(), 'synthetic': bool(bundle.config.get('synthetic', False)),
        'native_precision_condition': 'cuda_float16_AMP_reference' if device.startswith('cuda:') else 'cpu_float32_reference',
        'full_pretrained_validation': 'not_run', 'real_wsi_validation': 'not_run',
        'paper_reproduction': 'not_run', 'formal_admission': False}
    out = Path(output).expanduser().resolve()
    executable, env = environment(cfg, out)
    request['runtime'] = runtime(executable, env)
    child = request['runtime']['software']
    if (child['source_sha256'] != request['software']['source_sha256']
            or child['sources_lock_sha256'] != request['software']['sources_lock_sha256']):
        raise ValueError('FEATHER isolated runtime code differs from its common entry')
    if resume:
        if read_json(out / 'request.json') != request or read_json(out / 'status.json')['state'] == 'completed':
            raise ValueError('FEATHER recovery changed actual inputs/recipe/runtime or already completed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'request.json', request)
    lock = out / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('Inspect the actual FEATHER child before recovering a locked run') from None
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    process = None
    try:
        command = [executable, '-s', '-u', '-m', 'milplus.feather_runtime',
                   '--request', str(out / 'request.json'), '--output', str(out)]
        if resume:
            command.append('--resume')
        if stop_batch is not None:
            command += ['--stop-after-batch', str(stop_batch)]
        write_json(out / 'status.json', {'state': 'running', 'backend': request['backend'], 'command': command})
        with (out / ('resume_native.log' if resume else 'native.log')).open('a') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(out / 'native_process.json', {'pid': process.pid, 'command': command})
            code = process.wait()
        if code:
            raise RuntimeError('The full original-pretrained FEATHER runtime failed; inspect its native log')
        status = read_json(out / 'native_status.json')
        if status['state'] not in {'completed', 'interrupted'}:
            raise ValueError('FEATHER returned an invalid native run state')
        write_json(out / 'status.json', {**status, 'backend': request['backend']})
        return out
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(error), 'backend': request['backend']})
        raise
    finally:
        if process is None or process.poll() is not None:
            lock.unlink(missing_ok=True)


def evaluate(run, subset='test', device='cpu'):
    from .engine import score
    out = Path(run).expanduser().resolve()
    request = read_json(out / 'request.json')
    if (subset not in {'val', 'test'} or read_json(out / 'status.json')['state'] != 'completed'
            or device != request['device'] or request['software'] != software_identity()):
        raise ValueError('FEATHER evaluation requires completed training with its same actual source/device')
    bundle = Bundle(request['data'], request['fm'], request['split'])
    if bundle.feather.contract != request['input_contract']:
        raise ValueError('Actual FEATHER evaluation patient population/native feature pairing changed')
    bundle.feather.verify(deep=True)
    executable, env = environment(request['config'], out)
    if runtime(executable, env) != request['runtime']:
        raise ValueError('FEATHER evaluation dependency stack changed')
    command = [executable, '-s', '-u', '-m', 'milplus.feather_runtime',
        '--request', str(out / 'request.json'), '--output', str(out), '--evaluate', subset]
    with (out / (subset + '_evaluation.log')).open('a') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    frame = pd.DataFrame(read_json(out / (subset + '_predictions_native.json')))
    rows = bundle.records(subset)
    if (frame.slide_id.tolist() != [r['slide_id'] for r in rows]
            or frame.global_patient_id.tolist() != [r['global_patient_id'] for r in rows]
            or frame.target.tolist() != [int(r['target']) for r in rows]):
        raise ValueError('FEATHER actual prediction identities or common label order changed')
    metrics = score(bundle, frame)
    frame.to_csv(out / (subset + '_predictions.csv'), index=False)
    write_json(out / (subset + '_metrics.json'), metrics)
    return metrics

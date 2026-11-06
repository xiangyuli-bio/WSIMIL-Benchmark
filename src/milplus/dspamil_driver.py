"""Public full DSPA2025 reference entry, without original-asset or DSPA-U claims."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pandas as pd
import yaml

from .data import Bundle
from .dspamil_configuration import PROFILES, resolve, validate
from .dspamil_metrics import metrics
from .io import read_json, write_json
from .provenance import software_identity


def environment(config, out, *, probe=False):
    executable = config['runtime_python'] or sys.executable
    if not Path(executable).is_file():
        raise FileNotFoundError('Declare the real isolated DSPA runtime executable')
    output = Path(out)
    cache = (output.parent / '.milplus_dspamil_runtime_probes' / output.name
             if probe else output / 'private_runtime_cache')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parent.parent),
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': str(config['threads']), 'MKL_NUM_THREADS': str(config['threads']),
        'MKL_CBWR': os.environ.get('MKL_CBWR', 'COMPATIBLE'), 'GIT_OPTIONAL_LOCKS': '0',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    for key in ['TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
                'XDG_CACHE_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HUB_CACHE', 'HF_MODULES_CACHE']:
        env[key] = str(cache / key.lower())
    return str(executable), env


def runtime(executable, env):
    result = subprocess.run([executable, '-s', '-c',
        'import json;from milplus.dspamil_runtime import runtime_identity;print(json.dumps(runtime_identity()))'],
        capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError('DSPA-MIL runtime preflight failed; configure model.runtime_python '
                           'with the separate .[dspamil] environment.\n' + result.stderr[-6000:])
    return json.loads(result.stdout)


def process_identity(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        # The executable name can contain spaces/parentheses. starttime is the
        # twentieth field after the final closing parenthesis, not split()[21].
        text = path.read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rfind(')') + 2:].split()
    if fields[0] == 'Z':
        return None
    return {'pid': pid, 'starttime_ticks': int(fields[19]), 'host': socket.gethostname()}


def claim(out):
    lock = out / 'RUNNING.lock'
    if lock.exists():
        owner = read_json(lock)
        child_path = out / 'native_process.json'
        child = read_json(child_path).get('identity') if child_path.exists() else None
        for item in [owner, child]:
            if item is None:
                continue
            if item.get('host') != socket.gethostname():
                raise ValueError('DSPA existing run belongs to another host; verify that actual process before recovery')
            if process_identity(item['pid']) == item:
                raise ValueError('An actual DSPA owner/child is still running; do not duplicate or replace it')
        lock.rename(out / f'verified_terminal_previous_lock_{time.time_ns()}.json')
    owner = process_identity(os.getpid())
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(owner, stream)
    return lock, owner


def train(data, method, output, *, fm=None, split=None, seed=2026, device='cpu',
          config=None, resume=False, stop_after=None):
    bundle = Bundle(data, fm, split)
    inputs = bundle.dspamil
    if inputs is None:
        raise ValueError('Prepare actual DSPA native UNI/concept/TRAIN-pattern/patient inputs first')
    cfg, stop_batch = resolve(method, bundle, config)
    if (type(seed) is not int or not 0 <= seed < 2**32
            or (device != 'cpu' and (not device.startswith('cuda:') or not device[5:].isdigit()))):
        raise ValueError('DSPA needs an actual cpu/cuda:N device and32bit seed')
    if stop_after is not None:
        raise ValueError('DSPA recovery checkpoints actual updates; use model.stop_after_train_batches')
    if stop_batch is not None and (type(stop_batch) is not int or stop_batch < 1):
        raise ValueError('DSPA interruption requires a positive actual TRAIN minibatch count')
    inputs.verify(deep=True)
    out = Path(output).expanduser().resolve()
    executable, env = environment(cfg, out)
    probe_executable, probe_env = environment(cfg, out, probe=True)
    observed = runtime(probe_executable, probe_env)
    entry = software_identity()
    if any(entry[key] != observed['software'][key] for key in ['source_sha256', 'sources_lock_sha256']):
        raise ValueError('DSPA isolated runtime source differs from its common entry')
    request = {'schema_version': 1, 'backend': 'dspamil2025_full_PAT_reference',
        'method': method, 'data': str(bundle.root), 'fm': bundle.fm, 'split': bundle.split,
        'seed': seed, 'device': device, 'config': cfg, 'input_contract': inputs.contract,
        'runtime': observed, 'software': observed['software'], 'entry_software': entry,
        'synthetic': bool(bundle.config.get('synthetic', False)),
        'original_expert_assets_verified': False, 'DSPA_U_implemented': False,
        'unpublished_details_are_explicit_references': True, 'paper_reproduction': False,
        'real_wsi_validation': 'not_run', 'formal_admission': False}
    if resume:
        if read_json(out / 'request.json') != request or read_json(out / 'status.json')['state'] == 'completed':
            raise ValueError('DSPA resume changed actual inputs, full recipe/runtime or already completed the run')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'request.json', request)
        write_json(out / 'status.json', {'state': 'initializing', 'backend': request['backend'],
            'formal_admission': False})
    lock, owner = claim(out)
    process = None
    try:
        for key in ['TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
                    'XDG_CACHE_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HUB_CACHE', 'HF_MODULES_CACHE']:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        command = [executable, '-s', '-u', '-m', 'milplus.dspamil_runtime',
                   '--request', str(out / 'request.json'), '--output', str(out)]
        if resume:
            command.append('--resume')
        if stop_batch is not None:
            command += ['--stop-after-batch', str(stop_batch)]
        write_json(out / 'status.json', {'state': 'running', 'command': command,
            'backend': request['backend'], 'formal_admission': False})
        with (out / ('resume_native.log' if resume else 'native.log')).open('a') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(out / 'native_process.json', {'identity': process_identity(process.pid),
                'pid': process.pid, 'command': command})
            code = process.wait()
        if code:
            raise RuntimeError('Full DSPA reference runtime failed; inspect its native log and preserve its checkpoint')
        status = read_json(out / 'native_status.json')
        if status['state'] not in {'completed', 'interrupted'}:
            raise ValueError('DSPA returned an invalid full-model run state')
        write_json(out / 'status.json', {**status, 'backend': request['backend']})
        return out
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(error),
            'backend': request['backend'], 'formal_admission': False})
        raise
    finally:
        if process is None or process.poll() is not None:
            if lock.exists() and read_json(lock) == owner:
                lock.unlink()


def evaluate(run, subset='test', device='cpu'):
    out = Path(run).expanduser().resolve()
    request = read_json(out / 'request.json')
    validate(request['config'])
    if (subset not in {'val', 'test'} or read_json(out / 'status.json')['state'] != 'completed'
            or device != request['device']):
        raise ValueError('DSPA evaluation requires a completed full-model run and its recorded device')
    entry = software_identity()
    if any(entry[key] != request['software'][key] for key in ['source_sha256', 'sources_lock_sha256']):
        raise ValueError('DSPA evaluation entry source differs from the recorded run')
    bundle = Bundle(request['data'], request['fm'], request['split'])
    if bundle.dspamil is None or bundle.dspamil.contract != request['input_contract']:
        raise ValueError('DSPA evaluation concepts or actual patient population changed')
    bundle.dspamil.verify(deep=True)
    executable, env = environment(request['config'], out)
    if runtime(executable, env) != request['runtime']:
        raise ValueError('DSPA evaluation actual runtime changed')
    command = [executable, '-s', '-u', '-m', 'milplus.dspamil_runtime',
               '--request', str(out / 'request.json'), '--output', str(out), '--evaluate', subset]
    with (out / (subset + '_evaluation.log')).open('a') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    frame = pd.DataFrame(read_json(out / (subset + '_predictions_native.json')))
    rows = bundle.records(subset)
    if (frame.slide_id.tolist() != [r['slide_id'] for r in rows]
            or frame.global_patient_id.tolist() != [r['global_patient_id'] for r in rows]
            or frame.target.tolist() != [int(r['target']) for r in rows]
            or set(frame.subset) != {subset}):
        raise ValueError('DSPA actual prediction identities or held-out membership changed')
    result = metrics(frame,
        rule=request['config']['recipe']['reference_choices']['ordinal_decision'],
        unit=bundle.config.get('evaluation_unit', 'slide'))
    if result != read_json(out / (subset + '_metrics_native.json')):
        raise ValueError('DSPA exported ordinal decisions differ from its actual validation/evaluation runtime')
    frame.to_csv(out / (subset + '_predictions.csv'), index=False)
    write_json(out / (subset + '_metrics.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('demo', help='Explicit synthetic shape/control inputs, not original expert assets')
    p.add_argument('--output', required=True)
    p.add_argument('--patches', type=int, default=1025)
    p = sub.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--method', choices=list(PROFILES), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--fm')
    p.add_argument('--split')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('evaluate')
    p.add_argument('--run', required=True)
    p.add_argument('--subset', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.command == 'demo':
        from .dspamil_demo import create_demo
        result = create_demo(args.output, patches=args.patches)
    elif args.command == 'train':
        from .config_io import load_training_config
        result = train(args.data, args.method, args.output, fm=args.fm, split=args.split,
            seed=args.seed, device=args.device, config=load_training_config(args.config), resume=args.resume)
    else:
        result = evaluate(args.run, args.subset, args.device)
    print(json.dumps(str(result) if isinstance(result, Path) else result, indent=2))


if __name__ == '__main__':
    main()

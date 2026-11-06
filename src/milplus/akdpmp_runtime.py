"""Per-method CPU CNR startup; other MIL/text-encoder runtimes stay untouched."""
import json
import os
from pathlib import Path
import subprocess
import sys

MARKER='MILPLUS_AKDPMP_CPU_STARTUP'


def desired_policy(args):
    if not str(getattr(args,'device','cpu')).startswith('cpu'):return None
    if args.command=='run':
        name=''.join(c for c in args.method.lower() if c.isalnum())
        if not name.startswith('akdpmp'):return None
        from .config_io import load_training_config
        config=load_training_config(args.config)
        threads=config.get('threads',2)
    elif args.command=='evaluate':
        request=json.loads((Path(args.run)/'request.json').read_text())
        if request.get('backend')!='akdpmp_complete_continual_reference':return None
        # Preserve an explicitly requested in-process author/default CPU run.
        # Canonical CLI-created runs carry the reproducible CNR policy.
        policy=request.get('cpu_numerical_policy') or {}
        if not policy.get('conditional_reproducibility',False):return None
        threads=request['threads']
    else:return None
    if type(threads) is not int or threads<1:raise ValueError('AKD-PMP CPU threads must be a positive integer')
    return {'MKL_CBWR':'COMPATIBLE','MKL_DYNAMIC':'FALSE','OMP_DYNAMIC':'FALSE',
        'MKL_NUM_THREADS':str(threads),'OMP_NUM_THREADS':str(threads),MARKER:'cnr_v1'}


def maybe_restart(args,argv=None):
    desired=desired_policy(args)
    if desired is None or all(os.environ.get(k)==v for k,v in desired.items()):return None
    env={**os.environ,**desired};arguments=list(sys.argv[1:] if argv is None else argv)
    command=[sys.executable,'-m','milplus',*arguments]
    if argv is None:os.execvpe(sys.executable,command,env)
    # Embedded CLI callers retain their own interpreter and its BLAS settings.
    return subprocess.run(command,env=env).returncode


def numerical_policy(threads):
    keys=['MKL_CBWR','MKL_DYNAMIC','OMP_DYNAMIC','MKL_NUM_THREADS','OMP_NUM_THREADS']
    env={k:os.environ.get(k) for k in keys}
    return {'conditional_reproducibility':env['MKL_CBWR']=='COMPATIBLE' and
        env['MKL_DYNAMIC']=='FALSE' and env['OMP_DYNAMIC']=='FALSE' and
        env['MKL_NUM_THREADS']==str(threads) and env['OMP_NUM_THREADS']==str(threads),
        'threads':threads,'environment':env,'precision':'original_float32',
        'scope':'CPU BLAS startup only; no model/loss/gradient/epoch change'}

"""Build the entire pinned original CPU/CUDA backend in its own directory.

No source surgery, reduced kernels, global installation or existing environment
changes. CUDA selection follows the original setup.py on the actual GPU host.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from milplus.io import digest_file, write_json
from milplus.sparseconvmil_source import SCN_COMMIT
from milplus.upstream import verified_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError('A positive compile-job count is required')
    source, commit = verified_source('sparseconvnet')
    if commit != SCN_COMMIT:
        raise ValueError('SparseConvNet source commit changed')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    build = output/'source'
    shutil.copytree(source, build, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'build', 'dist'))
    hashes = {str(p.relative_to(build)): digest_file(p) for p in build.rglob('*') if p.is_file()}
    command = [sys.executable, 'setup.py', 'build_ext', '--build-lib', str(output/'overlay'),
               '--build-temp', str(output/'temp')]
    state = {'state': 'building', 'source_commit': commit, 'python': sys.executable, 'command': command,
             'source_files': hashes, 'source_modified': False, 'pid': os.getpid()}
    write_json(output/'process.json', state)
    try:
        with (output/'build.log').open('x') as log:
            child = subprocess.Popen(command, cwd=build,
                env={**os.environ, 'MAX_JOBS': str(args.jobs), 'TMPDIR': str(output), 'PYTHONDONTWRITEBYTECODE': '1'},
                stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid; write_json(output/'process.json', state)
            result = child.wait()
        if result:
            raise RuntimeError('Original SparseConvNet build failed; inspect build.log, no fallback backend used')
        target = output/'overlay/sparseconvnet'
        if len(list(target.glob('SCN*.so'))) != 1:
            raise ValueError('Original SparseConvNet compiled extension is missing')
        for path in (build/'sparseconvnet').glob('*.py'):
            shutil.copy2(path, target/path.name)
        if hashes != {str(p.relative_to(build)): digest_file(p) for p in build.rglob('*') if p.is_file()}:
            raise ValueError('Original source changed during SparseConvNet compilation')
        files = {str(p.relative_to(output/'overlay')): digest_file(p) for p in target.rglob('*') if p.is_file()}
        write_json(output/'manifest.json', {'state': 'compiled_original_backend', 'source_commit': commit,
            'source_modified': False, 'files': files, 'python': sys.executable})
        state.update(state='passed', overlay=str(output/'overlay'))
        write_json(output/'process.json', state)
        print(json.dumps({'state': 'passed', 'PYTHONPATH_addition': str(output/'overlay')}))
    except BaseException as error:
        state.update(state='failed', error=repr(error)); write_json(output/'process.json', state)
        raise


if __name__ == '__main__':
    main()

"""Successful temporary tensors are reclaimable while numerical evidence survives."""
import os
import subprocess
import sys
from pathlib import Path


def test_receipts_survive_pytest_cleanup_and_failed_fixture_is_retained(tmp_path):
    source = tmp_path/'test_probe.py'
    source.write_text('''import json

def test_success(tmp_path):
    (tmp_path/'weights.bin').write_bytes(b'x'*1000000)
    (tmp_path/'author_training.json').write_text(json.dumps({'state':'passed','updates':14}))

def test_failure(tmp_path):
    (tmp_path/'failure_debug.bin').write_bytes(b'failed fixture')
    assert False, 'controlled retention probe'
''')
    destination = tmp_path/'receipts'
    base = tmp_path/'generated'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'scripts.pytest_receipts',
        str(source), '--basetemp='+str(base), '-o', 'tmp_path_retention_policy=failed',
        '--junitxml='+str(tmp_path/'probe.xml')], check=False, cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'MILPLUS_TEST_RECEIPTS': str(destination)}, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout+result.stderr
    assert len(list(destination.rglob('author_training.json'))) == 1
    assert not list(base.rglob('weights.bin'))
    assert len(list(base.rglob('failure_debug.bin'))) == 1
    assert (tmp_path/'probe.xml').is_file()

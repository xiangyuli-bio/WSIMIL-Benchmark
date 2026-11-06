"""Generate actual full catalogs from a cold tree, preserving all entry data."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_cold_and_repeated_catalog_builds_preserve_complete_current_specs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    expected = {str(p.relative_to(root)):p.read_bytes() for p in [
        root/'src/milplus/resources/methods.json',root/'src/milplus/resources/profiles.json',
        *sorted((root/'method_specs').glob('*.json'))]}
    profiles = json.loads(expected['src/milplus/resources/profiles.json'])
    assert {'rmdl_source_sum_full_reference','rmdl_paper_mean_full_reference'} <= {p['id'] for p in profiles}
    assert len(json.loads(expected['src/milplus/resources/methods.json'])) == 164
    cold = tmp_path/'cold'
    shutil.copytree(root,cold,ignore=shutil.ignore_patterns('.git','.venv','build','*.egg-info',
        '__pycache__','.pytest_cache'))
    for relative in expected:
        (cold/relative).unlink()
    environment = {**os.environ,'PYTHONPATH':str(cold/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
                   'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for repeat in range(2):
        result = subprocess.run([sys.executable,'-s',str(cold/'scripts/build_catalog.py')],
            cwd=cold,env=environment,capture_output=True,text=True)
        assert result.returncode == 0,result.stdout+result.stderr
        for relative,original in expected.items():
            assert (cold/relative).read_bytes() == original,(repeat,relative)

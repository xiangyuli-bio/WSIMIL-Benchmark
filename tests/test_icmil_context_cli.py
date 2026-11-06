import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from milplus.io import digest_file,read_json
from test_icmil_context_data import fixture


def cli(*args):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],cwd=root,env=env,capture_output=True,text=True)


def test_public_prepare_run_and_two_fresh_native_evaluations(tmp_path):
    data=fixture(tmp_path/'data',32);prepared=tmp_path/'prepared';run=tmp_path/'run'
    p=cli('prepare-icmil-context','--data',data,'--output',prepared,'--projection','train_pca25_reference','--bag-size',10)
    assert p.returncode==0,p.stderr+p.stdout
    p=cli('run','--data',prepared,'--method','ICMIL In-context','--output',run)
    assert p.returncode==0,p.stderr+p.stdout
    p=cli('evaluate','--run',run);assert p.returncode==0,p.stderr+p.stdout
    first=digest_file(run/'test_predictions.csv')
    p=cli('evaluate','--run',run);assert p.returncode==0,p.stderr+p.stdout
    assert digest_file(run/'test_predictions.csv')==first
    assert read_json(run/'status.json')['downstream_optimizer_updates']==0


def test_native_call_rejects_any_query_label_field_before_loading(tmp_path):
    from milplus.icmil_context_driver import environment
    python,env=environment({'threads':2});call=tmp_path/'call.json';call.write_text(json.dumps({'query_labels':[1]}))
    p=subprocess.run([python,'-m','milplus.icmil_context_native','--call',str(call)],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'unlabelled queries' in p.stderr


def test_original_three_checkpoints_and_full_old_scope_remain_distinct():
    from milplus.icmil_context_source import WEIGHTS
    from milplus.registry import methods,profiles
    assert len(WEIGHTS)==3 and len(set(WEIGHTS.values()))==3
    assert len(methods())==164
    assert len([p for p in profiles() if p['family']=='icmil_context'])==3

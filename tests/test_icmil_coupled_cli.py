import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from milplus.io import read_json,digest_file

WEIGHTS='/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/development/icmil_coupled_v92/legacy14_assets_r1/resnet50-11ad3fa6.pth'


def cli(*args):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],cwd=root,env=env,capture_output=True,text=True)


@pytest.mark.parametrize('method',['ICMIL Coupled','icmil_coupled_maxpool_patient_reference','icmil_coupled_dtfd_legacy14_patient_reference'])
def test_public_complete_pipeline_and_two_fresh_selected_evaluations(tmp_path,method):
    data=tmp_path/'data';run=tmp_path/'run';config=tmp_path/'software_budget.yaml'
    config.write_text(yaml.safe_dump({'epochs':1,'threads':2,'model':{'iterations':1,'distill_epochs':1}}))
    r=cli('demo-icmil-coupled','--output',data,'--weights',WEIGHTS);assert r.returncode==0,r.stderr+r.stdout
    r=cli('check','--data',data,'--method',method,'--deep');assert r.returncode==0,r.stderr+r.stdout
    r=cli('run','--data',data,'--method',method,'--output',run,'--config',config);assert r.returncode==0,r.stderr+r.stdout
    assert read_json(run/'status.json')['MIL_phases']==2
    r=cli('evaluate','--run',run);assert r.returncode==0,r.stderr+r.stdout
    first=digest_file(run/'test_predictions.csv');r=cli('evaluate','--run',run)
    assert r.returncode==0 and digest_file(run/'test_predictions.csv')==first,r.stderr+r.stdout


def test_two_ICMIL_methods_are_distinct_and_native_widths_cannot_be_replaced(tmp_path):
    from milplus.registry import inspect_method
    assert inspect_method('ICMIL Coupled')['family']=='icmil_coupled'
    assert inspect_method('ICMIL In-context')['family']=='icmil_context'
    data=tmp_path/'data';r=cli('demo-icmil-coupled','--output',data,'--weights',WEIGHTS);assert r.returncode==0,r.stderr
    config=tmp_path/'wrong.yaml';config.write_text(yaml.safe_dump({'model':{'hidden':32}}))
    r=cli('run','--data',data,'--method','ICMIL Coupled','--output',tmp_path/'run','--config',config)
    assert r.returncode!=0 and 'source Adam' in r.stderr

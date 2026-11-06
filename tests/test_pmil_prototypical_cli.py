import os
from pathlib import Path
import subprocess
import sys

import yaml

from milplus.io import digest_file,read_json
from milplus.registry import inspect_method,profiles,resource

WEIGHTS=os.environ.get('MILPLUS_PMIL_WEIGHTS',str(Path.home()/'.cache/milplus/pmil/resnet34-333f7ec4.pth'))


def cli(*args):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],cwd=root,env=env,capture_output=True,text=True)


def test_public_raw_PMIL_pipeline_and_two_independent_cli_evaluations(tmp_path):
    data=tmp_path/'data';run=tmp_path/'run';config=tmp_path/'software_only.yaml'
    config.write_text(yaml.safe_dump({'epochs':1,'threads':2,'model':{'encoder_epochs':1}}))
    result=cli('demo-pmil-prototypical','--output',data,'--weights',WEIGHTS)
    assert result.returncode==0,result.stderr+result.stdout
    result=cli('check','--data',data,'--method','PMIL Prototypical','--deep')
    assert result.returncode==0,result.stderr+result.stdout
    result=cli('run','--data',data,'--method','PMIL Prototypical','--output',run,'--config',config)
    assert result.returncode==0,result.stderr+result.stdout
    assert read_json(run/'status.json')['state']=='completed'
    assert not (run/'evaluation_test').exists()
    result=cli('evaluate','--run',run)
    assert result.returncode==0,result.stderr+result.stdout
    before=digest_file(run/'test_predictions.csv');result=cli('evaluate','--run',run)
    assert result.returncode==0,result.stderr+result.stdout
    assert digest_file(run/'test_predictions.csv')==before


def test_PMIL_paper_id_and_all_existing_profiles_survive_catalog_regeneration(tmp_path):
    assert inspect_method('PMIL Prototypical')['family']=='pmil_prototypical'
    before=profiles();root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/build_catalog.py'],cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr+result.stdout
    assert profiles()==before
    assert any(p['family']=='pmil_shapley' for p in before)
    catalog={r['id']:r for r in resource('methods.json')}
    assert catalog['protomil']['name']=='ProtoMIL'
    assert catalog['protomil']['id']!=inspect_method('PMIL Prototypical')['family']

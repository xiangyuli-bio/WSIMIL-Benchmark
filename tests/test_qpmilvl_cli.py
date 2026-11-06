import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from milplus.qpmilvl_driver import PROFILES
from test_qpmilvl_source import WEIGHTS,SHA,DEVICE


def call(*args):
    p=subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,env=os.environ.copy())
    assert p.returncode==0,p.stdout+p.stderr
    return p


@pytest.mark.parametrize('profile',list(PROFILES))
def test_actual_public_CLI_both_complete_profiles_and_two_fresh_final_model_reloads(tmp_path,profile):
    data=tmp_path/'data';call('demo-qpmilvl','--output',data,'--weights',WEIGHTS,'--sha256',SHA)
    cfg=tmp_path/'epochs.yaml';cfg.write_text(yaml.safe_dump({'epochs':2,'threads':2}));out=tmp_path/'run'
    call('check','--data',data,'--method',profile)
    call('run','--data',data,'--method',profile,'--output',out,'--config',cfg,'--device',DEVICE)
    previous=None
    for _ in range(2):
        call('evaluate','--run',out,'--device',DEVICE);actual=(out/'test_predictions.csv').read_bytes()
        if previous is not None:assert actual==previous
        previous=actual
    status=json.loads((out/'status.json').read_text());assert status['completed_tasks']==4 and status['updates']==8


def test_catalog_preserves_all_prior313_profile_bytes_and_all164_planned_families(tmp_path):
    from milplus.registry import profiles,methods,inspect_method
    root=Path(__file__).resolve().parents[1];prior=json.loads((root/'tests/fixtures/qpmilvl_prior313_profiles.json').read_text())
    # Keep this historical QPMIL census on its exact 0.96 profile domain;
    # the successor KMP test separately asserts the expanded full catalog.
    current={p['id']:p for p in profiles() if p['id'] in {r['id'] for r in prior}|set(PROFILES)};assert len(prior)==313 and all(current[p['id']]==p for p in prior)
    assert len(current)==315 and len({p['family'] for p in current.values()})==149 and len(methods())==164
    assert {p['id'] for p in current.values() if p['family']=='qpmilvl'}==set(PROFILES)
    assert inspect_method('QPMIL-VL')['id']=='qpmilvl_conch_max_query_patient_reference'
    path=root/'src/milplus/resources/profiles.json';before=path.read_bytes()
    result=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],capture_output=True,text=True,cwd=root)
    assert result.returncode==0,result.stdout+result.stderr
    assert path.read_bytes()==before

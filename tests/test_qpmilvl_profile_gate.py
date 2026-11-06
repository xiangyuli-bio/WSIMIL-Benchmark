import json
import os
from pathlib import Path
import subprocess
import sys
from milplus.qpmilvl_driver import PROFILES
from test_qpmilvl_source import WEIGHTS,SHA,DEVICE


def test_full_library_verifier_actually_runs_both_full_CONCH_continual_recipes(tmp_path):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'matrix'
    cmd=[sys.executable,str(root/'scripts/validate_all_profiles.py'),'--output',str(out),'--device',DEVICE,
         '--qpmilvl-weights',str(WEIGHTS),'--qpmilvl-sha256',SHA]
    for name in PROFILES:cmd+=['--profile',name]
    result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy());assert result.returncode==0,result.stdout+result.stderr
    rows=json.loads((out/'validation.json').read_text())['profiles'];assert {r['id'] for r in rows}==set(PROFILES)
    for row in rows:
        assert row['state']=='passed';status=json.loads((out/row['id']/'status.json').read_text())
        assert status['completed_tasks']==4 and status['epochs_completed']==8 and status['updates']==8

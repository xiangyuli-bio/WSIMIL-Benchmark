import json
import os
from pathlib import Path
import subprocess
import sys


def test_full_library_verifier_executes_all_four_complete_author_recipes(tmp_path):
    from milplus.akdpmp_driver import PROFILES
    root=Path(__file__).resolve().parents[1];out=tmp_path/'matrix';cmd=[sys.executable,str(root/'scripts/validate_all_profiles.py'),'--output',str(out)]
    for name in PROFILES:cmd+=['--profile',name]
    result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy());assert result.returncode==0,result.stdout+result.stderr
    rows=json.loads((out/'validation.json').read_text())['profiles'];assert {r['id'] for r in rows}==set(PROFILES)
    for row in rows:
        assert row['state']=='passed';status=json.loads((out/row['id']/'status.json').read_text())
        assert status['completed_tasks']==3 and status['epochs_completed']==6 and status['updates']>12 and status['memory_samples']>0

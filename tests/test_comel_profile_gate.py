import json
import os
from pathlib import Path
import subprocess
import sys


def test_full_profile_verifier_runs_both_complete_continual_variants(tmp_path):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'gate'
    cmd=[sys.executable,str(root/'scripts/validate_all_profiles.py'),'--output',str(out)]
    for profile in ['comel_owlora_source_patient_reference','comel_owlora_svd_basis_repaired']:cmd+=['--profile',profile]
    actual=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy())
    assert actual.returncode==0,actual.stdout+actual.stderr
    rows=json.loads((out/'validation.json').read_text())['profiles'];assert len(rows)==2
    for row in rows:
        assert row['state']=='passed'
        state=json.loads((out/row['id']/'status.json').read_text())
        assert state['completed_tasks']==3 and state['epochs_completed']==6 and state['updates']==12
        assert len(row['metrics']['matrix'])==3

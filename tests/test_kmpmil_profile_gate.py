import os,json,sys,subprocess
from pathlib import Path
from milplus.kmpmil_driver import PROFILES
from test_kmpmil_data import WEIGHTS,SHA
DEVICE=os.environ.get('MILPLUS_KMPMIL_TEST_DEVICE','cpu')


def test_full_library_verifier_executes_both_actual_KMP_protocols(tmp_path):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'matrix'
    cmd=[sys.executable,str(root/'scripts/validate_all_profiles.py'),'--output',str(out),'--device',DEVICE,'--kmpmil-weights',str(WEIGHTS),'--kmpmil-sha256',SHA]
    for name in PROFILES:cmd+=['--profile',name]
    p=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy());assert p.returncode==0,p.stdout+p.stderr
    rows=json.loads((out/'validation.json').read_text())['profiles'];assert {r['id'] for r in rows}==set(PROFILES)
    for row in rows:
        assert row['state']=='passed';state=json.loads((out/row['id']/'status.json').read_text())
        assert state['completed_tasks']==4 and state['epochs_completed']==8 and state['updates']==16

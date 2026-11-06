import sys,os,json,subprocess
from pathlib import Path
from milplus.protomil_driver import PROFILES


def test_full_library_verifier_runs_all_four_complete_concept_and_intervention_protocols(tmp_path):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'all'
    command=[sys.executable,str(root/'scripts/validate_all_profiles.py'),'--output',str(out),'--device','cpu']
    for name in PROFILES:command+=['--profile',name]
    p=subprocess.run(command,env=os.environ.copy(),capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
    rows=json.loads((out/'validation.json').read_text())['profiles'];assert {r['id'] for r in rows}==set(PROFILES)
    for row in rows:
        assert row['state']=='passed';run=out/row['id'];assert json.loads((run/'status.json').read_text())['state']=='completed'
        history=json.loads((run/'MIL/history.json').read_text());assert len(history)==2 and history[-1]['updates']==12
        assert json.loads((run/'SAE/history.json').read_text())['actual_completed_pieces']==6

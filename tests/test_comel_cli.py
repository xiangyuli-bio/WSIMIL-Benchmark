import json
import os
from pathlib import Path
import subprocess
import sys

from milplus.comel_demo import create_demo
from milplus.registry import inspect_method


def test_actual_public_cli_complete_sequence_and_two_fresh_process_evaluations(tmp_path):
    data=create_demo(tmp_path/'data',tasks=2);run=tmp_path/'run'
    command=[sys.executable,'-m','milplus']
    for args in [
        ['run','--data',str(data),'--method','CoMEL','--output',str(run),'--epochs','2'],
        ['evaluate','--run',str(run),'--subset','test'],
        ['evaluate','--run',str(run),'--subset','test']]:
        r=subprocess.run(command+args,capture_output=True,text=True,env=os.environ.copy())
        assert r.returncode==0,r.stdout+r.stderr
    result=json.loads((run/'test_metrics.json').read_text());assert len(result['matrix'])==2
    assert json.loads((run/'status.json').read_text())['updates']==8
    assert inspect_method('CoMEL')['id']=='comel_owlora_source_patient_reference'


def test_all_prior_profiles_and_catalog_regeneration_preserved(tmp_path):
    from milplus.registry import profiles,methods
    root=Path(__file__).resolve().parents[1];current={p['id']:p for p in profiles()}
    # This test freezes the CoMEL admission scope. Additional explicitly
    # registered method families are checked by their own scope ledgers.
    prior_family_ids={p['family'] for p in json.loads((Path(__file__).parent/'fixtures/comel_prior307_profiles.json').read_text())}|{'comel'}
    assert len([p for p in current.values() if p['family'] in prior_family_ids])==309 and len(methods())==164
    prior=root.parent.parent/'pmil_prototypical_v93/source_r4/src/milplus/resources/profiles.json'
    # This package-level test also runs from an independently installed wheel;
    # compare the frozen prior JSON embedded in this test's adjacent fixture.
    fixture=Path(__file__).parent/'fixtures/comel_prior307_profiles.json'
    expected=json.loads(fixture.read_text())
    assert len(expected)==307 and all(current[p['id']]==p for p in expected)
    before=(root/'src/milplus/resources/profiles.json').read_bytes()
    r=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],cwd=root,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert (root/'src/milplus/resources/profiles.json').read_bytes()==before

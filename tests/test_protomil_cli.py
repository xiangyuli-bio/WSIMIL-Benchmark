import os,sys,json,subprocess
from pathlib import Path
import pytest,yaml
from milplus.protomil_driver import PROFILES


def call(*args):
    p=subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,env=os.environ.copy())
    assert p.returncode==0,p.stdout+p.stderr
    return p


def supply_review(run):
    from milplus.io import read_json,write_json
    request=read_json(run/'review_request.json');concept=request['active_concepts'][0]
    write_json(request['review_file'],{'schema_version':1,'sae_state_sha256':request['sae_state_sha256'],'concept_bank_sha256':request['concept_bank_sha256'],
        'reviewer':'synthetic_validation_only','provenance':'Unit numeric fixture only; not pathologist review','removed_concepts':[concept],
        'concept_names':{str(concept):'synthetic decision only'}})


@pytest.mark.parametrize('profile',list(PROFILES))
def test_complete_public_CLI_three_stages_actual_review_and_two_fresh_model_reloads(tmp_path,profile):
    data=tmp_path/'data';call('demo-protomil','--output',data);task=data/'task';out=tmp_path/'run'
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'sae_epochs':1,'mil_epochs':2,'probe_per_class':5,'threads':2}))
    call('check','--data',task,'--method',profile)
    call('run','--data',task,'--method',profile,'--output',out,'--config',config,'--stop-after',1)
    assert json.loads((out/'status.json').read_text())['state']=='interrupted'
    call('run','--data',task,'--method',profile,'--output',out,'--config',config,'--resume')
    if PROFILES[profile]['intervened']:
        assert json.loads((out/'status.json').read_text())['state']=='awaiting_concept_review' and not (out/'MIL').exists()
        supply_review(out);call('run','--data',task,'--method',profile,'--output',out,'--config',config,'--resume')
    assert json.loads((out/'status.json').read_text())['state']=='completed'
    previous=None
    for _ in range(2):
        call('evaluate','--run',out);actual=(out/'test_predictions.csv').read_bytes()
        if previous is not None:assert actual==previous
        previous=actual
    request=json.loads((out/'request.json').read_text());assert request['recipe']['native_sae_batch']==4096 and request['recipe']['classifier_mean_L1']==1.
    assert not request['test_used_for_training_or_selection']


def test_all_prior317_profile_records_and164_families_preserved_and_expanded_catalog_reproducible():
    from milplus.registry import profiles,methods,inspect_method
    root=Path(__file__).resolve().parents[1];prior=json.loads((root/'tests/fixtures/protomil_prior317_profiles.json').read_text());current={p['id']:p for p in profiles() if p['id'] in {r['id'] for r in prior}|set(PROFILES)}
    assert len(prior)==317 and all(current[p['id']]==p for p in prior)
    assert len(current)==321 and len({p['family'] for p in current.values()})==151 and len(methods())==164
    assert {p['id'] for p in current.values() if p['family']=='protomil'}==set(PROFILES)
    assert inspect_method('ProtoMIL')['id']=='protomil_concept_paper_patient_reference'
    before=(root/'src/milplus/resources/profiles.json').read_bytes()
    p=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],cwd=root,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
    assert (root/'src/milplus/resources/profiles.json').read_bytes()==before

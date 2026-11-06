import json,os,sys,subprocess
from pathlib import Path
import pytest,yaml
from milplus.kmpmil_driver import PROFILES
from test_kmpmil_data import WEIGHTS,SHA
DEVICE=os.environ.get('MILPLUS_KMPMIL_TEST_DEVICE','cpu')


def call(*args):
    p=subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,env=os.environ.copy())
    assert p.returncode==0,p.stdout+p.stderr
    return p


@pytest.mark.parametrize('profile',list(PROFILES))
def test_public_complete_KMP_training_both_query_profiles_and_two_fresh_class_and_task_CL_reloads(tmp_path,profile):
    data=tmp_path/'data';call('demo-kmpmil','--output',data,'--weights',WEIGHTS,'--sha256',SHA)
    cfg=tmp_path/'epochs.yaml';cfg.write_text(yaml.safe_dump({'epochs':3,'threads':2}));out=tmp_path/'run'
    call('check','--data',data,'--method',profile)
    call('run','--data',data,'--method',profile,'--output',out,'--config',cfg,'--device',DEVICE)
    previous=None
    for _ in range(2):
        call('evaluate','--run',out,'--device',DEVICE);actual=[(out/(f+'.csv')).read_bytes() for f in ['test_predictions','test_task_cl_predictions']]
        if previous is not None:assert actual==previous
        previous=actual
    state=json.loads((out/'status.json').read_text());assert state['completed_tasks']==4 and state['updates']==24
    req=json.loads((out/'request.json').read_text());assert req['recipe']['load_best_ckpt'] and req['flush_incomplete_tail'] and not req['old_WSI_replay']


def test_all_prior315_profiles_and_all164_families_are_preserved_and_catalog_is_reproducible():
    from milplus.registry import profiles,methods,inspect_method
    root=Path(__file__).resolve().parents[1];prior=json.loads((root/'tests/fixtures/kmpmil_prior315_profiles.json').read_text());current={r['id']:r for r in profiles() if r['id'] in {p['id'] for p in prior}|set(PROFILES)}
    assert len(prior)==315 and all(current[r['id']]==r for r in prior)
    assert len(current)==317 and len({r['family'] for r in current.values()})==150 and len(methods())==164
    assert {r['id'] for r in current.values() if r['family']=='kmpmil'}==set(PROFILES)
    assert inspect_method('KMP-MIL')['id']=='kmpmil_conch_max_query_patient_reference'
    p=root/'src/milplus/resources/profiles.json';before=p.read_bytes();result=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],capture_output=True,text=True,cwd=root)
    assert result.returncode==0,result.stdout+result.stderr
    assert p.read_bytes()==before

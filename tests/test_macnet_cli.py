import json,os,subprocess,sys
from pathlib import Path
import pytest,yaml
from milplus.macnet_driver import PROFILES
WEIGHTS=Path(os.environ.get('MILPLUS_MACNET_WEIGHTS','/root/MIL/development/macnet_v99_assets_r1/swin_tiny_patch4_window7_224_22k.pth'))
DEVICE=os.environ.get('MILPLUS_MACNET_TEST_DEVICE','cuda:0')


def call(*args,okay=True):
    p=subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,env=os.environ.copy())
    if okay:assert p.returncode==0,p.stdout+p.stderr
    else:assert p.returncode!=0
    return p


@pytest.fixture(scope='module')
def prepared_tasks(tmp_path_factory):
    root=tmp_path_factory.mktemp('macnet_actual_raw_inputs');tasks={}
    for condition in ['public','lscc81','cscc64']:
        target=root/condition;call('demo-macnet','--output',target,'--condition',condition,'--weights',WEIGHTS);tasks[condition]=target
    return tasks


@pytest.mark.parametrize('method',list(PROFILES))
def test_public_CLI_all6_original_raw_conditions_actual_epoch_recovery_and_two_fresh_reloads(tmp_path,prepared_tasks,method):
    data=prepared_tasks[PROFILES[method]['condition']];out=tmp_path/'run';cfg=tmp_path/'epochs.yaml';cfg.write_text(yaml.safe_dump({'epochs':2,'threads':2}))
    call('check','--data',data,'--method',method,'--deep')
    call('run','--data',data,'--method',method,'--output',out,'--device',DEVICE,'--seed',0,'--config',cfg,'--stop-after',1)
    assert json.loads((out/'status.json').read_text())['state']=='interrupted'
    call('run','--data',data,'--method',method,'--output',out,'--device',DEVICE,'--seed',0,'--config',cfg,'--resume')
    assert json.loads((out/'status.json').read_text())['state']=='completed'
    previous=None
    for _ in range(2):
        call('evaluate','--run',out,'--device',DEVICE);current=(out/'test_predictions.csv').read_bytes()
        if previous is not None:assert previous==current
        previous=current
    history=json.loads((out/'history.json').read_text());assert len(history)==2 and history[-1]['updates']==(8 if PROFILES[method]['condition']=='public' else 6)
    request=json.loads((out/'request.json').read_text());assert request['recipe']['optimizer_reset_every_epoch'] and request['recipe']['native_epochs']==100
    assert not request['test_task_labels_used_for_fit_or_selection'] and request['guidance_scope']=='train_bank'


def test_family_routes_actual_private_guide_condition_and_rejects_frozen_wrong_layout_and_CPU(tmp_path,prepared_tasks):
    from milplus.data import Bundle
    from milplus.registry import resolve_for_bundle
    for condition,data in prepared_tasks.items():assert resolve_for_bundle('MacNet',Bundle(data))['id']=='macnet_'+condition+'_source_cached_patient_reference'
    p=call('check','--data',prepared_tasks['public'],'--method','macnet_lscc81_source_cached_patient_reference',okay=False);assert 'guide conditions differ' in p.stderr
    call('run','--data',prepared_tasks['public'],'--method','MacNet','--output',tmp_path/'cpu_disallowed','--device','cpu',okay=False);assert not (tmp_path/'cpu_disallowed').exists()
    plain=tmp_path/'plain';call('demo','--output',plain)
    p=call('check','--data',plain,'--method','MacNet',okay=False);assert 'raw RGB' in p.stderr


def test_all321_prior_profile_bytes_and164_families_preserved_and_expanded327_152_catalog_is_reproducible():
    from milplus.registry import profiles,methods,inspect_method
    root=Path(__file__).resolve().parents[1];prior=json.loads((root/'tests/fixtures/macnet_prior321_profiles.json').read_text());current={p['id']:p for p in profiles() if p['id'] in {r['id'] for r in prior}|set(PROFILES)}
    assert len(prior)==321 and all(current[p['id']]==p for p in prior)
    assert len(current)==327 and len({p['family'] for p in current.values()})==152 and len(methods())==164
    assert {p['id'] for p in current.values() if p['family']=='macnet'}==set(PROFILES)
    assert inspect_method('MacNet')['id']=='macnet_public_source_cached_patient_reference'
    before=(root/'src/milplus/resources/profiles.json').read_bytes();p=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],cwd=root,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr;assert before==(root/'src/milplus/resources/profiles.json').read_bytes()

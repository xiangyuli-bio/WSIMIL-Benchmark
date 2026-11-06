import os,sys,subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize('script',[
    'validate_protomil_components.py',
    'validate_protomil_sae_recovery.py',
    'validate_protomil_author_MIL.py',
    'validate_protomil_pipeline.py'])
def test_original_full_components_literal_SAE_and_MIL_trainers_and_complete_lifecycle(tmp_path,script):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'oracle'
    command=[sys.executable,str(root/'scripts'/script),'--output',str(out)]
    if script in {'validate_protomil_author_MIL.py','validate_protomil_pipeline.py'}:command+=['--device','cpu']
    p=subprocess.run(command,capture_output=True,text=True,env=os.environ.copy());assert p.returncode==0,p.stdout+p.stderr
    assert (out/'validation.json').exists()


def test_native_protocol_preserves_original_dimensions_budgets_and_extra_information_track():
    from milplus.protomil_driver import PROFILES
    from milplus.protomil_spec import PROFILES as SPECS
    assert len(PROFILES)==4 and {r['family'] for r in SPECS}=={'protomil'}
    for name,cfg in PROFILES.items():
        assert (cfg['sae_epochs'],cfg['mil_epochs'])==((60,200) if '_paper_' in name else (200,1))
        assert cfg['intervened']==('_intervened_' in name)
    for row in SPECS:
        assert row['class_count']==2 and row['tasks']==['classification']
        assert 'TRAIN_region_annotated_RGB256_probes' in row['required_inputs']
        assert ('actual_SAE_bound_concept_review' in row['required_inputs'])==PROFILES[row['id']]['intervened']

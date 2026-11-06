"""The standard complete-profile validator must run the real raw PMIL pipeline."""
import os
from pathlib import Path
import subprocess
import sys

from milplus.io import read_json


def test_all_profiles_validator_supports_complete_raw_PMIL_without_FM_fallback(tmp_path):
    root=Path(__file__).resolve().parents[1];output=tmp_path/'profile_gate'
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'scripts/validate_all_profiles.py','--output',str(output),
        '--profile','pmil_prototypical_raw_patient_reference'],cwd=root,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr+result.stdout
    report=read_json(output/'validation.json');assert len(report['profiles'])==1
    entry=report['profiles'][0];assert entry['id']=='pmil_prototypical_raw_patient_reference' and entry['state']=='passed'
    run=output/entry['id'];request=read_json(run/'request.json');status=read_json(run/'status.json')
    assert request['backend']=='pmil_prototypical_complete_reference'
    assert request['config']['encoder_epochs']==1 and request['config']['bag_epochs']==1
    assert status['stages']==['raw_max_MIL','actual_512D_feature_extraction','local_AP','global_AP','dual_CE_PMIL']
    assert (run/'prototype_discovery/discovery.json').is_file() and (run/'test_predictions.csv').is_file()

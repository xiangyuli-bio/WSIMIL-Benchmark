import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from milplus.akdpmp_driver import PROFILES
from milplus.akdpmp_demo import create_demo


@pytest.mark.parametrize('profile',list(PROFILES))
def test_public_CLI_complete_source_recipe_and_two_fresh_selected_model_reloads(tmp_path,profile):
    data=create_demo(tmp_path/'data',preset=PROFILES[profile]['preset']);out=tmp_path/'run'
    cfg=tmp_path/'software_epochs.yaml';cfg.write_text(yaml.safe_dump({'epochs':2,'threads':2}))
    command=[sys.executable,'-m','milplus'];env=os.environ.copy()
    actual=subprocess.run(command+['run','--data',str(data),'--method',profile,'--output',str(out),'--config',str(cfg)],capture_output=True,text=True,env=env)
    assert actual.returncode==0,actual.stdout+actual.stderr
    previous=None
    for _ in range(2):
        result=subprocess.run(command+['evaluate','--run',str(out)],capture_output=True,text=True,env=env)
        assert result.returncode==0,result.stdout+result.stderr
        current=(out/'test_predictions.csv').read_bytes()
        if previous is not None:assert current==previous
        previous=current
    assert len(json.loads((out/'test_metrics.json').read_text())['matrix'])==3


def test_catalog_preserves_all_previous309_profiles_and_regenerates_new_scope(tmp_path):
    from milplus.registry import profiles,methods,inspect_method
    root=Path(__file__).resolve().parents[1];fixture=json.loads((root/'tests/fixtures/akdpmp_prior309_profiles.json').read_text())
    current={p['id']:p for p in profiles()};assert len(fixture)==309 and all(current[p['id']]==p for p in fixture)
    assert {p['id'] for p in profiles() if p['family']=='akdpmp'}==set(PROFILES)
    assert len(methods())==164 and inspect_method('AKD-PMP')['id']=='akdpmp_transmil_tcga_patient_reference'
    before=(root/'src/milplus/resources/profiles.json').read_bytes()
    result=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],cwd=root,capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,result.stderr
    assert (root/'src/milplus/resources/profiles.json').read_bytes()==before

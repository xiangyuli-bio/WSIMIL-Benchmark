import json
import subprocess
import sys

import pytest

from milplus.cli import main
from milplus.data import Bundle
from milplus.hdmil_demo import create_hdmil_demo
from milplus.models import build_bundle_model
from milplus.registry import inspect_method


def test_public_alias_and_four_actual_task_conditions():
    run=subprocess.run([sys.executable,'-m','milplus','inspect','--method','AHDMIL'],capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['id']=='ahdmil_brca_source_repaired'
    for name in ['camelyon16','nsclc','brca','rcc']:
        profile=inspect_method('ahdmil_'+name+'_source_repaired')
        assert profile['runnable'] and profile['external_source']=='ahdmil'
        assert profile['class_count']==(3 if name=='rcc' else 2)
        assert 'hdmil_paired_low_rgb' in profile['required_inputs']


def test_setup_fetches_both_actual_pretraining_and_joint_sources(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr('milplus.execution.fetch_source',lambda source,path:seen.append(source) or path)
    monkeypatch.setenv('MILPLUS_SOURCE_ROOT',str(tmp_path))
    main(['setup','--method','AHDMIL'])
    assert seen==['ahdmil','hdmil']


def test_model_refuses_missing_real_rgb_and_mismatched_physical_assets(tmp_path):
    import yaml
    data=create_hdmil_demo(tmp_path/'data',patients=20)
    p=data/'task.yaml';task=yaml.safe_load(p.read_text())
    fm=task['defaults']['fm'];task['encoders'][fm].pop('hdmil');p.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='paired RGB'):build_bundle_model('ahdmil_brca_source_repaired',Bundle(data))

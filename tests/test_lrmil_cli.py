import json
import subprocess
import sys

from milplus.data import Bundle
from milplus.lrmil_demo import create_lrmil_demo
from milplus.registry import inspect_method


def cli(*arguments):
    return subprocess.run([sys.executable,'-m','milplus',*map(str,arguments)],
                          capture_output=True,text=True,timeout=180)


def test_public_alias_resolves_complete_three_phase_source_profile():
    result=cli('inspect','--method','LRMIL')
    assert result.returncode==0,result.stdout+result.stderr
    spec=json.loads(result.stdout)
    assert spec['id']=='lrmil_source_repaired' and spec['runnable'] is True
    assert inspect_method('LRMIL')['external_source']=='lrmil'


def test_public_demo_binds_actual_pretrained_initialization_and_all_layers(tmp_path):
    result=cli('demo','--lrmil','--patients','27','--classes','3','--output',tmp_path/'data')
    assert result.returncode==0,result.stdout+result.stderr
    bundle=Bundle(tmp_path/'data')
    assert bundle.lrmil is not None and bundle.dim==768 and bundle.config['synthetic'] is True
    assert bundle.lrmil.schema['teacher_layers']==[0,1,10]


def test_public_prepare_creates_a_new_identical_input_contract(tmp_path):
    root,manifest=create_lrmil_demo(tmp_path/'data',patients=18)
    original=json.loads(manifest.read_text())
    schema=root/'schema.json';schema.write_text(json.dumps(original['schema']))
    result=cli('prepare-lrmil','--data',root,'--index',root/'lrmil_index.csv',
               '--schema',schema,'--output',root/'new.json')
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((root/'new.json').read_text())==original


def test_incompatible_input_modes_are_not_silently_combined(tmp_path):
    result=cli('demo','--lrmil','--kmgmamba','--output',tmp_path/'invalid')
    assert result.returncode!=0 and not (tmp_path/'invalid').exists()

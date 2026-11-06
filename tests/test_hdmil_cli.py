import json
import subprocess
import sys

from milplus.data import Bundle
from milplus.hdmil_demo import create_hdmil_demo
from milplus.registry import inspect_method


def cli(*args):
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,timeout=180)


def test_public_alias_and_all_four_task_recipes():
    result=cli('inspect','--method','HDMIL')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['id']=='hdmil_brca_source'
    for name in ['camelyon16','nsclc','brca','rcc']:
        spec=inspect_method('hdmil_'+name+'_source')
        assert spec['runnable'] and spec['external_source']=='hdmil'
        assert 'hdmil_paired_low_rgb' in spec['required_inputs']


def test_public_demo_and_preparation_create_same_bound_assets(tmp_path):
    result=cli('demo','--hdmil','--patients','20','--classes','2','--output',tmp_path/'data')
    assert result.returncode==0,result.stdout+result.stderr
    b=Bundle(tmp_path/'data');assert b.config['synthetic'] and b.hdmil is not None
    schema=tmp_path/'schema.json';schema.write_text(json.dumps(b.hdmil.schema))
    result=cli('prepare-hdmil','--data',b.root,'--index',b.root/'hdmil_index.csv','--schema',schema,'--output',b.root/'second.json')
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((b.root/'second.json').read_text())==b.hdmil.meta


def test_mixed_data_modes_are_rejected(tmp_path):
    result=cli('demo','--hdmil','--mgtrans','--output',tmp_path/'data')
    assert result.returncode!=0 and not (tmp_path/'data').exists()

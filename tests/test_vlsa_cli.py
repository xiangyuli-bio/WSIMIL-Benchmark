"""Public VLSA discovery, explicit input declaration and CLI validation."""
import json
import subprocess
import sys

import pytest

from milplus.registry import inspect_method
from milplus.vlsa_source import weights_path


def cli(*args):
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],
                          capture_output=True,text=True,timeout=180)


def test_public_vlsa_demo_check_and_source_discovery(tmp_path):
    output=tmp_path/'inputs'
    result=cli('demo','--vlsa','--vlsa-weights',weights_path(),'--patients','20','--output',output)
    assert result.returncode==0,result.stdout+result.stderr
    result=cli('check','--data',output,'--method','VLSA','--deep')
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report['patients']==20 and report['deep'] is True
    spec=inspect_method('VLSA')
    assert spec['id']=='vlsa_conch_source_repaired' and spec['tasks']==['survival']
    assert spec['status']=='implemented_unvalidated' and spec['paper_reproduction']=='not_run'


@pytest.mark.parametrize('extra',[[],['--slip'],['--survival'],['--raw-images']])
def test_public_vlsa_demo_rejects_absent_weights_or_mixed_tasks(tmp_path,extra):
    args=['demo','--vlsa','--output',tmp_path/'bad',*extra]
    if extra:args+=['--vlsa-weights',weights_path()]
    result=cli(*args)
    assert result.returncode!=0 and not (tmp_path/'bad').exists()

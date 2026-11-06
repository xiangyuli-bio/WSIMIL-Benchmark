"""CPU startup contract; preserve every original strict numerical test."""
import json
from types import SimpleNamespace

import pytest

from milplus import akdpmp_runtime as runtime


def args(**kwargs):
    return SimpleNamespace(**{'command':'run','device':'cpu','method':'AKD-PMP','config':None,**kwargs})


@pytest.mark.parametrize('method',['AKD-PMP','akdpmp_clam_skin_patient_reference','akdpmp_transmil_tcga_patient_reference'])
def test_only_canonical_AKD_CPU_uses_fixed_startup_and_original_precision(method):
    policy=runtime.desired_policy(args(method=method))
    assert policy=={'MKL_CBWR':'COMPATIBLE','MKL_DYNAMIC':'FALSE','OMP_DYNAMIC':'FALSE',
        'MKL_NUM_THREADS':'2','OMP_NUM_THREADS':'2',runtime.MARKER:'cnr_v1'}


@pytest.mark.parametrize('change',[{'method':'ConMIL'},{'method':'ABMIL'},{'device':'cuda:0'},{'command':'setup'},{'command':'methods'}])
def test_other_methods_commands_and_GPU_do_not_change_startup(change):
    assert runtime.desired_policy(args(**change)) is None


def test_config_threads_and_recorded_policy_agree(tmp_path,monkeypatch):
    config=tmp_path/'config.yaml';config.write_text('threads: 3\n')
    policy=runtime.desired_policy(args(config=str(config)))
    for k,v in policy.items():monkeypatch.setenv(k,v)
    recorded=runtime.numerical_policy(3)
    assert recorded['conditional_reproducibility'] and recorded['precision']=='original_float32'
    assert not runtime.numerical_policy(2)['conditional_reproducibility']
    config.write_text('threads: true\n')
    with pytest.raises(ValueError,match='positive integer'):runtime.desired_policy(args(config=str(config)))


@pytest.mark.parametrize('policy',[None,{}, {'conditional_reproducibility':False}])
def test_GPU_or_unpinned_API_run_is_not_reinterpreted_as_CNR_evaluation(tmp_path,policy):
    (tmp_path/'request.json').write_text(json.dumps({'backend':'akdpmp_complete_continual_reference','cpu_numerical_policy':policy,'threads':2}))
    assert runtime.desired_policy(args(command='evaluate',run=str(tmp_path))) is None


def test_CNR_evaluation_reuses_recorded_thread_policy(tmp_path):
    (tmp_path/'request.json').write_text(json.dumps({'backend':'akdpmp_complete_continual_reference',
        'cpu_numerical_policy':{'conditional_reproducibility':True},'threads':3}))
    assert runtime.desired_policy(args(command='evaluate',run=str(tmp_path)))['OMP_NUM_THREADS']=='3'


def test_existing_correct_startup_does_not_restart(monkeypatch):
    a=args()
    for k,v in runtime.desired_policy(a).items():monkeypatch.setenv(k,v)
    monkeypatch.setattr(runtime.os,'execvpe',lambda *a:pytest.fail('unexpected restart'))
    assert runtime.maybe_restart(a) is None


def test_real_CLI_execs_same_arguments_before_training(monkeypatch):
    captured={}
    monkeypatch.delenv('MKL_CBWR',raising=False)
    monkeypatch.setattr(runtime.sys,'argv',['milplus','run','--method','AKD-PMP','--data','with space'])
    class Executed(Exception):pass
    def execute(program,command,env):
        captured.update(program=program,command=command,env=env);raise Executed
    monkeypatch.setattr(runtime.os,'execvpe',execute)
    with pytest.raises(Executed):runtime.maybe_restart(args())
    assert captured['command']==[runtime.sys.executable,'-m','milplus','run','--method','AKD-PMP','--data','with space']
    assert captured['env']['MKL_CBWR']=='COMPATIBLE'


def test_embedded_CLI_preserves_callers_process_and_environment(monkeypatch):
    monkeypatch.delenv('MKL_CBWR',raising=False);observed={}
    def run(command,env):
        observed.update(command=command,env=env);return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runtime.subprocess,'run',run)
    assert runtime.maybe_restart(args(),['run','--method','AKD-PMP'])==7
    assert 'MKL_CBWR' not in runtime.os.environ
    assert observed['env']['MKL_CBWR']=='COMPATIBLE'

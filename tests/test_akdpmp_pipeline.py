import json
from pathlib import Path

import pytest

from milplus.akdpmp_demo import create_demo
from milplus.akdpmp_driver import train,evaluate,PROFILES
from milplus.akdpmp_data import AKDInputs
from milplus.icmil_coupled_training import load
from test_akdpmp_source import same


@pytest.mark.parametrize('profile',list(PROFILES))
def test_every_original_model_and_preset_complete_memory_teacher_resume_and_reload(tmp_path,profile,monkeypatch):
    preset=PROFILES[profile]['preset'];data=create_demo(tmp_path/'data',preset=preset);cfg={'epochs':2,'threads':2};observed=[]
    original=AKDInputs.features
    def audited(self,row):
        assert row['subset']!='test';observed.append((row['subset'],row['target']));return original(self,row)
    monkeypatch.setattr(AKDInputs,'features',audited)
    whole=train(data,profile,tmp_path/'whole',config=cfg,seed=83);baseline=load(whole/'last.pt')
    assert baseline['next_task']==3 and len(baseline['history'])==6
    from milplus.data import Bundle
    current_only_updates=2*len(Bundle(data).records('train'))
    assert baseline['updates']>current_only_updates # Actual old-memory optimizer updates execute.
    memory=baseline['memory'];samples=[r for group in memory['buffers'].values() for r in group] if 'buffers' in memory else memory['buffer']
    assert samples and all('row_positions' in s for s in samples)
    assert all(len(s['features'])==4 for s in samples) if preset=='tcga' else all(len(s['features'])==41 for s in samples)
    for stop in [1,2,3]:
        out=train(data,profile,tmp_path/('resumed'+str(stop)),config=cfg,seed=83,stop_after=stop)
        assert json.loads((out/'status.json').read_text())['state']=='interrupted'
        train(data,profile,out,config=cfg,seed=83,resume=True);actual=load(out/'last.pt')
        for key in ['model','modes','gradients','next_task','next_epoch','optimizer','teacher','teacher_gradients','early',
                    'memory','train_generator','val_generator','history','updates','rng']:
            same(baseline[key],actual[key])
        for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']))
    monkeypatch.setattr(AKDInputs,'features',original)
    a=evaluate(whole);p=(whole/'test_predictions.csv').read_bytes();b=evaluate(whole)
    assert a==b and p==(whole/'test_predictions.csv').read_bytes();assert [len(r) for r in a['matrix']]==[1,2,3]
    assert not json.loads((whole/'request.json').read_text())['test_used_for_training_or_selection']


def test_future_and_heldout_features_are_not_decoded_before_first_checkpoint(tmp_path,monkeypatch):
    data=create_demo(tmp_path/'data');native=AKDInputs.features;seen=[]
    def guarded(self,row):
        assert row['subset']!='test' and row['target']<2;seen.append(row['slide_id']);return native(self,row)
    monkeypatch.setattr(AKDInputs,'features',guarded)
    run=train(data,'akdpmp_transmil_tcga_patient_reference',tmp_path/'run',config={'epochs':2},stop_after=1)
    assert seen and json.loads((run/'status.json').read_text())['state']=='interrupted'
    with pytest.raises(ValueError,match='Complete all'):evaluate(run)


def test_short_native_pseudo_bags_fail_instead_of_changing_sampling_or_padding(tmp_path):
    data=create_demo(tmp_path/'data',patches=17)
    with pytest.raises(ValueError,match='too short'):train(data,'akdpmp_transmil_tcga_patient_reference',tmp_path/'run',config={'epochs':2})


def test_singleton_incremental_classes_are_real_tasks_and_complete_prediction_matrix(tmp_path):
    data=create_demo(tmp_path/'data',preset='skin',patches=17,task_sizes=[2,1,1,1,1]);profile='akdpmp_clam_skin_patient_reference'
    run=train(data,profile,tmp_path/'run',config={'epochs':[1]*5},seed=83);result=evaluate(run)
    assert len(result['matrix'])==5 and result['matrix'][-1][-1]['macro_auroc'] is None
    assert json.loads((run/'status.json').read_text())['completed_tasks']==5

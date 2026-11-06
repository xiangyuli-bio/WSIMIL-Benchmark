"""Complete endpoint labels, grouping safeguards and independent sigmoid metrics."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from milplus.cli import main
from milplus.data import Bundle
from milplus.metrics import classification_metrics
from milplus.multilabel import multilabel_metrics
from milplus.registry import check_compatibility, coverage, require_runnable, resolve_for_bundle
from milplus.smmile_demo import create_smmile_demo


def task_fixture(tmp_path):
    root=create_smmile_demo(tmp_path/'data','multilabel',dimension=4)
    return root,yaml.safe_load((root/'task.yaml').read_text())


@pytest.mark.parametrize('bad',['missing','unknown','nonbinary','float_text','missing_column','column_order_count','duplicate_column',
                                'no_provenance','empty_semantics','missing_endpoint_support','inconsistent_patient','global_leakage'])
def test_invalid_supervision_is_never_replaced_with_negatives(tmp_path,bad):
    root,task=task_fixture(tmp_path)
    frame=pd.read_csv(root/'slides.csv',dtype=str)
    if bad=='missing':frame.loc[0,'endpoint_0']=''
    if bad=='unknown':frame.loc[0,'endpoint_0']='not_tested'
    if bad=='nonbinary':frame.loc[0,'endpoint_0']='2'
    if bad=='float_text':frame.loc[0,'endpoint_0']='0.0'
    if bad=='missing_column':frame=frame.drop(columns='endpoint_0')
    if bad=='column_order_count':task['multilabel']['columns']=['endpoint_0']
    if bad=='duplicate_column':task['multilabel']['columns']=['endpoint_0']*3
    if bad=='no_provenance':task['multilabel']['label_provenance']=''
    if bad=='empty_semantics':task['multilabel']['empty_target_semantics']=''
    if bad=='missing_endpoint_support':frame['endpoint_0']='0'
    if bad=='inconsistent_patient':
        patient=frame.patient_id.value_counts().idxmax();indices=frame.index[frame.patient_id.eq(patient)]
        frame.loc[indices[0],'endpoint_0']=str(1-int(frame.loc[indices[0],'endpoint_0']))
    if bad=='global_leakage':
        split=pd.read_csv(root/'split.csv');a=split[split.subset.eq('train')].patient_id.iloc[0];b=split[split.subset.eq('test')].patient_id.iloc[0]
        frame['global_patient_id']=frame.patient_id
        frame.loc[frame.patient_id.eq(b),'global_patient_id']=a
    frame.to_csv(root/'slides.csv',index=False)
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError):Bundle(root)


def predictions():
    y=np.array([[1,1,0],[0,0,1],[1,0,1],[0,1,0]])
    p=np.array([[.8,.7,.2],[.1,.4,.9],[.6,.3,.8],[.2,.9,.1]])
    frame=pd.DataFrame({**{f'target_{i}':y[:,i] for i in range(3)},**{f'p_{i}':p[:,i] for i in range(3)}})
    frame['global_patient_id']=['a','b','c','d']
    return frame,y,p


def test_independent_probabilities_match_primary_metric_definitions():
    frame,y,p=predictions()
    assert not np.allclose(p.sum(1),1)
    result=multilabel_metrics(frame,3)
    assert result['macro_auc']==roc_auc_score(y,p,average='macro')
    assert result['micro_auc']==roc_auc_score(y,p,average='micro')
    assert result['macro_ap']==average_precision_score(y,p,average='macro')
    assert result['subset_accuracy']==1.
    assert result['log_loss']==pytest.approx(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
    exclusive=frame.copy();exclusive['target']=[0,1,0,1]
    with pytest.raises(ValueError,match='probabilities'):classification_metrics(exclusive,3)


def test_patient_aggregation_and_exact_half_threshold():
    frame,_y,_p=predictions()
    repeated=pd.concat([frame,frame],ignore_index=True)
    assert multilabel_metrics(repeated,3,'patient')=={**multilabel_metrics(frame,3,'patient'),'n':4}
    repeated.loc[0,'target_0']=0
    with pytest.raises(ValueError,match='consistent'):multilabel_metrics(repeated,3,'patient')
    frame[[f'p_{i}' for i in range(3)]]=.5
    assert multilabel_metrics(frame,3)['micro_f1']==0.


@pytest.mark.parametrize('bad',['nan','range','target','missing','unit'])
def test_metrics_reject_invalid_predictions(bad):
    frame,_y,_p=predictions()
    if bad=='nan':frame.loc[0,'p_0']=np.nan
    if bad=='range':frame.loc[0,'p_0']=1.1
    if bad=='target':
        frame['target_0']=frame.target_0.astype(float)
        frame.loc[0,'target_0']=.5
    if bad=='missing':frame=frame.drop(columns='p_1')
    with pytest.raises(ValueError):multilabel_metrics(frame,3,'bad_unit' if bad=='unit' else 'slide')


def test_multilabel_unsupported_methods_are_blocked_not_relabelled(tmp_path):
    root,_task=task_fixture(tmp_path);bundle=Bundle(root)
    check_compatibility(require_runnable('smmile_multilabel'),bundle)
    for name in ['abmil','mean_linear','smmile','smmile_binary']:
        with pytest.raises(ValueError):check_compatibility(require_runnable(name),bundle)
    assert coverage()['planned_records']==164
    assert resolve_for_bundle('SMMILe',bundle)['id']=='smmile_multilabel'
    assert resolve_for_bundle('smmile_multiclass',bundle)['id']=='smmile'


def test_cli_multilabel_demo_is_explicitly_synthetic(tmp_path,capsys):
    assert main(['demo','--smmile','multilabel','--output',str(tmp_path/'demo'),'--patients','30'])==0
    record=json.loads(capsys.readouterr().out)
    b=Bundle(record['created'])
    assert b.task_type=='multilabel' and b.config['synthetic'] is True
    assert (0,0,0) in set(b.rows.targets)
    config=copy.deepcopy(b.config)
    assert config['multilabel']['label_provenance'].startswith('synthetic')


def test_all_method_matrix_keeps_164_records_and_resolves_multilabel(tmp_path):
    from milplus.execution import matrix
    root,_task=task_fixture(tmp_path)
    records=matrix(root,['all'],['toy'],[2026],tmp_path/'matrix.json')
    assert len(records)==164
    ready=[r for r in records if r['state']=='ready']
    assert [r['method'] for r in ready]==['smmile_multilabel']
    assert ready[0]['family']=='smmile'


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
def test_generic_method_name_resolves_declared_task_route(tmp_path,variant):
    data=create_smmile_demo(tmp_path/'data',variant,dimension=4)
    b=Bundle(data)
    expected={'binary':'smmile_binary','multiclass':'smmile','multilabel':'smmile_multilabel'}[variant]
    assert resolve_for_bundle('SMMILe',b)['id']==expected


@pytest.mark.parametrize('unit',['training_unit','evaluation_unit'])
def test_invalid_multilabel_unit_is_rejected_at_registration(tmp_path,unit):
    root,task=task_fixture(tmp_path)
    task[unit]='unknown'
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='units'):Bundle(root)

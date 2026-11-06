"""Actual dataset+method CLI, full native widths and explicit FM-dimension port."""
import pytest
import yaml

from milplus.cli import dispatch,parser
from milplus.data import Bundle
from milplus.engine import evaluate
from milplus.io import digest_file,read_json
from milplus.registry import inspect_method


@pytest.mark.parametrize('dimension',[2048,1024])
def test_actual_unified_full_model_train_select_and_two_independent_reload(tmp_path,dimension):
    result=dispatch(parser().parse_args(['demo','--m4','--m4-dimension',str(dimension),
        '--patients','20','--output',str(tmp_path/'data')]))
    bundle=Bundle(result['created']);assert bundle.task_type=='multilabel' and bundle.dim==dimension and len(bundle.classes)==10
    assert inspect_method('M4')['id']=='m4_multilabel_source'
    cfg=tmp_path/'software.yaml';cfg.write_text(yaml.safe_dump({'epochs':2,'threads':1}))
    run=tmp_path/'run';dispatch(parser().parse_args(['run','--data',str(bundle.root),'--method','M4',
        '--config',str(cfg),'--output',str(run)]))
    assert read_json(run/'status.json')['state']=='completed' and not (run/'test_predictions.csv').exists()
    request=read_json(run/'request.json');settings=request['config']
    assert settings['lr']==.0001 and settings['weight_decay']==.00004 and settings['grad_clip'] is None
    assert settings['max_instances'] is None and settings['selection_metric']=='m4_val_mean_auc'
    source=read_json(run/'method_protocol.json')
    assert source['expert_width']==512 and source['experts']==5 and source['gate_slices']==4 and source['tower_hidden']==32
    assert source['completed_epochs']==2 and source['optimizer_updates']==2*len(bundle.records('train'))
    import torch
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    assert best['epoch']==max(range(2),key=lambda i:last['history'][i]['validation']['m4_val_mean_auc'])
    assert len(last['model']['_extra_state']['val_orders'])==2
    first=evaluate(run);sha=digest_file(run/'test_predictions.csv')
    assert first==evaluate(run) and digest_file(run/'test_predictions.csv')==sha
    assert first['macro_auc']==first['m4_test_mean_auc']


def test_original32epoch_default_and_no_first_zero_checkpoint_selection(tmp_path):
    from milplus.m4 import M4MIL
    from milplus.m4_demo import create_m4_demo
    bundle=Bundle(create_m4_demo(tmp_path/'data'))
    model=M4MIL(bundle)
    assert not model.selection_improved(0.,-float('inf'))
    assert model.selection_improved(.5,0.) and not model.selection_improved(.5,.5)
    assert yaml.safe_load((__import__('pathlib').Path(__file__).resolve().parents[1]/'configs/m4.yaml').read_text())['epochs']==32


@pytest.mark.parametrize('mode',['--convl','--survival','--miladapter','--center-embeddings'])
def test_other_modes_do_not_change_m4_input_supervision(tmp_path,mode):
    with pytest.raises(ValueError,match='cannot be mixed'):
        dispatch(parser().parse_args(['demo','--m4',mode,'--output',str(tmp_path/'bad')]))
    assert not (tmp_path/'bad').exists()


def test_all_method_matrix_keeps_full_scope_and_declared_slide_evaluation(tmp_path):
    from milplus.execution import matrix
    from milplus.m4_demo import create_m4_demo
    root=create_m4_demo(tmp_path/'data')
    records=matrix(root,['all'],['toy'],[2026],tmp_path/'matrix.json')
    assert len(records)==164
    record=next(row for row in records if row['family']=='m4')
    assert record['state']=='ready' and record['method']=='m4_multilabel_source'
    task=yaml.safe_load((root/'task.yaml').read_text());task['evaluation_unit']='patient'
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    records=matrix(root,['M4'],['toy'],[2026],tmp_path/'patient_matrix.json')
    assert records[0]['state']!='ready' and 'slide' in records[0]['reason']


def test_real_engine_interruption_resume_keeps_complete_source_state_and_predictions(tmp_path):
    import torch
    from milplus.engine import train
    from milplus.m4_demo import create_m4_demo
    from test_m4_source import assert_equal
    data=create_m4_demo(tmp_path/'data')
    cfg={'epochs':2,'threads':1}
    continuous=train(data,'M4',tmp_path/'continuous',seed=42,config=cfg)
    recovered=train(data,'M4',tmp_path/'recovered',seed=42,config=cfg,stop_after=1)
    assert read_json(recovered/'status.json')['state']=='interrupted'
    train(data,'M4',recovered,seed=42,config=cfg,resume=True)
    a=torch.load(continuous/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(recovered/'last.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizers','rng','history','epoch','best','bad']:
        assert_equal(a[key],b[key])
    assert evaluate(continuous)==evaluate(recovered)
    assert digest_file(continuous/'test_predictions.csv')==digest_file(recovered/'test_predictions.csv')

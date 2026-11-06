"""Public selection, prerequisite boundaries and actual source validation metrics."""
from types import SimpleNamespace

import pandas as pd
import pytest

from milplus.cli import parser
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.nciemil import NcIEMIL,source_loop
from milplus.nciemil_prepare import prepare
from milplus.registry import inspect_method,check_compatibility,profiles


def test_public_method_selects_full_pipeline_and_names_both_distinct_kv_conditions():
    selected=inspect_method('NcIEMIL')
    assert selected['id']=='nciemil_full_reference_released_kv'
    assert selected['external_source']=='nciemil'
    assert 'nciemil_tasktrained_selected_cache' in selected['required_inputs']
    assert {p['id'] for p in profiles() if p['family']=='nciemil'}=={
        'nciemil_full_reference_released_kv','nciemil_full_reference_independent_kv_repaired'}


def test_arbitrary_frozen_features_cannot_substitute_for_task_trained_swin_filter(tmp_path):
    bundle=Bundle(create_demo(tmp_path/'features',patients=20))
    with pytest.raises(ValueError,match='TRAIN-only Swin'):
        check_compatibility(inspect_method('NcIEMIL'),bundle)


def test_public_preparation_requires_explicit_missing_author_pretraining_parameters(tmp_path):
    args=parser().parse_args(['prepare-nciemil','--data','raw','--weights','swin.pth','--output','cache',
        '--pretrain-config','declared.yaml','--k','128'])
    assert args.k==128 and args.pretrain_config=='declared.yaml' and not args.resume
    with pytest.raises(ValueError,match='explicitly declare'):
        prepare('raw','weights',tmp_path/'cache',{'epochs':3},128)
    assert not (tmp_path/'cache').exists()


def test_literal_source_micro_auc_and_macro_f1_with_explicit_finite_zero_harmonic():
    owner=SimpleNamespace(bundle=SimpleNamespace(classes=['a','b','c','d']),loop=source_loop())
    logits=[[-1.,3.,-1.,-1.],[-1.,-1.,3.,-1.],[-1.,-1.,-1.,3.],[3.,-1.,-1.,-1.]]
    frame=pd.DataFrame(logits,columns=[f'nciemil_logit_{i}' for i in range(4)])
    frame['target']=[0,1,2,3]
    result=NcIEMIL.validation_metrics(owner,frame)
    assert result['nciemil_source_accuracy']==result['nciemil_source_macro_f1']==0
    assert result['nciemil_source_auc']>0 and result['nciemil_val_harmonic']==0
    for i in range(4):frame[f'nciemil_logit_{i}']=[3. if j==i else -1. for j in range(4)]
    result=NcIEMIL.validation_metrics(owner,frame)
    assert set(result.values())=={1.}

from pathlib import Path
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from milplus.engine import train,evaluate
from milplus.icmil_context_data import ContextInputs
from milplus.icmil_context_driver import environment,PROFILES
from milplus.icmil_context_prepare import prepare
from milplus.icmil_context_source import checkpoint
from milplus.io import read_json,write_json
from milplus.registry import inspect_method

from test_icmil_context_data import fixture


@pytest.mark.parametrize('profile',list(PROFILES))
def test_actual_complete_pretrained_context_fit_resume_and_two_fresh_reloads(tmp_path,profile):
    data=prepare(fixture(tmp_path/'data'),tmp_path/'prepared',10,'train_pca25_reference')
    common={'threads':2};continuous=train(data,profile,tmp_path/'continuous',config=common)
    interrupted=train(data,profile,tmp_path/'interrupted',config=common,stop_after=1)
    assert read_json(interrupted/'status.json')['state']=='interrupted'
    train(data,profile,interrupted,config=common,resume=True)
    a=pd.read_csv(continuous/'val_predictions.csv');b=pd.read_csv(interrupted/'val_predictions.csv')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    x=evaluate(continuous);first=(continuous/'test_predictions.csv').read_bytes()
    y=evaluate(continuous);assert first==(continuous/'test_predictions.csv').read_bytes() and x==y
    request=read_json(continuous/'request.json')
    assert request['downstream_optimizer_updates']==0
    assert request['model_seed']==PROFILES[profile]
    assert request['synthetic'] is True
    assert request['pretrained_weights_sha256']
    for p in continuous.glob('prediction_*/native_call.json'):
        call=read_json(p);assert not any('label' in key or 'target' in key for key in call)
        receipt=read_json(p.parent/'logits.receipt.json')
        assert receipt['parameters']==8257552 and receipt['pretraining_epochs']==8000
        assert receipt['query_labels_available_to_native_process'] is False
        assert receipt['downstream_optimizer_updates']==0


def test_original_algorithm_rejects_invented_downstream_epochs_and_alias_is_unambiguous(tmp_path):
    assert inspect_method('ICMIL In-context')['id']=='icmil_context_c5trd795_source'
    assert inspect_method('icmil_coupled')['runnable'] is False
    data=prepare(fixture(tmp_path/'data',16),tmp_path/'prepared',10,'native')
    with pytest.raises(ValueError,match='zero downstream'):train(data,'icmil_context',tmp_path/'run',config={'epochs':2})


def test_modified_support_cannot_be_used_for_evaluation(tmp_path):
    data=prepare(fixture(tmp_path/'data',16),tmp_path/'prepared',10,'native')
    run=train(data,'icmil_context',tmp_path/'run',config={'threads':2})
    with np.load(run/'support.npz') as f:x=f['features'].copy();y=f['targets'].copy()
    x[0,0,0]+=1;np.savez(run/'support.npz',features=x,targets=y)
    with pytest.raises(RuntimeError,match='native context'):evaluate(run)


def test_changed_native_runtime_cannot_replace_recorded_context_runtime(tmp_path):
    data=prepare(fixture(tmp_path/'data',16),tmp_path/'prepared',10,'native')
    run=train(data,'icmil_context',tmp_path/'run',config={'threads':2})
    request=read_json(run/'request.json');request['native_runtime']['numpy']='different-runtime'
    write_json(run/'request.json',request)
    with pytest.raises(ValueError,match='runtime changed'):evaluate(run)

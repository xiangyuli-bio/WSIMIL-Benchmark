import json,os
import pytest
from milplus.kmpmil_driver import train,evaluate,recipe
from milplus.kmpmil_data import KMPInputs
from milplus.kmpmil_demo import create_demo
from milplus.data import Bundle
from milplus.icmil_coupled_training import load
from test_kmpmil_data import WEIGHTS,SHA
DEVICE=os.environ.get('MILPLUS_KMPMIL_TEST_DEVICE','cpu')


def test_actual_source_hyperparameters_and_no_silent_mechanism_configuration(tmp_path):
    a=Bundle(create_demo(tmp_path/'data',WEIGHTS,SHA)).kmpmil;c=recipe(a,{})
    assert c['epochs']==[20]*4 and c['load_best_ckpt'] and c['bp_every_batch']==16
    assert c['adam_lr']==.005 and c['adam_eps']==1e-8 and c['adam_weight_decay']==.0005
    assert c['prompt_length']==24 and c['match_size']==5 and c['max_norm']==1
    assert c['es_patience']==[10]*4 and c['lrs_patience']==[3]*4
    assert c['pfc_reg_weight']==.1 and c['lambda_route']==.5 and c['route_margin']==.05 and c['route_temperature']==.07
    with pytest.raises(ValueError,match='Preserve original'):recipe(a,{'use_pfc':False})


def test_first_epoch_decodes_only_current_TRAIN_VAL_and_keeps_actual_incomplete_tail(tmp_path,monkeypatch):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);native=KMPInputs.features;observed=[]
    def guard(self,row):
        assert row['subset']!='test' and row['target']<3;observed.append(row['slide_id']);return native(self,row)
    monkeypatch.setattr(KMPInputs,'features',guard)
    out=train(data,tmp_path/'run',config={'epochs':3},device=DEVICE,stop_after=1);state=load(out/'last.pt')
    assert observed and state['next_task']==0 and state['next_epoch']==1 and state['updates']==2
    assert state['history'][0]['native_minibatch_sizes']==[16,11]
    assert json.loads((out/'status.json').read_text())['state']=='interrupted'
    assert not (out/'test_predictions.csv').exists()
    with pytest.raises(ValueError,match='Complete every'):evaluate(out,device=DEVICE)

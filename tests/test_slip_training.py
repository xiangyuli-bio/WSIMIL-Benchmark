"""Actual native text, complete source prompt SGD and strict continuation."""
import copy
import json

import pytest
import torch

from milplus.data import Bundle
from milplus.engine import evaluate,train
from milplus.io import read_json
from milplus.slip import SLIPMIL
from milplus.slip_demo import create_slip_demo


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load_model(data):
    bundle=Bundle(data)
    return SLIPMIL(bundle,'slip_'+bundle.slip.architecture+'_source_repaired')


@pytest.mark.parametrize('architecture',['clip_vit_b16','clip_rn50'])
def test_full_native_entry_final_selection_reload_and_bitwise_resume(tmp_path,architecture):
    data=create_slip_demo(tmp_path/'inputs',architecture=architecture)
    cfg={'epochs':3,'threads':1}
    full=train(data,'SLIP',tmp_path/'full',config=cfg,seed=2066)
    part=train(data,'SLIP',tmp_path/'resumed',config=cfg,seed=2066,stop_after=1)
    train(data,'SLIP',part,config=cfg,seed=2066,resume=True)
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(part/'last.pt',weights_only=False)
    assert set(a['model'])==set(b['model'])
    for key,value in a['model'].items():
        if torch.is_tensor(value):assert torch.equal(value,b['model'][key]),key
        else:assert value==b['model'][key],key
    assert evaluate(full)==evaluate(part)
    assert (full/'test_predictions.csv').read_bytes()==(part/'test_predictions.csv').read_bytes()
    before=(full/'test_predictions.csv').read_bytes();evaluate(full)
    assert before==(full/'test_predictions.csv').read_bytes()
    best=torch.load(full/'best.pt',weights_only=False)
    assert best['epoch']==2
    assert best['model']['_extra_state']['completed_epochs']==3
    assert a['model']['_extra_state']['optimizer_updates']==len(Bundle(data).records('train'))*3
    assert set(a['model'])=={'_extra_state','core.prompt_learner.ctx',
        'core.prompt_learner.token_prefix','core.prompt_learner.token_suffix',
        'core.prompt_learner.tokenized_prompts','core.tokenized_prompts','core.tissue_weights'}
    protocol=read_json(full/'method_protocol.json')
    assert protocol['text_layers']==12 and protocol['completed_epochs']==3
    assert len(a['optimizers'])==1 and a['optimizers'][0]['state']=={}


@pytest.mark.parametrize('field,value',[
    ('classes',['wrong_class','class_1']),('synthetic',False),('weights_sha256','0'*64),
    ('architecture','arbitrary_fm'),('feature_preprocessing','different pipeline'),
    ('slide_classnames',[['only one class']]),('tissue_classnames',[]),('templates',['no placeholder']),
    ('n_shots','1'),('few_shot_provenance',''),
])
def test_reject_wrong_native_prompt_feature_and_few_shot_binding(tmp_path,field,value):
    data=create_slip_demo(tmp_path/'inputs');p=data/'slip_inputs.json';meta=read_json(p);meta[field]=value;p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):Bundle(data)


@pytest.mark.parametrize('damage',['prompt_nan','prompt_dtype','fixed_tissue','extra_foundation','wrong_train_history'])
def test_strict_checkpoint_rejects_corrupted_prompts_buffers_or_training_scope(tmp_path,damage):
    m=load_model(create_slip_demo(tmp_path/'inputs'));state=copy.deepcopy(m.state_dict())
    if damage=='prompt_nan':state['core.prompt_learner.ctx'].flatten()[0]=float('nan')
    elif damage=='prompt_dtype':state['core.prompt_learner.ctx']=state['core.prompt_learner.ctx'].double()
    elif damage=='fixed_tissue':state['core.tissue_weights'].flatten()[0]+=1
    elif damage=='extra_foundation':state['core.native.logit_scale']=torch.tensor(1.)
    else:state['_extra_state'].update(completed_epochs=1,epoch_index=0,train_orders=[['test_slide']],optimizer_updates=1)
    with pytest.raises(ValueError):m.load_state_dict(state)


def test_optimizer_contract_and_before_optimizer_device_conversion_guard(tmp_path):
    m=load_model(create_slip_demo(tmp_path/'inputs'));optimizer=m.make_optimizer(.002,0.)
    m.validate_optimizer_state();optimizer.param_groups[0]['lr']=.004
    with pytest.raises(ValueError,match='SGD configuration'):m.validate_optimizer_state()
    with pytest.raises(ValueError,match='before constructing'):m.to('cpu')


def test_future_parameter_conversion_preserves_actual_shared_frozen_text(tmp_path):
    m=load_model(create_slip_demo(tmp_path/'inputs'))
    previous=torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True);m.to('cpu')
    finally:torch.__future__.set_overwrite_module_params_on_conversion(previous)
    assert m.core.text_encoder.text_projection is m.core.native.text_projection
    assert m.core.text_encoder.positional_embedding is m.core.native.positional_embedding
    assert m.core.text_encoder.transformer is m.core.native.transformer
    optimizer=m.make_optimizer(.002,0.);m.start_epoch(0)
    m.epoch(m.bundle,optimizer,'cpu');m.validate_optimizer_state()
    assert m.optimizer_updates==len(m.bundle.records('train'))


def test_prompt_mutation_after_binding_and_wrong_profile_are_rejected(tmp_path):
    data=create_slip_demo(tmp_path/'inputs');m=load_model(data)
    with pytest.raises(ValueError):SLIPMIL(m.bundle,'slip_clip_rn50_source_repaired')
    p=data/'slip_inputs.json';meta=read_json(p);meta['templates']=['changed {}'];p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='changed after loading'):m.start_epoch(0)

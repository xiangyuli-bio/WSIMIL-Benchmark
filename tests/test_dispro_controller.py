"""Controller-only doubles plus exact full-stage/resume invariants.

Tiny modules here exercise orchestration without downloading assets. They are
never production adapters or native-network validation. Separate native probes
execute the actual 12-layer BioBERT and all331 pathway encoders.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from milplus import dispro as implementation
from milplus.data import Bundle
from milplus.dispro import DisPro, concordance
from milplus.engine import restore_rng, rng_state, seed_everything
from test_dispro_data import fixture


class TinyUni(nn.Module):
    def __init__(self,dim,foundation):
        super().__init__()
        self.model=nn.Module()
        self.model.modality_encoder=nn.Linear(dim,4)
        self.model.project=nn.Linear(4,4)
        self.model.ctx=nn.Parameter(torch.randn(4))
        self.foundation=foundation
        self.drop=nn.Dropout(.2)

    def forward(self,value,c):
        if isinstance(value,dict):value=torch.cat(list(value.values()),-1)
        if value.ndim==3:value=value.mean(1)
        value=self.model.modality_encoder(value)
        return self.model.project(self.drop(value))+self.model.ctx+(1-c.reshape(-1,1))*.1


class TinyMulti(nn.Module):
    def __init__(self,wsi,omics,foundation):
        super().__init__()
        self.uni_model_wsi=wsi.requires_grad_(False)
        self.uni_model_omics=omics.requires_grad_(False)
        self.wsi_adapter=wsi.model.modality_encoder.requires_grad_(True)
        self.omics_adapter=omics.model.modality_encoder.requires_grad_(True)
        self.Transformer=foundation
        self.holder_path=nn.Parameter(torch.randn(1,2,4))
        self.holder_omic=nn.Parameter(torch.randn(1,2,4))
        self.cls_token=nn.Parameter(torch.randn(1,1,4))
        self.wsi_attn_head=nn.Linear(4,1);self.omics_attn_head=nn.Linear(4,1)
        self.linear_head=nn.Sequential(nn.Linear(4,4),nn.Dropout(.2))
        self.classifier=nn.Linear(4,4)

    def forward(self,x_WSI,x_Omics,censor,label):
        w=(self.wsi_adapter(x_WSI.mean(1)) if x_WSI is not None else 0)+self.holder_path.mean(1)
        o=(self.omics_adapter(torch.cat(list(x_Omics.values()),-1)) if x_Omics is not None else 0)+self.holder_omic.mean(1)
        result={}
        def outputs(logits,suffix=''):
            h=logits.sigmoid();result['hazards'+suffix]=h;result['S'+suffix]=(1-h).cumprod(1)
        outputs(self.classifier(self.linear_head(w+o+self.cls_token[:,0])))
        if x_WSI is None:outputs(self.uni_model_wsi.model.project(w),'_wsi')
        if x_Omics is None:outputs(self.uni_model_omics.model.project(o),'_omics')
        return result


@pytest.fixture
def controller(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    foundation=nn.Linear(4,4).requires_grad_(False)
    monkeypatch.setattr(implementation,'backbone',lambda:foundation)
    monkeypatch.setattr(implementation,'build_unimodal',lambda modal,dimension,sizes:
        TinyUni(dimension if modal=='WSI' else sum(sizes),foundation))
    monkeypatch.setattr(implementation,'build_multimodal',lambda w,o,sizes,dimension,repaired_mask:
        TinyMulti(w,o,foundation))
    bundle=Bundle(fixture(tmp_path/'data'))
    def build():
        model=DisPro(bundle,stage_epochs=[2,2,2])
        cfg={'epochs':6,'grad_clip':None,'max_instances':None,'weight_decay':1e-5,
             'lr':2e-4,'selection_metric':'dispro_c_index'}
        model.configure_training(cfg);model.prepare(bundle,2026,'cpu')
        optimizers=model.make_optimizers(cfg['lr'],cfg['weight_decay'])
        return model,optimizers,cfg
    return bundle,build


def advance(bundle,model,optimizers,cfg,epoch):
    model.start_epoch(epoch);model.epoch(bundle,optimizers,'cpu')
    frame=model.predict_subset(bundle,'val','cpu',cfg)
    metrics=model.validation_metrics(frame);model.after_validation(metrics)
    return metrics


def same(left,right):
    if torch.is_tensor(left):return torch.equal(left,right)
    if isinstance(left,np.ndarray):return np.array_equal(left,right)
    if isinstance(left,dict):return set(left)==set(right) and all(same(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right


@pytest.mark.parametrize('boundary',[1,2,3,4,5])
def test_all_three_stages_bitwise_resume_at_and_around_both_transitions(controller,boundary):
    bundle,build=controller;seed_everything(2026);full,opts,cfg=build()
    checkpoint=None
    for epoch in range(6):
        advance(bundle,full,opts,cfg,epoch)
        if epoch+1==boundary:
            checkpoint={'model':copy.deepcopy(full.state_dict()),
                'optimizers':copy.deepcopy([o.state_dict() for o in opts]),'rng':copy.deepcopy(rng_state())}
    expected=copy.deepcopy(full.state_dict());expected_opts=copy.deepcopy([o.state_dict() for o in opts])
    restored,newopts,cfg=build();restored.load_state_dict(checkpoint['model'])
    for o,s in zip(newopts,checkpoint['optimizers']):o.load_state_dict(s)
    restore_rng(checkpoint['rng'])
    for epoch in range(boundary,6):advance(bundle,restored,newopts,cfg,epoch)
    assert same(expected,restored.state_dict()) and same(expected_opts,[o.state_dict() for o in newopts])
    assert full.stage_progress==[2,2,2] and full.bound_teachers==full.teacher_best
    assert full.teacher_states==[None,None] and full.optimizers[0].state==full.optimizers[1].state=={}
    assert all(s.optimizer is o and s.last_epoch==2 for s,o in zip(full.schedulers,opts))


def test_stage_selection_does_not_write_final_best_until_fusion_and_reloads_both_teachers(controller):
    bundle,build=controller;model,opts,cfg=build()
    assert not model.selection_improved(.9,-float('inf'))
    advance(bundle,model,opts,cfg,0)
    selected=copy.deepcopy(model.teacher_states[0]);meta=copy.deepcopy(model.teacher_best[0])
    model.start_epoch(1);model.epoch(bundle,opts,'cpu')
    # A lower VAL score cannot overwrite the genuine first-stage teacher.
    model.after_validation({'dispro_c_index':-1.})
    assert same(selected,model.teacher_states[0]) and model.teacher_best[0]==meta
    model.start_epoch(2)
    assert same(selected,implementation._portable_state(model.wsi))
    for epoch in [2,3]:advance(bundle,model,opts,cfg,epoch)
    gene=copy.deepcopy(model.teacher_states[1]);model.start_epoch(4)
    assert same(gene,implementation._portable_state(model.omics))
    assert same(selected,implementation._portable_state(model.wsi))
    assert model.selection_improved(.9,.8) and not model.selection_improved(.8,.8)


def test_actual_parameters_holders_CLS_and_bound_schedules_not_detached_data(controller):
    bundle,build=controller;model,opts,cfg=build()
    for epoch in range(4):advance(bundle,model,opts,cfg,epoch)
    model.start_epoch(4)
    names={g['name'] for g in opts[2].param_groups}
    assert {'cls_token','holder_path','holder_omic'}<=names
    for key in ['cls_token','holder_path','holder_omic']:
        assert any(getattr(model.network,key) is p for g in opts[2].param_groups for p in g['params'])
    before=[getattr(model.network,k).detach().clone() for k in ['cls_token','holder_path','holder_omic']]
    records=model.epoch(bundle,opts,'cpu')
    assert all(not torch.equal(a,getattr(model.network,k)) for a,k in zip(before,['cls_token','holder_path','holder_omic']))
    assert any('missing_wsi_unipro_distillation' in r for r in records)
    assert any('missing_omics_unipro_distillation' in r for r in records)
    assert all(s.optimizer is o for s,o in zip(model.schedulers,opts))
    for i,phase in enumerate(model.phase_parameters):
        assert len(phase)==len(set(phase)) and len(phase)>0


def test_stage_populations_are_exact_observed_TRAIN_and_VAL_patients_no_zero_imputation(controller):
    bundle,build=controller;model,opts,cfg=build()
    for epoch in range(6):advance(bundle,model,opts,cfg,epoch)
    for order in model.train_orders:
        allowed={g[0]['global_patient_id'] for g in model._groups('train',order['phase'])}
        assert set(order['patients'])==allowed and len(order['patients'])==len(allowed)
    assert all(meta['val_patients']==[g[0]['global_patient_id'] for g in model._groups('val',i)]
               for i,meta in enumerate(model.teacher_best))
    assert model.optimizer_steps==[len(model._groups('train',i))*2 for i in range(3)]
    frame=model.predict_subset(bundle,'test','cpu',cfg)
    assert len(frame)==len(model._groups('test',2)) and (frame.wsi_observed|frame.omics_observed).all()


@pytest.mark.parametrize('kind',['order','counter','phase','schedule','teacher','weights','asset'])
def test_corrupted_resume_refuses_wrong_patients_counters_teachers_or_foundation(controller,kind):
    bundle,build=controller;model,opts,cfg=build();advance(bundle,model,opts,cfg,0)
    state=copy.deepcopy(model.state_dict());extra=state['_extra_state']
    if kind=='order':extra['train_orders'][0]['patients'][0]='foreign_TEST_patient'
    elif kind=='counter':extra['optimizer_steps'][0]+=1
    elif kind=='phase':extra['phase']=2
    elif kind=='schedule':extra['schedulers'][0]['T_max']+=1
    elif kind=='teacher':extra['teacher_best'][0]['val_patients']=['foreign_TEST_patient']
    elif kind=='weights':next(iter(extra['teacher_states'][0].values())).add_(1)
    else:extra['biobert_sha256']={}
    restored,_,_=build()
    with pytest.raises(ValueError,match='DisPro'):restored.load_state_dict(state)


def test_constant_foundation_not_duplicated_but_all_learned_weights_strict(controller):
    bundle,build=controller;model,opts,cfg=build();advance(bundle,model,opts,cfg,0)
    state=model.state_dict()
    assert not any('foundation.' in n or '.Transformer.' in n for n in state)
    assert any('modality_encoder.weight' in n for n in state)
    broken=copy.deepcopy(state);broken.pop(next(n for n in broken if 'modality_encoder.weight' in n))
    restored,_,_=build()
    with pytest.raises(ValueError,match='strict complete'):restored.load_state_dict(broken)
    restored.load_state_dict(state)
    assert same(restored.state_dict(),state)


@pytest.mark.parametrize('stage_epochs',[(0,50,50),(30,50),(30,True,50),[30,50,-1]])
def test_no_stage_can_be_omitted_or_replaced_by_zero_budget(controller,stage_epochs):
    bundle,_=controller
    with pytest.raises(ValueError,match='all three stages'):DisPro(bundle,stage_epochs=stage_epochs)


def test_patient_TEST_cannot_bypass_pretraining_and_final_stage(controller):
    bundle,build=controller;model,opts,cfg=build()
    with pytest.raises(ValueError,match='three-stage'):model.predict_subset(bundle,'test','cpu',cfg)
    with pytest.raises(ValueError,match='skip/repeat'):model.start_epoch(2)
    altered={**cfg,'epochs':2}
    with pytest.raises(ValueError,match='all declared'):model.configure_training(altered)


def test_concordance_time_ties_risk_tolerance_and_no_comparable_pairs():
    assert concordance([1,1],[1,0],[2,1])==1.
    assert concordance([1,2],[1,1],[1.,1.+1e-9])==.5
    assert concordance([1,2],[1,1],[1.,2.])==0.
    assert concordance([1,2],[1,0],[2.,1.])==1.
    with pytest.raises(ValueError,match='no comparable'):concordance([1,1],[1,1],[2,1])
    with pytest.raises(ValueError,match='no comparable'):concordance([1,2],[0,0],[2,1])
    with pytest.raises(ValueError,match='finite'):concordance([1,2],[1,0],[np.nan,1])


@pytest.mark.parametrize('boundary',[2,4])
def test_actual_unified_engine_class_training_interruption_reload_and_TEST_export(controller,tmp_path,boundary):
    from milplus.engine import evaluate,train
    from milplus.io import read_json
    bundle,_=controller
    cfg={'model':{'stage_epochs':[2,2,2]},'threads':1}
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    train(bundle.root,'DisPro',full,config=cfg)
    train(bundle.root,'DisPro',resumed,config=cfg,stop_after=boundary)
    assert read_json(resumed/'status.json')['state']=='interrupted'
    assert not (resumed/'best.pt').exists(),'No unimodal stage may produce final MultiPro best.pt'
    train(bundle.root,'DisPro',resumed,config=cfg,resume=True)
    assert read_json(resumed/'status.json')['completed_epochs']==6
    assert evaluate(full)==evaluate(resumed)
    assert (full/'test_predictions.csv').read_bytes()==(resumed/'test_predictions.csv').read_bytes()
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    assert same(a,b)
    selected=torch.load(resumed/'best.pt',weights_only=False)
    assert 'optimizers' not in selected and 'rng' not in selected
    assert 'resume_from_last.pt' in selected['checkpoint_role']
    assert len(b['optimizers'])==3 and 'rng' in b
    protocol=read_json(resumed/'survival_protocol.json')
    assert [s['completed_epochs'] for s in protocol['stages']]==[2,2,2]
    assert protocol['bound_teachers']==protocol['teacher_best']


@pytest.mark.parametrize('corruption',['LR','missing_active_moments','wrong_decay','step','holder_steps'])
def test_optimizer_resume_rejects_missing_moments_bad_LR_ownership_and_counters(controller,corruption):
    bundle,build=controller;model,opts,cfg=build()
    for epoch in range(5):advance(bundle,model,opts,cfg,epoch)
    model.validate_optimizer_state()
    if corruption=='LR':opts[2].param_groups[0]['lr']+=.01
    elif corruption=='missing_active_moments':opts[2].state.clear()
    elif corruption=='wrong_decay':opts[2].param_groups[0]['weight_decay']=.3
    elif corruption=='step':next(iter(opts[2].state.values()))['step'].fill_(99999)
    else:opts[2].state[model.network.holder_omic]['step'].sub_(1)
    with pytest.raises(ValueError,match='DisPro'):model.validate_optimizer_state()


def test_stage_optimizers_follow_actual_parameters_after_device_conversion(controller):
    bundle,_=controller
    model=DisPro(bundle,stage_epochs=[2,2,2])
    model.network.direct_teacher_ctx=model.wsi.model.ctx
    previous=torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        # CPU simulation of parameter-replacing device conversions. Production
        # builds optimizers AFTER device transfer, so cached Python Param lists
        # must follow the modules, regardless of PyTorch conversion policy.
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        model.to('cpu')
    finally:torch.__future__.set_overwrite_module_params_on_conversion(previous)
    actual={id(p) for p in model.network.parameters()}
    assert all(id(p) in actual for ps in model._phase_params for p in ps)
    assert all(id(p) in actual for _,ps,_ in model._fusion_groups for p in ps)
    assert model.network.direct_teacher_ctx is model.wsi.model.ctx
    cfg={'epochs':6,'grad_clip':None,'max_instances':None,'weight_decay':1e-5,
         'lr':2e-4,'selection_metric':'dispro_c_index'}
    model.configure_training(cfg);model.prepare(bundle,2026,'cpu')
    opts=model.make_optimizers(2e-4,1e-5)
    for epoch in range(6):advance(bundle,model,opts,cfg,epoch)
    model.validate_optimizer_state()
    assert model.stage_progress==[2,2,2]

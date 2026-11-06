"""Independent released PathOmics trainer, phase transition and leakage checks."""
import ast
import copy
import random
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.pathomics import MODEL_FILE, PROFILES, PathOmicsSurvival, fine_trainable_names, pathomics_epoch
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, **options):
    dimension=options.pop('dimension',16)
    data = create_demo(tmp_path/'data',patients=20,dimension=dimension,survival=True,omics=True)
    cfg = yaml.safe_load((data/'task.yaml').read_text())
    cfg['omics']['normalization'] = 'none'
    (data/'task.yaml').write_text(yaml.safe_dump(cfg))
    bundle = Bundle(data)
    model = PathOmicsSurvival(bundle.dim,bundle.omics,**{
        'hidden':16,'pretrain_epochs':2,'accumulation_steps':2,**options})
    model.configure_training({'epochs':4,'max_instances':None})
    model.prepare(bundle,41,'cpu')
    return model,bundle


def independent_scope(dim=1024, hidden=256):
    root,sha=verified_source('pathomics')
    scope = {**vars(F),'torch':torch,'nn':nn,'F':F,'np':np,'random':random,'Optional':Optional,
             'Tensor':torch.Tensor,'Module':nn.Module,'Parameter':nn.Parameter,
             'NonDynamicallyQuantizableLinear':nn.modules.linear.NonDynamicallyQuantizableLinear,
             'xavier_uniform_':nn.init.xavier_uniform_,'xavier_normal_':nn.init.xavier_normal_,
             'constant_':nn.init.constant_,'tqdm':lambda x:x,
             'concordance_index_censored':lambda *a,**k:(.5,)}
    for path in ['PathOmics/model_and_training_utils/PathOmics_model_utils.py',MODEL_FILE,
                 'PathOmics/model_and_training_utils/Customized_Loss.py',
                 'PathOmics/model_and_training_utils/train_and_eval_utils_core.py']:
        code=pinned_text(root,sha,path)
        if dim!=1024 or hidden!=256:
            code=code.replace('[1024, 256, 256]',f'[{dim}, {hidden}, {hidden}]')
            code=code.replace("'small': [256, 256]",f"'small': [{hidden}, {hidden}]")
            code=code.replace('embed_dim=256',f'embed_dim={hidden}')
            code=code.replace('d_model=256, nhead=8, dim_feedforward=512',f'd_model={hidden}, nhead=8, dim_feedforward={2*hidden}')
            code=code.replace('nn.Linear(256*2, size[2])',f'nn.Linear({hidden}*2, size[2])')
        definitions=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent pinned source
    # Execute exactly the released zip loop, without calling the adapter's freeze selector.
    main=ast.parse(pinned_text(root,sha,'PathOmics/bash_main.py'))
    loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and
              'zip(best_pretrain_model.named_parameters(), best_pretrain_model.children())' in ast.unparse(n.iter))
    scope['freeze_loop']=compile(ast.Module(body=[loop],type_ignores=[]),'source_freeze','exec')
    return scope


def freeze_reference(reference, policy, scope):
    reference.requires_grad_(True)
    if policy=='source_zip':
        scope.update(best_pretrain_model=reference,names=[],params=SimpleNamespace(parameter_reset=False))
        exec(scope['freeze_loop'],scope)  # noqa: S102 -- exact author freeze loop
    else:
        prefixes=('path_attention_head.','omic_attention_head.','path_rho.','omic_rho.','mm.','classifier.','path_proj.','omic_proj.')
        for n,p in reference.named_parameters():
            p.requires_grad_(n=='logit_scale' or n.startswith(prefixes))


def assert_state_equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_state_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state_equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('profile',sorted(PROFILES))
@pytest.mark.parametrize('policy',['source_zip','heads'])
def test_author_two_stage_trainer_eight_adam_updates(tmp_path,profile,policy):
    model,bundle=fixture(tmp_path,profile=profile,freeze_policy=policy)
    scope=independent_scope(model.dim,model.hidden)
    reference=scope['PathOmics_Surv'](**model.network_args)
    reference.load_state_dict(model.network.state_dict())
    opt=model.make_optimizers(1e-4,0.)
    pre_ref=torch.optim.Adam(reference.parameters(),lr=1e-4,weight_decay=0.)
    groups=units(bundle,'train')[:4]
    loader=[]
    for rows in groups:
        x,molecular=model.inputs(bundle,rows,'cpu')
        loader.append((x.unsqueeze(0),[molecular[idx].unsqueeze(0) for idx in model.group_indices],
                       torch.tensor([1-rows[0]['event']]),rows[0]['time'],
                       model.time_label([rows[0]['time']]),[rows[0]['global_patient_id']]))
    histories=[[],[]]
    def record(_optimizer, histories, network):
        histories.append({n:(p.detach().clone(),None if p.grad is None else p.grad.clone())
                          for n,p in network.named_parameters()})
    pre_ref.register_step_post_hook(lambda o,a,k:record(o,histories[0],reference))
    opt[0].register_step_post_hook(lambda o,a,k:record(o,histories[1],model.network))
    saved=None
    for epoch in range(4):
        if epoch==2:
            reference.load_state_dict(saved)
            freeze_reference(reference,policy,scope)
            fine_ref=torch.optim.Adam([p for p in reference.parameters() if p.requires_grad],lr=5e-5,weight_decay=0.)
            fine_ref.register_step_post_hook(lambda o,a,k:record(o,histories[0],reference))
            opt[1].register_step_post_hook(lambda o,a,k:record(o,histories[1],model.network))
        model.train();model.start_epoch(epoch)
        initial=rng_state()
        stage='pretrain' if epoch<2 else 'Finetune'
        criterion=nn.MSELoss() if epoch<2 else scope['NLLSurvLoss'](alpha=0.)
        scope['train_loop'](reference,loader,pre_ref if epoch<2 else fine_ref,criterion,None,
                            torch.device('cpu'),epoch,gc=2,model_mode=stage)
        restore_rng(initial)
        pathomics_epoch(model,bundle,groups,opt,'cpu',{'grad_clip':None})
        assert_state_equal(histories[0],histories[1])
        if epoch==0:saved=copy.deepcopy(reference.state_dict())
        if epoch<2:model.after_validation({'alignment_loss':1.+epoch})
    assert len(histories[0])==8
    assert model.stage_updates==[4,4]
    assert model.pretrain_best_epoch==0
    assert {n for n,p in reference.named_parameters() if p.requires_grad}==model.fine_names


@pytest.mark.parametrize('profile',sorted(PROFILES))
@pytest.mark.parametrize('stop',[1,2,3])
def test_exact_resume_before_at_and_after_transition(tmp_path,profile,stop):
    data=create_demo(tmp_path/'data',patients=20,survival=True,omics=True)
    cfg={'epochs':4,'threads':1,'model':{'hidden':16,'pretrain_epochs':2,'accumulation_steps':5}}
    train(data,profile,tmp_path/'full',config=cfg)
    train(data,profile,tmp_path/'resume',config=cfg,stop_after=stop)
    train(data,profile,tmp_path/'resume',config=cfg,resume=True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/last.pt',weights_only=False)
    for key in ['model','optimizers','history','best']:assert_state_equal(a[key],b[key])
    evaluate(tmp_path/'full');evaluate(tmp_path/'resume')
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'full/test_predictions.csv'),pd.read_csv(tmp_path/'resume/test_predictions.csv'))
    assert set(a['history'][0]['validation'])=={'alignment_loss','stage'}
    assert a['model']['_extra_state']['stage_updates']==[6,6]


@pytest.mark.parametrize('omic_bag',['SNN','Attention','SNN_Attention'])
@pytest.mark.parametrize('use_gap',[True,False])
def test_native_author_forward_and_alignment_gradients(tmp_path,omic_bag,use_gap):
    model,_=fixture(tmp_path,dimension=1024,hidden=256,omic_bag=omic_bag,use_gap=use_gap)
    # Full native hidden width, arbitrary FM width equals adapter bundle dimension.
    scope=independent_scope(model.dim,256)
    ref=scope['PathOmics_Surv'](**model.network_args)
    ref.load_state_dict(model.network.state_dict())
    x=torch.randn(20,model.dim);y=torch.randn(len(model.columns))
    model.train();ref.train()
    initial=rng_state()
    a,b=ref(x,[y[g] for g in model.group_indices],[],mode='pretrain')
    loss=F.mse_loss(a,b);loss.backward()
    restore_rng(initial)
    actual=model.alignment(x,y);actual.backward()
    torch.testing.assert_close(actual,loss,rtol=0,atol=0)
    for name,p in model.network.named_parameters():
        expected=dict(ref.named_parameters())[name]
        if expected.grad is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected.grad,rtol=0,atol=0)
    assert model.network.classifier.weight.grad is None
    assert model.network.path_proj.weight.grad.abs().sum()>0
    assert model.network.omic_proj.weight.grad.abs().sum()>0
    assert model.network.path_transformer.layers[0].linear1.weight.grad.abs().sum()>0


def test_frozen_pretrain_snapshot_and_heads_repair(tmp_path):
    model,_=fixture(tmp_path,freeze_policy='heads')
    model.start_epoch(0);model.after_validation({'alignment_loss':1.})
    saved=copy.deepcopy(model.pretrain_best)
    with torch.no_grad():
        for p in model.parameters():p.add_(.1)
    model.start_epoch(1);model.after_validation({'alignment_loss':2.})
    assert_state_equal(saved,model.pretrain_best)
    model.start_epoch(2)
    assert_state_equal(saved,model.network.state_dict())
    assert not model.network.path_transformer.layers[0].linear1.weight.requires_grad
    assert model.network.classifier.weight.requires_grad
    assert len(fine_trainable_names(model.network,'source_zip'))>len(model.fine_names)


def test_pretrain_tail_sum_and_finetune_tail_mean(tmp_path):
    model,bundle=fixture(tmp_path,accumulation_steps=4)
    groups=units(bundle,'train')[:3]
    opt=model.make_optimizers(1e-4,0.)
    for epoch in [0,2]:
        if epoch==2:model.after_validation({'alignment_loss':1.})
        model.train();model.start_epoch(epoch)
        reference=copy.deepcopy(model)
        opt_ref=reference.make_optimizers(1e-4,0.)[0 if epoch==0 else 1]
        initial=rng_state()
        for rows in groups:
            x,y=reference.inputs(bundle,rows,'cpu',train=True)
            loss=reference.alignment(x,y) if epoch==0 else reference.objective(reference(x,y),rows[0]['time'],rows[0]['event'])/3
            loss.backward()
        opt_ref.step()
        restore_rng(initial)
        pathomics_epoch(model,bundle,groups,opt,'cpu',{})
        assert_state_equal(model.network.state_dict(),reference.network.state_dict())
        assert all(p.grad is None for p in model.parameters())
    assert model.stage_updates==[1,1]


def test_inference_rng_and_single_branch_independence(tmp_path):
    model,bundle=fixture(tmp_path,profile='pathomics_image_source')
    model.eval();x,y=model.inputs(bundle,units(bundle,'val')[0],'cpu')
    state=rng_state();actual=model(x,y)
    # No survival labels or times supplied to the model, and molecular values cannot alter image-only output.
    changed=model(x,y+1000)
    torch.testing.assert_close(actual['hazard_logits'],changed['hazard_logits'],rtol=0,atol=0)
    assert torch.equal(state['torch'],rng_state()['torch']) and state['python']==rng_state()['python']
    model.fusion='omics';model.network.fusion='omics'
    a=model(x,y);b=model(x+1000,y)
    torch.testing.assert_close(a['hazard_logits'],b['hazard_logits'],rtol=0,atol=0)


@pytest.mark.parametrize('options',[{'hidden':15},{'pretrain_epochs':0},{'omic_bag':'bad'},
                                  {'freeze_policy':'bad'},{'profile':'pathomics_bilinear'},
                                  {'proj_ratio':0},{'finetune_lr':0}])
def test_invalid_method_contract(tmp_path,options):
    with pytest.raises(ValueError):fixture(tmp_path,**options)


def test_short_bags_and_stage_budget_rejected(tmp_path):
    model,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='per gene group'):
        model(torch.randn(1,model.dim),torch.randn(len(model.columns)))
    with pytest.raises(ValueError,match='BOTH'):
        model.configure_training({'epochs':2,'max_instances':None})
    with pytest.raises(ValueError,match='all patches'):
        model.configure_training({'epochs':4,'max_instances':2})
    with pytest.raises(RuntimeError,match='validated'):
        model.start_epoch(2)


def test_pretrain_ignores_outcomes_and_validation_does_not_call_survival_head(tmp_path):
    model,bundle=fixture(tmp_path)
    model.train();model.start_epoch(0)
    groups=units(bundle,'train')[:2]
    # The alignment stage must run even if outcome keys are absent from its rows.
    for rows in groups:
        for row in rows:
            row.pop('time');row.pop('event')
    optimizer=model.make_optimizers(1e-4,0.)
    handle=model.network.classifier.register_forward_hook(lambda *args:pytest.fail('survival head used in alignment stage'))
    pathomics_epoch(model,bundle,groups,optimizer,'cpu',{})
    result=model.alignment_validation(bundle,'cpu')
    handle.remove()
    assert result['alignment_loss']>=0 and result['stage']=='pretrain'


def test_train_only_statistics_and_intervals(tmp_path):
    model,bundle=fixture(tmp_path)
    bundle.omics.normalization='train_standard'
    model.normalization='train_standard'
    # Use the real table with train-only normalization; change every held-out covariate and outcome.
    model.prepare(bundle,12,'cpu')
    mean=model.omics_mean.clone();scale=model.omics_scale.clone();edges=model.bin_edges.clone()
    held=bundle.rows.subset!='train'
    bundle.rows.loc[held,'time']=1e9
    bundle.rows.loc[held,'event']=1
    held_ids=sorted(set(bundle.rows.loc[held,'global_patient_id']))
    bundle.omics.values.loc[held_ids,:]+=1e6
    model.prepare(bundle,12,'cpu')
    torch.testing.assert_close(mean,model.omics_mean,rtol=0,atol=0)
    torch.testing.assert_close(scale,model.omics_scale,rtol=0,atol=0)
    torch.testing.assert_close(edges,model.bin_edges,rtol=0,atol=0)


def test_omics_shared_mha_and_source_freeze_are_visible(tmp_path):
    model,_=fixture(tmp_path,omic_bag='SNN_Attention')
    assert all(m is model.network.MHA for m in model.network.mutiheadattention_networks)
    protocol=model.protocol()
    assert protocol['fine_tune_freeze_policy']=='source_zip'
    assert 'path_transformer.layers.1.linear1.weight' in protocol['fine_tune_trainable']
    assert protocol['source_bilinear'].startswith('not admitted')


def test_missing_molecular_signatures_rejected(tmp_path):
    _,bundle=fixture(tmp_path)
    with pytest.raises(ValueError,match='gene groups'):PathOmicsSurvival(bundle.dim,None)
    bundle.omics.group_indices=[]
    with pytest.raises(ValueError,match='gene groups'):PathOmicsSurvival(bundle.dim,bundle.omics)

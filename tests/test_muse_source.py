"""Complete89M MUSE, all600 descriptions and literal source mechanism checks."""
import copy
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from milplus.muse_source import (COMMIT, bank_assets, build_original,
                                 diagnostic_rng_port, checked_source)

WEIGHTS=Path('/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth')


@pytest.fixture(scope='module')
def full_source():
    torch.set_num_threads(8);torch.manual_seed(2026)
    model=build_original(WEIGHTS,'tcga_brca',dimension=512)
    return model


@pytest.mark.parametrize('cohort',['tcga_brca','tcga_nsclc','camelyon_all'])
def test_every_complete_original300_description_bank(cohort):
    assets=bank_assets(cohort)
    assert len(assets['descriptions'])==2 and all(len(x)==300 for x in assets['descriptions'])
    assert all(len(text)>50 for rows in assets['descriptions'] for text in rows)
    assert len(assets['class_prompts'])==2 and len(assets['sha256'])==2


def test_actual_full12layer8expert_native_model_and_source_initialization(full_source):
    model=full_source
    assert checked_source()[1]==COMMIT
    assert sum(p.numel() for p in model.parameters())==88985107
    assert len(model.text_encoder.transformer.resblocks)==12
    assert model.prompt_learner.ctx.shape==(16,768)
    assert model.adapter[0].weight.shape==(512,512)
    assert model.num_heads==8 and model.moe.top_k==2 and len(model.moe.experts)==8
    assert model.text_base_0.shape==model.text_base_1.shape==(300,512)
    assert model.text_base.shape==(600,512)
    original=torch.load(WEIGHTS,map_location='cpu',weights_only=True,mmap=True)
    assert torch.equal(model.text_encoder.text_projection,original['text.text_projection'])
    assert torch.equal(model.text_encoder.positional_embedding,original['text.positional_embedding'])
    assert not torch.equal(model.text_encoder.transformer.resblocks[0].mlp.c_fc.weight,
                           original['text.transformer.resblocks.0.mlp.c_fc.weight'])


def test_twenty_separate_complete_view_updates_and_actual_sparse_Adam_counts(full_source):
    model=copy.deepcopy(full_source);model.train();model.is_training=True
    torch.manual_seed(2026);feature=torch.randn(257,512);label=torch.tensor([0])
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001,weight_decay=.00001)
    counts={name:0 for name,_ in model.named_parameters()}
    before=model.prompt_learner.ctx.detach().clone()
    model.retrieval_k(feature,label);cache=model.k_vectors
    assert cache.shape==(20,512) and len(set(model.top_k_indices.tolist()))==20
    for i in range(20):
        probabilities,prediction,loss=model(feature,label,cache[i:i+1])
        assert probabilities.shape==(1,2) and torch.isfinite(loss)
        loss.backward()
        for name,parameter in model.named_parameters():counts[name]+=int(parameter.grad is not None)
        optimizer.step();optimizer.zero_grad()
    assert not torch.equal(before,model.prompt_learner.ctx)
    assert counts['prompt_learner.ctx']==20
    for name,parameter in model.named_parameters():
        if counts[name]:assert int(optimizer.state[parameter]['step'])==counts[name]
        else:assert parameter not in optimizer.state
    assert counts['q_proj.weight']==counts['agg.0.weight']==0


def test_released_least_cosine_retrieval_over_all300_entries(full_source):
    model=copy.deepcopy(full_source);model.eval();torch.manual_seed(17)
    x=torch.randn(257,512);label=torch.tensor([1]);state=torch.get_rng_state()
    with torch.no_grad():model.retrieval_k(x,label)
    actual=model.top_k_indices.clone();torch.set_rng_state(state)
    with torch.no_grad():
        feature=model.adapter(x)
        prompts=model.text_encoder(model.prompt_learner())
        query=model.aggregator_by_text(feature,prompts,agg_type='img',if_topk=False,
                                       no_aggregator=False,is_moe=True,is_shared=False)
        score=torch.nn.functional.normalize(query,dim=1) @ torch.nn.functional.normalize(model.text_base_1,dim=1).T
        expected=torch.topk(score,20,dim=1,largest=False).indices.squeeze(0)
    assert torch.equal(actual,expected)


def test_private_diagnostic_exact651_draw_shapes_and_Python_state(full_source):
    model=copy.deepcopy(full_source);model.eval();model.is_training=False
    torch.manual_seed(41);feature=torch.randn(257,512)
    scope=model.moe_visualization.__func__.__globals__;real_torch,real_os=scope['torch'],scope['os']
    draws=[];exports=[]
    class Proxy:
        def __getattr__(self,name):return getattr(real_torch,name)
        def load(self,*args,**kwargs):return feature.clone()
        def save(self,value,path,*args,**kwargs):exports.append(str(path))
        def randn_like(self,value,**kwargs):
            draws.append(list(value.shape));return real_torch.randn_like(value,**kwargs)
    start_torch,start_python=torch.get_rng_state(),random.getstate()
    scope['torch']=Proxy();scope['os']=SimpleNamespace(path=os.path,listdir=lambda path:[str(i)+'.pt' for i in range(1000)])
    try:
        with torch.no_grad():model.moe_visualization(0)
        end_torch,end_python=torch.get_rng_state(),random.getstate()
    finally:scope['torch'],scope['os']=real_torch,real_os
    torch.set_rng_state(start_torch);random.setstate(start_python);diagnostic_rng_port(model)
    assert torch.equal(end_torch,torch.get_rng_state()) and end_python==random.getstate()
    assert draws==[[2,8]]+[[8]]*600+[[1,2,8]]*50 and len(exports)==1


def test_eval_router_noise_is_preserved_and_seed_replay_is_exact(full_source):
    model=copy.deepcopy(full_source);model.eval();model.is_training=False
    torch.manual_seed(29);feature=torch.randn(257,512);label=torch.tensor([0]);rng=torch.get_rng_state()
    with torch.no_grad():first=model(feature,label)[0];different=model(feature,label)[0]
    torch.set_rng_state(rng)
    with torch.no_grad():replayed=model(feature,label)[0]
    assert torch.equal(first,replayed) and not torch.equal(first,different)

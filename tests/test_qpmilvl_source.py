"""Actual full CONCH and original update, not a toy text encoder."""
import copy
import os
from pathlib import Path
import json

import numpy as np
import pytest
import torch
from milplus import qpmilvl_source as source
from milplus.upstream import verified_source,pinned_text

WEIGHTS=Path(os.environ.get('MILPLUS_QPMILVL_WEIGHTS',Path.home()/'.cache/milplus/qpmilvl/conch.pth'))
SHA=os.environ.get('MILPLUS_QPMILVL_SHA256','40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c')
DEVICE=os.environ.get('MILPLUS_QPMILVL_TEST_DEVICE','cpu')


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.fixture(scope='module')
def base():
    torch.set_num_threads(2);torch.manual_seed(83)
    return source.load_base(WEIGHTS,SHA,DEVICE)


def test_actual_full_original_CoCa_constructor_and_pretrained_text_identity(base):
    assert len(base.text.transformer.resblocks)==len(base.visual.trunk.blocks)==12
    assert base.text_decoder is not None and type(base.text.ln_final) is torch.nn.LayerNorm
    assert base.text.width==768 and base.text.heads==12 and base.text.context_length==128
    assert sum(p.numel() for p in base.text.parameters())==110129664
    assert not any(p.requires_grad for p in base.parameters())
    d=source.definitions();tokens=d.tokenizer.tokenize(d.tokenizer.get_tokenizer(),['lung adenocarcinoma.','breast invasive ductal carcinoma.'])[:,:-1].to(DEVICE)
    with torch.no_grad():
        native=base.text(tokens)[0]
        actual=d.model.CONCHPromptEncoder(base)(base.text.token_embedding(tokens),tokens)
    assert torch.equal(native,actual)


def test_all_original_clinical_ensembles_and_literal_templates_are_preserved():
    root,sha=verified_source('qpmilvl');text=json.loads(pinned_text(root,sha,'class_ensemble/class_ensemble.json'))['0']
    assert len(text['templates'])==22
    cumulative=[]
    for task,classes in text['classnames'].items():
        for names in classes.values():cumulative.extend([t.replace('CLASSNAME',name) for name in names for t in text['templates']])
        actual=source.ensemble_through(task)
        assert actual['ensemble_classes']==cumulative
        assert sum(actual['count'])==len(cumulative)
    assert len(cumulative)==660


@pytest.mark.parametrize('pooling',['max','mean'])
def test_literal16bag_objective_every_gradient_Adam_moment_and_zero_parameter_exclusion(base,pooling):
    torch.manual_seed(83);cfg=source.config();cfg.update(task_num=1,pooling=pooling)
    pools=source.parameter_pools(cfg,base);model=source.model(cfg,base,pools,source.ensemble_through('tcga_lung'),{k:[] for k in cfg['dataset_names']}).eval()
    clone=copy.deepcopy(model)
    def optimizer(m):return torch.optim.Adam(m.parameters(),lr=cfg['adam_lr'],eps=cfg['adam_eps'],weight_decay=cfg['adam_weight_decay'])
    a=optimizer(model);b=optimizer(clone)
    bags=[torch.randn(1,17+i%3,512,device=DEVICE) for i in range(16)];labels=[torch.tensor([[i%2]],device=DEVICE) for i in range(16)]
    (logits,indices,features),logged=source.original_update(model,a,cfg,[x.clone() for x in bags],labels)
    expected,losses,chosen,expected_features=clone([x.clone() for x in bags])
    b.zero_grad();ce=torch.nn.functional.cross_entropy(expected,torch.cat(labels).squeeze(-1))
    total=ce+cfg['lambda']*losses['matching_loss']+cfg['beta']*losses['class_sim_loss'];total.backward()
    torch.nn.utils.clip_grad_norm_(clone.parameters(),cfg['max_norm'])
    for p in clone.parameters():
        if p.grad is not None and torch.all(p.grad==0):p.grad=None
    b.step()
    same(logits,expected);same(indices,chosen);same(features,expected_features)
    same(model.state_dict(),clone.state_dict());same(a.state_dict(),b.state_dict())
    for (_,p),(_,q) in zip(model.named_parameters(),clone.named_parameters()):same(p.grad,q.grad)
    assert logged['train/total loss']==float(total.detach())
    selected=set(indices[0].tolist());assert len(selected)==5
    assert all(torch.equal(indices[0],row) for row in indices)
    for i in range(20):
        assert (pools[0][i].grad is not None)==(i in selected)
        assert (pools[1][i].grad is not None)==(i in selected)
    assert all(p.grad is None for p in base.parameters())


def test_second_task_penalty_is_actual_last_epoch_frequency_and_old_vectors_frozen(base):
    cfg=source.config();cfg['task_num']=1;pools=source.parameter_pools(cfg,base)
    source.model(cfg,base,pools,source.ensemble_through('tcga_lung'),{k:[] for k in cfg['dataset_names']})
    pools[2][0].grad=torch.ones_like(pools[2][0]);cfg['task_num']=2
    counts=torch.arange(1,21,dtype=torch.int);frequency={k:[] for k in cfg['dataset_names']};frequency['tcga_lung']=[torch.ones(20,dtype=torch.int),counts]
    model=source.model(cfg,base,pools,source.ensemble_through('tcga_brca'),frequency).eval()
    assert torch.equal(model.penalty_table,(counts/counts.sum()).to(DEVICE))
    assert not pools[2][0].requires_grad and pools[2][0].grad is None and pools[2][1].requires_grad
    x=[torch.randn(1,19,512,device=DEVICE) for _ in range(16)]
    _,loss=model._query_prototype_pool(x,16,False);_,evaluation_loss=model._query_prototype_pool(x,16,True)
    assert torch.isfinite(loss) and evaluation_loss is None


def test_wrong_actual_checkpoint_hash_is_rejected_without_random_fallback():
    with pytest.raises(ValueError,match='hash mismatch'):source.load_base(WEIGHTS,'0'*64,DEVICE)


def test_actual_timm_and_torch_runtime_are_explicit_and_wrong_timm_is_rejected(monkeypatch):
    import timm
    actual=source.runtime_identity()
    assert actual['timm']=='1.0.11' and actual['torch_actual']==torch.__version__ and actual['precision']=='original_float32'
    monkeypatch.setattr(timm,'__version__','0.9.12')
    with pytest.raises(ValueError,match='isolated timm1.0.11'):source.runtime_identity()

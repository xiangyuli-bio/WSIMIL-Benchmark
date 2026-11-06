"""Full native released network and the actual label-disambiguation machinery."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus.iibmil_source import (COMMIT,author_components,build_original,
                                  checked_source,component_protocol,source_config)


def literal_module(path,name):
    """Execute original complete file, without the adapter's AST selection."""
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('condition',['luad_gm','tcga_nsclc','tcga_rcc'])
def test_native1280_full500instance_forward_loss_and_every_gradient_equal_literal_source(condition,monkeypatch):
    torch.set_num_threads(2);root,sha=checked_source();assert sha==COMMIT
    _,nested,_=author_components()
    # Resolve only the original annotation dependency. The network file itself
    # executes literally, with every full original class and parameter intact.
    package=ModuleType('util');misc=ModuleType('util.misc');misc.NestedTensor=nested
    monkeypatch.setitem(sys.modules,'util',package);monkeypatch.setitem(sys.modules,'util.misc',misc)
    original=literal_module(root/'main/models/iibmil.py','_iibmil_literal_network')
    torch.manual_seed(8601);actual=build_original(condition)
    cfg=source_config(condition)['MODEL_IIBMIL']
    torch.manual_seed(8601);reference=original.IIBMIL(cfg['NUM_CLASSES'],1280,cfg['HIDDEN_DIM'],cfg['DEPTH'],5)
    assert actual.state_dict().keys()==reference.state_dict().keys()
    for name,value in actual.state_dict().items():assert torch.equal(value,reference.state_dict()[name]),name
    assert actual.patch_encoder.depth==cfg['DEPTH']
    assert actual.query_embed.weight.shape==(5,cfg['HIDDEN_DIM'])
    assert len(actual.wsi_aggregator.decoder.layers)==2
    assert len(actual.patch_encoder._modules['depth_0'])==20
    # Exact source500-token bag cap, including a real padding mask. Attention
    # ignores that mask in the release; adapter does not silently change it.
    x=torch.randn(2,500,1280,requires_grad=True);y=x.detach().clone().requires_grad_(True)
    mask=torch.ones(2,500,dtype=torch.bool);mask[1,437:]=False
    torch.manual_seed(8602);a=actual(nested(x,mask));state=torch.get_rng_state()
    torch.manual_seed(8602);b=reference(nested(y,mask));assert torch.equal(state,torch.get_rng_state())
    assert len(a)==4 and a[0].shape==(2,500,cfg['NUM_CLASSES']) and a[3].shape==(2,cfg['NUM_CLASSES'])
    for left,right in zip(a,b):assert torch.equal(left,right)
    labels=torch.tensor([0,1]);pseudo=torch.full_like(a[0],1/cfg['NUM_CLASSES'])
    loss_a=F.cross_entropy(a[3],labels)-(a[0].log_softmax(-1)*pseudo).sum(-1).mean()
    loss_b=F.cross_entropy(b[3],labels)-(b[0].log_softmax(-1)*pseudo).sum(-1).mean()
    loss_a.backward();loss_b.backward();assert torch.equal(x.grad,y.grad)
    for (name,p),(other,q) in zip(actual.named_parameters(),reference.named_parameters()):
        assert name==other and ((p.grad is None)==(q.grad is None))
        if p.grad is not None:assert torch.equal(p.grad,q.grad),name
    assert actual.patch_encoder.mlp_head[1].weight.grad is not None
    assert actual.wsi_classifier.weight.grad is not None
    assert actual.patch_encoder.patch_to_embedding.weight.grad is None  # source-unused layer retained


def test_complete_partial_loss_confidence_EMA_and_epoch_schedule_match_literal_source():
    root,_=checked_source();_,_,confidence=author_components()
    literal=literal_module(root/'main/util/loss.py','_iibmil_literal_confidence')
    initial=torch.tensor([[1-1e-6,5e-7,5e-7],[5e-7,1-1e-6,5e-7],[5e-7,5e-7,1-1e-6]])
    actual=confidence(initial.clone(),{'a':0,'b':1,'c':2})
    original=literal.PartialLoss(initial.clone(),{'a':0,'b':1,'c':2})
    logits=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True)
    index=torch.tensor([2,0]);other=logits.detach().clone().requires_grad_(True)
    a=actual(logits,index);b=original(other,index)
    assert torch.equal(a,b);a.backward();b.backward();assert torch.equal(logits.grad,other.grad)
    # Exercise full epoch-dependent schedule, including beyond the source's
    # hardcoded200 denominator; do not silently rescale it to YAML300epochs.
    for epoch in [0,10,100,200,299]:
        actual.set_conf_ema_m(epoch);original.set_conf_ema_m(epoch)
        assert actual.conf_ema_m==original.conf_ema_m==epoch/200*(.8-.95)+.95
        scores=torch.tensor([[.1,.2,.7],[.6,.3,.1]])
        actual.confidence_update(scores,index,torch.ones(2,3))
        original.confidence_update(scores,index,torch.ones(2,3))
        assert torch.equal(actual.confidence,original.confidence)
    assert not torch.equal(actual.confidence,initial)
    assert torch.equal(actual.confidence[1],initial[1])  # no unrelated-instance update


def test_original_momentum_prototypes_respect_valid_instances_and_class_matched_bags():
    torch.set_num_threads(2);model=build_original('tcga_rcc')
    # Independently reconstruct the literal source top10percent and sequential
    # momentum update, with a valid source NumPy WSI index (its .size is an int).
    scores=torch.randn(3,20,3);features=torch.randn(3,20,256)
    mask=torch.ones(3,20,dtype=torch.bool);mask[2,15:]=False
    matched=np.array([[0],[2]],dtype=np.int64);proto=model.prototypes.clone();c=2
    chosen=scores[matched.squeeze(-1)][mask[matched.squeeze(-1)]]
    selected_features=features[matched.squeeze(-1)][mask[matched.squeeze(-1)]]
    top=torch.topk(chosen.softmax(1)[:,c],len(chosen)//10).indices
    momentum=1-(1-.99)*3/matched.size
    for value in selected_features[top]:proto[c]=proto[c]*momentum+(1-momentum)*value
    model.update_prototypes(c,matched,scores,features,mask,.99)
    assert torch.equal(model.prototypes,proto) and torch.count_nonzero(proto[:2])==0


def test_FM_width_port_retains_every_native_transformer_decoder_and_query_dimension():
    model=build_original('tcga_nsclc',feature_dimension=1024)
    assert model.patch_encoder.patch_dim==1024 and model.patch_encoder.z_dim==256
    for depth in range(3):
        assert model.patch_encoder._modules['res_'+str(depth)].weight.shape==(256,1024)
    assert model.query_embed.weight.shape==(5,256)
    assert model.wsi_aggregator.decoder.layers[0].linear1.weight.shape==(512,256)
    protocol=component_protocol('tcga_nsclc',1024)
    assert protocol['training_epoch_in_released_tree']=='absent'
    assert protocol['runnable_profile_registered'] is False and protocol['formal_admission'] is False


def test_no_silent_unknown_config_or_feature_width():
    for value in ['unknown','tcga_brca',None]:
        with pytest.raises(ValueError):source_config(value)
    for width in [0,-1,True,1.5]:
        with pytest.raises(ValueError):build_original('tcga_rcc',width)

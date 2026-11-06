"""Full native GMMamba graph against literal source and independent operators."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import numpy as np
import pytest
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

from milplus.gmmamba_source import (COMMIT,GMMambaNetwork,checked_source,dependency,
                                   validated_groups)


def literal(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def coordinates(n=257):
    return torch.tensor([[float(i%10)*1000+float(i//10),float(i//10)%7] for i in range(n)])


@pytest.mark.parametrize('profile,dimension,classes',[
    ('gmmamba_mambamil_bimamba_source',512,2),
    ('gmmamba_mambamil_bimamba_source',512,3),
    ('gmmamba_fm_projection_repaired',1024,2)])
def test_full257patch_10group_512width_source_logits_every_gradient_and_RNG(profile,dimension,classes,monkeypatch):
    torch.set_num_threads(2);root,sha=checked_source();assert sha==COMMIT
    bimamba,identity=dependency()
    parent=ModuleType('mamba');parent.__path__=[]
    child=ModuleType('mamba.mamba_ssm');child.BiMamba=bimamba
    def forbidden(*args,**kwargs):raise AssertionError('Unused source imports must not become a plain-Mamba substitution')
    child.Mamba=child.SRMamba=forbidden
    monkeypatch.setitem(sys.modules,'mamba',parent);monkeypatch.setitem(sys.modules,'mamba.mamba_ssm',child)
    source_imm=literal(root/'models/IMM.py','_gmmamba_literal_imm')
    source_css=literal(root/'models/CSS.py','_gmmamba_literal_css')
    source_cls=literal(root/'models/classmodel.py','_gmmamba_literal_classifier')
    torch.manual_seed(8603);actual=GMMambaNetwork(dimension,classes,profile)
    torch.manual_seed(8603);reference=nn.ModuleDict({
        'imm':source_imm.IMM_block(dimension,classes,.25,'gelu',layer=1,mask_ratio=.3),
        'css':source_css.CSS_block(512,classes,n_iter=1),
        'classifier':source_cls.ClassMultiMILTransformer(SimpleNamespace(embed_dim=512,n_classes=classes,num_layers=1))})
    mapping=lambda name:name.replace('imm.source.','imm.')
    current={mapping(name):value for name,value in actual.state_dict().items()}
    assert current.keys()==reference.state_dict().keys()
    for name,value in reference.state_dict().items():assert torch.equal(current[name],value),name
    imm=actual.imm.source if hasattr(actual.imm,'source') else actual.imm
    for block in [imm.BiMamba1[1],imm.BiMamba2[1]]:
        assert (block.d_model,block.d_inner,block.d_state,block.d_conv,block.dt_rank)==(512,1024,16,4,32)
    assert len(actual.classifier.GroupsSuper_Layer)==1 and actual.css.stoken_refine.num_heads==8
    assert identity['source']=='mambamil' and identity['backend']['name']=='torch'
    x=torch.randn(257,dimension,requires_grad=True);y=x.detach().clone().requires_grad_(True);coords=coordinates()
    torch.manual_seed(8604);output=actual(x,coords);after=torch.get_rng_state()
    labels=KMeans(n_clusters=10,random_state=0,n_init='auto').fit(coords.numpy()).labels_
    torch.manual_seed(8604);groups=[]
    for index in range(10):
        features=y[np.argwhere(labels==index).squeeze()].unsqueeze(0)
        if profile=='gmmamba_fm_projection_repaired':features=reference['imm']._fc1(features.float())
        groups.append(reference['imm'](features,first=True))
    group_tensor=torch.stack(groups,dim=1).view(1,-1,1,512)
    expected=reference['classifier'](reference['css'].stoken_forward(group_tensor))
    assert torch.equal(after,torch.get_rng_state())
    for name in ['logits','Y_prob','Y_hat']:assert torch.equal(output[name],expected[name]),name
    target=torch.tensor([1]);a=F.cross_entropy(output['logits'],target);b=F.cross_entropy(expected['logits'],target)
    assert torch.equal(a,b);a.backward();b.backward();assert torch.equal(x.grad,y.grad)
    ref=dict(reference.named_parameters())
    for name,value in actual.named_parameters():
        other=ref[mapping(name)];assert (value.grad is None)==(other.grad is None),name
        if value.grad is not None:assert torch.equal(value.grad,other.grad),name
    assert imm.mask_attention[0].weight.grad is None  # literal nondifferentiable hard-index path
    assert (imm._fc1[0].weight.grad is None)==(profile=='gmmamba_mambamil_bimamba_source')
    assert imm.BiMamba1[1].A_log.grad is not None and imm.BiMamba2[1].A_b_log.grad is not None
    assert actual.css.stoken_refine.in_proj_weight.grad is not None
    assert actual.classifier.fc2.weight.grad is not None and not actual.memory.msg_states


def test_coordinate_groups_preserve_all_rows_and_match_source_KMeans_membership():
    torch.set_num_threads(2);x=torch.arange(257*512).reshape(257,512).float();coords=coordinates()
    groups=validated_groups(x,coords);labels=KMeans(n_clusters=10,random_state=0,n_init='auto').fit(coords.numpy()).labels_
    assert len(groups)==10 and sum(group.shape[1] for group in groups)==257
    for index,group in enumerate(groups):assert torch.equal(group[0],x[np.flatnonzero(labels==index)])


def test_CSS_full_eight_head_output_matches_equations8_to12():
    torch.set_num_threads(2);model=GMMambaNetwork();x=torch.randn(1,10,1,512,requires_grad=True)
    result=model.css.stoken_forward(x)
    feature=x.squeeze(2);maximum=feature.max(1,keepdim=True).values
    association=(feature@maximum.transpose(1,2)).softmax(1)
    cross=association.transpose(1,2)@feature
    mha=model.css.stoken_refine(cross.transpose(0,1),feature.transpose(0,1),feature.transpose(0,1))[0].transpose(0,1)
    expected=(association@mha+cross).view(1,-1,512)
    assert torch.equal(result,expected)


@pytest.mark.parametrize('kind',['missing','misaligned','singleton','insufficient'])
def test_invalid_original_spatial_groups_fail_without_fabricated_instances(kind):
    coords=coordinates();x=torch.randn(257,512)
    if kind=='missing':coords=torch.full_like(coords,float('nan'))
    if kind=='misaligned':coords=coords[:-1]
    if kind=='insufficient':coords=coords[:10];x=x[:10]
    if kind=='singleton':
        coords=torch.tensor([[i/1000,0.] for i in range(90)]+[[1000.*i,0.] for i in range(1,10)])
        x=x[:len(coords)]
    with pytest.raises(ValueError):validated_groups(x,coords)


def test_released_source_rejects_non512_FM_and_unknown_or_plain_Mamba_backend():
    with pytest.raises(ValueError,match='512D'):GMMambaNetwork(1024)
    with pytest.raises(ValueError):GMMambaNetwork(backend='automatic')
    with pytest.raises(ValueError):GMMambaNetwork(profile='gmmamba_generic')

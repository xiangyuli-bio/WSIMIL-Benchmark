import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.icmil_coupled_source import BagModel,Distiller,backbone,bag_step,definitions,views,source_views
from milplus.upstream import load_definitions

WEIGHTS='/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/development/icmil_coupled_v92/legacy14_assets_r1/resnet50-11ad3fa6.pth'


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('kind',['abmil','maxpool'])
def test_full_native_MIL_source_constructor_gradients_and_two_Adam_updates(kind):
    torch.set_num_threads(2);network,attention,_=definitions();torch.manual_seed(73)
    model=BagModel(kind);initial_rng=torch.get_rng_state().clone();optimizers=model.optimizers()
    torch.manual_seed(73);classifier=network.Classifier_1fc(512,2,0.)
    attn=nn.AdaptiveMaxPool1d(1) if kind=='maxpool' else attention.Attention_Gated(512)
    reduction=network.DimReduction(1024,512,numLayer_Res=0)
    assert torch.equal(torch.get_rng_state(),initial_rng)
    optimizer=torch.optim.Adam(list(classifier.parameters())+list(attn.parameters())+list(reduction.parameters()),
        lr=2e-4 if kind=='abmil' else 1e-4,weight_decay=1e-4)
    for n,target in [(17,0),(23,1)]:
        x=torch.randn(n,1024);y=torch.tensor([target])
        z=reduction(x.unsqueeze(0)).squeeze(0)
        if kind=='abmil':
            a=attn(z).squeeze(0);features=torch.sum(torch.einsum('ns,n->ns',z,a),dim=0).unsqueeze(0)
        else:features=attn(z.t()).squeeze(0).t()
        prediction=classifier(features);loss=nn.CrossEntropyLoss(reduction='none')(prediction,y).mean()
        optimizer.zero_grad();loss.backward()
        for block in [reduction,attn,classifier]:nn.utils.clip_grad_norm_(block.parameters(),5.)
        optimizer.step()
        actual=bag_step(model,x,y,optimizers)
        assert actual['loss']==float(loss.detach()) and actual['optimizer_updates']==1
        for a,b in [(model.classifier,classifier),(model.attention,attn),(model.dim_reduction,reduction)]:
            equal(a.state_dict(),b.state_dict())
            for p,q in zip(a.parameters(),b.parameters()):equal(p.grad,q.grad)
        equal(optimizers[0].state_dict(),optimizer.state_dict())


def test_original_full_ImageNet_encoder_has_native_1024_output_and_all_loaded_weights():
    torch.set_num_threads(2);model=backbone(WEIGHTS).eval();_,_,original=definitions()
    literal=original.ResNet_Baseline(original.Bottleneck_Baseline,[3,4,6,3])
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True);literal.load_state_dict(state,strict=False);literal.eval()
    image=torch.rand(2,3,256,256)
    with torch.no_grad():a=model(image);b=literal(image)
    assert a.shape==(2,1024) and torch.equal(a,b)
    equal(model.state_dict(),literal.state_dict())
    assert sum(p.numel() for p in model.parameters())>8_000_000
    assert not hasattr(model,'layer4') and not hasattr(model,'fc')


def test_native_three_KL_losses_every_student_gradient_and_Adam_equal_literal_source():
    torch.set_num_threads(2);torch.manual_seed(114)
    encoder=backbone(WEIGHTS);bag=BagModel('abmil');model=Distiller(WEIGHTS,encoder.state_dict(),bag)
    literal=copy.deepcopy(model);optimizer=model.optimizer();reference=literal.optimizer()
    original=torch.rand(2,3,256,256);augmented=original.flip(-1)*.9
    e=literal.teacher_encoder(original);z=literal.teacher_reduction(e);p=literal.teacher_classifier(z)
    student_z=literal.student_reduction(literal.student_encoder(augmented));student_p=literal.student_classifier(student_z)
    consistency_z=literal.student_reduction(e);consistency_p=literal.student_classifier(consistency_z)
    criterion=nn.KLDivLoss(reduction='none')
    loss0=criterion(F.log_softmax(student_p,dim=-1),F.softmax(p.detach(),dim=-1)).mean()
    loss1=criterion(F.log_softmax(consistency_p,dim=-1),F.softmax(p,dim=-1)).mean()
    loss2=criterion(F.log_softmax(consistency_z,dim=-1),F.softmax(z,dim=-1)).mean()
    loss=loss0+.5*loss1+.5*loss2
    reference.zero_grad();loss.backward()
    for block in [literal.student_reduction,literal.student_classifier]:nn.utils.clip_grad_norm_(block.parameters(),5.)
    reference.step();actual=model.step(original,augmented,optimizer)
    for k,v in [('loss',loss),('patch_prediction_kl',loss0),('classifier_consistency_kl',loss1),('feature_consistency_kl',loss2)]:
        assert actual[k]==float(v.detach())
    equal(model.state_dict(),literal.state_dict());equal(optimizer.state_dict(),reference.state_dict())
    for (n,p),(m,q) in zip(model.named_parameters(),literal.named_parameters()):
        assert n==m
        if n.startswith('teacher_'):assert p.grad is None and q.grad is None and not p.requires_grad
        else:assert p.grad is not None and torch.isfinite(p.grad).all();equal(p.grad,q.grad)


def test_both_original_augmented_views_and_unused_third_view_RNG_are_preserved(tmp_path):
    from PIL import Image
    import glob,random,PIL.Image
    import torchvision.transforms as T
    image=np.random.default_rng(7).integers(0,256,(256,256,3),dtype=np.uint8)
    p=tmp_path/'patch.png';Image.fromarray(image).save(p)
    source=load_definitions('icmil_coupled','dataset/RandomPatchDistillationDataset.py',
        ['RandomPatchDistillationDataset'],{'torch':torch,'glob':glob,'random':random,'PIL':PIL,'T':T})
    dataset=source.RandomPatchDistillationDataset.__new__(source.RandomPatchDistillationDataset)
    dataset.data=[str(p)];dataset.transform,dataset.transform1,dataset.transform2=source_views()
    torch.manual_seed(17);a=dataset[0];expected=torch.get_rng_state().clone()
    torch.manual_seed(17);b=views(image);assert torch.equal(torch.get_rng_state(),expected)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    torch.manual_seed(17);source_views()[0](Image.fromarray(image));source_views()[1](Image.fromarray(image))
    assert not torch.equal(torch.get_rng_state(),expected)


def test_DTFD_cannot_silently_use_a_different_or_unverified_update_rule():
    model=BagModel('dtfd')
    with pytest.raises(ValueError,match='Torch1.4'):bag_step(model,torch.randn(17,1024),torch.tensor([1]),model.optimizers())


def test_unmodified_Adam14_preserves_actual_gradient_decay_and_parameter_version_semantics():
    from milplus.vendor.torch14_adam import Adam
    from milplus.io import digest_file
    import milplus.vendor.torch14_adam.adam as original
    assert digest_file(original.__file__)=='97067d8382378053cbd13084c84f1da6dac7f584b7fbceff5f61b2070e776091'
    p=nn.Parameter(torch.tensor([1.]));p.grad=torch.tensor([2.]);version=p._version
    opt=Adam([p],lr=1e-3,weight_decay=1e-4);opt.step()
    assert p._version==version and p.item()!=1.
    assert torch.equal(p.grad,torch.tensor([2.0001]))
    assert opt.state[p]['step']==1 and isinstance(opt.state[p]['step'],int)
    opt.zero_grad();assert p.grad is not None and torch.equal(p.grad,torch.zeros(1))


def test_DTFD_entire_original_training_body_runs_unchanged_with_original_Adam14():
    import ast,types
    from milplus.vendor.torch14_adam import Adam
    from milplus.upstream import verified_source,pinned_text
    torch.set_num_threads(2);torch.manual_seed(319);model=BagModel('dtfd');literal=copy.deepcopy(model)
    opts=model.optimizers();opt0=Adam(list(literal.classifier.parameters())+list(literal.attention.parameters())+
        list(literal.dim_reduction.parameters()),lr=1e-4,weight_decay=1e-4)
    opt1=Adam(literal.att_classifier.parameters(),lr=1e-4,weight_decay=1e-4)
    source,sha=verified_source('icmil_coupled');tree=ast.parse(pinned_text(source,sha,'step1_train_classifier_dtfdmil.py'))
    outer=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ii')
    inner=next(n for n in outer.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
    code=compile(ast.Module(body=inner.body,type_ignores=[]),'actual_original_DTFD_training_AST','exec')
    ns={'torch':torch,'np':np,'classifier':literal.classifier,'attention':literal.attention,'dimReduction':literal.dim_reduction,
        'attCls':literal.att_classifier,'optimizer0':opt0,'optimizer1':opt1,'ce_cri':nn.CrossEntropyLoss(reduction='none'),
        'params':types.SimpleNamespace(device='cpu',numGroup=5,grad_clipping=5)}
    for step,n in enumerate([17,23,17,23]):
        x=torch.randn(n,1024);y=torch.tensor([step%2]);ns.update(data=(x.unsqueeze(0),y),i=step,ii=step//2)
        exec(code,ns);actual=bag_step(model,x,y,opts,legacy_adam14=True)
        assert actual['bag_ce']==float(ns['loss0'].detach()) and actual['second_tier_ce']==float(ns['loss1'].detach())
        equal(model.state_dict(),literal.state_dict());equal(opts[0].state_dict(),opt0.state_dict());equal(opts[1].state_dict(),opt1.state_dict())
        for p,q in zip(model.parameters(),literal.parameters()):equal(p.grad,q.grad)

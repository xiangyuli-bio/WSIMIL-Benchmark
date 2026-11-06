import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from milplus import akdpmp_source as a
from milplus.engine import rng_state,restore_rng


def same(x,y):
    if isinstance(x,torch.Tensor):assert torch.equal(x,y)
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:same(x[k],y[k])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for xx,yy in zip(x,y):same(xx,yy)
    else:assert x==y


@pytest.mark.parametrize('method',['random','max','maxmin','maxminrand','maxrand'])
def test_actual_pseudo_bag_indices_match_author_selector_values_and_rng(method):
    torch.set_num_threads(2);torch.manual_seed(19);x=torch.randn(41,1024);attention=torch.rand(1,41);state=rng_state()
    expected=a.definitions().utilities.distill_slide(x,attention,size=8,method=method);after=rng_state()
    restore_rng(state);idx=a.definitions().selection_indices(x,attention,8,method)
    assert torch.equal(x[idx],expected);same(after,rng_state())
    if method=='maxrand':assert len(idx)==8 # Actual top4+random4; duplicates are not removed.


@pytest.mark.parametrize('large',[False,True])
def test_actual_smooth_SVM_both_branches_values_and_gradients_unchanged(large):
    torch.set_num_threads(2);torch.manual_seed(7);x=torch.randn(9,2)*(20 if large else 1);y=torch.arange(9)%2
    left=x.clone().requires_grad_();right=x.clone().requires_grad_()
    original=a.smooth_svm().original(2);compatible=a.smooth_svm().compatible(2)
    u=original(left,y);v=compatible(right,y);u.backward();v.backward();assert torch.equal(u,v) and torch.equal(left.grad,right.grad)
    compatible=compatible.to(dtype=torch.float64);assert compatible.labels.dtype==torch.int64
    assert torch.isfinite(compatible(right.double(),y))


@pytest.mark.parametrize('balanced',[False,True])
def test_native_reservoir_quota_sampling_and_every_state_field_restore(balanced):
    from milplus.akdpmp_driver import buffer_state,_buffer
    torch.manual_seed(1);np.random.seed(1);cfg=a.arguments('transmil','tcga',4,{'buffer_size':8},2);cfg.buffer_balance=balanced
    buffer=_buffer(cfg);buffer.adjust_buffer_size_by_new_classes(2)
    for i in range(30):
        label=i%2;index=buffer.add_judge(label)
        if index>=0:buffer.add({'features':torch.full((4,1024),float(i)),'label':torch.tensor([label])},index)
    buffer.start_epoch();buffer.get_next_batch();saved=buffer_state(buffer);restored=_buffer(cfg,saved)
    same(saved,buffer_state(restored));state=rng_state()
    values=[]
    while (value:=buffer.get_next_batch()) is not None:values.append(value)
    restore_rng(state);actual=[]
    while (value:=restored.get_next_batch()) is not None:actual.append(value)
    same(values,actual)


def manual_step(kind,network,teacher,opt,batch,seen,old_seen,replay):
    opt.zero_grad()
    if kind=='clam':
        out=network(batch['features'],batch['label'],instance_eval=True,return_features=True,seen_classes=seen)
        loss=.7*F.cross_entropy(out['logits'],batch['label'])+.3*out['instance_loss']
        if replay:
            old=teacher(batch['features'],batch['label'],instance_eval=True,return_features=True,seen_classes=old_seen)
            logits_loss=a.definitions().utilities.kd_loss_fn(out['logits'][:,old_seen],old['logits'][:,old_seen])
            attn_loss=a.definitions().utilities.kd_loss_fn(out['A'],old['A'])
            loss=loss+attn_loss+10*logits_loss
    else:
        out=network(batch['features'],return_attn=replay);unseen=list(set(range(7))-set(seen));out['logits'][:,unseen]=-100
        loss=F.cross_entropy(out['logits'],batch['label'])
        if replay:
            old=teacher(batch['features'],return_attn=True);kd=a.definitions().utilities.kd_loss_fn
            logits_loss=10*kd(out['logits'][:,old_seen],old['logits'][:,old_seen])
            attn_loss=10*(kd(out['attn1'],old['attn1'],ta=1)+kd(out['attn2'],old['attn2'],ta=1))
            h_loss=10*(F.mse_loss(out['h1'],old['h1'])+F.mse_loss(out['h2'],old['h2']))
            loss=loss+logits_loss+attn_loss+h_loss
    loss.backward();opt.step();return float(loss)


@pytest.mark.parametrize('kind',['clam','transmil'])
def test_full_original_current_and_replay_objectives_every_gradient_and_Adam_oracle(kind):
    torch.set_num_threads(2);torch.manual_seed(83);np.random.seed(83);left=a.model(kind,1024,7);right=copy.deepcopy(left)
    old=copy.deepcopy(left).eval()
    for p in old.parameters():p.requires_grad=False
    current={'features':torch.randn(41,1024),'label':torch.tensor([2])};previous={'features':torch.randn(21,1024),'label':torch.tensor([1])}
    one=torch.optim.Adam(left.parameters(),lr=2e-4,weight_decay=1e-5);two=torch.optim.Adam(right.parameters(),lr=2e-4,weight_decay=1e-5)
    cfg=a.arguments(kind,'tcga',7,{'epochs':2},3);buffer=a.definitions().utilities.SimpleBuffer(8);buffer.add(previous,0)
    left.train();right.train();state=rng_state();first=manual_step(kind,left,old,one,current,[0,1,2,3],[0,1],False)
    buffer.start_epoch();replay=buffer.get_next_batch()[0];second=manual_step(kind,left,old,one,replay,[0,1,2,3],[0,1],True)
    restore_rng(state);metric=a.Mean();log=a.Logger();fabric=a.Fabric('cpu')
    a.definitions().epoch(cfg,right,two,[current],buffer,old,np.array([0,1,2,3]),np.array([0,1]),[4,5,6],1,0,fabric,metric,log)
    assert fabric.updates==2 and metric.values==[first,second]
    same(left.state_dict(),right.state_dict());same(one.state_dict(),two.state_dict())
    same({n:p.grad for n,p in left.named_parameters()},{n:p.grad for n,p in right.named_parameters()})
    assert all(not p.requires_grad and p.grad is None for p in old.parameters())
    assert 'attn_loss' in log.rows[-1] and 'logits_loss' in log.rows[-1]


def test_all_four_actual_recipes_and_original_epoch_minimum_are_preserved():
    for kind in ['clam','transmil']:
        tcga=a.arguments(kind,'tcga',7,{},3);skin=a.arguments(kind,'skin',6,{},3)
        assert tcga.epochs==[50]*3 and tcga.buffer_size==300 and tcga.buffer_balance and tcga.weighted_sample and tcga.buffer_slide_size==.1
        assert skin.buffer_size==42 and not skin.buffer_balance and not skin.weighted_sample and skin.buffer_slide_size==-1
        assert skin.epochs==([40,50,50] if kind=='clam' else [20,40,25])
    early=a.definitions().utilities.EarlyStopping(patience=10,stop_epoch=25)
    for epoch in range(26):early(epoch,float(epoch+1));assert not early.early_stop
    early(26,100.);assert early.early_stop


def test_validation_selection_preserves_original_float32_aggregation_rounding():
    metric=a.Mean()
    for v in [1e8,1.,-1e8]:metric.update(v)
    assert metric.mean()==0.0 # torchmetrics CPU float32; a float64 mean would be 1/3.

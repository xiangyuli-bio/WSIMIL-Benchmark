"""Independent fixed author graph, joint objective and complete train-loop checks."""
import ast
import copy
import logging
import math
import os
import subprocess

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler

from milplus.ahdmil_source import RECIPES, arguments, build_joint, dmin_objective, inference, lipn_objective
from milplus.engine import rng_state, restore_rng, seed_everything
from milplus.mammoth import BinarySmoothTop1
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def reference(path,scope,replacements=()):
    root,sha=verified_source('ahdmil')
    code=subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)
    for old,new in replacements:
        assert code.count(old)==1
        code=code.replace(old,new)
    nodes=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    scope=dict(scope);exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)
    return scope


def original_models(profile,dim):
    scope=dict(torch=torch,nn=nn,F=F,np=np,math=math,copy=copy)
    scope.update(reference('models/mobilenetv4.py',scope));scope.update(reference('models/hdmil.py',scope))
    args=arguments(profile,dim)
    dmin=scope['KAN_CLAM_MB_v5'](I=dim,dropout=True,n_classes=args.n_classes,subtyping=args.subtyping,
            instance_loss_fn=BinarySmoothTop1(),k_sample=args.k_sample,args=args)
    return dmin,scope['ImageProcessor_MoE'](args)


def original_runner(profile,dim,dmin,lipn):
    from sklearn import metrics
    scope=dict(torch=torch,np=np,F=F,Dataset=Dataset,logging=logging,os=os,tqdm=lambda x:x)
    scope.update({n:getattr(metrics,n) for n in ['roc_auc_score','roc_curve','auc','accuracy_score','precision_score','recall_score','f1_score']})
    source=reference('Joint.py',scope)['HADMIL'];runner=source.__new__(source)
    runner.args=arguments(profile,dim);runner.args.device='cpu'
    runner.DMIN,runner.LIPN=dmin,lipn
    runner.l1_loss=nn.L1Loss();runner.l2_loss=nn.MSELoss()
    runner.cross_entropy_loss=nn.CrossEntropyLoss();runner.cross_entropy_loss_ls=nn.CrossEntropyLoss()
    return runner


@pytest.mark.parametrize('profile,dim',[(p,1536 if 'nsclc' in p else 1024) for p in RECIPES])
@pytest.mark.parametrize('mode',[0.,1.])
def test_native_all_experts_losses_gradients_bn_rng_and_inference(profile,dim,mode):
    seed_everything(2026)
    dmin,lipn=build_joint(profile,dim);other,experts=original_models(profile,dim)
    other.load_state_dict(dmin.state_dict());experts.load_state_dict(lipn.state_dict())
    runner=original_runner(profile,dim,other,experts);runner.args.attn_ratio=mode
    x=torch.randn(1,40,dim);rgb=torch.randint(256,(1,40,16,16,3),dtype=torch.uint8)
    label=torch.tensor([RECIPES[profile]['classes']-1])
    rng=rng_state();loss,parts,manner=lipn_objective(dmin,lipn,x,rgb,mode);loss.backward();after=rng_state()
    restore_rng(rng);expected,rate,score=runner.train_LIPN(x,label,rgb);expected.backward()
    equal(loss,expected);equal(parts['mask_rate'],rate);equal(parts['score_rate'],score);equal(after,rng_state())
    assert manner==('hard_train' if mode==0 else 'attn_train')
    for (n,p),(m,q) in zip(lipn.named_parameters(),experts.named_parameters()):
        assert n==m;equal(p.grad,q.grad)
    assert len(lipn.extractors)==2
    assert all(e.features[0].block[0].weight.grad.abs().sum()>0 for e in lipn.extractors)
    assert all(p.grad is None for p in dmin.parameters())
    rng=rng_state();ce=dmin_objective(dmin,lipn,x,rgb,label);ce.backward();after=rng_state()
    restore_rng(rng);expected=runner.tune_DMIN(x,label,rgb);expected.backward();equal(ce,expected);equal(after,rng_state())
    for (n,p),(m,q) in zip(dmin.named_parameters(),other.named_parameters()):
        assert n==m;equal(p.grad,q.grad)
    equal(lipn.state_dict(),experts.state_dict())
    assert dmin.attention_net[0].out_features==512 and dmin.attention_net[-1].attention_a[0].out_features==256
    assert dmin.classifiers[0].degree==(16 if 'nsclc' in profile else 12)
    dmin.eval();lipn.eval();other.eval();experts.eval()
    logits,rate=inference(dmin,lipn,x,rgb);_,prob,expected_rate=runner.test_inference(x,label,rgb,other,experts)
    equal(logits.softmax(1),prob);equal(rate,expected_rate)


def test_cross_ema_is_simultaneous_parameter_only_and_preserves_bn_buffers_and_rng():
    _,experts=build_joint('ahdmil_camelyon16_source_repaired',1024)
    with torch.no_grad():
        for index,expert in enumerate(experts.extractors):
            for p in expert.parameters():p.fill_(1. if index==0 else 3.)
            for b in expert.buffers():b.fill_(index+4)
    before=copy.deepcopy(experts.state_dict());rng=rng_state();experts.cross_EMA();equal(rng,rng_state())
    for index,expert in enumerate(experts.extractors):
        for name,p in expert.named_parameters():
            a=before[f'extractors.{index}.{name}'];b=before[f'extractors.{1-index}.{name}']
            equal(p,.8*a+.2*b)
        for name,b in expert.named_buffers():equal(b,before[f'extractors.{index}.{name}'])


def test_actual_author_joint_loop_all_updates_ema_validation_and_rng(tmp_path):
    from milplus.data import Bundle
    from milplus.engine import train
    from milplus.hdmil_demo import create_hdmil_demo
    import os
    data=create_hdmil_demo(tmp_path/'data',patients=20,dimension=1024)
    config=dict(epochs=5,threads=1,model=dict(dmin_epochs=2,joint_epochs=3,dmin_selection_start=1))
    train(data,'AHDMIL',tmp_path/'run',seed=2026,config=config)
    actual=torch.load(tmp_path/'run/last.pt',weights_only=False);extra=actual['model']['_extra_state']
    seed_everything(2026);dmin,lipn=original_models('ahdmil_brca_source_repaired',1024)
    dmin.load_state_dict(extra['selected_dmin']);runner=original_runner('ahdmil_brca_source_repaired',1024,dmin,lipn)
    runner.args.n_epochs=3;runner.args.dataset='TCGA-BRCA'
    runner.counter=0;runner.patience=5;runner.stop_epoch=0;runner.best_loss=np.inf;runner.flag=1
    runner.best_valid_metrics=None;runner.lower_ratio=.3;runner.upper_ratio=1.
    runner.DMIN_path=str(tmp_path/'selected_dmin.pt');runner.LIPN_path=str(tmp_path/'selected_lipn.pt')
    runner.optimizer1=torch.optim.Adam(lipn.parameters(),lr=1e-5,weight_decay=1e-5)
    runner.optimizer2=torch.optim.Adam(dmin.parameters(),lr=1e-5,weight_decay=1e-5)
    bundle=Bundle(data)
    class AuthorSlides(Dataset):
        def __init__(self,subset):self.rows=bundle.records(subset)
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            row=self.rows[index];x,img=bundle.hdmil.load(row)
            return x,row['target'],row['slide_id'],img,row['slide_id']
    rows=bundle.records('train');counts=np.bincount([r['target'] for r in rows])
    weights=torch.DoubleTensor([len(rows)/counts[r['target']] for r in rows])
    dataset=AuthorSlides('train');val=AuthorSlides('val')
    runner.train_loader=DataLoader(dataset,batch_size=1,sampler=WeightedRandomSampler(weights,len(weights),replacement=True))
    runner.valid_loader=DataLoader(val,batch_size=1,sampler=SequentialSampler(val))
    runner.train()
    for prefix,module in [('dmin',dmin),('lipn',lipn)]:
        equal({k[len(prefix)+1:]:v for k,v in actual['model'].items() if k.startswith(prefix+'.')},module.state_dict())
    equal(actual['optimizers'][1:],[runner.optimizer1.state_dict(),runner.optimizer2.state_dict()]);equal(actual['rng'],rng_state())
    selected=torch.load(tmp_path/'run/best.pt',weights_only=False)['model']
    for prefix,path in [('dmin',runner.DMIN_path),('lipn',runner.LIPN_path)]:
        assert os.path.isfile(path)
        equal({k[len(prefix)+1:]:v for k,v in selected.items() if k.startswith(prefix+'.')},torch.load(path,weights_only=True))
    assert extra['ema_calls']==4 and extra['generation_updates']==3*len(rows)


@pytest.mark.parametrize('task,epoch,classes',[('brca',5,2),('rcc',55,3)])
def test_actual_author_validation_reset_and_explicit_rcc_attribute_repair(tmp_path,task,epoch,classes):
    import types
    from milplus.ahdmil import AHDMIL
    from milplus.data import Bundle
    from milplus.hdmil import snapshot,tensor_state_hash
    from milplus.hdmil_demo import create_hdmil_demo
    bundle=Bundle(create_hdmil_demo(tmp_path/'data',patients=10*classes,classes=classes))
    profile='ahdmil_'+task+'_source_repaired';port=AHDMIL(bundle,profile=profile)
    port.make_optimizers(port.native['lr'],1e-5)
    port.selected_dmin=snapshot(port.pretrain);port.selected_dmin_hash=tensor_state_hash(port.selected_dmin)
    port.stage='joint';port.apply_phase();port.stage_epoch=epoch;port.epoch_index=100+epoch;port.stage_bad['joint']=4
    dmin,lipn=original_models(profile,bundle.dim);source=original_runner(profile,bundle.dim,dmin,lipn)
    source.args.dataset='TCGA-RCC' if classes==3 else 'TCGA-BRCA'
    source.args.pretrain_dir=str(tmp_path/'original');source.args.k=0
    path=tmp_path/'original/fold-0';path.mkdir(parents=True)
    torch.save(port.selected_dmin,path/'best_epoch.pth')
    source.init_model=lambda:original_models(profile,bundle.dim)
    source.counter=4;source.patience=5;source.flag=1;source.best_loss=np.inf
    source.best_valid_metrics=None;source.lower_ratio=.3;source.upper_ratio=1.
    source.valid_loader=[(torch.zeros(1,1,bundle.dim),torch.tensor([c]),'',torch.zeros(1,1,16,16,3),'') for c in range(classes)]
    source.test_inference=lambda *unused:(torch.tensor(1.),torch.ones(1,classes)/classes,torch.tensor(0.))
    if classes==3:
        # This targeted port is explicit, and the failing original attribute is
        # independently executed before checking the corrected source branch.
        before=rng_state()
        with pytest.raises(AttributeError,match='lower_ratio'):source.valid(epoch)
        restore_rng(before);source.counter=4
        bindings=dict(torch=torch,np=np,F=F,Dataset=Dataset,logging=logging,tqdm=lambda x:x)
        from sklearn import metrics
        bindings.update({n:getattr(metrics,n) for n in ['roc_auc_score','roc_curve','auc','accuracy_score','precision_score','recall_score','f1_score']})
        repaired=reference('Joint.py',bindings,[('self.args.lower_ratio -= 0.05','self.lower_ratio -= 0.05')])['HADMIL']
        source.valid=types.MethodType(repaired.valid,source)
    before=rng_state();port.after_validation(dict(ahdmil_source_ce=1.,ahdmil_retention=0.));after=rng_state()
    restore_rng(before);source.valid(epoch);equal(rng_state(),after)
    equal(port.dmin.state_dict(),source.DMIN.state_dict());equal(port.lipn.state_dict(),source.LIPN.state_dict())
    equal([o.state_dict() for o in port.optimizers[1:]],[source.optimizer1.state_dict(),source.optimizer2.state_dict()])
    assert source.lower_ratio==port.lower_ratio and source.counter==port.stage_bad['joint']==0
    assert source.best_valid_metrics is None and port.stage_best['joint'] is None and source.flag==1

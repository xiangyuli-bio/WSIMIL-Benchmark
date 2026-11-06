"""Full original epoch/validation loops, stage transitions, continuation and exports."""
import ast
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from milplus.data import Bundle
from milplus.engine import evaluate, predict, restore_rng, rng_state, train, units
from milplus.smmile import PROFILES, SMMILe, sample_loader, smmile_epoch, source_bitempered
from milplus.smmile_demo import create_smmile_demo
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture(tmp_path,profile,**options):
    variant=PROFILES[profile]
    data=create_smmile_demo(tmp_path/'data',variant,dimension=4)
    if options.get('consistency'):
        task=yaml.safe_load((data/'task.yaml').read_text())
        task['smmile_negative_bag_semantics']='synthetic explicit negative bag; no clinical assertion'
        (data/'task.yaml').write_text(yaml.safe_dump(task))
    bundle=Bundle(data)
    model=SMMILe(bundle,profile,**{'warmup_epochs':1,'groups':2,**options})
    return model,bundle


class Logger:
    def __init__(self,*args,**kwargs):pass
    def log(self,*args,**kwargs):pass


def reference(variant):
    root,sha=verified_source('smmile')
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'pd':pd,'randint':random.randint,
           'random':random.random,'sample':random.sample,'device':torch.device('cpu'),
           'optim':torch.optim,'DataLoader':DataLoader,'WeightedRandomSampler':WeightedRandomSampler,
           'RandomSampler':RandomSampler,'SequentialSampler':SequentialSampler,
           'Accuracy_Logger':Logger,'calculate_error':lambda *args:0.,'print':lambda *args,**kwargs:None,
           'safe_span':lambda v:v.masked_fill(v==0,1)}
    branch='multi' if variant=='multilabel' else 'single'
    def read(path,names,network=False):
        code=subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)
        if network:
            code=code.replace('.cuda()', '.to(device)')
            code=code.replace('(final_score[:,i].max()-final_score[:,i].min())','safe_span(final_score[:,i].max()-final_score[:,i].min())')
            code=code.replace('(cls_final_score.max()-cls_final_score.min())','safe_span(cls_final_score.max()-cls_final_score.min())')
            code=code.replace('(valid_loss.max() - valid_loss.min())','safe_span(valid_loss.max() - valid_loss.min())')
            line=next(line for line in code.splitlines() if 'adj_sp_scores = sp_scores[(one_adj==1)]' in line)
            spaces=' '*(len(line)-len(line.lstrip()))
            code=code.replace(line,line+'\n'+spaces+'if len(adj_sp_scores) == 0:\n'+spaces+'    continue')
        nodes=[node for node in ast.parse(code).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent locked author Git AST
    read(branch+'/utils/utils.py',{'initialize_weights','collate_MIL','get_split_loader','make_weights_for_balanced_classes_split','get_optim'})
    read(branch+'/models/model_smmile.py',{'Attn_Net','Attn_Net_Gated','IAMIL','SMMILe','SMMILe_SINGLE'},network=True)
    read(branch+'/utils/bi_tempered_loss_pytorch.py',{'log_t','bi_tempered_binary_logistic_loss','bi_tempered_logistic_loss'})
    filename=branch+'/utils/core_utils.py'
    text=subprocess.check_output(['git','show',sha+':'+filename],cwd=root,text=True)
    text=text.replace('device=torch.device("cuda" if torch.cuda.is_available() else "cpu")', "device=torch.device('cpu')")
    suffix='_single' if variant=='binary' else ''
    names={'train_loop_smmile'+suffix,'validate_smmile'+suffix}
    nodes=[]
    for node in ast.parse(text).body:
        if not isinstance(node,ast.FunctionDef) or node.name not in names:continue
        # Retain every statement through the original optimization/validation loop.
        # Discard only post-loop annotation-dependent reporting and file writes.
        stop=next(i for i,stmt in enumerate(node.body) if 'enumerate(loader)' in ast.unparse(stmt))
        node.body=node.body[:stop+1]
        if node.name.startswith('validate'):
            node.body.append(ast.parse('return val_loss / len(loader)').body[0])
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),filename,'exec'),scope)  # noqa: S102 -- numerical author loop intact; metrics-only tail removed
    scope['train_loop']=scope['train_loop_smmile'+suffix]
    scope['validate_loop']=scope['validate_smmile'+suffix]
    return scope


class SourceDataset:
    def __init__(self,bundle,groups,variant):
        self.bundle,self.groups,self.variant=bundle,sorted(groups,key=lambda rows:rows[0]['slide_id']),variant
        keys=[tuple(rows[0]['targets']) if variant=='multilabel' else rows[0]['target'] for rows in self.groups]
        unique=sorted(set(keys));self.codes=[unique.index(k) for k in keys]
        self.slide_cls_ids=[np.flatnonzero(np.asarray(self.codes)==i) for i in range(len(unique))]
    def __len__(self):return len(self.groups)
    def getlabel(self,idx):return self.codes[idx]
    def __getitem__(self,idx):
        row=self.groups[idx][0]
        x,coords,layout,sp,adj=self.bundle.smmile.load(row)
        label=list(row['targets']) if self.variant=='multilabel' else row['target']
        return layout.scatter(x),label,[coords.numpy(),layout.mask,sp,adj],[]


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def capture(network,optimizer,destination):
    destination.append({'state':copy.deepcopy(network.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                        'grads':{name:None if p.grad is None else p.grad.detach().clone() for name,p in network.named_parameters()}})


@pytest.mark.parametrize('profile',sorted(PROFILES))
@pytest.mark.parametrize('weighted',[True,False])
@pytest.mark.parametrize('bag_loss',['bce','bibce'])
def test_complete_author_training_validation_and_stage_reset(tmp_path,profile,weighted,bag_loss):
    consistency=bag_loss=='bibce' and profile!='smmile'
    model,bundle=fixture(tmp_path,profile,weighted_sampling=weighted,bag_loss=bag_loss,mrf=True,inst_rate=.2,consistency=consistency)
    scope=reference(model.variant)
    cls=scope['SMMILe_SINGLE'] if model.variant=='binary' else scope['SMMILe']
    original=cls(fea_dim=4,n_classes=len(bundle.classes),n_refs=3)
    original.load_state_dict(model.network.state_dict())
    cfg={'epochs':2,'max_instances':None,'grad_clip':None,'selection_metric':'smmile_source_loss'}
    model.configure_training(cfg)
    opt=model.make_optimizer(2e-5,1e-5)
    refopt=scope['get_optim'](original,SimpleNamespace(opt='adam',lr=2e-5,reg=1e-5))
    refsched=torch.optim.lr_scheduler.ReduceLROnPlateau(refopt,mode='min',factor=.5,patience=10)
    args=SimpleNamespace(n_classes=len(bundle.classes),bi_loss=bag_loss=='bibce',drop_with_score=True,D=1,
                         superpixel=True,sp_smooth=True,G=2,inst_refinement=True,inst_rate=.2,
                         mrf=True,tau=.1,consistency=consistency,multi_label=model.variant=='multilabel',results_dir=str(tmp_path))
    groups=units(bundle,'train')[:9]
    valgroups=units(bundle,'val')
    snapshot_a,snapshot_b=[],[]
    opt.register_step_post_hook(lambda o,a,k:capture(model.network,o,snapshot_a))
    refopt.register_step_post_hook(lambda o,a,k:capture(original,o,snapshot_b))
    best_ref=None
    for epoch in range(2):
        model.train();model.start_epoch(epoch)
        if epoch==1:
            assert best_ref is not None
            original.load_state_dict(best_ref)
            refopt.state.clear()
            refopt.param_groups[0]['lr']=2e-5
            refsched=torch.optim.lr_scheduler.ReduceLROnPlateau(refopt,mode='min',factor=.5,patience=10)
        loader=scope['get_split_loader'](SourceDataset(bundle,groups,model.variant),training=True,weighted=weighted)
        state=rng_state()
        scope['train_loop'](epoch,original,loader,refopt,loss_fn=F.binary_cross_entropy,ref_start=epoch>0,args=args)
        restore_rng(state)
        smmile_epoch(model,bundle,groups,opt,'cpu',cfg)
        equal(snapshot_a,snapshot_b)
        original.eval()
        val_loader=scope['get_split_loader'](SourceDataset(bundle,valgroups,model.variant))
        state=rng_state()
        refval=scope['validate_loop'](0,epoch,original,val_loader,loss_fn=F.binary_cross_entropy,
                                    ref_start=epoch>0,args=args)
        source_rng=rng_state()
        restore_rng(state)
        pred=predict(model,bundle,profile,'val','cpu',cfg)
        val=model.validation_metrics(pred)['smmile_source_loss']
        assert val==refval
        equal(rng_state()['torch'],source_rng['torch']);assert rng_state()['python']==source_rng['python']
        refsched.step(refval);model.after_validation({'smmile_source_loss':val})
        equal(refsched.state_dict(),model.scheduler.state_dict())
        if epoch==0:best_ref=copy.deepcopy(original.state_dict())
    assert len(snapshot_a)==18 and model.stage_transitions==1
    assert model.optimizer_steps==18


@pytest.mark.parametrize('profile',sorted(PROFILES))
@pytest.mark.parametrize('stop_after',[1,2])
def test_exact_continuation_both_sides_of_transition_and_export(tmp_path,profile,stop_after):
    _model,bundle=fixture(tmp_path,profile)
    cfg={'epochs':3,'threads':1,'model':{'warmup_epochs':1,'groups':2}}
    train(bundle.root,profile,tmp_path/'full',config=cfg)
    train(bundle.root,profile,tmp_path/'resume',config=cfg,stop_after=stop_after)
    train(bundle.root,profile,tmp_path/'resume',config=cfg,resume=True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/last.pt',weights_only=False)
    for key in ['model','optimizers','history','best','bad','rng']:equal(a[key],b[key])
    first=evaluate(tmp_path/'full');second=evaluate(tmp_path/'resume')
    assert first==second
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'full/test_predictions.csv'),pd.read_csv(tmp_path/'resume/test_predictions.csv'))
    assert evaluate(tmp_path/'full')==first
    manifest=json.loads((tmp_path/'full/test_instances/manifest.json').read_text())
    assert manifest['synthetic'] and len(manifest['records'])==len(bundle.records('test'))
    for record in manifest['records']:
        row=next(row for row in bundle.records('test') if row['slide_id']==record['slide_id'])
        x,coords=bundle.load(row)
        with h5py.File(tmp_path/'full/test_instances'/record['file']) as h:
            np.testing.assert_array_equal(h['coords'][:],coords.numpy())
            assert len(h['refined_scores'])==len(x)
            assert h.attrs['coordinate_frame']=='level0_pixels'
    protocol=json.loads((tmp_path/'full/method_protocol.json').read_text())
    assert protocol['stage_transitions']==1 and protocol['optimizer_steps']>0


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_query_targets_cannot_change_native_or_refined_instance_predictions(tmp_path,profile):
    model,bundle=fixture(tmp_path,profile)
    model.epoch=1;model.eval()
    row=bundle.records('test')[0]
    other=copy.deepcopy(row)
    if model.variant=='multilabel':other['targets']=tuple(1-x for x in row['targets'])
    else:other['target']=(row['target']+1)%len(bundle.classes)
    a=model.forward_unit(bundle,[row],'cpu',{})
    b=model.forward_unit(bundle,[other],'cpu',{})
    for key in ['bag_scores','probabilities','instance_attention_product','instance_class_scores','attention','coords','refined_scores']:
        equal(a[key],b[key])
    contents=[]
    for index,candidate in enumerate([row,other]):
        directory=tmp_path/f'exports_{index}'
        record=model.export_instances(model.forward_unit(bundle,[candidate],'cpu',{}),candidate,directory)
        with h5py.File(directory/record['file']) as h:
            contents.append({'arrays':{key:h[key][:] for key in h},'attrs':dict(h.attrs)})
    equal(contents[0],contents[1])
    assert model.network.training is False


def test_stage_requires_best_snapshot_and_source_budget(tmp_path):
    model,_bundle=fixture(tmp_path,'smmile')
    model.make_optimizer(2e-5,1e-5)
    with pytest.raises(ValueError,match='stage1 best'):model.start_epoch(1)
    base={'epochs':3,'max_instances':None,'grad_clip':None,'selection_metric':'smmile_source_loss'}
    for key,value in [('epochs',1),('max_instances',4),('grad_clip',5.),('selection_metric','macro_auc')]:
        with pytest.raises(ValueError):model.configure_training({**base,key:value})
    assert model.selection_improved(-1.,-1.)
    model.epoch=0;model.after_validation({'smmile_source_loss':1.})
    first=model.warmup_best['cls_net.weight'].clone()
    with torch.no_grad():model.network.cls_net.weight.add_(1)
    assert not torch.equal(model.network.cls_net.weight,first)
    model.after_validation({'smmile_source_loss':1.})
    model.start_epoch(1)
    assert torch.equal(model.network.cls_net.weight,model.warmup_best['cls_net.weight'])


@pytest.mark.parametrize('profile',['smmile_binary','smmile_multilabel'])
def test_negative_consistency_requires_semantics_and_uses_native_weight(tmp_path,profile):
    model,bundle=fixture(tmp_path,profile)
    with pytest.raises(ValueError,match='semantics'):SMMILe(bundle,profile,consistency=True)
    task=yaml.safe_load((bundle.root/'task.yaml').read_text())
    task['smmile_negative_bag_semantics']='synthetic negative bags; no clinical claim'
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    bundle=Bundle(bundle.root);model=SMMILe(bundle,profile,consistency=True,warmup_epochs=1,groups=2)
    if model.variant=='multilabel':
        row=next(r for r in bundle.records('train') if not any(r['targets']))
    else:
        row=next(r for r in bundle.records('train') if r['target']==0)
    result=model.raw_forward(bundle,row,'cpu',True)
    raw=result['source_result'];consistency=raw[4]['consist_loss']
    assert consistency>0 and len(raw[1])==1
    target=torch.zeros_like(raw[1][0])
    expected=F.binary_cross_entropy(raw[1][0],target)+consistency*(.1 if model.variant=='multilabel' else 1.)
    torch.testing.assert_close(result['loss'],expected,rtol=0,atol=0)


def test_sampler_weights_are_by_full_training_label_combination(tmp_path):
    _model,bundle=fixture(tmp_path,'smmile_multilabel')
    groups=units(bundle,'train')
    loader=sample_loader(groups,'multilabel',True)
    keys=[tuple(rows[0]['targets']) for rows in sorted(groups,key=lambda rows:rows[0]['slide_id'])]
    expected=torch.tensor([len(keys)/keys.count(k) for k in keys],dtype=torch.double)
    assert torch.equal(loader.sampler.weights,expected)
    assert loader.sampler.replacement and loader.sampler.num_samples==len(groups)


def test_bitempered_saturated_endpoint_repair_is_explicit():
    raw=reference('multilabel')['bi_tempered_binary_logistic_loss']
    p=torch.tensor([1.,.2],requires_grad=True);target=torch.tensor([1.,0.])
    source=raw(p,target,.2,1.);source.backward()
    assert not torch.isfinite(p.grad).all()
    safe=p.detach().clone().requires_grad_()
    loss=source_bitempered('multilabel')(safe,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(safe.grad).all()


def test_nonfinite_gradient_cannot_reach_adam(tmp_path):
    model,bundle=fixture(tmp_path,'smmile')
    optimizer=model.make_optimizer(2e-5,1e-5);model.start_epoch(0)
    before={n:p.detach().clone() for n,p in model.network.named_parameters()}
    handle=model.network.cls_net.weight.register_hook(lambda grad:grad*float('nan'))
    with pytest.raises(FloatingPointError,match='Adam update refused'):
        smmile_epoch(model,bundle,units(bundle,'train')[:1],optimizer,'cpu',{})
    handle.remove()
    assert model.optimizer_steps==0 and len(optimizer.state)==0
    for name,p in model.network.named_parameters():equal(p,before[name])


def test_two_class_native_ranking_uses_source_positive_score(tmp_path):
    data=create_smmile_demo(tmp_path/'data','multiclass',classes=2,dimension=4)
    model=SMMILe(Bundle(data),warmup_epochs=1)
    frame=pd.DataFrame({'target':[0,0,1,1],'raw_p_0':[.1,.2,.9,.8],'raw_p_1':[.2,.3,.9,.8],
                        'smmile_bag_bce':[.5]*4})
    assert model.validation_metrics(frame)['smmile_raw_slide_macro_auc']==1.


@pytest.mark.parametrize('variant',['binary','multiclass','multilabel'])
@pytest.mark.parametrize('size',['small','big'])
def test_original_constructor_initialization_and_rng(variant,size):
    from milplus.smmile_kernels import author_network
    scope=reference(variant)
    cls=scope['SMMILe_SINGLE'] if variant=='binary' else scope['SMMILe']
    classes=2 if variant=='binary' else 3
    torch.manual_seed(2026)
    original=cls(fea_dim=4,n_classes=classes,size_arg=size)
    expected_rng=torch.get_rng_state()
    torch.manual_seed(2026)
    actual=author_network(variant,dimension=4,classes=classes,size=size)
    equal(original.state_dict(),actual.state_dict())
    equal(expected_rng,torch.get_rng_state())
    if variant=='binary':assert actual.n_classes==2 and actual.cls_net.out_features==1


def test_original_epoch0_reset_can_discard_first_refinement_best(tmp_path):
    root,sha=verified_source('smmile')
    source=subprocess.check_output(['git','show',sha+':single/utils/core_utils.py'],cwd=root,text=True).replace('np.Inf','np.inf')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='EarlyStopping')
    scope={'torch':torch,'np':np,'print':lambda *args,**kwargs:None}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'source/EarlyStopping','exec'),scope)  # noqa: S102 -- independent original control policy, NumPy compatibility only
    cls=scope['EarlyStopping'];model=nn.Linear(1,1)
    first=cls(patience=20,stop_epoch=50);first(0,1.,model,str(tmp_path/'first.pt'))
    reset=cls(patience=50,stop_epoch=100);reset(1,2.,model,str(tmp_path/'reset.pt'))
    assert reset.best_score==-2. and first.best_score==-1.
    # PULS keeps all validation-scored refinement candidates under one best tracker.
    assert not SMMILe.selection_improved(None,-2.,-1.)


@pytest.mark.parametrize('bad',[{'warmup_epochs':0},{'groups':False},{'drop_times':0},{'inst_rate':0},
    {'inst_rate':float('nan')},{'tau':-1},{'tau':float('nan')},{'mrf':'true'},
    {'weighted_sampling':1},{'bag_loss':'ce'},{'refs':0},{'size':'unknown'}])
def test_invalid_recipe_options_rejected(tmp_path,bad):
    _model,bundle=fixture(tmp_path,'smmile')
    with pytest.raises(ValueError):SMMILe(bundle,**bad)


def test_cuda_checker_records_not_run_with_devices_disabled(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/validate_smmile_cuda.py'
    subprocess.run([sys.executable,str(script),'--output',str(tmp_path/'cuda')],check=True,
                   env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
    result=json.loads((tmp_path/'cuda/validation.json').read_text())
    assert result['state']=='not_run' and 'unavailable' in result['reason']

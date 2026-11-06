"""Actual full AGDR source train/validation loops, native gradients and recovery."""
import ast
import copy

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn import metrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from milplus.agdrmil import AGDRMIL, COMMIT, PRESETS
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, predict, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json, write_json
from milplus.registry import require_runnable
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def data(path,classes=2,dim=1024):
    return Bundle(create_demo(path,patients=10*classes,classes=classes,dimension=dim))


def original():
    root,sha=verified_source('agdrmil');assert sha==COMMIT
    scope={'__name__':__name__,'torch':torch,'nn':nn,'F':F,'np':np}
    scope.update({n:getattr(metrics,n) for n in ['accuracy_score','precision_score','recall_score','f1_score','confusion_matrix','roc_auc_score']})
    def definitions(path,names):
        nodes=[n for n in ast.parse(pinned_text(root,sha,path)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'independent_actual_agdrmil.py','exec'),scope)  # noqa: S102 -- exact named fixed-source definitions, no numerical rewrites
    definitions('utils/variate.py',['get_act','get_optimizer','get_criterion'])
    definitions('model.py',['initialize_weights','att_split','CosineSimilarityWeighted','RowMask','mil_aggregation','AGDR_MIL'])
    definitions('utils/utils.py',['train_one_epoch_loss','val_one_epoch_loss','Earlystopping'])
    return scope


def reference(model,scope):
    hi,lo,mask,_=PRESETS[model.profile]
    result=scope['AGDR_MIL'](len(model.bundle.classes),.25,'relu',model.bundle.dim,hi,lo,mask,model.high_pool,'attention')
    result.load_state_dict(model.network.state_dict());return result


def exact(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def snapshot(model,optimizer):
    return {'weights':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
        'gradients':{n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()},'rng':rng_state()}


def loader(bundle,subset,generator=None):
    rows=bundle.records(subset)
    class Bags(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):return bundle.load(rows[i])[0],rows[i]['target']
    return DataLoader(Bags(),batch_size=1,shuffle=subset=='train',generator=generator)


@pytest.mark.parametrize('profile',list(PRESETS))
@pytest.mark.parametrize('classes',[2,3])
def test_actual_author_full_trainer_every_update(tmp_path,profile,classes):
    bundle=data(tmp_path/'data',classes);seed_everything(2026)
    model=AGDRMIL(bundle,profile=profile);scope=original();ref=reference(model,scope);traces=[]
    initial={n:p.detach().clone() for n,p in model.network.named_parameters() if n.startswith('att.')}
    class TracedAdam(torch.optim.Adam):
        def step(self,closure=None):
            result=super().step(closure);traces.append(snapshot(ref,self));return result
    ref_opt=TracedAdam(ref.parameters(),lr=2e-4,weight_decay=1e-5)
    optimizer=model.make_optimizer(2e-4,1e-5);generator=torch.Generator().manual_seed(2000)
    source_loader=loader(bundle,'train',generator);checked=0
    for epoch in range(2):
        before=rng_state();traces.clear()
        loss=scope['train_one_epoch_loss'](ref,source_loader,nn.CrossEntropyLoss(),ref_opt,torch.device('cpu'),model.loss_weight)
        expected=list(traces);restore_rng(before);step=optimizer.step
        def checked_step(*args,_step=step,_expected=expected,**kwargs):
            nonlocal checked
            result=_step(*args,**kwargs);exact(snapshot(model.network,optimizer),_expected.pop(0));checked+=1;return result
        optimizer.step=checked_step;losses=model.epoch(bundle,optimizer);optimizer.step=step
        assert not expected and loss==sum(x['loss'] for x in losses)/len(losses)
        exact(generator.get_state(),model.generator.get_state());model.finish_epoch(bundle,'cpu',{})
    assert checked==2*len(bundle.records('train'))
    for name,value in initial.items():exact(model.network.state_dict()[name],value)
    assert all(p.grad is None for n,p in model.network.named_parameters() if n.startswith('att.'))
    write_json(tmp_path/'agdrmil_author_training.json',{'state':'passed','source_commit':COMMIT,'profile':profile,
        'classes':classes,'input_dimension':1024,'hidden':512,'epochs':2,'optimizer_updates':checked,
        'comparison':'Actual original train_one_epoch_loss, native1024/512, all parameters/gradients/Adam moments/all RNG after every update and private DataLoader generator after every epoch are bitwise equal. Untrained hard grader unchanged.'})


@pytest.mark.parametrize('profile',list(PRESETS))
@pytest.mark.parametrize('classes',[2,3])
def test_actual_source_validation_loss_outputs_and_sequential_loader_rng(tmp_path,profile,classes):
    bundle=data(tmp_path/'data',classes);seed_everything(2026)
    model=AGDRMIL(bundle,profile=profile);scope=original();ref=reference(model,scope);captured=[]
    ref.register_forward_hook(lambda module,inputs,out:captured.append(tuple(v.detach().clone() for v in out)))
    before=rng_state()
    expected=scope['val_one_epoch_loss'](ref,loader(bundle,'val'),nn.CrossEntropyLoss(),torch.device('cpu'),classes,model.loss_weight)
    expected_rng=rng_state();restore_rng(before);model.eval();rows=bundle.records('val');index=0
    original_forward=model.forward_unit
    def checked_forward(*args,**kwargs):
        nonlocal index
        value=original_forward(*args,**kwargs);high,all_bag=captured[index];index+=1
        exact(value['high_logits'],high);exact(value['logits'],all_bag);exact(value['probabilities'],all_bag.softmax(1));return value
    model.forward_unit=checked_forward
    frame=predict(model,bundle,profile,'val','cpu',{'max_instances':None})
    assert index==len(rows);exact(rng_state(),expected_rng)
    actual=model.validation_metrics(frame)
    assert actual['agdr_val_loss']==expected[0] and actual['agdr_native_slide_accuracy']==expected[1]
    assert actual['agdr_native_slide_auc']==expected[2]
    for (high,all_bag),row in zip(captured,frame.to_dict('records')):
        exact(high[0].tolist(),[row[f'agdr_high_sigmoid_{i}'] for i in range(classes)])
        exact(all_bag[0].tolist(),[row[f'agdr_full_sigmoid_{i}'] for i in range(classes)])
    write_json(tmp_path/'agdrmil_author_validation.json',{'state':'passed','profile':profile,'classes':classes,
        'slides':len(rows),'native_loss':expected[0],'native_auc':expected[2],
        'comparison':'Actual original val_one_epoch_loss and sequential DataLoader; all native raw/probability outputs, weighted dual loss, native slide accuracy/AUC and full RNG exact, including the CPU base-seed draw and eval RowMask.'})


@pytest.mark.parametrize('classes',[2,3])
@pytest.mark.parametrize('dimension',[16,1024])
@pytest.mark.parametrize('pool',['max','mean','attention','gate'])
def test_complete_network_all_source_gradients_and_high_pool_variants(tmp_path,classes,dimension,pool):
    bundle=data(tmp_path/'data',classes,dimension);seed_everything(77)
    model=AGDRMIL(bundle,profile='agdrmil_tcga_source',high_pool=pool);ref=reference(model,original())
    x=torch.randn(31,dimension,requires_grad=True);other=x.detach().clone().requires_grad_();target=torch.tensor([classes-1])
    before=rng_state();actual=model(x);after=rng_state();restore_rng(before);high,all_bag=ref(other);exact(rng_state(),after)
    exact(actual['high_logits'],high);exact(actual['logits'],all_bag)
    expected=.4*F.cross_entropy(high,target)+.6*F.cross_entropy(all_bag,target)
    loss=model.objective(actual,target);exact(loss,expected);loss.backward();expected.backward();exact(x.grad,other.grad)
    for (name,p),(other_name,q) in zip(model.network.named_parameters(),ref.named_parameters()):
        assert name==other_name;exact(p.grad,q.grad)
    assert all(p.grad is None for n,p in model.network.named_parameters() if n.startswith('att.'))
    assert all(p.grad is not None for n,p in model.network.named_parameters() if n.startswith(('classifier_1.','classifier_2.','reduce_dim.')))


@pytest.mark.parametrize('profile',list(PRESETS))
@pytest.mark.parametrize('classes',[2,3])
@pytest.mark.parametrize('boundary',[1,2])
def test_actual_engine_exact_resume_selection_raw_scores_and_structural_export(tmp_path,profile,classes,boundary):
    bundle=data(tmp_path/'data',classes,32);cfg={'epochs':3,'threads':1}
    full=train(bundle.root,profile,tmp_path/'full',config=cfg)
    part=train(bundle.root,profile,tmp_path/'part',config=cfg,stop_after=boundary)
    train(bundle.root,profile,part,config=cfg,resume=True)
    a,b=[torch.load(p/'last.pt',weights_only=False) for p in [full,part]]
    for key in ['model','optimizers','history','rng','best','bad']:exact(a[key],b[key])
    assert a['model']['_extra_state']['optimizer_steps']==3*len(bundle.records('train'))
    losses=[h['validation']['agdr_val_loss'] for h in a['history']]
    assert torch.load(full/'best.pt',weights_only=False)['epoch']==max(i for i,v in enumerate(losses) if v==min(losses))
    assert not (full/'test_predictions.csv').exists()
    assert evaluate(full)==evaluate(part)
    frame=pd.read_csv(full/'test_predictions.csv');pd.testing.assert_frame_equal(frame,pd.read_csv(part/'test_predictions.csv'))
    sha=digest_file(full/'test_predictions.csv');evaluate(full);assert sha==digest_file(full/'test_predictions.csv')
    for entry in read_json(full/'test_instances/manifest.json')['records']:
        with h5py.File(entry['path']) as h:
            order=h['retained_order'][:];attention=h['attention'][:];grades=h['grades'][:]
            assert h['coords'].shape==(entry['instances'],2) and len(np.unique(order))==len(order)
            np.testing.assert_array_equal(h['partition'][:],np.where(grades>=PRESETS[profile][0],2,np.where(grades>=PRESETS[profile][1],1,0)))
            np.testing.assert_allclose(attention.sum(),1.,rtol=0,atol=1e-6)
            assert (attention[np.setdiff1d(np.arange(entry['instances']),order)]==0).all()
    assert read_json(full/'method_protocol.json')['completed_epochs']==3


@pytest.mark.parametrize('kind',['empty','one','constant','nan','infinite','dimension','batch'])
def test_undefined_inputs_fail_instead_of_fabricating_grading(tmp_path,kind):
    bundle=data(tmp_path/'data',dim=16);model=AGDRMIL(bundle).eval();x=torch.randn(9,16)
    if kind=='empty':x=x[:0]
    elif kind=='one':x=x[:1]
    elif kind=='constant':x[:]=1
    elif kind=='nan':x[0,0]=float('nan')
    elif kind=='infinite':x[0,0]=float('inf')
    elif kind=='dimension':x=x[:,:15]
    else:x=x.unsqueeze(0)
    with pytest.raises(ValueError):model(x)


@pytest.mark.parametrize('change',['profile','pool','dropout','sampler_shape','sampler_dtype'])
def test_checkpoint_contract_and_private_sampler_state_cannot_be_changed(tmp_path,change):
    bundle=data(tmp_path/'data',dim=16);model=AGDRMIL(bundle);state=copy.deepcopy(model.state_dict());kwargs={}
    if change=='profile':kwargs['profile']='agdrmil_tcga_source'
    elif change=='pool':kwargs['high_pool']='attention'
    elif change=='dropout':kwargs['dropout']=0.
    elif change=='sampler_shape':state['_extra_state']['sampler_generator']=torch.zeros(1,dtype=torch.uint8)
    else:state['_extra_state']['sampler_generator']=state['_extra_state']['sampler_generator'].float()
    other=AGDRMIL(bundle,**kwargs)
    with pytest.raises(ValueError):other.load_state_dict(state)


@pytest.mark.parametrize('change',['max_instances','grad_clip','selection_metric'])
def test_native_recipe_rejects_silent_generic_training_changes(tmp_path,change):
    model=AGDRMIL(data(tmp_path/'data',dim=16))
    cfg={'max_instances':None,'grad_clip':None,'selection_metric':'agdr_val_loss','epochs':3}
    cfg[change]={'max_instances':16,'grad_clip':5.,'selection_metric':'macro_auc'}[change]
    with pytest.raises(ValueError):model.configure_training(cfg)


def test_original_early_stop_ties_and_first_eligible_stop_epoch(tmp_path):
    model=AGDRMIL(data(tmp_path/'data',dim=16));callback=original()['Earlystopping'];checks=[]
    for values,expected_best,expected_stop in [([.7]+[.9]*30,0,21),([.8,.7,.7]+[.9]*30,2,22)]:
        ref=callback(patience=20,stop_epoch=20,verbose=False)
        dummy=nn.Linear(1,1);best=-float('inf');bad=0;last_best=None;stop=None
        for epoch,loss in enumerate(values):
            ref(epoch,loss,dummy,ckpt_name=tmp_path/'source.pt')
            if model.selection_improved(-loss,best):best,bad,last_best=-loss,0,epoch
            else:bad+=1
            actual=bad>=20 and epoch+1>model.stop_after_epoch
            assert actual==ref.early_stop
            assert -best==ref.best_score*-1 and bad==ref.counter
            if actual:stop=epoch;break
        assert last_best==expected_best and stop==expected_stop
        checks.append({'last_tied_best_epoch':last_best,'first_actual_stop_epoch':stop})
    write_json(tmp_path/'agdrmil_author_selection.json',{'state':'passed','checks':checks,
        'comparison':'Actual released Earlystopping callback against public last-tie selector and engine eligibility convention, including forbidden epoch20 versus first eligible epoch21; no numerical rewrite of callback.'})


def test_test_model_inference_stays_sealed_until_evaluate(tmp_path,monkeypatch):
    bundle=data(tmp_path/'data',dim=16);test={r['slide_id'] for r in bundle.records('test')};forward=AGDRMIL.forward_unit
    def guarded(self,bundle,rows,*args,**kwargs):
        if any(row['slide_id'] in test for row in rows):raise AssertionError('TEST reached model during training/selection')
        return forward(self,bundle,rows,*args,**kwargs)
    monkeypatch.setattr(AGDRMIL,'forward_unit',guarded)
    run=train(bundle.root,'AGDR-MIL',tmp_path/'run',config={'epochs':2,'threads':1})
    assert not (run/'test_predictions.csv').exists()
    monkeypatch.setattr(AGDRMIL,'forward_unit',forward);evaluate(run)
    assert set(pd.read_csv(run/'test_predictions.csv').slide_id)==test


def test_real_cli_and_undefined_native_auc_are_explicit(tmp_path):
    from milplus.cli import main
    bundle=data(tmp_path/'data',dim=16);assert require_runnable('AGDR-MIL')['id']=='agdrmil_yaml_source'
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'epochs':2,'threads':1}))
    assert main(['run','--data',str(bundle.root),'--method','AGDR-MIL','--config',str(config),'--output',str(tmp_path/'run')])==0
    assert main(['evaluate','--run',str(tmp_path/'run')])==0
    frame=pd.read_csv(tmp_path/'run/test_predictions.csv');model=AGDRMIL(bundle)
    frame.target=0
    assert model.validation_metrics(frame)['agdr_native_slide_auc'] is None

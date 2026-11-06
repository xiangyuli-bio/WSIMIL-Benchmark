"""Complete original trainer/selection/RNG and real public recovery checks."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.engine import evaluate,seed_everything,train
from milplus.gmmamba import (GMMambaMIL,SourceSelection,author_training,
                             deferred_test_rng_port,load_bag)
from milplus.gmmamba_demo import create_gmmamba_demo
from milplus.gmmamba_source import checked_source
from milplus.io import digest_file
from milplus.registry import inspect_method
from milplus.upstream import pinned_text


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    torch.set_num_threads(2)
    if os.environ.get('MILPLUS_GMMAMBA_TEST_BUNDLE'):
        path=Path(os.environ['MILPLUS_GMMAMBA_TEST_BUNDLE'])
        bundle=Bundle(path)
        assert bundle.config['synthetic'] is True and bundle.dim==512 and len(bundle.rows)==6
        assert all(bundle.load(row,require_coords=True)[0].shape==(257,512) for row in bundle.rows.to_dict('records'))
        return path
    return create_gmmamba_demo(tmp_path_factory.mktemp('gmmamba_training')/'data',patients=6,patches=257)


def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for name in a:equal(a[name],b[name])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_two_full_epochs_match_original_loop_weights_AdamW_RNG_and_complete_groups(data):
    seed_everything(8606);owner=GMMambaMIL(Bundle(data));reference=copy.deepcopy(owner.network)
    cfg={'epochs':7,'lr':1e-4,'weight_decay':1e-5,'patience':20,'threads':2,
         'selection_metric':'gmmamba_val_auc_f1','grad_clip':None,'max_instances':None}
    owner.configure_training(cfg);actual=owner.make_optimizer(1e-4,1e-5)
    parameters=list(reference.imm.parameters())+list(reference.classifier.parameters())+list(reference.css.parameters())
    expected=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-5)
    source=author_training();args=SimpleNamespace(n_classes=2,num_subbags=10)
    start=torch.get_rng_state()
    for epoch in range(2):
        owner.start_epoch(epoch);owner.epoch(owner.bundle,actual,torch.device('cpu'))
        frame=owner.predict_subset(owner.bundle,'val',torch.device('cpu'),cfg)
        owner.selection_objective(owner.validation_metrics(frame),epoch,0.)
    end=torch.get_rng_state();torch.set_rng_state(start)
    for epoch in range(2):
        source.train_function(args,reference.imm,reference.css,reference.classifier,reference.memory,
            epoch,owner.loader('train')[2],torch.device('cpu'),torch.nn.CrossEntropyLoss(),expected)
        source.val_function(args,reference.imm,reference.css,reference.classifier,reference.memory,
            owner.loader('val')[2],torch.device('cpu'),epoch)
    equal(owner.network.state_dict(),reference.state_dict());equal(actual.state_dict(),expected.state_dict())
    assert torch.equal(end,torch.get_rng_state());assert owner.optimizer_updates==4
    assert owner.train_orders and owner.val_orders and not owner.network.memory.msg_states
    owner.validate_optimizer_state()


@pytest.mark.parametrize('scenario',['changing','zero'])
def test_selection_and_termination_match_literal_entire_author_epoch_loop(scenario):
    root,sha=checked_source();tree=ast.parse(pinned_text(root,sha,'main.py'))
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train_main')
    initial=[x for x in function.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'best_state','patience','counter'} for t in x.targets)]
    loop=next(x for x in function.body if isinstance(x,ast.For))
    events=[];model=SimpleNamespace(state_dict=lambda:{})
    series=[]
    for epoch in range(100):
        score=0. if scenario=='zero' else .6 if epoch<10 else .7 if epoch<49 else .9
        series.append((score/2,0.,score/2,0.,0.))
    def training(*args):return (0.,0.,0.,0.,0.)
    def validation(*args,**kwargs):
        return (0.,0.,0.,0.,0.) if kwargs.get('currenttype')=='test' else series[args[-1]]
    scope={'args':SimpleNamespace(epoch=100,save_dir='/unused_control_oracle'),
        'basedmodel':model,'classifymodel':model,'super_token_model':model,'memory':None,
        'train_loader':None,'valid_loader':None,'test_loader':None,'device':None,
        'classify_loss_fn':None,'optimizer_classify_loss':None,
        'train_function':training,'val_function':validation,
        'torch':SimpleNamespace(save=lambda record,path:events.append(record['epoch']))}
    exec(compile(ast.Module(body=initial+[loop],type_ignores=[]),str(root/'main.py'),'exec'),scope)
    selection=SourceSelection();observed=[]
    for epoch,(auc,_,f1,_,_) in enumerate(series):
        selection.observe(epoch,auc,f1)
        if selection.improved:observed.append(epoch)
        if selection.done:break
    assert observed==events and selection.counter==scope['counter'] and selection.epoch==scope['epoch']
    assert selection.best==scope['best_state']['val_auc']+scope['best_state']['val_f1']
    assert selection.selected_epoch==(events[-1] if events else None)


def test_deferred_TEST_RNG_port_matches_actual_literal_HDF5_dataset_source_inference(data,tmp_path):
    root,_=checked_source();bundle=Bundle(data);owner=GMMambaMIL(bundle)
    rows=bundle.records('test')
    table=tmp_path/'author_test.csv';pd.DataFrame({'test':[x['slide_id'] for x in rows],
        'test_label':[x['target'] for x in rows]}).to_csv(table,index=False)
    spec=importlib.util.spec_from_file_location('_gmmamba_literal_dataset',root/'datasets/toloda_datasets.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dataset=module.h5file_Dataset(table,data/'features','test')
    loader=DataLoader(dataset,batch_size=1,num_workers=2,pin_memory=True,shuffle=False)
    before=torch.get_rng_state();numpy_state=np.random.get_state();python_state=random.getstate()
    states={name:value.clone() for name,value in owner.network.state_dict().items()}
    author_training().val_function(SimpleNamespace(n_classes=2,num_subbags=10),owner.network.imm,
        owner.network.css,owner.network.classifier,owner.network.memory,loader,torch.device('cpu'),6,currenttype='test')
    after=torch.get_rng_state();torch.set_rng_state(before);deferred_test_rng_port()
    assert torch.equal(after,torch.get_rng_state());assert python_state==random.getstate()
    now=np.random.get_state();assert numpy_state[0]==now[0] and np.array_equal(numpy_state[1],now[1]) and numpy_state[2:]==now[2:]
    equal(states,owner.network.state_dict())


def test_original_integer_coordinate_dtype_and_low_bits_are_preserved(tmp_path):
    data=create_gmmamba_demo(tmp_path/'data',patients=6)
    bundle=Bundle(data);row=bundle.records('train')[0];path=data/row['path']
    with h5py.File(path,'r+') as archive:
        coords=archive['coords'][:];coords[:,0]+=2**25+3;archive['coords'][:]=coords
    _,actual=load_bag(bundle,row)
    assert actual.dtype==torch.int64 and np.array_equal(actual.numpy(),coords)


def test_actual_uninterrupted_vs_resume_complete_state_and_independent_selected_reloads(data,tmp_path):
    cfg={'epochs':7,'threads':2};method='gmmamba_mambamil_bimamba_source'
    full=train(data,method,tmp_path/'full',seed=2026,config=cfg)
    partial=train(data,method,tmp_path/'partial',seed=2026,config=cfg,stop_after=3)
    assert json.loads((partial/'status.json').read_text())['state']=='interrupted'
    train(data,method,partial,seed=2026,config=cfg,resume=True)
    a=torch.load(full/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(partial/'last.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizers','history','best','bad']:equal(a[key],b[key])
    assert torch.equal(a['rng']['torch'],b['rng']['torch']) and a['rng']['python']==b['rng']['python']
    state=b['model']['_extra_state'];assert state['completed_epochs']==7 and state['optimizer_updates']==14
    assert state['selection']['selected_epoch']==6 and state['deferred_test_rng_epochs']==[6]
    assert not (partial/'test_predictions.csv').exists()
    del a,b
    metrics=evaluate(partial);prediction=digest_file(partial/'test_predictions.csv')
    assert evaluate(partial)==metrics and digest_file(partial/'test_predictions.csv')==prediction


def test_method_alias_and_source_recipe_cannot_silently_subsample_or_use_generic_CE(data):
    assert inspect_method('GMMamba')['id']=='gmmamba_mambamil_bimamba_source'
    model=GMMambaMIL(Bundle(data));cfg={'epochs':7,'lr':1e-4,'weight_decay':1e-5,
        'patience':20,'threads':2,'selection_metric':'gmmamba_val_auc_f1','grad_clip':None,'max_instances':None}
    for key,value in [('max_instances',32),('grad_clip',5.),('epochs',2),('lr',1e-3),('selection_metric','macro_auc')]:
        with pytest.raises(ValueError):model.configure_training({**cfg,key:value})

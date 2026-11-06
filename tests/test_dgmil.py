"""Independent DGMIL density/selection/trainer evidence and patient ranking behavior."""
import ast
import copy
import heapq
import json

import faiss
import h5py
import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm
import torch
import yaml
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.dgmil import DGMIL, density_scores, dgmil_epoch, fit_density
from milplus.engine import evaluate, restore_rng, rng_state, train
from milplus.metrics import classification_metrics, paired_bootstrap
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1);faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):yield


def reference(dim):
    root,sha=verified_source('dgmil')
    scope={'np':np,'torch':torch,'nn':nn,'F':F,'faiss':faiss,'heapq':heapq,
           'train_test_split':train_test_split,'skm':skm,'print_':lambda *a:None}
    selections={'model.py':{'Linear_projection_MAE'},
                'utils.py':{'get_scores_one_cluster'},
                'pos_score_calculator.py':{'get_clusters','get_scores','get_scores_multi_cluster','get_eval_results_dis_feats','get_score_and_dis_feats'},
                'preprocessor.py':{'sort_and_generate_feats_bank_forpos_dynamic','sort_and_generate_feats_bank_forneg_dynamic','main_generator_dynamic'},
                'dynamic_MAE_fintuning_addslidelevel.py':{'model_train'}}
    for path,names in selections.items():
        code=pinned_text(root,sha,path)
        if path=='model.py':code=code.replace('768',str(dim))
        tree=ast.parse(code)
        defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=defs,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independently read locked definitions
    return scope


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def bundle_at(path,dim=16):
    return Bundle(create_demo(path,patients=20,dimension=dim))


@pytest.mark.parametrize('dim,clusters',[(7,1),(16,3),(23,2),(768,1)])
def test_cached_density_matches_complete_author_formula(dim,clusters):
    rng=np.random.default_rng(48)
    negative=rng.normal(size=(220,dim)).astype('float32')
    positive=rng.normal(1,.5,size=(75,dim)).astype('float32')
    before=rng_state();state=fit_density(negative,clusters)
    actual=[density_scores(state,v,block_size=23) for v in (negative,positive)]
    restore_rng(before)
    expected=reference(dim)['get_score_and_dis_feats'](clusters,2026,negative.copy(),positive.copy())
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
    assert sum(state['counts'])==len(negative)


@pytest.mark.parametrize('profile',['dgmil','dgmil_source_extremes'])
@pytest.mark.parametrize('dim,clusters',[(16,1),(23,2)])
def test_three_updates_match_independent_author_training_path(tmp_path,profile,dim,clusters):
    bundle=bundle_at(tmp_path/'data',dim)
    model=DGMIL(bundle,profile=profile,clusters=clusters)
    ref=reference(dim);network=ref['Linear_projection_MAE']();network.load_state_dict(model.network.state_dict())
    optimizer=model.make_optimizer(.01,0.)
    original_optimizer=torch.optim.Adam(network.parameters(),lr=.01)
    banks=model.training_banks(bundle);arrays=[x.numpy().copy() for x in banks]
    state=rng_state();model.prepare(bundle,2026,'cpu');restore_rng(state)
    distance=ref['get_score_and_dis_feats'](clusters,2026,*[x.copy() for x in arrays])
    expected=ref['main_generator_dynamic'](distance[0],arrays[0],distance[1],arrays[1])
    for epoch in range(3):
        x,_,y,_=train_test_split(*expected,test_size=.1,random_state=12345,shuffle=True)
        actual_x,actual_y=model.selected_training_batch(bundle,'cpu')
        equal(actual_x,torch.from_numpy(x));equal(actual_y,torch.from_numpy(y))
        before=rng_state()
        expected_loss=ref['model_train'](torch.from_numpy(x),torch.from_numpy(y),network,nn.CrossEntropyLoss(),original_optimizer)
        restore_rng(before);actual=dgmil_epoch(model,bundle,[],optimizer,'cpu',{})
        assert actual[0]['loss']==expected_loss
        equal(model.network.state_dict(),network.state_dict())
        for p,q in zip(model.network.parameters(),network.parameters()):equal(p.grad,q.grad)
        equal(optimizer.state_dict(),original_optimizer.state_dict())
        before=rng_state();model.finish_epoch(bundle,'cpu',{})
        projected=[network.projection_head(torch.from_numpy(x)).detach().numpy() for x in arrays]
        restore_rng(before)
        distance=ref['get_score_and_dis_feats'](clusters,2026,*[x.copy() for x in projected])
        expected=ref['main_generator_dynamic'](distance[0],arrays[0],distance[1],arrays[1])
        assert model.optimizer_steps==epoch+1 and model.iteration==epoch+1
        for x,d in zip(projected,distance):np.testing.assert_allclose(density_scores(model.density,x),d,rtol=1e-7,atol=1e-7)


@pytest.mark.parametrize('profile',['dgmil','dgmil_source_extremes'])
def test_resume_is_exact_and_restored_predictions_have_raw_scores(tmp_path,profile):
    bundle=bundle_at(tmp_path/'data')
    config={'epochs':3,'threads':1,'model':{'clusters':2}}
    train(bundle.root,profile,tmp_path/'full',config=config)
    train(bundle.root,profile,tmp_path/'resume',config=config,stop_after=1)
    train(bundle.root,profile,tmp_path/'resume',config=config,resume=True)
    a,b=[torch.load(tmp_path/p/'last.pt',weights_only=False) for p in ('full','resume')]
    for k in ('model','optimizers','history','best'):equal(a[k],b[k])
    assert json.loads((tmp_path/'full/method_protocol.json').read_text())['optimizer_steps']==3
    assert evaluate(tmp_path/'full')==evaluate(tmp_path/'resume')
    left=pd.read_csv(tmp_path/'full/test_predictions.csv');right=pd.read_csv(tmp_path/'resume/test_predictions.csv')
    pd.testing.assert_frame_equal(left,right)
    assert 'ranking_score' in left and left.ranking_score.notna().all()
    first=(tmp_path/'full/test_predictions.csv').read_bytes();evaluate(tmp_path/'full')
    assert first==(tmp_path/'full/test_predictions.csv').read_bytes()


def test_native_network_gradients_and_one_update(tmp_path):
    bundle=bundle_at(tmp_path/'data',768)
    model=DGMIL(bundle,clusters=1);ref=reference(768)['Linear_projection_MAE']()
    ref.load_state_dict(model.network.state_dict())
    x=torch.randn(11,768);target=torch.arange(11)%2
    optim1=model.make_optimizer(.01,0.);optim2=torch.optim.Adam(ref.parameters(),lr=.01)
    for net,opt in [(model.network,optim1),(ref,optim2)]:
        opt.zero_grad();F.cross_entropy(net(x),target).backward();opt.step()
    equal(model.network.state_dict(),ref.state_dict());equal(optim1.state_dict(),optim2.state_dict())


def test_repaired_ties_have_unique_indices_and_source_repeats_first(tmp_path,monkeypatch):
    bundle=bundle_at(tmp_path/'data')
    import milplus.dgmil as module
    monkeypatch.setattr(module,'density_scores',lambda state,x:np.ones(len(x)))
    for profile in ('dgmil','dgmil_source_extremes'):
        model=DGMIL(bundle,profile,clusters=1);model.prepare(bundle,2026,'cpu')
        for indices in model.selected:
            if profile=='dgmil':np.testing.assert_array_equal(indices,np.arange(len(indices)))
            else:assert not indices.any()


def test_only_training_features_fit_density_selection_and_probability_mapping(tmp_path,monkeypatch):
    bundle=bundle_at(tmp_path/'data');model=DGMIL(bundle,clusters=1)
    original=bundle.load;seen=[]
    def restricted(row,*a,**kw):
        assert row['subset']=='train';seen.append(row['slide_id']);return original(row,*a,**kw)
    monkeypatch.setattr(bundle,'load',restricted)
    model.prepare(bundle,2026,'cpu')
    optimizer=model.make_optimizer(.01,0.)
    dgmil_epoch(model,bundle,[],optimizer,'cpu',{})
    model.finish_epoch(bundle,'cpu',{})
    assert set(seen)=={r['slide_id'] for r in bundle.records('train')}
    assert {r['global_patient_id'] for r in model.fit_info['training_inventory']}=={r['global_patient_id'] for r in bundle.records('train')}


def test_prediction_is_permutation_invariant_rng_pure_and_does_not_refit(tmp_path,monkeypatch):
    bundle=bundle_at(tmp_path/'data');model=DGMIL(bundle,clusters=1)
    model.prepare(bundle,2026,'cpu');model.finish_epoch(bundle,'cpu',{})
    x,_=bundle.load(bundle.records('test')[0]);perm=x.flip(0)
    before=rng_state();weights=copy.deepcopy(model.state_dict())
    def forbidden(*a,**kw):raise AssertionError('inference must not refit density')
    monkeypatch.setattr(model,'refit',forbidden)
    a=model(x);b=model(perm)
    torch.testing.assert_close(a['ranking_score'],b['ranking_score'],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(a['instance_scores'],b['instance_scores'].flip(0),rtol=0,atol=0)
    equal(before,rng_state());equal(weights,model.state_dict())


def test_projection_excludes_bn_relu_and_classifier(tmp_path):
    bundle=bundle_at(tmp_path/'data');model=DGMIL(bundle,clusters=1,projection_batch=7)
    x=torch.randn(23,16)
    torch.testing.assert_close(model.project(x),model.network.projection_head(x),rtol=1e-6,atol=1e-6)
    with torch.no_grad():model.network.bn.bias.fill_(100);model.network.class_layer.weight.fill_(100)
    torch.testing.assert_close(model.project(x),model.network.projection_head(x),rtol=1e-6,atol=1e-6)


def test_raw_score_controls_auc_pr_and_patient_bootstrap():
    frame=pd.DataFrame({'slide_id':['a','b','c','d'],'global_patient_id':['p0','p1','p2','p3'],
                        'target':[0,0,1,1],'subset':'test','p_0':[.01,.1,.8,.9],
                        'p_1':[.99,.9,.2,.1],'ranking_score':[0.,1.,2.,3.]})
    actual=classification_metrics(frame,2,'patient')
    assert actual['macro_auc']==1. and actual['auprc']==1. and actual['accuracy']==0.
    reverse=frame.copy();reverse.ranking_score=-reverse.ranking_score
    result=paired_bootstrap(frame,reverse,2,repetitions=40,unit='patient')
    assert result['difference_left_minus_right']==1.
    expanded=pd.concat([frame,frame.assign(slide_id=['e','f','g','h'],ranking_score=[1.,0.,4.,3.])])
    assert classification_metrics(expanded,2,'patient')['macro_auc']==1.
    expanded.loc[0,'ranking_score']=np.nan
    with pytest.raises(ValueError,match='finite binary'):classification_metrics(expanded,2,'patient')


@pytest.mark.parametrize('damage',['feature','split','fm'])
def test_changed_inputs_block_resume(tmp_path,damage):
    bundle=bundle_at(tmp_path/'data');config={'epochs':2,'threads':1,'model':{'clusters':1}}
    train(bundle.root,'dgmil',tmp_path/'run',config=config,stop_after=1)
    if damage=='feature':
        with h5py.File(bundle.root/bundle.records('train')[0]['path'],'r+') as f:f['features'][0]=f['features'][0]+1
    elif damage=='fm':
        cfg=yaml.safe_load((bundle.root/'task.yaml').read_text());cfg['encoders']['toy']['weights']='changed'
        (bundle.root/'task.yaml').write_text(yaml.safe_dump(cfg))
    else:
        path=bundle.root/'split.csv';frame=pd.read_csv(path);frame.loc[frame.subset=='val','subset']='test';frame.to_csv(path,index=False)
    with pytest.raises(ValueError):train(bundle.root,'dgmil',tmp_path/'run',config=config,resume=True)


@pytest.mark.parametrize('clusters',[0,-1,True,1.5])
def test_invalid_clusters_rejected(tmp_path,clusters):
    with pytest.raises(ValueError):DGMIL(bundle_at(tmp_path/'data'),clusters=clusters)


@pytest.mark.parametrize('array,clusters',[(np.ones((30,8),'float32'),2),(np.eye(5,dtype='float32'),5),
                                        (np.full((30,8),np.nan,'float32'),1)])
def test_degenerate_negative_density_rejected(array,clusters):
    with pytest.raises(ValueError):fit_density(array,clusters)


def test_explicit_binary_contract_and_no_hidden_patch_budget(tmp_path):
    data=create_demo(tmp_path/'multiclass',patients=30,classes=3)
    with pytest.raises(ValueError,match='binary'):DGMIL(Bundle(data))
    model=DGMIL(bundle_at(tmp_path/'data'),clusters=1)
    with pytest.raises(ValueError,match='all training patches'):model.configure_training({'max_instances':10})
    with pytest.raises(ValueError,match='finite nonempty'):model.project(torch.empty(0,16))

"""Original six-head TTIS math, complete optimization, asset integrity and evaluation."""
import ast
import json
import math
import subprocess
import types
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pytest
import torch
import yaml
from einops import rearrange, reduce, repeat
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.registry import check_compatibility, require_runnable
from milplus.ttis import PROFILES, TTISMIL, FrozenSemantic, ttis_epoch
from milplus.ttis_data import hierarchical_clusters, prepare_ttis, semantic_metadata, validate_clusters
from milplus.ttis_demo import create_ttis_demo
from milplus.ttis_source import HEADS
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def data(path, dimension=16, classes=2, secondary=False):
    return Bundle(create_ttis_demo(path, dimension, classes, secondary))


def options(profile, native=False):
    return {'hidden':512 if native or 'rrt' in profile else 32,'attention_dim':384 if native else 16}


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def raw(path,scope):
    root,sha=verified_source('ttis')
    text=subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)
    tree=ast.parse(text)
    defs=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    ns=dict(scope)
    exec(compile(ast.Module(body=defs,type_ignores=[]),'independent-author/'+path,'exec'),ns)  # noqa: S102 -- separately compiled original Git definitions
    return ns


def reference_network(kind,dim,classes,opts):
    from torch_geometric.nn import GlobalAttention, global_max_pool, global_mean_pool
    root,sha=verified_source('ttis')
    scope={'torch':torch,'nn':nn,'F':F,'Tensor':torch.Tensor,'np':np,'math':math,
           'rearrange':rearrange,'reduce':reduce,'repeat':repeat,'Optional':Optional,'Union':Union,
           'ABC':ABC,'abstractmethod':abstractmethod,'List':list,'defaultdict':defaultdict,
           'DropPath':None,'_c2r':torch.view_as_real,'_r2c':torch.view_as_complex,
           'global_max_pool':global_max_pool,'global_mean_pool':global_mean_pool,'GeoGlobalAttention':GlobalAttention}
    scope.update(raw('src/models/modules/mil_template.py',scope))
    scope.update(raw('src/models/modules/layers.py',scope))
    scope.update(raw('src/models/modules/nystrom_attention.py',{**scope,'ceil':math.ceil,'einsum':torch.einsum}))
    scope.update(raw('src/ttis/sampling.py',scope));scope.update(raw('src/ttis/aggregation.py',scope))
    head=HEADS[kind];scope=raw('src/models/'+head+'.py',scope)
    folder={'clam':'clamsb','rrt':'rrtmil','s4':'s4mil'}.get(kind,kind)
    cfg=yaml.safe_load(subprocess.check_output(['git','show',sha+f':configs/models/{folder}/uni_v1/brca.yaml'],cwd=root,text=True))['Model']
    import inspect
    cfg.update(in_dim=dim,num_classes=classes,embed_dim=opts['hidden'])
    if kind=='rrt':cfg['mlp_dim']=opts['hidden']
    if kind=='abmil':cfg['attn_dim']=opts['attention_dim']
    if kind=='clam':cfg['attention_dim']=opts['attention_dim']
    cfg={k:v for k,v in cfg.items() if k in inspect.signature(scope[head]).parameters}
    return scope[head](**cfg)


def reference_optimizer(model):
    scope={'torch':torch,'Optimizer':torch.optim.Optimizer,'OrderedDict':OrderedDict,
           'defaultdict':defaultdict,'Dict':dict,'Callable':Callable,'optim':torch.optim}
    scope.update(raw('src/optimizers/lookahead.py',scope));scope.update(raw('src/optimizers/optim_factory.py',scope))
    return scope['create_optimizer'](types.SimpleNamespace(opt='lookahead_adamw',lr=1e-4,
        weight_decay=.001,opt_eps=1e-10,opt_betas=None),model)


def original_training_step(network):
    root,sha=verified_source('ttis')
    text=subprocess.check_output(['git','show',sha+':src/models/model_interface.py'],cwd=root,text=True)
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ModelInterface')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'training_step','_shared_step'}]
    ns={};exec(compile(ast.Module(body=methods,type_ignores=[]),'author/ModelInterface','exec'),ns)  # noqa: S102 -- actual original training methods without framework/logging imports
    obj=types.SimpleNamespace(model=network,loss=nn.CrossEntropyLoss(),train_acc=lambda *args:None,log=lambda *args,**kwargs:None)
    obj._shared_step=types.MethodType(ns['_shared_step'],obj)
    return types.MethodType(ns['training_step'],obj)


@pytest.mark.parametrize('profile',list(PROFILES))
@pytest.mark.parametrize('native',[False,True])
def test_all_six_author_heads_whole_training_forward_gradients_and_ttis(tmp_path,profile,native):
    opts=options(profile,native);b=data(tmp_path/'data',1024 if native else 16,classes=3)
    initial=rng_state();m=TTISMIL(b,profile,**opts);after=rng_state();restore_rng(initial)
    ref=reference_network(m.kind,b.dim,3,opts)
    equal(m.network.state_dict(),ref.state_dict());equal(after,rng_state())
    x=b.load(b.records('train')[0])[0];label=torch.tensor([2])
    m.train();ref.train();initial=rng_state();out=m(x,label=label);out['total_loss'].backward();after=rng_state();restore_rng(initial)
    native_result,_=ref(x[None],label=label,loss_fn=nn.CrossEntropyLoss())
    loss=native_result['loss'] if profile=='ttis_clam_full_loss' else F.cross_entropy(native_result['logits'],label)
    loss.backward();equal(after,rng_state());equal(out['logits'],native_result['logits']);equal(out['total_loss'],loss)
    for (name,p),(other,q) in zip(m.network.named_parameters(),ref.named_parameters()):
        assert name==other
        if p.grad is None:assert q.grad is None
        else:equal(p.grad,q.grad)
    if m.kind=='clam':
        active=[p.grad is not None and p.grad.abs().sum()>0 for p in m.network.instance_classifiers.parameters()]
        assert any(active)==(profile=='ttis_clam_full_loss')
    m.eval();ref.eval();row=b.records('test')[0];x=b.load(row)[0]
    clusters,guidance=b.ttis.load(row)
    scores=m.semantic_scale*F.normalize(guidance,dim=-1) @ m.semantic_text.T
    expected_views={};run=ref.sampling.run
    def capture(*args,**kwargs):
        result=run(*args,**kwargs);expected_views.update(result);return result
    ref.sampling.run=capture
    kwargs={'label':torch.tensor([0]),'loss_fn':nn.CrossEntropyLoss()} if m.kind=='clam' else {}
    with torch.no_grad():
        actual=m.forward_unit(b,[row],'cpu',{})
        expected,_=ref(x[None],use_ttis=True,clusters=clusters,zeroshot_model=FrozenSemantic(scores[None]),**kwargs)
    equal(actual['logits'],expected['logits']);equal(actual['selected_indices'],expected_views)
    changed={**row,'target':1-row['target']}
    equal(actual,m.forward_unit(b,[changed],'cpu',{}))
    assert len(expected_views)==2+len(b.ttis.semantic['prompts'])


@pytest.mark.parametrize('profile',list(PROFILES))
def test_original_training_step_all_lookahead_updates_and_cosine_state(tmp_path,profile):
    b=data(tmp_path/'data',classes=3);opts=options(profile)
    m=TTISMIL(b,profile,**opts);ref=reference_network(m.kind,b.dim,3,opts)
    ref.load_state_dict(m.network.state_dict())
    cfg={'grad_clip':None,'max_instances':None,'selection_metric':'ttis_val_loss','lr':1e-4}
    m.configure_training(cfg);opt=m.make_optimizer(1e-4,.001);refopt=reference_optimizer(ref)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(refopt,T_max=50,eta_min=1e-7)
    source_step=original_training_step(ref)
    original_forward=m.forward_unit;observed={};steps=0
    def forward(bundle,rows,device,config,train=False):
        observed.update(rows=rows,rng=rng_state())
        return original_forward(bundle,rows,device,config,train)
    m.forward_unit=forward
    actual_step=opt.step
    def compare(*args,**kwargs):
        nonlocal steps
        grads=[None if p.grad is None else p.grad.detach().clone() for p in m.network.parameters()]
        result=actual_step(*args,**kwargs);after=rng_state();restore_rng(observed['rng'])
        row=observed['rows'][0];x=b.load(row)[0][None];target=torch.tensor([row['target']])
        refopt.zero_grad()
        if profile=='ttis_clam_full_loss':loss=ref(x,label=target,loss_fn=nn.CrossEntropyLoss())[0]['loss']
        else:loss=source_step({'embeddings':x,'label':target},0)
        loss.backward()
        for actual,p in zip(grads,ref.parameters()):
            if actual is None:assert p.grad is None
            else:equal(actual,p.grad)
        refopt.step();equal(m.network.state_dict(),ref.state_dict());equal(opt.state_dict(),refopt.state_dict());equal(after,rng_state())
        steps+=1;return result
    opt.step=compare
    groups=[next(g for g in units(b,'train') if g[0]['target']==i) for i in [0,2]]
    for _ in range(7):
        m.train();ref.train();ttis_epoch(m,b,groups,opt,'cpu',{})
        m.finish_epoch(b,'cpu',{});schedule.step();equal(m.scheduler.state_dict(),schedule.state_dict())
    assert steps==14 and m.optimizer_steps==14 and m.completed_epochs==7
    assert all(g['lookahead_step']==14 for g in opt.param_groups)
    assert any('lookahead_slow_buff' in value for value in opt.state_dict()['state'].values())
    write_json(tmp_path/'ttis_author_training.json',{'state':'passed','profile':profile,'optimizer_updates':steps,
        'epochs':7,'scope':'All original ModelInterface.training_step (or named full CLAM loss), original Lookahead/AdamW step gradients/weights/moments/slow buffers/cosine and RNG bitwise equal; CPU synthetic two-slide training fixture'})


@pytest.mark.parametrize('profile',list(PROFILES))
@pytest.mark.parametrize('boundary',[1,2])
def test_full_train_exact_resume_source_validation_selection_and_repeat_csv(tmp_path,profile,boundary):
    b=data(tmp_path/'data');cfg={'epochs':3,'threads':1,'model':options(profile)}
    full=train(b.root,profile,tmp_path/'full',seed=9,config=cfg)
    part=train(b.root,profile,tmp_path/'part',seed=9,config=cfg,stop_after=boundary)
    assert read_json(part/'status.json')['state']=='interrupted'
    train(b.root,profile,part,seed=9,config=cfg,resume=True)
    a=torch.load(full/'last.pt',weights_only=False);c=torch.load(part/'last.pt',weights_only=False)
    for key in ['model','optimizers','history','rng']:equal(a[key],c[key])
    assert a['model']['_extra_state']['completed_epochs']==3
    assert a['model']['_extra_state']['optimizer_steps']==3*len(units(b,'train'))
    best=torch.load(full/'best.pt',weights_only=False)
    history=read_json(full/'history.json')
    assert best['epoch']==min(range(3),key=lambda e:history[e]['validation']['ttis_val_loss'])
    metrics=evaluate(full);sha=digest_file(full/'test_predictions.csv')
    equal(metrics,evaluate(full));assert sha==digest_file(full/'test_predictions.csv')
    equal(metrics,evaluate(part));assert sha==digest_file(part/'test_predictions.csv')
    manifest=read_json(full/'test_instances/manifest.json')
    assert manifest['synthetic'] and len(manifest['records'])==len(b.records('test'))
    for record in manifest['records']:
        with h5py.File(record['path']) as f:
            assert len(f['indices'])==2+len(b.ttis.semantic['prompts'])
            assert all((v[:]<len(f['coords'])).all() for v in f['indices'].values())
    check_compatibility(require_runnable(profile),b)


@pytest.mark.parametrize('secondary',[False,True])
def test_semantic_policy_test_only_sampling_and_prediction_rng(tmp_path,secondary):
    b=data(tmp_path/'data',secondary=secondary);m=TTISMIL(b,**options('ttis_abmil_source'));m.eval()
    state=rng_state()
    for subset in ['train','val']:
        row=b.records(subset)[0]
        out=m.forward_unit(b,[row],'cpu',{})
        equal(out['logits'],m(b.load(row)[0])['logits'])
        assert m.trace is None
    row=b.records('test')[0]
    out=m.forward_unit(b,[row],'cpu',{})
    equal(out,m.forward_unit(b,[row],'cpu',{}));equal(state,rng_state())
    assert m.assets.policy==('aligned_secondary' if secondary else 'source_primary')


@pytest.mark.parametrize('mutation',['missing','duplicate','negative','bool','empty'])
def test_invalid_partition_is_never_silently_replaced(mutation):
    value={'0':[0,1],'1':[2,3]}
    if mutation=='missing':value['1']=[2]
    elif mutation=='duplicate':value['1']=[1,2,3]
    elif mutation=='negative':value['1']=[-1,3]
    elif mutation=='bool':value['0']=[False,1]
    else:value['0']=[]
    with pytest.raises(ValueError,match='TTIS'):validate_clusters(value,4)


@pytest.mark.parametrize('secondary',[False,True])
@pytest.mark.parametrize('asset',['features','coords','text','clusters','guidance'])
def test_asset_mismatch_blocks_evaluation_inputs(tmp_path,secondary,asset):
    if asset=='guidance' and not secondary:return
    b=data(tmp_path/'data',secondary=secondary);row=b.records('test')[0]
    if asset in {'features','coords'}:
        with h5py.File(b.root/row['path'],'r+') as f:
            key='features' if asset=='features' else 'coords';f[key][0,0]+=1
    elif asset=='text':
        path=Path(b.ttis.semantic['text_path']);value=np.load(path);np.save(path,-value)
    elif asset=='clusters':
        key=next(iter(b.ttis.rows[row['slide_id']]['clusters']));b.ttis.rows[row['slide_id']]['clusters'][key].append(0)
    else:
        with h5py.File(b.ttis.rows[row['slide_id']]['guidance_path'],'r+') as f:f['features'][0,0]+=1
    with pytest.raises(ValueError,match='TTIS'):b.ttis.load(row)


@pytest.mark.parametrize('change',[{'dropout':.3},{'hidden':31},{'sampling_ratio':0.},{'sampling_ratio':True},
                                   {'eta_min':float('nan')},{'scheduler_epochs':0}])
def test_unknown_or_invalid_model_options_fail(tmp_path,change):
    b=data(tmp_path/'data')
    with pytest.raises((ValueError,TypeError)):TTISMIL(b,**change)


def test_native_cuml_never_silently_switches_to_cpu(tmp_path):
    if torch.cuda.is_available():return
    with pytest.raises(ValueError,match='no implicit CPU'):
        hierarchical_clusters(torch.randn(10,4),np.arange(20).reshape(10,2),4,tmp_path/'clusters.json','slide','cuml')


@pytest.mark.parametrize('clusters',[1,4,16])
def test_actual_two_stage_cpu_preparation_against_independent_kmeans(tmp_path,clusters):
    from sklearn.cluster import KMeans
    rng=np.random.default_rng(11);x=torch.tensor(rng.normal(size=(32,4)).astype(np.float32))
    coords=np.column_stack([np.arange(32)*256,np.arange(32)%4*256])
    actual=hierarchical_clusters(x,coords,clusters,tmp_path/'clusters.json','s','sklearn_cpu',seed=17,n_init=2)
    count=math.isqrt(clusters);outer=KMeans(n_clusters=count,init='k-means++',n_init=2,random_state=17).fit(coords).labels_
    expected={};counter=0
    for group in range(count):
        index=np.flatnonzero(outer==group)
        if not len(index):continue
        inner=KMeans(n_clusters=min(len(index),count),init='k-means++',n_init=2,random_state=17).fit(x[index].numpy()).labels_
        for subgroup in range(min(len(index),count)):
            members=index[inner==subgroup]
            if len(members):expected[str(counter)]=members.tolist();counter+=1
    assert actual==expected
    assert read_json(tmp_path/'clusters.json')['s']==expected


def test_preparation_never_reads_test_targets_and_preserves_large_physical_coords(tmp_path):
    b=data(tmp_path/'data')
    for row in b.records('test'):
        with h5py.File(b.root/row['path'],'r+') as f:f['coords'][:]=f['coords'][:]+2**26
    first=prepare_ttis(b,tmp_path/'first',b.root/'semantic.json','sklearn_cpu',clusters=16,n_init=2)
    b.rows.loc[b.rows.subset=='test','target']=1-b.rows.loc[b.rows.subset=='test','target']
    second=prepare_ttis(b,tmp_path/'second',b.root/'semantic.json','sklearn_cpu',clusters=16,n_init=2)
    assert read_json(first)['slides']==read_json(second)['slides']
    assert read_json(first)['clustering']['initialization']=='explicit CPU k-means++ variant'


@pytest.mark.parametrize('change',[{'classes':['wrong','order']},{'dimension':-1},{'synthetic':False},
                                  {'logit_scale':float('nan')},{'weights_sha256':'unversioned'},
                                  {'prompt_class_indices':[0,0]},{'prompts':['one']}])
def test_semantic_provenance_and_prompt_schema_required(tmp_path,change):
    b=data(tmp_path/'data');path=b.root/'semantic.json'
    original=read_json(path);path.write_text(json.dumps({**original,**change}))
    with pytest.raises(ValueError,match='TTIS|semantic dimension'):
        semantic_metadata(path,b)


def test_changed_manifest_or_semantics_blocks_previous_run(tmp_path):
    b=data(tmp_path/'data');run=train(b.root,'ttis_abmil_source',tmp_path/'run',config={'epochs':2,'threads':1,'model':options('ttis_abmil_source')})
    meta=read_json(b.ttis.path);meta['clustering']['seed']+=1;write_json(b.ttis.path,meta)
    with pytest.raises(ValueError,match='inputs differ'):
        evaluate(run)


def test_source_cross_encoder_policy_is_explicit_and_secondary_alignment_is_enforced(tmp_path):
    b=data(tmp_path/'data')
    assert b.ttis.policy=='source_primary'
    doc=read_json(b.ttis.path)
    registration=Path(doc['semantic_metadata']);semantic=read_json(registration)
    semantic['visual_embedding_space']='different_space'
    write_json(registration,semantic);doc['semantic_metadata_sha256']=digest_file(registration)
    doc['semantic_policy']='aligned_secondary';write_json(b.ttis.path,doc)
    with pytest.raises(ValueError,match='matching declared visual/text'):
        Bundle(b.root)


@pytest.mark.parametrize('change',['missing','extra','space','coordinates','zero','dimension','nonfinite'])
def test_secondary_preparation_rejects_invalid_assets_before_publishing_manifest(tmp_path,change):
    b=data(tmp_path/'data',secondary=True)
    guidance={key:row['guidance_path'] for key,row in b.ttis.rows.items()}
    if change=='missing':guidance.pop(next(iter(guidance)))
    elif change=='extra':guidance['train_slide']=next(iter(guidance.values()))
    elif change=='space':
        meta=read_json(b.root/'semantic.json');meta['visual_embedding_space']='unaligned'
        write_json(b.root/'semantic.json',meta)
    else:
        path=next(iter(guidance.values()))
        with h5py.File(path,'r+') as f:
            if change=='coordinates':f['coords'][0,0]+=1
            elif change=='zero':f['features'][0]=0
            elif change=='nonfinite':f['features'][0,0]=float('nan')
            else:
                array=f['features'][:,:-1];del f['features'];f['features']=array
    out=tmp_path/'invalid'
    with pytest.raises(ValueError,match='TTIS'):
        prepare_ttis(b,out,b.root/'semantic.json','sklearn_cpu',clusters=16,n_init=2,guidance=guidance)
    assert not (out/'manifest.json').exists()

"""Independent pinned ReCo-MIL networks, original trainer updates and protocol invariants."""
import ast
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo, create_dual_scale_demo
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.recomil import PROFILES, ReCoMIL, fit_prototypes, recomil_epoch
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def raw_source():
    root, sha = verified_source('recomil')
    ns = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'os': os, 'math': math, 'random': random,
          'time': time, 'Optional': Optional, 'Literal': Literal, 'Dict': dict, 'Any': Any,
          'autocast': autocast}
    def source(path):
        return subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)
    for path in ['recomil_model.py', 'dual_recomil.py', 'rarity_gate.py', 'orth_regularizer.py', 'rare_weight.py']:
        tree = ast.parse(source(path))
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'author/'+path, 'exec'), ns)  # noqa: S102 -- independent ASTs from fixed author Git blobs
    main = next(n for n in ast.parse(source('train_recomil_v3.py')).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    for name in ['lr_factor', 'tau_now', 'set_proto_trainable']:
        node = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'author/train_schedule', 'exec'), ns)  # noqa: S102 -- independent ASTs from fixed author Git blobs
    loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    body = []
    for node in loop.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'val_metrics':
            break
        body.append(node)
    function = ast.parse('def author_epoch(epoch):\n    pass').body[0]
    function.body = body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), 'author/train_epoch', 'exec'), ns)  # noqa: S102 -- independent ASTs from fixed author Git blobs
    return ns


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def fixture(path, profile='recomil_single', **options):
    dual = profile.startswith('recomil_dual')
    data = create_dual_scale_demo(path, high_dimension=24) if dual else create_demo(path, patients=20)
    model = ReCoMIL(Bundle(data), profile, **{'hidden': 16, 'prototypes': 8, 'cluster_iterations': 5, **options})
    bundle = Bundle(data)
    model.prepare(bundle, 2026, 'cpu')
    model.configure_training({'epochs': 50, 'lr': 1e-4, 'max_instances': None})
    model.start_epoch(12)
    return model, bundle


def original_network(model, path, raw):
    path.mkdir(exist_ok=True)
    settings = {}
    for branch, params in model.settings.items():
        location = path/(branch+'.pt'); torch.save(getattr(model, 'initial_'+branch), location)
        settings[branch] = {'dim_in': model.dims[branch], 'embed_dim': params['hidden'], 'num_protos': params['prototypes'],
            'proto_init_path': str(location), 'similarity_method': params['similarity'], 'tau': params['tau_start'],
            'route_topk': params['route_topk'], 'use_graph_reducer': params['graph_reducer'], 'graph_thresh': params['graph_threshold']}
    common = {'num_classes': 2, 'num_enhancers': model.settings['high']['enhancers'], 'drop': .25,
              'hard': model.branches()['high'].hard_assignment}
    network = raw['DualReCoMIL'](**{key+'_'+branch: value for branch in settings for key, value in settings[branch].items()}, **common) if model.dual else raw['ReCoMIL'](**settings['high'], **common)
    network.load_state_dict(model.network.state_dict())
    for branch, original in model.branches().items():
        target = getattr(network, branch) if model.dual else network
        target.tau = original.tau
        target.prototypes.requires_grad_(original.prototypes.requires_grad)
    return network


def original_forward(model, net, values, raw, training):
    kwargs = {}
    for branch in model.settings:
        network = getattr(net, branch) if model.dual else net
        x, p = values[branch], model.settings[branch]
        gate, weights = None, None
        if training:
            if p['outlier_top_p'] > 0 and p['outlier_alpha'] > 0:
                with torch.no_grad():
                    a, b = (network.patch_proj(x), network.proto_proj(network.prototypes)) if p['gate_in_embed'] else (x, network.prototypes.detach())
                    gate = raw['rarity_gate'](a, b, p['outlier_top_p'], p['outlier_alpha'], method=p['similarity'], in_embed=p['gate_in_embed'])
            if p['use_frequencies'] and p['rare_beta'] > 0:
                freq = getattr(model, 'frequency_'+branch).numpy().astype(np.float32)
                w = (freq+1.)**(-p['rare_beta']); weights = torch.from_numpy(w/(w.mean()+1e-6))
        suffix = '_'+branch if model.dual else ''
        kwargs.update({'data'+suffix: x, 'instance_gate'+suffix: gate, 'proto_weights'+suffix: weights})
    return net(**kwargs)


@pytest.mark.parametrize('profile', sorted(PROFILES))
@pytest.mark.parametrize('frequencies', [False, True])
def test_full_source_outputs_objective_and_active_gradients(tmp_path, profile, frequencies):
    model, bundle = fixture(tmp_path/'data', profile, use_frequencies=frequencies, gate_in_embed=frequencies)
    raw = raw_source(); net = original_network(model, tmp_path/'source', raw)
    for branch, network in model.branches().items():
        other = getattr(net, branch) if model.dual else net
        other.tau = network.tau
    values = model.bags(bundle, units(bundle, 'train')[0], 'cpu')
    label = torch.tensor([1]); state = rng_state()
    out = model(**values, train=True); actual = model.objective(out, label)
    restore_rng(state); expected = original_forward(model, net, values, raw, True)
    loss = F.cross_entropy(expected['logits'], label, label_smoothing=.03)
    for branch in model.settings:
        branch_net = getattr(net, branch) if model.dual else net
        loss = loss+.02*raw['orth_loss'](branch_net.prototypes)
    torch.testing.assert_close(out['logits'], expected['logits'], rtol=0, atol=0)
    torch.testing.assert_close(actual['loss'], loss, rtol=0, atol=0)
    actual['loss'].backward(); loss.backward()
    for (name, a), (other, b) in zip(model.network.named_parameters(), net.named_parameters()):
        assert name == other
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None and name.endswith('similarity_scale'), name
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=lambda message, name=name: name+': '+message)


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_default_1024_1536_inputs_512_hidden_source_gradients(tmp_path, profile):
    dual = profile.startswith('recomil_dual')
    data = create_dual_scale_demo(tmp_path/'data', dimension=1024, high_dimension=1536) if dual else create_demo(tmp_path/'data', patients=20, dimension=1024)
    bundle = Bundle(data); model = ReCoMIL(bundle, profile); model.prepare(bundle, 2026, 'cpu'); model.start_epoch(12)
    raw = raw_source(); net = original_network(model, tmp_path/'raw', raw)
    for branch, network in model.branches().items():
        (getattr(net, branch) if dual else net).tau = network.tau
    values = {branch: torch.randn(5, dimension) for branch, dimension in model.dims.items()}
    model.eval(); net.eval()
    a = model(**values)['logits']; b = original_forward(model, net, values, raw, False)['logits']
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward(); b.square().sum().backward()
    for (name, p), (other, q) in zip(model.network.named_parameters(), net.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None and name.endswith('similarity_scale')
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_original_full_epoch_schedule_adam_updates(tmp_path, profile):
    model, bundle = fixture(tmp_path/'data', profile, use_frequencies=True, class_weight=True, gate_in_embed=True)
    raw = raw_source(); net = original_network(model, tmp_path/'source', raw)
    args = {'fp16': False, 'print_every': 0, 'epochs': 50, 'warmup_epochs': 12, 'lr': 1e-4,
            'label_smoothing': .03, 'grad_clip': 1., 'ortho_on_projected': False}
    names = {'freeze_epochs':'freeze_protos_epochs', 'similarity':'similarity_method'}
    for branch, params in model.settings.items():
        for key in ['tau_start','tau_end','tau_epochs','freeze_epochs','outlier_top_p','outlier_alpha','gate_in_embed','similarity','lambda_ortho']:
            args[names.get(key, key)+'_'+branch] = params[key]
    groups = units(bundle, 'train')[:2]; batches = []
    for rows in groups:
        values = model.bags(bundle, rows, 'cpu'); label = torch.tensor([rows[0]['target']])
        x = (values['high'][None], values['low'][None]) if model.dual else values['high'][None]
        batches.append((x, label, ['synthetic']))
    opt = model.make_optimizer(1e-4, 5e-4); other = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)
    raw.update(model=net, args=SimpleNamespace(**args), multi_scale=model.dual, opt=other,
        scaler=GradScaler('cuda', enabled=False), device='cpu', dl_tr=batches,
        class_w=model.class_weights.clone(), w_high=model.rare_weight_high.clone(),
        w_low=model.rare_weight_low.clone() if model.dual else None)
    for epoch in [1, 2, 12, 13, 14, 35, 50]:
        state = rng_state(); raw['author_epoch'](epoch)
        restore_rng(state); model.start_epoch(epoch-1)
        recomil_epoch(model, bundle, groups, opt, 'cpu', {'grad_clip': 1.})
        equal(model.network.state_dict(), net.state_dict()); equal(opt.state_dict(), other.state_dict())
        for branch, network in model.branches().items():
            original = getattr(net, branch) if model.dual else net
            assert network.tau == original.tau and network.prototypes.requires_grad == original.prototypes.requires_grad
    assert model.optimizer_steps == 14


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_exact_continuation_across_freeze_boundary(tmp_path, profile):
    model, bundle = fixture(tmp_path/'data', profile)
    config = {'epochs': 3, 'threads': 1, 'model': {'hidden': 16, 'prototypes': 8, 'cluster_iterations': 5,
        'freeze_epochs': 1, 'use_frequencies': True, 'low': {'freeze_epochs': 2} if model.dual else None}}
    full = train(bundle.root, profile, tmp_path/'full', config=config)
    paused = train(bundle.root, profile, tmp_path/'resumed', config=config, stop_after=1)
    train(bundle.root, profile, paused, config=config, resume=True)
    for kind in ['best','last']:
        a, b = [torch.load(path/(kind+'.pt'), weights_only=False) for path in [full, paused]]
        equal(a['model'], b['model']); equal(a['optimizers'], b['optimizers']); assert a['history'] == b['history']
    assert evaluate(full) == evaluate(paused)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(paused/'test_predictions.csv'))


@pytest.mark.parametrize('ratio,limit', [(1.,0),(.5,0),(1.,9),(.5,9)])
def test_original_prototype_sampling_and_minibatch_kmeans(tmp_path, ratio, limit):
    root, sha = verified_source('recomil')
    code = subprocess.check_output(['git','show',sha+':minibatch_kmeans_recomil.py'],cwd=root,text=True)
    ns = {'np':np,'os':os,'random':random}
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in {'load_ids','iter_slide_arrays','sample_arrays'}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'author/clustering','exec'),ns)  # noqa: S102 -- independent ASTs from fixed author Git blobs
    rng = np.random.default_rng(21); arrays = [rng.normal(size=(20+i,8)).astype(np.float32) for i in range(3)]
    for i, array in enumerate(arrays):
        directory=tmp_path/f'slide{i}';directory.mkdir();np.save(directory/'features.npy',array)
    args=SimpleNamespace(root_dir=str(tmp_path),feat_dim=8,sample_ratio=ratio,max_per_slide=limit,scale='high',dataset='legacy',ids_file=None)
    state=np.random.get_state();np.random.seed(2025)
    try:
        stacked=ns['sample_arrays'](args)
    finally:
        np.random.set_state(state)
    km=MiniBatchKMeans(n_clusters=8,random_state=2025,batch_size=4096,max_iter=200,verbose=0).fit(stacked)
    centers,freq,_=fit_prototypes(arrays,8,2025,ratio,limit)
    np.testing.assert_array_equal(centers,km.cluster_centers_)
    np.testing.assert_array_equal(freq,np.bincount(km.predict(stacked),minlength=8))


@pytest.mark.parametrize('method',['l2','cosine','dot'])
@pytest.mark.parametrize('hard',[False,True])
def test_routing_variants_and_inactive_scale(tmp_path,method,hard):
    model,bundle=fixture(tmp_path/'data',similarity=method,hard=hard,use_frequencies=True)
    model.eval();values=model.bags(bundle,units(bundle,'train')[0],'cpu')
    raw=raw_source();net=original_network(model,tmp_path/'source',raw);net.eval()
    expected=original_forward(model,net,values,raw,True)
    result=model(**values,train=True)
    torch.testing.assert_close(result['logits'],expected['logits'],rtol=0,atol=0)
    model.objective(result,torch.tensor([1]))['loss'].backward()
    assert model.network.similarity_scale.grad is None
    assert model.network.prototypes.grad is not None and torch.isfinite(model.network.prototypes.grad).all()


@pytest.mark.parametrize('method',['l2','cosine','dot'])
def test_rarity_ties_and_projection_orthogonal_source_behavior(tmp_path,method):
    model,_=fixture(tmp_path/'data',similarity=method,ortho_projected=True)
    x=torch.zeros(7,16);p=torch.zeros(8,16)
    assert torch.equal(model.gate(x,p,top_p=.015,alpha=.2,method=method),torch.full((7,),1.2))
    assert torch.equal(model.gate(x,p,top_p=0,alpha=.2,method=method),torch.ones(7))
    out=model(high=torch.randn(5,16));objective=model.objective(out,torch.tensor([1]))
    assert not objective['orth'].requires_grad and objective['ce'].requires_grad


def test_deferred_initialization_matches_loaded_source_rng(tmp_path):
    data=create_dual_scale_demo(tmp_path/'data');bundle=Bundle(data)
    raw=raw_source()
    for profile in ['recomil_single','recomil_dual_graph']:
        torch.manual_seed(2026);model=ReCoMIL(bundle,profile,hidden=16,prototypes=8)
        # Original source must read valid prototype tensors rather than taking its random fallback.
        directory=tmp_path/profile;directory.mkdir()
        for branch in model.settings:
            torch.save(torch.zeros(8,16),directory/(branch+'.pt'))
        torch.manual_seed(2026)
        common={'num_classes':2,'num_enhancers':3,'drop':.25,'hard':False}
        args={'dim_in':16,'embed_dim':16,'num_protos':8,'similarity_method':'l2','tau':1.2,'route_topk':3,
              'use_graph_reducer':profile.endswith('graph'),'graph_thresh':.15}
        if model.dual:
            net=raw['DualReCoMIL'](**{k+'_'+b:v for b in model.settings for k,v in {**args,'proto_init_path':str(directory/(b+'.pt'))}.items()},**common)
        else:
            net=raw['ReCoMIL'](**args,proto_init_path=str(directory/'high.pt'),**common)
        equal(model.network.state_dict(),net.state_dict())


def test_training_only_prototypes_and_no_inference_gate(tmp_path):
    model,bundle=fixture(tmp_path/'data','recomil_dual_graph',use_frequencies=True)
    before=copy.deepcopy(model.ledger)
    for subset in ['val','test']:
        for row in bundle.records(subset):
            import h5py
            with h5py.File(bundle.root/row['path'],'r+') as f:
                f['features'][...]=np.full(f['features'].shape,9999.,dtype=np.float32)
            np.save(bundle.root/bundle.dual_scale.rows[row['slide_id']]['path'],np.full((17,24),-9999.,dtype=np.float32))
    other=ReCoMIL(Bundle(bundle.root),'recomil_dual_graph',hidden=16,prototypes=8,cluster_iterations=5,use_frequencies=True)
    other.prepare(Bundle(bundle.root),2026,'cpu');assert before==other.ledger
    values=model.bags(bundle,units(bundle,'train')[0],'cpu');model.eval()
    a=model(**values)['logits'];b=model.network(data_high=values['high'],data_low=values['low'])['logits']
    assert torch.equal(a,b)
    training=model(**values,train=True)['logits'];assert not torch.equal(a,training)


@pytest.mark.parametrize('options',[{'prototypes':4},{'tau_start':0},{'outlier_top_p':1.1},{'low':{}},{'warmup_epochs':True},{'sample_ratio':0}])
def test_invalid_configuration_is_rejected(tmp_path,options):
    bundle=Bundle(create_demo(tmp_path/'data',patients=20))
    with pytest.raises(ValueError):
        ReCoMIL(bundle,**options)


def test_missing_data_insufficient_training_and_global_truncation(tmp_path):
    bundle=Bundle(create_demo(tmp_path/'data',patients=20))
    with pytest.raises(ValueError,match='dual_scale'):
        ReCoMIL(bundle,'recomil_dual')
    model=ReCoMIL(bundle,hidden=16,prototypes=2048)
    with pytest.raises(ValueError,match='sampled training'):
        model.prepare(bundle,2026,'cpu')
    with pytest.raises(ValueError,match='max_instances'):
        model.configure_training({'max_instances':4,'epochs':1,'lr':1e-4})
    with pytest.raises(RuntimeError,match='prototype preparation'):
        model(torch.ones(2,16))


def test_family_cli_and_validation_joint(tmp_path):
    data=create_demo(tmp_path/'data',patients=20)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'epochs':1,'threads':1,'model':{'hidden':16,'prototypes':8,'cluster_iterations':5}}))
    for args in [['check','--data',str(data),'--method','ReCo-MIL','--deep'],
                 ['run','--data',str(data),'--method','ReCo-MIL','--config',str(config),'--output',str(tmp_path/'run')],
                 ['evaluate','--run',str(tmp_path/'run')]]:
        out=subprocess.run([sys.executable,'-m','milplus',*args],capture_output=True,text=True,check=True)
    assert json.loads(out.stdout)['n']>0
    metrics=json.loads((tmp_path/'run/validation_metrics.json').read_text())
    assert metrics['reco_joint']==(metrics['macro_auc']+metrics['accuracy'])/2
    protocol=json.loads((tmp_path/'run/method_protocol.json').read_text())
    assert protocol['optimizer']=='Adam' and protocol['prepared'] and protocol['optimizer_steps']>0


def test_asymmetric_low_branch_full_source_objective(tmp_path):
    model,bundle=fixture(tmp_path/'data','recomil_dual_graph',use_frequencies=True,
        low={'hidden':24,'prototypes':16,'graph_reducer':False,'tau_start':.9,'tau_end':.4,
             'lambda_ortho':.04,'similarity':'cosine','sample_ratio':.5,'gate_in_embed':True})
    raw=raw_source();net=original_network(model,tmp_path/'source',raw)
    values=model.bags(bundle,units(bundle,'train')[0],'cpu');state=rng_state()
    out=model(**values,train=True);actual=model.objective(out,torch.tensor([1]))['loss']
    restore_rng(state);expected=original_forward(model,net,values,raw,True)
    loss=F.cross_entropy(expected['logits'],torch.tensor([1]),label_smoothing=.03)
    for branch,params in model.settings.items():
        loss=loss+params['lambda_ortho']*raw['orth_loss'](getattr(net,branch).prototypes)
    torch.testing.assert_close(actual,loss,rtol=0,atol=0)
    actual.backward();loss.backward()
    for (name,a),(_,b) in zip(model.network.named_parameters(),net.named_parameters()):
        if a.grad is None:
            assert b.grad is None and name.endswith('similarity_scale')
        else:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

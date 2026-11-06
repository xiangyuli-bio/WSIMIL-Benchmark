"""Whole pinned MicroMIL numerical loops, independent graph equations and data contract."""
import ast
import copy
import json
import subprocess
from types import SimpleNamespace
from typing import Optional

import dgl
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from dgl.nn import GATConv
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, predict, restore_rng, rng_state, seed_everything, train, units
from milplus.micromil import MicroMIL, micromil_epoch
from milplus.registry import inspect_method
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, patient=False, instances=None, **kwargs):
    root = create_demo(tmp_path/'data', patients=20, dimension=4)
    if patient:
        task = yaml.safe_load((root/'task.yaml').read_text());task['training_unit'] = 'patient'
        (root/'task.yaml').write_text(yaml.safe_dump(task))
    b = Bundle(root)
    if instances is not None:
        for row in b.rows.to_dict('records'):
            with h5py.File(root/row['path'], 'r+') as h:
                for key in ('features', 'coords'):
                    value = h[key][:instances];del h[key];h[key] = value
    opts = {'hidden': 8, 'clusters': 4, 'dropout': .5, **kwargs}
    return MicroMIL(b, **opts), b, opts


def original_gat_class():
    root,sha=verified_source('dgl081')
    code=subprocess.check_output(['git','show',sha+':python/dgl/nn/pytorch/conv/gatconv.py'],cwd=root,text=True)
    nodes=[n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name=='GATConv']
    from dgl.nn.functional import edge_softmax
    from dgl.utils import expand_as_pair
    namespace={'th': torch,'nn': nn,'fn': dgl.function,'edge_softmax': edge_softmax,
                   'DGLError': dgl.DGLError,'Identity': nn.Identity,'expand_as_pair': expand_as_pair}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'dgl-0.8.1/gatconv.py','exec'),namespace)  # noqa: S102 -- original external Git AST
    return namespace['GATConv']


def native(model):
    """Original full class, including its unsqueezed GAT heads, at the FM boundary."""
    root, sha = verified_source('micromil')
    source = subprocess.check_output(['git', 'show', sha+':model.py'], cwd=root, text=True)
    tree = ast.parse(source)
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {'DEC', 'ClusterAttentionModel'}]
    node = next(n for n in definitions if n.name == 'ClusterAttentionModel')
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    init.body = [n for n in init.body if not isinstance(n, ast.If)]
    init.body = [n for n in init.body if not isinstance(n, ast.Assign) or ast.unparse(n.targets[0]) != 'self.feature_extractor.fc']
    init.body.insert(1, ast.parse('self.feature_extractor = nn.Identity()').body[0])
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'Parameter': Parameter, 'Optional': Optional, 'dgl': dgl, 'GATConv': original_gat_class()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), 'author/model.py', 'exec'), namespace)  # noqa: S102 -- independent author AST
    args = SimpleNamespace(model_name='frozen_fm', device='cpu', cluster_number=model.clusters,
                           layer=model.layers, shuffle=False)
    network = namespace['ClusterAttentionModel'](args, in_dim=model.dim, hidden_dim=model.hidden, n_classes=2, drop_rate=model.dropout)
    network.load_state_dict(model.network.state_dict())
    return network


def equal(a, b, atol=0):
    if torch.is_tensor(a):torch.testing.assert_close(a, b, rtol=0, atol=atol)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:equal(a[key], b[key], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):equal(x, y, atol)
    elif isinstance(a, np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a == b


@pytest.mark.parametrize('n,clusters,layers', [(1,1,1),(1,4,2),(7,4,2),(27,16,3)])
def test_whole_author_forward_and_gradients(tmp_path, n, clusters, layers):
    seed_everything(17)
    m, _b, _ = fixture(tmp_path, clusters=clusters, layers=layers)
    author = native(m)
    x = torch.randn(n, 4)
    state = rng_state();out = m(x);loss = F.binary_cross_entropy(out['logits'].softmax(1)[0,1], torch.tensor(1.));loss.backward()
    restore_rng(state);reference = author(x.unsqueeze(0));ref_loss = F.binary_cross_entropy(reference.softmax(0)[1], torch.tensor(1.));ref_loss.backward()
    equal(out['logits'].squeeze(), reference)
    equal(loss, ref_loss)
    for (name, p), (other, q) in zip(m.network.named_parameters(), author.named_parameters()):
        assert name == other and p.grad is not None
        equal(p.grad, q.grad, atol=1e-7)
    selected = out['cluster_assignments']
    assert selected.shape == (max(n, 2) if n == 1 else n, clusters)
    assert (selected.sum(0) == 1).all() and ((selected == 0) | (selected == 1)).all()
    assert (out['adjacency'].sum(1) == 1).all()
    if n > 1:
        assert m.network.dec.cluster_centers.grad.abs().sum() > 0


@pytest.mark.parametrize('dropout', [0., .5])
@pytest.mark.parametrize('patient', [False, True])
def test_complete_native_author_epoch_updates(tmp_path, dropout, patient):
    seed_everything(71)
    m, b, _ = fixture(tmp_path, patient=patient, dropout=dropout)
    author = native(m)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-6)
    ref_opt = torch.optim.Adam(author.parameters(), lr=1e-4, weight_decay=1e-6)
    groups = units(b, 'train');update_count = 0;maximum = 0.
    for epoch in range(3):
        m.train();author.train()  # Documented repair: source forgets this after first eval.
        initial = rng_state();losses = micromil_epoch(m, b, groups, opt, torch.device('cpu'), {})
        after = rng_state();restore_rng(initial)
        ref_losses = []
        # Original batch-size-one DataLoader, independent of the adapter's source_loader.
        for batch in torch.utils.data.DataLoader(groups, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0]):
            x = torch.cat([b.load(row)[0] for row in batch]);label = torch.tensor(float(batch[0]['target']))
            result = author(x.unsqueeze(0));loss = F.binary_cross_entropy(result.softmax(0)[1], label)
            ref_opt.zero_grad();loss.backward();ref_opt.step();ref_losses.append(float(loss.detach()))
        for key, p in m.network.state_dict().items():
            q = author.state_dict()[key];maximum = max(maximum, float((p-q).abs().max()))
            equal(p, q, atol=2e-7)
        equal(opt.state_dict(), ref_opt.state_dict(), atol=2e-7)
        np.testing.assert_allclose([l['loss'] for l in losses], ref_losses, rtol=0, atol=1e-7)
        equal(torch.get_rng_state(), after['torch'])
        # Evaluation consumes no future training randomness in the registered protocol.
        predict(m, b, 'micromil_frozen', 'val', torch.device('cpu'), {'max_instances': None})
        equal(torch.get_rng_state(), after['torch'])
        update_count += len(groups)
    assert m.optimizer_steps == update_count
    (tmp_path/'micromil_author_comparison.json').write_text(json.dumps({'state':'passed','epochs':3,
        'training_unit':'patient' if patient else 'slide', 'dropout':dropout,'optimizer_updates':update_count,
        'maximum_parameter_abs':maximum,'scope':'Original complete network and Adam/BCE loop at explicit FM boundary; source mode and validation repairs disclosed'}))


@pytest.mark.parametrize('edges', [([0,0,1,2,2],[0,1,1,1,2]),([0,1,1,2,0,1,2],[0,0,1,2,0,1,2])])
def test_independent_gat_equations_with_duplicate_directed_self_edges(edges):
    torch.manual_seed(16)
    net = GATConv(4, 3, num_heads=1)
    reference = copy.deepcopy(net)
    x = torch.randn(3,4,requires_grad=True);y = x.detach().clone().requires_grad_()
    source, dest = (torch.tensor(e) for e in edges)
    graph = dgl.graph((source,dest),num_nodes=3)
    output = net(graph,x).squeeze(1)
    # Explicit neighbor softmax and message sum: no DGL kernels in this oracle.
    feature = F.linear(y, reference.fc.weight)
    scores = F.leaky_relu((feature[source]*reference.attn_l.flatten()).sum(1)+
                         (feature[dest]*reference.attn_r.flatten()).sum(1),negative_slope=.2)
    manual = torch.stack([(feature[source[dest==node]]*scores[dest==node].softmax(0).unsqueeze(1)).sum(0)
                          for node in range(3)])+reference.bias
    torch.testing.assert_close(output,manual,rtol=1e-6,atol=1e-7)
    output.square().sum().backward();manual.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-6)
    for p,q in zip(net.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('option,value', [('clusters',0),('clusters',True),('clusters',1025),('layers',0),
    ('hidden',0),('dropout',1),('dropout',float('nan')),('dropout',True),('prediction_samples',0),('prediction_seed',-1)])
def test_reject_invalid_config(tmp_path,option,value):
    with pytest.raises(ValueError):fixture(tmp_path,**{option:value})


@pytest.mark.parametrize('kind',['empty','wrong_dim','nan','inf','rank'])
def test_reject_bad_features(tmp_path,kind):
    m,_,_=fixture(tmp_path)
    x={'empty':torch.empty(0,4),'wrong_dim':torch.randn(2,3),'nan':torch.full((3,4),float('nan')),
       'inf':torch.full((3,4),float('inf')),'rank':torch.randn(4)}[kind]
    with pytest.raises(ValueError):m(x)


@pytest.mark.parametrize('key,value',[('max_instances',5),('grad_clip',1.),('selection_metric','auroc')])
def test_reject_incompatible_recipe(tmp_path,key,value):
    m,_,_=fixture(tmp_path)
    cfg={'max_instances':None,'grad_clip':None,'selection_metric':'micromil_f1',key:value}
    with pytest.raises(ValueError):m.configure_training(cfg)


@pytest.mark.parametrize('samples',[1,3])
def test_label_independent_reproducible_stochastic_prediction(tmp_path,samples):
    m,b,_=fixture(tmp_path,prediction_samples=samples)
    cfg={'max_instances':None};state=rng_state()
    one=predict(m,b,'micromil_frozen','test',torch.device('cpu'),cfg)
    two=predict(m,b,'micromil_frozen','test',torch.device('cpu'),cfg)
    pd.testing.assert_frame_equal(one,two,check_exact=True)
    equal(torch.get_rng_state(),state['torch'])
    original=b.rows.target.copy();b.rows['target']=1-b.rows.target
    changed=predict(m,b,'micromil_frozen','test',torch.device('cpu'),cfg)
    np.testing.assert_array_equal(one[['p_0','p_1']],changed[['p_0','p_1']]);b.rows['target']=original


@pytest.mark.parametrize('patient',[False,True])
def test_full_engine_exact_resume_and_evaluate(tmp_path,patient):
    _,b,opts=fixture(tmp_path,patient=patient)
    cfg={'epochs':3,'threads':1,'patience':5,'model':opts}
    direct=train(b.root,'MicroMIL',tmp_path/'direct',seed=46,config=cfg)
    resumed=train(b.root,'MicroMIL',tmp_path/'resume',seed=46,config=cfg,stop_after=1)
    train(b.root,'MicroMIL',resumed,seed=46,config=cfg,resume=True)
    a=torch.load(direct/'last.pt',weights_only=False);c=torch.load(resumed/'last.pt',weights_only=False)
    for key in ['model','optimizers','history','rng','best','bad']:equal(a[key],c[key])
    assert json.loads((direct/'request.json').read_text())['method']=='micromil_frozen'
    assert not json.loads((direct/'status.json').read_text())['test_evaluated']
    first=evaluate(direct,subset='test');pred=(direct/'test_predictions.csv').read_bytes()
    second=evaluate(direct,subset='test');assert first==second and pred==(direct/'test_predictions.csv').read_bytes()
    protocol=json.loads((direct/'method_protocol.json').read_text())
    assert protocol['optimizer_steps']==3*len(units(b,'train')) and protocol['paper_reproduction']=='not_run'


def test_validation_patient_f1_not_slide_or_test_selection(tmp_path):
    m,_,_=fixture(tmp_path)
    frame=pd.DataFrame({'global_patient_id':['p0','p0','p0','p1'], 'target':[0,0,0,1],
                        'p_0':[.1,.8,.8,.9],'p_1':[.9,.2,.2,.1]})
    assert m.validation_metrics(frame)=={'micromil_f1':0.}
    assert inspect_method('MicroMIL')['class_count']==2


@pytest.mark.parametrize('task',['survival','multiclass','ordinal'])
def test_reject_unsupported_task(tmp_path,task):
    root=create_demo(tmp_path/'data',patients=30,dimension=4,classes=3 if task=='multiclass' else 2,survival=task=='survival')
    if task=='ordinal':
        p=root/'task.yaml';s=yaml.safe_load(p.read_text());s['task_type']='ordinal';p.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError):MicroMIL(Bundle(root))


def test_nonfinite_gradients_do_not_step(tmp_path):
    m,b,_=fixture(tmp_path);p=m.network.dec.cluster_centers
    p.register_hook(lambda grad:torch.full_like(grad,float('nan')))
    before={k:v.clone() for k,v in m.network.state_dict().items()}
    with pytest.raises(ValueError,match='gradient'):
        micromil_epoch(m,b,units(b,'train'),torch.optim.Adam(m.parameters()),torch.device('cpu'),{})
    assert m.optimizer_steps==0
    equal(before,m.network.state_dict())


@pytest.mark.parametrize('clusters',[2,7])
def test_independent_dce_equation_and_center_gradients(tmp_path,clusters):
    model,_,_=fixture(tmp_path,clusters=clusters,dropout=0.)
    x=torch.randn(9,4,requires_grad=True)
    expected_centers=model.network.dec.cluster_centers.detach().clone().requires_grad_()
    result=model.network.dec(x)
    numerator=1/(1+((x[:,None,:]-expected_centers[None,:,:])**2).sum(-1))
    expected=numerator/numerator.sum(1,keepdim=True)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    weights=torch.linspace(.1,1,result.numel()).reshape_as(result)
    (result*weights).sum().backward(retain_graph=True)
    (expected*weights).sum().backward()
    torch.testing.assert_close(model.network.dec.cluster_centers.grad,expected_centers.grad,rtol=0,atol=0)


@pytest.mark.parametrize('bad',[{}, {'optimizer_steps':-1}, {'optimizer_steps':True}])
def test_reject_invalid_checkpoint_counter(tmp_path,bad):
    model,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='checkpoint'):model.set_extra_state(bad)


@pytest.mark.parametrize('dtype',[torch.int64,torch.complex64])
def test_reject_nonreal_or_nonfloating_raw_forward(tmp_path,dtype):
    model,_,_=fixture(tmp_path)
    with pytest.raises(ValueError):model(torch.ones(3,4,dtype=dtype))


def test_sparse_pinned_reference_fetch_does_not_checkout_unrequested_files(tmp_path,monkeypatch):
    from milplus.execution import fetch_source
    origin=tmp_path/'origin';origin.mkdir()
    subprocess.run(['git','init','--quiet'],cwd=origin,check=True)
    (origin/'keep').mkdir();(origin/'keep/model.py').write_text('x=1\n');(origin/'unrelated.txt').write_text('not needed')
    subprocess.run(['git','add','.'],cwd=origin,check=True)
    subprocess.run(['git','-c','user.name=fixture','-c','user.email=fixture@example.com','commit','-qm','fixture'],cwd=origin,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=origin,text=True).strip()
    monkeypatch.setattr('milplus.registry.resource',lambda name:{'fixture':{'url':str(origin),'commit':sha,'sparse_files':['/keep/model.py']}})
    target=fetch_source('fixture',tmp_path/'checkout')
    assert (target/'keep/model.py').read_text()=='x=1\n' and not (target/'unrelated.txt').exists()
    assert subprocess.check_output(['git','status','--porcelain'],cwd=target,text=True)==''
    assert fetch_source('fixture',target)==target

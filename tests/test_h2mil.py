"""Real three-level H2 inputs, complete source training and unified recovery."""
import ast
import copy
import json
import math
import types
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GlobalAttention, LayerNorm, LEConv, global_max_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.h2mil import H2MIL, SparseTensor, WidthNorm
from milplus.h2mil_data import COMMIT, H2Hierarchy, prepare
from milplus.h2mil_demo import create_h2mil_demo
from milplus.io import digest_file, read_json, write_json
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import pinned_text, verified_source

PROFILES=['h2mil_source_compat','h2mil_repaired']

@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)

def data(path,classes=2,dim=32,side=2):
    return Bundle(create_h2mil_demo(path,patients=10*classes,classes=classes,dimension=dim,coarse_side=side))

def original():
    root,sha=verified_source('h2mil');assert sha==COMMIT
    scope={'__name__':__name__,'torch':torch,'nn':nn,'F':F,'math':math,'np':np,'Tensor':Tensor,
        'Parameter':nn.Parameter,'Linear':nn.Linear,'Optional':Optional,'Union':Union,'Tuple':tuple,'Callable':object,
        'Adj':Adj,'OptPairTensor':OptPairTensor,'OptTensor':OptTensor,'Size':Size,'MessagePassing':MessagePassing,
        'softmax':softmax,'scatter':scatter,'SparseTensor':SparseTensor,'LEConv':LEConv,'LayerNorm':LayerNorm,
        'global_mean_pool':global_mean_pool,'global_max_pool':global_max_pool,'GlobalAttention':GlobalAttention,
        'add_remaining_self_loops':add_remaining_self_loops,'device':torch.device('cpu'),'roc_auc_score':roc_auc_score}
    def definitions(path,names):
        nodes=[n for n in ast.parse(pinned_text(root,sha,'code/'+path)).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'independent_original_h2mil.py','exec'),scope)  # noqa: S102 -- actual fixed source definitions
    definitions('RAConv.py',['glorot','zeros','RAConv'])
    definitions('IHPool.py',['uniform','euclidean_dist','IHPool'])
    definitions('main.py',['GCN','return_auc'])
    tree=ast.parse(pinned_text(root,sha,'code/main.py'))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch_num')
    stop=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='auc_of_train' for t in n.targets))
    function=ast.FunctionDef(name='epoch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in
        ['model','optimizer','loss_fun','device','data_for_train','label_for_train']],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=copy.deepcopy(loop.body[:stop+1])+[ast.Return(value=ast.Name(id='auc_of_train',ctx=ast.Load()))],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'actual_h2mil_main_epoch.py','exec'),scope)  # noqa: S102 -- actual full outer train body, no numerical rewrites
    return scope

def reference(model,scope):
    hidden=model.network.conv1.out_channels
    net=scope['GCN'](model.bundle.dim,hidden,hidden,drop_out_ratio=.3,pool1_ratio=.1,pool2_ratio=4)
    net.norm=WidthNorm(net.norm,[model.bundle.dim,hidden,hidden//2])
    if model.corrected:net.lin2=nn.Linear(hidden//2,len(model.bundle.classes));net.softmax=nn.Identity()
    for module in net.modules():
        if isinstance(module,MessagePassing):module.propagate=types.MethodType(MessagePassing.propagate,module)
    net.load_state_dict(model.network.state_dict(),strict=True)
    return net

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

@pytest.mark.parametrize('profile',PROFILES)
def test_actual_author_summed_batch_loop_every_update(tmp_path,profile):
    bundle=data(tmp_path/'data',dim=1024);seed_everything(2026)
    model=H2MIL(bundle,profile=profile,hidden=16,batch_size=4);scope=original();ref=reference(model,scope)
    traces=[]
    class TracedAdam(torch.optim.Adam):
        def step(self,closure=None):
            value=super().step(closure);traces.append(snapshot(ref,self));return value
    ref_optimizer=TracedAdam(ref.parameters(),lr=5e-5,weight_decay=5e-4)
    optimizer=model.make_optimizer(5e-5,5e-4);rows=bundle.records('train')
    graphs=[[model.assets.load(r) for r in rows[i:i+4]] for i in range(0,len(rows),4)]
    labels=[[r['target'] for r in rows[i:i+4]] for i in range(0,len(rows),4)]
    checked=0
    for epoch in range(4):
        seed_everything(2026+epoch);before=rng_state();traces.clear()
        auc=scope['epoch'](ref,ref_optimizer,nn.CrossEntropyLoss(),torch.device('cpu'),graphs,labels)
        expected=list(traces);restore_rng(before);step=optimizer.step
        def checked_step(*a,_step=step,_expected=expected,**kw):
            nonlocal checked
            value=_step(*a,**kw);exact(snapshot(model.network,optimizer),_expected.pop(0));checked+=1;return value
        optimizer.step=checked_step;losses=model.epoch(bundle,optimizer);optimizer.step=step
        assert not expected and len(losses)==len(graphs)
        if not model.corrected:assert model.train_auc==auc
        model.finish_epoch(bundle,'cpu',{})
    assert checked==20 and model.optimizer_steps==20
    write_json(tmp_path/'h2mil_author_training.json',{'state':'passed','profile':profile,'source_commit':COMMIT,
        'epochs':4,'optimizer_updates':checked,'input_dimension':1024,
        'comparison':'Actual main.py epoch/batch body and original dense IHPool; all parameters/gradients/Adam moments/all RNG exact after every summed-batch update. Explicit common shape and corrected-profile head compatibility only.'})

@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('boundary',[1,2])
def test_actual_unified_resume_selection_and_structural_export(tmp_path,profile,boundary):
    bundle=data(tmp_path/'data');cfg={'epochs':3,'threads':1,'model':{'hidden':16,'batch_size':4}}
    full=train(bundle.root,profile,tmp_path/'full',config=cfg,seed=7)
    part=train(bundle.root,profile,tmp_path/'part',config=cfg,seed=7,stop_after=boundary)
    train(bundle.root,profile,part,config=cfg,seed=7,resume=True)
    a,b=[torch.load(p/'last.pt',weights_only=False) for p in [full,part]]
    for k in ['model','optimizers','history','rng','best','bad']:exact(a[k],b[k])
    expected=max(i for i,h in enumerate(a['history']) if h['validation']['h2_val_auc']==max(r['validation']['h2_val_auc'] for r in a['history']))
    assert torch.load(full/'best.pt',weights_only=False)['epoch']==expected
    assert a['model']['_extra_state']['optimizer_steps']==15
    assert evaluate(full)==evaluate(part)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'),pd.read_csv(part/'test_predictions.csv'))
    for entry in read_json(full/'test_instances/manifest.json')['records']:
        with h5py.File(entry['path']) as h:
            np.testing.assert_array_equal(h['all_nodes_final_cluster'][:],h['stage_1/membership'][:][h['stage_0/membership'][:]])
            np.testing.assert_array_equal(h['fine_final_cluster'][:],h['all_nodes_final_cluster'][:][h['original_node_type'][:]==2])
            assert h['coords'].shape==(16,2) and h['stage_0/fitness_before'].shape==(21,)
    assert read_json(full/'method_protocol.json')['completed_epochs']==3

@pytest.mark.parametrize('classes',[2,3])
def test_dense_network_full_forward_and_all_gradients(tmp_path,classes):
    bundle=data(tmp_path/'data',classes=classes,dim=1024)
    model=H2MIL(bundle,hidden=16);ref=reference(model,original())
    row=next(r for r in bundle.records('train') if r['target']==classes-1)
    graph=model.assets.load(row);graph.x.requires_grad_();other=graph.clone();other.x=graph.x.detach().clone().requires_grad_()
    torch.manual_seed(77);result=model.forward_graph(graph)
    torch.manual_seed(77);expected=ref(other)[0]
    exact(result['logits'],expected)
    target=torch.tensor([classes-1]);F.cross_entropy(result['logits'],target).backward();F.cross_entropy(expected,target).backward()
    exact(graph.x.grad,other.x.grad)
    for p,q in zip(model.network.parameters(),ref.parameters()):exact(p.grad,q.grad)
    assert model.network.pool_1.weight_1.grad is None and model.network.conv1.t_att_l.grad is not None

@pytest.mark.parametrize('change',['wrong_parent','missing_parent','duplicate_grid','fractional_grid','wrong_coordinates','wrong_dimension','missing_thumbnail','outcome_column'])
def test_real_three_level_inputs_required_before_preparation(tmp_path,change):
    bundle=data(tmp_path/'data');row=bundle.records('train')[0];schema=copy.deepcopy(bundle.h2mil.meta['schema'])
    path=bundle.root/'hierarchies'/f"{row['slide_id']}.h5"
    if change=='wrong_coordinates':
        with h5py.File(bundle.root/row['path'],'r+') as h:h['coords'][0,0]+=1
    elif change=='outcome_column':
        frame=pd.read_csv(bundle.root/'hierarchies.csv');frame['label']='class_0';frame.to_csv(bundle.root/'bad.csv',index=False)
    else:
        with h5py.File(path,'r+') as h:
            if change=='wrong_parent':h['fine_parent'][0]=(h['fine_parent'][0]+1)%4
            elif change=='missing_parent':h['fine_parent'][:]=0
            elif change=='duplicate_grid':h['fine_grid'][0]=h['fine_grid'][1]
            elif change=='fractional_grid':
                v=h['fine_grid'][:].astype(float);v[0,0]+=.5;del h['fine_grid'];h['fine_grid']=v
            elif change=='wrong_dimension':del h['coarse_features'];h['coarse_features']=np.zeros((4,31),dtype='float32')
            else:del h['thumbnail_features']
    with pytest.raises(ValueError):prepare(bundle,bundle.root/('bad.csv' if change=='outcome_column' else 'hierarchies.csv'),schema,tmp_path/'bad.json')

def test_rehashed_invalid_geometry_cannot_bypass_semantic_checks(tmp_path):
    bundle=data(tmp_path/'data');row=bundle.records('train')[0]
    path=bundle.root/'hierarchies'/f"{row['slide_id']}.h5"
    with h5py.File(path,'r+') as h:h['fine_parent'][0]=(h['fine_parent'][0]+1)%4
    meta=read_json(bundle.h2mil.path);meta['slides'][row['slide_id']]['sha256']=digest_file(path)
    write_json(bundle.h2mil.path,meta)
    with pytest.raises(ValueError):H2Hierarchy(bundle).load(row)

def test_missing_hierarchy_is_not_a_generic_pooling_fallback(tmp_path):
    from milplus.demo import create_demo
    bundle=Bundle(create_demo(tmp_path/'data',patients=20))
    assert require_runnable('H2-MIL')['id']=='h2mil_repaired'
    with pytest.raises(ValueError,match='H2-MIL'):check_compatibility(require_runnable('H2-MIL'),bundle)


@pytest.mark.parametrize('side',[2,4])
def test_actual_author_notebook_graph_constructor_all_rows(tmp_path,side):
    bundle=data(tmp_path/'data',side=side);root,sha=verified_source('h2mil')
    notebook=json.loads(pinned_text(root,sha,'code/generate_heterougeneous_graph.ipynb'))
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    function=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='get_edge_index_2')
    scope={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_original_h2_graph.py','exec'),scope)  # noqa: S102 -- pinned author's actual notebook function
    for row in bundle.rows.to_dict('records'):
        graph=bundle.h2mil.load(row);path=bundle.root/'hierarchies'/f"{row['slide_id']}.h5"
        with h5py.File(path) as h:
            coarse=h['coarse_grid'][:];fine=h['fine_grid'][:];parents=h['fine_parent'][:]
            values={'-1':h['thumbnail_features'][0]}
            for i,(x,y) in enumerate(coarse):values[f'{2*x}-{2*x+1}-{2*y}-{2*y+1}-{x}-{y}.png']=h['coarse_features'][i]
        x,_=bundle.load(row)
        for i,(a,b) in enumerate(fine):values[f'{a}-{b}.png']=x[i].numpy()
        scope['t_all_feature']={row['slide_id']:values}
        expected=scope['get_edge_index_2'](row['slide_id'])
        assert sorted(set(zip(*expected)))==list(map(tuple,graph.edge_index_tree_8nb.T.tolist()))
        exact(graph.x,torch.from_numpy(np.stack(list(values.values()))))
        exact(graph.node_tree,torch.tensor(np.r_[-1,np.zeros(len(coarse),dtype=np.int64),parents+1]))
        expected_xy=np.r_[np.zeros((1,2)),coarse/coarse.max(0),fine/fine.max(0)].astype('float32')
        exact(graph.x_y_index,torch.from_numpy(expected_xy))
    write_json(tmp_path/'h2mil_author_graph.json',{'state':'passed','source_commit':COMMIT,'slides':len(bundle.rows),
        'nodes_per_slide':1+5*side*side,'scope':'Actual author notebook constructor, every edge/feature/parent/normalized coordinate on shuffled physical fine rows'})


def independent_pool(pool,x,edge,node_type,tree,coordinates):
    """Independent NumPy hard assignments, Python edge contraction and explicit means."""
    features=x.detach().numpy();kind=node_type.numpy();parents=tree.numpy();xy=coordinates.numpy()
    mids=np.flatnonzero(kind==1);fine=np.flatnonzero(kind==2)
    w1=pool.weight_1.detach().numpy();w2=pool.weight_2.detach().numpy()
    fit1=np.tanh((features[mids]*w1).sum(-1)/np.linalg.norm(w1))
    fit2=np.tanh((features[fine]*w2).sum(-1)/np.linalg.norm(w2))
    def assign(positions,fitness,seeds):
        a=np.c_[positions,fitness].astype(np.float64);selected=a[seeds]
        distance=np.linalg.norm(selected[:,None,:2]-a[None,:,:2],axis=-1)+np.abs(selected[:,None,2]-a[None,:,2])
        return distance.argmin(0)
    n=len(mids);step=math.ceil(1/pool.ratio) if pool.ratio<1 else 1 if n<pool.ratio else math.ceil(n/pool.ratio)
    first=assign(xy[mids],fit1,np.argsort(fit1)[::step]);second_parent=first[parents[fine]-parents[fine].min()]
    second=np.zeros(len(fine),dtype=np.int64)
    new_xy=[np.zeros(2),*[xy[mids][first==i].mean(0) for i in np.unique(first)]]
    new_tree=[-1]+[0]*len(np.unique(first))
    for k in range(len(np.unique(first))):
        selected=np.flatnonzero(second_parent==k);fitness=fit2[selected];n=len(selected)
        step=math.ceil(1/pool.ratio) if pool.ratio<1 else 1 if n==1 else n-1
        local=assign(xy[fine][selected],fitness,np.argsort(fitness)[::step])
        new_xy.extend(xy[fine][selected][local==i].mean(0) for i in np.unique(local))
        local=local+2*len(np.unique(second));second[second_parent==k]=local
        new_tree.extend([k+1]*len(np.unique(local)))
    cluster=np.empty(len(x),dtype=np.int64);cluster[kind==0]=0;cluster[mids]=first+1;cluster[fine]=second+len(first)+100
    _,cluster=np.unique(cluster,return_inverse=True)
    original_edges=set(map(tuple,edge.T.tolist()))|{(i,i) for i in range(len(x))}
    pooled_edges=sorted({(int(cluster[a]),int(cluster[b])) for a,b in original_edges})
    values=torch.stack([x[torch.from_numpy(cluster==i)].mean(0) for i in range(cluster.max()+1)])
    kinds=np.r_[0,np.ones(len(np.unique(first)),dtype=np.int64),np.full(len(np.unique(second)),2,dtype=np.int64)]
    return {'features':values,'edges':torch.tensor(pooled_edges,dtype=torch.long).T,'cluster':torch.from_numpy(cluster),
        'node_type':torch.from_numpy(kinds),'tree':torch.tensor(new_tree,dtype=torch.long),
        'fitness':torch.tensor(np.r_[0,fit1,fit2]),'coordinates':torch.tensor(np.array(new_xy))}


@pytest.mark.parametrize('ratio',[.1,.2,.3,.4,.5,1.,4.])
@pytest.mark.parametrize('seed',[7,2026,9901])
def test_two_real_pool_stages_against_independent_assignments_edges_and_gradients(tmp_path,ratio,seed):
    bundle=data(tmp_path/'data',dim=8,side=4);graph=bundle.h2mil.load(bundle.records('train')[0])
    torch.manual_seed(seed);model=H2MIL(bundle,hidden=8)
    pool_type=type(model.network.pool_1);pool=pool_type(8,ratio=ratio)
    x=torch.randn(len(graph.x),8,requires_grad=True)
    edge,kind,tree,xy=graph.edge_index_tree_8nb,graph.node_type,graph.node_tree,graph.x_y_index*2-1
    checks=[]
    for stage in range(2):
        actual=pool(x,edge,kind,tree,xy);expected=independent_pool(pool,x,edge,kind,tree,xy)
        torch.testing.assert_close(actual[0],expected['features'],rtol=1e-6,atol=1e-6)
        for index,key in [(1,'edges'),(4,'cluster'),(5,'node_type'),(6,'tree')]:exact(actual[index],expected[key])
        for index,key in [(7,'fitness'),(8,'coordinates')]:
            torch.testing.assert_close(actual[index],expected[key].to(actual[index].dtype),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(torch.autograd.grad(actual[0].square().sum(),x)[0],
            torch.autograd.grad(expected['features'].square().sum(),x)[0],rtol=1e-6,atol=1e-6)
        checks.append({'stage':stage,'nodes_before':len(x),'nodes_after':len(actual[0]),'state':'passed'})
        x=actual[0].detach().requires_grad_();edge,kind,tree,xy=actual[1],actual[5],actual[6],actual[8]
        pool=pool_type(8,ratio=4)
    write_json(tmp_path/'h2mil_pool_oracle.json',{'state':'passed','seed':seed,'ratio':ratio,'checks':checks,
        'scope':'Independent hard-assignment, sparse-contraction and input-gradient equations; released ratio rounding preserved'})


def test_resolution_and_node_attention_independent_primary_equations(tmp_path):
    bundle=data(tmp_path/'data',dim=8);graph=bundle.h2mil.load(bundle.records('train')[0])
    seed_everything(2026);model=H2MIL(bundle,hidden=8);conv=model.network.conv1
    x=graph.x.detach().clone().requires_grad_();edge=graph.edge_index_tree_8nb;kind=graph.node_type
    actual=conv(x,edge,kind);actual.square().sum().backward()
    gradients={name:p.grad.clone() for name,p in conv.named_parameters()};input_grad=x.grad.clone();conv.zero_grad()
    other=x.detach().clone().requires_grad_();n=len(other)
    # Match the author's branch construction order: the two shared-input
    # backward paths otherwise accumulate their float32 gradients in reverse.
    projected=conv.lin_l(other).view(-1,1,8);node_left=(projected*conv.att_l).sum(-1);node_right=(projected*conv.att_r).sum(-1)
    groups=kind[edge[0]]+3*edge[1];grouped=scatter(other[edge[0]],groups,dim=0,reduce='mean')
    t=torch.cat([other,grouped]);projected_type=conv.t_lin_l(t).view(-1,1,8)
    left=(projected_type*conv.t_att_l).sum(-1);right=(projected_type*conv.t_att_r).sum(-1)
    used=torch.unique(groups,sorted=True);targets=used//3
    type_attention=softmax(F.leaky_relu(left[used+n]+right[targets],.2),targets,num_nodes=len(t))
    node_attention=softmax(F.leaky_relu(node_left[edge[0]]+node_right[edge[1]],.2),groups)
    attention=node_attention*type_attention[torch.searchsorted(used,groups)]
    expected=scatter(projected[edge[0]]*attention.unsqueeze(-1),edge[1],dim=0,dim_size=n,reduce='sum').view(n,8)+conv.bias
    exact(actual,expected);expected.square().sum().backward();exact(input_grad,other.grad)
    for name,p in conv.named_parameters():exact(gradients[name],p.grad)
    write_json(tmp_path/'h2mil_attention_oracle.json',{'state':'passed','scope':'Independent resolution- and node-attention equations, bitwise full outputs/all parameter/input gradients'})


@pytest.mark.parametrize('profile',PROFILES)
def test_original_selector_exact_boundaries_and_last_tie(tmp_path,profile):
    bundle=data(tmp_path/'data');model=H2MIL(bundle,profile=profile,hidden=8)
    root,sha=verified_source('h2mil');tree=ast.parse(pinned_text(root,sha,'code/main.py'))
    selection=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and
        'auc_for_val >= best_auc_val_fold' in ast.unparse(n.test))
    condition=compile(ast.Expression(selection),'actual_original_h2_selection.py','eval')
    checked=0
    for epoch in [0,49,50,51,59]:
        for training_auc in [.799,.8,.800001]:
            for objective in [.69,.7,.71]:
                scope={'epoch_num':epoch,'epochs':60,'auc_of_train':training_auc,
                    'auc_for_val':objective,'best_auc_val_fold':.7}
                model.completed_epochs=epoch+1;model.train_auc=training_auc
                assert model.selection_improved(objective,.7)==eval(condition,{},scope)
                checked+=1
    write_json(tmp_path/'h2mil_selector.json',{'state':'passed','profile':profile,
        'actual_original_condition':ast.unparse(selection),'boundary_checks':checked})


@pytest.mark.parametrize('which',['primary','hierarchy'])
def test_input_changes_after_binding_are_rejected(tmp_path,which):
    bundle=data(tmp_path/'data');model=H2MIL(bundle,hidden=8);row=bundle.records('train')[0]
    path=bundle.root/row['path'] if which=='primary' else bundle.root/'hierarchies'/f"{row['slide_id']}.h5"
    with h5py.File(path,'r+') as h:h['features' if which=='primary' else 'thumbnail_features'][0,0]+=1
    with pytest.raises(ValueError,match='changed'):model.forward_unit(bundle,[row],'cpu',{})


def test_checkpoint_cannot_be_loaded_with_a_different_pooling_contract(tmp_path):
    bundle=data(tmp_path/'data');model=H2MIL(bundle,hidden=8)
    altered=H2MIL(bundle,hidden=8,ratio1=.2)
    with pytest.raises(ValueError,match='contract'):altered.load_state_dict(model.state_dict())


def test_training_never_forwards_held_out_test_graphs(tmp_path,monkeypatch):
    bundle=data(tmp_path/'data');test={r['slide_id'] for r in bundle.records('test')};forward=H2MIL.forward_unit
    def guarded(self,bundle,rows,*args,**kwargs):
        if any(row['slide_id'] in test for row in rows):raise AssertionError('TEST entered training or validation inference')
        return forward(self,bundle,rows,*args,**kwargs)
    # Static geometry/hash validation is allowed; TEST model inference is not.
    monkeypatch.setattr(H2MIL,'forward_unit',guarded)
    run=train(bundle.root,'H2-MIL',tmp_path/'run',config={'epochs':2,'threads':1,'model':{'hidden':8,'batch_size':4}})
    assert not (run/'test_predictions.csv').exists()
    monkeypatch.setattr(H2MIL,'forward_unit',forward)
    evaluate(run)
    assert set(pd.read_csv(run/'test_predictions.csv').slide_id)==test


def test_actual_prepare_cli_and_three_class_training(tmp_path):
    from milplus.cli import dispatch, parser
    bundle=data(tmp_path/'data',classes=3);schema=bundle.h2mil.meta['schema']
    write_json(tmp_path/'schema.json',schema)
    result=dispatch(parser().parse_args(['prepare-h2mil','--data',str(bundle.root),
        '--index',str(bundle.root/'hierarchies.csv'),'--schema',str(tmp_path/'schema.json'),
        '--output',str(bundle.root/'cli_hierarchy.json')]))
    assert read_json(result['manifest'])==read_json(bundle.h2mil.path)
    run=train(bundle.root,'H2-MIL',tmp_path/'run',config={'epochs':2,'threads':1,'model':{'hidden':8,'batch_size':4}})
    result=evaluate(run);assert np.isfinite(result['macro_auc'])
    frame=pd.read_csv(run/'test_predictions.csv')
    np.testing.assert_allclose(frame[['p_0','p_1','p_2']].sum(1),1.,atol=1e-6,rtol=0)
    with pytest.raises(ValueError,match='binary'):H2MIL(bundle,profile='h2mil_source_compat',hidden=8)

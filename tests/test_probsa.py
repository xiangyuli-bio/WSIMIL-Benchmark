"""ProbSA author whole networks/loops, independent KL and variable-bag repairs."""
import ast
import copy
import json
import subprocess
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scipy.spatial import KDTree
from torch import Tensor, nn
from torch.nn.attention import SDPBackend

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, predict, restore_rng, rng_state, train, units
from milplus.probsa import PROFILES, ProbSA, build_graph, definitions, pad_batch, probsa_epoch
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def single_thread():torch.set_num_threads(1)


def fixture(tmp_path, profile='probsa_abmil_probmean', equal_length=False, **options):
    root=create_demo(tmp_path/'data', patients=20, dimension=8)
    b=Bundle(root)
    if equal_length:
        for row in b.rows.to_dict('records'):
            with h5py.File(root/row['path'],'r+') as h:
                for key in ['features','coords']:
                    array=h[key][:16];del h[key];h[key]=array
    opts={'hidden':16,'attention_dim':8,'train_samples':4,'test_samples':8,'cycles':2,**options}
    return ProbSA(b,profile,**opts),b,opts


def reference_definitions():
    root,sha=verified_source('probsamil')
    scope={'torch':torch,'np':np,'Tensor':Tensor,'TensorDict':dict,'Any':Any,'KDTree':KDTree,
           'SDP_BACKEND':[SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]}
    def load(path,names):
        source=subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)
        selected=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        assert {n.name for n in selected}==set(names)
        exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent pinned Git definitions
    for path,names in [('torchmil/nn/utils.py',['LazyLinear','masked_softmax','get_feat_dim']),
      ('torchmil/models/mil_model.py',['MILModel','MILModelWrapper','get_args_names']),
      ('torchmil/nn/attention/prob_smooth_attention_pool.py',['ProbSmoothAttentionPool']),
      ('torchmil/nn/attention/multihead_self_attention.py',['MultiheadSelfAttention']),
      ('torchmil/nn/transformers/encoder.py',['Encoder']),('torchmil/nn/transformers/layer.py',['Layer']),
      ('torchmil/nn/transformers/conventional_transformer.py',['TransformerLayer','TransformerEncoder']),
      ('torchmil/models/prob_smooth_abmil.py',['ProbSmoothABMIL']),
      ('torchmil/models/transformer_prob_smooth_abmil.py',['TransformerProbSmoothABMIL']),
      ('torchmil/utils/annealing_scheduler.py',['AnnealingScheduler','ConstantAnnealingScheduler','CyclicalAnnealingScheduler'])]:load(path,names)
    return scope,load


def native(model):
    scope,load=reference_definitions()
    feature=nn.Sequential(nn.Linear(model.dim,model.hidden),nn.ReLU())
    common={'in_shape':(model.dim,),'feat_ext':feature,'n_samples_train':model.train_samples,'n_samples_test':model.test_samples}
    net=(scope['TransformerProbSmoothABMIL'](pool_att_dim=model.attention_dim,transf_att_dim=model.hidden,transf_n_layers=2,transf_n_heads=8,**common)
         if model.transformer else scope['ProbSmoothABMIL'](att_dim=model.attention_dim,**common))
    net.load_state_dict(model.network.state_dict())
    return net,scope,load


def nested(a,b,tol=0):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=tol)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:nested(a[k],b[k],tol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):nested(x,y,tol)
    else:assert a==b


@pytest.mark.parametrize('transformer',[False,True])
@pytest.mark.parametrize('n',[1,16])
def test_complete_author_network_loss_and_all_gradients(tmp_path,transformer,n):
    m,b,_=fixture(tmp_path,profile='probsa_transformer_source' if transformer else 'probsa_abmil_source')
    author,_,_=native(m)
    row=b.records('train')[0];x,c=b.load(row);x,c=x[:n],c[:n]
    graph=build_graph(x,c,256.).to_dense().unsqueeze(0)
    state=rng_state();pred,terms=m.network.compute_loss(torch.ones(1),x.unsqueeze(0),graph)
    sum(terms.values()).backward();restore_rng(state)
    q,ref=author.compute_loss(torch.ones(1),x.unsqueeze(0),graph)
    sum(ref.values()).backward()
    nested(pred,q,tol=1e-7);nested(terms,ref,tol=1e-7)
    for (name,p),(other,r) in zip(m.network.named_parameters(),author.named_parameters()):
        assert name==other and p.grad is not None
        nested(p.grad,r.grad,tol=2e-7)


@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('lengths',[(1,1),(3,7),(7,7)])
def test_variable_bag_kl_matches_independent_dense_equation(lengths,sparse):
    pool=definitions()['ProbSmoothAttentionPool'](3,att_dim=4,n_samples_train=3)
    bags=[torch.randn(n,3) for n in lengths]
    graphs=[build_graph(x,torch.stack([torch.arange(len(x)),torch.zeros(len(x))],1),1.) for x in bags]
    _x,mask,adj=pad_batch(bags,graphs,'cpu');pool.valid_mask=mask
    mu=torch.randn(*mask.shape,1,requires_grad=True);logv=torch.randn_like(mu,requires_grad=True)
    result=pool._kl_div(mu,logv,adj if sparse else adj.to_dense())
    manual=[]
    for i,n in enumerate(lengths):
        q=mu[i,:n,0];l=logv[i,:n,0];v=l.exp();a=graphs[i].to_dense()
        # Original library KL convention (constant omitted); no padding in this oracle.
        manual.append((q@(torch.eye(n)-a)@q+((1-a.diag())*v).sum()-.5*l.sum())/n)
    expected=torch.stack(manual).mean()
    torch.testing.assert_close(result,expected,rtol=1e-6,atol=1e-7)
    grads=torch.autograd.grad(result,(mu,logv),retain_graph=True)
    refs=torch.autograd.grad(expected,(mu,logv))
    for g,r in zip(grads,refs):torch.testing.assert_close(g,r,rtol=1e-5,atol=1e-7)
    assert not grads[0][~mask].any() and not grads[1][~mask].any()


def test_sparse_kl_never_densifies_graph(monkeypatch):
    pool=definitions()['ProbSmoothAttentionPool'](3,att_dim=4)
    pool.valid_mask=torch.ones(1,1000,dtype=torch.bool)
    adj=torch.sparse_coo_tensor(torch.empty(3,0,dtype=torch.long),torch.empty(0),(1,1000,1000)).coalesce()
    monkeypatch.setattr(torch.Tensor,'to_dense',lambda *a,**k:(_ for _ in ()).throw(AssertionError('dense graph allocation')))
    loss=pool._kl_div(torch.zeros(1,1000,1),torch.zeros(1,1000,1),adj)
    assert loss==1


class MeanMetric:
    def __init__(self,*a,**k):self.values=[]
    def reset(self):self.values=[]
    def update(self,value,*a):self.values.append(float(value))
    def compute(self):return torch.tensor(np.mean(self.values))


class Progress:
    def __init__(self,iterator,**kwargs):self.iterator=iterator
    def __iter__(self):return iter(self.iterator)
    def set_description(self,*a):pass
    def set_postfix(self,*a):pass
    def close(self):pass


class Batch(dict):
    def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})


@pytest.mark.parametrize('transformer',[False,True])
@pytest.mark.parametrize('kl_weight',['cyclical',1.])
def test_original_trainer_full_epochs_adam_and_schedules(tmp_path,transformer,kl_weight):
    m,b,_=fixture(tmp_path,profile='probsa_transformer_source' if transformer else 'probsa_abmil_source',equal_length=True,kl_weight=kl_weight,batch_size=2)
    author,scope,load=native(m)
    cfg={'epochs':3,'max_instances':None,'grad_clip':None,'selection_metric':'probsa_bag_auc','weight_decay':0.}
    m.configure_training(cfg);opt=m.make_optimizer(1e-4,0.)
    refopt=torch.optim.Adam(author.parameters(),lr=1e-4)
    lr=torch.optim.lr_scheduler.LinearLR(refopt,start_factor=.1,total_iters=10)
    anneal=scope['CyclicalAnnealingScheduler'](m.total_steps/m.cycles,cycle_prop=.8) if kl_weight=='cyclical' else scope['ConstantAnnealingScheduler'](1.)
    scope.update(torchmetrics=SimpleNamespace(MeanMetric=MeanMetric,Metric=MeanMetric,Accuracy=MeanMetric),tqdm=Progress,deepcopy=copy.deepcopy)
    scope['AnnealingScheduler']=scope['AnnealingScheduler'];load('torchmil/utils/trainer.py',['Trainer'])
    trainer=scope['Trainer'].__new__(scope['Trainer'])
    trainer.model=scope['MILModelWrapper'](author);trainer.optimizer=refopt;trainer.device='cpu'
    trainer.disable_pbar=True;trainer.metrics_dict={};trainer.annealing_scheduler_dict={'KLDiv':anneal}
    rows=b.records('train');max_difference=0.;updates=[]
    def collate(batch):
        x,mask,adj=m.batch(batch,'cpu')
        return Batch(Y=torch.tensor([r['target'] for r in batch]),X=x,mask=mask,adj=adj.to_dense())
    for epoch in range(3):
        m.train();m.start_epoch(epoch);initial=rng_state()
        losses=probsa_epoch(m,b,units(b,'train'),opt,torch.device('cpu'),cfg);after=rng_state();m.finish_epoch(b,'cpu',cfg)
        restore_rng(initial)
        loader=torch.utils.data.DataLoader(rows,batch_size=2,shuffle=True,collate_fn=collate)
        metrics=trainer._shared_loop(loader,epoch=epoch,mode='train')
        lr.step()
        for name,p in m.network.state_dict().items():
            q=author.state_dict()[name];max_difference=max(max_difference,float((p-q).abs().max()))
            nested(p,q,tol=2e-6)
        nested(opt.state_dict(),refopt.state_dict(),tol=2e-6)
        nested(torch.get_rng_state(),after['torch'])
        assert float(np.mean([v['loss'] for v in losses]))==pytest.approx(metrics['train/loss'],abs=1e-6)
        assert m.coefficient()==float(anneal()) and opt.param_groups[0]['lr']==refopt.param_groups[0]['lr']
        updates.extend([v['kl_coefficient'] for v in losses])
    assert m.network.pool.log_diag_Sigma_nn.weight.grad.abs().sum()>0
    (tmp_path/'probsa_author_training.json').write_text(json.dumps({'state':'passed','transformer':transformer,'kl_weight':kl_weight,
        'epochs':3,'optimizer_updates':m.optimizer_steps,'max_parameter_abs':max_difference,'coefficients':updates,
        'scope':'Direct pinned author Trainer._shared_loop, MILModelWrapper and complete model; independent dense graph reference versus sparse repair, fixed 16-instance bags'}))


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_engine_resume_and_label_independent_repeated_inference(tmp_path,profile):
    _,b,opts=fixture(tmp_path,profile=profile)
    cfg={'epochs':3,'threads':1,'model':opts}
    a=train(b.root,profile,tmp_path/'full',seed=27,config=cfg)
    c=train(b.root,profile,tmp_path/'resume',seed=27,config=cfg,stop_after=1)
    train(b.root,profile,c,seed=27,config=cfg,resume=True)
    first=torch.load(a/'last.pt',weights_only=False);second=torch.load(c/'last.pt',weights_only=False)
    for key in ['model','optimizers','history','rng']:nested(first[key],second[key])
    result=evaluate(a);sha=(a/'test_predictions.csv').read_bytes();assert result==evaluate(a) and sha==(a/'test_predictions.csv').read_bytes()
    manifest=json.loads((a/'test_instances/manifest.json').read_text())
    for entry in manifest['records']:
        with h5py.File(entry['path']) as h:
            assert np.isfinite(h['posterior_variance'][:]).all() and h['attention'][:].sum()==pytest.approx(1.)
    assert len(json.loads((a/'history.json').read_text()))==3


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_prediction_labels_do_not_enter_model(tmp_path,profile):
    m,b,_=fixture(tmp_path,profile=profile);m.eval();cfg={'max_instances':None}
    state=rng_state();one=predict(m,b,profile,'test',torch.device('cpu'),cfg)
    b.rows['target']=1-b.rows.target
    two=predict(m,b,profile,'test',torch.device('cpu'),cfg)
    pd.testing.assert_frame_equal(one[['p_0','p_1']],two[['p_0','p_1']],check_exact=True)
    nested(state['torch'],torch.get_rng_state())


@pytest.mark.parametrize('option,value',[('hidden',0),('train_samples',0),('test_samples',True),('cycles',0),('kl_weight',0.),
    ('kl_weight',float('nan')),('batch_size',0),('transformer_dropout',1.),('max_graph_edges',0),('max_graph_cache_bytes',0)])
def test_reject_invalid_model_parameters(tmp_path,option,value):
    with pytest.raises(ValueError):fixture(tmp_path,**{option:value})


def test_physical_graph_guard_and_singleton():
    x=torch.randn(5,8);c=torch.zeros(5,2)
    with pytest.raises(ValueError):build_graph(x,c,256.)
    c=torch.stack([torch.arange(5),torch.zeros(5)],1)
    with pytest.raises(ValueError,match='before allocation'):build_graph(x,c,256.,max_edges=2)
    graph=build_graph(x[:1],c[:1],256.)
    assert graph.shape==(1,1) and graph._nnz()==0


@pytest.mark.parametrize('task',['survival','multiclass','patient'])
def test_reject_unsupported_task(tmp_path,task):
    r=create_demo(tmp_path/'data',patients=30,dimension=8,classes=3 if task=='multiclass' else 2,survival=task=='survival')
    if task=='patient':
        p=r/'task.yaml';s=yaml.safe_load(p.read_text());s['training_unit']='patient';p.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError):ProbSA(Bundle(r))


def test_graph_cache_eviction_retains_same_graph(tmp_path):
    m,b,_=fixture(tmp_path,max_graph_cache_bytes=800)
    row=b.records('train')[0];x,a=m.load(row)
    for r in b.records('train')[1:]:m.load(r)
    _,q=m.load(row)
    torch.testing.assert_close(a.to_dense(),q.to_dense(),rtol=0,atol=0)
    assert sum(g._nnz()*20 for g in m.graphs.values())<=800
    assert x.shape[1]==8


@pytest.mark.parametrize('source',[False,True])
def test_source_logit_mean_and_paper_probability_mean_match_declared_oracles(tmp_path,source):
    from milplus.io import digest_object
    from milplus.stochastic import inference_rng
    m,b,_=fixture(tmp_path,profile='probsa_abmil_source' if source else 'probsa_abmil_probmean',test_samples=128)
    m.eval()
    with torch.no_grad():m.network.classifier.module.weight.mul_(30.)
    row=b.records('test')[0];x,_=b.load(row)
    seed=int(digest_object({'seed':m.prediction_seed,'slide':row['slide_id']})[:15],16)
    with inference_rng('cpu',seed):logits=m.network(x.unsqueeze(0),adj=None,return_samples=True)
    result=m.forward_unit(b,[row],torch.device('cpu'),{},False)
    expected=logits.mean(1).sigmoid() if source else logits.sigmoid().mean(1)
    torch.testing.assert_close(result['probabilities'][:,1],expected,rtol=0,atol=0)
    assert (logits.mean(1).sigmoid()-logits.sigmoid().mean(1)).abs().max()>1e-5


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_all_routes_reject_mc_oversize_before_network(tmp_path,profile,monkeypatch):
    m,_,_=fixture(tmp_path,profile=profile,max_sample_bytes=1)
    monkeypatch.setattr(m.network,'forward',lambda *a,**k:(_ for _ in ()).throw(AssertionError('network was allocated')))
    with pytest.raises(ValueError,match='working set'):m(torch.randn(10,8))


def test_transformer_attention_budget_before_forward(tmp_path):
    m,_,_=fixture(tmp_path,profile='probsa_transformer_source',max_attention_bytes=1)
    with pytest.raises(ValueError,match='transformer working set'):m(torch.randn(10,8))


@pytest.mark.parametrize('epoch',[0,1,5,10,11])
def test_linear_lr_warmup_matches_source_paper(tmp_path,epoch):
    m,_,_=fixture(tmp_path,kl_weight=1.);opt=m.make_optimizer(.0001,0.)
    for _ in range(epoch):opt.step();m.lr_scheduler.step()
    assert opt.param_groups[0]['lr']==pytest.approx(.0001*(.1+.9*min(epoch,10)/10),rel=1e-12)


@pytest.mark.parametrize('step',[0,1,2,5,8,9,10,11,18,19,20])
def test_author_cyclic_scheduler_boundaries(step):
    scheduler=definitions()['CyclicalAnnealingScheduler'](10,cycle_prop=.8)
    for _ in range(step):scheduler.step()
    expected=0. if step==0 else min(((step-1)%10)/8,1.)
    assert scheduler()==expected


def test_optional_class_balance_uses_only_train_rows(tmp_path):
    m,b,_=fixture(tmp_path,balance_loss=True)
    train_indices=b.rows.index[b.rows.subset=='train']
    b.rows.loc[train_indices[:3],'target']=1
    y=b.rows[b.rows.subset=='train'].target.tolist();expected=y.count(0)/y.count(1)
    b.rows.loc[b.rows.subset=='test','target']=1
    m.prepare(b,0,'cpu');assert float(m.network.criterion.pos_weight)==pytest.approx(expected)
    assert 'network.criterion.pos_weight' in m.state_dict()


def test_coordinate_precision_and_instance_order_preserved(tmp_path):
    m,b,_=fixture(tmp_path)
    row=b.records('train')[0]
    with h5py.File(b.root/row['path'],'r+') as h:
        n=len(h['coords']);coords=np.column_stack([np.arange(n)+2**24,np.zeros(n)]).astype('int64')
        del h['coords'];h['coords']=coords
    m.patch_size=1.
    _,graph=m.load(row)
    assert graph._nnz()>0


def test_infinite_posterior_variance_rejected_in_prediction(tmp_path):
    m,_,_=fixture(tmp_path);m.eval()
    with torch.no_grad():m.network.pool.log_diag_Sigma_nn.bias.fill_(1000.)
    with pytest.raises(ValueError,match='nonfinite'):m(torch.ones(3,8))

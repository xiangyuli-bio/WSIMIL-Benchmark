"""Independent original multi-depth attention and complete notebook training-loop checks."""
import ast
import copy
import json
import subprocess

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, predict, restore_rng, rng_state, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.msattention import PROFILES, MultiAttentionMIL, msattention_epoch
from milplus.registry import check_compatibility, inspect_method, require_runnable
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def data(path, dimension=7, classes=2):
    return Bundle(create_demo(path, dimension=dimension, classes=classes,
                              patients=30 if classes == 3 else 20))


def equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def original_class():
    root, sha = verified_source('msattention')
    text = subprocess.check_output(['git', 'show', sha+':model.py'], cwd=root, text=True)
    scope = {'torch': torch, 'nn': nn}
    exec(compile(text, 'author/model.py', 'exec'), scope)  # noqa: S102 -- independent fixed full original file
    return scope['MultiAttentionMIL']


@pytest.mark.parametrize('profile', sorted(PROFILES))
@pytest.mark.parametrize('dimension,classes,n', [(512,2,1), (1280,3,33), (7,2,27)])
@pytest.mark.parametrize('dropout', [0., .5])
def test_native_constructor_forward_three_attentions_and_all_gradients(tmp_path, profile, dimension, classes, n, dropout):
    b = data(tmp_path/'data', dimension, classes)
    state = rng_state()
    m = MultiAttentionMIL(b, profile, dropout=dropout)
    after = rng_state();restore_rng(state)
    ref = original_class()(classes, dimension, use_dropout=dropout > 0, n_dropout=dropout)
    equal(m.network.state_dict(), ref.state_dict());equal(after, rng_state())
    x = torch.randn(n, dimension)
    m.train();ref.train()
    state = rng_state();out = m(x);loss = nn.CrossEntropyLoss()(out['logits'], torch.tensor([classes-1]));loss.backward()
    after = rng_state();restore_rng(state)
    y = ref(x[None]);ref_loss = nn.CrossEntropyLoss()(y[0], torch.tensor([classes-1]));ref_loss.backward()
    equal(out['logits'], y[0]);equal(out['attention'], torch.cat(y[1:]).T);equal(loss, ref_loss);equal(after,rng_state())
    assert out['attention'].shape == (n,3)
    torch.testing.assert_close(out['attention'].sum(0), torch.ones(3))
    for (name,p),(other,q) in zip(m.network.named_parameters(),ref.named_parameters()):
        assert name == other and p.grad is not None
        equal(p.grad,q.grad)


def capture(opt, parameters):
    original = opt.step;records=[]
    def step(*args, **kwargs):
        grads=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
        result=original(*args,**kwargs)
        records.append({'gradients':grads,'parameters':[p.detach().clone() for p in parameters],
                        'optimizer':copy.deepcopy(opt.state_dict())})
        return result
    opt.step=step
    return records


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_actual_notebook_outer_training_loop_all_adam_updates(tmp_path, profile):
    b=data(tmp_path/'data')
    m=MultiAttentionMIL(b,profile)
    ref=original_class()(2,b.dim,use_dropout=True,n_dropout=.5)
    ref.load_state_dict(m.network.state_dict())
    opt=m.make_optimizer(1e-4,0.);refopt=torch.optim.Adam(ref.parameters(),lr=1e-4,betas=(.9,.999))
    actual=capture(opt,list(m.network.parameters()));expected=capture(refopt,list(ref.parameters()))
    rows=units(b,'train')
    ds=[(b.load(g[0])[0],torch.tensor([g[0]['target']])) for g in rows]
    root,sha=verified_source('msattention')
    book=json.loads(subprocess.check_output(['git','show',sha+':main.ipynb'],cwd=root,text=True))
    cell=''.join(book['cells'][2]['source'])
    outer=next(n for n in ast.parse(cell).body if isinstance(n,ast.For))
    class OmitCheckpointIO(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.save':
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    outer=OmitCheckpointIO().visit(outer)
    class Progress:
        def __init__(self,loader):self.loader=loader
        def __iter__(self):return iter(self.loader)
        def set_description(self,*args,**kwargs):pass
    state=rng_state()
    for _ in range(3):
        m.train();msattention_epoch(m,b,rows,opt,'cpu',{})
    after=rng_state();restore_rng(state)
    scope={'model':ref,'optimizer':refopt,'criterion':nn.CrossEntropyLoss(),'epochs':3,'device':'cpu',
           'torch':torch,'nn':nn,'dloader':DataLoader(ds,batch_size=1,shuffle=True,num_workers=0),
           'tqdm':Progress,'print':lambda *args,**kwargs:None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[])),'author/main.ipynb/cell2','exec'),scope)  # noqa: S102 -- actual original complete training loop; checkpoint IO omitted, FM dataset boundary explicit
    equal(actual,expected);equal(m.network.state_dict(),ref.state_dict());equal(after,rng_state())
    assert len(actual)==3*len(rows)
    write_json(tmp_path/'msattention_author_training.json',{'state':'passed','profile':profile,
        'epochs':3,'optimizer_updates':len(actual),'scope':'Original notebook outer training loop, all gradients/Adam moments/parameters/RNG exact; frozen feature data boundary; no WSI score claim'})


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_dropout_eval_semantics_and_rng_preserving_unlabeled_prediction(tmp_path,profile):
    b=data(tmp_path/'data');m=MultiAttentionMIL(b,profile,prediction_samples=3 if profile.endswith('source') else 1)
    m.eval();x=torch.randn(51,b.dim)
    a=m(x);c=m(x)
    assert torch.equal(a['logits'],c['logits']) == profile.endswith('repaired')
    state=rng_state();rows=[b.records('val')[0]]
    a=m.forward_unit(b,rows,'cpu',{});equal(state,rng_state())
    equal(a,m.forward_unit(b,rows,'cpu',{}))
    equal(a,m.forward_unit(b,[{**rows[0],'target':1-rows[0]['target']}],'cpu',{}))
    other={**rows[0],'slide_id':'independent-prediction-id'}
    if profile.endswith('source'):
        assert not torch.equal(a['logits'],m.forward_unit(b,[other],'cpu',{})['logits'])


@pytest.mark.parametrize('profile', sorted(PROFILES))
@pytest.mark.parametrize('evaluation_unit', ['slide', 'patient'])
def test_train_resume_best_checkpoint_export_and_repeat_csv(tmp_path,profile,evaluation_unit):
    b=data(tmp_path/'data')
    task=yaml.safe_load((b.root/'task.yaml').read_text());task['evaluation_unit']=evaluation_unit
    (b.root/'task.yaml').write_text(yaml.safe_dump(task));b=Bundle(b.root)
    cfg={'epochs':3,'threads':1}
    full=train(b.root,profile,tmp_path/'full',config=cfg,seed=9)
    part=train(b.root,profile,tmp_path/'part',config=cfg,seed=9,stop_after=1)
    train(b.root,profile,part,config=cfg,seed=9,resume=True)
    a=torch.load(full/'last.pt',weights_only=False);c=torch.load(part/'last.pt',weights_only=False)
    for key in ['model','optimizers','history','rng']:equal(a[key],c[key])
    assert a['model']['_extra_state']['optimizer_steps']==3*len(units(b,'train'))
    first=evaluate(full);sha=digest_file(full/'test_predictions.csv')
    equal(first,evaluate(full));assert sha==digest_file(full/'test_predictions.csv')
    equal(first,evaluate(part))
    assert digest_file(part/'test_predictions.csv')==sha
    values=read_json(full/'request.json')['config']
    assert values['lr']==1e-4 and values['weight_decay']==0 and values['grad_clip'] is None
    state=torch.load(full/'best.pt',weights_only=False)
    m=MultiAttentionMIL(b,profile);m.load_state_dict(state['model'])
    assert m.protocol()['evaluation_unit']==evaluation_unit
    predict(m,b,profile,'test','cpu',{'max_instances':None},instance_output=tmp_path/'attention')
    report=read_json(tmp_path/'attention/manifest.json')
    for r in report['records']:
        with h5py.File(r['path']) as h:
            assert h['attention'].shape[1]==3 and len(h['coords'])==len(h['attention'])
            np.testing.assert_allclose(h['attention'][:].sum(0),np.ones(3),atol=2e-7)
    check_compatibility(require_runnable(profile),b)


@pytest.mark.parametrize('opts',[{'dropout':float('nan')},{'dropout':-1},{'dropout':1},
                                 {'prediction_samples':0},{'prediction_samples':True},{'prediction_seed':-1},
                                 {'prediction_seed':False}])
def test_invalid_options_rejected(tmp_path,opts):
    with pytest.raises(ValueError):MultiAttentionMIL(data(tmp_path/'data'),**opts)


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_no_feature_dimension_squeezing_or_fake_instances(tmp_path,profile):
    b=data(tmp_path/'data');m=MultiAttentionMIL(b,profile)
    for x in [torch.empty(0,b.dim),torch.randn(2,b.dim+1),torch.zeros(3,b.dim,dtype=torch.long),
              torch.full((3,b.dim),float('nan'))]:
        with pytest.raises(ValueError):m(x)
    for option in [{'max_instances':4},{'grad_clip':1.},{'weight_decay':1e-4},{'selection_metric':'macro_auc'}]:
        cfg={'epochs':3,'max_instances':None,'grad_clip':None,'weight_decay':0.,'selection_metric':'log_loss',**option}
        with pytest.raises(ValueError):m.configure_training(cfg)
    for value in [-1,True,1.5]:
        with pytest.raises(ValueError):m.set_extra_state({'optimizer_steps':value})


def test_registry_alias_and_classification_only_contract():
    assert inspect_method('msattention')['id']=='msattention_repaired'
    for profile in PROFILES:
        spec=require_runnable(profile)
        assert spec['family']=='msattention' and spec['tasks']==['classification']
        assert spec['required_inputs']==['patch_features','single_slide_bag']


def test_repaired_route_refuses_fictitious_stochastic_budget(tmp_path):
    with pytest.raises(ValueError, match='prediction_samples=1'):
        MultiAttentionMIL(data(tmp_path/'data'), 'msattention_repaired', prediction_samples=3)

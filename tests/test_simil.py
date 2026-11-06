"""Full SI-MIL source training, independent derivative oracle, aligned assets and recovery."""
import ast
import contextlib
import copy
import io
import math
import subprocess
import sys
import types

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.io import digest_file, read_json, write_json
from milplus.registry import check_compatibility, require_runnable
from milplus.simil import SIMIL
from milplus.simil_data import prepare
from milplus.simil_demo import create_simil_demo
from milplus.simil_normalization import PathExpertNormalization
from milplus.simil_source import COMMIT, load_model
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def source(path):
    root,sha=verified_source('simil')
    assert sha==COMMIT
    return subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)


def data(path,native=False):
    return Bundle(create_simil_demo(path,dimension=384 if native else 16,descriptors=246 if native else 12))


def exact(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def original_training():
    class CpuTensor(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if ast.unparse(node)=='torch.cuda.FloatTensor':
                return ast.copy_location(ast.parse('torch.FloatTensor',mode='eval').body,node)
            return self.generic_visit(node)
    tree=ast.parse(source('train.py'))
    tree=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and
                         n.name in {'train','dropout_patches'}],type_ignores=[])
    tree=ast.fix_missing_locations(CpuTensor().visit(tree))
    namespace={'torch':torch,'nn':nn,'F':F,'np':np,'Variable':Variable,'sys':sys}
    exec(compile(tree,'original_simil_training.py','exec'),namespace)  # noqa: S102 -- original Git reference
    return namespace['train']


def snapshot(model,opt):
    return {'weights':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),
        'gradients':{n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()},
        'rng':rng_state()}


@pytest.mark.parametrize('profile',['simil_source','simil_full_vjp'])
def test_actual_author_outer_loop_every_update(tmp_path,profile):
    """Actual train.py, native 384/246/K20/four mixers; every weight, gradient, Adam and RNG."""
    bundle=data(tmp_path/'data',native=True);seed_everything(2026)
    model=SIMIL(bundle,profile=profile);model.configure_training({'epochs':39,'max_instances':None,'grad_clip':None})
    model.prepare(bundle,2026,'cpu');opt=model.make_optimizer(.0002,.005)
    reference=copy.deepcopy(model.network);traces=[]
    class TracedAdam(torch.optim.Adam):
        def step(self,closure=None):
            result=super().step(closure);traces.append(snapshot(reference,self));return result
    ref_opt=TracedAdam(reference.parameters(),lr=.0002,betas=(.5,.9),weight_decay=.005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(ref_opt,40,eta_min=.000005)
    aux=[];deep=[];index={};offset=0
    for row in bundle.records('train'):
        x,raw,_=model.assets.load(row);a=model.normalization.transform(raw)
        index[row['slide_id']]=[offset,offset+len(x),row['target']];offset+=len(x)
        aux.append(a);deep.append(x.numpy())
    aux=np.concatenate(aux);deep=np.concatenate(deep);order=list(index)
    checked=0;original=original_training()
    for epoch in range(1,4):
        seed_everything(2026+epoch);initial=rng_state();np.random.shuffle(order);traces.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            original(order,index,index,aux,deep,reference,nn.BCELoss(),ref_opt,246,384,
                     types.SimpleNamespace(dropout_patch=.2),epoch)
        expected=list(traces);restore_rng(initial);step=opt.step
        def checked_step(*args,_step=step,_expected=expected,**kwargs):
            nonlocal checked
            result=_step(*args,**kwargs);exact(snapshot(model.network,opt),_expected.pop(0));checked+=1;return result
        opt.step=checked_step
        losses=model.epoch(bundle,opt);opt.step=step
        assert not expected and len(losses)==len(order) and model.order==order
        model.finish_epoch(bundle,'cpu',{});scheduler.step();exact(model.scheduler.state_dict(),scheduler.state_dict())
    assert checked==54 and model.optimizer_steps==54
    write_json(tmp_path/'simil_author_training.json',{'state':'passed','profile':profile,'source_commit':COMMIT,
        'input_dimension':384,'pathexpert_dimension':246,'top_k':20,'mixer_layers':4,'epochs':3,
        'optimizer_updates':checked,'comparison':'actual original train.py, all weights, gradients, Adam moments, scheduler and RNG after every step',
        'scope':'CPU synthetic; named derivative repair checked separately against a Gaussian closed-form oracle'})


@pytest.mark.parametrize('profile',['source','full_vjp'])
def test_two_choice_gaussian_closed_form(profile):
    """Analytic derivative, independent of both original and repaired backward code."""
    module=load_model(profile);torch.manual_seed(2026)
    x=torch.zeros(1,2,requires_grad=True)
    indicator=module.PerturbedTopK(k=1,num_samples=262144,sigma=.5)(x)
    indicator[0,0,0].backward();gradient=x.grad[0]
    assert abs(float(gradient[0])-1/math.sqrt(math.pi))<.01
    if profile=='full_vjp':
        assert abs(float(gradient[1])+1/math.sqrt(math.pi))<.01
    else:
        assert gradient[1]==0  # Explicit source defect; never silently claim a full derivative.


def test_forward_unchanged_by_full_vjp():
    original=load_model('source');repaired=load_model('full_vjp')
    x=torch.tensor([[.3,-.7,.1,.2]],requires_grad=True)
    torch.manual_seed(77);a=original.PerturbedTopK(k=2,num_samples=100,sigma=.002)(x)
    torch.manual_seed(77);b=repaired.PerturbedTopK(k=2,num_samples=100,sigma=.002)(x)
    exact(a,b)


@pytest.mark.parametrize('seed',[2026,7,42])
def test_actual_author_normalization(seed):
    rng=np.random.default_rng(seed);raw=rng.normal(size=(48,246));raw[rng.random(raw.shape)<.1]=np.nan
    raw[:,0]=np.nan;raw[:,1]=0;raw[:,2]=np.round(raw[:,2]);raw[-1,3:5]=[1e6,-1e6]
    tree=ast.parse(source('data_curation/data_filtering.py'))
    node=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="norm_feat == 'max'")
    namespace={'np':np,'pd':pd,'norm_feat':'bin','bins':10,'tqdm':lambda x:x,
        'features_csv':pd.DataFrame(raw),'train_index_select_list':list(range(28))}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'original_filter.py','exec'),namespace)  # noqa: S102
    namespace['features_array']=namespace['normalized_df'].to_numpy()
    node=next(n for n in ast.walk(ast.parse(source('train.py'))) if isinstance(n,ast.For)
              and ast.unparse(n.iter)=='range(features_array.shape[1])')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original_zscore.py','exec'),namespace)  # noqa: S102
    norm=PathExpertNormalization().fit(raw[:28]);result=norm.transform(raw)
    np.testing.assert_array_equal(result,namespace['features_array'])
    assert (result[:,:2]==-1).all() and (result[-1,3:5]==-1).all()
    exact(result,PathExpertNormalization.restore(norm.state()).transform(raw))


def test_blocked_quantiles_exact_and_heldout_isolation(tmp_path):
    bundle=data(tmp_path/'data');assets=bundle.simil
    expected=PathExpertNormalization().fit(np.concatenate([assets.load(r)[1] for r in bundle.records('train')]))
    n=sum(assets.meta['slides'][r['slide_id']]['instances'] for r in bundle.records('train'))
    actual=assets.fit_training_normalization(n*8*6*2)
    exact(actual.state(),expected.state());assert assets.fit_plan['column_block_size']==2
    for row in bundle.records('val')+bundle.records('test'):
        entry=assets.meta['slides'][row['slide_id']];path=assets.path.parent/entry['path']
        with h5py.File(path,'r+') as h:h['features'][:]=1e6
        entry['sha256']=digest_file(path)
    write_json(assets.path,assets.meta)
    changed=Bundle(bundle.root).simil.fit_training_normalization(n*8*6*2)
    exact(actual.state(),changed.state())
    with pytest.raises(ValueError,match='budget'):assets.fit_training_normalization(1)


@pytest.mark.parametrize('kind',['coords','descriptors','primary','manifest'])
def test_changed_or_permuted_assets_rejected(tmp_path,kind):
    bundle=data(tmp_path/'data');row=bundle.records('train')[0];assets=bundle.simil
    path=assets.path.parent/assets.meta['slides'][row['slide_id']]['path']
    if kind=='manifest':
        metadata=read_json(assets.path);metadata['schema']['provenance']='changed';write_json(assets.path,metadata)
    else:
        if kind=='primary':path=bundle.root/row['path']
        with h5py.File(path,'r+') as h:
            if kind=='coords':h['coords'][:]=h['coords'][:][::-1]
            else:h['features'][0,0]+=1
    with pytest.raises(ValueError,match='changed'):assets.load(row)


@pytest.mark.parametrize('kind',['fov','names','synthetic','weights','coverage','duplicates','coords'])
def test_invalid_preparation_rejected(tmp_path,kind):
    bundle=data(tmp_path/'data');metadata=read_json(bundle.simil.path);schema=metadata['schema']
    index=pd.read_csv(bundle.root/'pathexpert.csv')
    if kind=='fov':schema['field_of_view_um']+=1
    elif kind=='names':schema['feature_names'][1]=schema['feature_names'][0]
    elif kind=='synthetic':schema['synthetic']=False
    elif kind=='weights':schema['segmentation_weights_sha256']='invalid'
    elif kind=='coverage':index=index.iloc[:-1]
    elif kind=='duplicates':index.loc[1,'path']=index.loc[0,'path']
    else:
        with h5py.File(bundle.root/index.iloc[0].path,'r+') as h:h['coords'][:]=h['coords'][:][::-1]
    index.to_csv(bundle.root/'index_bad.csv',index=False)
    with pytest.raises(ValueError):prepare(bundle,bundle.root/'index_bad.csv',schema,bundle.root/'bad.json')


@pytest.mark.parametrize('n',[1,3,20,24])
def test_native_short_bags_and_interpretation(tmp_path,n):
    bundle=data(tmp_path/'data');model=SIMIL(bundle);model.prepare(bundle,2026,'cpu');row=bundle.records('test')[0]
    x,raw,xy=model.assets.load(row);a=model.normalization.transform(raw)
    x=x[:n];a=a[:n];xy=xy[:n];model.eval()
    with torch.no_grad():
        out=model(torch.tensor(a,dtype=torch.float32),x,train=False);model.trace={'normalized':a,'coords':xy}
        record=model.export_instances(out,row,tmp_path/'instances')
    with h5py.File(record['path']) as h:
        logit=h['feature_contributions'][:].sum()+len(h['prediction_indices'])*h['per_instance_bias'][0]
        assert abs(1/(1+np.exp(-logit))-record['prediction'])<1e-5
        assert len(h['gate_input_indices'])==20
        assert len(h['prediction_indices'])==min(len(x),20)
        np.testing.assert_array_equal(h['coords'][:],xy)


def test_distillation_detaches_target_not_selection(tmp_path):
    bundle=data(tmp_path/'data');model=SIMIL(bundle,top_k=2);model.prepare(bundle,2026,'cpu')
    out=model.forward_unit(bundle,[bundle.records('train')[0]],'cpu',{},train=True)
    loss=model.objective(out,torch.tensor([0]))['distillation']
    gradients=torch.autograd.grad(loss,[out['deep'],out['main']],allow_unused=True,retain_graph=True)
    assert gradients[0] is None and gradients[1] is not None
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.network.b_classifier.attention_deep.parameters())
    assert all(p.grad is None for p in model.network.b_classifier.classification_head_deep.parameters())


@pytest.mark.parametrize('profile',['simil_source','simil_full_vjp'])
def test_cli_training_exact_resume_and_predictions(tmp_path,profile):
    bundle=data(tmp_path/'data');cfg={'epochs':3,'threads':1}
    full=train(bundle.root,profile,tmp_path/'full',config=cfg)
    partial=train(bundle.root,profile,tmp_path/'resume',config=cfg,stop_after=1)
    train(bundle.root,profile,partial,config=cfg,resume=True)
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(partial/'last.pt',weights_only=False)
    for key in ['model','optimizers','rng','history','best','bad']:exact(a[key],b[key])
    evaluate(full)
    evaluate(partial)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'),pd.read_csv(partial/'test_predictions.csv'))
    report=read_json(full/'method_protocol.json')
    assert report['optimizer_steps']==54 and report['normalization']['training_instances']>0
    assert read_json(full/'request.json')['synthetic'] is True
    assert read_json(full/'test_instances'/'manifest.json')['records']


def test_alias_and_missing_input_contract(tmp_path):
    assert require_runnable('SI-MIL')['id']=='simil_full_vjp'
    bundle=data(tmp_path/'data');task=yaml.safe_load((bundle.root/'task.yaml').read_text())
    del task['encoders']['toy']['simil'];(bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='PathExpert'):check_compatibility(require_runnable('SI-MIL'),Bundle(bundle.root))


@pytest.mark.parametrize('kwargs',[{'top_k':1},{'top_k':True},{'dropout_patch':.5},
    {'dropout_patch':float('nan')},{'normalization_workspace_bytes':True},{'profile':'other'}])
def test_native_configuration_guards(tmp_path,kwargs):
    bundle=data(tmp_path/'data')
    with pytest.raises(ValueError):SIMIL(bundle,**kwargs)


def test_prepare_command_parser_and_deep_inventory(tmp_path):
    from milplus.cli import parser
    bundle=data(tmp_path/'data');inventory=bundle.check(deep=True)
    assert all('simil' in row for row in inventory['features'])
    args=parser().parse_args(['prepare-simil','--data',str(bundle.root),'--index','index.csv',
                             '--schema','schema.json','--output','manifest.json'])
    assert args.command=='prepare-simil'
    write_json(bundle.root/'schema.json',bundle.simil.meta['schema'])
    result=subprocess.run([sys.executable,'-m','milplus','prepare-simil','--data',str(bundle.root),
        '--index',str(bundle.root/'pathexpert.csv'),'--schema',str(bundle.root/'schema.json'),
        '--output',str(bundle.root/'cli_manifest.json')],text=True,capture_output=True,check=True)
    assert 'cli_manifest.json' in result.stdout
    exact(read_json(bundle.root/'cli_manifest.json'),read_json(bundle.simil.path))

"""Original native 2D model/recipes, independent primary equations and patient-safe recovery."""
import ast
import copy
import itertools
import subprocess
import sys
import types

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.autograd.gradcheck import GradcheckError

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, train, units
from milplus.io import digest_file, read_json, write_json
from milplus.mamba2d import Mamba2DMIL, mamba2d_epoch
from milplus.mamba2d_data import PhysicalGrid
from milplus.mamba2d_demo import create_mamba2d_demo
from milplus.mamba2d_scan import build_paper_scan, build_scan
from milplus.mamba2d_source import VARIANTS, author_model
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import verified_source

NAMES = itertools.count()
PROFILES = list(VARIANTS)
SHAPES = [(1,1,1,1,1),(1,2,2,2,3),(2,3,5,2,3),(2,4,8,2,3),
          (1,8,4,2,3),(1,9,7,1,2),(1,1,7,2,2),(1,7,1,2,2)]


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def source(path):
    root,sha=verified_source('mamba2dmil')
    return subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)


def module(name,text):
    value=types.ModuleType(name);value.__file__=name+'.py';sys.modules[name]=value
    exec(compile(text,value.__file__,'exec'),value.__dict__)  # noqa: S102 -- independent original Git reference, not adapter loader
    return value


def primary_equations(a,x):
    horizontal=[]
    for i in range(x.shape[1]):
        row=[]
        for j in range(x.shape[2]):row.append(x[:,i,j]+(a[:,i,j]*row[j-1] if j else 0))
        horizontal.append(row)
    vertical=[]
    for i in range(x.shape[1]):
        vertical.append([horizontal[i][j]+(a[:,i,j]*vertical[i-1][j] if i else 0) for j in range(x.shape[2])])
    return torch.stack([torch.stack(row,dim=1) for row in vertical],dim=1)


def reference(bundle,profile,hidden=16,state=4,layers=1):
    prefix='_independent_mamba2d_'+str(next(NAMES))
    scan=module(prefix+'_scan',source('2DMambaMIL/models/pscan.py'))
    # For autograd-only reference, tests use height<=2 where the literal CPU vertical downsweep is inactive.
    # Thus independent primary equations describe the same forward, with independently correct derivatives.
    if profile!='mamba2dmil_source':scan.pscan_2d=primary_equations
    model_name=prefix+'_model'
    module(model_name,source('2DMambaMIL/models/mamba_simple.py').replace(
        'from models.pscan import pscan, pscan_2d','from '+scan.__name__+' import pscan, pscan_2d'))
    cls=module(prefix+'_head',source('2DMambaMIL/models/MambaMIL_2D.py').replace(
        'from models.mamba_simple import','from '+model_name+' import')).MambaMIL_2D
    extent=bundle.encoder['mamba2d']['max_coordinates']
    args=types.SimpleNamespace(in_dim=bundle.dim,mambamil_dim=hidden,mambamil_state_dim=state,
        mambamil_layer=layers,drop_out=.25,patch_encoder_batch_size=128,mambamil_inner_layernorms=False,
        pscan=True,cuda_pscan=False,model_type='2DMambaMIL',mamba_2d_max_w=extent[0],
        mamba_2d_max_h=extent[1],mamba_2d_patch_size=bundle.encoder['spatial']['patch_size'],
        mamba_2d_pad_token='trainable',n_classes=4 if bundle.task_type=='survival' else len(bundle.classes),
        survival=bundle.task_type=='survival',pos_emb_type=None,pos_emb_dropout=0.)
    return cls(args),args


def data(path,survival=False,dim=16,short=False):
    root=create_mamba2d_demo(path,dimension=dim,survival=survival)
    if short:
        import yaml
        task=__import__('yaml').safe_load((root/'task.yaml').read_text())
        task['encoders']['toy']['mamba2d']['max_coordinates'][1]=1023
        for row in Bundle(root).rows.to_dict('records'):
            with h5py.File(root/row['path'],'r+') as h:
                good=h['coords'][:,1]<1024;x,xy=h['features'][:][good],h['coords'][:][good]
                del h['features'];del h['coords'];h['features']=x;h['coords']=xy
        (root/'task.yaml').write_text(yaml.safe_dump(task))
    return Bundle(root)


def exact(a,b,tolerant=False):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=3e-4 if tolerant else 0,atol=3e-6 if tolerant else 0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:exact(a[k],b[k],tolerant)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y,tolerant)
    else:assert a==b


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('shape',SHAPES)
def test_real_parallel_scan_against_original_or_primary_equations_and_numeric_gradient(profile,shape):
    text=source('2DMambaMIL/models/pscan.py')
    original=module('_raw_scan_'+str(next(NAMES)),text).pscan_2d
    fn=original if profile=='mamba2dmil_source' else (build_scan(text)[0] if profile=='mamba2dmil_autograd' else build_paper_scan(text)[0])
    torch.manual_seed(21)
    a=(torch.rand(shape,dtype=torch.float64)*.2).requires_grad_();x=torch.randn_like(a).requires_grad_()
    out=fn(a,x)
    expected=primary_equations(a,x) if profile=='mamba2dmil_paper' else original(a,x)
    torch.testing.assert_close(out,expected,rtol=1e-12 if profile=='mamba2dmil_paper' else 0,atol=1e-12 if profile=='mamba2dmil_paper' else 0)
    if profile!='mamba2dmil_source':
        assert torch.autograd.gradcheck(fn,(a,x),fast_mode=True,eps=1e-6,atol=1e-5,rtol=1e-4)


def test_literal_cpu_backward_defect_is_explicit_not_rebranded_as_correct():
    raw=module('_defective_scan',source('2DMambaMIL/models/pscan.py')).pscan_2d
    torch.manual_seed(21)
    a=(torch.rand(1,2,2,1,1,dtype=torch.float64)*.2).requires_grad_();x=torch.randn_like(a).requires_grad_()
    with pytest.raises(GradcheckError):
        torch.autograd.gradcheck(raw,(a,x),eps=1e-6,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('survival',[False,True])
@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('layers',[1,2])
def test_actual_full_native_model_initialization_all_gradients_and_physical_bridge(tmp_path,profile,survival,native,layers):
    b=data(tmp_path/'data',survival,1024 if native else 16,short=profile=='mamba2dmil_autograd')
    hidden,state=(128,16) if native else (16,4)
    before=rng_state();m=Mamba2DMIL(b,profile,hidden=hidden,state_size=state,layers=layers);after=rng_state();restore_rng(before)
    ref,_=reference(b,profile,hidden,state,layers)
    exact(m.network.state_dict(),ref.state_dict());exact(after,rng_state())
    if survival:m.prepare(b,7,'cpu')
    row=b.records('train')[0];x,grid,xy=m.grid.inputs(row)
    m.train();ref.train();before=rng_state();actual=m(x,grid)
    loss=m.objective(actual,row['time'],row['event'])/32 if survival else nn.functional.cross_entropy(actual['logits'],torch.tensor([row['target']]))
    loss.backward();after=rng_state();restore_rng(before)
    result=ref(x,torch.as_tensor(xy))
    if survival:
        nll=reference_definitions('2DMambaMIL/utils/survival_utils.py',{'nll_loss'})['nll_loss']
        expected=nll(result[0],result[1],m.time_label([row['time']]),torch.tensor([1-row['event']]),alpha=0.)/32
    else:expected=nn.functional.cross_entropy(result[0],torch.tensor([row['target']]))
    expected.backward();exact(after,rng_state())
    torch.testing.assert_close(loss,expected,rtol=2e-5,atol=2e-6)
    for (name,p),(other,q) in zip(m.network.named_parameters(),ref.named_parameters()):
        assert name==other
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)
    m.eval();ref.eval()
    with torch.no_grad():
        out=m(x,grid);expected=ref(x,torch.as_tensor(xy))
    torch.testing.assert_close(out['hazards'] if survival else out['logits'],expected[0],rtol=2e-5,atol=2e-6)
    assert m.trace.shape==(m.grid.width*m.grid.height,) and torch.isclose(m.trace.sum(),torch.tensor(1.))
    changed={**row,'target':999,'time':100000,'event':0}
    exact(out,m.forward_unit(b,[changed],'cpu',{}))


def reference_definitions(path,names,extra=None):
    tree=ast.parse(source(path));defs=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in defs}==names
    ns={'torch':torch,'np':np,'optim':torch.optim,**(extra or {})}
    exec(compile(ast.Module(body=defs,type_ignores=[]),'independent/'+path,'exec'),ns)  # noqa: S102 -- actual original loss/optimizer/training definitions
    return ns


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('survival',[False,True])
def test_complete_original_outer_training_updates_optimizer_scheduler_and_rng(tmp_path,profile,survival):
    b=data(tmp_path/'data',survival,short=profile=='mamba2dmil_autograd')
    m=Mamba2DMIL(b,profile,hidden=16,state_size=4)
    cfg={'epochs':4,'max_instances':None,'grad_clip':None,'selection_metric':'c_index' if survival else 'mamba2d_bag_auc'}
    m.configure_training(cfg)
    if survival:m.prepare(b,3,'cpu')
    ref,args=reference(b,profile);ref.load_state_dict(m.network.state_dict())
    if profile!='mamba2dmil_source':
        # Full architecture/gradients are independently checked above against primary equations.
        # For the outer-loop test use the validated repaired parallel kernel in both branches:
        # serial-vs-parallel roundoff in analytically zero softmax biases is amplified by Adam.
        ref=copy.deepcopy(m.network)
    opt=m.make_optimizer(1e-4,1e-5)
    factory=reference_definitions('2DMambaMIL/utils/utils.py',{'get_optim'})['get_optim']
    args.opt='adamw';args.lr=1e-4;args.reg=1e-5;args.device=0;args.n_classes=4 if survival else 3
    refopt=factory(ref,args)
    schedule=None if survival else torch.optim.lr_scheduler.CosineAnnealingLR(refopt,T_max=4,eta_min=0)
    class Progress:
        def update(self):pass
        def close(self):pass
    extra={'tqdm':lambda **kw:Progress(),'wandb':types.SimpleNamespace(log=lambda *a,**kw:None),
        'calculate_error':lambda *a:0.,'calculate_metrics':lambda *a:(0.,0.,0.),
        'concordance_index_censored':lambda *a,**kw:(.5,0,0,0,0)}
    name='train_loop_survival' if survival else 'train_loop'
    code=source('2DMambaMIL/utils/core_utils.py').replace("torch.device(f'cuda:{args.device}')","torch.device('cpu')")
    tree=ast.parse(code);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'torch':torch,'np':np,**extra};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original/train_loop','exec'),ns)  # noqa: S102 -- unmodified numerical original loop with explicit CPU device compatibility
    groups=units(b,'train')[:4] if survival else [next(g for g in units(b,'train') if g[0]['target']==i) for i in range(3)]
    batches=[]
    for group in groups:
        row=group[0];x,grid,xy=m.grid.inputs(row);xy=grid if profile!='mamba2dmil_source' else torch.as_tensor(xy)
        batches.append((x,xy,torch.zeros(1),m.time_label([row['time']]),row['time'],torch.tensor([1-row['event']])) if survival else (x[None],xy[None],torch.tensor([row['target']])))
    criterion=reference_definitions('2DMambaMIL/utils/survival_utils.py',{'nll_loss','NLLSurvLoss'})['NLLSurvLoss'](alpha=0)
    snapshots=[];step=refopt.step
    def capture(*a,**kw):
        gradients=[None if p.grad is None else p.grad.detach().clone() for p in ref.parameters()]
        result=step(*a,**kw)
        snapshots.append((gradients,copy.deepcopy(ref.state_dict()),copy.deepcopy(refopt.state_dict()),rng_state()))
        return result
    refopt.step=capture
    before=rng_state();schedules=[]
    for epoch in range(4):
        if survival:
            from torch.utils.data import DataLoader, RandomSampler
            loader=DataLoader(batches,batch_size=1,sampler=RandomSampler(batches),num_workers=0,collate_fn=lambda values:values[0])
            ns[name](epoch,ref,loader,refopt,loss_fn=criterion,args=args)
        else:ns[name](epoch,ref,batches,refopt,schedule,nn.CrossEntropyLoss(),args)
        if schedule is not None:schedules.append(copy.deepcopy(schedule.state_dict()))
    restore_rng(before);index=0;candidate_step=opt.step
    def compare(*a,**kw):
        nonlocal index
        expected=snapshots[index]
        exact([p.grad for p in m.network.parameters()],expected[0])
        value=candidate_step(*a,**kw)
        exact(m.network.state_dict(),expected[1])
        exact(opt.state_dict(),expected[2]);exact(rng_state(),expected[3])
        index+=1;return value
    opt.step=compare
    for epoch in range(4):
        m.train();mamba2d_epoch(m,b,groups,opt,'cpu',cfg);m.finish_epoch(b,'cpu',cfg)
        if schedule is not None:exact(m.scheduler.state_dict(),schedules[epoch])
    assert index==4*len(groups) and m.optimizer_steps==index and m.completed_epochs==4
    write_json(tmp_path/'mamba2d_author_training.json',{'state':'passed','profile':profile,'task':'survival' if survival else 'classification',
        'optimizer_updates':index,'epochs':4,'scope':'Actual original full outer numerical loop, AdamW/state/cosine/RNG and all gradients/weights; independent primary-equation architecture checks separate; repaired parallel kernels shared only for this strict outer-loop comparison. CPU synthetic real-grid bags. Metrics/logging noops only.'})


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('survival',[False,True])
@pytest.mark.parametrize('boundary',[1,2])
def test_exact_resume_native_selection_repeat_csv_and_real_pad_mass(tmp_path,profile,survival,boundary):
    b=data(tmp_path/'data',survival);cfg={'epochs':3,'threads':1,'model':{'hidden':16,'state_size':4}}
    full=train(b.root,profile,tmp_path/'full',seed=7,config=cfg)
    part=train(b.root,profile,tmp_path/'part',seed=7,config=cfg,stop_after=boundary)
    assert read_json(part/'status.json')['state']=='interrupted'
    train(b.root,profile,part,seed=7,config=cfg,resume=True)
    a,c=[torch.load(path/'last.pt',weights_only=False) for path in [full,part]]
    for key in ['model','optimizers','history','rng']:exact(a[key],c[key])
    assert a['model']['_extra_state']['completed_epochs']==3
    assert a['model']['_extra_state']['optimizer_steps']==3*len(units(b,'train'))
    history=read_json(full/'history.json');metric='c_index' if survival else 'mamba2d_bag_auc'
    values=[h['validation'][metric] for h in history]
    best=max(values);expected=values.index(best) if survival else max(i for i,v in enumerate(values) if v==best)
    assert torch.load(full/'best.pt',weights_only=False)['epoch']==expected
    result=evaluate(full);sha=digest_file(full/'test_predictions.csv')
    exact(result,evaluate(full));assert sha==digest_file(full/'test_predictions.csv')
    exact(result,evaluate(part));assert sha==digest_file(part/'test_predictions.csv')
    export=read_json(full/'test_instances/manifest.json')
    for record in export['records']:
        with h5py.File(record['path']) as f:
            weight=f['whole_grid_attention'][:];mask=f['tissue_mask'][:]
            assert weight.shape==(4,6) and mask.sum()==len(f['coords'])
            np.testing.assert_allclose(weight.sum(),1.,rtol=1e-5)
            assert 0<record['pad_attention_mass']<1
            np.testing.assert_array_equal(f['attention'][:],weight[f['grid_indices'][:,1],f['grid_indices'][:,0]])
            assert weight[mask].sum()<1
    check_compatibility(require_runnable(profile),b)


@pytest.mark.parametrize('change',['missing','negative','fractional_pixels','collision','nan','out_of_extent'])
def test_real_coordinates_required_and_never_replaced_by_a_pseudo_grid(tmp_path,change):
    b=data(tmp_path/'data');row=b.records('train')[0]
    with h5py.File(b.root/row['path'],'r+') as f:
        values=f['coords'][:].astype(float)
        if change=='missing':del f['coords']
        else:
            if change=='negative':values[0,0]=-1
            elif change=='fractional_pixels':values[0,0]+=.5
            elif change=='collision':values[1]=values[0]+1
            elif change=='nan':values[0,0]=float('nan')
            else:values[0,0]=10000
            del f['coords'];f['coords']=values
    with pytest.raises((ValueError,FileNotFoundError),match='2DMamba|coordinate'):
        PhysicalGrid(b).inputs(row)


@pytest.mark.parametrize('change',[{'max_coordinates':None},{'max_coordinates':[True,100]},
    {'max_coordinates':[-1,1]},{'max_coordinates':[float('inf'),1]},{'provenance':''}])
def test_unregistered_or_invalid_extent_fails(tmp_path,change):
    b=data(tmp_path/'data');b.encoder['mamba2d'].update(change)
    with pytest.raises(ValueError,match='2DMamba'):PhysicalGrid(b)


def test_large_raw_coordinates_keep_exact_cell_identity(tmp_path):
    b=data(tmp_path/'data');row=b.records('train')[0]
    with h5py.File(b.root/row['path'],'r+') as f:
        xy=f['coords'][:]+2**26;f['coords'][:]=xy
    b.encoder['mamba2d']['max_coordinates']=[2**26+3071,2**26+2047]
    _,grid,physical=PhysicalGrid(b).inputs(row)
    np.testing.assert_array_equal(physical,xy)
    np.testing.assert_array_equal(grid,xy//512)
    with pytest.raises(ValueError,match='workspace budget'):Mamba2DMIL(b)


@pytest.mark.parametrize('change',[{'hidden':0},{'state_size':False},{'layers':0},{'dropout':True},
    {'dropout':float('nan')},{'pad_token':'fake'},{'inner_layernorms':1},{'workspace_budget_bytes':1},
    {'cuda_binary':'unused.so'},{'backend':'automatic'},{'position_embedding':'linear'}])
def test_invalid_or_silently_substituted_model_options_rejected(tmp_path,change):
    b=data(tmp_path/'data')
    with pytest.raises((ValueError,TypeError)):Mamba2DMIL(b,**change)


def test_native_cuda_never_implicitly_falls_back_without_a_gpu():
    if torch.cuda.is_available():return
    with pytest.raises(RuntimeError,match='no implicit torch substitution'):
        author_model('mamba2dmil_source','cuda')


def test_survival_never_concatenates_unrelated_slide_frames(tmp_path):
    b=data(tmp_path/'data',True)
    b.rows.loc[b.rows.index[1],'global_patient_id']=b.rows.iloc[0].global_patient_id
    with pytest.raises(ValueError,match='one registered WSI per patient'):PhysicalGrid(b)


def test_training_only_time_bins_and_input_hash_binding(tmp_path):
    b=data(tmp_path/'data',True);m=Mamba2DMIL(b,hidden=16,state_size=4);m.prepare(b,3,'cpu')
    edges=m.bin_edges.clone();b.rows.loc[b.rows.subset!='train','time']*=1000
    other=Mamba2DMIL(b,hidden=16,state_size=4);other.prepare(b,3,'cpu');exact(edges,other.bin_edges)
    run=train(b.root,'mamba2dmil_paper',tmp_path/'run',config={'epochs':2,'threads':1,'model':{'hidden':16,'state_size':4}})
    row=b.records('test')[0]
    with h5py.File(b.root/row['path'],'r+') as f:f['coords'][0,0]+=512
    with pytest.raises(ValueError):evaluate(run)


@pytest.mark.parametrize('field',['source_commit','sha256','torch','cuda'])
def test_cuda_binary_build_provenance_checked_before_import(tmp_path,monkeypatch,field):
    from milplus.upstream import verified_source
    _,sha=verified_source('mamba2dmil')
    path=tmp_path/'pscan.so';path.write_bytes(b'invalid binary fixture; must never be imported')
    proof={'source_commit':sha,'sha256':digest_file(path),'torch':str(torch.__version__),'cuda':torch.version.cuda}
    proof[field]='wrong'
    write_json(path.with_suffix('.so.json'),proof)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    with pytest.raises(ValueError,match='build source, binary hash or Torch/CUDA ABI'):
        author_model('mamba2dmil_source','cuda',str(path))


def test_cli_bundle_entry_point_and_alias_runs_the_declared_method(tmp_path):
    import json
    root=create_mamba2d_demo(tmp_path/'data')
    output=tmp_path/'run';cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps({'epochs':2,'threads':1,'model':{'hidden':16,'state_size':4}}))
    result=subprocess.run([sys.executable,'-m','milplus','run','--data',str(root),'--method','2DMambaMIL',
        '--config',str(cfg),'--output',str(output)],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    assert read_json(output/'request.json')['method']=='mamba2dmil_paper'
    result=subprocess.run([sys.executable,'-m','milplus','evaluate','--run',str(output)],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['n']>0

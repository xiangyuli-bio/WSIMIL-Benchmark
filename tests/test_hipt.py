"""Actual pretrained HIPT, independent original training and real hierarchy contracts."""
import argparse
import ast
import contextlib
import copy
import io
import math
import os
import subprocess
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from milplus.data import Bundle
from milplus.engine import evaluate, restore_rng, rng_state, seed_everything, train
from milplus.hipt import HIPTMIL
from milplus.hipt_data import prepare
from milplus.hipt_demo import create_hipt_demo
from milplus.hipt_source import COMMIT, PROFILES, TENSOR_SHA, export_backbone, load_backbone, tensor_map_hash
from milplus.io import digest_file, read_json, write_json
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.fixture
def weights():
    path=Path(os.environ.get('MILPLUS_HIPT_WEIGHTS','/data/milplus_model_weights/hipt_vit4k_xs_teacher_backbone.pt'))
    if not path.is_file():pytest.skip('Prepare the actual HIPT teacher and set MILPLUS_HIPT_WEIGHTS; random weights are not a native substitute')
    return path


def data(path,weights,classes=2,regions=2):
    return Bundle(create_hipt_demo(path,weights,classes=classes,regions=regions))


def source(path):
    root,sha=verified_source('hipt');assert sha==COMMIT
    return subprocess.check_output(['git','show',sha+':'+path],cwd=root,text=True)


def definitions(path,names,scope):
    nodes=[n for n in ast.parse(source(path)).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'independent_original_hipt.py','exec'),scope)  # noqa: S102 -- independent fixed original Git definitions
    return scope


def original():
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'partial':partial,'warnings':warnings}
    definitions('HIPT_4K/vision_transformer4k.py',['_no_grad_trunc_normal_','trunc_normal_','drop_path',
        'DropPath','Mlp','Attention','Block','VisionTransformer4K','vit4k_xs'],scope)
    definitions('2-Weakly-Supervised-Subtyping/models/model_utils.py',['Attn_Net_Gated'],scope)
    definitions('2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py',['HIPT_LGP_FC'],scope)
    scope.update(DataLoader=DataLoader,WeightedRandomSampler=WeightedRandomSampler,device=torch.device('cpu'),
        calculate_error=lambda p,y:1.-p.float().eq(y.float()).float().mean().item())
    definitions('2-Weakly-Supervised-Subtyping/utils/utils.py',
        ['make_weights_for_balanced_classes_split','collate_MIL','get_split_loader'],scope)
    definitions('2-Weakly-Supervised-Subtyping/utils/core_utils.py',['Accuracy_Logger','train_loop'],scope)
    return scope


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


def snapshot(network,optimizer):
    return {'weights':copy.deepcopy(network.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
        'gradients':{n:None if p.grad is None else p.grad.clone() for n,p in network.named_parameters()},'rng':rng_state()}


@pytest.mark.parametrize('profile',list(PROFILES))
def test_complete_original_balanced_training_every_update(tmp_path,weights,profile):
    bundle=data(tmp_path/'data',weights);seed_everything(2026)
    model=HIPTMIL(bundle,profile=profile);optimizer=model.make_optimizer(2e-4,1e-5)
    scope=original()
    with contextlib.redirect_stdout(io.StringIO()):reference=scope['HIPT_LGP_FC'](n_classes=2,freeze_4k=PROFILES[profile])
    reference.load_state_dict(model.network.state_dict());traces=[]
    class TracedAdam(torch.optim.Adam):
        def step(self,closure=None):
            result=super().step(closure);traces.append(snapshot(reference,self));return result
    opt_scope=definitions('2-Weakly-Supervised-Subtyping/utils/utils.py',['get_optim'],
                         {'optim':types.SimpleNamespace(Adam=TracedAdam)})
    ref_opt=opt_scope['get_optim'](reference,types.SimpleNamespace(opt='adam',lr=2e-4,reg=1e-5))
    rows=bundle.records('train')
    class Dataset:
        slide_cls_ids: ClassVar[list]=[np.flatnonzero([r['target']==c for r in rows]) for c in range(2)]
        def __len__(self):return len(rows)
        def __getitem__(self,i):return model.assets.load(rows[i])[0],rows[i]['target']
        def getlabel(self,i):return rows[i]['target']
    loader=scope['get_split_loader'](Dataset(),training=True,weighted=True,mode='pyramid')
    checked=0
    for epoch in range(3):
        seed_everything(2026+epoch);initial=rng_state();traces.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            scope['train_loop'](epoch,reference,loader,ref_opt,2,loss_fn=nn.CrossEntropyLoss())
        expected=list(traces);restore_rng(initial);step=optimizer.step
        def checked_step(*args,_step=step,_expected=expected,**kwargs):
            nonlocal checked
            result=_step(*args,**kwargs);exact(snapshot(model.network,optimizer),_expected.pop(0));checked+=1;return result
        optimizer.step=checked_step;losses=model.epoch(bundle,optimizer);optimizer.step=step
        assert not expected and len(losses)==len(rows)
        model.finish_epoch(bundle,'cpu',{})
    assert checked==54 and model.optimizer_steps==54
    assert (tensor_map_hash(model.network.local_vit.state_dict())==TENSOR_SHA)==PROFILES[profile]
    write_json(tmp_path/'hipt_author_training.json',{'state':'passed','profile':profile,'source_commit':COMMIT,
        'actual_pretrained_teacher':TENSOR_SHA,'epochs':3,'optimizer_updates':checked,'input_shape':[2,256,384],
        'comparison':'actual author get_split_loader, get_optim and train_loop; every weight/gradient/Adam moment/all RNG after every update',
        'scope':'CPU synthetic physical pyramids and actual canonical pretrained teacher; no natural WSI or paper-number claim'})


def test_actual_original_teacher_loader_and_forward(tmp_path,weights):
    state=load_backbone(weights);scope=original()
    with contextlib.redirect_stdout(io.StringIO()):model=scope['vit4k_xs']()
    model.load_state_dict(state,strict=True);model.eval()
    class TorchIO:
        def __getattr__(self,name):return getattr(torch,name)
        def load(self,path,**kwargs):return {'teacher':{'backbone.'+k:v for k,v in state.items()}}
    namespace={'torch':TorchIO(),'os':os,'vits4k':types.SimpleNamespace(vit4k_xs=scope['vit4k_xs'])}
    definitions('HIPT_4K/hipt_model_utils.py',['get_vit4k'],namespace)
    with contextlib.redirect_stdout(io.StringIO()):reference=namespace['get_vit4k'](str(weights))
    exact(model.state_dict(),reference.state_dict())
    x=torch.randn(2,384,16,16)
    with torch.no_grad():exact(model(x),reference(x))
    assert len(state)==78 and len(model.blocks)==6 and model.pos_embed.shape==(1,197,192)


@pytest.mark.parametrize('profile',list(PROFILES))
def test_multiclass_full_model_and_gradients(tmp_path,weights,profile):
    bundle=data(tmp_path/'data',weights,classes=3);model=HIPTMIL(bundle,profile=profile);scope=original()
    with contextlib.redirect_stdout(io.StringIO()):reference=scope['HIPT_LGP_FC'](n_classes=3,freeze_4k=PROFILES[profile])
    reference.load_state_dict(model.network.state_dict())
    x=torch.randn(3,256,384,requires_grad=True);other=x.detach().clone().requires_grad_(True);target=torch.tensor([2])
    torch.manual_seed(77);actual=model(x)
    torch.manual_seed(77);expected=reference(other)
    exact(actual['logits'],expected[0]);exact(actual['probabilities'],expected[1])
    F.cross_entropy(actual['logits'],target).backward();F.cross_entropy(expected[0],target).backward()
    exact(x.grad,other.grad)
    for p,q in zip(model.network.parameters(),reference.parameters()):exact(p.grad,q.grad)


@pytest.mark.parametrize('profile',list(PROFILES))
@pytest.mark.parametrize('early',[False,True])
def test_unified_engine_exact_resume_and_native_selection(tmp_path,weights,profile,early):
    bundle=data(tmp_path/'data',weights)
    cfg={'epochs':3,'threads':1,'model':{'early_stopping':early}}
    full=train(bundle.root,profile,tmp_path/'full',config=cfg)
    partial=train(bundle.root,profile,tmp_path/'partial',config=cfg,stop_after=1)
    train(bundle.root,profile,partial,config=cfg,resume=True)
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(partial/'last.pt',weights_only=False)
    for key in ['model','optimizers','rng','history','best','bad']:exact(a[key],b[key])
    best=torch.load(full/'best.pt',weights_only=False)
    values=[e['validation']['hipt_val_loss'] for e in a['history']]
    expected=max(i for i,v in enumerate(values) if v==min(values)) if early else 2
    assert best['epoch']==expected and a['model']['_extra_state']['optimizer_steps']==54
    assert evaluate(full)==evaluate(partial)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'),pd.read_csv(partial/'test_predictions.csv'))
    manifest=read_json(full/'test_instances/manifest.json')
    for entry in manifest['records']:
        assert entry['regions']==2 and entry['instances']==512
        assert entry['patch_attention_mass']+entry['cls_self_attention_mass']==pytest.approx(1.,abs=1e-6)
        with h5py.File(entry['path']) as h:
            rows=h['canonical_region_rows'][:];g=h['region_attention'][:];local=h['local_cls_attention_all_heads'][:].mean(1)
            np.testing.assert_allclose(h['hierarchical_attention_product'][:][rows],g[:,None]*local[:,1:],rtol=0,atol=0)
    report=read_json(full/'method_protocol.json')
    assert report['completed_epochs']==3 and report['optimizer_steps']==54


@pytest.mark.parametrize('regions',[1,2,3])
def test_physical_row_order_and_native_unfold_equation(tmp_path,weights,regions):
    bundle=data(tmp_path/'data',weights,regions=regions);row=bundle.records('test')[0]
    sequence,coords,origins,order=bundle.hipt.load(row)
    raw,_=bundle.load(row)
    assert sequence.shape==(regions,256,384)
    expected=np.array([[j*512,i*512] for i in range(16) for j in range(16)])
    for r,origin in enumerate(origins):
        np.testing.assert_array_equal(coords[order[r]],expected+origin)
        exact(sequence[r],raw[order[r]])
    marker=torch.arange(regions*256*384).reshape(regions,256,384)
    grid=marker.unfold(1,16,16).transpose(1,2)
    for i in range(16):
        for j in range(16):assert grid[0,17,i,j]==marker[0,i*16+j,17]


@pytest.mark.parametrize('kind',['assignment','duplicate_origin','off_grid','missing_cell','wrong_frame','wrong_embedding','wrong_parent','targets_in_index'])
def test_invalid_hierarchy_cannot_be_prepared(tmp_path,weights,kind):
    bundle=data(tmp_path/'data',weights);schema=copy.deepcopy(bundle.hipt.meta['schema'])
    row=bundle.records('train')[0];path=bundle.root/'pyramids'/f"{row['slide_id']}.h5"
    if kind=='assignment':
        with h5py.File(path,'r+') as h:h['region_ids'][0]=1-h['region_ids'][0]
    elif kind=='duplicate_origin':
        with h5py.File(path,'r+') as h:h['region_origins'][1]=h['region_origins'][0]
    elif kind=='off_grid':
        with h5py.File(bundle.root/row['path'],'r+') as h:h['coords'][0,0]+=1
    elif kind=='missing_cell':
        with h5py.File(path,'r+') as h:
            ids=h['region_ids'][:][:-1];del h['region_ids'];h['region_ids']=ids
    elif kind=='wrong_frame':schema['coordinate_frame']='micrometers'
    elif kind=='wrong_embedding':schema['patch_encoder']='different_384_dimension_encoder'
    elif kind=='wrong_parent':schema['patch_encoder_weights_sha256']='0'*64
    else:
        index=pd.read_csv(bundle.root/'pyramids.csv');index['target']=0;index.to_csv(bundle.root/'pyramids.csv',index=False)
    with pytest.raises(ValueError):prepare(bundle,bundle.root/'pyramids.csv',schema,weights,bundle.root/'bad.json')


def test_even_rehashed_geometry_still_rejected(tmp_path,weights):
    bundle=data(tmp_path/'data',weights);row=bundle.records('train')[0]
    path=bundle.root/'pyramids'/f"{row['slide_id']}.h5"
    with h5py.File(path,'r+') as h:h['region_ids'][0]=1-h['region_ids'][0]
    meta=read_json(bundle.hipt.path);meta['slides'][row['slide_id']]['sha256']=digest_file(path)
    write_json(bundle.hipt.path,meta)
    with pytest.raises(ValueError):Bundle(bundle.root).hipt.load(row)


def test_weight_hash_export_and_no_random_fallback(tmp_path,weights):
    original=load_backbone(weights);portable=export_backbone(weights,tmp_path/'teacher.pt')
    exact(original,load_backbone(portable));assert tensor_map_hash(original)==TENSOR_SHA
    saved=torch.load(portable,weights_only=True);saved['backbone']['cls_token'].zero_()
    torch.save(saved,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='differ'):load_backbone(tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='new'):export_backbone(weights,portable)


def test_original_legacy_checkpoint_is_loaded_safely_when_available():
    path=Path('/data/milplus_model_weights/hipt_vit4k_xs_dino.pth')
    if not path.is_file():pytest.skip('Optional original public full checkpoint unavailable; portable teacher tested separately')
    before=list(torch.serialization.get_safe_globals())
    state=load_backbone(path)
    assert tensor_map_hash(state)==TENSOR_SHA and torch.serialization.get_safe_globals()==before


@pytest.mark.parametrize('concurrent',[False,True])
def test_legacy_conversion_preserves_existing_allowlist_and_order(concurrent):
    path=Path('/data/milplus_model_weights/hipt_vit4k_xs_dino.pth')
    if not path.is_file():pytest.skip('Actual public full HIPT checkpoint required for legacy conversion checks')
    original=list(torch.serialization.get_safe_globals())
    try:
        torch.serialization.add_safe_globals([dict,np.dtype,argparse.Namespace])
        before=list(torch.serialization.get_safe_globals())
        if concurrent:
            with ThreadPoolExecutor(max_workers=2) as executor:states=list(executor.map(load_backbone,[path,path]))
        else:states=[load_backbone(path)]
        assert all(tensor_map_hash(state)==TENSOR_SHA for state in states)
        assert torch.serialization.get_safe_globals()==before
    finally:
        torch.serialization.clear_safe_globals();torch.serialization.add_safe_globals(original)


def test_legacy_conversion_failure_has_no_allowlist_mutation_or_unsafe_fallback(monkeypatch):
    import milplus.hipt_source as module
    path=Path('/data/milplus_model_weights/hipt_vit4k_xs_dino.pth')
    if not path.is_file():pytest.skip('Actual public full HIPT checkpoint required for conversion failure check')
    before=list(torch.serialization.get_safe_globals())
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=1,stdout=b'',stderr=b'child private details'))
    with pytest.raises(ValueError,match='isolated weights-only conversion failed') as error:load_backbone(path)
    assert 'private' not in str(error.value) and torch.serialization.get_safe_globals()==before


def test_original_early_stopper_boundary_and_ties(tmp_path,weights):
    bundle=data(tmp_path/'data',weights);model=HIPTMIL(bundle,early_stopping=True)
    class TorchIO:
        def __getattr__(self,name):return getattr(torch,name)
        def save(self,*args,**kwargs):pass
    scope={'np':types.SimpleNamespace(Inf=float('inf')),'torch':TorchIO()}
    definitions('2-Weakly-Supervised-Subtyping/utils/core_utils.py',['EarlyStopping'],scope)
    reference=scope['EarlyStopping'](patience=20,stop_epoch=50)
    stub=nn.Linear(1,1);best=-float('inf');bad=0;observed=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for epoch in range(60):
            loss=.5 if epoch==0 else .7
            reference(epoch,loss,stub,ckpt_name='unused.pt')
            objective=model.selection_objective({'hipt_val_loss':loss},epoch,loss)
            improved=model.selection_improved(objective,best)
            best,bad=(objective,0) if improved else (best,bad+1)
            stop=bad>=20 and epoch+1>model.stop_after_epoch
            assert stop==reference.early_stop
            if stop:observed.append(epoch);break
    assert observed==[51] and model.selection_improved(-.5,-.5)


def test_alias_and_missing_or_wrong_native_embedding_space(tmp_path,weights):
    assert require_runnable('HIPT')['id']=='hipt_lgp_frozen_source'
    bundle=data(tmp_path/'data',weights);task=yaml.safe_load((bundle.root/'task.yaml').read_text())
    del task['encoders']['toy']['hipt'];task['encoders']['toy']['embedding_space']='another_384_encoder'
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task));other=Bundle(bundle.root)
    with pytest.raises(ValueError,match='HIPT'):check_compatibility(require_runnable('HIPT'),other)
    with pytest.raises(ValueError,match='embedding'):prepare(other,bundle.root/'pyramids.csv',bundle.hipt.meta['schema'],weights,bundle.root/'bad.json')


@pytest.mark.parametrize('changes',[{'profile':'other'},{'early_stopping':1}])
def test_configuration_contract(tmp_path,weights,changes):
    bundle=data(tmp_path/'data',weights)
    with pytest.raises(ValueError):HIPTMIL(bundle,**changes)


def test_cli_preparation_and_actual_native_run(tmp_path,weights):
    bundle=data(tmp_path/'data',weights);write_json(bundle.root/'schema.json',bundle.hipt.meta['schema'])
    def cli(*args):
        return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,check=True)
    cli('prepare-hipt-weights','--checkpoint',weights,'--output',tmp_path/'teacher.pt')
    exact(load_backbone(weights),load_backbone(tmp_path/'teacher.pt'))
    cli('prepare-hipt','--data',bundle.root,'--index',bundle.root/'pyramids.csv',
        '--schema',bundle.root/'schema.json','--weights',weights,'--output',bundle.root/'cli_manifest.json')
    exact(read_json(bundle.root/'cli_manifest.json'),read_json(bundle.hipt.path))
    cli('run','--data',bundle.root,'--method','HIPT','--epochs',1,'--output',tmp_path/'run')
    cli('evaluate','--run',tmp_path/'run')
    assert read_json(tmp_path/'run'/'status.json')['state']=='completed'
    assert (tmp_path/'run'/'test_predictions.csv').is_file()

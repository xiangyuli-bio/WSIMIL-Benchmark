"""PAMoE original trainer equivalence, routing repair and prototype provenance."""
import ast
import copy
import json
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from nystrom_attention import NystromAttention
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from milplus.cli import main
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, load_unit, restore_rng, rng_state, train, units
from milplus.io import digest_file
from milplus.pamoe import PROFILES, PAMoE, pamoe_batches, pamoe_epoch
from milplus.pamoe_data import add_synthetic_prototypes, build_tissue_prototypes
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, survival=True, dimension=16, **options):
    data=add_synthetic_prototypes(create_demo(tmp_path/'data',patients=20,dimension=dimension,survival=survival))
    bundle=Bundle(data)
    model=PAMoE(bundle,**{'hidden':16,'batch_size':4,**options})
    model.prepare(bundle,2026,'cpu')
    return model,bundle


def reference_scope(dim=16,hidden=16,repaired=False):
    root,sha=verified_source('pamoe')
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'copy':copy,'random':random,'Tensor':Tensor,'Sampler':Sampler,
           'LayerNorm':nn.LayerNorm,'pad_sequence':pad_sequence,'NystromAttention':NystromAttention,
           'device':torch.device('cpu'),'optim':torch.optim,
           'cindex_metric_train':lambda *a,**k:(.5,.5),'acc_metric_train':lambda *a,**k:{}}
    for path in ['models/pamoe_layers/pamoe_utils.py','models/pamoe_layers/pamoe.py',
                 'models/models_demo/model_transmil_pamoe.py','models/models_demo/model_transformer_pamoe.py',
                 'utils/utils_loss.py','utils/utils.py','utils/core_funcs.py']:
        code=pinned_text(root,sha,path).replace('.cuda()', '.to(device)')
        if path.endswith('pamoe.py') and '/pamoe_layers/' in path and repaired:
            code=code.replace('expert_input = x_in_reshaped[i * batch_size: (i + 1) * batch_size]',
                              'expert_input = x_in_reshaped.reshape(batch_size, self.n_experts, top_k, embed_dim)[:,i]')
        if path.endswith('model_transmil_pamoe.py') and (dim!=1024 or hidden!=512):
            code=code.replace('dim=512',f'dim={hidden}').replace('nn.Linear(1024, 512)',f'nn.Linear({dim}, {hidden})')
            code=code.replace('torch.randn(1, 1, 512)',f'torch.randn(1, 1, {hidden})').replace('nn.LayerNorm(512)',f'nn.LayerNorm({hidden})').replace('nn.Linear(512, self.n_classes)',f'nn.Linear({hidden}, self.n_classes)')
        names=None if path not in {'utils/utils.py','utils/core_funcs.py'} else (
            {'get_optim','Sampler_custom'} if path=='utils/utils.py' else {'forward_model','train_loop_survival'})
        definitions=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),path,'exec'),scope)  # noqa: S102 -- independent locked source
    return scope


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('profile',sorted(PROFILES))
@pytest.mark.parametrize('survival',[True,False])
def test_independent_full_trainer_six_epochs(tmp_path,profile,survival):
    model,bundle=fixture(tmp_path,profile=profile,survival=survival)
    ref=reference_scope(model.dim,model.hidden,model.repaired)
    tensor_file=tmp_path/'prototypes.pt';torch.save(bundle.pamoe.tensor,tensor_file)
    cls=ref['transmil_pmoe'] if 'transmil' in profile else ref['TransformerEncoder']
    network=cls(**model.network_args,prototype_pth=tensor_file)
    state=model.network.state_dict();state.pop('proto_types');network.load_state_dict(state)
    optim_ref=ref['get_optim'](network,SimpleNamespace(opt='adamw',lr=2e-4,reg=1e-2))
    sched_ref=torch.optim.lr_scheduler.CosineAnnealingLR(optim_ref,T_max=5,eta_min=1e-5)
    optimizer=model.make_optimizer(2e-4,1e-2)
    groups=units(bundle,'train')[:8]
    criterion=ref['CoxPHLoss'](0.) if survival else nn.CrossEntropyLoss()
    args=SimpleNamespace(mode='patch',task='survival' if survival else 'classification',use_fp16=False,alpha_pamoe=.01,num_classes=2)
    records=[[],[]]
    def record(net,target):
        target.append({n:(p.detach().clone(),None if p.grad is None else p.grad.detach().clone()) for n,p in net.named_parameters()})
    optim_ref.register_step_post_hook(lambda o,a,k:record(network,records[0]))
    optimizer.register_step_post_hook(lambda o,a,k:record(model.network,records[1]))
    def scaler(loss,opt,**kw):loss.backward();opt.step()
    for epoch in range(6):
        initial=rng_state()
        indices=list(ref['Sampler_custom']([1-r[0]['event'] for r in groups],4)) if survival else list(range(len(groups)))
        loader=[]
        for start in range(0,len(indices)//4*4,4):
            batch=[groups[i] for i in indices[start:start+4]]
            bags=[load_unit(bundle,rows,'cpu')[0] for rows in batch]
            label=[rows[0]['time'] if survival else rows[0]['target'] for rows in batch]
            loader.append((bags,[torch.zeros(len(x),2) for x in bags],label,
                           [1-int(rows[0]['event']) for rows in batch],[rows[0]['global_patient_id'] for rows in batch]))
        ref['train_loop_survival'](epoch,network,loader,optim_ref,args,loss_fn=criterion,gc=2,loss_scaler=scaler,bar=False)
        sched_ref.step()
        restore_rng(initial);model.train()
        pamoe_epoch(model,bundle,groups,optimizer,'cpu',{})
        model.finish_epoch(bundle,'cpu',{})
        equal(records[0],records[1])
        equal(optim_ref.state_dict(),optimizer.state_dict())
        assert optimizer.param_groups[0]['lr']==optim_ref.param_groups[0]['lr']
    assert len(records[0])>=6
    assert optimizer.param_groups[0]['lr']>1e-5  # Source five-epoch cosine turns upwards again.


@pytest.mark.parametrize('profile',sorted(PROFILES))
@pytest.mark.parametrize('survival',[True,False])
def test_exact_resume_and_predict(tmp_path,profile,survival):
    data=add_synthetic_prototypes(create_demo(tmp_path/'data',patients=20,survival=survival))
    cfg={'epochs':3,'threads':1,'model':{'hidden':16,'batch_size':4}}
    train(data,profile,tmp_path/'full',config=cfg)
    train(data,profile,tmp_path/'resume',config=cfg,stop_after=1)
    train(data,profile,tmp_path/'resume',config=cfg,resume=True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/last.pt',weights_only=False)
    for key in ['model','optimizers','history','best']:equal(a[key],b[key])
    evaluate(tmp_path/'full');evaluate(tmp_path/'resume')
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'full/test_predictions.csv'),pd.read_csv(tmp_path/'resume/test_predictions.csv'))


def manual_experts(m,x):
    b,t,_d=x.shape
    logits=m.router(x);prob=logits.softmax(-1)
    k=max(1,int(m.capacity_factor*t/m.n_experts))
    out=torch.zeros_like(x)
    for batch in range(b):
        for expert in range(m.n_experts):
            weight,idx=prob[batch,:,expert].topk(k)
            selected=x[batch,idx]
            value=m.experts[expert](selected)*weight[:,None]
            out[batch].index_add_(0,idx,value)
    return out


@pytest.mark.parametrize('batch',[1,3])
def test_expert_repair_matches_manual_patient_expert_indices(tmp_path,batch):
    model,_bundle=fixture(tmp_path,profile='pamoe_transmil')
    fixed=model.network.layer1.ffn
    old=reference_scope()['PAMoE'](dim=16,num_expert_proto=4,num_expert_extra=2,capacity_factor=2.)
    old.load_state_dict(fixed.state_dict())
    fixed.eval();old.eval()
    x=torch.randn(batch,11,16,requires_grad=True)
    result=fixed(x)[0];manual=manual_experts(fixed,x)
    torch.testing.assert_close(result,manual,rtol=1e-5,atol=1e-6)
    ga=torch.autograd.grad(result.square().sum(),x,retain_graph=True)[0]
    gb=torch.autograd.grad(manual.square().sum(),x)[0]
    torch.testing.assert_close(ga,gb,rtol=2e-5,atol=3e-6)
    original=old(x)[0]
    if batch==1:torch.testing.assert_close(original,result,rtol=0,atol=0)
    else:assert not torch.allclose(original,result)


def test_prior_is_batch_token_distribution_and_drives_router(tmp_path):
    model,bundle=fixture(tmp_path)
    model.eval()
    bags=[torch.randn(10,16),torch.randn(10,16)]
    result=model.forward_batch(bags)
    gradient=torch.autograd.grad(result['prior_loss'],model.network.layer1.ffn.router.weight)[0]
    assert gradient.abs().sum()>0
    assert not model.network.proto_types.requires_grad
    ref=reference_scope()
    x=torch.stack(bags)
    targets=ref['get_x_cos_similarity'](x,4,bundle.pamoe.tensor)
    for target in targets:torch.testing.assert_close(target.sum(),torch.tensor(1.))
    logits=torch.randn(2,10,6,requires_grad=True)
    expected=sum(-(t*logits[:,:,i].flatten().log_softmax(0)).sum() for i,t in enumerate(targets))/4
    torch.testing.assert_close(ref['cal_prior_loss_ce'](targets,logits,4),expected)


def test_prototypes_are_training_only_means(tmp_path):
    _,bundle=fixture(tmp_path)
    features=[]
    for row in bundle.records('train'):features.append(bundle.load(row)[0])
    for k in range(4):
        expected=torch.cat([x[torch.arange(len(x))%4==k] for x in features]).double().mean(0).float()
        equal(expected,bundle.pamoe.tensor[k])
    assert set(bundle.pamoe.metadata['fit_patient_ids'])=={r['global_patient_id'] for r in bundle.records('train')}
    path=bundle.root/'synthetic_tissues/index.csv'
    table=pd.read_csv(path);table.loc[0,'coords_sha256']='0'*64;table.to_csv(path,index=False)
    with pytest.raises(ValueError,match='coordinate order'):
        build_tissue_prototypes(bundle,path,tmp_path/'new_asset',['a','b','c','d'],'test')


@pytest.mark.parametrize('damage',['fm','hash','leak','names','zero','external_overlap'])
def test_prototype_contract_rejects_invalid_assets(tmp_path,damage):
    _,bundle=fixture(tmp_path)
    manifest=bundle.root/bundle.encoder['pamoe']['manifest'];m=json.loads(manifest.read_text())
    if damage=='fm':m['encoder_sha256']='wrong'
    if damage=='hash':m['sha256']='wrong'
    if damage=='leak':m['fit_patient_ids'].append(bundle.records('test')[0]['global_patient_id'])
    if damage=='names':m['names']=['x']*4
    if damage=='external_overlap':m['fit_scope']='external_disjoint'
    if damage=='zero':
        p=manifest.parent/'prototypes.npy';np.save(p,np.zeros((4,16),dtype=np.float32));m['sha256']=digest_file(p)
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):Bundle(bundle.root)


def test_changed_asset_blocks_resume_and_evaluation(tmp_path):
    data=add_synthetic_prototypes(create_demo(tmp_path/'data',patients=20,survival=True))
    cfg={'epochs':2,'threads':1,'model':{'hidden':16,'batch_size':4}}
    train(data,'pamoe',tmp_path/'run',config=cfg,stop_after=1)
    train(data,'pamoe',tmp_path/'complete',config=cfg)
    bundle=Bundle(data);manifest=bundle.root/bundle.encoder['pamoe']['manifest'];m=json.loads(manifest.read_text())
    m['provenance']='changed';manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Resume refused'):train(data,'pamoe',tmp_path/'run',config=cfg,resume=True)
    with pytest.raises(ValueError,match='inputs differ'):evaluate(tmp_path/'complete')


@pytest.mark.parametrize('profile',sorted(PROFILES))
def test_native_dimensions_author_forward_and_gradients(tmp_path,profile):
    model,bundle=fixture(tmp_path,dimension=1024,hidden=512,profile=profile)
    ref=reference_scope(dim=1024,hidden=512,repaired=model.repaired)
    path=tmp_path/'prototypes.pt';torch.save(bundle.pamoe.tensor,path)
    cls=ref['transmil_pmoe'] if 'transmil' in profile else ref['TransformerEncoder']
    network=cls(**model.network_args,prototype_pth=path)
    state=model.network.state_dict();state.pop('proto_types');network.load_state_dict(state)
    x=torch.randn(2,9,1024,requires_grad=True)
    model.eval();network.eval()
    actual=model.forward_batch([x[0],x[1]])
    _,prediction,prior=network(x)
    equal(actual['logits'],prediction);equal(actual['prior_loss'],prior)
    (actual['logits'].square().sum()+.01*actual['prior_loss']).backward()
    dx=x.grad.clone();x.grad=None
    (prediction.square().sum()+.01*prior).backward()
    equal(dx,x.grad)
    for (a,p),(b,q) in zip(model.network.named_parameters(),network.named_parameters()):
        assert a==b
        equal(p.grad,q.grad)


def test_training_feature_change_requires_recomputed_prototypes(tmp_path):
    _,bundle=fixture(tmp_path)
    bundle.check(deep=True)
    path=bundle.root/bundle.records('train')[0]['path']
    with h5py.File(path,'r+') as f:
        f['features'][0]=f['features'][0]+1
    with pytest.raises(ValueError,match='training features changed'):Bundle(bundle.root).check(deep=True)


def test_prepare_tissues_cli_matches_training_means(tmp_path):
    _,bundle=fixture(tmp_path)
    destination=tmp_path/'cli_asset'
    main(['prepare-tissues','--data',str(bundle.root),'--assignments',str(bundle.root/'synthetic_tissues/index.csv'),
          '--output',str(destination),'--names','a','b','c','d','--provenance','synthetic CLI validation'])
    equal(torch.from_numpy(np.load(destination/'prototypes.npy')),bundle.pamoe.tensor)
    m=json.loads((destination/'manifest.json').read_text())
    assert m['synthetic'] and m['fit_scope']=='training_patients'


def test_source_cox_tie_order_and_nonfinite_rejection(tmp_path):
    model,_=fixture(tmp_path)
    score=torch.tensor([[.1],[.8]],requires_grad=True)
    censor=torch.tensor([0.,0.]);times=torch.tensor([3.,3.])
    forward=model.criterion(S=score,c=censor,event_time=times)
    reverse=model.criterion(S=score.flip(0),c=censor,event_time=times)
    assert not torch.allclose(forward,reverse)  # Released cumulative-set approximation, not Breslow ties.
    expected=-sum(score.flatten()-(score.flatten().sub(.8).exp().cumsum(0)+1e-7).log()-.8)/2
    torch.testing.assert_close(forward,expected)
    assert not torch.isfinite(model.criterion(S=score,c=torch.ones(2),event_time=times))
    with pytest.raises(ValueError,match='finite nonempty'):model.forward_batch([])


@pytest.mark.parametrize('options',[{'hidden':15},{'batch_size':0},{'capacity':100},{'prior_weight':0},{'extra_experts':-1}])
def test_invalid_recipe_rejected(tmp_path,options):
    with pytest.raises(ValueError):fixture(tmp_path,**options)


def test_sampler_event_requirement_tail_and_rng(tmp_path):
    model,bundle=fixture(tmp_path)
    groups=units(bundle,'train')
    before=rng_state();a=pamoe_batches(model,groups);restore_rng(before)
    ref=reference_scope()['Sampler_custom']([1-r[0]['event'] for r in groups],4)
    indices=list(ref)
    assert [[r[0]['global_patient_id'] for r in b] for b in a]==[[groups[i][0]['global_patient_id'] for i in indices[j:j+4]] for j in range(0,len(indices),4)]
    assert all(len(b)==4 and any(r[0]['event'] for r in b) for b in a)
    assert len({r[0]['global_patient_id'] for b in a for r in b})==4*len(a)
    for rows in groups:rows[0]['event']=0
    with pytest.raises(ValueError,match='no complete'):pamoe_batches(model,groups)


def test_train_truncation_and_eval_full_bag(tmp_path):
    model,_=fixture(tmp_path,max_train_patches=10)
    model.eval();bags=[torch.randn(14,16)]
    truncated=model.forward_batch(bags,training=True)
    expected=model.forward_batch([bags[0][:10]],training=False)
    equal(truncated['logits'],expected['logits'])
    assert not torch.allclose(model.forward_batch(bags)['logits'],expected['logits'])

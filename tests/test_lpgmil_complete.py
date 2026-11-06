"""Native dimensions and independent source equations, not clinical validation."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.engine import seed_everything
from milplus.lpgmil import LPGMIL
from milplus.lpgmil_data import prepare
from milplus.lpgmil_demo import create_demo
from milplus.lpgmil_source import native_mamba, native_network, source_inner_reference
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, inspect_method


@pytest.fixture
def bundle(tmp_path):
    return Bundle(create_demo(tmp_path/'data',patches=17))


def config():return {'epochs':2,'max_instances':None,'grad_clip':None}


def model(bundle):
    x=LPGMIL(bundle,'lpgmil_tls1024_reference');x.configure_training(config());return x


def exact(a,b):
    if torch.is_tensor(a):assert torch.is_tensor(b)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b)and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert type(a)==type(b)and a==b


def test_native_dimensions_and_all_core_losses(bundle):
    torch.set_num_threads(2);m=model(bundle)
    assert len(m.native.mamba_layers)==2 and len(m.native.token_attentions)==6
    assert m.native.prototypes.shape==(1,6,1024)
    assert m.native.projection.fc.weight.shape==(512,1024)
    for layer in m.native.mamba_layers:
        s=layer[1];assert(s.d_model,s.d_state,s.d_conv,s.expand,s.d_inner)==(512,16,4,2,1024)
        assert not s.dt_proj.bias.detach().any()  # literal author initializer overwrites original special bias
    x,_=bundle.load(bundle.records('train')[0]);out=m(x)
    loss,parts=m.objective(out,torch.tensor([0]));loss.backward()
    assert set(parts)=={'bag_ce','six_branch_ce','all_parameter_l1'}
    assert all(v.grad is not None for v in m.native.parameters())
    for name in ['prototypes','projection.fc.weight','mamba_layers.0.1.A_log','mamba_layers.1.1.in_proj.weight',
                 'token_attentions.0.q_proj.weight','token_classifiers.5.fc.weight','slide_classifier.fc.weight']:
        assert dict(m.native.named_parameters())[name].grad.abs().sum()>0


def test_literal_unrepaired_source_width_failure_is_not_hidden(bundle):
    cls,_,_=native_network(repaired=False)
    source=cls(prototypes=bundle.lpgmil.prototypes.clone())
    with pytest.raises(RuntimeError,match='mat1 and mat2 shapes'):
        source(torch.randn(17,1024))


def test_original_mamba111_reference_equations_and_gradients():
    torch.set_num_threads(2);seed_everything(107)
    cls,_=native_mamba();native=cls(512,d_state=16,d_conv=4,expand=2)
    x=torch.randn(1,17,512,requires_grad=True)
    xz=native.in_proj.weight@x.reshape(17,512).t()
    xz=xz.reshape(2048,1,17).permute(1,0,2)
    ref=source_inner_reference()
    expected=ref(xz,native.conv1d.weight,native.conv1d.bias,native.x_proj.weight,native.dt_proj.weight,
        native.out_proj.weight,native.out_proj.bias,-native.A_log.float().exp(),None,None,native.D.float(),
        delta_bias=native.dt_proj.bias.float(),delta_softplus=True)
    actual=native(x)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    params=[x]+list(native.parameters())
    a=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),params)
    for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=2e-5,atol=2e-6)


def test_literal_repaired_source_full_loss_gradient_and_Adam_state(bundle):
    seed_everything(109);m=model(bundle);source=copy.deepcopy(m.native)
    optimizer=m.make_optimizer(1e-4,1e-5)
    original=torch.optim.Adam(source.parameters(),lr=1e-4,weight_decay=1e-5)
    row=bundle.records('train')[0];x,_=bundle.load(row);target=torch.tensor([row['target']])
    rng=torch.get_rng_state();out=m(x);loss,parts=m.objective(out,target);loss.backward()
    grads={n:p.grad.clone()for n,p in m.native.named_parameters()}
    optimizer.step();optimizer.zero_grad()
    torch.set_rng_state(rng)
    logits,_,_,_,sub_preds,_=source(x)
    # Actual source loop repaired only for its six-return versus seven-unpack,
    # and single-bag dimension in the six-branch cross entropy.
    expected=F.cross_entropy(logits,target)+F.cross_entropy(sub_preds[:,0,:],target.repeat_interleave(6))
    expected+=m.l1(source)*1e-5
    expected.backward()
    assert torch.equal(loss.detach(),expected.detach())
    for name,p in source.named_parameters():assert torch.equal(p.grad,grads[name]),name
    original.step();original.zero_grad()
    exact(m.native.state_dict(),source.state_dict());exact(optimizer.state_dict(),original.state_dict())


def test_complete_epoch_resume_keeps_sampling_masking_Adam_and_prototypes(bundle):
    seed_everything(117);a=model(bundle);oa=a.make_optimizer(1e-4,1e-5)
    a.start_epoch(0);a.epoch(bundle,oa,torch.device('cpu'))
    checkpoint={'model':copy.deepcopy(a.state_dict()),'optimizer':copy.deepcopy(oa.state_dict()),'rng':torch.get_rng_state()}
    a.start_epoch(1);a.epoch(bundle,oa,torch.device('cpu'));rng=torch.get_rng_state()
    b=model(bundle);b.load_state_dict(checkpoint['model']);ob=b.make_optimizer(1e-4,1e-5)
    ob.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['rng'])
    b.start_epoch(1);b.epoch(bundle,ob,torch.device('cpu'))
    exact(a.state_dict(),b.state_dict());exact(oa.state_dict(),ob.state_dict());assert torch.equal(rng,torch.get_rng_state())
    assert a.updates==8 and a.native.prototypes.grad is None


def test_full1025_patch_top1000_attention_and_frozen_selector(bundle):
    seed_everything(113);m=model(bundle);m.eval();before=m.native.prototype_selector.clone()
    x=torch.randn(1025,1024);out=m(x)
    assert out['attention'].shape==(8,6,1000)
    assert out['token_logits'].shape==(6,2)
    out['logits'].sum().backward()
    assert torch.equal(before,m.native.prototype_selector)and m.native.prototype_selector.grad is None
    assert m.native.prototypes.grad is not None and m.native.prototypes.grad.abs().sum()>0
    assert m.native.topk_selector.q_proj.weight.grad is None  # source hard selection has no task gradient


def test_training_only_candidates_and_exact_centroid_provenance(bundle):
    m=bundle.lpgmil.manifest
    assert len(m['candidates'])==20 and len(m['train_patients'])==4
    assert all(s.startswith('synthetic_train_')for s,_,_ in m['candidates'])
    assignment=np.load(bundle.lpgmil.root/m['assignment'],allow_pickle=False)
    rows={r['slide_id']:r for r in bundle.records('train')}
    values=np.stack([bundle.load(rows[s])[0][i].numpy()for s,_,i in m['candidates']])
    centers=np.stack([values[assignment==i].mean(0)for i in range(6)]).astype(np.float32)
    assert np.array_equal(centers,bundle.lpgmil.prototypes.numpy())


@pytest.mark.parametrize('case',['val_slide','duplicate','wrong_patch','negative','exceeds_all','not_integer'])
def test_reject_invalid_or_leaking_nucleus_counts(bundle,tmp_path,case):
    table=pd.read_csv(bundle.root/'train_counts.csv')
    if case=='val_slide':table.loc[0,'slide_id']=bundle.records('val')[0]['slide_id']
    elif case=='duplicate':table=pd.concat([table,table.iloc[:1]],ignore_index=True)
    elif case=='wrong_patch':table.loc[0,'patch_id']='different_actual_patch'
    elif case=='negative':table.loc[0,'n_lymphocytes']=-1
    elif case=='exceeds_all':table.loc[0,'n_lymphocytes']=999
    elif case=='not_integer':
        table['n_lymphocytes']=table.n_lymphocytes.astype(float)
        table.loc[0,'n_lymphocytes']=1.5
    path=tmp_path/'counts.csv';table.to_csv(path,index=False)
    with pytest.raises(ValueError):prepare(bundle,path,bundle.root/'detector.json',tmp_path/'bad_assets',.75,10)
    assert not(tmp_path/'bad_assets').exists()


def test_changed_actual_feature_or_prototype_bytes_rejected(bundle):
    path=bundle.root/bundle.records('train')[0]['path']
    with h5py.File(path,'r+')as f:f['features'][0,0]+=1
    with pytest.raises(ValueError,match='feature bytes'):bundle.lpgmil.verify()
    np.save(bundle.lpgmil.root/'prototypes.npy',np.zeros((6,1024),dtype=np.float32))
    with pytest.raises(ValueError,match='asset changed'):Bundle(bundle.root)


def test_reject_synthetic_detectors_on_clinical_data(bundle):
    task=yaml.safe_load((bundle.root/'task.yaml').read_text());task['synthetic']=False
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='Synthetic LPGMIL'):Bundle(bundle.root)


def test_fm_and_class_port_keeps_complete_native_architecture(tmp_path):
    data=Bundle(create_demo(tmp_path/'fm_data',17,768,3))
    m=LPGMIL(data,'lpgmil_fm_width_reference');m.configure_training(config())
    assert m.native.projection.fc.weight.shape==(512,768)
    assert len(m.native.mamba_layers)==2 and m.native.mamba_layers[0][1].d_model==512
    x,_=data.load(data.records('train')[0]);result=m(x)
    assert result['logits'].shape==(1,3)and result['token_logits'].shape==(6,3)
    with pytest.raises(ValueError,match='1024D'):LPGMIL(data,'lpgmil_tls1024_reference')


def test_profiles_no_core_truncation_or_regularizer_removal(bundle):
    spec=inspect_method('LPGMIL');assert spec['id']=='lpgmil_tls1024_reference'
    check_compatibility(spec,bundle);m=build_bundle_model(spec['id'],bundle)
    assert spec['native_recipe']['epochs']==50 and spec['native_recipe']['hidden']==512
    assert m.selection_objective({},49,0)==49.
    with pytest.raises(ValueError,match='every input patch'):m.configure_training({**config(),'max_instances':100})
    with pytest.raises(ValueError,match='no gradient clipping'):m.configure_training({**config(),'grad_clip':1.})
    with pytest.raises(ValueError,match='all-parameter'):LPGMIL(bundle,spec['id'],l1_coefficient=0.)
    with pytest.raises(ValueError,match='explicitly'):LPGMIL(bundle,spec['id'],scan_backend='auto')


def test_public_prepare_and_check_entrypoints(bundle,tmp_path):
    output=tmp_path/'prepared_public'
    command=[sys.executable,'-s','-m','milplus','prepare-lpgmil','--data',str(bundle.root),
        '--counts',str(bundle.root/'train_counts.csv'),'--detector-provenance',str(bundle.root/'detector.json'),
        '--output',str(output),'--min-fraction','.75','--min-nuclei','10']
    process=subprocess.run(command,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert Path(json.loads(process.stdout)['manifest'])==output/'manifest.json'
    check=subprocess.run([sys.executable,'-s','-m','milplus','check','--data',str(bundle.root),
        '--method','LPGMIL','--deep'],capture_output=True,text=True)
    assert check.returncode==0,check.stderr

"""Full original native2048/512/5expert/10task M4, not a reduced stand-in."""
import copy
import gc
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from milplus.data import Bundle
from milplus.engine import restore_rng, rng_state, seed_everything
from milplus.m4 import M4MIL, author_training
from milplus.m4_demo import create_m4_demo
from milplus.m4_source import COMMIT, author_classes, build_original
from milplus.upstream import load_definitions, verified_source


def config(epochs=2):
    return {'epochs':epochs,'lr':.0001,'weight_decay':.00004,'patience':epochs,
            'threads':1,'selection_metric':'m4_val_mean_auc','max_instances':None,'grad_clip':None}


def assert_equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    else:assert a==b


def test_original_full_source_revision_and_native_core_257patch_gradients():
    torch.set_num_threads(1);seed_everything(42)
    root,revision=verified_source('m4');assert revision==COMMIT
    module=author_classes();assert all(hasattr(module,key) for key in ['M4','MP_AMIL','Aggregator','ConvPosEnc'])
    network=build_original();assert len(network.experts)==5 and len(network.towers)==10
    assert network.gate1_fc1[0].weight.shape==(512,2048)
    assert all(gate[0].weight.shape==(5,128) for gate in network.gate1_fc2)
    assert all(expert.fc[0].weight.shape==(512,2048) and expert.hierarchical.seg==4 for expert in network.experts)
    features=torch.randn(257,2048);outputs=network(features)
    assert len(outputs)==10 and all(p.shape==(1,) and 0<p.item()<1 for p in outputs)
    target=torch.arange(10).remainder(2).float()
    loss=torch.stack([torch.nn.functional.binary_cross_entropy(p,target[i:i+1]) for i,p in enumerate(outputs)]).sum()
    loss.backward()
    for expert in network.experts:
        assert expert.classifier.weight.grad is None and expert.classifier.bias.grad is None
        for layer in [expert.fc[0],expert.cpe.proj,expert.hierarchical.agg1.conv1,
                      expert.hierarchical.agg2.pointwise_conv,expert.hierarchical.agg3.conv1,
                      expert.attention_net.attention_c]:
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0
    assert all(tower.fc2.weight.grad is not None for tower in network.towers)
    assert all(gate[0].weight.grad is not None for gate in network.gate1_fc2)
    # Sum-of-task BCE is part of the actual source method, not its mean.
    assert loss.item()==pytest.approx(sum(torch.nn.functional.binary_cross_entropy(p,target[i:i+1]).item() for i,p in enumerate(outputs)))
    del network,outputs,loss;gc.collect()


def test_actual_original_twoepoch_training_loader_cosine_val_and_rng_match(tmp_path):
    torch.set_num_threads(1);seed_everything(42)
    bundle=Bundle(create_m4_demo(tmp_path/'data'))
    actual=M4MIL(bundle);actual.configure_training(config());opt=actual.make_optimizer(.0001,.00004)
    after_init=rng_state();initial=copy.deepcopy(actual.network.state_dict())
    losses=[];validations=[]
    for epoch in range(2):
        actual.start_epoch(epoch);losses.append(actual.epoch(bundle,opt,'cpu')[0]['loss'])
        actual.finish_epoch(bundle,'cpu',config());validations.append(actual.predict_subset(bundle,'val','cpu',config()))
        actual.validate_optimizer_state()
    expected_rng=rng_state()
    # Original unmodified Dataset/collate and training/evaluate bodies, with the
    # same full source network, complete saved tensor inputs and actual DataLoader.
    source=load_definitions('m4','Dataset.py',['Datasets','collate_MIL'],{'torch':torch,'Dataset':Dataset})
    reference=build_original();reference.load_state_dict(initial)
    reference_opt=torch.optim.Adam(reference.parameters(),lr=.0001,weight_decay=.00004)
    scheduler=torch.optim.lr_scheduler.LambdaLR(reference_opt,lr_lambda=lambda i:((1+math.cos(i*math.pi/2))/2)*.9+.1)
    inputs={}
    for subset in ['train','val']:
        paths=[];targets=[]
        for row in bundle.records(subset):
            path=tmp_path/(row['slide_id']+'.pt');torch.save(bundle.load(row)[0],path)
            paths.append(str(path));targets.append(np.asarray(row['targets']))
        dataset=source.Datasets(None,paths,targets)
        inputs[subset]=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,pin_memory=True,
                                  drop_last=True,collate_fn=source.collate_MIL)
    restore_rng(after_init)
    for epoch in range(2):
        value=author_training().train(SimpleNamespace(num_classes=10),reference,reference_opt,inputs['train'],'cpu',epoch)
        assert value==losses[epoch];scheduler.step()
        metrics=author_training().evaluate(SimpleNamespace(num_classes=10),reference,inputs['val'],'cpu')
        if epoch==1:
            assert list(metrics[0])==actual.last_source_metrics['auc']
            assert sum(metrics[0])/len(metrics[0])==actual.validation_metrics(validations[-1])['m4_val_mean_auc']
    assert_equal(actual.network.state_dict(),reference.state_dict())
    assert_equal(opt.state_dict(),reference_opt.state_dict());assert_equal(actual.scheduler.state_dict(),scheduler.state_dict())
    assert_equal(expected_rng,rng_state())
    assert actual.completed_epochs==2 and actual.optimizer_updates==2*len(bundle.records('train'))


def test_actual_complete_disk_resume_and_unconfigured_inference_model(tmp_path):
    torch.set_num_threads(1);seed_everything(42)
    bundle=Bundle(create_m4_demo(tmp_path/'data'))
    actual=M4MIL(bundle);actual.configure_training(config());opt=actual.make_optimizer(.0001,.00004)
    actual.start_epoch(0);actual.epoch(bundle,opt,'cpu');actual.finish_epoch(bundle,'cpu',config())
    actual.predict_subset(bundle,'val','cpu',config())
    path=tmp_path/'full.pt';torch.save({'model':actual.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state()},path)
    assert path.stat().st_size>50_000_000
    actual.start_epoch(1);actual.epoch(bundle,opt,'cpu');actual.finish_epoch(bundle,'cpu',config())
    expected_val=actual.predict_subset(bundle,'val','cpu',config());expected_rng=rng_state()
    restored=M4MIL(bundle);restored.configure_training(config());restored_opt=restored.make_optimizer(.0001,.00004)
    state=torch.load(path,map_location='cpu',weights_only=False)
    restored.load_state_dict(state['model']);restored_opt.load_state_dict(state['optimizer']);restored.validate_optimizer_state();restore_rng(state['rng'])
    restored.start_epoch(1);restored.epoch(bundle,restored_opt,'cpu');restored.finish_epoch(bundle,'cpu',config())
    assert expected_val.equals(restored.predict_subset(bundle,'val','cpu',config()))
    assert_equal(actual.state_dict(),restored.state_dict());assert_equal(opt.state_dict(),restored_opt.state_dict());assert_equal(expected_rng,rng_state())
    restored.validate_optimizer_state()
    # Actual evaluation constructs no optimizer; source state must still load.
    inference=M4MIL(bundle);inference.load_state_dict(actual.state_dict())
    predicted=inference.predict_subset(bundle,'test','cpu',config())
    assert predicted.equals(actual.predict_subset(bundle,'test','cpu',config()))
    assert 'm4_test_mean_auc' in inference.validation_metrics(predicted)
    assert 'm4_val_mean_auc' not in inference.validation_metrics(predicted)


@pytest.mark.parametrize('key,value',[('grad_clip',1.),('max_instances',32),('lr',.001),('weight_decay',1e-5),('selection_metric','macro_auc')])
def test_no_generic_recipe_or_truncation_substitution(tmp_path,key,value):
    torch.set_num_threads(1)
    bundle=Bundle(create_m4_demo(tmp_path/'data'))
    actual=M4MIL(bundle);cfg=config();cfg[key]=value
    with pytest.raises(ValueError,match='M4'):actual.configure_training(cfg)


@pytest.mark.parametrize('mutation',['label_provenance','feature_path','feature_bytes'])
def test_immutable_endpoint_provenance_and_feature_identity(tmp_path,mutation):
    torch.set_num_threads(1)
    bundle=Bundle(create_m4_demo(tmp_path/'data'))
    actual=M4MIL(bundle)
    if mutation=='label_provenance':
        bundle.config['multilabel']['label_provenance']='silently replaced assays'
    elif mutation=='feature_path':
        bundle.rows.loc[0,'path']=bundle.rows.loc[1,'path']
    else:
        import h5py
        from milplus.data import resolve
        with h5py.File(resolve(bundle.root,bundle.rows.iloc[0]['path']),'r+') as handle:
            handle['features'][0,0]+=1
    with pytest.raises(ValueError,match='M4'):actual.check_inputs()

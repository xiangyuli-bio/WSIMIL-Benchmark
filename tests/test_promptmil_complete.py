"""Original RGB prompt/DINO/DSMIL operators, accumulated gradients and real input contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from milplus.data import Bundle
from milplus.engine import rng_state,restore_rng,seed_everything
from milplus.models import build_bundle_model
from milplus.promptmil import PromptMIL,RECIPES
from milplus.promptmil_data import PromptInputs,pack
from milplus.promptmil_demo import create_demo
from milplus.promptmil_source import WEIGHTS,author_modules
from milplus.registry import inspect_method,resolve_for_bundle,check_compatibility
from milplus.upstream import load_definitions

ASSETS=Path('/data/xiangyuli/mil_baseline_puls_work/assets/promptmil_author_weights_20261026_r3')


def exact(a,b):
    if torch.is_tensor(a):assert torch.is_tensor(b)and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray)and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert isinstance(b,dict)and a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)is type(b)and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert type(a)is type(b)and a==b


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root=tmp_path_factory.mktemp('prompt_original_rgb')
    return {c:create_demo(root/c,ASSETS/WEIGHTS[c]['filename'],c,patches=3)for c in ['brca','bright','crc']}


def model(root,edition='source',**kwargs):
    b=Bundle(root);m=PromptMIL(b,'promptmil_'+b.promptmil.condition+'_dino_dsmil_'+edition,**kwargs)
    m.configure_training({'epochs':2,'max_instances':None,'grad_clip':None})
    o=m.make_optimizer(m.recipe['lr'],m.recipe['weight_decay']);return b,m,o


@pytest.mark.parametrize('condition',['brca','bright','crc'])
def test_actual_original_teacher_and_full_prompt_network_are_present(data,condition):
    b,m,_=model(data[condition])
    assert m.backbone.num_features==192 and len(m.backbone.blocks)==12
    assert m.backbone.blocks[0].attn.num_heads==3
    assert m.backbone.patch_embed.proj.weight.shape==(192,3,16,16)
    assert m.backbone.prompt_embeddings.shape==(1,1,192)
    assert [n for n,p in m.backbone.named_parameters()if p.requires_grad]==['prompt_embeddings']
    assert m.classifier.b_classifier.q.weight.shape==(128,192)
    assert m.classifier.b_classifier.v[1].weight.shape==(192,192)
    assert m.classifier.b_classifier.fcc.weight.shape==(len(b.classes),len(b.classes),192)
    assert m.protocol()['full_original_pretrained_backbone_frozen']
    assert m.assets.condition==condition
    assert resolve_for_bundle('Prompt-MIL',b)['id']=='promptmil_'+condition+'_dino_dsmil_paper_reference'
    check_compatibility(resolve_for_bundle('Prompt-MIL',b),b)
    with pytest.raises(ValueError,match='raw-RGB'):
        m(torch.randn(3,192))


def literal_trainer(actual,optimizer,epochs):
    # Execute the author's complete trainer class. Only Lightning's explicit
    # manual-backward/optimizer hooks and logging are bridged to plain Torch;
    # no forward, cached-gradient or training_step body is reimplemented.
    _,_,functions=author_modules()
    cls=load_definitions('promptmil','pl_model/mil_e2e_trainer.py',['MilE2EModule'],
        {'pl':SimpleNamespace(LightningModule=nn.Module),'np':np,'torch':torch,'nn':nn,
         'F':torch.nn.functional,'lr_scheduler':torch.optim.lr_scheduler}).MilE2EModule
    owner=cls.__new__(cls);nn.Module.__init__(owner)
    owner.backbone=copy.deepcopy(actual.backbone);owner.classifier=copy.deepcopy(actual.classifier)
    owner.loss=nn.CrossEntropyLoss()if actual.recipe['loss']=='CE'else nn.BCEWithLogitsLoss()
    owner.classifer_fuc=functions.dsmil_forward_ce if actual.recipe['loss']=='CE'else functions.dsmil_forward
    owner.num_classes=actual.classes;owner.freeze_backbone=False
    owner.batch_size_train=actual.recipe['train_chunk'];owner.batch_size_eval=actual.recipe['eval_augmentation_chunk']
    owner.accumulate_grad_batches=8;owner.transforms_train=nn.Sequential();owner.transforms_eval=nn.Sequential()
    owner.manual_backward=lambda loss:loss.backward()
    owner.log=lambda *a,**k:None;owner.log_dict=lambda *a,**k:None;owner.train_metrics=lambda *a,**k:None
    owner.lr=actual.recipe['lr'];owner.args=SimpleNamespace(adam=actual.recipe['adam'],lr_factor=1.,epochs=epochs,weight_decay=actual.recipe['weight_decay'])
    native_optimizers=owner.configure_optimizers();opt=native_optimizers['optimizer'];scheduler=native_optimizers['lr_scheduler']
    owner.optimizers=lambda:opt;owner.lr_schedulers=lambda:scheduler
    return owner,opt,scheduler


@pytest.mark.parametrize('condition',['brca','bright'])
def test_every_dual_loss_prompt_gradient_and_8plus1_source_update_match_literal_trainer(data,condition):
    torch.set_num_threads(2);seed_everything(10611)
    b,actual,opt=model(data[condition]);owner,refopt,refsched=literal_trainer(actual,opt,2)
    actual.train();owner.train()
    for epoch in [0,1]:
        actual.start_epoch(epoch)
        for i,row in enumerate(b.records('train')):
            pixels=b.promptmil.pixels(row);label=torch.tensor([row['target']])
            before=rng_state();terms,_,_=actual.train_bag(pixels,label,opt,i);after=rng_state()
            restore_rng(before);reference_loss=owner.training_step((pixels.unsqueeze(0),label),i)
            exact(after,rng_state());assert terms['loss']==float(reference_loss.detach())
            exact(actual.backbone.state_dict(),owner.backbone.state_dict())
            exact(actual.classifier.state_dict(),owner.classifier.state_dict())
            exact(opt.state_dict(),refopt.state_dict())
            for n,p in actual.named_parameters():
                q=dict(owner.named_parameters()).get(n)
                if q is not None:
                    assert (p.grad is None)==(q.grad is None),n
                    if p.grad is not None:exact(p.grad,q.grad)
        actual.finish_epoch(b,'cpu',{});owner.on_train_epoch_end();exact(actual.scheduler.state_dict(),refsched.state_dict())
        assert actual.pending_bags==1 and actual.optimizer_updates==epoch+1
        assert actual.backbone.prompt_embeddings.grad is not None
    assert actual.backbone.prompt_embeddings.grad.abs().max()>0
    assert actual.classifier.i_classifier.fc[0].weight.grad is not None
    assert all(p.grad is None for p in actual.backbone.parameters()if not p.requires_grad)


@pytest.mark.parametrize('edition',['source','paper_reference'])
def test_checkpoint_keeps_pending_prompt_and_classifier_gradients_scheduler_and_next_RGB_order(data,tmp_path,edition):
    torch.set_num_threads(2);seed_everything(10621)
    b,first,opt=model(data['brca'],edition)
    first.start_epoch(0);first.epoch(b,opt,'cpu');first.finish_epoch(b,'cpu',{})
    assert first.pending_bags==(1 if edition=='source'else 0)
    assert first.optimizer_updates==(1 if edition=='source'else 2)
    p=tmp_path/'actual_complete.pt';torch.save({'model':first.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state()},p)
    first.start_epoch(1);first.epoch(b,opt,'cpu');first.finish_epoch(b,'cpu',{});expected=rng_state()
    _,second,other=model(data['brca'],edition)
    state=torch.load(p,weights_only=False);second.load_state_dict(state['model']);other.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    second.start_epoch(1);second.epoch(b,other,'cpu');second.finish_epoch(b,'cpu',{})
    exact(first.state_dict(),second.state_dict());exact(opt.state_dict(),other.state_dict());exact(expected,rng_state())
    assert len(first.draw_history)==18 and first.actual_bags==18


def test_complete_deep_prompt_extension_retains_all_twelve_transformers_and_real_gradient(data):
    torch.set_num_threads(2);b,m,opt=model(data['brca'],deep_prompt=True);m.train()
    assert m.backbone.deep_prompt_embeddings.shape==(11,1,192)
    row=b.records('train')[0]
    m.train_bag(b.promptmil.pixels(row),torch.tensor([row['target']]),opt,0)
    assert m.backbone.deep_prompt_embeddings.grad is not None and torch.isfinite(m.backbone.deep_prompt_embeddings.grad).all()
    assert m.backbone.deep_prompt_embeddings.grad.abs().sum()>0
    assert sum(p.numel()for p in m.backbone.parameters())>5_500_000


def test_predict_exports_true_source_Sigmoid_separately_and_never_consumes_pending_gradients(data):
    torch.set_num_threads(2);b,m,opt=model(data['bright']);m.train();row=b.records('train')[0]
    m.train_bag(b.promptmil.pixels(row),torch.tensor([row['target']]),opt,0)
    state=copy.deepcopy(m.state_dict());frame=m.predict_subset(b,'test','cpu',{})
    exact(state,m.state_dict())
    assert not np.allclose(frame[[f'promptmil_source_score_{i}'for i in range(3)]].sum(1),1.)
    assert np.allclose(frame[['p_0','p_1','p_2']].sum(1),1.)
    metrics=m.validation_metrics(frame);assert np.isfinite(metrics['promptmil_source_AUROC'])
    assert len(frame)==6 and set(frame.slide_id)=={r['slide_id']for r in b.records('test')}


def test_multiclass_source_AUROC_uses_each_raw_sigmoid_column_without_probability_renormalization(data):
    _,m,_=model(data['brca'])
    frame=pd.DataFrame({'global_patient_id':['a','b','c','d'],'target':[0,0,1,1],
        'promptmil_source_score_0':[.9,.8,.2,.1],'promptmil_source_score_1':[.6,.3,.8,.2],
        'source_loss':[1.,1.,1.,1.]})
    from sklearn.metrics import roc_auc_score
    expected=(roc_auc_score(frame.target==0,frame.promptmil_source_score_0)+roc_auc_score(frame.target==1,frame.promptmil_source_score_1))/2
    assert m.validation_metrics(frame)['promptmil_source_AUROC']==expected


def test_raw_bytes_ids_splits_and_original_weights_are_bound_and_invalid_data_refused(data,tmp_path):
    b=Bundle(data['brca']);assert b.promptmil.inventory(b.records('train')[0],True)['shape']==[3,224,224,3]
    with pytest.raises(ValueError,match='complete prompt-tuning'):
        b.load(b.records('train')[0])
    with pytest.raises(ValueError,match='not frozen spatial'):
        b.check(deep=True,require_coords=True)
    _,m,_=model(data['brca'])
    with pytest.raises(ValueError,match='no gradient clipping'):
        m.configure_training({'epochs':2,'grad_clip':5,'max_instances':None})
    with pytest.raises(ValueError,match='every RGB patch'):
        m.configure_training({'epochs':2,'grad_clip':None,'max_instances':2})
    from milplus.promptmil_source import build_original
    bad=tmp_path/'not_original.pth';bad.write_bytes(b'not pretrained')
    with pytest.raises(ValueError,match='actual original cohort'):
        build_original(bad,'brca',2)


def test_packer_preserves_original_RGB_bytes_and_table_order_without_resize(tmp_path):
    from PIL import Image
    images=[]
    for i in range(3):
        pixel=np.random.default_rng(i).integers(0,256,(224,224,3),dtype=np.uint8)
        path=tmp_path/f'{i}.png';Image.fromarray(pixel).save(path)
        images.append((path,pixel))
    table=tmp_path/'patches.csv';pd.DataFrame([{'slide_id':'wsi','patch_id':str(i),'path':str(p)}for i,(p,x)in enumerate(images)]).to_csv(table,index=False)
    index=pack(table,tmp_path/'packed');p=pd.read_csv(index).iloc[0]['path']
    with h5py.File(p,'r')as f:
        assert np.array_equal(f['rgb'][:],np.stack([x for p,x in images]))
        assert f['patch_ids'].asstr()[:].tolist()==['0','1','2']


def test_native_recipe_budgets_and_balanced_chunks_are_not_silently_shrunk(data):
    b,m,_=model(data['brca'])
    assert [len(x)for x in m.split_tensor(torch.zeros(1025,1))]==[342,342,341]
    assert RECIPES['promptmil_brca_dino_dsmil_source']['epochs']==50
    assert RECIPES['promptmil_bright_dino_dsmil_source']['epochs']==50
    assert RECIPES['promptmil_crc_dino_dsmil_source']['epochs']==40
    assert RECIPES['promptmil_crc_dino_dsmil_paper_reference']['lr']==.0005
    assert m.recipe['accumulation']==8 and m.recipe['train_chunk']==512


def test_explicit_five_class_output_port_keeps_real_pretrained_backbone_and_both_losses(tmp_path):
    torch.set_num_threads(2)
    root=create_demo(tmp_path/'five_classes',ASSETS/WEIGHTS['bright']['filename'],'bright',patches=3,classes=5)
    b,m,opt=model(root);m.train();row=b.records('train')[4]
    terms,_,_=m.train_bag(b.promptmil.pixels(row),torch.tensor([row['target']]),opt,0)
    assert np.isfinite(terms['loss'])and m.classes==5
    assert m.classifier.b_classifier.fcc.weight.shape==(5,5,192)
    assert m.backbone.prompt_embeddings.grad is not None
    assert m.classifier.i_classifier.fc[0].weight.grad.shape==(5,192)
    assert m.classifier.b_classifier.fcc.weight.grad is not None
    assert len(m.backbone.blocks)==12 and m.backbone.blocks[0].attn.num_heads==3


def test_changed_original_pixel_or_patch_identity_bytes_are_rejected(tmp_path):
    root=create_demo(tmp_path/'integrity',ASSETS/WEIGHTS['brca']['filename'],patches=3)
    b=Bundle(root);row=b.records('train')[0]
    with h5py.File(root/row['path'],'a')as f:f['rgb'][0,0,0,0]^=1
    with pytest.raises(ValueError,match='RGB or aligned patch identity bytes changed'):
        b.promptmil.inventory(row,True)


def test_public_cached_asset_command_keeps_the_exact_existing_author_file():
    import os,subprocess,sys
    command=[sys.executable,'-m','milplus','prepare-promptmil-assets','--condition','brca','--output',str(ASSETS)]
    result=subprocess.run(command,env=os.environ.copy(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert str(ASSETS/WEIGHTS['brca']['filename'])in result.stdout


def test_synthetic_validation_and_test_pixels_are_distinct_from_training_pixels(data):
    import hashlib
    b=Bundle(data['brca']);seen={}
    for subset in ['train','val','test']:
        values=set()
        for row in b.records(subset):
            with h5py.File(b.root/row['path'],'r')as f:
                values.add(hashlib.sha256(f['rgb'][:].tobytes()).hexdigest())
        assert len(values)==len(b.records(subset))
        seen[subset]=values
    assert seen['train'].isdisjoint(seen['val'])and seen['train'].isdisjoint(seen['test'])
    assert seen['val'].isdisjoint(seen['test'])

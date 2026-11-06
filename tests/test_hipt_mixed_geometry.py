"""Explicit mixed-scan geometry: synthetic controls, actual teacher integration.

Coordinate scaling below is confined to declared synthetic software fixtures.
Clinical features/coordinates are never converted to satisfy the contract.
"""
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.hipt_data import aligned_inputs, patch_stride_for, prepare, schema_contract
from milplus.hipt_demo import create_hipt_demo
from milplus.hipt_source import COMMIT, PATCH_SHA
from milplus.io import read_json, write_json


def schema():
    return dict(schema_version=2,kind='hipt_complete_16x16_pyramid',source_commit=COMMIT,
        synthetic=True,grid_shape=[16,16],coordinate_frame='level0_pixels',
        patch_stride_by_slide={'scan40':512,'scan20':256},
        patch_encoder='HIPT_ViT256_DINO',patch_encoder_weights_sha256=PATCH_SHA,
        **{k:'Explicit synthetic unit-test geometry, not clinical extraction' for k in
           ['patching_provenance','patch_encoder_provenance','preprocessing_provenance']})


def owner():
    return SimpleNamespace(rows=pd.DataFrame({'slide_id':['scan40','scan20']}),
        task_type='classification',dim=384,config={'synthetic':True,'training_unit':'slide'},
        encoder=dict(embedding_space='HIPT_ViT256_DINO',weights_parent_sha256=PATCH_SHA,
            spatial={'unit':'level0_pixels','origin':'top_left'}))


@pytest.mark.parametrize('kind',['missing','extra','empty','not_map','both','v1_map','bool_version','future_version'])
def test_stride_schema_rejects_ambiguous_or_incomplete_metadata(kind):
    value=schema()
    if kind=='missing':del value['patch_stride_by_slide']['scan20']
    elif kind=='extra':value['patch_stride_by_slide']['not_a_task_slide']=256
    elif kind=='empty':value['patch_stride_by_slide']={}
    elif kind=='not_map':value['patch_stride_by_slide']=[512,256]
    elif kind=='both':value['patch_stride']=512
    elif kind=='v1_map':value.update(schema_version=1,patch_stride=512)
    elif kind=='bool_version':value['schema_version']=True
    else:value['schema_version']=3
    with pytest.raises(ValueError,match='HIPT'):schema_contract(value,owner())


@pytest.mark.parametrize('stride',[True,False,0,-1,None,'512',[],{},float('nan'),float('inf'),-float('inf'),.5,2**80])
def test_invalid_declared_level0_stride_rejected(stride):
    value=schema();value['patch_stride_by_slide']['scan20']=stride
    with pytest.raises(ValueError,match='HIPT'):schema_contract(value,owner())


def test_v1_and_unversioned_diagnostic_scalar_geometry_remain_supported():
    value=schema();value.pop('patch_stride_by_slide');value.update(schema_version=1,patch_stride=512)
    schema_contract(value,owner())
    assert patch_stride_for(value)==512
    assert patch_stride_for(dict(patch_stride=512,coordinate_frame='level0_pixels'))==512
    value['coordinate_frame']='micrometers';value['patch_stride']=.5
    assert patch_stride_for(value)==.5


@pytest.mark.parametrize('row',[None,{},dict(slide_id='not_a_task_slide'),dict(slide_id=20)])
def test_v2_does_not_fall_back_to_another_slide_stride(row):
    with pytest.raises(ValueError,match='missing'):patch_stride_for(schema(),row)


def test_mixed_physical_arrays_recover_original_features_without_resampling(tmp_path):
    value=schema();bundle=owner();bundle.root=tmp_path
    def load(row,require_coords=False):
        with h5py.File(tmp_path/row['path'],'r') as archive:
            return torch.from_numpy(archive['features'][:]),archive['coords'][:]
    bundle.load=load
    features=np.arange(256*384,dtype=np.float32).reshape(256,384)/1000
    rng=np.random.default_rng(2026)
    for slide_id,stride in value['patch_stride_by_slide'].items():
        origin=np.array([[1024,2048]],dtype=np.int64)
        coords=origin[0]+np.array([[x*stride,y*stride] for y in range(16) for x in range(16)])
        permutation=rng.permutation(256)
        path=tmp_path/(slide_id+'.h5');pyramid=tmp_path/(slide_id+'_pyramid.h5')
        with h5py.File(path,'w') as archive:
            archive['features']=features[permutation];archive['coords']=coords[permutation]
        with h5py.File(pyramid,'w') as archive:
            archive['region_ids']=np.zeros(256,dtype=np.int64);archive['region_origins']=origin
        row=dict(slide_id=slide_id,path=path.name)
        result,actual,origins,order=aligned_inputs(bundle,row,pyramid,value)
        np.testing.assert_array_equal(result[0].numpy(),features)
        np.testing.assert_array_equal(actual[order[0]],coords)
        np.testing.assert_array_equal(origins,origin)
        wrong=copy.deepcopy(value);wrong['patch_stride_by_slide'][slide_id]=256 if stride==512 else 512
        with pytest.raises(ValueError,match='physical 16x16'):aligned_inputs(bundle,row,pyramid,wrong)


@pytest.fixture
def weights():
    path=Path(os.environ.get('MILPLUS_HIPT_WEIGHTS','/data/milplus_model_weights/hipt_vit4k_xs_teacher_backbone.pt'))
    if not path.is_file():pytest.skip('Actual registered HIPT teacher required; no random replacement')
    return path


@pytest.fixture
def mixed(tmp_path,weights):
    torch.set_num_threads(1)
    root=create_hipt_demo(tmp_path/'synthetic',weights,regions=2)
    bundle=Bundle(root)
    value=copy.deepcopy(bundle.hipt.meta['schema']);value.pop('patch_stride')
    value.update(schema_version=2,patch_stride_by_slide={})
    config=yaml.safe_load((root/'task.yaml').read_text())
    del config['encoders']['toy']['hipt']
    config['encoders']['toy']['spatial'].pop('patch_size',None)
    slides=pd.read_csv(root/'slides.csv')
    for index,row in enumerate(bundle.rows.to_dict('records')):
        stride=256 if index%2 else 512
        value['patch_stride_by_slide'][row['slide_id']]=stride
        if stride==256:
            with h5py.File(root/row['path'],'r+') as archive:archive['coords'][:]=archive['coords'][:]/2
            with h5py.File(root/'pyramids'/f"{row['slide_id']}.h5",'r+') as archive:
                archive['region_origins'][:]=archive['region_origins'][:]/2
            selected=slides.slide_id==row['slide_id']
            slides.loc[selected,['width_level0','height_level0']]/=2
    slides.to_csv(root/'slides.csv',index=False)
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    manifest=prepare(Bundle(root),root/'pyramids.csv',value,weights,root/'hipt_mixed.json')
    config['encoders']['toy']['hipt']={'manifest':manifest.name}
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    return Bundle(root)


def exact(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def test_mixed_manifest_round_trip_and_tamper_rejection(mixed):
    assert mixed.hipt.meta['schema_version']==mixed.hipt.meta['schema']['schema_version']==2
    assert set(mixed.hipt.meta['schema']['patch_stride_by_slide'].values())=={256,512}
    for row in mixed.rows.to_dict('records'):
        result,coords,origins,order=mixed.hipt.load(row)
        stride=patch_stride_for(mixed.hipt.meta['schema'],row)
        offsets=np.array([[x*stride,y*stride] for y in range(16) for x in range(16)])
        assert result.shape==(2,256,384)
        for index,origin in enumerate(origins):np.testing.assert_array_equal(coords[order[index]],origin+offsets)
    row=mixed.records('train')[0]
    altered=read_json(mixed.hipt.path)
    previous=altered['schema']['patch_stride_by_slide'][row['slide_id']]
    altered['schema']['patch_stride_by_slide'][row['slide_id']]=256 if previous==512 else 512
    write_json(mixed.hipt.path,altered)
    with pytest.raises(ValueError,match='manifest changed'):mixed.hipt.load(row)
    with pytest.raises(ValueError,match='physical 16x16'):Bundle(mixed.root).hipt.load(row)


def test_manifest_version_cannot_hide_per_slide_schema(mixed):
    altered=read_json(mixed.hipt.path);altered['schema_version']=1
    write_json(mixed.hipt.path,altered)
    with pytest.raises(ValueError,match='versions differ'):Bundle(mixed.root)


@pytest.mark.parametrize('profile',['hipt_lgp_frozen_source','hipt_lgp_finetune_source'])
def test_both_original_models_exact_resume_on_mixed_geometry(mixed,tmp_path,profile):
    cfg=dict(epochs=2,threads=1)
    full=train(mixed.root,profile,tmp_path/'full',config=cfg)
    partial=train(mixed.root,profile,tmp_path/'partial',config=cfg,stop_after=1)
    train(mixed.root,profile,partial,config=cfg,resume=True)
    a,b=[torch.load(path/'last.pt',weights_only=False) for path in [full,partial]]
    for key in ['model','optimizers','rng','history','best','bad']:exact(a[key],b[key])
    assert a['model']['_extra_state']['optimizer_steps']==2*len(mixed.records('train'))
    assert evaluate(full)==evaluate(partial)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'),pd.read_csv(partial/'test_predictions.csv'))


def test_cli_mixed_geometry_preparation(mixed,weights):
    root=mixed.root
    write_json(root/'schema_v2.json',mixed.hipt.meta['schema'])
    result=subprocess.run([sys.executable,'-m','milplus','prepare-hipt','--data',str(root),
        '--index',str(root/'pyramids.csv'),'--schema',str(root/'schema_v2.json'),
        '--weights',str(weights),'--output',str(root/'hipt_cli_v2.json')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    exact(read_json(root/'hipt_cli_v2.json'),read_json(mixed.hipt.path))

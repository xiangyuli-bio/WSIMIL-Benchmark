import copy

import h5py
import numpy as np
import pandas as pd
import pytest

from milplus.data import Bundle
from milplus.hdmil_data import prepare, read_rgb, validate_schema
from milplus.hdmil_demo import create_hdmil_demo
from milplus.io import read_json, write_json


@pytest.fixture
def bundle(tmp_path):
    return Bundle(create_hdmil_demo(tmp_path/'data',patients=20,dimension=1024))


def test_full_registered_rgb_and_true_feature_order(bundle):
    inventory=bundle.check(deep=True)
    assert inventory['feature_inventory_sha256']
    for row in bundle.records('train'):
        x,rgb=bundle.hdmil.load(row)
        assert x.shape==(40,1024) and rgb.shape==(40,16,16,3)
        assert bundle.hdmil.inventory(row,True)['physical']['side_level0']==256


@pytest.mark.parametrize('failure',['permutation','rgb_float','wrong_pixels','rgb_missing','duplicated_coords','wrong_mpp','absent_mpp','negative_coords'])
def test_measured_input_faults_refused_before_training(bundle,failure):
    row=bundle.records('train')[0]; asset=bundle.hdmil.meta['slides'][row['slide_id']]
    with h5py.File(asset['path'],'r+') as f:
        if failure=='permutation':f['coords'][:]=f['coords'][:][::-1]
        elif failure=='rgb_float':
            rgb=f['rgb'][:].astype(np.float32);del f['rgb'];f['rgb']=rgb
        elif failure=='wrong_pixels':
            rgb=f['rgb'][:,:,:8];del f['rgb'];f['rgb']=rgb
        elif failure=='rgb_missing':del f['rgb']
        elif failure=='duplicated_coords':f['coords'][1]=f['coords'][0]
        elif failure=='wrong_mpp':f.attrs['low_mpp']=4.
        elif failure=='absent_mpp':del f.attrs['level0_mpp']
        else:f['coords'][0]=[-256,0]
    with pytest.raises(ValueError):read_rgb(bundle,row,asset['path'],bundle.hdmil.schema)
    with pytest.raises(ValueError,match='archive changed'):bundle.hdmil.load(row)


@pytest.mark.parametrize('field,value',[('low_magnification',2.5),('high_patch_pixels',224),('rgb_layout','NCHW'),('coordinate_provenance','')])
def test_incompatible_extraction_contracts_rejected(bundle,field,value):
    schema=copy.deepcopy(bundle.hdmil.schema);schema[field]=value
    with pytest.raises(ValueError):validate_schema(schema)


def test_mixed_scanner_mpp_preserves_per_slide_same_physical_extent(bundle):
    row=bundle.records('train')[0]; asset=bundle.hdmil.meta['slides'][row['slide_id']]
    with h5py.File(asset['path'],'r+') as f:f.attrs.update(level0_mpp=.4,high_mpp=.4,low_mpp=6.4)
    _,_,physical=read_rgb(bundle,row,asset['path'],bundle.hdmil.schema)
    assert physical['side_level0']==256 and physical['low_mpp']==6.4


@pytest.mark.parametrize('field',['global_patient_id','subset','target','preparer_sha256'])
def test_registered_task_or_preparer_identity_cannot_be_reassigned(bundle,field):
    path=bundle.hdmil.path; meta=read_json(path)
    if field=='preparer_sha256':meta[field]='0'*64
    else:
        row=bundle.records('train')[0];meta['slides'][row['slide_id']][field]='changed'
    write_json(path,meta)
    with pytest.raises(ValueError):Bundle(bundle.root)


def test_duplicate_or_missing_index_row_and_manifest_overwrite_are_rejected(bundle):
    original=pd.read_csv(bundle.hdmil.meta['index'])
    for i,table in enumerate([original.iloc[:-1],pd.concat([original,original.iloc[:1]])]):
        index=bundle.root/f'bad{i}.csv';table.to_csv(index,index=False)
        with pytest.raises(ValueError,match='exact task'):prepare(bundle,index,bundle.hdmil.schema,bundle.root/f'bad{i}.json')
    with pytest.raises(ValueError,match='manifest exists'):
        prepare(bundle,bundle.hdmil.meta['index'],bundle.hdmil.schema,bundle.hdmil.path)

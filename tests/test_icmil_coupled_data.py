import copy

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from milplus.data import Bundle
from milplus.icmil_coupled_demo import create_demo
from milplus.icmil_coupled_pack import pack
from milplus.io import read_json

WEIGHTS='/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/development/icmil_coupled_v92/legacy14_assets_r1/resnet50-11ad3fa6.pth'


def test_original_raw_pixels_cannot_enter_frozen_feature_methods(tmp_path):
    from milplus.registry import require_runnable,check_compatibility
    data=create_demo(tmp_path/'data',WEIGHTS);b=Bundle(data)
    with pytest.raises(ValueError,match='raw RGB'):check_compatibility(require_runnable('abmil'),b)
    with pytest.raises(ValueError,match='full extraction'):b.load(b.records('train')[0])
    report=b.check(deep=True);assert len(report['features'])==6 and all(r['icmil_coupled_rgb']['shape']==[5,256,256,3] for r in report['features'])


def test_changed_raw_patch_ids_or_pixels_are_rejected(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS);b=Bundle(data);p=data/'pixels/synthetic_0.h5'
    with h5py.File(p,'r+') as f:f['rgb'][0,0,0,0]^=1
    with pytest.raises(ValueError,match='RGB bytes'):b.check(deep=True)


def test_patient_partition_leakage_is_rejected_before_any_encoder_training(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS);frame=pd.read_csv(data/'slides.csv');frame['global_patient_id']='same_patient';frame.to_csv(data/'slides.csv',index=False)
    with pytest.raises(ValueError,match='crosses splits'):Bundle(data)


def test_frozen_initialization_or_input_normalization_cannot_change(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS);c=yaml.safe_load((data/'task.yaml').read_text())
    p=data/'icmil_coupled_manifest.json';meta=read_json(p);meta['schema']['normalization']='imagenet_mean_std'
    import json
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='ToTensor'):Bundle(data)


def test_literal_PIL_pack_preserves_every_RGB_pixel_path_order_and_source_hash(tmp_path):
    rng=np.random.default_rng(4);images={}
    for name in ['b','a']:
        array=rng.integers(0,256,(256,256,3),dtype=np.uint8);p=tmp_path/(name+'.png');Image.fromarray(array).save(p);images[str(p)]=array
    table=tmp_path/'patches.csv';pd.DataFrame([{'slide_id':'slideA','path':'b.png'},{'slide_id':'slideA','path':'a.png'}]).to_csv(table,index=False)
    index=pack(table,tmp_path/'packed');path=pd.read_csv(index).iloc[0].path
    with h5py.File(path) as f:
        ids=f['patch_ids'].asstr()[:].tolist();assert ids==sorted(images)
        for n,key in enumerate(ids):assert np.array_equal(f['rgb'][n],images[key])
    assert read_json(tmp_path/'packed/pack_receipt.json')['state']=='completed'


def test_pack_refuses_silent_color_or_resolution_changes(tmp_path):
    Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(tmp_path/'wrong.png')
    table=tmp_path/'patches.csv';pd.DataFrame([{'slide_id':'slideA','path':'wrong.png'}]).to_csv(table,index=False)
    with pytest.raises(ValueError,match='RGB256'):pack(table,tmp_path/'packed')

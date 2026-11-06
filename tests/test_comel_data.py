import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from milplus.comel_demo import create_demo
from milplus.comel_data import prepare
from milplus.data import Bundle


def original(data):
    config=yaml.safe_load((data/'task.yaml').read_text());config['encoders']['synthetic_fm'].pop('comel')
    (data/'task.yaml').write_text(yaml.safe_dump(config));bundle=Bundle(data)
    manifest=json.loads((data/'comel_manifest.json').read_text())
    return bundle,{'schema_version':1,'tasks':manifest['tasks'],
        'slide_tasks':{r['slide_id']:manifest['tasks'][r['task']]['id'] for r in manifest['rows']},'provenance':'Synthetic software input audit'}


def test_manifest_binds_actual_unique_patch_order(tmp_path):
    data=create_demo(tmp_path/'data');bundle,schema=original(data);row=bundle.records('train')[0]
    with h5py.File(row['path'],'r+') as f:f['patch_ids'][1]=f['patch_ids'][0]
    with pytest.raises(ValueError,match='unique'):prepare(bundle,schema,tmp_path/'manifest.json')


def test_training_patch_ground_truth_is_rejected(tmp_path):
    data=create_demo(tmp_path/'data');bundle,schema=original(data);p=data/'features.csv';d=pd.read_csv(p).fillna('')
    row=bundle.records('train')[0];labels=data/'forbidden_labels.npy';np.save(labels,np.zeros(17,dtype='int64'))
    d.loc[d.slide_id==row['slide_id'],'instance_labels_path']=str(labels);d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='TRAIN uses bag labels only'):prepare(Bundle(data),schema,tmp_path/'manifest.json')


def test_patient_shared_across_continual_tasks_rejected(tmp_path):
    data=create_demo(tmp_path/'data');bundle,schema=original(data);p=data/'slides.csv';d=pd.read_csv(p)
    d['global_patient_id']=d.patient_id;d.loc[d.slide_id.isin(['t0_train_0','t1_train_0']),'global_patient_id']='same_actual_patient'
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='multiple continual tasks'):prepare(Bundle(data),schema,tmp_path/'manifest.json')


def test_wrong_binary_class_offset_and_truncated_task_mapping_rejected(tmp_path):
    data=create_demo(tmp_path/'data');bundle,schema=original(data)
    schema['slide_tasks']['t0_train_0']='task1'
    with pytest.raises(ValueError,match='binary task offset'):prepare(bundle,schema,tmp_path/'manifest.json')
    schema['slide_tasks'].pop('t0_train_0')
    with pytest.raises(ValueError,match='exactly all'):prepare(bundle,schema,tmp_path/'missing.json')


def test_actual_fm_dimension_and_joined_metadata_are_bound(tmp_path):
    data=create_demo(tmp_path/'data');p=data/'task.yaml';d=yaml.safe_load(p.read_text());d['encoders']['synthetic_fm']['dimension']=768
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError,match='selected FM'):Bundle(data)


def test_eval_annotation_tamper_is_rejected(tmp_path):
    data=create_demo(tmp_path/'data');b=Bundle(data);p=Path(b.records('test')[0]['instance_labels_path'])
    a=np.load(p);a[0]=1-a[0];np.save(p,a)
    with pytest.raises(ValueError,match='ground truth changed'):Bundle(data)

import json
from pathlib import Path

import h5py
import pandas as pd
import pytest
import yaml

from milplus.akdpmp_demo import create_demo
from milplus.akdpmp_data import prepare
from milplus.akdpmp_driver import train
from milplus.data import Bundle


def original(path):
    cfg=yaml.safe_load((path/'task.yaml').read_text());cfg['encoders']['synthetic_fm'].pop('akdpmp')
    (path/'task.yaml').write_text(yaml.safe_dump(cfg));bundle=Bundle(path);m=json.loads((path/'akdpmp_manifest.json').read_text())
    return bundle,{'schema_version':1,'tasks':m['tasks'],'slide_tasks':{r['slide_id']:m['tasks'][r['task']]['id'] for r in m['rows']},'provenance':'Explicit software input integrity audit'}


def test_feature_bytes_and_real_patch_identity_are_bound(tmp_path):
    data=create_demo(tmp_path/'data');b=Bundle(data);row=b.records('test')[0]
    with h5py.File(row['path'],'r+') as f:f['features'][0,0]+=1
    with pytest.raises(ValueError,match='feature bytes'):Bundle(data)


def test_duplicate_patch_ids_rejected_before_memory_selection(tmp_path):
    data=create_demo(tmp_path/'data');b,schema=original(data);row=b.records('train')[0]
    with h5py.File(row['path'],'r+') as f:f['patch_ids'][1]=f['patch_ids'][0]
    with pytest.raises(ValueError,match='unique'):prepare(b,schema,tmp_path/'manifest.json')


def test_patient_crossing_continual_task_boundaries_is_rejected(tmp_path):
    data=create_demo(tmp_path/'data');b,schema=original(data);p=data/'slides.csv';d=pd.read_csv(p);d['global_patient_id']=d.patient_id
    d.loc[d.slide_id.isin(['t0_train_0','t1_train_0']),'global_patient_id']='same_real_patient';d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='crosses continual'):prepare(Bundle(data),schema,tmp_path/'manifest.json')


def test_task_mapping_cannot_move_a_class_into_a_different_task(tmp_path):
    data=create_demo(tmp_path/'data');b,schema=original(data);schema['slide_tasks']['t0_train_0']='task1'
    with pytest.raises(ValueError,match='does not belong'):prepare(b,schema,tmp_path/'manifest.json')


def test_selected_FM_or_dimension_cannot_change_after_input_preparation(tmp_path):
    data=create_demo(tmp_path/'data');p=data/'task.yaml';d=yaml.safe_load(p.read_text());d['encoders']['synthetic_fm']['dimension']=768;p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError,match='metadata changed'):Bundle(data)


def test_undeclared_extra_skin_task_budgets_and_tiny_memory_fail_explicitly(tmp_path):
    data=create_demo(tmp_path/'data',preset='skin',task_sizes=[2,1,1,1,1])
    with pytest.raises(ValueError,match='every actual task'):train(data,'akdpmp_clam_skin_patient_reference',tmp_path/'missing')
    with pytest.raises(ValueError,match='capacity'):train(data,'akdpmp_clam_skin_patient_reference',tmp_path/'small',config={'epochs':[1]*5,'buffer_size':2})

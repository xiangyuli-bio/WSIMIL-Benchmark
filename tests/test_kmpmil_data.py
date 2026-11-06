import json,os
from pathlib import Path
import h5py
import pytest
from milplus.kmpmil_demo import create_demo
from milplus.data import Bundle
from milplus.io import digest_object
from milplus.registry import check_compatibility,inspect_method

WEIGHTS=Path(os.environ.get('MILPLUS_KMPMIL_WEIGHTS',Path.home()/'.cache/milplus/qpmilvl/conch.pth'))
SHA=os.environ.get('MILPLUS_KMPMIL_SHA256','40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c')


def mutate(data,change):
    p=Path(data)/'kmpmil_manifest.json';value=json.loads(p.read_text());value.pop('contract');change(value);value['contract']=digest_object(value);p.write_text(json.dumps(value))


def test_actual_four_task_nine_classes_semantic_descriptors_clinical_ensemble_and_compatibility(tmp_path):
    b=Bundle(create_demo(tmp_path/'data',WEIGHTS,SHA));a=b.kmpmil
    assert [len(t['classes']) for t in a.tasks]==[3,2,2,2] and a.seen(3)==list(range(9))
    assert [len(a.descriptors(t)) for t in range(4)]==[10]*4
    assert a.ensemble(0)['count']==[176,154,154] and sum(a.ensemble(3)['count'])==1144
    assert len(a.records(0,'train'))==27
    check_compatibility(inspect_method('KMP-MIL'),b)
    with pytest.raises(ValueError,match='KMP-MIL tasks require'):check_compatibility(inspect_method('ABMIL'),b)


def test_mutated_actual_feature_bytes_rejected(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);b=Bundle(data)
    with h5py.File(b.records('train')[0]['path'],'r+') as f:f['features'][0,0]=42
    with pytest.raises(ValueError,match='feature bytes changed'):Bundle(data)


def test_mutated_actual_patch_order_rejected(tmp_path):
    b=Bundle(create_demo(tmp_path/'data',WEIGHTS,SHA));r=b.records('train')[0]
    with h5py.File(r['path'],'r+') as f:f['patch_ids'][:]=f['patch_ids'][:][::-1]
    with pytest.raises(ValueError,match='features changed'):b.kmpmil.features(r)


@pytest.mark.parametrize('change,message',[
    (lambda v:v['conch'].update(sha256='0'*64),'checkpoint hash mismatch'),
    (lambda v:v['semantic_descriptors'].update(bracs=['missing']),'semantic descriptors'),
    (lambda v:v['class_descriptions'].pop('LUAD'),'clinical descriptions'),
    (lambda v:v['conch']['feature_encoding'].update(proj_contrast=False),'CONCH projection'),
    (lambda v:v['rows'][0].update(global_patient_id='other-patient'),'joined metadata changed')])
def test_actual_weight_text_descriptor_feature_and_patient_contracts(tmp_path,change,message):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);mutate(data,change)
    with pytest.raises(ValueError,match=message):Bundle(data)

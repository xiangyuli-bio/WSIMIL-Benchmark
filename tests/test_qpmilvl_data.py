import json
from pathlib import Path
import h5py
import pytest
import yaml
from milplus.qpmilvl_demo import create_demo
from milplus.data import Bundle
from milplus.qpmilvl_data import prepare
from milplus.io import digest_object
from milplus.registry import check_compatibility,inspect_method
from test_qpmilvl_source import WEIGHTS,SHA


def manifest(data):
    p=Path(data)/'qpmilvl_manifest.json';return p,json.loads(p.read_text())


def rewrite(p,x):
    x.pop('contract',None);x['contract']=digest_object(x);p.write_text(json.dumps(x))


def test_actual_patient_task_feature_patch_text_weight_contract_and_compatible_profile(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);b=Bundle(data);a=b.qpmilvl
    assert len(a.tasks)==4 and a.seen(3)==list(range(8)) and len(a.records(0,'train'))==16
    assert a.ensemble(0)['count']==[66,66] and sum(a.ensemble(3)['count'])==660
    check_compatibility(inspect_method('QPMIL-VL'),b)
    with pytest.raises(ValueError,match='QPMIL-VL tasks require'):check_compatibility(inspect_method('ABMIL'),b)


def test_mutated_real_feature_bytes_rejected(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);b=Bundle(data);row=b.records('train')[0]
    with h5py.File(row['path'],'r+') as f:f['features'][0,0]=10
    with pytest.raises(ValueError,match='feature bytes changed'):Bundle(data)


def test_mutated_ordered_real_patch_ids_rejected(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);b=Bundle(data);r=b.records('train')[0];a=b.qpmilvl
    with h5py.File(r['path'],'r+') as f:
        values=f['patch_ids'][:];f['patch_ids'][:]=values[::-1]
    with pytest.raises(ValueError,match='features changed'):a.features(r)


def test_pretrained_checkpoint_binding_cannot_be_relabelled(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);p,x=manifest(data);x['conch']['sha256']='0'*64;rewrite(p,x)
    with pytest.raises(ValueError,match='checkpoint hash mismatch'):Bundle(data)


def test_undeclared_clinical_class_text_or_malformed_template_rejected(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);p,x=manifest(data);x['class_descriptions'].pop('LUAD');rewrite(p,x)
    with pytest.raises(ValueError,match='clinical descriptions'):Bundle(data)


def test_encoder_projection_and_actual_feature_dimension_are_required(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);p,x=manifest(data);x['conch']['feature_encoding']['proj_contrast']=False;rewrite(p,x)
    with pytest.raises(ValueError,match='CONCH projection'):Bundle(data)


def test_global_patient_and_partition_metadata_cannot_be_changed(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS,SHA);p,x=manifest(data);x['rows'][0]['global_patient_id']='another-patient';rewrite(p,x)
    with pytest.raises(ValueError,match='joined metadata changed'):Bundle(data)

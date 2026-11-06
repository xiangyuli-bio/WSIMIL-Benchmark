import copy,json
from pathlib import Path
import h5py,numpy as np,pytest,yaml
from milplus.data import Bundle
from milplus.io import read_json,write_json,digest_object,digest_file
from milplus.protomil_demo import create_demo
from milplus.protomil_data import actual_features,normalized_features


def mutate_contract(data,change):
    p=Path(data)/'protomil_manifest.json';v=read_json(p);v.pop('contract');change(v);v['contract']=digest_object(v);write_json(p,v)


def test_actual_TRAIN_populations_and_annotated_probes_preserve_original_patient_and_patch_identities(tmp_path):
    data=create_demo(tmp_path/'input');assets=Bundle(data).protomil
    assert len(assets.population)==8 and {r['subset'] for r in assets.population}=={'train'}
    assert len(assets.probes)==10 and {p['region_label'] for p in assets.probes}=={0,1}
    assert all(p['patch_id'].endswith('_patch_'+str(p['patch_index'])) for p in assets.probes)
    assert len({p['global_patient_id'] for p in assets.probes})==2
    train={r['global_patient_id'] for r in assets.rows.values() if r['subset']=='train'}
    assert {p['global_patient_id'] for p in assets.probes}<=train


@pytest.mark.parametrize('field',['patient_id','global_patient_id','subset','path'])
def test_rehashed_main_patient_or_feature_manifest_cannot_change_actual_join(tmp_path,field):
    data=create_demo(tmp_path/'input')
    mutate_contract(data,lambda v:v['rows'][0].__setitem__(field,'forged_identity'))
    with pytest.raises(ValueError,match='record changed'):Bundle(data)


@pytest.mark.parametrize('field',['patient_id','global_patient_id','subset','source_root'])
def test_rehashed_SAE_population_cannot_change_source_TRAIN_patient_identity(tmp_path,field):
    data=create_demo(tmp_path/'input')
    mutate_contract(data,lambda v:v['sae_population'][0].__setitem__(field,'forged_identity'))
    with pytest.raises(ValueError,match='source patient/partition/feature'):Bundle(data)


def test_rehashed_population_cannot_omit_actual_downstream_TRAIN_patients(tmp_path):
    data=create_demo(tmp_path/'input');mutate_contract(data,lambda v:v['sae_population'].pop())
    with pytest.raises(ValueError,match='incomplete or repeated'):Bundle(data)


@pytest.mark.parametrize('field',['patch_id','patch_index','region_label','global_patient_id'])
def test_rehashed_annotated_probe_cannot_forge_patch_or_patient_identity(tmp_path,field):
    data=create_demo(tmp_path/'input')
    def change(v):v['probes'][0][field]=999 if field in {'patch_index','region_label'} else 'forged_identity'
    mutate_contract(data,change)
    with pytest.raises(ValueError,match='probe|concept'):Bundle(data)


def test_rehashed_probe_cannot_change_actual_annotation_counts(tmp_path):
    data=create_demo(tmp_path/'input');mutate_contract(data,lambda v:v['probes'][0]['actual_annotation'].__setitem__('tumor_pixels',0))
    with pytest.raises(ValueError,match='mask counts'):Bundle(data)


@pytest.mark.parametrize('kind',['feature','annotation','metadata'])
def test_changed_actual_input_bytes_are_rejected_without_silent_repreparation(tmp_path,kind):
    data=create_demo(tmp_path/'input');assets=Bundle(data).protomil
    if kind=='feature':
        with h5py.File(assets.population[0]['path'],'r+') as f:f['features'][0,0]+=1
    elif kind=='annotation':
        with h5py.File(data/'actual_probe_RGB_masks.h5','r+') as f:f['mask'][0,0,0]=0
    else:
        p=data/'slides.csv';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='bytes changed'):Bundle(data)


def test_SAE_source_task_configuration_is_checked_after_prepare(tmp_path):
    data=create_demo(tmp_path/'input');other=data.parent/'additional_SAE_cohort/task.yaml';v=yaml.safe_load(other.read_text());v['task_id']='different_population';other.write_text(yaml.safe_dump(v))
    with pytest.raises(ValueError,match='configuration'):Bundle(data)


def test_actual_rowwise_preprocessing_and_zero_norm_rejection(tmp_path):
    data=create_demo(tmp_path/'input');row=Bundle(data).records('train')[0];p=Path(row['path']);x=actual_features(p)
    import torch
    assert torch.equal(normalized_features(p),x/torch.norm(x,dim=-1,keepdim=True))
    with h5py.File(p,'r+') as f:f['features'][0]=0
    with pytest.raises(ValueError,match='zero-norm'):actual_features(p)


def test_rehashed_invalid_CONCH_weight_identity_is_rejected(tmp_path):
    data=create_demo(tmp_path/'input');mutate_contract(data,lambda v:v['encoding'].__setitem__('checkpoint_sha256','z'*64))
    with pytest.raises(ValueError,match='CONCH'):Bundle(data)

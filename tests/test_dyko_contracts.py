"""Native source semantics/input rejection; no substitute pretrained models."""
import copy
import subprocess
from types import SimpleNamespace

import pytest

from milplus.dyko_assets import git_blob_sha1, native_lock, prepare
from milplus.dyko_data import VISUAL_WEIGHT_SHA256, _visual_metadata, class_mapping, shot_plan
from milplus.io import write_json
from milplus.registry import resource


def bundle(classes=None,dimension=768):
    return SimpleNamespace(task_type='classification',dim=dimension,
        classes=classes or ['LUAD','LUSC'],config={})


def rows():
    return [{'slide_id':f's{index}','global_patient_id':f'p{index//2}',
             'subset':'train' if index < 8 else 'val' if index < 10 else 'test',
             'target':index % 2} for index in range(12)]


def test_native_NSCLC_prompt_order_is_preserved_at_common_class_boundary():
    source_to_common,common_to_source = class_mapping(bundle(),'NSCLC',{'LUSC':'LUSC','LUAD':'LUAD'})
    assert source_to_common == [1,0]
    assert common_to_source == [1,0]
    for native,common in enumerate(source_to_common):
        assert common_to_source[common] == native


@pytest.mark.parametrize('mapping',[
    {'LUSC':'LUAD','LUAD':'LUAD'}, {'LUSC':'LUSC'},
    {'LUSC':'LUSC','LUAD':'LUAD','OTHER':'OTHER'},
])
def test_class_boundary_rejects_ambiguous_or_incomplete_prompt_semantics(mapping):
    with pytest.raises(ValueError,match='exactly one'):
        class_mapping(bundle(),'NSCLC',mapping)


def test_native_CONCH_alignment_cannot_be_obtained_by_matching_shape_only():
    with pytest.raises(ValueError,match='native768D'):
        class_mapping(bundle(dimension=1024),'NSCLC',{'LUSC':'LUSC','LUAD':'LUAD'})


def test_shot_plan_selects_only_TRAIN_and_counts_actual_patients():
    population = rows()
    plan = shot_plan(population,[1,0],99,1024)
    assert {r['slide_id'] for r in plan['selected']} == {f's{i}' for i in range(8)}
    assert plan['actual_slide_count'] == 8 and plan['actual_patient_count'] == 4
    assert plan['class_slide_counts'] == {'0':4,'1':4}
    assert all(r['source_target'] == 1-r['common_target'] for r in plan['selected'])


def test_heldout_labels_cannot_change_the_TRAIN_few_shot_population():
    original = rows();changed = copy.deepcopy(original)
    for row in changed:
        if row['subset'] != 'train':
            row['target'] = 1-row['target']
    assert shot_plan(original,[1,0],2,2026) == shot_plan(changed,[1,0],2,2026)


def test_few_shot_inputs_reject_patient_leakage_before_selection():
    population = rows();population[-1]['global_patient_id'] = population[0]['global_patient_id']
    with pytest.raises(ValueError,match='leaks a patient'):
        shot_plan(population,[1,0],2,42)


def test_duplicate_WSI_identity_is_rejected_before_selection():
    population = rows();population[-1]['slide_id'] = population[0]['slide_id']
    with pytest.raises(ValueError,match='slide identities'):
        shot_plan(population,[1,0],2,42)


def test_original_TITAN_assets_missing_reject_without_producing_manifest(tmp_path):
    actual = tmp_path/'actual';actual.mkdir()
    output = tmp_path/'manifest.json'
    with pytest.raises(FileNotFoundError,match='Original TITAN files missing'):
        prepare(actual,output)
    assert not output.exists()


def test_git_object_fingerprint_matches_actual_git_hash_object(tmp_path):
    actual = tmp_path/'binary';actual.write_bytes(bytes(range(256))+b'\0\r\n')
    expected = subprocess.check_output(['git','hash-object',str(actual)],text=True).strip()
    assert git_blob_sha1(actual) == expected


@pytest.mark.parametrize('change',[
    {'encoder_id':'UNI2'}, {'weights_sha256':'0'*64},
    {'native_feature_space':'generic_768'}, {'extraction_provenance':''},
])
def test_native_visual_contract_rejects_wrong_FM_weights_and_missing_provenance(tmp_path,change):
    value = {'encoder_id':'CONCHv1.5','feature_dim':768,'weights_sha256':VISUAL_WEIGHT_SHA256,
             'native_feature_space':'TITAN_CONCHv15','extraction_provenance':'fixed extraction run'}
    value.update(change)
    path = tmp_path/'visual.json';write_json(path,value)
    with pytest.raises(ValueError,match='shape alone is insufficient'):
        _visual_metadata(bundle(),path)


def test_integrated_DyKo_code_does_not_claim_unexecuted_full_validation():
    method = next(row for row in resource('methods.json') if row['id'] == 'dyko')
    assert method['status'] == 'adapted_under_verification' and len(method['profiles']) == 8
    assert method['formal_admission'] is False and method['validation_blockers']
    lock = native_lock()
    assert lock['original_DyKo_code_does_not_pin_TITAN_revision'] is True
    assert lock['files']['model.safetensors']['bytes'] == 635494504

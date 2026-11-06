"""Native image/text alignment, real prompt identity and leakage refusal."""
import copy

import h5py
import pytest
import yaml

from milplus.conmil_data import prepare
from milplus.conmil_demo import create_conmil_demo, default_text
from milplus.conmil_text import load_text, validate_prompts
from milplus.data import Bundle
from milplus.io import digest_file, digest_object, read_json, write_json


def unbound(data):
    config = yaml.safe_load((data/'task.yaml').read_text())
    config['encoders']['synthetic_conch_dimension'].pop('conmil')
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    return Bundle(data)


def modified_text(tmp_path, transform):
    original = read_json(default_text())
    spec = copy.deepcopy(original['prompts']); transform(spec)
    path = tmp_path/'prompt_spec.json'; write_json(path, spec)
    original.update(prompts=spec, input_spec={'path': str(path), 'sha256': digest_file(path)})
    original['contract'] = digest_object({k: v for k, v in original.items() if k != 'contract'})
    output = tmp_path/'text.json'; write_json(output, original)
    return output


def test_actual_prepared_text_native_dimensions_and_complete_visual_inventory(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    bundle = Bundle(data); inventory = bundle.check(deep=True)
    assert inventory['slides'] == 10 and inventory['patients'] == 10
    assert bundle.conmil.text.positive.shape == (4, 512) and bundle.conmil.text.negative.shape == (2, 512)
    assert bundle.conmil.text.metadata['encoding'] == 'original_CONCH_causal_padding_CLS_project512_L2_normalized'


@pytest.mark.parametrize('key,value', [('classes', ['wrong', 'labels']), ('alignment_policy', 'unrelated_FM'),
    ('synthetic', False), ('class_roles', [{'class': 'class_0', 'role': 'positive'}, {'class': 'class_1', 'role': 'negative'}]),
    ('feature_provenance', '')])
def test_wrong_task_space_semantics_and_roles_are_refused(tmp_path, key, value):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    bundle = unbound(data); meta = read_json(data/'conmil_metadata.json'); meta[key] = value
    path = data/'invalid.json'; write_json(path, meta)
    with pytest.raises(ValueError):
        prepare(bundle, path, data/'new_manifest.json')


def test_normalized_source_pixels_are_not_implicitly_repaired(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    bundle = unbound(data); row = bundle.records('train')[0]
    with h5py.File(row['path'], 'r+') as file:
        file['features'][0] *= 2
    with pytest.raises(ValueError, match='L2 normalization'):
        prepare(bundle, data/'conmil_metadata.json', data/'new_manifest.json')


def test_actual_feature_bytes_and_patient_targets_are_bound(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    bundle = Bundle(data); row = bundle.records('train')[0]
    bundle.conmil.load(row)
    with h5py.File(row['path'], 'r+') as file:
        file['features'][0] = file['features'][0][::-1]
    with pytest.raises(ValueError, match='original normalized patch features'):
        bundle.conmil.load(row)


def test_prompt_refinement_cannot_use_validation_patients(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10); bundle = unbound(data)
    def mutate(spec):
        spec['review'] = {'policy': 'train_only_patch_retrieval', 'selection_subset': 'train',
            'global_patient_ids': [bundle.records('val')[0]['global_patient_id']],
            'retrieval_receipt_sha256': 'a'*64, 'provenance': 'Intentionally invalid held-out review fixture'}
    folder = tmp_path/'invalid_text'; folder.mkdir(); text = modified_text(folder, mutate)
    meta = read_json(data/'conmil_metadata.json'); meta['text_assets'] = str(text)
    path = data/'invalid.json'; write_json(path, meta)
    with pytest.raises(ValueError, match='non-TRAIN patient'):
        prepare(bundle, path, data/'new_manifest.json')


@pytest.mark.parametrize('subset', ['val', 'test', 'all'])
def test_review_metadata_explicitly_refuses_heldout_selection(subset):
    spec = copy.deepcopy(load_text(default_text()).metadata['prompts'])
    spec['review']['selection_subset'] = subset
    with pytest.raises(ValueError, match='TRAIN only'):
        validate_prompts(spec)


def test_changed_actual_prompt_wording_or_tokens_are_rejected(tmp_path):
    metadata = read_json(default_text())
    spec = copy.deepcopy(metadata['prompts']); path = tmp_path/'prompts.json'; write_json(path, spec)
    metadata['input_spec'] = {'path': str(path), 'sha256': digest_file(path)}
    metadata['contract'] = digest_object({k: v for k, v in metadata.items() if k != 'contract'})
    output = tmp_path/'text.json'; write_json(output, metadata)
    load_text(output)
    spec['positive'][0] = 'Different original query wording'; write_json(path, spec)
    with pytest.raises(ValueError, match='wording or review provenance changed'):
        load_text(output)


def test_source_binary_roles_are_distinct_from_explicit_multiclass_targets(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=15, classes=3)
    bundle = Bundle(data)
    assert bundle.conmil.meta['metadata']['class_roles'] == [
        {'class': name, 'role': 'task_class'} for name in bundle.classes]
    from milplus.registry import check_compatibility, require_runnable
    with pytest.raises(ValueError, match='2 classes'):
        check_compatibility(require_runnable('conmil_posneg_source'), bundle)
    check_compatibility(require_runnable('conmil_posneg_multiclass_repaired'), bundle)


def test_manifest_cannot_be_overwritten_or_mutated_during_loading(tmp_path):
    data = create_conmil_demo(tmp_path/'data', patients=10)
    bundle = Bundle(data)
    with pytest.raises(FileExistsError):
        prepare(bundle, data/'conmil_metadata.json', data/'conmil/manifest.json')
    bundle.conmil.path.write_text(bundle.conmil.path.read_text()+'\n')
    with pytest.raises(ValueError, match='manifest bytes changed'):
        bundle.conmil.load(bundle.records('train')[0])

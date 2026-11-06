"""Leakage, semantic alignment and actual-file drift checks for DSPA inputs."""
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.dspamil_data import DSPAInputs, prepare
from milplus.feather_assets import native_lock
from milplus.io import digest_object, read_json, write_json


def fixture(root):
    rng = np.random.default_rng(83)
    slides, index, splits, annotations = [], [], [], []
    for subset in ['train', 'val', 'test']:
        for grade in range(6):
            slide = subset + str(grade)
            path = root / f'{slide}.npy'
            # One actual long bag verifies that preparation/loading keeps its
            # final patch. This is an input-contract test, not model training.
            count = 1025 if slide == 'train5' else 3 + grade
            np.save(path, rng.normal(size=(count, 1024)).astype('float32'))
            slides.append({'slide_id': slide, 'patient_id': slide,
                'global_patient_id': 'global_' + slide, 'label': f'GG{grade}'})
            index.append({'slide_id': slide, 'path': str(path)})
            splits.append({'patient_id': slide, 'subset': subset})
            if subset == 'train':
                annotations.append({'slide_id': slide, 'global_patient_id': 'global_' + slide,
                    'grade_group': grade, 'pattern_3': grade % 2,
                    'pattern_4': (grade // 2) % 2, 'pattern_5': (grade // 4) % 2,
                    'annotation_provenance': 'Synthetic measured-target fixture, not clinical annotation'})
    pd.DataFrame(slides).to_csv(root / 'slides.csv', index=False)
    pd.DataFrame(index).to_csv(root / 'features.csv', index=False)
    pd.DataFrame(splits).to_csv(root / 'split.csv', index=False)
    pd.DataFrame(annotations).to_csv(root / 'patterns.csv', index=False)
    config = {'schema_version': 1, 'task_id': 'synthetic_DSPA_contract_only',
        'task_type': 'ordinal', 'classes': [f'GG{i}' for i in range(6)],
        'defaults': {'fm': 'uni', 'split': 'fixed'}, 'splits': {'fixed': 'split.csv'},
        'encoders': {'uni': {'dimension': 1024, 'index': 'features.csv'}}}
    (root / 'task.yaml').write_text(yaml.safe_dump(config))
    visual = native_lock()['native_encoders']['uni']
    sha = next(iter(visual['files'].values()))['sha256']
    write_json(root / 'visual.json', {'schema_version': 1, 'encoder_id': 'uni',
        'feature_dim': 1024, 'model_id': visual['model_id'], 'revision': visual['revision'],
        'weights_sha256': sha, 'extraction_provenance': 'Synthetic software-contract fixture only'})
    np.save(root / 'pattern_concepts.npy', rng.normal(size=(3, 4, 1024)).astype('float32'))
    np.save(root / 'grade_concepts.npy', rng.normal(size=(6, 1024)).astype('float32'))
    (root / 'text_checkpoint.bin').write_bytes(b'Explicit synthetic file identity; not a pretrained model')
    (root / 'alignment.txt').write_text('Synthetic contract fixture; no original Quilt-to-UNI alignment claimed')
    schema = {'schema_version': 1, 'grade_map': {str(i): f'GG{i}' for i in range(6)},
        'visual_extraction_manifest': 'visual.json', 'training_pattern_table': 'patterns.csv',
        'provenance': 'Synthetic software-only input validation', 'concepts': {
            'pattern_embeddings': 'pattern_concepts.npy', 'grade_embeddings': 'grade_concepts.npy',
            'pattern_texts': [{'pattern': k, 'texts': [f'Synthetic pattern{k} concept{j}' for j in range(4)]}
                              for k in [3, 4, 5]],
            'grade_texts': [{'grade': i, 'text': f'Synthetic grade{i} concept'} for i in range(6)],
            'text_encoder': {'model_id': 'explicit_synthetic_not_pretrained', 'revision': 'fixture1',
                'checkpoint_files': ['text_checkpoint.bin'], 'provenance': 'Not original paper assets'},
            'alignment': {'target_encoder_id': 'uni', 'target_weights_sha256': sha,
                'procedure': 'Synthetic data generated for contract checks only',
                'evidence_file': 'alignment.txt', 'projection_checkpoint': None},
            'provenance': 'No original concepts or clinical labels fabricated'}}
    return Bundle(root), schema


def attach(bundle, path):
    config = yaml.safe_load((bundle.root / 'task.yaml').read_text())
    config['encoders']['uni']['dspamil'] = {'manifest': str(path)}
    (bundle.root / 'task.yaml').write_text(yaml.safe_dump(config))
    return Bundle(bundle.root)


def test_actual_bundle_loads_all1025_patches_and_keeps_teacher_inputs_TRAIN_only(tmp_path):
    bundle, schema = fixture(tmp_path)
    path = prepare(bundle, schema, tmp_path / 'binding.json')
    bound = attach(bundle, path)
    inputs = bound.dspamil
    assert isinstance(inputs, DSPAInputs)
    assert inputs.pattern_concepts.shape == (3, 4, 1024)
    row = next(row for row in bound.records('train') if row['slide_id'] == 'train5')
    actual = inputs.load_features(row)
    assert actual.shape == (1025, 1024)
    assert torch.equal(actual[-1], torch.from_numpy(np.load(row['path']))[-1])
    grade, patterns, concept = inputs.training_supervision(row)
    assert grade.item() == 5 and patterns.tolist() == [1, 0, 1]
    # GG4/5 pattern presence is read from the measured table, never guessed.
    row4 = next(row for row in bound.records('train') if row['slide_id'] == 'train4')
    assert inputs.training_supervision(row4)[1].tolist() == [0, 0, 1]
    assert torch.equal(concept, torch.from_numpy(np.load(tmp_path / 'grade_concepts.npy'))[5])
    assert not concept.requires_grad
    for subset in ['val', 'test']:
        row = bound.records(subset)[0]
        assert len(inputs.load_features(row)) > 0
        with pytest.raises(ValueError, match='held-out'):
            inputs.training_supervision(row)
        with pytest.raises(ValueError, match='row, patient'):
            inputs.training_supervision({**row, 'subset': 'train'})
    value = read_json(path)
    assert value['original_paper_assets_verified'] is value['formal_admission'] is False
    assert len(value['training_patterns']) == 6
    assert all(not name.startswith(('test', 'val')) for name in value['training_patterns'])


@pytest.mark.parametrize('change', ['heldout_row', 'missing_row', 'wrong_patient', 'wrong_grade', 'unknown_binary'])
def test_pattern_supervision_requires_exact_measured_TRAIN_population(tmp_path, change):
    bundle, schema = fixture(tmp_path)
    path = tmp_path / 'patterns.csv'
    table = pd.read_csv(path)
    if change == 'heldout_row':
        table.loc[0, 'slide_id'] = 'test0'
    elif change == 'missing_row':
        table = table.iloc[1:]
    elif change == 'wrong_patient':
        table.loc[0, 'global_patient_id'] = 'global_test0'
    elif change == 'wrong_grade':
        table.loc[0, 'grade_group'] = 4
    else:
        table.loc[0, 'pattern_5'] = -1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='TRAIN|measured binary'):
        prepare(bundle, schema, tmp_path / 'binding.json')
    assert not (tmp_path / 'binding.json').exists()


@pytest.mark.parametrize('change', ['visual', 'alignment', 'pattern_texts', 'grade_order', 'dimension', 'class_order'])
def test_visual_dimension_and_concept_order_do_not_replace_semantic_evidence(tmp_path, change):
    bundle, schema = fixture(tmp_path)
    if change == 'visual':
        value = read_json(tmp_path / 'visual.json')
        value['weights_sha256'] = '0' * 64
        write_json(tmp_path / 'visual.json', value)
    elif change == 'alignment':
        schema['concepts']['alignment']['target_weights_sha256'] = '0' * 64
    elif change == 'pattern_texts':
        schema['concepts']['pattern_texts'][0]['texts'] = ['repeated'] * 4
    elif change == 'grade_order':
        schema['concepts']['grade_texts'].reverse()
    elif change == 'dimension':
        np.save(tmp_path / 'grade_concepts.npy', np.zeros((6, 768), dtype='float32'))
    else:
        schema['grade_map']['4'], schema['grade_map']['5'] = schema['grade_map']['5'], schema['grade_map']['4']
    with pytest.raises(ValueError):
        prepare(bundle, schema, tmp_path / 'binding.json')
    assert not (tmp_path / 'binding.json').exists()


@pytest.mark.parametrize('change', ['global_patient', 'identical_feature_bags'])
def test_patient_identity_and_actual_duplicate_content_prevent_partition_leakage(tmp_path, change):
    bundle, schema = fixture(tmp_path)
    if change == 'global_patient':
        bundle.rows.loc[bundle.rows.slide_id == 'test0', 'global_patient_id'] = 'global_train0'
    else:
        (tmp_path / 'test0.npy').write_bytes((tmp_path / 'train0.npy').read_bytes())
    with pytest.raises(ValueError, match='patient|byte-identical'):
        prepare(bundle, schema, tmp_path / 'binding.json')


@pytest.mark.parametrize('filename', ['train5.npy', 'pattern_concepts.npy', 'patterns.csv', 'alignment.txt'])
def test_deep_verification_detects_same_size_same_mtime_actual_file_mutation(tmp_path, filename):
    bundle, schema = fixture(tmp_path)
    bound = attach(bundle, prepare(bundle, schema, tmp_path / 'binding.json'))
    path = tmp_path / filename
    original = path.read_bytes()
    stat = path.stat()
    changed = bytearray(original)
    changed[-1] ^= 1
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='changed'):
        bound.dspamil.verify(deep=True)


def test_manifest_cannot_relabel_a_reference_as_original_assets_or_formal_admission(tmp_path):
    bundle, schema = fixture(tmp_path)
    path = prepare(bundle, schema, tmp_path / 'binding.json')
    for key in ['original_paper_assets_verified', 'full_method_validation', 'formal_admission']:
        value = read_json(path)
        value.pop('contract')
        value[key] = True
        value['contract'] = digest_object(value)
        write_json(path, value)
        bundle.encoder['dspamil'] = {'manifest': str(path)}
        with pytest.raises(ValueError, match='provenance'):
            DSPAInputs(bundle)
        value[key] = False
        value.pop('contract')
        value['contract'] = digest_object(value)
        write_json(path, value)


def test_public_prepare_command_binds_the_actual_task_and_preserves_pending_method(tmp_path, capsys):
    from milplus.cli import main
    from milplus.registry import resource
    bundle, schema = fixture(tmp_path)
    schema_path = tmp_path / 'schema.yaml'
    schema_path.write_text(yaml.safe_dump(schema))
    output = tmp_path / 'cli_binding.json'
    assert main(['prepare-dspamil', '--data', str(tmp_path), '--schema', str(schema_path),
                 '--output', str(output)]) == 0
    assert read_json(output)['identity']['task_id'] == bundle.config['task_id']
    assert read_json(output)['formal_admission'] is False
    item = next(row for row in resource('methods.json') if row['id'] == 'dspamil')
    assert item['status'] == 'adapted_under_verification' and len(item['profiles']) == 2
    assert item['DSPA_U_implemented'] is False and item['formal_admission'] is False
    assert 'manifest' in capsys.readouterr().out


def test_model_cannot_mutate_the_bound_frozen_pattern_or_grade_concept_cache(tmp_path):
    bundle, schema = fixture(tmp_path)
    bound = attach(bundle, prepare(bundle, schema, tmp_path / 'binding.json'))
    inputs = bound.dspamil
    original = inputs.pattern_concepts
    changed = inputs.pattern_concepts
    changed.zero_()
    assert torch.equal(inputs.pattern_concepts, original) and torch.count_nonzero(original)
    row = bound.records('train')[0]
    expected = inputs.training_supervision(row)[2]
    changed = inputs.training_supervision(row)[2]
    changed.fill_(99)
    assert torch.equal(inputs.training_supervision(row)[2], expected)


def test_actual_bag_change_between_hash_check_and_loader_is_rejected(tmp_path):
    bundle, schema = fixture(tmp_path)
    bound = attach(bundle, prepare(bundle, schema, tmp_path / 'binding.json'))
    original = bound.load
    def changing_loader(row):
        features, coords = original(row)
        path = tmp_path / (row['slide_id'] + '.npy')
        np.save(path, features.numpy() + 1)
        return features, coords
    bound.load = changing_loader
    with pytest.raises(ValueError, match='bag changed'):
        bound.dspamil.load_features(bound.records('train')[0])

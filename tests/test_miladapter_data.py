"""Original text bytes, class order, support leakage and immutable bags."""
import json

import numpy as np
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.miladapter import MILAdapter
from milplus.miladapter_data import import_author_prototypes, prepare, text_assets
from milplus.miladapter_demo import create_miladapter_demo
from milplus.miladapter_source import fewshot_indices


@pytest.mark.parametrize('encoder', ['CONCH', 'KEEP'])
@pytest.mark.parametrize('project,classes', [('BRCA', ['IDC', 'ILC']), ('NSCLC', ['LUAD', 'LUSC']),
    ('RCC', ['CHRCC', 'CCRCC', 'PRCC'])])
def test_six_actual_source_matrices_are_imported_without_text_regeneration(tmp_path, encoder, project, classes):
    path = import_author_prototypes(tmp_path/'text', project, encoder)
    meta, value, files = text_assets(path)
    assert meta['classes'] == classes and value.shape == (512 if encoder == 'CONCH' else 768, len(classes))
    assert value.dtype == np.float32 and len(files) == 2
    assert meta['prompt_provenance_status'] == 'author_import_generation_details_undisclosed'
    with pytest.raises(FileExistsError):
        import_author_prototypes(tmp_path/'text', project, encoder)
    value[0, 0] += .1; np.save(meta['path'], value)
    with pytest.raises(ValueError, match='bytes changed'):
        text_assets(path)


@pytest.mark.parametrize('mutation', ['class_order', 'synthetic', 'support_policy', 'unrelated_fm', 'provenance', 'text_bytes'])
def test_preparation_rejects_incompatible_or_untraceable_inputs(tmp_path, mutation):
    root = create_miladapter_demo(tmp_path/'data')
    # Prepare another immutable asset version from the unbound input bundle.
    config = yaml.safe_load((root/'task.yaml').read_text())
    del config['encoders']['synthetic_native512']['miladapter']
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    meta = json.loads((root/'miladapter_metadata.json').read_text())
    if mutation == 'class_order':
        meta['classes'].reverse()
    elif mutation == 'synthetic':
        meta['synthetic'] = False
    elif mutation == 'support_policy':
        meta['support_policy'] = 'select_from_test'
    elif mutation == 'unrelated_fm':
        meta['alignment_policy'] = 'arbitrary_FM'
    elif mutation == 'provenance':
        meta['feature_provenance'] = ''
    else:
        p = root/'author_text/prototypes.npy'; x = np.load(p); x[0, 0] += .2; np.save(p, x)
    (root/'modified.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        prepare(Bundle(root), root/'modified.json', root/'another/manifest.json')


def test_support_is_exact_classwise_train_only_with_source_numpy_rng(tmp_path):
    root = create_miladapter_demo(tmp_path/'data', patients=30)
    bundle = Bundle(root); pool = bundle.records('train')
    labels = np.array([r['target'] for r in pool])
    np.random.seed(802); expected = []
    for cls in range(2):
        expected.extend(np.random.choice(np.flatnonzero(labels == cls), 4, replace=False))
    expected_state = np.random.get_state()
    actual = fewshot_indices(labels, 4, 802)
    assert actual == expected
    assert np.array_equal(np.random.get_state()[1], expected_state[1])
    model = MILAdapter(bundle, k_shots=4, sampling_seed=802)
    assert model.support_ids == [pool[i]['slide_id'] for i in expected]
    support_patients = {r['global_patient_id'] for r in model.support}
    assert not support_patients & {r['global_patient_id'] for r in bundle.records('val')+bundle.records('test')}
    assert all(sum(r['target'] == cls for r in model.support) == 4 for cls in range(2))
    with pytest.raises(ValueError, match='at least k'):
        MILAdapter(bundle, k_shots=16)


def test_unregistered_multi_slide_support_patient_is_refused(tmp_path):
    root = create_miladapter_demo(tmp_path/'data')
    import pandas as pd
    slides = pd.read_csv(root/'slides.csv')
    first, second = slides.iloc[0].patient_id, slides.iloc[2].patient_id
    slides['global_patient_id'] = slides.patient_id
    slides.loc[slides.patient_id == second, 'global_patient_id'] = first
    slides.to_csv(root/'slides.csv', index=False)
    with pytest.raises(ValueError, match='one eligible TRAIN slide'):
        Bundle(root)


def test_feature_bytes_and_cached_manifest_are_checked_at_actual_use(tmp_path):
    root = create_miladapter_demo(tmp_path/'data'); bundle = Bundle(root)
    row = bundle.records('train')[0]
    before = bundle.miladapter.load(row).clone()
    value = np.load(row['path']); value[0, 0] += .25; np.save(row['path'], value)
    with pytest.raises(ValueError, match='bag bytes'):
        bundle.miladapter.load(row)
    assert not torch.equal(before, torch.from_numpy(value))


def test_missing_assets_and_unsupported_k_are_not_silent_fallbacks(tmp_path):
    root = create_miladapter_demo(tmp_path/'data'); bundle = Bundle(root)
    for k in [0, 1, 3, True]:
        with pytest.raises(ValueError):
            MILAdapter(bundle, k_shots=k)
    with pytest.raises(ValueError):
        import_author_prototypes(tmp_path/'titan', 'BRCA', 'TITAN')

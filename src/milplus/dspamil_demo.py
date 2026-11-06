"""Explicitly synthetic full-shape DSPA inputs; never original expert assets."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .dspamil_data import prepare
from .feather_assets import native_lock
from .io import write_json


def create_demo(output, *, patches=1025):
    if type(patches) is not int or patches < 1:
        raise ValueError('Synthetic DSPA bags require a positive actual instance count')
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(764)
    slides, index, partitions, annotations = [], [], [], []
    # Explicit fictional annotations, not a rule used to infer real GG4/GG5.
    fictional_patterns = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]]
    for subset in ['train', 'val', 'test']:
        for grade in range(6):
            slide = f'synthetic_{subset}_GG{grade}'
            patient = 'fictional_patient_' + slide
            path = root / (slide + '.npy')
            np.save(path, rng.normal(0, .1, size=(patches, 1024)).astype('float32'))
            slides.append({'slide_id': slide, 'patient_id': patient,
                'global_patient_id': patient, 'label': f'GG{grade}'})
            index.append({'slide_id': slide, 'path': str(path)})
            partitions.append({'patient_id': patient, 'subset': subset})
            if subset == 'train':
                annotations.append({'slide_id': slide, 'global_patient_id': patient,
                    'grade_group': grade, **{f'pattern_{k}': fictional_patterns[grade][k - 3] for k in [3, 4, 5]},
                    'annotation_provenance': 'Explicit fictional software-control targets; no patient annotation or clinical inference'})
    pd.DataFrame(slides).to_csv(root / 'slides.csv', index=False)
    pd.DataFrame(index).to_csv(root / 'features.csv', index=False)
    pd.DataFrame(partitions).to_csv(root / 'split.csv', index=False)
    pd.DataFrame(annotations).to_csv(root / 'TRAIN_patterns.csv', index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_DSPA_full_graph_software_control',
        'synthetic': True, 'synthetic_scope': 'random_full_shape_features_concepts_and_fictional_labels_NOT_original_assets',
        'task_type': 'ordinal', 'classes': [f'GG{i}' for i in range(6)],
        'training_unit': 'slide', 'evaluation_unit': 'slide',
        'defaults': {'fm': 'synthetic_native_shape_UNI', 'split': 'fictional'},
        'splits': {'fictional': 'split.csv'},
        'encoders': {'synthetic_native_shape_UNI': {'dimension': 1024, 'index': 'features.csv'}}}
    (root / 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    native = native_lock()['native_encoders']['uni']
    sha = next(iter(native['files'].values()))['sha256']
    write_json(root / 'visual.json', {'schema_version': 1, 'encoder_id': 'uni', 'feature_dim': 1024,
        'model_id': native['model_id'], 'revision': native['revision'], 'weights_sha256': sha,
        'extraction_provenance': 'SYNTHETIC shape-only Gaussian fixture: no UNI model, extraction or original visual weights were executed'})
    np.save(root / 'pattern_concepts.npy', rng.normal(0, .1, size=(3, 4, 1024)).astype('float32'))
    np.save(root / 'grade_concepts.npy', rng.normal(0, .1, size=(6, 1024)).astype('float32'))
    (root / 'SYNTHETIC_NOT_A_TEXT_MODEL.bin').write_bytes(b'Explicit synthetic file identity; no Quilt or pretrained text encoder')
    (root / 'SYNTHETIC_NOT_AN_ALIGNMENT.txt').write_text('Shape-only Gaussian vectors; no semantic text-to-UNI alignment asserted\n')
    schema = {'schema_version': 1, 'grade_map': {str(i): f'GG{i}' for i in range(6)},
        'visual_extraction_manifest': 'visual.json', 'training_pattern_table': 'TRAIN_patterns.csv',
        'provenance': 'Synthetic complete-graph software validation; not a clinical/reference reproduction dataset',
        'concepts': {'pattern_embeddings': 'pattern_concepts.npy', 'grade_embeddings': 'grade_concepts.npy',
            'pattern_texts': [{'pattern': k, 'texts': [f'SYNTHETIC_NOT_CLINICAL_pattern{k}_concept{j}' for j in range(4)]}
                              for k in [3, 4, 5]],
            'grade_texts': [{'grade': k, 'text': f'SYNTHETIC_NOT_CLINICAL_grade{k}'} for k in range(6)],
            'text_encoder': {'model_id': 'SYNTHETIC_NOT_QUILT', 'revision': 'fictional_control1',
                'checkpoint_files': ['SYNTHETIC_NOT_A_TEXT_MODEL.bin'], 'provenance': 'No text encoding or pretrained model'},
            'alignment': {'target_encoder_id': 'uni', 'target_weights_sha256': sha,
                'procedure': 'Random shape-only control; no semantic alignment claimed',
                'evidence_file': 'SYNTHETIC_NOT_AN_ALIGNMENT.txt', 'projection_checkpoint': None},
            'provenance': 'Random synthetic tensors and placeholder control texts, not original paper expert concepts'}}
    (root / 'input_schema.yaml').write_text(yaml.safe_dump(schema, sort_keys=False))
    bundle = Bundle(root)
    binding = prepare(bundle, schema, root / 'dspamil_binding.json')
    task['encoders']['synthetic_native_shape_UNI']['dspamil'] = {'manifest': str(binding)}
    (root / 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    Bundle(root).dspamil.verify(deep=True)
    write_json(root / 'synthetic_control_provenance.json', {'synthetic': True, 'actual_patches_per_bag': patches,
        'feature_dimension': 1024, 'slides': 18, 'TRAIN_slides': 6, 'VAL_slides': 6, 'TEST_slides': 6,
        'original_expert_assets_verified': False, 'real_WSI_experiment': False,
        'paper_reproduction': False, 'formal_admission': False})
    return root

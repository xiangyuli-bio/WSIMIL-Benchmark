"""Explicit synthetic512D bags with unmodified actual author text prototypes."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .io import write_json
from .miladapter_data import KIND, ENCODER_DIMENSIONS, PROJECT_CLASSES, import_author_prototypes, prepare


def create_miladapter_demo(output, patients=20, project='BRCA', encoder='CONCH', patches=33):
    if project not in PROJECT_CLASSES:
        raise ValueError('MIL-Adapter demo supports BRCA/NSCLC/RCC source prototype columns')
    if encoder not in ENCODER_DIMENSIONS:
        raise ValueError('MIL-Adapter needs original CONCH or KEEP prototypes')
    classes = PROJECT_CLASSES[project]; dimension = ENCODER_DIMENSIONS[encoder]
    if (type(patients) is not int or patients < 10*len(classes)
            or type(patches) is not int or patches < 6):
        raise ValueError('MIL-Adapter demo needs at least10patients/class and6patches/slide')
    root = Path(output); root.mkdir(parents=True, exist_ok=False); (root/'features').mkdir()
    text_path = import_author_prototypes(root/'author_text', project, encoder)
    slides, features, splits = [], [], []
    for i in range(patients):
        slide, patient = f'miladapter_slide_{i:04d}', f'miladapter_patient_{i:04d}'
        target = i % len(classes); group = (i//len(classes)) % 10
        subset = 'val' if group == 8 else 'test' if group == 9 else 'train'
        value = np.random.default_rng(8000+i).normal(size=(patches, dimension)).astype(np.float32)
        value[:, target] += 4.
        path = root/'features'/f'{slide}.npy'; np.save(path, value)
        slides.append({'slide_id': slide, 'patient_id': patient, 'label': classes[target]})
        features.append({'slide_id': slide, 'path': str(path)})
        splits.append({'patient_id': patient, 'subset': subset})
    for name, rows in [('slides.csv', slides), ('features.csv', features), ('split.csv', splits)]:
        pd.DataFrame(rows).to_csv(root/name, index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_miladapter_native512', 'task_type': 'classification',
        'classes': classes, 'synthetic': True, 'training_unit': 'slide', 'evaluation_unit': 'slide',
        'label_provenance': 'Synthetic only; actual source text matrix, no clinical WSI or natural image/text alignment claim',
        'slides': 'slides.csv', 'splits': {'fold0': 'split.csv'},
        'defaults': {'fm': 'synthetic_native512', 'split': 'fold0'},
        'encoders': {'synthetic_native512': {'dimension': dimension, 'index': 'features.csv',
            'weights': 'Synthetic visual bags; original public text prototypes bound separately'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'classes': classes,
        'text_assets': str(text_path), 'alignment_policy': 'synthetic_native_dimension_fixture',
        'support_policy': 'source_classwise_sampling_from_train_pool_only_patient_disjoint_extension',
        'feature_preprocessing': 'Synthetic native-width Gaussian bags, no patch truncation or silent normalization',
        'feature_provenance': 'Explicit software fixture, not actual CONCH/KEEP extraction',
        'task_semantics': 'Synthetic class discrimination, not real clinical task validation'}
    write_json(root/'miladapter_metadata.json', metadata)
    prepare(Bundle(root), root/'miladapter_metadata.json', root/'miladapter/manifest.json')
    task['encoders']['synthetic_native512']['miladapter'] = {'manifest': 'miladapter/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

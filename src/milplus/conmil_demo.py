"""Synthetic native-width visual bags with actual frozen CONCH text queries."""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .conmil_data import KIND, prepare
from .conmil_text import load_text
from .data import Bundle
from .io import write_json


def default_text():
    return Path(os.environ.get('MILPLUS_CONMIL_TEXT',
        Path.home()/'.cache/milplus/assets/conmil/text.json')).expanduser().resolve()


def create_conmil_demo(output, patients=20, classes=2, patches=33, text_assets=None):
    if (type(classes) is not int or classes < 2 or type(patients) is not int or patients < 5*classes
            or type(patches) is not int or patches < 1):
        raise ValueError('ConMIL demo needs native-width bags and at least5patients/class')
    text_path = Path(text_assets or default_text()).expanduser().resolve()
    load_text(text_path)
    root = Path(output); root.mkdir(parents=True, exist_ok=False); (root/'features').mkdir()
    slides, features, splits = [], [], []
    for index in range(patients):
        slide, patient = f'conmil_slide_{index:04d}', f'conmil_patient_{index:04d}'
        target = index % classes
        subset = ['train', 'train', 'train', 'val', 'test'][(index//classes) % 5]
        random = np.random.default_rng(7900+index)
        value = random.normal(size=(patches, 512)).astype(np.float32)
        value[:, target] += 4.
        value /= np.linalg.norm(value, axis=1, keepdims=True)
        path = root/'features'/f'{slide}.h5'
        with h5py.File(path, 'w') as archive:
            archive['features'] = value
        slides.append({'slide_id': slide, 'patient_id': patient, 'label': f'class_{target}'})
        features.append({'slide_id': slide, 'path': str(path)})
        splits.append({'patient_id': patient, 'subset': subset})
    for name, rows in [('slides.csv', slides), ('features.csv', features), ('split.csv', splits)]:
        pd.DataFrame(rows).to_csv(root/name, index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_conmil_native512', 'task_type': 'classification',
        'classes': [f'class_{n}' for n in range(classes)], 'synthetic': True,
        'label_provenance': 'Synthetic only; no clinical WSI, natural visual/text alignment or FM extraction claim',
        'evaluation_unit': 'slide', 'training_unit': 'slide', 'slides': 'slides.csv',
        'defaults': {'fm': 'synthetic_conch_dimension', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'synthetic_conch_dimension': {'dimension': 512, 'index': 'features.csv',
            'weights': 'synthetic visual features, actual native CONCH text separately bound',
            'normalization': 'L2', 'preprocessing': 'Explicit synthetic normalized Gaussian vectors'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'classes': task['classes'],
        'text_assets': str(text_path), 'alignment_policy': 'synthetic_native_dimension_fixture',
        'feature_preprocessing': 'Synthetic512D Gaussian features normalized per patch; not real CONCH output',
        'feature_provenance': 'Explicit software fixture with real pretrained-text encoding; no natural pathology image/text alignment claim',
        'task_semantics': 'Synthetic class discrimination for mechanical verification only',
        'class_roles': [{'class': name, 'role': role} for name, role in zip(task['classes'],
            ['negative', 'positive'] if classes == 2 else ['task_class']*classes)]}
    write_json(root/'conmil_metadata.json', metadata)
    prepare(Bundle(root), root/'conmil_metadata.json', root/'conmil/manifest.json')
    task['encoders']['synthetic_conch_dimension']['conmil'] = {'manifest': 'conmil/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

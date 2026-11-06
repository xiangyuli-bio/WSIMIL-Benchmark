"""Actual synthetic spatial-feature bags; never clinical WSI evidence."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .io import digest_file


def create_demo(output, patches=1025, dimension=384, survival=False, alibi_path=None):
    if type(patches) is not int or patches < 1 or type(dimension) is not int or dimension < 1:
        raise ValueError('Declare positive actual synthetic patch counts and feature dimensions')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'features').mkdir()
    rng = np.random.default_rng(104)
    side = int(np.ceil(np.sqrt(patches)))
    coords = np.asarray([(x, y) for x in range(side) for y in range(side)][:patches], dtype=np.int64) * 896
    slides, features, split = [], [], []
    for group, subset in enumerate(['train', 'val', 'test']):
        for i in range(4):
            patient = f'synthetic_{subset}_{i}'
            split.append({'patient_id': patient, 'subset': subset})
            for j in range(1 + int(survival and i % 2 == 0)):
                slide = patient + f'_slide{j}'
                path = 'features/' + slide + '.h5'
                value = rng.normal(0, 1, (patches, dimension)).astype('float32')
                value[:, i % 2] += .15
                with h5py.File(root/path, 'w') as file:
                    file['features'] = value
                    file['coords'] = coords
                    file['patch_ids'] = np.asarray([f'{slide}_x{x}_y{y}'.encode() for x,y in coords])
                slides.append({'slide_id': slide, 'patient_id': patient, 'global_patient_id': patient,
                    'label': f'class_{i%2}', 'time': float(10 + 3*i + group), 'event': int(i != 3),
                    'width_level0': side*896, 'height_level0': side*896})
                features.append({'slide_id': slide, 'path': path})
    pd.DataFrame(slides).to_csv(root/'slides.csv', index=False)
    pd.DataFrame(features).to_csv(root/'features.csv', index=False)
    pd.DataFrame(split).to_csv(root/'split.csv', index=False)
    physical = {'coordinate_frame': 'level0_xy', 'patch_step_level0': 896, 'grid_origin_xy': [0, 0]}
    if alibi_path is not None:
        asset = Path(alibi_path).resolve()
        physical.update(v1_alibi_path=str(asset), v1_alibi_sha256=digest_file(asset))
    config = {'schema_version': 1, 'task_id': 'synthetic_longmil_survival' if survival else 'synthetic_longmil_classification',
        'synthetic': True, 'task_type': 'survival' if survival else 'classification',
        'classes': [] if survival else ['class_0', 'class_1'],
        'evaluation_unit': 'patient', 'training_unit': 'patient' if survival else 'slide',
        'label_provenance': 'synthetic deterministic fixture; not real WSI or a paper-score experiment',
        'slides': 'slides.csv', 'defaults': {'fm': 'synthetic_fm', 'split': 'fold0'},
        'splits': {'fold0': 'split.csv'}, 'encoders': {'synthetic_fm': {'dimension': dimension,
        'index': 'features.csv', 'weights': 'synthetic_no_FM_checkpoint', 'longmil': physical,
        'preprocessing': 'actual generated synthetic2D coordinates and aligned ordered random feature rows'}}}
    if survival:
        config['time_unit'] = 'months'
    (root/'task.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    return root

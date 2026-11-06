"""Synthetic SMMILe task bundles for reproducible software checks, never clinical evidence."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .smmile_data import prepare_superpixels


def create_smmile_demo(output, variant='multiclass', patients=30, dimension=16, classes=None):
    if variant not in {'binary', 'multiclass', 'multilabel'}:
        raise ValueError('Unknown synthetic SMMILe task variant')
    classes = (2 if variant == 'binary' else 3) if classes is None else classes
    if type(classes) is not int or classes < 2 or variant == 'binary' and classes != 2:
        raise ValueError('Invalid synthetic SMMILe class count')
    if variant == 'multilabel' and (classes != 3 or patients < 30):
        raise ValueError('The synthetic multilabel fixture uses three endpoints and at least30 patients')
    root = create_demo(output, patients=patients, dimension=dimension, classes=2 if variant != 'multiclass' else classes)
    task = yaml.safe_load((root/'task.yaml').read_text())
    slides = pd.read_csv(root/'slides.csv')
    if variant == 'multilabel':
        columns = [f'endpoint_{i}' for i in range(3)]
        for i, name in enumerate(columns):
            slides[name] = slides.patient_id.map(lambda p, bit=i: (int(p.split('_')[-1]) >> bit) & 1)
        task.update(task_type='multilabel', task_id='synthetic_smmile_multilabel', classes=columns)
        task['multilabel'] = {'columns': columns, 'label_provenance': 'synthetic bit-pattern endpoints only',
                              'empty_target_semantics': 'explicitly negative for every synthetic endpoint'}
    task['training_unit'] = 'slide'
    task['smmile_variant'] = variant
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    slides.to_csv(root/'slides.csv', index=False)
    rng = np.random.default_rng(221)
    for row in Bundle(root).rows.to_dict('records'):
        with h5py.File(root/row['path'], 'r+') as h:
            count = len(h['features'])
            coords = np.array([[256*x+512,256*y+768] for x in range(8) for y in range(7)
                               if (x,y) not in {(2,2),(4,3)}], dtype=np.int64)[:count]
            rng.shuffle(coords)
            h['coords'][:] = coords
    prepare_superpixels(Bundle(root), root/'smmile_assets', 256, provenance='synthetic physical grid fixture; not real WSI')
    task['encoders']['toy']['smmile'] = {'manifest': 'smmile_assets/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return Path(root)

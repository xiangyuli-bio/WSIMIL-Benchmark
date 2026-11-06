"""Synthetic-only PAMIL annotations; never create annotations for clinical bags."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .pamil_data import prepare_annotations


def create_pamil_demo(output, multilabel=False, dimension=16, patients=30):
    root = create_demo(output, patients=patients, dimension=dimension, classes=2 if multilabel else 3)
    task = yaml.safe_load((root/'task.yaml').read_text())
    if multilabel:
        slides = pd.read_csv(root/'slides.csv')
        columns = ['endpoint_'+str(i) for i in range(3)]
        for i, column in enumerate(columns):
            slides[column] = slides.patient_id.map(lambda p, bit=i: (int(p.split('_')[-1]) >> bit) & 1)
        slides.to_csv(root/'slides.csv', index=False)
        task.update(task_type='multilabel', classes=columns)
        task['multilabel'] = {'columns': columns, 'label_provenance': 'synthetic bit-pattern only',
                             'empty_target_semantics': 'explicit synthetic all-negative endpoints'}
    task['task_id'] = 'synthetic_pamil'
    task['training_unit'] = 'slide'
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    bundle = Bundle(root)
    (root/'instance_annotations').mkdir()
    records = []
    for row in bundle.records('train'):
        count = len(bundle.load(row)[0])
        classes = np.flatnonzero(row['targets']) if multilabel else [row['target']]
        labels = np.resize(classes, count).astype(np.int64) if len(classes) else np.full(count, -1, dtype=np.int64)
        path = Path('instance_annotations')/(row['slide_id']+'.npy')
        np.save(root/path, labels)
        records.append({'slide_id': row['slide_id'], 'path': str(path)})
    pd.DataFrame(records).to_csv(root/'annotations.csv', index=False)
    prepare_annotations(bundle, root/'annotations.csv', root/'pamil_annotations.json',
                        'SYNTHETIC-only row labels for software checks; no clinical interpretation')
    task['encoders']['toy']['pamil'] = {'manifest': 'pamil_annotations.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

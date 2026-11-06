"""Synthetic real-coordinate rectangular bags for 2DMamba software verification."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .demo import create_demo


def create_mamba2d_demo(output, dimension=16, classes=3, survival=False):
    root = create_demo(output, patients=30, dimension=dimension, classes=classes, survival=survival)
    slides, features = [pd.read_csv(root/name) for name in ['slides.csv', 'features.csv']]
    if survival:
        slides = slides.drop_duplicates('patient_id')
        features = features[features.slide_id.isin(slides.slide_id)]
        slides.to_csv(root/'slides.csv', index=False);features.to_csv(root/'features.csv', index=False)
    coords = np.asarray([(x*512, y*512) for y in range(4) for x in range(6)
                         if (x+2*y) % 3 != 0], dtype=np.int64)
    rng = np.random.default_rng(63)
    labels = dict(zip(slides.slide_id, slides.label))
    for row in features.to_dict('records'):
        x = rng.normal(size=(len(coords), dimension)).astype(np.float32)
        x[:, int(labels[row['slide_id']].split('_')[-1])] += 1.25
        with h5py.File(root/row['path'], 'w') as f:f['features'] = x;f['coords'] = coords
    path = root/'task.yaml';task = yaml.safe_load(path.read_text())
    task['encoders']['toy']['spatial'] = {'unit': 'level0_pixels', 'origin': 'top_left',
        'patch_size': 512, 'provenance': 'synthetic sparse 6x4 physical grid; not inferred WSI topology'}
    task['encoders']['toy']['mamba2d'] = {'max_coordinates': [3071, 2047],
        'provenance': 'synthetic fixed complete canvas including known non-tissue grid cells'}
    if survival:task['representative_slide_rule'] = 'synthetic first slide per generated patient, fixed before any training'
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

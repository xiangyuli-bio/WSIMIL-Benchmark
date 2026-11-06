"""Synthetic true-patient four-morphology bags; no tissue/clinical claim."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .demo import create_demo


def create_amiml_demo(output, patients=20, dimension=256, classes=2, per_cluster=104):
    if type(per_cluster) is not int or per_cluster < 11:
        raise ValueError('AMIML synthetic clusters need >10 actual patches')
    root = Path(create_demo(output, patients=patients, dimension=dimension, classes=classes))
    path = root/'task.yaml'; cfg = yaml.safe_load(path.read_text())
    cfg.update(training_unit='patient', evaluation_unit='patient')
    cfg['encoders']['toy']['preprocessing'] = 'Synthetic four Gaussian morphologies, actual stored rows only'
    labels = pd.read_csv(root/'slides.csv').set_index('slide_id').label.to_dict()
    rng = np.random.default_rng(2064)
    centers = np.zeros((4, dimension), dtype=np.float32)
    centers[:, :4] = np.eye(4, dtype=np.float32)*20
    for row in pd.read_csv(root/'features.csv').to_dict('records'):
        patch = np.repeat(centers, per_cluster, axis=0)+rng.normal(0, .05, (4*per_cluster, dimension)).astype(np.float32)
        # Shape signal does not alter the cluster-defining leading coordinates.
        patch[:, 4+cfg['classes'].index(labels[row['slide_id']])] += .5
        with h5py.File(root/row['path'], 'w') as f: f['features'] = patch
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root

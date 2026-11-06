"""Explicitly synthetic stable-order feature bags for dynamic-policy PAMIL."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .demo import create_demo


def create_pamil_policy_demo(output, patients=20, dimension=1024, classes=2, rows_per_slide=None):
    if rows_per_slide is not None and (type(rows_per_slide) is not int or rows_per_slide < 1):
        raise ValueError('PAMIL fixture row count must be positive')
    root = Path(create_demo(output, patients=patients, dimension=dimension, classes=classes))
    path = root/'task.yaml'; cfg = yaml.safe_load(path.read_text())
    cfg['evaluation_unit'] = 'slide'
    cfg['encoders']['toy']['row_order_provenance'] = 'Fixed synthetic feature-index order, content-bound by Bundle inventory; no real spatial claim'
    if rows_per_slide is not None:
        slides = pd.read_csv(root/'slides.csv'); labels = slides.set_index('slide_id').label.to_dict()
        rng = np.random.default_rng(2062)
        for row in pd.read_csv(root/'features.csv').to_dict('records'):
            x = rng.normal(0, 1, (rows_per_slide, dimension)).astype(np.float32)
            x[:, cfg['classes'].index(labels[row['slide_id']])] += 1.25
            with h5py.File(root/row['path'], 'w') as f:
                f['features'] = x
                f['coords'] = np.column_stack([np.arange(rows_per_slide)*256, np.zeros(rows_per_slide)]).astype(np.int64)
        slides['width_level0'] = rows_per_slide*256
        slides.to_csv(root/'slides.csv', index=False)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root

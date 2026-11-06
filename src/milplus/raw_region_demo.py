"""Explicit synthetic tiled WSI/real-coordinate fixtures for raw two-stage MIL."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .raw_regions import RESNET34_V1_SHA256


def create_raw_region_demo(output, patients_per_class=4, initial_weights=None, mult=1.):
    import tifffile
    if type(patients_per_class) is not int or patients_per_class < 3:
        raise ValueError('Raw-region demo requires at least three independent patients per class')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(31)
    slides, index, split = [], [], []
    for subset, count in [('train', patients_per_class-2), ('val', 1), ('test', 1)]:
        for patient in range(count):
            for label in range(2):
                sid = f'{subset}_{patient}_{label}'
                pixels = rng.integers(0, 35, (672, 896, 3), dtype=np.uint8)
                yy, xx = np.indices(pixels.shape[:2])
                base = np.array([140, 95, 150] if label else [185, 145, 175], dtype=np.uint8)
                pixels += base + (((xx//16+yy//21) % 2)*20).astype(np.uint8)[..., None]
                tifffile.imwrite(root/f'{sid}.tiff', pixels, tile=(224, 224), photometric='rgb', metadata=None)
                np.save(root/f'{sid}.npy', np.array([(x*224, y*224) for y in range(3) for x in range(4)], dtype=np.int64))
                slides.append({'slide_id': sid, 'patient_id': sid, 'label': ['negative', 'positive'][label]})
                index.append({'slide_id': sid, 'path': sid+'.tiff', 'coords_path': sid+'.npy'})
                split.append({'patient_id': sid, 'subset': subset})
    for rows, name in [(slides, 'slides'), (index, 'images'), (split, 'split')]:
        pd.DataFrame(rows).to_csv(root/f'{name}.csv', index=False)
    backbone = {'architecture': 'resnet34', 'initialization': 'explicit_random'}
    if initial_weights:
        backbone.update(initialization='imagenet1k_v1', path=str(Path(initial_weights).resolve()), sha256=RESNET34_V1_SHA256)
    task = {'schema_version': 1, 'task_id': 'synthetic_raw_regions', 'synthetic': True, 'task_type': 'classification',
            'classes': ['negative', 'positive'], 'training_unit': 'slide', 'evaluation_unit': 'slide', 'slides': 'slides.csv',
            'splits': {'fold0': 'split.csv'}, 'defaults': {'fm': 'raw_resnet34', 'split': 'fold0'},
            'encoders': {'raw_resnet34': {'dimension': 3, 'index': 'images.csv', 'raw_regions': {
                'kind': 'openslide', 'coordinate_unit': 'level0_pixels', 'coordinate_order': 'xy', 'level': 0,
                'mult': mult, 'provenance': 'synthetic tiled TIFF with explicitly generated physical level0 pixel grid; not natural WSI',
                'backbone': backbone}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

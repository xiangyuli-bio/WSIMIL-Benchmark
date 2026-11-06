"""Synthetic full RGB slides for end-to-end backbone/ordinal protocol checks."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .io import digest_file

IMAGENET_RESNET50_SHA256 = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'


def create_raw_demo(output, patients=30, classes=3, tile_size=32, seed=71, hdf5=False, initial_weights=None):
    if patients < 5*classes or classes < 2 or tile_size < 32:
        raise ValueError('Raw demo needs at least5patients/class and genuine RGB patches of32px or more')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'images').mkdir()
    rng = np.random.default_rng(seed)
    slides, images, splits = [], [], []
    for i in range(patients):
        patient, slide = f'raw_patient_{i:04d}', f'raw_slide_{i:04d}'
        target = i % classes
        subset = ['train', 'train', 'train', 'val', 'test'][(i//classes) % 5]
        pixels = rng.integers(10+target*10, 170+target*10, (3, 3*tile_size+7, 3*tile_size+11), dtype=np.uint8)
        pixels[:, :tile_size, :tile_size] = 255
        relative = 'images/'+slide+('.h5' if hdf5 else '.npy')
        if hdf5:
            with h5py.File(root/relative, 'w') as f:
                f['pixels'] = pixels
        else:
            np.save(root/relative, pixels)
        slides.append({'slide_id': slide, 'patient_id': patient, 'label': f'grade_{target}'})
        images.append({'slide_id': slide, 'path': relative, 'sha256': digest_file(root/relative)})
        splits.append({'patient_id': patient, 'subset': subset})
    for name, records in [('slides.csv', slides), ('images.csv', images), ('split.csv', splits)]:
        pd.DataFrame(records).to_csv(root/name, index=False)
    backbone = {'architecture': 'resnet50', 'initialization': 'explicit_random'}
    if initial_weights is not None:
        backbone.update(initialization='imagenet1k_v1', path=str(Path(initial_weights).resolve()), sha256=IMAGENET_RESNET50_SHA256)
    task = {'schema_version': 1, 'task_id': 'synthetic_raw_rgb_ordinal', 'task_type': 'ordinal',
            'classes': [f'grade_{i}' for i in range(classes)], 'synthetic': True,
            'label_provenance': 'synthetic ordinal grades only; no clinical or WSI reproduction claim',
            'evaluation_unit': 'slide', 'training_unit': 'slide', 'slides': 'slides.csv',
            'defaults': {'fm': 'raw_resnet', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
            'encoders': {'raw_resnet': {'dimension': 3, 'index': 'images.csv',
                'raw_images': {'kind': 'decoded_rgb', 'level': 1, 'downsample': 4.,
                    'provenance': 'synthetic complete RGB level, no real patient/patch coordinates', 'backbone': backbone}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

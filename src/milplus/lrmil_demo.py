"""Synthetic complete LRMIL inputs; public training integration remains pending."""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .focus_network import WEIGHTS_SHA
from .lrmil_data import KIND, prepare
from .lrmil_source import SOURCE_LAYERS


def create_lrmil_demo(output, patients=20, classes=2, low_patches=2, checkpoint=None):
    if type(low_patches) is not int or low_patches < 2:
        raise ValueError('LRMIL fixture needs at least two actual measured-in-fixture LR regions')
    checkpoint = Path(checkpoint or os.environ.get('MILPLUS_FOCUS_WEIGHTS',
        Path.home()/'.cache/milplus/assets/focus/conch.pth')).expanduser().resolve()
    root = create_demo(output, patients=patients, dimension=768, classes=classes)
    low = np.asarray([[1024*(i % 16), 1024*(i//16)] for i in range(low_patches)], dtype=np.int64)
    offsets = np.asarray([[x*256, y*256] for y in range(4) for x in range(4)])
    high = np.concatenate([coord[None]+offsets for coord in low], axis=0)
    bundle = Bundle(root)
    records = []
    for index, row in enumerate(bundle.rows.to_dict('records')):
        random = np.random.default_rng(7500+index)
        final = random.normal(size=(len(high), 768)).astype(np.float32)
        final[:, row['target']] += 2
        images = random.integers(0, 256, size=(low_patches, 256, 256, 3), dtype=np.uint8)
        path = root/'lrmil_input'/f'{row["slide_id"]}.h5'
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, 'w') as archive:
            archive['images'] = images
            archive['coords_low'], archive['coords_high'] = low, high
            archive['teacher_final'] = final
            for layer in SOURCE_LAYERS:
                archive[f'layer_{layer}'] = random.normal(size=(len(high), 768)).astype(np.float32)
            archive.attrs['level0_mpp'] = .5
            archive.attrs['high_mpp'] = .5
            archive.attrs['low_mpp'] = 2.
        with h5py.File(root/row['path'], 'w') as primary:
            primary['features'], primary['coords'] = final, high
        records.append({'slide_id': row['slide_id'], 'path': str(path)})
    slides = pd.read_csv(root/'slides.csv')
    slides['width_level0'] = min(16, low_patches)*1024
    slides['height_level0'] = ((low_patches+15)//16)*1024
    slides.to_csv(root/'slides.csv', index=False)
    path = root/'task.yaml'
    task = yaml.safe_load(path.read_text())
    task['evaluation_unit'] = 'slide'
    task['encoders']['toy']['spatial'] = {'unit': 'level0_pixels', 'origin': 'top_left',
        'patch_size': 256, 'provenance': 'Explicit synthetic measured-in-fixture20x/5x4by4 grid; not actual WSI'}
    task['encoders']['toy']['feature_preprocessing'] = 'Independent synthetic Gaussian final/layer0/1/10 teacher values; not actual CONCH extraction'
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    pd.DataFrame(records).to_csv(root/'lrmil_index.csv', index=False)
    schema = {'schema_version': 1, 'kind': KIND, 'synthetic': True,
        'coordinate_frame': 'level0_top_left_pixels', 'rgb_dtype': 'uint8', 'rgb_layout': 'NHWC',
        'low_patch_pixels': 256, 'high_patch_pixels': 256, 'low_magnification': 5, 'high_magnification': 20,
        'teacher_representation': 'CONCH_native_visual_trunk_normalized_CLS_768',
        'teacher_layers': list(SOURCE_LAYERS), 'pretrained_weights_sha256': WEIGHTS_SHA,
        'pretrained_checkpoint': str(checkpoint), 'rgb_key': 'images', 'low_coords_key': 'coords_low',
        'high_coords_key': 'coords_high', 'teacher_final_key': 'teacher_final', 'teacher_layer_prefix': 'layer_',
        'coordinate_provenance': 'Explicit synthetic complete4x4 measured-in-fixture physical grid; not a natural WSI',
        'image_provenance': 'Synthetic random uint8 RGB256, not a pathology patch',
        'teacher_feature_provenance': 'Independent synthetic Gaussian768 feature values for full architecture software checks; NOT actual CONCH encoding',
        'weights_provenance': 'Actual fixed public FOCUS-linked CONCH pretrained checkpoint; original CONCH terms apply'}
    manifest = prepare(Bundle(root), root/'lrmil_index.csv', schema, root/'lrmil/manifest.json')
    task['encoders']['toy']['lrmil'] = {'manifest': 'lrmil/manifest.json'}
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root, manifest

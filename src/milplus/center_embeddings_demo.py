"""Synthetic measured-coordinate RGB fixture, full source256 pixels and K32."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .center_embeddings_assets import default_root, verify
from .center_embeddings_data import KIND, prepare
from .data import Bundle
from .io import write_json


def create_center_embeddings_demo(output, patients=6, patches=33, test_patches=None, asset_root=None):
    if (type(patients) is not int or patients < 6 or type(patches) is not int or patches < 32
            or (test_patches is not None and (type(test_patches) is not int or test_patches < 1))):
        raise ValueError('Center Embeddings fixture requires at least6patients and32actual256RGB TRAIN/VAL tiles')
    assets = verify(asset_root or default_root())
    root = Path(output); root.mkdir(parents=True, exist_ok=False); (root/'images').mkdir()
    slides, images, splits = [], [], []
    yy, xx = np.indices((256, 256))
    for i in range(patients):
        slide, patient = f'center_slide_{i:04d}', f'center_patient_{i:04d}'
        target = i % 2; subset = ['train', 'val', 'test'][(i//2) % 3]
        count = test_patches if subset == 'test' and test_patches is not None else patches
        random = np.random.default_rng(8100+i); path = root/'images'/f'{slide}.h5'
        with h5py.File(path, 'w') as archive:
            pixels = archive.create_dataset('pixels', (count, 256, 256, 3), dtype=np.uint8,
                chunks=(1, 256, 256, 3), compression='gzip', compression_opts=1)
            for n in range(count):
                color = random.integers(25, 145, 3); color[target] += 50
                pattern = ((xx//(3+n % 17)+yy//(4+n % 9)) % 2)*30
                pixels[n] = np.clip(color[None, None, :]+pattern[:, :, None], 0, 255).astype(np.uint8)
            archive['patch_ids'] = np.asarray([f'{slide}/original_{n:05d}' for n in range(count)], dtype=h5py.string_dtype())
            archive['coords'] = np.array([[512+(n % 8)*1024, 512+(n//8)*1024] for n in range(count)], dtype=np.int64)
            archive.attrs['extraction_provenance'] = 'Explicit synthetic256RGB and synthetic level0 centers; no actual WSI or FM extraction'
        slides.append({'slide_id': slide, 'patient_id': patient, 'label': f'class_{target}'})
        images.append({'slide_id': slide, 'path': str(path)})
        splits.append({'patient_id': patient, 'subset': subset})
    for name, rows in [('slides.csv', slides), ('images.csv', images), ('split.csv', splits)]:
        pd.DataFrame(rows).to_csv(root/name, index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_center_embeddings_source_RGB', 'task_type': 'classification',
        'classes': ['class_0', 'class_1'], 'synthetic': True, 'training_unit': 'slide', 'evaluation_unit': 'slide',
        'label_provenance': 'Synthetic only, no clinical WSI or historical paper metric claim', 'slides': 'slides.csv',
        'defaults': {'fm': 'raw_resnet34', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'raw_resnet34': {'dimension': 3, 'index': 'images.csv',
            'weights': 'Actual original ImageNet ResNet34-333f7ec4, trained end to end',
            'preprocessing': 'Full256RGB source torchvision0.7 augmentation/normalization'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    schema = {'schema_version': 1, 'kind': KIND, 'layout': 'NHWC', 'dtype': 'uint8', 'color': 'RGB',
        'patch_pixels': 256, 'pixel_key': 'pixels', 'patch_ids_key': 'patch_ids', 'coords_key': 'coords',
        'normalization': 'torchvision070_resize256_RHF05_Tensor_norm05_01', 'coordinate_frame': 'level0_centers',
        'positive_class': 'class_1', 'synthetic_coordinates': True,
        'image_provenance': 'Explicit synthetic patterned RGB, not real WSI/FM features',
        'coordinate_provenance': 'Explicit synthetic row-aligned level0 centers, not clinical acquisition',
        'asset_root': str(assets)}
    write_json(root/'center_schema.json', schema)
    prepare(Bundle(root), schema, root/'center_embeddings/manifest.json')
    task['encoders']['raw_resnet34']['center_embeddings'] = {'manifest': 'center_embeddings/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

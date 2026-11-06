"""Explicit synthetic RGB bags with real historical pretrained initialization."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .c2c_assets import default_root, verify
from .c2c_data import KIND, prepare
from .data import Bundle
from .io import write_json


def create_c2c_demo(output, patients=20, classes=2, patches=64, patch_pixels=64, asset_root=None):
    if (type(classes) is not int or classes < 2 or type(patients) is not int or patients < 5*classes
            or type(patches) is not int or patches < 1 or type(patch_pixels) is not int or patch_pixels < 32):
        raise ValueError('C2C synthetic RGB fixture needs valid bags and at least5patients/class')
    assets = verify(asset_root or default_root()).resolve()
    root = Path(output); root.mkdir(parents=True, exist_ok=False); (root/'images').mkdir()
    slides, images, splits = [], [], []
    for i in range(patients):
        slide, patient = f'c2c_slide_{i:04d}', f'c2c_patient_{i:04d}'
        label = i % classes; subset = ['train', 'train', 'train', 'val', 'test'][(i//classes) % 5]
        path = root/'images'/f'{slide}.h5'
        random = np.random.default_rng(7700+i)
        # Compressible procedural image structures keep the native512 RGB
        # fixture's storage bounded without changing image/patch/model shapes.
        grid_y, grid_x = np.indices((patch_pixels, patch_pixels))
        with h5py.File(path, 'w') as archive:
            pixels = archive.create_dataset('pixels', (patches, patch_pixels, patch_pixels, 3),
                dtype=np.uint8, chunks=(1, patch_pixels, patch_pixels, 3), compression='gzip', compression_opts=1)
            for index in range(patches):
                color = random.integers(20, 150, 3)
                color[label % 3] += 45
                period = 3+index % 13
                texture = ((grid_x//period+grid_y//(2+index % 7)) % 2)*30
                pixels[index] = np.clip(texture[:, :, None]+color, 0, 255).astype(np.uint8)
            archive['patch_ids'] = np.asarray([f'{slide}/original_patch_{n:05d}.png' for n in range(patches)], dtype=h5py.string_dtype())
        slides.append({'slide_id': slide, 'patient_id': patient, 'label': f'class_{label}'})
        images.append({'slide_id': slide, 'path': str(path)})
        splits.append({'patient_id': patient, 'subset': subset})
    for name, records in [('slides.csv', slides), ('images.csv', images), ('split.csv', splits)]:
        pd.DataFrame(records).to_csv(root/name, index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_c2c_raw_RGB', 'task_type': 'classification',
        'classes': [f'class_{n}' for n in range(classes)], 'synthetic': True,
        'label_provenance': 'Synthetic only; no clinical/WSI/FM-extraction results',
        'evaluation_unit': 'slide', 'training_unit': 'slide', 'slides': 'slides.csv',
        'defaults': {'fm': 'raw_resnet18', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'raw_resnet18': {'dimension': 3, 'index': 'images.csv',
            'weights': 'original torchvision0.6.1 ImageNet ResNet18, trained end-to-end',
            'preprocessing': 'Original RGB patch bytes, Albumentations0.4.6 ToTensor only'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    schema = {'schema_version': 1, 'kind': KIND, 'layout': 'NHWC', 'dtype': 'uint8', 'color': 'RGB',
        'normalization': 'albumentations046_ToTensor_no_normalize', 'coordinate_frame': 'path_identity_only',
        'patch_pixels': patch_pixels, 'pixel_key': 'pixels', 'patch_ids_key': 'patch_ids',
        'image_provenance': 'Explicit synthetic patterned RGB; NOT actual WSI images or FM features',
        'patch_provenance': 'Explicit synthetic stable patch identities; no physical coordinates invented',
        'asset_root': str(assets)}
    write_json(root/'c2c_schema.json', schema)
    prepare(Bundle(root), schema, root/'c2c/manifest.json')
    task['encoders']['raw_resnet18']['c2c'] = {'manifest': 'c2c/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

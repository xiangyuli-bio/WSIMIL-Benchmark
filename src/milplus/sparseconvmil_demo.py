"""Explicit synthetic spatial RGB fixture for the complete trainable network."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .io import write_json
from .sparseconvmil_data import KIND, prepare


def create_sparseconvmil_demo(output, patients=20, classes=2, patches=16, patch_pixels=64):
    if (type(classes) is not int or classes < 2 or type(patients) is not int or patients < 5*classes
            or type(patches) is not int or patches < 1 or type(patch_pixels) is not int or patch_pixels < 32):
        raise ValueError('SparseConvMIL demo needs valid RGB/spatial bags and at least5patients/class')
    root = Path(output); root.mkdir(parents=True, exist_ok=False); (root/'images').mkdir()
    slides, images, splits = [], [], []
    for i in range(patients):
        slide, patient = f'scn_slide_{i:04d}', f'scn_patient_{i:04d}'
        label = i % classes; subset = ['train', 'train', 'train', 'val', 'test'][(i//classes) % 5]
        path = root/'images'/f'{slide}.h5'
        rng = np.random.default_rng(7800+i)
        yy, xx = np.indices((patch_pixels, patch_pixels))
        with h5py.File(path, 'w') as archive:
            values = archive.create_dataset('pixels', (patches, patch_pixels, patch_pixels, 3),
                dtype=np.uint8, chunks=(1, patch_pixels, patch_pixels, 3), compression='gzip', compression_opts=1)
            for index in range(patches):
                color = rng.integers(20, 140, 3); color[label % 3] += 45
                texture = ((xx//(3+index % 13)+yy//(2+index % 7)) % 2)*30
                values[index] = np.clip(texture[:, :, None]+color, 0, 255).astype(np.uint8)
            coords = np.array([[(n % 4)*8, (n//4)*8] for n in range(patches)], dtype=np.int64)
            archive['coords'] = coords
            archive['patch_ids'] = np.asarray([f'{slide}/{x}_{y}.png' for x, y in coords], dtype=h5py.string_dtype())
        slides.append({'slide_id': slide, 'patient_id': patient, 'label': f'class_{label}'})
        images.append({'slide_id': slide, 'path': str(path)})
        splits.append({'patient_id': patient, 'subset': subset})
    for name, rows in [('slides.csv', slides), ('images.csv', images), ('split.csv', splits)]:
        pd.DataFrame(rows).to_csv(root/name, index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_sparseconvmil_spatial_RGB', 'task_type': 'classification',
        'classes': [f'class_{n}' for n in range(classes)], 'synthetic': True,
        'label_provenance': 'Synthetic software validation; no clinical WSI/FM/paper results',
        'evaluation_unit': 'slide', 'training_unit': 'slide', 'slides': 'slides.csv',
        'defaults': {'fm': 'raw_resnet18', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'raw_resnet18': {'dimension': 3, 'index': 'images.csv',
            'weights': 'full random-initialized ResNet18, trained end-to-end',
            'preprocessing': 'Original RGB bytes, ToTensor only; no resize'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    schema = {'schema_version': 1, 'kind': KIND, 'layout': 'NHWC', 'dtype': 'uint8', 'color': 'RGB',
        'normalization': 'ToTensor_only', 'coordinate_frame': 'tile_grid_indices',
        'patch_pixels': patch_pixels, 'pixel_key': 'pixels', 'patch_ids_key': 'patch_ids', 'coords_key': 'coords',
        'image_provenance': 'Explicit synthetic patterned RGB, no actual pathology images',
        'patch_provenance': 'Synthetic patch IDs; not actual WSI sampling',
        'coordinate_provenance': 'Explicit synthetic tile-grid positions; no invented real slide geometry'}
    write_json(root/'sparseconvmil_schema.json', schema)
    prepare(Bundle(root), schema, root/'sparseconvmil/manifest.json')
    task['encoders']['raw_resnet18']['sparseconvmil'] = {'manifest': 'sparseconvmil/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

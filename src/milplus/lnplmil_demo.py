"""Explicit synthetic full RGB224/LPA input; never clinical data or fake FM RGB."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .lnplmil_raw import KIND, prepare_bundle


def create_demo(output, assets, patches=401):
    if type(patches) is not int or patches < 401:
        raise ValueError('This complete SP/TOD fixture uses at least401 actual RGB224 patches per slide')
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'pixels').mkdir()
    slides, index, splits = [], [], []
    for i in range(6):
        name = 'synthetic_lnpl_' + str(i)
        target, subset = i % 2, ['train', 'val', 'test'][i // 2]
        path = root / 'pixels' / (name + '.h5')
        with h5py.File(path, 'x') as file:
            pixels = file.create_dataset('rgb', (patches, 224, 224, 3), dtype='uint8',
                chunks=(1, 224, 224, 3), compression='gzip')
            for j in range(patches):
                level = 64 + target * 64 + j * 7 if j < 2 else 32 + j % 180
                image = np.full((224, 224, 3), level, dtype=np.uint8)
                image[32:96, 11:119, target if j < 2 else j % 3] = 200 if j < 2 else 180
                pixels[j] = image
            file.create_dataset('ids', data=np.asarray([name + '_p' + str(j) for j in range(patches)], dtype=h5py.string_dtype('utf-8')))
            file.create_dataset('coords', data=np.asarray([[(j % 100) * 1200, (j // 100) * 1200] for j in range(patches)], dtype=np.int64))
            file.create_dataset('labelled', data=np.asarray([0, 1] if subset == 'train' else [], dtype=np.int64))
            file.create_dataset('patch_labels', data=np.asarray([target, target] if subset == 'train' else [], dtype=np.int64))
        slides.append({'slide_id': name, 'patient_id': name, 'global_patient_id': 'synthetic_lnpl:' + name,
                       'label': ['negative', 'positive'][target]})
        index.append({'slide_id': name, 'path': str(path)})
        splits.append({'patient_id': name, 'subset': subset})
    pd.DataFrame(slides).to_csv(root / 'slides.csv', index=False)
    pd.DataFrame(index).to_csv(root / 'pixels.csv', index=False)
    pd.DataFrame(splits).to_csv(root / 'split.csv', index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_lnpl_full_reference', 'synthetic': True,
        'task_type': 'classification', 'classes': ['negative', 'positive'], 'training_unit': 'slide',
        'evaluation_unit': 'slide', 'slides': 'slides.csv', 'label_provenance': 'Synthetic software fixture only',
        'defaults': {'fm': 'raw_rgb', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'raw_rgb': {'dimension': 3, 'index': 'pixels.csv'}}}
    (root / 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    schema = {'kind': KIND, 'schema_version': 1, 'layout': 'NHWC', 'color': 'RGB', 'patch_pixels': 224,
        'coordinate_frame': 'level0_top_left_pixels', 'magnification': 40, 'mpp': .25,
        'pixels_key': 'rgb', 'patch_ids_key': 'ids', 'coords_key': 'coords',
        'labelled_indices_key': 'labelled', 'patch_labels_key': 'patch_labels',
        'image_provenance': 'Explicit synthetic actual RGB224 arrays, not pathology data',
        'annotation_provenance': 'Explicit synthetic observed TRAIN-only patch labels',
        'patch_provenance': 'Explicit synthetic ordered identities and measured1200-grid coordinates',
        'initial_assets': str(Path(assets).resolve()), 'weak_classes': ['negative', 'positive'],
        'positive_weak_class_ids': [1], 'synthetic': True}
    (root / 'schema.yaml').write_text(yaml.safe_dump(schema, sort_keys=False))
    from .data import Bundle
    prepare_bundle(Bundle(root), schema, 'lnplmil_manifest.json')
    task['encoders']['raw_rgb']['lnplmil'] = {'manifest': 'lnplmil_manifest.json'}
    (root / 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    Bundle(root).check(deep=True)
    return root

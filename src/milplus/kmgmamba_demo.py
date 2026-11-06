"""Explicit synthetic dual-scale KMG-Mamba data; no actual WSI/FM extraction."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .kmgmamba_data import KIND, prepare


def create_kmgmamba_demo(output, patients=36, dimension=1024, classes=2, roots=9):
    if type(roots) is not int or roots < 1:
        raise ValueError('KMG-Mamba synthetic root count must be a positive integer')
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    bundle = Bundle(root)
    coarse = np.asarray([[(i % 32)*2048, (i//32)*2048] for i in range(roots)], dtype=np.int64)
    fine = np.asarray([coord+offset for coord in coarse
                       for offset in np.asarray([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])])
    records = []
    for number, row in enumerate(bundle.rows.to_dict('records')):
        path = root/'dual_scale'/f'{row["slide_id"]}.h5'
        path.parent.mkdir(exist_ok=True)
        random = np.random.default_rng(7300+number)
        low = random.normal(size=(roots, dimension)).astype(np.float32)
        high = random.normal(size=(len(fine), dimension)).astype(np.float32)
        low[:, row['target']] += 2.; high[:, row['target']] += 2.
        with h5py.File(path, 'w') as archive:
            archive['features_low'], archive['coords_low'] = low, coarse
            archive['features_high'], archive['coords_high'] = high, fine
        with h5py.File(root/row['path'], 'w') as primary:
            primary['features'], primary['coords'] = low, coarse
        records.append({'slide_id': row['slide_id'], 'path': str(path)})
    slides = pd.read_csv(root/'slides.csv')
    slides['width_level0'], slides['height_level0'] = min(32, roots)*2048, ((roots+31)//32)*2048
    slides.to_csv(root/'slides.csv', index=False)
    task_path = root/'task.yaml'; task = yaml.safe_load(task_path.read_text())
    task['encoders']['toy']['spatial'] = {'unit': 'level0_pixels', 'origin': 'top_left', 'patch_size': 2048,
        'provenance': 'Explicit synthetic two-dimensional measured-in-fixture coarse2048/fine1024 grid; not real WSI coordinates'}
    task_path.write_text(yaml.safe_dump(task, sort_keys=False))
    pd.DataFrame(records).to_csv(root/'kmgmamba_index.csv', index=False)
    schema = {'schema_version': 1, 'kind': KIND, 'coordinate_frame': 'level0_top_left_pixels',
        'levels': [{'magnification': 5, 'patch_size_level0': 2048,
                    'feature_key': 'features_low', 'coords_key': 'coords_low'},
                   {'magnification': 10, 'patch_size_level0': 1024,
                    'feature_key': 'features_high', 'coords_key': 'coords_high'}],
        **{key: 'Explicit synthetic Gaussian two-scale features and measured-in-fixture pixel grid; NOT real WSI/FM extraction'
           for key in ['coordinate_provenance', 'weights_provenance', 'preprocessing', 'patch_size_provenance']}}
    prepare(Bundle(root), root/'kmgmamba_index.csv', schema, root/'kmgmamba/manifest.json')
    path = root/'task.yaml'; task = yaml.safe_load(path.read_text())
    task['encoders']['toy']['kmgmamba'] = {'manifest': 'kmgmamba/manifest.json'}
    task['evaluation_unit'] = 'slide'
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

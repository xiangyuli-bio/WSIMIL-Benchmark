"""Explicit synthetic recursive pyramids; not real image/FM extraction."""
import itertools

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .io import write_json
from .zoommil_data import KIND, prepare


def create_zoommil_demo(output, patients=18, dimension=1024, classes=2, roots=16, geometry='bright'):
    if type(roots) is not int or roots < 1 or geometry not in {'bright', 'crc'}:
        raise ValueError('ZoomMIL fixture needs positive actual root count and named native hierarchy')
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    bundle = Bundle(root)
    mags = [1.25, 2.5, 10.] if geometry == 'bright' else [5., 10., 20.]
    depths = [round(np.log2(mag/mags[0])) for mag in mags]
    quadrants = [(0, 0), (1, 0), (0, 1), (1, 1)]
    base_coords = np.asarray([(index//4, index % 4) for index in range(roots)], dtype=np.int64)
    index = []
    for number, row in enumerate(bundle.rows.to_dict('records')):
        path = root/'pyramids'/f'{row["slide_id"]}.h5'
        path.parent.mkdir(exist_ok=True)
        generator = np.random.default_rng(7200+number)
        with h5py.File(path, 'w') as archive:
            for level, (mag, depth) in enumerate(zip(mags, depths)):
                coords = np.asarray([(*coord, *(value for child in children for value in child))
                    for coord in base_coords for children in itertools.product(quadrants, repeat=depth)], dtype=np.int64)
                values = generator.normal(size=(len(coords), dimension)).astype(np.float32)
                values[:, row['target']] += 1.25
                archive[f'{mag}x_patches'] = values
                archive[f'{mag}x_coords'] = coords
                if level == 0:
                    with h5py.File(root/row['path'], 'w') as primary:
                        primary['features'] = values
                        primary['coords'] = base_coords[:, [1, 0]]*256
        index.append({'slide_id': row['slide_id'], 'path': str(path)})
    slides = pd.read_csv(root/'slides.csv')
    slides['width_level0'] = min(roots, 4)*256
    slides['height_level0'] = ((roots+3)//4)*256
    slides.to_csv(root/'slides.csv', index=False)
    pd.DataFrame(index).to_csv(root/'zoommil_index.csv', index=False)
    schema = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'patch_pixels': 256,
        'coordinate_frame': 'source_recursive_row_col_indices',
        'levels': [{'magnification': mag, 'feature_key': f'{mag}x_patches', 'coords_key': f'{mag}x_coords'} for mag in mags],
        **{key: 'Explicit synthetic complete spatial quadrants/Gaussian features; NOT real WSI, FM or image extraction'
           for key in ['coordinate_provenance', 'weights_provenance', 'preprocessing', 'multiscale_provenance']}}
    write_json(root/'zoommil_schema.json', schema)
    prepare(Bundle(root), root/'zoommil_index.csv', schema, root/'zoommil/manifest.json')
    path = root/'task.yaml'; task = yaml.safe_load(path.read_text())
    task['evaluation_unit'] = 'slide'
    task['encoders']['toy']['zoommil'] = {'manifest': 'zoommil/manifest.json'}
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

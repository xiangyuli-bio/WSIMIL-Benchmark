"""Synthetic measured-in-fixture matched RGB/features, never WSI validation."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .hdmil_data import KIND, prepare


def create_hdmil_demo(output, patients=20, dimension=1024, classes=2, patches=40):
    if type(patches) is not int or patches < 32:
        raise ValueError('HDMIL fixture preserves native k32 genuine instances')
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    bundle = Bundle(root); records = []
    coordinates = np.column_stack([np.arange(patches)*256, np.zeros(patches)]).astype(np.int64)
    for number, row in enumerate(bundle.rows.to_dict('records')):
        rng = np.random.default_rng(14500+number)
        features = rng.normal(size=(patches, dimension)).astype(np.float32)
        features[:, row['target']] += 1.25
        images = rng.integers(0, 256, size=(patches, 16, 16, 3), dtype=np.uint8)
        path = root/'low_rgb'/f'{row["slide_id"]}.h5'; path.parent.mkdir(exist_ok=True)
        with h5py.File(path, 'w') as f:
            f['rgb'], f['coords'] = images, coordinates
            f.attrs.update(level0_mpp=.5, high_mpp=.5, low_mpp=8.)
        with h5py.File(root/row['path'], 'w') as f:
            f['features'], f['coords'] = features, coordinates
        records.append(dict(slide_id=row['slide_id'], path=str(path)))
    slides = pd.read_csv(root/'slides.csv'); slides['width_level0'], slides['height_level0'] = patches*256, 256
    slides.to_csv(root/'slides.csv', index=False)
    pd.DataFrame(records).to_csv(root/'hdmil_index.csv', index=False)
    schema = dict(schema_version=1, kind=KIND, coordinate_frame='level0_top_left_pixels',
        rgb_dtype='uint8', rgb_layout='NHWC', low_patch_pixels=16, high_patch_pixels=256,
        low_magnification=1.25, high_magnification=20, rgb_key='rgb', coords_key='coords',
        **{k: 'Explicit synthetic independent RGB pixels and Gaussian frozen features; not natural WSI/FM extraction'
           for k in ['coordinate_provenance', 'image_provenance', 'weights_provenance']})
    prepare(Bundle(root), root/'hdmil_index.csv', schema, root/'hdmil/manifest.json')
    path = root/'task.yaml'; task = yaml.safe_load(path.read_text())
    task['encoders']['toy']['hdmil'] = dict(manifest='hdmil/manifest.json')
    task['evaluation_unit'] = 'slide'
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

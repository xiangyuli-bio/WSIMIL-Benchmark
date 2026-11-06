"""Permanently marked synthetic SAM memberships, never actual SAM outputs."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .sammil_data import KIND, prepare


def create_sammil_demo(output, patients=30, dimension=1024, classes=3, patches=64):
    if type(patches) is not int or patches < 16:
        raise ValueError('Synthetic SAM-MIL fixture needs at least16 explicit patch coordinates')
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    bundle = Bundle(root); records = []
    coords = np.asarray([[(i % 16)*512, (i//16)*512] for i in range(patches)], dtype=np.int64)
    for number, row in enumerate(bundle.rows.to_dict('records')):
        random = np.random.default_rng(7600+number)
        feature = random.normal(size=(patches, dimension)).astype(np.float32)
        feature[:, row['target']] += 1.25
        with h5py.File(root/row['path'], 'w') as archive:
            archive['features'], archive['coords'] = feature, coords
        path = root/'sammil_groups'/f'{row["slide_id"]}.h5'; path.parent.mkdir(exist_ok=True)
        with h5py.File(path, 'w') as archive:
            archive['coords'] = coords
            group = archive.create_group('groups')
            for i, (start, stop) in enumerate([(0, patches//2), (patches//4, 3*patches//4), (patches//2, patches)]):
                item = group.create_dataset(f'seg_{i}', data=coords[start:stop]); item.attrs['area'] = [7000, 11000, 15000][i]
        records.append({'slide_id': row['slide_id'], 'path': str(path)})
    pd.DataFrame(records).to_csv(root/'sammil_index.csv', index=False)
    schema = {'schema_version': 1, 'kind': KIND, 'coordinate_frame': 'level0_top_left_pixels',
        'group_feature': 'arithmetic_mean_of_registered_member_patch_features',
        'overlap': 'minimum_original_mask_area', 'sam_model': 'vit_h',
        'sam_weights_sha256': '0'*64,
        **{key: 'Explicit synthetic Gaussian features, fixture pixel grid and overlapping region memberships. NOT a SAM segmentation or actual FM/WSI experiment.'
           for key in ['segmentation_provenance', 'membership_provenance', 'feature_provenance']}}
    from .io import write_json
    write_json(root/'sammil_schema.json', schema)
    prepare(Bundle(root), root/'sammil_index.csv', schema, root/'sammil/manifest.json')
    path = root/'task.yaml'; task = yaml.safe_load(path.read_text())
    task['encoders']['toy']['sammil'] = {'manifest': 'sammil/manifest.json'}
    task['evaluation_unit'] = 'slide'
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

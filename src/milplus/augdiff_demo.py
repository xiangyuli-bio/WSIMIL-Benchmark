"""Explicit synthetic seven-view software fixture, not image/FM extraction."""
import h5py
import numpy as np

from .augdiff_data import KIND, PAIRED, VIEWS
from .data import Bundle
from .demo import create_demo
from .io import digest_file, digest_object, write_json


def create_augdiff_demo(output, patients=18, dimension=512, instances=3, classes=2):
    if type(instances) is not int or instances < 1:
        raise ValueError('AugDiff software fixture patch count must be a positive integer')
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    bundle = Bundle(root)
    records = []
    train_patients = sorted({row['global_patient_id'] for row in bundle.records('train')})
    for i, row in enumerate(bundle.rows.to_dict('records')):
        path = root/row['path']
        x = np.random.default_rng(i).normal(size=(instances, dimension)).astype(np.float32)
        x[:, row['target']] += 1.25
        with h5py.File(path, 'w') as archive:
            archive['features'] = x
            archive['coords'] = np.stack((np.arange(instances)*256, np.zeros(instances)), axis=1).astype(np.int64)
        if row['subset'] != 'train':
            continue
        order = digest_object([(row['slide_id'], index) for index in range(instances)])
        views = {}
        for j, name in enumerate(VIEWS):
            if j == 0:
                view_path = path
            else:
                view_path = root/'synthetic_views'/f'{row["slide_id"]}_{name}.npy'
                view_path.parent.mkdir(exist_ok=True)
                np.save(view_path, x+j*.01)
            views[name] = {'path': str(view_path), 'sha256': digest_file(view_path),
                           'ordered_patch_ids_sha256': order}
        subset = 'val' if row['global_patient_id'] == train_patients[-1] else 'train'
        records.append({'slide_id': row['slide_id'], 'diffusion_subset': subset,
                        'ordered_patch_ids_sha256': order, 'views': views})
    meta = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'dimension': dimension,
        'view_pairing_policy': PAIRED,
        'fm': bundle.fm, 'encoder_identity': digest_object(bundle.encoder), 'records': records,
        **{key: 'Synthetic software fixture only; offsets are NOT real image-augmentation/FM features' for key in
           ['augmentation_provenance', 'weights_provenance', 'feature_preprocessing',
            'row_correspondence_provenance', 'diffusion_split_provenance']}}
    write_json(root/'augdiff_views.json', meta)
    return root

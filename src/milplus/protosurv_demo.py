"""Explicitly synthetic ProtoSurv graphs, tissue labels and optional fixed prototypes."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .protosurv_data import feature_space, prepare


def create_protosurv_demo(output, dimension=16, patients=30, nodes=15, fixed_prototypes=False):
    root = Path(create_demo(output, patients=patients, dimension=dimension, survival=True))
    slides = pd.read_csv(root/'slides.csv').drop_duplicates('patient_id')
    slides.to_csv(root/'slides.csv', index=False)
    index = pd.read_csv(root/'features.csv')
    index = index[index.slide_id.isin(slides.slide_id)]
    index.to_csv(root/'features.csv', index=False)
    category_dir = root/'categories'
    category_dir.mkdir()
    assignments, rng = [], np.random.default_rng(63)
    for row in index.to_dict('records'):
        coords = np.column_stack([np.arange(nodes)*256, (np.arange(nodes)%3)*256]).astype('int64')
        features = rng.normal(size=(nodes, dimension)).astype('float32')
        with h5py.File(root/row['path'], 'w') as stream:
            stream['features'] = features
            stream['coords'] = coords
        path = category_dir/(row['slide_id']+'.h5')
        with h5py.File(path, 'w') as stream:
            stream['coords'] = coords
            stream['node_types'] = np.arange(nodes, dtype='int64')%5
        assignments.append({'slide_id': row['slide_id'], 'path': str(path) if not fixed_prototypes else ''})
    category_index = root/'categories.csv'
    pd.DataFrame(assignments).to_csv(category_index, index=False)
    schema = {'schema_version': 1, 'kind': 'protosurv_spatial_graph', 'synthetic': True,
        'tissue_classes': ['tumor', 'stroma', 'infiltrate', 'necrosis', 'others'],
        'node_type_mode': 'unknown_fixed_prototype' if fixed_prototypes else 'predicted',
        'provenance': 'synthetic Gaussian software fixture, not biological tissue classifications',
        'graph_provenance': 'declared synthetic real-coordinate arrays and original spatial HNSW with1build thread',
        'category_model': {'name': 'synthetic_index_modulo5', 'revision': 'fixture1',
            'weights_sha256': '0'*64, 'prompt_or_training_provenance': 'synthetic fixture; no model or annotations'}}
    prototypes = provenance = None
    if fixed_prototypes:
        prototypes = root/'prototypes.npy'
        np.save(prototypes, rng.normal(size=(5, dimension)).astype('float32'))
        provenance = {'fit_subset': 'external_declared', 'provenance': 'synthetic Gaussian prototypes; no real cohort or pretrained classifier',
                      'feature_space': feature_space(Bundle(root))}
    manifest = prepare(Bundle(root), category_index, schema, root/'protosurv_assets', prototypes, provenance)
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['task_id'] = 'synthetic_protosurv_survival'
    task['encoders']['toy']['protosurv'] = {'manifest': str(manifest.relative_to(root))}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

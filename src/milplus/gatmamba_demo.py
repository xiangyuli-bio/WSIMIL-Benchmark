"""Synthetic GAT-Mamba multi-slide patient bags; no real subtype labels."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .gatmamba_data import CLASSES, prepare


def create_gatmamba_demo(output, dimension=16, patients=20, nodes=12):
    if type(nodes) is not int or nodes < 1:
        raise ValueError('Synthetic GAT-Mamba nodes must be positive')
    root = Path(create_demo(output, patients=patients, dimension=dimension, survival=True))
    path = root/'task.yaml'
    task = yaml.safe_load(path.read_text())
    task['task_id'] = 'synthetic_gatmamba_survival'
    task['encoders']['toy']['spatial'].update(coordinate_order='xy', patch_size=512,
        provenance='synthetic declared real coordinate arrays; not tissue extracted from WSI')
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    (root/'categories').mkdir()
    rng, assignments = np.random.default_rng(18), []
    for row in pd.read_csv(root/'features.csv').to_dict('records'):
        n = nodes+int(row['slide_id'].endswith('_1'))
        # Deliberately unsorted raw rows: preparation must reorder every array
        # together, using real supplied xy, not fabricate positions from order.
        order = rng.permutation(n)
        points = np.column_stack([np.arange(n)%4*512, np.arange(n)//4*512])[order]
        features = rng.normal(size=(n, dimension)).astype(np.float32)
        types = (np.arange(n)%6).astype(np.int64)[order]
        with h5py.File(root/row['path'], 'w') as stream:
            stream['features'], stream['coords'] = features, points
        category = root/'categories'/(row['slide_id']+'.h5')
        with h5py.File(category, 'w') as stream:
            stream['coords'], stream['node_types'] = points, types
        assignments.append({'slide_id': row['slide_id'], 'path': str(category)})
    index = root/'categories.csv'
    pd.DataFrame(assignments).to_csv(index, index=False)
    schema = {'schema_version': 1, 'kind': 'gatmamba_spatial_subtype_graph', 'synthetic': True,
        'tissue_classes': CLASSES, 'category_protocol': 'explicit_external_classifier',
        'provenance': 'Synthetic Gaussian software fixture, no LUAD classifier or real pathology interpretation',
        'graph_provenance': 'actual synthetic xy arrays, exact source8NN graph,21class pair types and physical/feature distances',
        'patch_order_provenance': 'source natural r/c tile ordering realized by physical y/x row-major order',
        'classifier': {'ensemble_provenance': 'synthetic_index_modulo6; no biological classification',
            'fit_subset': 'external_declared',
            'training_population': 'none; synthetic fixture',
            'models': [{'name': 'synthetic_modulo6', 'revision': 'fixture1', 'weights_sha256': '0'*64}]}}
    manifest = prepare(Bundle(root), index, schema, root/'gatmamba_assets')
    task['encoders']['toy']['gatmamba'] = {'manifest': str(manifest.relative_to(root))}
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

"""Synthetic AGCN feature graph survival task; never represents natural WSI."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .deepgraphsurv_data import prepare
from .demo import create_demo
from .io import write_json


def create_deepgraphsurv_demo(output, patients=20, dimension=128, nodes=12, maximum=500):
    if type(nodes) is not int or nodes < 4:
        raise ValueError('Original DeepGraphSurv requires at least4actual patches per WSI')
    root = Path(create_demo(output, patients=patients, dimension=dimension, survival=True))
    slides, features = pd.read_csv(root/'slides.csv'), pd.read_csv(root/'features.csv')
    slides = slides.drop_duplicates('patient_id', keep='first').copy()
    features = features[features.slide_id.isin(slides.slide_id)]
    for index, row in slides.iterrows():
        number = int(row.patient_id.rsplit('_', 1)[1])
        slides.loc[index, 'event'] = number % 2
        slides.loc[index, 'time'] = 1+number if number % 2 else 100+number
    slides.to_csv(root/'slides.csv', index=False)
    features.to_csv(root/'features.csv', index=False)
    rng = np.random.default_rng(2018)
    for i, row in enumerate(features.to_dict('records')):
        n = nodes-i % min(3, nodes-3)
        x = rng.normal(size=(n, dimension)).astype(np.float32)
        with h5py.File(root/row['path'], 'w') as stream:
            stream['features'] = x
    path = root/'task.yaml'
    task = yaml.safe_load(path.read_text())
    task['task_id'] = 'synthetic_deepgraphsurv_agcn'
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    bundle = Bundle(root)
    schema = {'schema_version': 1, 'kind': 'deepgraphsurv_feature_graph', 'synthetic': True,
        'representation': 'frozen_features_full_dimension',
        'representation_provenance': 'Synthetic Gaussian full-dimensional features; not actual VGG16/PCA or FM outputs',
        'selection_provenance': 'Explicit first synthetic WSI per patient; original NumPy sampling and native feature-distance graph',
        'max_nodes': maximum, 'sampling_seed': 1,
        'slide_order': {s: bundle.rows.loc[bundle.rows.subset == s, 'slide_id'].tolist() for s in ['train', 'val', 'test']}}
    write_json(root/'deepgraphsurv_schema.json', schema)
    manifest = prepare(bundle, schema, root/'deepgraphsurv_assets')
    task['encoders']['toy']['deepgraphsurv'] = {'manifest': str(manifest.relative_to(root))}
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root

"""Synthetic C²MIL software fixture, including explicitly synthetic thumbnails."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .c2mil_data import feature_space, prepare
from .data import Bundle
from .demo import create_demo
from .io import write_json


def create_c2mil_demo(output, patients=30, dimension=16, nodes=10):
    if type(nodes) is not int or nodes < 1:
        raise ValueError('Synthetic C²MIL requires positive node counts')
    root = Path(create_demo(output, patients=patients, dimension=dimension, survival=True))
    # Native data chooses one named diagnostic DX1 WSI, not a mean of slides.
    slides, index = pd.read_csv(root/'slides.csv'), pd.read_csv(root/'features.csv')
    slides = slides.drop_duplicates('patient_id', keep='first')
    index = index[index.slide_id.isin(slides.slide_id)]
    slides.to_csv(root/'slides.csv', index=False);index.to_csv(root/'features.csv', index=False)
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['task_id'] = 'synthetic_c2mil_survival'
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    rng, thumbnails = np.random.default_rng(21), []
    (root/'thumbnails').mkdir()
    for row in index.to_dict('records'):
        coords = np.column_stack([np.arange(nodes)%4*256, np.arange(nodes)//4*256]).astype(np.int64)
        with h5py.File(root/row['path'], 'w') as stream:
            stream['features'] = rng.normal(size=(nodes, dimension)).astype(np.float32)
            stream['coords'] = coords
        thumb = root/'thumbnails'/(row['slide_id']+'.npy')
        np.save(thumb, rng.normal(size=(1, dimension)).astype(np.float32))
        thumbnails.append({'slide_id': row['slide_id'], 'thumbnail_path': str(thumb)})
    pd.DataFrame(thumbnails).to_csv(root/'thumbnails.csv', index=False)
    bundle = Bundle(root)
    schema = {'schema_version': 1, 'kind': 'c2mil_graph_thumbnail_reference', 'synthetic': True,
        'selection_provenance': 'Synthetic first slide per patient; no actual TCGA diagnostic specimen',
        'graph_provenance': 'Declared synthetic xy arrays and original HNSW8NN; not natural WSI coordinates',
        'thumbnail': {'kind': 'whole_slide_image_embedding', 'feature_space': feature_space(bundle),
            'image_generation': 'independent Gaussian synthetic thumbnail fixture; no OpenSlide image',
            'encoder_preprocessing': 'synthetic Gaussian only',
            'provenance': 'explicitly synthetic; never patch feature averaging'},
        'patient_order': {s: bundle.rows.loc[bundle.rows.subset == s, 'global_patient_id'].tolist()
                          for s in ['train', 'val', 'test']}}
    write_json(root/'c2mil_schema.json', schema)
    manifest = prepare(bundle, root/'thumbnails.csv', schema, root/'c2mil_assets')
    task['encoders']['toy']['c2mil'] = {'manifest': str(manifest.relative_to(root))}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

"""Synthetic actual-coordinate MASS/hypergraph survival software fixture."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .io import write_json
from .triagemil_data import prepare


def create_triagemil_demo(output, patients=20, dimension=16, nodes=48):
    if type(nodes) is not int or nodes < 1:
        raise ValueError('Synthetic TRIAGE-MIL requires actual positive node count')
    root = Path(create_demo(output, patients=patients, dimension=dimension, survival=True))
    slides, features = pd.read_csv(root/'slides.csv'), pd.read_csv(root/'features.csv')
    slides = slides.drop_duplicates('patient_id', keep='first')
    features = features[features.slide_id.isin(slides.slide_id)]
    slides.to_csv(root/'slides.csv', index=False);features.to_csv(root/'features.csv', index=False)
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['task_id'] = 'synthetic_triagemil_survival'
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    rng = np.random.default_rng(35)
    for row in features.to_dict('records'):
        order = rng.permutation(nodes)
        x = rng.normal(size=(nodes, dimension)).astype(np.float32)
        coords = np.column_stack([np.arange(nodes)%8*256, np.arange(nodes)//8*256]).astype(np.int64)[order]
        with h5py.File(root/row['path'], 'w') as stream:
            stream['features'], stream['coords'] = x, coords
    bundle = Bundle(root)
    schema = {'schema_version': 1, 'kind': 'triagemil_mass_hypergraph', 'synthetic': True,
        'selection_provenance': 'Synthetic declared first diagnostic-like WSI; Gaussian features, no actual tissue interpretation',
        'graph_provenance': 'Original MASS quality/four-axis selection and semantic hypergraph on supplied synthetic xy',
        'patient_order': {s: bundle.rows.loc[bundle.rows.subset == s, 'global_patient_id'].tolist()
                          for s in ['train', 'val', 'test']}}
    write_json(root/'triagemil_schema.json', schema)
    manifest = prepare(bundle, schema, root/'triagemil_assets')
    task['encoders']['toy']['triagemil'] = {'manifest': str(manifest.relative_to(root))}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

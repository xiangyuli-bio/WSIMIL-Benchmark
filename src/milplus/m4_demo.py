"""Explicit synthetic complete multi-endpoint bags for full M4 software checks."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .demo import create_demo


def create_m4_demo(output, patients=20, dimension=2048, endpoints=10, patches=None):
    if type(endpoints) is not int or endpoints < 2:
        raise ValueError('Synthetic M4 requires at least two explicit binary endpoints')
    if patches is not None and (type(patches) is not int or patches < 1):
        raise ValueError('Synthetic M4 patch count must be a positive integer')
    root = create_demo(output, patients=patients, dimension=dimension)
    task = yaml.safe_load((root/'task.yaml').read_text())
    slides = pd.read_csv(root/'slides.csv')
    columns = [f'synthetic_endpoint_{i}' for i in range(endpoints)]
    # Deliberately correlated duplicate/complement controls guarantee support in
    # each source split. They are software probes, never biological outcomes.
    for i, name in enumerate(columns):
        slides[name] = slides.patient_id.map(lambda p, bit=i: (int(p.split('_')[-1])+bit//2)%2)
    task.update(task_id='synthetic_m4_complete_source', task_type='multilabel', classes=columns,
                training_unit='slide', evaluation_unit='slide')
    task['multilabel'] = {'columns': columns,
        'label_provenance': 'synthetic correlated duplicate/complement endpoints for software checks; no molecular assay data',
        'empty_target_semantics': 'explicit measured synthetic binary controls only; missing real measurements are not negatives'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False));slides.to_csv(root/'slides.csv', index=False)
    if patches is not None:
        rng = np.random.default_rng(7303)
        features = pd.read_csv(root/'features.csv')
        for row in features.to_dict('records'):
            with h5py.File(root/row['path'], 'r+') as handle:
                del handle['features'];del handle['coords']
                handle['features'] = rng.normal(size=(patches, dimension)).astype(np.float32)
                handle['coords'] = np.column_stack([np.arange(patches)*256, np.zeros(patches)]).astype(np.int64)
    return root

"""Explicit synthetic patient bags, original full ConVL text/CONCH weights."""
from pathlib import Path

import h5py
import numpy as np
import yaml

from .convl_data import KIND, prepare
from .convl_source import original_concepts
from .data import Bundle
from .demo import create_demo
from .io import write_json


def create_convl_demo(output, weights, patients=20, cohort='BRCA', patches=None):
    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError('ConVL fixture requires the actual original pretrained CONCH checkpoint')
    original_concepts(cohort)
    if patches is not None and (type(patches) is not int or patches < 1):
        raise ValueError('ConVL fixture patch count must be a positive explicit integer')
    root = create_demo(output, patients=patients, dimension=512, survival=True)
    if patches is not None:
        random = np.random.default_rng(8200)
        for file in sorted((root/'features').glob('*.h5')):
            with h5py.File(file, 'w') as archive:
                archive['features'] = random.normal(size=(patches, 512)).astype(np.float32)
                archive['coords'] = np.column_stack([np.arange(patches)*256, np.zeros(patches)]).astype(np.int64)
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'checkpoint': str(weights),
        'alignment_policy': 'synthetic_native_dimension_fixture', 'concept_cohort': cohort,
        'feature_preprocessing': 'Synthetic full native512 Gaussian vectors, not actual WSI CONCH extraction',
        'feature_provenance': 'Explicit deterministic software fixture, not clinical tissue',
        'patient_order_provenance': 'Every registered slide in stable slide-ID order, each true patient once',
        'task_semantics': 'Synthetic patient survival, months/event1; no historical or clinical metric claim',
        'binning': {'policy': 'train_observed_event_quartiles',
            'provenance': 'Explicit unified preparation extension: three cuts from TRAIN observed events only; original external label generation not disclosed'}}
    write_json(root/'convl_metadata.json', metadata)
    prepare(Bundle(root), 'convl_metadata.json', 'convl/manifest.json')
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['encoders']['toy']['convl'] = {'manifest': 'convl/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

"""Synthetic full native-dimensional bags and actual pretrained language assets."""
from pathlib import Path

import h5py
import numpy as np
import yaml

from .data import Bundle
from .demo import create_demo
from .io import write_json
from .libramil_data import KIND, prepare


def create_libramil_demo(output, weights, patients=30, patches=257):
    weights = Path(weights).resolve()
    if not weights.is_file() or type(patches) is not int or patches < 1:
        raise ValueError('Libra-MIL fixture needs real pretrained CONCH weights and a positive patch count')
    root = create_demo(output, patients=patients, dimension=512, classes=3)
    task = yaml.safe_load((root/'task.yaml').read_text()); task['evaluation_unit'] = 'slide'
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    rng = np.random.default_rng(8400)
    for path in sorted((root/'features').glob('*.h5')):
        with h5py.File(path, 'w') as archive:
            archive['features'] = rng.normal(size=(patches, 512)).astype(np.float32)
            archive['coords'] = np.column_stack([np.arange(patches)*256, np.zeros(patches)]).astype(np.int64)
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True,
        'checkpoint': str(weights), 'class_order': task['classes'], 'text_mode': 'author_RCC',
        'alignment_policy': 'synthetic_native_dimension_fixture',
        'feature_preprocessing': 'Synthetic native512D Gaussian bags, not real CONCH feature extraction',
        'feature_provenance': 'Deterministic software controls only',
        'text_provenance': 'Actual complete46original RCC structural and3high-resolution descriptions, actual pretrained CONCH text encoding',
        'label_semantics': 'Synthetic class0/1/2 controls in source RCC prompt order; no clinical interpretation or measured RCC label'}
    write_json(root/'libramil_metadata.json', metadata)
    prepare(Bundle(root), 'libramil_metadata.json', 'libramil/manifest.json')
    task['encoders']['toy']['libramil'] = {'manifest': 'libramil/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root

"""Synthetic bags/prompts, with actual native pretrained FOCUS text weights."""
import os
from pathlib import Path

import yaml

from .data import Bundle
from .demo import create_demo
from .focus_data import KIND, prepare
from .focus_network import WEIGHTS_SHA
from .io import write_json


def create_focus_demo(output, patients=20, dimension=512, classes=2, checkpoint=None):
    checkpoint = Path(checkpoint or os.environ.get('MILPLUS_FOCUS_WEIGHTS',
        Path.home()/'.cache/milplus/assets/focus/conch.pth')).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('FOCUS demo requires actual pretrained CONCH weights; set MILPLUS_FOCUS_WEIGHTS')
    root = Path(create_demo(output, patients=patients, dimension=dimension, classes=classes))
    path = root/'task.yaml'; cfg = yaml.safe_load(path.read_text())
    cfg['evaluation_unit'] = 'slide'
    encoder = cfg['encoders']['toy']
    encoder.update(model='synthetic_full_dim_features', weights_sha256='synthetic',
        feature_preprocessing='Synthetic Gaussian frozen features, not real CONCH extraction')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    bundle = Bundle(root)
    folder = root/'focus_assets'; folder.mkdir()
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'classes': bundle.classes,
        'checkpoint': str(checkpoint), 'weights_sha256': WEIGHTS_SHA,
        'weights_provenance': 'Public author FOCUS README-linked CONCH checkpoint; original CONCH terms apply',
        'prompt_provenance': 'Synthetic fixed class descriptions for mechanical verification only',
        'feature_order_provenance': 'Deterministic synthetic row order, unchanged for selection and compression',
        'feature_preprocessing': encoder['feature_preprocessing'],
        'alignment_policy': 'cross_fm_projection_extension',
        'alignment_provenance': 'Explicit synthetic full-D projection, actual pretrained text; no natural visual-language alignment claim',
        'prompts': [{'role': role, 'class': name, 'text': f'Synthetic {role} magnification morphology of diagnostic {name} tissue, used only for software validation.'}
                    for role in ['low', 'high'] for name in bundle.classes]}
    write_json(folder/'text.json', metadata)
    manifest = prepare(bundle, folder/'text.json', folder/'manifest.json')
    cfg['encoders']['toy']['focus'] = {'manifest': str(Path(manifest).relative_to(root))}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root

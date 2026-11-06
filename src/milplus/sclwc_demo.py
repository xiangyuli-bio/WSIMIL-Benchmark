"""Synthetic normal/reference classes for software checks, never natural WSI."""
from pathlib import Path

import yaml

from .demo import create_demo


def create_sclwc_demo(output, patients=30, dimension=768, classes=3):
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    path = Path(root)/'task.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg.update(evaluation_unit='slide', sclwc={'negative_class': 'class_0',
        'negative_class_role': 'normal_tissue', 'provenance': 'synthetic class roles; no clinical normal annotation'})
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root

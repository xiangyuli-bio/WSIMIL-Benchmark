"""Synthetic frozen features for the explicitly registered INS input condition."""
from pathlib import Path

import yaml

from .demo import create_demo


def create_ins_demo(output, patients=20, dimension=1024):
    if type(dimension) is not int or dimension < 32 or dimension % 32:
        raise ValueError('INS synthetic features must satisfy the actual full-D source layout')
    root = create_demo(output, patients=patients, dimension=dimension, classes=2)
    path = Path(root)/'task.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['evaluation_unit'] = 'slide'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root

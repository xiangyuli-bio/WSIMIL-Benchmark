"""Complete pinned TRIAGE-MIL selection, hypergraph, network and epoch definitions."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple  # noqa: UP035 -- preserve original source annotation namespace

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn

from .upstream import load_definitions, verified_source

COMMIT = 'e97e52a32f1358974b0145532fdec5f5d75c6864'


def preprocessing():
    _, sha = verified_source('triagemil')
    if sha != COMMIT:
        raise ValueError('TRIAGE-MIL source differs from the reviewed original commit')
    core = load_definitions('triagemil', 'src/mass_core.py', ['MASSConfig', '_normalize01',
        'compute_feature_entropy', 'quality_filter_embeddings', 'filter_artifacts_comprehensive',
        'compute_mass_axis_scores', 'farthest_point_sampling'], {'dataclass': dataclass,
        'Optional': Optional, 'Tuple': Tuple, 'np': np, 'NearestNeighbors': NearestNeighbors})  # noqa: UP006 -- original annotations
    selector = load_definitions('triagemil', 'src/mass_selector.py',
        ['_budget_from_fractions', 'select_mass_tiles'], {'np': np, 'Dict': Dict, 'Optional': Optional,  # noqa: UP006 -- original annotations
        **{k: getattr(core, k) for k in ['MASSConfig', 'compute_mass_axis_scores',
                                       'farthest_point_sampling', 'quality_filter_embeddings']}})
    hypergraph = load_definitions('triagemil', 'src/precompute_hypergraphs.py',
        ['_safe_kmeans', 'build_semantic_hierarchical_hypergraph'], {'np': np, 'torch': torch,
        'Dict': Dict, 'Tuple': Tuple, 'KMeans': KMeans, 'NearestNeighbors': NearestNeighbors})  # noqa: UP006 -- original annotations
    return core, selector, hypergraph


def metric():
    tree = load_definitions('lifelines_metric', 'lifelines/utils/btree.py', ['_BTree'], {'np': np})._BTree
    return load_definitions('lifelines_metric', 'lifelines/utils/concordance.py',
        ['concordance_index', '_preprocess_scoring_data', '_concordance_summary_statistics',
         '_handle_pairs', '_concordance_ratio'], {'np': np, '_BTree': tree}).concordance_index


def trainer():
    _, sha = verified_source('triagemil')
    if sha != COMMIT:
        raise ValueError('TRIAGE-MIL source differs from its declared original commit')
    return load_definitions('triagemil', 'src/train_triage_mil.py',
        ['c_index', 'cox_ph_loss', 'HypergraphGatedMILSurvival', 'collate_bags', 'train_one_epoch', 'evaluate'],
        {'np': np, 'torch': torch, 'nn': nn, 'Optional': Optional, 'concordance_index': metric(),
         'pd': pd})

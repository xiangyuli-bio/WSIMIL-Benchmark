"""Frozen per-slide MamMIL topology assets, built from the pinned author's notebook."""
import ast
import hashlib
import importlib.metadata
import json
import os
import random
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric import utils

from .io import digest_file, digest_object
from .upstream import pinned_text, verified_source

ARRAYS = ('edge_index', 'edge_latent', 'edge_weight', 'dfs_postorder_index', 'dfs_preorder_index', 'bfs_levelorder_index')


def author_graph_functions():
    import nmslib
    root, sha = verified_source('mammil')
    notebook = json.loads(pinned_text(root, sha, 'h5toPyG.ipynb'))
    nodes = []
    names = {'Hnsw', 'cosine_similarity', 'pt2graph'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell.get('source', []))).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                    nodes.append(node)
    if {node.name for node in nodes} != names:
        raise ValueError('Pinned MamMIL graph notebook definitions changed')
    scope = {'nmslib': nmslib, 'torch': torch, 'np': np, 'nx': nx, 'utils': utils, 'random': random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'MamMIL/h5toPyG.ipynb', 'exec'), scope)  # noqa: S102 -- verified notebook AST
    original = scope['Hnsw']
    def serial_index(**kwargs):
        return original(**kwargs, index_params={'M': 16, 'post': 0, 'efConstruction': 400,
                                               'indexThreadQty': 1}, print_progress=False)
    scope['Hnsw'] = serial_index
    return scope


def build_graph(features, coords, neighbors=8, graph_seed=2024):
    x, xy = np.asarray(features, dtype=np.float32), np.asarray(coords, dtype=np.float32)
    if x.ndim != 2 or not len(x) or xy.shape != (len(x), 2) or not np.isfinite(x).all() or not np.isfinite(xy).all():
        raise ValueError('MamMIL graph requires finite aligned nonempty features and physical coordinates')
    if type(neighbors) is not int or neighbors < 1 or len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('MamMIL neighbors must be positive; duplicate physical coordinates are not supported')
    state = random.getstate()
    try:
        random.seed(graph_seed)
        graph = author_graph_functions()['pt2graph']({'features': x, 'coords': xy}, radius=min(neighbors+1, len(x)))
    finally:
        random.setstate(state)
    return {name: getattr(graph, name).cpu().numpy() for name in ARRAYS}


def asset_checksum(arrays):
    return digest_object({name: [list(arrays[name].shape), str(arrays[name].dtype),
                                hashlib.sha256(arrays[name].tobytes()).hexdigest()] for name in ARRAYS})


def validate_arrays(arrays, count, neighbors):
    edges = count*min(neighbors, count-1)
    for name in ('edge_index', 'edge_latent'):
        value = arrays[name]
        if value.dtype != np.int64 or value.shape != (2, edges) or (value < 0).any() or (value >= count).any():
            raise ValueError('Invalid MamMIL graph edges: '+name)
    weights = arrays['edge_weight']
    if weights.dtype != np.float32 or weights.shape != (edges,) or not np.isfinite(weights).all():
        raise ValueError('Invalid MamMIL graph weights')
    for name in ARRAYS[3:]:
        if arrays[name].dtype != np.int64 or arrays[name].shape != (count,) or not np.array_equal(np.sort(arrays[name]), np.arange(count)):
            raise ValueError('Invalid MamMIL graph traversal: '+name)


def graph_asset(features, coords, directory, neighbors=8, graph_seed=2024, expected=None, allow_build=False):
    x, xy = np.ascontiguousarray(features), np.ascontiguousarray(coords)
    metadata = {'implementation': digest_file(Path(__file__)),
                'source_commit': verified_source('mammil')[1], 'neighbors': neighbors, 'graph_seed': graph_seed,
                'index': {'M': 16, 'post': 0, 'efConstruction': 400, 'indexThreadQty': 1, 'ef': 90},
                'versions': {p: importlib.metadata.version(p) for p in ('nmslib', 'networkx', 'torch-geometric')},
                'features': [list(x.shape), str(x.dtype), hashlib.sha256(x.tobytes()).hexdigest()],
                'coordinates': [list(xy.shape), str(xy.dtype), hashlib.sha256(xy.tobytes()).hexdigest()]}
    key = digest_object(metadata)
    if expected is not None and expected['key'] != key:
        raise ValueError('MamMIL graph inputs or builder identity changed')
    path = Path(directory)/(key+'.npz')
    if not path.exists():
        if not allow_build:
            raise FileNotFoundError('Frozen MamMIL graph asset missing: '+str(path))
        arrays = build_graph(x, xy, neighbors, graph_seed)
        validate_arrays(arrays, len(x), neighbors)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez(temporary_path, **arrays, metadata=json.dumps(metadata, sort_keys=True),
                     checksum=asset_checksum(arrays), key=key)
            try:
                os.link(temporary_path, path)  # concurrent preparation never overwrites another frozen asset
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in ARRAYS}
        checksum = asset_checksum(arrays)
        if str(payload['key']) != key or json.loads(str(payload['metadata'])) != metadata or str(payload['checksum']) != checksum:
            raise ValueError('Invalid MamMIL graph cache identity or checksum')
    validate_arrays(arrays, len(x), neighbors)
    record = {'key': key, 'checksum': checksum}
    if expected is not None and record != expected:
        raise ValueError('Frozen MamMIL graph asset differs from checkpoint')
    return {name: torch.from_numpy(arrays[name].copy()) for name in ARRAYS}, record


def prepare_graphs(bundle, directory=None, neighbors=8, graph_seed=2024):
    """CPU preparation shared by the explicit CLI and training; no labels enter a graph."""
    directory = Path(directory or bundle.root/'.milplus-cache/mammil').resolve()
    manifest = {}
    for row in bundle.rows.to_dict('records'):
        x, coords = bundle.load(row, require_coords=True)
        _, record = graph_asset(x.numpy(), coords.numpy(), directory, neighbors, graph_seed, allow_build=True)
        manifest[row['slide_id']] = record
    return directory, manifest

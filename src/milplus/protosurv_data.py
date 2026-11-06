"""Input-bound physical ProtoSurv graphs and explicit tissue-category assets."""
import copy
import importlib.metadata
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .data import load_array, resolve
from .io import digest_file, digest_object, read_json, write_json
from .spatial_attention import checked_points, spatial_metadata
from .upstream import load_definitions

COMMIT = '551eaa3c0a2844e2c3557ef412d8a1fd2fec6e20'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'encoder': {k: v for k, v in bundle.encoder.items() if k != 'protosurv'}})


def feature_space(bundle):
    return {'fm': bundle.fm, 'dimension': bundle.dim, 'weights': bundle.encoder.get('weights'),
            'preprocessing': bundle.encoder.get('preprocessing'),
            'weights_sha256': bundle.encoder.get('weights_sha256')}


def schema_contract(schema, bundle):
    if (schema.get('schema_version') != 1 or schema.get('kind') != 'protosurv_spatial_graph' or
            type(schema.get('synthetic')) is not bool or
            schema['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('ProtoSurv requires its versioned spatial graph schema and matching synthetic marker')
    names = schema.get('tissue_classes')
    if (not isinstance(names, list) or len(names) < 2 or
            any(not isinstance(v, str) or not v.strip() for v in names) or len(set(names)) != len(names)):
        raise ValueError('ProtoSurv needs ordered unique tissue-category names')
    mode = schema.get('node_type_mode')
    if mode not in {'predicted', 'unknown_fixed_prototype'}:
        raise ValueError('ProtoSurv node_type_mode must be predicted or unknown_fixed_prototype')
    for key in ['provenance', 'graph_provenance']:
        if not isinstance(schema.get(key), str) or not schema[key].strip():
            raise ValueError('ProtoSurv asset schema requires '+key)
    if mode == 'predicted':
        classifier = schema.get('category_model', {})
        for key in ['name', 'revision', 'prompt_or_training_provenance']:
            if not isinstance(classifier.get(key), str) or not classifier[key].strip():
                raise ValueError('ProtoSurv needs the actual category model and prompt/training provenance')
        sha = classifier.get('weights_sha256')
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('ProtoSurv needs an actual tissue-classifier weight SHA256')
        if not schema['synthetic'] and sha == '0'*64:
            raise ValueError('Placeholder tissue-classifier hashes are not real-data provenance')
    spatial_metadata(bundle)
    return names, mode


def prototype_contract(entry, bundle):
    if not isinstance(entry, dict) or not isinstance(entry.get('provenance'), str) or not entry['provenance'].strip():
        raise ValueError('ProtoSurv prototypes require explicit population and feature-space provenance')
    if entry.get('feature_space') != feature_space(bundle):
        raise ValueError('ProtoSurv prototype origin must declare this exact FM/weights/preprocessing space, not merely its dimension')
    if entry.get('fit_subset') == 'train':
        ids = entry.get('fit_patients')
        train = set(bundle.rows.loc[bundle.rows.subset == 'train', 'global_patient_id'])
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or
                len(set(ids)) != len(ids) or not set(ids) <= train):
            raise ValueError('ProtoSurv TRAIN prototype subjects must be a nonempty unique TRAIN subset')
    elif entry.get('fit_subset') != 'external_declared':
        raise ValueError('ProtoSurv prototypes may be TRAIN-fitted or declared external assets, never VAL/TEST-fitted')


def spatial_edges(coords):
    """Author HNSW L2 query and frozen order; explicit single build thread."""
    import nmslib
    points = checked_points(coords)
    n = len(points)
    if n == 1:
        return np.empty((2, 0), dtype=np.int64)
    original = load_definitions('protosurv', 'data_preparation/graph_construction.py', ['Hnsw'],
                                {'nmslib': nmslib}).Hnsw
    parameters = {'M': 16, 'post': 0, 'efConstruction': 400, 'indexThreadQty': 1}
    model = original(space='l2', index_params=parameters, query_params={'ef': 90}, print_progress=False)
    model.fit(points)
    count = min(5, n)
    neighbors = [np.asarray(model.query(point, topn=count), dtype=np.int64)[1:] for point in points]
    if any(len(v) != count-1 for v in neighbors):
        raise ValueError('Author HNSW returned an incomplete neighborhood')
    return np.stack([np.repeat(np.arange(n), count-1), np.concatenate(neighbors)])


def graph_file(bundle, row, path, names, mode):
    x, coords = bundle.load(row, require_coords=True)
    points = checked_points(coords)
    with h5py.File(path, 'r') as stream:
        expected = {'coords', 'edge_index', 'node_types'} if mode == 'predicted' else {'coords', 'edge_index'}
        if set(stream) != expected:
            raise ValueError('ProtoSurv graph assets need exactly the declared coordinate/edge/category arrays')
        xy = np.asarray(stream['coords'])
        edges = np.asarray(stream['edge_index'])
        types = np.asarray(stream['node_types']) if mode == 'predicted' else np.full(len(x), -1, dtype=np.int64)
    if not np.array_equal(xy, points) or len(points) != len(x):
        raise ValueError('ProtoSurv graph/category rows must match the real primary coordinates exactly')
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in 'iu' or
            (edges.size and (edges.min() < 0 or edges.max() >= len(x))) or
            np.any(edges[0] == edges[1]) or len(np.unique(edges.T, axis=0)) != edges.shape[1]):
        raise ValueError('ProtoSurv graph has invalid, duplicate or self edges')
    degree = np.bincount(edges[0].astype(np.int64), minlength=len(x))
    if not np.all(degree == min(4, len(x)-1)):
        raise ValueError('ProtoSurv graph requires the declared four-neighbor author spatial construction')
    if (types.shape != (len(x),) or types.dtype.kind not in 'iu' or
            (mode == 'predicted' and (types.min() < 0 or types.max() >= len(names)))):
        raise ValueError('ProtoSurv tissue categories must be row-aligned integer IDs in the declared vocabulary')
    graph = Data(x=x, edge_index=torch.from_numpy(edges.astype(np.int64)),
                 centroid=coords, patch_classify_type=torch.from_numpy(types.astype(np.int64)))
    graph.event_time = float(row.get('time', 0.))
    graph.c = float(1-row.get('event', 0))
    graph.slide_id, graph.case_id = row['slide_id'], row['global_patient_id']
    return graph


def prepare(bundle, index, schema, output, prototypes=None, prototype_provenance=None):
    names, mode = schema_contract(schema, bundle)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('ProtoSurv preparation output must be a new directory')
    table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (set(table) != {'slide_id', 'path'} or table.slide_id.duplicated().any() or
            set(table.slide_id) != set(bundle.rows.slide_id)):
        raise ValueError('ProtoSurv category index must cover the exact task slides without labels or duplicates')
    paths = {r.slide_id: resolve(Path(index).resolve().parent, r.path).resolve()
             for r in table.itertuples() if r.path}
    if mode == 'predicted' and (len(paths) != len(table) or len(set(paths.values())) != len(paths)):
        raise ValueError('ProtoSurv predicted categories need distinct existing files for each slide')
    prototype_meta = None
    if prototypes is not None:
        array = np.asarray(load_array(Path(prototypes)))
        if array.shape != (len(names), bundle.dim) or array.dtype.kind != 'f' or not np.isfinite(array).all():
            raise ValueError('ProtoSurv prototypes must match tissue vocabulary and actual FM dimension')
        prototype_contract(prototype_provenance, bundle)
        prototype_meta = {'path': os.path.relpath(Path(prototypes).resolve(), output),
                          'sha256': digest_file(prototypes), **copy.deepcopy(prototype_provenance)}
    elif mode == 'unknown_fixed_prototype':
        raise ValueError('Unknown-category ProtoSurv route requires fixed prototypes in the registered FM space')
    output.mkdir(parents=True)
    entries = {}
    for row in bundle.rows.to_dict('records'):
        x, coords = bundle.load(row, require_coords=True)
        path = output/(digest_object(row['slide_id'])[:24]+'.h5')
        with h5py.File(path, 'w') as stream:
            stream['coords'] = coords.numpy()
            stream['edge_index'] = spatial_edges(coords)
            if mode == 'predicted':
                category = paths[row['slide_id']]
                types = np.asarray(load_array(category, 'node_types'))
                xy = np.asarray(load_array(category, 'coords'))
                if not np.array_equal(xy, coords.numpy()):
                    raise ValueError('ProtoSurv category model output coordinates differ from the primary patch order')
                stream['node_types'] = types
        graph_file(bundle, row, path, names, mode)
        entries[row['slide_id']] = {'path': path.name, 'sha256': digest_file(path), 'nodes': len(x),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'coordinate_sha256': digest_file(resolve(bundle.root, row.get('coords_path') or row['path']))}
    manifest = {'schema_version': 1, 'source_commit': COMMIT, 'encoder_sha256': encoder_identity(bundle),
                'schema': copy.deepcopy(schema), 'slides': entries, 'prototype': prototype_meta,
                'graph_builder': {'name': 'author_HNSW_L2_spatial', 'nmslib': importlib.metadata.version('nmslib'),
                    'neighbors': 4, 'small_graph_neighbors': 'min(4,N-1)',
                    'index_parameters': {'M': 16, 'post': 0, 'efConstruction': 400, 'indexThreadQty': 1}, 'ef': 90,
                    'latent_edges': 'not constructed; source model reads only spatial edge_index'}}
    write_json(output/'manifest.json', manifest)
    return output/'manifest.json'


class ProtoSurvAssets:
    def __init__(self, bundle):
        registration = bundle.encoder.get('protosurv')
        if not isinstance(registration, dict) or set(registration) != {'manifest'}:
            raise ValueError('ProtoSurv requires encoder.protosurv.manifest')
        self.bundle, self.path = bundle, resolve(bundle.root, registration['manifest']).resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('source_commit') != COMMIT or
                self.meta.get('encoder_sha256') != encoder_identity(bundle)):
            raise ValueError('ProtoSurv source or actual encoder identity differs')
        self.names, self.mode = schema_contract(self.meta['schema'], bundle)
        if set(self.meta['slides']) != set(bundle.rows.slide_id):
            raise ValueError('ProtoSurv graph manifest must cover the exact task slides')
        self.contract = digest_file(self.path)
        self.files = [self.path, *[resolve(self.path.parent, e['path']).resolve() for e in self.meta['slides'].values()]]
        if len(set(self.files)) != len(self.files):
            raise ValueError('ProtoSurv graph assets must be distinct per slide')
        if self.meta.get('prototype'):
            prototype_contract(self.meta['prototype'], bundle)
            self.files.append(resolve(self.path.parent, self.meta['prototype']['path']).resolve())
        elif self.mode == 'unknown_fixed_prototype':
            raise ValueError('ProtoSurv unknown-category route requires actual fixed prototypes')

    def prototype(self):
        entry = self.meta.get('prototype')
        if not entry:
            return None
        path = resolve(self.path.parent, entry['path'])
        if digest_file(path) != entry['sha256']:
            raise ValueError('ProtoSurv registered prototype contents changed')
        array = np.asarray(load_array(path))
        if array.shape != (len(self.names), self.bundle.dim) or array.dtype.kind != 'f' or not np.isfinite(array).all():
            raise ValueError('ProtoSurv prototypes do not match the registered FM and category vocabulary')
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())

    def load(self, row):
        if digest_file(self.path) != self.contract:
            raise ValueError('ProtoSurv manifest changed')
        entry = self.meta['slides'][row['slide_id']]
        path = resolve(self.path.parent, entry['path'])
        if (digest_file(path) != entry['sha256'] or
                digest_file(resolve(self.bundle.root, row['path'])) != entry['primary_sha256'] or
                digest_file(resolve(self.bundle.root, row.get('coords_path') or row['path'])) != entry['coordinate_sha256']):
            raise ValueError('ProtoSurv graph, categories, primary features or coordinates changed')
        graph = graph_file(self.bundle, row, path, self.names, self.mode)
        if len(graph.x) != entry['nodes']:
            raise ValueError('ProtoSurv registered node count changed')
        return graph

    def inventory(self, row, deep=False):
        entry = self.meta['slides'][row['slide_id']]
        if deep:
            self.load(row)
            self.prototype()
        return {'manifest_sha256': self.contract, 'graph_sha256': entry['sha256'], 'nodes': entry['nodes']}

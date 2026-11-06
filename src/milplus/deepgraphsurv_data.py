"""Frozen original feature-space graph sampling; full FM channels and true WSIs."""
import copy
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import resolve
from .deepgraphsurv_source import COMMIT, graph_functions
from .io import digest_file, digest_object, read_json, write_json


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'deepgraphsurv'}})


def schema_contract(bundle, schema):
    if (schema.get('schema_version') != 1 or schema.get('kind') != 'deepgraphsurv_feature_graph' or
            type(schema.get('synthetic')) is not bool or
            schema['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('DeepGraphSurv needs its original feature-graph schema and matching synthetic marker')
    if bundle.task_type != 'survival' or bundle.dim < 2:
        raise ValueError('DeepGraphSurv needs survival outcomes and at least two feature channels')
    for key in ['representation_provenance', 'selection_provenance']:
        if not isinstance(schema.get(key), str) or not schema[key].strip():
            raise ValueError('DeepGraphSurv needs '+key)
    if schema.get('representation') != 'frozen_features_full_dimension':
        raise ValueError('Declare the actual full-dimensional frozen feature representation; no implicit FM truncation or undocumented PCA')
    maximum, seed = schema.get('max_nodes', 500), schema.get('sampling_seed', 1)
    if type(maximum) is not int or maximum < 4 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('DeepGraphSurv max_nodes>=4 and a valid original NumPy sampling seed are required')
    order = schema.get('slide_order')
    if not isinstance(order, dict) or set(order) != {'train', 'val', 'test'}:
        raise ValueError('Declare exact original slide order in every clinical subset')
    for subset, ids in order.items():
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or
                set(ids) != set(bundle.rows.loc[bundle.rows.subset == subset, 'slide_id'])):
            raise ValueError('DeepGraphSurv slide order differs from the actual clinical split')
    return maximum, seed


def prepare(bundle, schema, output):
    maximum, seed = schema_contract(bundle, schema)
    source = graph_functions()
    out = Path(output).resolve()
    if out.exists():
        raise ValueError('DeepGraphSurv preparation needs a new output directory')
    out.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    rows = {r['slide_id']: r for r in bundle.rows.to_dict('records')}
    entries = {}
    for subset in ['train', 'val', 'test']:
        for slide in schema['slide_order'][subset]:
            row = rows[slide]
            x, _ = bundle.load(row)
            if len(x) < 4:
                raise ValueError('Original DeepGraphSurv filters slides with fewer than4patches; select the clinical task explicitly')
            # Original loader permutes even TEST, samples one graph per WSI and
            # retains min(n,max_nodes) actual nodes. Persist indices once.
            selected = rng.permutation(len(x))[:maximum].astype(np.int64)
            adjacency, _ = source.get_adjacency(x[selected].numpy())
            lap = source.compute_laplacian(adjacency, normalized=False)
            path = out/(digest_object(slide)[:24]+'.h5')
            with h5py.File(path, 'w') as stream:
                stream['indices'], stream['laplacian'] = selected, lap
            entries[slide] = {'path': path.name, 'sha256': digest_file(path),
                'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
                'raw_nodes': len(x), 'nodes': len(selected)}
    manifest = {'schema_version': 1, 'source_commit': COMMIT, 'schema': copy.deepcopy(schema),
                'encoder_sha256': encoder_identity(bundle), 'max_nodes': maximum,
                'slides': entries, 'scope': 'Original30percent feature-distance cutoff including diagonal zeros; strict edges, normalized adjacency with self-loops followed by unnormalized degree-minus-adjacency. No spatial kNN or Chebyshev rescaling.'}
    write_json(out/'manifest.json', manifest)
    return out/'manifest.json'


class DeepGraphSurvAssets:
    def __init__(self, bundle):
        entry = bundle.encoder.get('deepgraphsurv')
        if not isinstance(entry, dict) or set(entry) != {'manifest'}:
            raise ValueError('DeepGraphSurv requires encoder.deepgraphsurv.manifest')
        self.bundle, self.path = bundle, resolve(bundle.root, entry['manifest']).resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('source_commit') != COMMIT or
                self.meta.get('encoder_sha256') != encoder_identity(bundle)):
            raise ValueError('DeepGraphSurv graph source or full FM identity changed')
        maximum, _ = schema_contract(bundle, self.meta['schema'])
        if self.meta.get('max_nodes') != maximum or set(self.meta['slides']) != set(bundle.rows.slide_id):
            raise ValueError('DeepGraphSurv graph node budget or actual selected slide set differs')
        self.maximum, self.order = maximum, copy.deepcopy(self.meta['schema']['slide_order'])
        self.rows = {r['slide_id']: r for r in bundle.rows.to_dict('records')}
        self.contract = digest_file(self.path)
        self.files = [self.path, *[resolve(self.path.parent, e['path']).resolve() for e in self.meta['slides'].values()]]
        if len(set(self.files)) != len(self.files):
            raise ValueError('DeepGraphSurv requires distinct actual per-slide graph files')

    def load(self, row, rebuild=False):
        e = self.meta['slides'][row['slide_id']]
        path = resolve(self.path.parent, e['path'])
        if (digest_file(self.path) != self.contract or digest_file(path) != e['sha256'] or
                digest_file(resolve(self.bundle.root, row['path'])) != e['primary_sha256']):
            raise ValueError('DeepGraphSurv graph or original full-dimensional feature contents changed')
        x, _ = self.bundle.load(row)
        with h5py.File(path, 'r') as stream:
            if set(stream) != {'indices', 'laplacian'} or stream.attrs:
                raise ValueError('DeepGraphSurv graph asset structure changed')
            indices, lap = np.asarray(stream['indices']), np.asarray(stream['laplacian'])
        n = len(indices)
        if (indices.ndim != 1 or indices.dtype != np.int64 or len(x) != e['raw_nodes'] or
                n != e['nodes'] or n != min(len(x), self.maximum) or n < 4 or
                len(set(indices.tolist())) != n or indices.min() < 0 or indices.max() >= len(x) or
                lap.shape != (n, n) or lap.dtype != np.float32 or not np.isfinite(lap).all() or
                not np.allclose(lap, lap.T, rtol=0, atol=1e-7) or
                not np.allclose(lap.sum(0), 0, rtol=0, atol=2e-6)):
            raise ValueError('DeepGraphSurv raw selected indices or literal intrinsic Laplacian are invalid')
        if rebuild:
            source = graph_functions()
            adjacency, _ = source.get_adjacency(x[indices].numpy())
            expected = source.compute_laplacian(adjacency, normalized=False)
            if not np.array_equal(lap, expected):
                raise ValueError('DeepGraphSurv stored Laplacian differs from its unchanged author graph builder')
        return x[torch.from_numpy(indices)], torch.from_numpy(lap)

    def arrays(self, subset):
        ids = self.order[subset]
        data = np.full((len(ids), self.maximum, self.bundle.dim), -1, np.float32)
        lap = np.full((len(ids), self.maximum, self.maximum), -1, np.float32)
        sizes = np.empty(len(ids), np.int32)
        for i, slide in enumerate(ids):
            features, intrinsic = self.load(self.rows[slide])
            n = len(features)
            data[i, :n], lap[i, :n, :n], sizes[i] = features.numpy(), intrinsic.numpy(), n
        return data, lap, sizes, np.asarray(ids)

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        e = self.meta['slides'][row['slide_id']]
        return {'manifest_sha256': self.contract, 'asset_sha256': e['sha256'],
                'raw_nodes': e['raw_nodes'], 'selected_nodes': e['nodes'], 'max_nodes': self.maximum}

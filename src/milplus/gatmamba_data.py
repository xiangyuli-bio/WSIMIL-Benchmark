"""Frozen physical GAT-Mamba graphs and explicit six-class tissue information."""
import copy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json
from .spatial_attention import checked_points, spatial_metadata
from .upstream import load_definitions

COMMIT = '18979497990a66ffe6a75880f0336f85914552f0'
CLASSES = ['lepidic', 'acinar', 'papillary', 'micropapillary', 'solid', 'nontumor']


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'gatmamba'}})


def schema_contract(schema, bundle):
    if (not isinstance(schema, dict) or schema.get('schema_version') != 1 or
            schema.get('kind') != 'gatmamba_spatial_subtype_graph' or
            type(schema.get('synthetic')) is not bool or
            schema['synthetic'] != bool(bundle.config.get('synthetic', False)) or
            schema.get('tissue_classes') != CLASSES):
        raise ValueError('GAT-Mamba needs its versioned physical graph and ordered six-class LUAD schema')
    for name in ['provenance', 'graph_provenance', 'patch_order_provenance']:
        if not isinstance(schema.get(name), str) or not schema[name].strip():
            raise ValueError('GAT-Mamba assets require '+name)
    if schema.get('category_protocol') not in {'author_three_model_ensemble', 'explicit_external_classifier'}:
        raise ValueError('Declare original three-model ensemble or an explicitly different category classifier')
    classifier = schema.get('classifier', {})
    for name in ['ensemble_provenance', 'training_population']:
        if not isinstance(classifier.get(name), str) or not classifier[name].strip():
            raise ValueError('GAT-Mamba classifier requires '+name)
    if classifier.get('fit_subset') == 'train':
        identities = classifier.get('training_patients')
        train_ids = set(bundle.rows.loc[bundle.rows.subset == 'train', 'global_patient_id'])
        if (not isinstance(identities, list) or not identities or
                any(not isinstance(v, str) or not v for v in identities) or
                len(set(identities)) != len(identities) or not set(identities) <= train_ids):
            raise ValueError('GAT-Mamba fitted classifier subjects must be a unique nonempty TRAIN-only subset')
    elif classifier.get('fit_subset') != 'external_declared':
        raise ValueError('GAT-Mamba classifier fitting must be TRAIN-only or explicitly declared external, never VAL/TEST')
    models = classifier.get('models')
    if not isinstance(models, list) or not models or (
            schema['category_protocol'] == 'author_three_model_ensemble' and len(models) != 3):
        raise ValueError('GAT-Mamba category model identities are incomplete')
    for model in models:
        if not isinstance(model, dict) or any(not isinstance(model.get(k), str) or not model[k].strip()
                                             for k in ['name', 'revision']):
            raise ValueError('GAT-Mamba needs actual category model names and fixed revisions')
        sha = model.get('weights_sha256')
        if (not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                or (not schema['synthetic'] and sha == '0'*64)):
            raise ValueError('GAT-Mamba requires real classifier weight SHA256 values')
    if len({(m['name'], m['revision']) for m in models}) != len(models):
        raise ValueError('Duplicate GAT-Mamba classifier identities')
    meta = spatial_metadata(bundle)
    if meta['unit'] != 'level0_pixels' or meta.get('coordinate_order') != 'xy':
        raise ValueError('GAT-Mamba original positional encoding requires declared level0-pixel xy coordinates')


def source_xy(bundle, coords):
    points = checked_points(coords).astype(np.float64)
    meta = spatial_metadata(bundle)
    if meta['origin'] == 'center':
        points = points-meta['patch_size']/2
    if (points < 0).any() or not np.equal(points, np.floor(points)).all():
        raise ValueError('GAT-Mamba needs real nonnegative integer top-left level0 positions, without interpolation')
    return points


def category_table():
    fn = load_definitions('gatmamba', 'modeling/create_dataset_uni.py', ['subtype_subtype_edge'], {}).subtype_subtype_edge
    table = np.empty((6, 6), dtype=np.int64)
    for a in range(6):
        for b in range(6):
            value = fn([a, b])
            if value == 'none':
                value = fn([b, a])
            if type(value) is not int or not 0 <= value <= 20:
                raise ValueError('Original GAT-Mamba category pair table differs')
            table[a, b] = value
    return table


def build_arrays(x, points, types, pair_types):
    order = np.lexsort((points[:, 0], points[:, 1])).astype(np.int64)
    xy = points[order]
    features = x.detach().cpu().numpy()[order].astype(np.float64)
    labels = types[order]
    if len(x) == 1:
        return order, np.empty((2, 0), dtype=np.int64), np.empty((0, 3), dtype=np.float32)
    adjacency = kneighbors_graph(xy, min(8, len(x)-1), mode='connectivity', include_self=False).tocoo()
    edges = np.stack([adjacency.row, adjacency.col]).astype(np.int64)
    a, b = edges
    attrs = np.column_stack([pair_types[labels[a], labels[b]],
        np.linalg.norm(xy[a]-xy[b], axis=1), np.linalg.norm(features[a]-features[b], axis=1)])
    edges, attrs = to_undirected(torch.from_numpy(edges), torch.tensor(attrs, dtype=torch.float32), reduce='mean')
    return order, edges.numpy(), attrs.numpy()


def prepare(bundle, index, schema, output):
    schema_contract(schema, bundle)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('GAT-Mamba graph preparation requires a new output directory')
    entries = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (set(entries) != {'slide_id', 'path'} or entries.slide_id.duplicated().any() or
            entries.path.eq('').any() or set(entries.slide_id) != set(bundle.rows.slide_id)):
        raise ValueError('GAT-Mamba category index must cover exactly the task slides')
    files = {r.slide_id: resolve(Path(index).resolve().parent, r.path).resolve() for r in entries.itertuples()}
    if len(set(files.values())) != len(files):
        raise ValueError('GAT-Mamba category files must be distinct for each WSI')
    output.mkdir(parents=True)
    slides = {}
    # Verify and load the original21class mapping once, not once per WSI.
    pair_types = category_table()
    for row in bundle.rows.to_dict('records'):
        x, coords = bundle.load(row, require_coords=True)
        points = source_xy(bundle, coords)
        with h5py.File(files[row['slide_id']], 'r') as stream:
            if set(stream) != {'coords', 'node_types'} or not np.array_equal(stream['coords'][:], coords.numpy()):
                raise ValueError('GAT-Mamba category coordinates must equal primary coordinates in original patch order')
            types = np.asarray(stream['node_types'])
        if types.shape != (len(x),) or types.dtype.kind not in 'iu' or types.min() < 0 or types.max() >= 6:
            raise ValueError('GAT-Mamba tissue types must be row-aligned integer IDs from the six-class vocabulary')
        order, edges, attrs = build_arrays(x, points, types, pair_types)
        path = output/(digest_object(row['slide_id'])[:24]+'.h5')
        with h5py.File(path, 'w') as stream:
            stream['coords'], stream['node_types'], stream['order'] = coords.numpy(), types, order
            stream['edge_index'], stream['edge_attr'] = edges, attrs
        slides[row['slide_id']] = {'path': path.name, 'sha256': digest_file(path), 'nodes': len(x),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'coordinate_sha256': digest_file(resolve(bundle.root, row.get('coords_path') or row['path']))}
    manifest = {'schema_version': 1, 'source_commit': COMMIT, 'encoder_sha256': encoder_identity(bundle),
        'schema': copy.deepcopy(schema), 'slides': slides,
        'graph_builder': {'name': 'source_sklearn8NN_PyG_undirected_mean',
            'neighbors': 8, 'small_graph_policy': 'min(8,N-1); source execution still requires >=10nodes per WSI',
            'order': 'real top-left level0 y/x row-major per slide, realizing source natural r/c tile-name ordering',
            'edge_features': ['original21unordered_subtype_pair', 'level0_pixel_L2', 'full_registered_FM_L2'],
            'distance_calculation': 'float64 then original float32 PyG edge attributes'}}
    write_json(output/'manifest.json', manifest)
    return output/'manifest.json'


class GATMambaAssets:
    def __init__(self, bundle):
        registration = bundle.encoder.get('gatmamba')
        if not isinstance(registration, dict) or set(registration) != {'manifest'}:
            raise ValueError('GAT-Mamba requires encoder.gatmamba.manifest')
        self.bundle, self.path = bundle, resolve(bundle.root, registration['manifest']).resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('source_commit') != COMMIT or
                self.meta.get('encoder_sha256') != encoder_identity(bundle) or
                set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('GAT-Mamba graph source, FM space or exact slide inventory differs')
        schema_contract(self.meta['schema'], bundle)
        self.contract = digest_file(self.path)
        self.files = [self.path, *[resolve(self.path.parent, e['path']).resolve() for e in self.meta['slides'].values()]]
        if len(set(self.files)) != len(self.files):
            raise ValueError('GAT-Mamba graph assets must be distinct per WSI')
        self.pairs = category_table()

    def load(self, row):
        if digest_file(self.path) != self.contract:
            raise ValueError('GAT-Mamba graph manifest changed')
        entry = self.meta['slides'][row['slide_id']]
        path = resolve(self.path.parent, entry['path'])
        if (digest_file(path) != entry['sha256'] or
                digest_file(resolve(self.bundle.root, row['path'])) != entry['primary_sha256'] or
                digest_file(resolve(self.bundle.root, row.get('coords_path') or row['path'])) != entry['coordinate_sha256']):
            raise ValueError('GAT-Mamba graph, tissue assignments, coordinates or FM features changed')
        x, coords = self.bundle.load(row, require_coords=True)
        points = source_xy(self.bundle, coords)
        with h5py.File(path, 'r') as stream:
            if set(stream) != {'coords', 'node_types', 'order', 'edge_index', 'edge_attr'}:
                raise ValueError('GAT-Mamba graph needs exactly its declared physical arrays')
            xy, types, order, edges, attrs = [np.asarray(stream[k]) for k in
                                            ['coords', 'node_types', 'order', 'edge_index', 'edge_attr']]
        expected_order = np.lexsort((points[:, 0], points[:, 1]))
        if (entry['nodes'] != len(x) or not np.array_equal(xy, coords.numpy()) or
                order.shape != (len(x),) or order.dtype.kind not in 'iu' or not np.array_equal(order, expected_order) or
                types.shape != (len(x),) or types.dtype.kind not in 'iu' or types.min() < 0 or types.max() >= 6):
            raise ValueError('GAT-Mamba graph rows, ordered patches or category assignments differ')
        if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in 'iu' or
                (edges.size and (edges.min() < 0 or edges.max() >= len(x))) or
                np.any(edges[0] == edges[1]) or len(np.unique(edges.T, axis=0)) != edges.shape[1] or
                attrs.shape != (edges.shape[1], 3) or attrs.dtype.kind != 'f' or
                not np.isfinite(attrs).all() or (attrs < 0).any()):
            raise ValueError('Invalid GAT-Mamba physical graph or edge features')
        if not np.array_equal(attrs[:, 0], self.pairs[types[order][edges[0]], types[order][edges[1]]]):
            raise ValueError('GAT-Mamba semantic edge categories do not match the real tissue assignments')
        pairs = {tuple(e) for e in edges.T}
        if any((b, a) not in pairs for a, b in pairs):
            raise ValueError('GAT-Mamba source graphs must be undirected')
        degree = np.bincount(edges[0], minlength=len(x))
        if (degree < min(8, len(x)-1)).any():
            raise ValueError('GAT-Mamba physical graph has insufficient8NN neighbors')
        features = torch.cat([torch.tensor(points[order], dtype=torch.float32), x[order]], 1)
        return Data(x=features, edge_index=torch.from_numpy(edges.astype(np.int64)),
                    edge_attr=torch.tensor(attrs, dtype=torch.float32))

    def patient(self, rows):
        if not rows or len({r['global_patient_id'] for r in rows}) != 1:
            raise ValueError('GAT-Mamba patient graphs require a nonempty single-patient set of slides')
        rows = sorted(rows, key=lambda r: r['slide_id'])
        batched = Batch.from_data_list([self.load(row) for row in rows])
        graph = Data(x=batched.x, edge_index=batched.edge_index, edge_attr=batched.edge_attr)
        graph.event = torch.tensor([rows[0]['event']], dtype=torch.float32)
        graph.days = torch.tensor([rows[0]['time']], dtype=torch.float32)
        for name in ['patient_id', 'global_patient_id']:
            setattr(graph, name, rows[0][name])
        graph.slide_id = rows[0]['slide_id']
        graph.source_slides = ';'.join(r['slide_id'] for r in rows)
        return graph

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        return {'manifest_sha256': self.contract, 'graph_sha256': self.meta['slides'][row['slide_id']]['sha256'],
                'nodes': self.meta['slides'][row['slide_id']]['nodes']}

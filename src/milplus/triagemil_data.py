"""Actual MASS-selected patch indices and semantic hierarchical hypergraph assets."""
import copy
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json
from .spatial_attention import checked_points, spatial_metadata
from .triagemil_source import COMMIT, preprocessing


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'triagemil'}})


def schema_contract(schema, bundle, source=None):
    if (schema.get('schema_version') != 1 or schema.get('kind') != 'triagemil_mass_hypergraph' or
            type(schema.get('synthetic')) is not bool or
            schema['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('TRIAGE-MIL requires its versioned MASS/hypergraph schema and matching synthetic marker')
    if bundle.task_type != 'survival' or bundle.rows.groupby('global_patient_id').size().max() != 1:
        raise ValueError('TRIAGE-MIL needs survival labels and one explicitly selected WSI per patient')
    spatial_metadata(bundle)
    for key in ['selection_provenance', 'graph_provenance']:
        if not isinstance(schema.get(key), str) or not schema[key].strip():
            raise ValueError('TRIAGE-MIL requires '+key)
    orders = schema.get('patient_order')
    if not isinstance(orders, dict) or set(orders) != {'train', 'val', 'test'}:
        raise ValueError('TRIAGE-MIL requires the declared source patient order in all clinical subsets')
    for subset, ids in orders.items():
        expected = set(bundle.rows.loc[bundle.rows.subset == subset, 'global_patient_id'])
        if (not isinstance(ids, list) or not ids or any(not isinstance(g, str) for g in ids) or
                len(ids) != len(set(ids)) or set(ids) != expected):
            raise ValueError('TRIAGE-MIL patient orders must match the exact clinical split without leakage or duplicates')
    source = source or preprocessing()
    config_type = source[0].MASSConfig
    settings = schema.get('mass', {})
    if not isinstance(settings, dict) or set(settings)-{f.name for f in fields(config_type)}:
        raise ValueError('Unknown TRIAGE-MIL MASS settings')
    cfg = config_type(**settings)
    params = asdict(cfg)
    fractions = [params[k] for k in ['mh_fraction', 'tr_fraction', 'mi_fraction', 'td_fraction']]
    if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in params.values()) or any(not 0 <= v <= 1 for v in fractions) or
            not math.isclose(sum(fractions), 1., abs_tol=1e-6) or type(cfg.mi_neighbors) is not int or
            cfg.mi_neighbors < 1 or not 0 <= cfg.magnitude_percentile_low < cfg.magnitude_percentile_high <= 100 or
            not 0 <= cfg.entropy_percentile_low <= 100 or cfg.outlier_z_threshold <= 0 or
            not 0 <= cfg.min_quality_threshold <= 1 or cfg.eps <= 0):
        raise ValueError('Invalid TRIAGE-MIL original MASS fractions/quality/neighbor settings')
    budget = schema.get('selection_budget', 4096)
    if type(budget) is not int or budget < 1:
        raise ValueError('TRIAGE-MIL requested selection budget must be positive')
    hyper = {'k_intra': 10, 'k_inter': 5, 'tiles_per_super': 12, 'random_state': 35}
    supplied = schema.get('hypergraph', {})
    if not isinstance(supplied, dict) or set(supplied)-set(hyper):
        raise ValueError('Unknown TRIAGE-MIL semantic hypergraph settings')
    hyper.update(supplied)
    if (any(type(v) is not int or v < 0 for v in hyper.values()) or
            hyper['tiles_per_super'] < 1 or hyper['random_state'] >= 2**32):
        raise ValueError('Invalid TRIAGE-MIL source hypergraph parameters')
    return cfg, budget, hyper


def original_arrays(x, coords, cfg, budget, hyper, source):
    points = checked_points(coords)
    indices, labels, proxy, selection = source[1].select_mass_tiles(x.numpy(), points, budget, cfg, verbose=False)
    H, hierarchy = source[2].build_semantic_hierarchical_hypergraph(
        x.numpy()[indices], points[indices], labels, **hyper)
    H = H.coalesce()
    return {'indices': indices, 'mass_labels': labels, 'proxy_scores': proxy,
        'H_indices': H.indices().numpy(), 'H_values': H.values().numpy(),
        'H_shape': np.asarray(H.shape, dtype=np.int64)}, {'selection': selection, 'hierarchy': hierarchy}


def prepare(bundle, schema, output):
    source = preprocessing()
    cfg, budget, hyper = schema_contract(schema, bundle, source)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('TRIAGE-MIL preparation requires a new output directory')
    output.mkdir(parents=True)
    entries = {}
    for row in bundle.rows.to_dict('records'):
        x, coords = bundle.load(row, require_coords=True)
        arrays, meta = original_arrays(x, coords, cfg, budget, hyper, source)
        path = output/(digest_object(row['slide_id'])[:24]+'.h5')
        with h5py.File(path, 'w') as stream:
            for k, value in arrays.items():stream[k] = value
            stream['coords'] = coords.numpy()
            stream.attrs['metadata_json'] = json.dumps(meta, allow_nan=False)
        entries[row['slide_id']] = {'path': path.name, 'sha256': digest_file(path), 'raw_nodes': len(x),
            'selected_nodes': len(arrays['indices']), 'hyperedges': int(arrays['H_shape'][1]),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'coordinate_sha256': digest_file(resolve(bundle.root, row.get('coords_path') or row['path']))}
    manifest = {'schema_version': 1, 'source_commit': COMMIT, 'encoder_sha256': encoder_identity(bundle),
        'schema': copy.deepcopy(schema), 'settings': {'mass': asdict(cfg), 'selection_budget': budget,
            'hypergraph': hyper}, 'slides': entries,
        'scope': 'Original per-WSI unsupervised MASS quality/four-axis selection and semantic hypergraphs; no cross-subset population fit'}
    write_json(output/'manifest.json', manifest)
    return output/'manifest.json'


class TRIAGEMILAssets:
    def __init__(self, bundle):
        entry = bundle.encoder.get('triagemil')
        if not isinstance(entry, dict) or set(entry) != {'manifest'}:
            raise ValueError('TRIAGE-MIL requires encoder.triagemil.manifest')
        self.bundle, self.path = bundle, resolve(bundle.root, entry['manifest']).resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('source_commit') != COMMIT or
                self.meta.get('encoder_sha256') != encoder_identity(bundle)):
            raise ValueError('TRIAGE-MIL original source or actual FM identity changed')
        cfg, budget, hyper = schema_contract(self.meta['schema'], bundle)
        if self.meta.get('settings') != {'mass': asdict(cfg), 'selection_budget': budget, 'hypergraph': hyper}:
            raise ValueError('TRIAGE-MIL frozen MASS/hypergraph parameters changed')
        if set(self.meta['slides']) != set(bundle.rows.slide_id):
            raise ValueError('TRIAGE-MIL hypergraphs must cover exactly the selected task WSIs')
        self.contract = digest_file(self.path)
        self.order = copy.deepcopy(self.meta['schema']['patient_order'])
        self.by_patient = {r['global_patient_id']: r for r in bundle.rows.to_dict('records')}
        self.files = [self.path, *[resolve(self.path.parent, e['path']).resolve() for e in self.meta['slides'].values()]]
        if len(set(self.files)) != len(self.files):
            raise ValueError('TRIAGE-MIL needs distinct actual selected-WSI hypergraph files')

    def load(self, row, rebuild=False):
        e = self.meta['slides'][row['slide_id']]
        path = resolve(self.path.parent, e['path'])
        if (digest_file(self.path) != self.contract or digest_file(path) != e['sha256'] or
                digest_file(resolve(self.bundle.root, row['path'])) != e['primary_sha256'] or
                digest_file(resolve(self.bundle.root, row.get('coords_path') or row['path'])) != e['coordinate_sha256']):
            raise ValueError('TRIAGE-MIL MASS/hypergraph/primary feature or coordinate contents changed')
        x, coords = self.bundle.load(row, require_coords=True)
        checked_points(coords)
        names = ['indices', 'mass_labels', 'proxy_scores', 'H_indices', 'H_values', 'H_shape']
        with h5py.File(path, 'r') as stream:
            if set(stream) != set(names+['coords']) or set(stream.attrs) != {'metadata_json'}:
                raise ValueError('TRIAGE-MIL assets must contain the complete declared MASS and sparse-hypergraph arrays')
            arrays = {k: np.asarray(stream[k]) for k in names}
            if not np.array_equal(np.asarray(stream['coords']), coords.numpy()):
                raise ValueError('TRIAGE-MIL hypergraph rows differ from original physical coordinates')
            meta = json.loads(stream.attrs['metadata_json'])
        indices, labels, proxy = [arrays[k] for k in names[:3]]
        n = len(indices)
        if (len(x) != e['raw_nodes'] or n != e['selected_nodes'] or indices.ndim != 1 or n < 1 or
                indices.dtype.kind not in 'iu' or len(set(indices.tolist())) != n or
                indices.min() < 0 or indices.max() >= len(x) or labels.shape != (n,) or
                labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() > 3 or
                proxy.shape != (n,) or proxy.dtype.kind != 'f' or not np.isfinite(proxy).all() or
                np.any(proxy < 0) or np.any(proxy > 1)):
            raise ValueError('TRIAGE-MIL selected raw indices, semantic labels or proxy scores are invalid')
        shape, ij, values = arrays['H_shape'], arrays['H_indices'], arrays['H_values']
        if (shape.shape != (2,) or shape.dtype.kind not in 'iu' or tuple(shape) != (n, e['hyperedges']) or
                shape[1] < 1 or ij.ndim != 2 or ij.shape[0] != 2 or ij.dtype.kind not in 'iu' or
                values.shape != (ij.shape[1],) or values.dtype.kind != 'f' or not np.equal(values, 1).all() or
                not ij.size or ij.min() < 0 or ij[0].max() >= n or ij[1].max() >= shape[1] or
                len(np.unique(ij.T, axis=0)) != ij.shape[1] or len(set(ij[0].tolist())) != n or
                len(set(ij[1].tolist())) != shape[1]):
            raise ValueError('TRIAGE-MIL needs a nonempty binary sparse incidence matrix covering all actual nodes and hyperedges')
        if (meta['selection']['selected_tiles'] != n or meta['selection']['total_tiles'] != len(x) or
                meta['hierarchy']['n_tiles'] != n or meta['hierarchy']['n_hyperedges'] != shape[1] or
                meta['selection']['axis_counts'] != {k: int((labels == i).sum())
                                                     for i, k in enumerate(['MH', 'TR', 'MI', 'TD'])}):
            raise ValueError('TRIAGE-MIL original preprocessing metadata is inconsistent with its actual arrays')
        if rebuild:
            source = preprocessing()
            cfg, budget, hyper = schema_contract(self.meta['schema'], self.bundle, source)
            expected, expected_meta = original_arrays(x, coords, cfg, budget, hyper, source)
            if any(not np.array_equal(arrays[k], expected[k]) for k in names) or meta != expected_meta:
                raise ValueError('TRIAGE-MIL frozen selection/hypergraph differs from the actual original source builder')
        H = torch.sparse_coo_tensor(torch.from_numpy(ij.astype(np.int64)),
            torch.from_numpy(values.astype(np.float32)), tuple(shape)).coalesce()
        return {'id': row['global_patient_id'], 'features': x[torch.from_numpy(indices.astype(np.int64))],
            'H': H, 'coords': coords[torch.from_numpy(indices.astype(np.int64))],
            'mass_labels': torch.from_numpy(labels.astype(np.int64)),
            'risk_scores': torch.from_numpy(proxy.astype(np.float32)),
            'time': torch.tensor(row['time'], dtype=torch.float32),
            'event': torch.tensor(row['event'], dtype=torch.long)}

    def inventory(self, row, deep=False):
        if deep:self.load(row)
        e = self.meta['slides'][row['slide_id']]
        return {'manifest_sha256': self.contract, 'asset_sha256': e['sha256'],
            'raw_nodes': e['raw_nodes'], 'selected_nodes': e['selected_nodes'], 'hyperedges': e['hyperedges']}

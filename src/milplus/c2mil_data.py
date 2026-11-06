"""Bound spatial graphs, actual whole-slide thumbnails and TRAIN reference population."""
import copy
import importlib.metadata
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .c2mil_source import COMMIT
from .data import load_array, resolve
from .io import digest_file, digest_object, read_json, write_json
from .spatial_attention import checked_points, spatial_metadata
from .upstream import load_definitions


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'c2mil'}})


def feature_space(bundle):
    return {'fm': bundle.fm, 'dimension': bundle.dim, 'weights': bundle.encoder.get('weights'),
        'weights_sha256': bundle.encoder.get('weights_sha256')}


def validate_schema(schema, bundle):
    if (schema.get('schema_version') != 1 or schema.get('kind') != 'c2mil_graph_thumbnail_reference' or
            type(schema.get('synthetic')) is not bool or
            schema['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('C²MIL requires its versioned schema and matching synthetic marker')
    if bundle.task_type != 'survival' or bundle.rows.groupby('global_patient_id').size().max() != 1:
        raise ValueError('C²MIL requires survival labels and one explicitly selected diagnostic WSI per patient')
    spatial_metadata(bundle)
    thumbnail = schema.get('thumbnail', {})
    if thumbnail.get('feature_space') != feature_space(bundle):
        raise ValueError('C²MIL thumbnails must use the declared patch FM/weights and dimension')
    for key in ['selection_provenance', 'graph_provenance']:
        if not isinstance(schema.get(key), str) or not schema[key].strip():
            raise ValueError('C²MIL requires '+key)
    for key in ['image_generation', 'encoder_preprocessing', 'provenance']:
        if not isinstance(thumbnail.get(key), str) or not thumbnail[key].strip():
            raise ValueError('C²MIL requires actual whole-slide thumbnail '+key)
    if thumbnail.get('kind') != 'whole_slide_image_embedding':
        raise ValueError('C²MIL requires an actual thumbnail embedding, never a mean of patch features')
    orders = schema.get('patient_order')
    if not isinstance(orders, dict) or set(orders) != {'train', 'val', 'test'}:
        raise ValueError('C²MIL requires explicit original patient order in all three split lists')
    for subset, ids in orders.items():
        expected = set(bundle.rows.loc[bundle.rows.subset == subset, 'global_patient_id'])
        if (not isinstance(ids, list) or any(not isinstance(v, str) for v in ids) or
                len(ids) != len(set(ids)) or set(ids) != expected):
            raise ValueError('C²MIL patient order must exactly match its split, without leakage or duplicates')
    if len(orders['train']) < 6:
        raise ValueError('C²MIL adaptive clustering requires at least six distinct TRAIN patients')


def spatial_edges(coords):
    import nmslib
    points = checked_points(coords)
    if len(points) == 1:
        return np.empty((2, 0), dtype=np.int64)
    author = load_definitions('c2mil', 'data_preprocess/to_Graph.py', ['Hnsw'], {'nmslib': nmslib}).Hnsw
    model = author(space='l2', index_params={'M': 16, 'post': 0, 'efConstruction': 400,
        'indexThreadQty': 1}, query_params={'ef': 90}, print_progress=False)
    model.fit(points)
    k = min(9, len(points))
    neighbors = [np.asarray(model.query(v, topn=k), dtype=np.int64)[1:] for v in points]
    return np.stack([np.repeat(np.arange(len(points)), k-1), np.concatenate(neighbors)])


def thumbnail_array(path, dimension):
    x = np.asarray(load_array(path))
    if x.shape == (dimension,):
        x = x[None]
    if x.shape != (1, dimension) or x.dtype.kind != 'f' or not np.isfinite(x).all():
        raise ValueError('C²MIL thumbnail must contain exactly one finite full-dimensional embedding')
    return torch.from_numpy(x.astype(np.float32).copy())


def prepare(bundle, index, schema, output, sampling_rng_state=None):
    validate_schema(schema, bundle)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('C²MIL preparation requires a new output directory')
    table = pd.read_csv(index, dtype=str, keep_default_na=False)
    if (set(table) != {'slide_id', 'thumbnail_path'} or table.slide_id.duplicated().any() or
            set(table.slide_id) != set(bundle.rows.slide_id) or not table.thumbnail_path.all()):
        raise ValueError('C²MIL thumbnail index must cover the exact selected task slides')
    thumbs = {r.slide_id: resolve(Path(index).resolve().parent, r.thumbnail_path).resolve()
              for r in table.itertuples()}
    if len(set(thumbs.values())) != len(thumbs):
        raise ValueError('C²MIL requires a distinct actual thumbnail per selected WSI')
    generator = torch.Generator().manual_seed(98)
    if sampling_rng_state is not None:
        state = np.asarray(load_array(Path(sampling_rng_state)))
        if state.ndim != 1 or state.dtype != np.uint8:
            raise ValueError('C²MIL sampling RNG state must be a saved CPU Torch uint8 generator state')
        generator.set_state(torch.from_numpy(state.copy()))
    rng_before = generator.get_state().clone()
    # Draw once per TRAIN patient, exactly in the supplied original fold order.
    by_patient = {r['global_patient_id']: r for r in bundle.rows.to_dict('records')}
    indices = {}
    for gid in schema['patient_order']['train']:
        x, _ = bundle.load(by_patient[gid], require_coords=True)
        indices[gid] = torch.randperm(len(x), generator=generator)[:1500].numpy()
    output.mkdir(parents=True)
    entries = {}
    for row in bundle.rows.to_dict('records'):
        x, coords = bundle.load(row, require_coords=True)
        thumbnail_array(thumbs[row['slide_id']], bundle.dim)
        path = output/(digest_object(row['slide_id'])[:24]+'.h5')
        with h5py.File(path, 'w') as stream:
            stream['coords'], stream['edge_index'] = coords.numpy(), spatial_edges(coords)
            if row['global_patient_id'] in indices:
                stream['reference_indices'] = indices[row['global_patient_id']]
        entries[row['slide_id']] = {'path': path.name, 'sha256': digest_file(path), 'nodes': len(x),
            'primary_sha256': digest_file(resolve(bundle.root, row['path'])),
            'coordinate_sha256': digest_file(resolve(bundle.root, row.get('coords_path') or row['path'])),
            'thumbnail_path': os.path.relpath(thumbs[row['slide_id']], output),
            'thumbnail_sha256': digest_file(thumbs[row['slide_id']])}
    np.save(output/'sampling_rng_before.npy', rng_before.numpy())
    np.save(output/'sampling_rng_after.npy', generator.get_state().numpy())
    manifest = {'schema_version': 1, 'source_commit': COMMIT, 'encoder_sha256': encoder_identity(bundle),
        'schema': copy.deepcopy(schema), 'slides': entries,
        'sampling': {'patches': 1500, 'padding': 'mean_of_selected_rows', 'seed': 98,
            'mode': 'provided_original_fold_rng' if sampling_rng_state else 'fresh_single_split_seed98',
            'before_sha256': digest_file(output/'sampling_rng_before.npy'),
            'after_sha256': digest_file(output/'sampling_rng_after.npy')},
        'graph_builder': {'name': 'author_HNSW_L2_spatial', 'neighbors': 8, 'small_graph': 'min(8,N-1)',
            'nmslib': importlib.metadata.version('nmslib'), 'single_build_thread': True,
            'latent_graph': 'unused by original source forward; not constructed'}}
    write_json(output/'manifest.json', manifest)
    return output/'manifest.json'


class C2MILAssets:
    def __init__(self, bundle):
        entry = bundle.encoder.get('c2mil')
        if not isinstance(entry, dict) or set(entry) != {'manifest'}:
            raise ValueError('C²MIL requires encoder.c2mil.manifest')
        self.bundle, self.path = bundle, resolve(bundle.root, entry['manifest']).resolve()
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('source_commit') != COMMIT or
                self.meta.get('encoder_sha256') != encoder_identity(bundle)):
            raise ValueError('C²MIL source or encoder identity changed')
        validate_schema(self.meta['schema'], bundle)
        if set(self.meta['slides']) != set(bundle.rows.slide_id):
            raise ValueError('C²MIL graph/thumbnail manifest differs from the selected task population')
        self.contract = digest_file(self.path)
        self.order = copy.deepcopy(self.meta['schema']['patient_order'])
        self.by_patient = {r['global_patient_id']: r for r in bundle.rows.to_dict('records')}
        self.files = [self.path]
        for e in self.meta['slides'].values():
            self.files.extend([resolve(self.path.parent, e['path']).resolve(),
                               resolve(self.path.parent, e['thumbnail_path']).resolve()])
        self.files.extend(self.path.parent/name for name in ['sampling_rng_before.npy', 'sampling_rng_after.npy'])
        if len(set(self.files)) != len(self.files):
            raise ValueError('C²MIL graph and thumbnail assets must be distinct per WSI')
        sampling = self.meta.get('sampling', {})
        if (sampling.get('patches') != 1500 or sampling.get('seed') != 98 or
                sampling.get('padding') != 'mean_of_selected_rows' or sampling.get('mode') not in
                {'provided_original_fold_rng', 'fresh_single_split_seed98'}):
            raise ValueError('C²MIL uses the original 1500-patch TRAIN reference protocol')
        for which in ['before', 'after']:
            if digest_file(self.path.parent/('sampling_rng_'+which+'.npy')) != self.meta['sampling'][which+'_sha256']:
                raise ValueError('C²MIL reference-sampling RNG provenance changed')
        generator = torch.Generator()
        generator.set_state(torch.from_numpy(np.load(self.path.parent/'sampling_rng_before.npy', allow_pickle=False)))
        if (sampling['mode'] == 'fresh_single_split_seed98' and
                not torch.equal(generator.get_state(), torch.Generator().manual_seed(98).get_state())):
            raise ValueError('C²MIL fresh single-split sampling must start from the declared seed98')
        for gid in self.order['train']:
            e = self.meta['slides'][self.by_patient[gid]['slide_id']]
            expected = torch.randperm(e['nodes'], generator=generator)[:1500].numpy()
            with h5py.File(resolve(self.path.parent, e['path']), 'r') as stream:
                if not np.array_equal(expected, np.asarray(stream['reference_indices'])):
                    raise ValueError('C²MIL TRAIN reference indices differ from the declared original RNG stream')
        if not np.array_equal(generator.get_state().numpy(), np.load(self.path.parent/'sampling_rng_after.npy', allow_pickle=False)):
            raise ValueError('C²MIL final reference-sampling RNG state differs')

    def load(self, row):
        e = self.meta['slides'][row['slide_id']]
        graph = resolve(self.path.parent, e['path'])
        thumb = resolve(self.path.parent, e['thumbnail_path'])
        if (digest_file(self.path) != self.contract or digest_file(graph) != e['sha256'] or
                digest_file(thumb) != e['thumbnail_sha256'] or
                digest_file(resolve(self.bundle.root, row['path'])) != e['primary_sha256'] or
                digest_file(resolve(self.bundle.root, row.get('coords_path') or row['path'])) != e['coordinate_sha256']):
            raise ValueError('C²MIL graph, thumbnail, reference or primary FM contents changed')
        x, coords = self.bundle.load(row, require_coords=True)
        train = row['global_patient_id'] in self.order['train']
        with h5py.File(graph, 'r') as stream:
            if set(stream) != ({'coords', 'edge_index', 'reference_indices'} if train else {'coords', 'edge_index'}):
                raise ValueError('C²MIL only TRAIN assets may contain reference-pool sampling indices')
            xy, edges = np.asarray(stream['coords']), np.asarray(stream['edge_index'])
            indices = np.asarray(stream['reference_indices']) if train else None
        if len(x) != e['nodes'] or not np.array_equal(xy, coords.numpy()):
            raise ValueError('C²MIL physical graph rows differ from actual patch coordinates')
        if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in 'iu' or
                (edges.size and (edges.min() < 0 or edges.max() >= len(x))) or
                np.any(edges[0] == edges[1]) or len(np.unique(edges.T, axis=0)) != edges.shape[1] or
                not np.all(np.bincount(edges[0].astype(np.int64), minlength=len(x)) == min(8, len(x)-1))):
            raise ValueError('C²MIL requires valid original eight-neighbor physical graph edges')
        if train and (indices.shape != (min(1500, len(x)),) or indices.dtype.kind not in 'iu' or
                len(set(indices.tolist())) != len(indices) or indices.min() < 0 or indices.max() >= len(x)):
            raise ValueError('C²MIL TRAIN reference indices must be actual unique selected patch rows')
        return (Data(x=x, edge_index=torch.from_numpy(edges.astype(np.int64)), edge_attr=None),
                thumbnail_array(thumb, self.bundle.dim), indices)

    def reference_population(self):
        thumbs, patches = [], []
        for gid in self.order['train']:
            graph, thumb, indices = self.load(self.by_patient[gid])
            selected = graph.x[torch.from_numpy(indices.astype(np.int64))]
            if len(selected) < 1500:
                selected = torch.cat([selected, selected.mean(0, keepdim=True).repeat(1500-len(selected), 1)])
            thumbs.append(thumb[0]);patches.append(selected)
        return torch.stack(thumbs), torch.stack(patches)

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        e = self.meta['slides'][row['slide_id']]
        return {'manifest_sha256': self.contract, 'graph_sha256': e['sha256'],
            'thumbnail_sha256': e['thumbnail_sha256'], 'nodes': e['nodes'],
            'TRAIN_reference_patient': row['global_patient_id'] in self.order['train']}

"""Bind real CoD low/high feature rows, physical maps and ordered text assets."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_array, resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'codmil_physical_chain_prompts'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'encoder': {k: v for k, v in bundle.encoder.items() if k != 'codmil'}})


def asset(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest_file(path)}


def checked_asset(entry):
    path = Path(entry['path'])
    if path.stat().st_size != entry['bytes'] or digest_file(path) != entry['sha256']:
        raise ValueError('CoD bound asset bytes changed: '+str(path))
    return path


def finite_array(path, dimension, name):
    x = np.asarray(load_array(path))
    if (x.ndim != 2 or not len(x) or x.shape[1] != dimension or x.dtype.kind not in 'fiu'
            or not np.isfinite(x).all()):
        raise ValueError('CoD '+name+' needs complete finite [N,D] features')
    return torch.tensor(x, dtype=torch.float32)


def points(path, count):
    x = np.asarray(load_array(path, 'coords'))
    if (x.shape != (count, 2) or x.dtype.kind not in 'fiu' or not np.isfinite(x).all()
            or len(np.unique(x, axis=0)) != count):
        raise ValueError('CoD real coordinate rows must be finite, unique and aligned with their features')
    return x.astype(np.float64)


def validate_schema(bundle, schema):
    if (schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or type(schema.get('synthetic')) is not bool
            or schema['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('CoD requires its versioned physical schema and matching synthetic marker')
    if bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
        raise ValueError('CoD author protocol requires classification of individual slide bags')
    if schema.get('coordinate_unit') not in {'level0_pixels', 'micrometers'} or schema.get('origin') not in {'top_left', 'center'}:
        raise ValueError('CoD scales require a common real coordinate frame and origin')
    sizes = [schema.get(k) for k in ['coarse_patch_size', 'fine_patch_size']]
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x) or x <= 0 for x in sizes) or sizes[0] <= sizes[1]:
        raise ValueError('CoD coarse physical patch size must exceed the positive fine size')
    spatial = bundle.encoder.get('spatial', {})
    if (spatial.get('unit') != schema['coordinate_unit'] or spatial.get('origin') != schema['origin']
            or spatial.get('patch_size') != sizes[0] or not spatial.get('provenance')):
        raise ValueError('CoD schema and registered coarse physical extraction metadata differ')
    for key in ['mapping_provenance', 'fine_extraction_provenance']:
        if not isinstance(schema.get(key), str) or not schema[key].strip():
            raise ValueError('CoD requires actual '+key)


def check_mapping(mapping, coarse, fine, schema):
    if (mapping.ndim != 2 or mapping.shape[0] != len(coarse) or mapping.shape[1] < 1
            or mapping.dtype.kind not in 'iu' or (mapping < -1).any() or (mapping >= len(fine)).any()):
        raise ValueError('CoD map needs actual fine row IDs or -1 for every coarse coordinate row')
    # Missing fine tissue is explicit (-1). No phantom children or duplicate
    # rows within one region. Repeated children across overlapping regions are
    # retained, as in the original selected-fine indexing path.
    for row in mapping:
        ids = row[row >= 0]
        if len(ids) != len(np.unique(ids)):
            raise ValueError('CoD duplicate fine child within one coarse map row')
    rows, cols = np.nonzero(mapping >= 0)
    if not len(rows):
        raise ValueError('CoD physical map contains no observed fine children')
    coarse_size, fine_size = schema['coarse_patch_size'], schema['fine_patch_size']
    c = coarse[rows].copy(); f = fine[mapping[rows, cols]].copy()
    if schema['origin'] == 'center':
        c -= coarse_size/2; f -= fine_size/2
    tolerance = max(coarse_size, fine_size)*1e-7
    if ((f < c-tolerance).any() or (f+fine_size > c+coarse_size+tolerance).any()):
        raise ValueError('CoD map associates a fine patch outside its declared coarse physical region')
    return torch.tensor(mapping, dtype=torch.long)


def text_assets(bundle, metadata, metadata_path):
    if (metadata.get('schema_version') != 1 or metadata.get('kind') != 'codmil_ordered_text_embeddings'
            or metadata.get('classes') != bundle.classes
            or type(metadata.get('synthetic')) is not bool
            or metadata['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('CoD text class order, schema or synthetic marker differs')
    for key in ['model', 'weights', 'preprocessing', 'text_provenance', 'embedding_space']:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError('CoD text metadata requires '+key)
    dimension = metadata.get('dimension')
    if type(dimension) is not int or dimension < 1:
        raise ValueError('CoD text dimension must be explicitly positive')
    prompts, classes = metadata.get('prompts', []), bundle.classes
    if not isinstance(prompts, list) or len(prompts) < 2*len(classes)+2:
        raise ValueError('CoD needs low/high class prompts, normal concepts and the excluded-last source prompt')
    for i, entry in enumerate(prompts):
        if not isinstance(entry, dict) or not isinstance(entry.get('text'), str) or not entry['text'].strip():
            raise ValueError('Every CoD embedding needs its actual nonempty prompt')
        if i < 2*len(classes):
            role = 'low' if i < len(classes) else 'high'
            if entry.get('role') != role or entry.get('class') != classes[i % len(classes)]:
                raise ValueError('CoD text rows must follow ordered low classes then high classes')
        elif entry.get('role') != ('excluded_last' if i == len(prompts)-1 else 'normal'):
            raise ValueError('CoD normal and source excluded-last text roles must be explicit')
    policy = metadata.get('alignment_policy')
    if policy == 'matched_vlm':
        if (dimension != bundle.dim or metadata['embedding_space'] != bundle.encoder.get('embedding_space')
                or metadata.get('image_model') != bundle.encoder.get('model')
                or metadata.get('image_weights') != bundle.encoder.get('weights')):
            raise ValueError('CoD matched VLM text/image model, weights, dimensions or embedding space differ')
    elif policy == 'learned_projection_extension':
        if not isinstance(metadata.get('alignment_provenance'), str) or not metadata['alignment_provenance'].strip():
            raise ValueError('CoD cross-FM text projection extension requires explicit alignment_provenance')
    else:
        raise ValueError('CoD text/image alignment policy must be explicitly matched or a named learned extension')
    path = resolve(bundle.root, metadata['features'])
    value = finite_array(path, dimension, 'text')
    if len(value) != len(prompts):
        raise ValueError('CoD text prompt and embedding row counts differ')
    return value, [asset(metadata_path), asset(path)]


def load_record(bundle, record, schema):
    coarse_path, coarse_coords_path, fine_path, fine_coords_path, map_path = [checked_asset(record[k]) for k in
        ['coarse_features', 'coarse_coords', 'fine_features', 'fine_coords', 'map']]
    coarse = finite_array(coarse_path, bundle.dim, 'coarse')
    fine = finite_array(fine_path, bundle.dim, 'fine')
    c, f = points(coarse_coords_path, len(coarse)), points(fine_coords_path, len(fine))
    mapping = check_mapping(np.asarray(load_array(map_path, 'mapping')), c, f, schema)
    if [len(coarse), len(fine)] != [record['coarse_count'], record['fine_count']]:
        raise ValueError('CoD full coarse/fine row counts differ from prepared assets')
    return coarse, fine, mapping


def prepare(bundle, index, schema_path, text_metadata, output):
    """Bind supplied real assets. No synthetic alternate scales or text generated."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('CoD output already exists; use a new manifest path')
    index, schema_path, text_metadata = [resolve(bundle.root, p).resolve() for p in [index, schema_path, text_metadata]]
    schema, metadata = read_json(schema_path), read_json(text_metadata)
    validate_schema(bundle, schema)
    text, text_files = text_assets(bundle, metadata, text_metadata)
    frame = pd.read_csv(index, dtype=str, keep_default_na=False)
    needed = {'slide_id', 'fine_path', 'fine_coords_path', 'map_path'}
    if not needed <= set(frame) or frame.slide_id.duplicated().any() or frame[list(needed)].eq('').any().any():
        raise ValueError('CoD index requires unique complete slide/fine/coordinate/map paths')
    if set(frame.slide_id) != set(bundle.rows.slide_id):
        raise ValueError('CoD index must contain every registered slide exactly once')
    source_rows = frame.set_index('slide_id').to_dict('index')
    coarse_paths = {resolve(bundle.root, r['path']).resolve() for r in bundle.rows.to_dict('records')}
    fine_paths, maps, records = set(), set(), []
    for row in bundle.rows.to_dict('records'):
        supplied = source_rows[row['slide_id']]
        fine_path, fine_coords_path, map_path = [resolve(bundle.root, supplied[k]).resolve()
            for k in ['fine_path', 'fine_coords_path', 'map_path']]
        if fine_path in fine_paths or fine_path in coarse_paths or map_path in maps:
            raise ValueError('CoD requires independent per-slide fine features/maps, distinct from coarse files')
        fine_paths.add(fine_path); maps.add(map_path)
        coarse_path = resolve(bundle.root, row['path'])
        coarse_coords_path = resolve(bundle.root, row['coords_path']) if row.get('coords_path') else coarse_path
        record = {'slide_id': row['slide_id'], 'coarse_features': asset(coarse_path),
            'coarse_coords': asset(coarse_coords_path), 'fine_features': asset(fine_path),
            'fine_coords': asset(fine_coords_path), 'map': asset(map_path)}
        coarse, fine = finite_array(coarse_path, bundle.dim, 'coarse'), finite_array(fine_path, bundle.dim, 'fine')
        record.update(coarse_count=len(coarse), fine_count=len(fine))
        load_record(bundle, record, schema)
        records.append(record)
    result = {'schema_version': 1, 'kind': KIND, 'schema': schema, 'encoder_identity': encoder_identity(bundle),
        'text_metadata': metadata, 'text_assets': text_files, 'text_dimension': text.shape[1],
        'sources': [asset(index), asset(schema_path)], 'records': records}
    result['contract'] = digest_object(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return str(output)


class CoDAssets:
    def __init__(self, bundle):
        self.bundle = bundle
        self.path = resolve(bundle.root, bundle.encoder['codmil']['manifest'])
        self.manifest_sha = digest_file(self.path)
        self.manifest = manifest = read_json(self.path)
        contract = {k: v for k, v in manifest.items() if k != 'contract'}
        if (manifest.get('schema_version') != 1 or manifest.get('kind') != KIND
                or manifest.get('encoder_identity') != encoder_identity(bundle)
                or manifest.get('contract') != digest_object(contract)):
            raise ValueError('CoD manifest contract, encoder or complete row binding differs')
        validate_schema(bundle, manifest['schema'])
        for entry in manifest['sources']+manifest['text_assets']:
            checked_asset(entry)
        if (manifest['schema'] != read_json(manifest['sources'][1]['path'])
                or manifest['text_metadata'] != read_json(manifest['text_assets'][0]['path'])):
            raise ValueError('CoD embedded schema/text metadata differs from bound source files')
        self.text, _ = text_assets(bundle, manifest['text_metadata'], Path(manifest['text_assets'][0]['path']))
        self.rows = {r['slide_id']: r for r in manifest['records']}
        if len(self.rows) != len(manifest['records']) or set(self.rows) != set(bundle.rows.slide_id):
            raise ValueError('CoD prepared slide identities differ')
        registered = bundle.rows.set_index('slide_id').to_dict('index')
        index = pd.read_csv(manifest['sources'][0]['path'], dtype=str, keep_default_na=False)
        if index.slide_id.duplicated().any() or set(index.slide_id) != set(self.rows):
            raise ValueError('CoD source index slide identities differ')
        index = index.set_index('slide_id').to_dict('index')
        coarse_paths = {resolve(bundle.root, row['path']).resolve() for row in registered.values()}
        fine_paths, maps = set(), set()
        for sid, record in self.rows.items():
            if Path(record['coarse_features']['path']) != resolve(bundle.root, registered[sid]['path']).resolve():
                raise ValueError('CoD coarse row path differs from the registered FM index')
            coords = registered[sid].get('coords_path') or registered[sid]['path']
            if Path(record['coarse_coords']['path']) != resolve(bundle.root, coords).resolve():
                raise ValueError('CoD coarse coordinate path differs from the registered FM index')
            for source_key, asset_key in [('fine_path', 'fine_features'), ('fine_coords_path', 'fine_coords'), ('map_path', 'map')]:
                if Path(record[asset_key]['path']) != resolve(bundle.root, index[sid][source_key]).resolve():
                    raise ValueError('CoD manifest and source index asset paths differ')
            fine, mapping = Path(record['fine_features']['path']), Path(record['map']['path'])
            if fine in coarse_paths or fine in fine_paths or mapping in maps:
                raise ValueError('CoD requires distinct per-slide fine files/maps, independent of coarse files')
            fine_paths.add(fine); maps.add(mapping)
        self.files = [self.path, *[Path(e['path']) for e in manifest['sources']+manifest['text_assets']]]
        self.contract = manifest['contract']

    def load(self, row):
        if digest_file(self.path) != self.manifest_sha:
            raise ValueError('CoD manifest changed after loading')
        return load_record(self.bundle, self.rows[row['slide_id']], self.manifest['schema'])

    def inventory(self, row, deep):
        if deep:
            self.load(row)
        record = self.rows[row['slide_id']]
        return {'contract': self.contract, 'assets': {k: {**record[k],
            'mtime_ns': checked_asset(record[k]).stat().st_mtime_ns} for k in
            ['coarse_features', 'coarse_coords', 'fine_features', 'fine_coords', 'map']},
            'coarse_count': record['coarse_count'], 'fine_count': record['fine_count']}

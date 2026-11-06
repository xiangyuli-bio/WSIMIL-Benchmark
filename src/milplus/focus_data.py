"""Bind actual pretrained FOCUS weights, ordered class descriptions and feature rows."""
from pathlib import Path

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'focus_native_prompt_text'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'encoder': {k: v for k, v in bundle.encoder.items() if k != 'focus'}})


def asset(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest_file(path)}


def checked_asset(entry):
    path = Path(entry['path'])
    if path.stat().st_size != entry['bytes'] or digest_file(path) != entry['sha256']:
        raise ValueError('FOCUS registered asset bytes changed: '+str(path))
    return path


def validate_metadata(bundle, metadata):
    from .focus_network import WEIGHTS_SHA
    if (metadata.get('schema_version') != 1 or metadata.get('kind') != KIND
            or metadata.get('classes') != bundle.classes
            or type(metadata.get('synthetic')) is not bool
            or metadata['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('FOCUS prompt schema, ordered classes or synthetic marker differs')
    if bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
        raise ValueError('FOCUS original complete recipe requires individual slide classification')
    for key in ['prompt_provenance', 'feature_order_provenance', 'feature_preprocessing', 'weights_provenance']:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError('FOCUS requires actual '+key)
    if metadata.get('weights_sha256') != WEIGHTS_SHA:
        raise ValueError('FOCUS requires actual registered native CONCH pretrained text weights')
    prompts = metadata.get('prompts')
    if not isinstance(prompts, list) or len(prompts) != 2*len(bundle.classes):
        raise ValueError('FOCUS requires exactly ordered low then high C descriptions')
    for i, entry in enumerate(prompts):
        if (not isinstance(entry, dict) or entry.get('role') != ('low' if i < len(bundle.classes) else 'high')
                or entry.get('class') != bundle.classes[i % len(bundle.classes)]
                or not isinstance(entry.get('text'), str) or not entry['text'].strip()):
            raise ValueError('FOCUS class prompt rows, roles and descriptions must be explicit and aligned')
    policy = metadata.get('alignment_policy')
    if policy == 'native_conch':
        if (bundle.dim != 512 or bundle.encoder.get('model') != 'conch_ViT-B-16'
                or bundle.encoder.get('weights_sha256') != WEIGHTS_SHA
                or bundle.encoder.get('feature_preprocessing') != metadata['feature_preprocessing']):
            raise ValueError('FOCUS native CONCH feature model, full dimension, weights or preprocessing differs')
    elif policy == 'cross_fm_projection_extension':
        if not isinstance(metadata.get('alignment_provenance'), str) or not metadata['alignment_provenance'].strip():
            raise ValueError('FOCUS cross-FM Dto512 extension requires explicit alignment provenance')
    else:
        raise ValueError('FOCUS needs a declared native CONCH or cross-FM extension condition')


def prepare(bundle, text_metadata, output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('FOCUS output exists; prepare a new immutable manifest')
    text_metadata = resolve(bundle.root, text_metadata).resolve()
    metadata = read_json(text_metadata)
    validate_metadata(bundle, metadata)
    weights = asset(resolve(bundle.root, metadata['checkpoint']))
    if weights['sha256'] != metadata['weights_sha256']:
        raise ValueError('FOCUS actual checkpoint bytes do not match declared original pretrained weights')
    records = []
    for row in bundle.rows.to_dict('records'):
        features, _ = bundle.load(row)
        records.append({'slide_id': row['slide_id'], 'features': asset(resolve(bundle.root, row['path'])),
                        'count': len(features), 'dimension': features.shape[1]})
    result = {'schema_version': 1, 'kind': KIND, 'metadata': metadata, 'source': asset(text_metadata),
        'weights': weights, 'encoder_identity': encoder_identity(bundle), 'records': records}
    result['contract'] = digest_object(result)
    write_json(output, result)
    return str(output)


class FOCUSAssets:
    def __init__(self, bundle):
        self.bundle = bundle
        self.path = resolve(bundle.root, bundle.encoder['focus']['manifest'])
        self.manifest_sha = digest_file(self.path)
        self.manifest = manifest = read_json(self.path)
        if (manifest.get('schema_version') != 1 or manifest.get('kind') != KIND
                or manifest.get('encoder_identity') != encoder_identity(bundle)
                or manifest.get('contract') != digest_object({k: v for k, v in manifest.items() if k != 'contract'})):
            raise ValueError('FOCUS immutable manifest or feature encoder binding differs')
        source = checked_asset(manifest['source'])
        if manifest['metadata'] != read_json(source):
            raise ValueError('FOCUS embedded metadata differs from the actual bound source')
        validate_metadata(bundle, manifest['metadata'])
        self.weights = checked_asset(manifest['weights'])
        if (self.weights != resolve(bundle.root, manifest['metadata']['checkpoint']).resolve()
                or manifest['weights']['sha256'] != manifest['metadata']['weights_sha256']):
            raise ValueError('FOCUS bound pretrained checkpoint differs')
        self.rows = {r['slide_id']: r for r in manifest['records']}
        if len(self.rows) != len(manifest['records']) or set(self.rows) != set(bundle.rows.slide_id):
            raise ValueError('FOCUS feature identity coverage differs')
        for row in bundle.rows.to_dict('records'):
            rec = self.rows[row['slide_id']]
            if Path(rec['features']['path']) != resolve(bundle.root, row['path']).resolve() or rec['dimension'] != bundle.dim:
                raise ValueError('FOCUS registered full feature path/dimension differs')
        self.contract = manifest['contract']
        self.prompts = [r['text'] for r in manifest['metadata']['prompts']]
        self.files = [self.path, source]  # Checkpoint identity is content-bound above, not part of data root.

    def load(self, row):
        if digest_file(self.path) != self.manifest_sha:
            raise ValueError('FOCUS immutable manifest changed after loading')
        record = self.rows[row['slide_id']]
        checked_asset(record['features'])
        value, _ = self.bundle.load(row)
        if len(value) != record['count'] or value.shape[1] != record['dimension']:
            raise ValueError('FOCUS ordered feature rows differ from prepared inventory')
        return value

    def inventory(self, row, deep):
        record = self.rows[row['slide_id']]
        if deep:
            self.load(row)
        return {'contract': self.contract, 'ordered_features': {**record['features'],
            'mtime_ns': checked_asset(record['features']).stat().st_mtime_ns}, 'count': record['count'],
            'dimension': record['dimension'], 'weights_sha256': self.manifest['weights']['sha256']}

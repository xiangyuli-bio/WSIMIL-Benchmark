"""Bind FEATHER's actual pretrained feature space and patient partitions.

Shape compatibility does not establish native encoder compatibility. This
manifest records a provenance assertion by the data producer and hashes every
actual feature file; it cannot prove that an undocumented extraction was run.
"""
from pathlib import Path

from .feather_assets import OriginalFEATHERAsset, native_lock
from .feather_source import ENCODERS
from .io import digest_file, digest_object, read_json, write_json


def resolve(root, value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def identity(bundle):
    return {'task_id': bundle.config['task_id'], 'task_type': bundle.task_type,
            'classes': bundle.classes, 'fm': bundle.fm, 'dimension': bundle.dim,
            'split': bundle.split, 'rows': bundle.rows.to_dict('records'),
            'training_unit': bundle.config.get('training_unit', 'slide'),
            'evaluation_unit': bundle.config.get('evaluation_unit', 'slide')}


def verify_visual_metadata(bundle, encoder, value):
    if (encoder not in ENCODERS or bundle.dim != ENCODERS[encoder]
            or bundle.task_type not in {'classification', 'ordinal'}
            or bundle.config.get('training_unit', 'slide') != 'slide'):
        raise ValueError('FEATHER requires its actual native encoder and slide classification/grading bags')
    row = native_lock()['native_encoders'][encoder]
    expected = next(iter(row['files'].values()))['sha256']
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or value.get('encoder_id') != encoder or value.get('feature_dim') != ENCODERS[encoder]
            or value.get('model_id') != row['model_id'] or value.get('revision') != row['revision']
            or value.get('weights_sha256') != expected
            or not isinstance(value.get('extraction_provenance'), str)
            or not value['extraction_provenance'].strip()):
        raise ValueError('Bind the native FEATHER visual checkpoint and extraction provenance; equal dimensions are insufficient')
    return value


def feature_population(bundle):
    rows = bundle.rows.to_dict('records')
    patients = {}
    records = {}
    paths = set()
    hashes = {}
    for row in rows:
        if row['subset'] not in {'train', 'val', 'test'}:
            raise ValueError('FEATHER requires explicit TRAIN/VAL/TEST partitions')
        patient = row['global_patient_id']
        previous = patients.setdefault(patient, row['subset'])
        if not patient or previous != row['subset']:
            raise ValueError('FEATHER global patient identity crosses partitions')
        path = resolve(bundle.root, row['path'])
        if path in paths or row['slide_id'] in records:
            raise ValueError('FEATHER physical feature paths and slide identities must be unique')
        paths.add(path)
        x, unused = bundle.load(row)
        sha = digest_file(path)
        prior = hashes.setdefault(sha, row['subset'])
        if prior != row['subset']:
            raise ValueError('Byte-identical FEATHER bags cross partitions; resolve duplicate slides explicitly')
        stat = path.stat()
        records[row['slide_id']] = {'path': str(path), 'sha256': sha,
            'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'instances': len(x),
            'dimension': bundle.dim, 'subset': row['subset'], 'global_patient_id': patient,
            'target': int(row['target'])}
    if set(patients.values()) != {'train', 'val', 'test'}:
        raise ValueError('FEATHER needs nonempty patient-disjoint TRAIN/VAL/TEST partitions')
    return records


def prepare(bundle, schema, output):
    keys = {'schema_version', 'pretrained_assets', 'visual_extraction_manifest', 'provenance'}
    if (not isinstance(schema, dict) or set(schema) != keys or schema['schema_version'] != 1
            or not isinstance(schema['provenance'], str) or not schema['provenance'].strip()):
        raise ValueError('Declare FEATHER pretrained assets, native extraction manifest and provenance')
    if 'feather' in bundle.encoder:
        raise ValueError('Prepare before adding encoder.feather to the task bundle')
    asset = OriginalFEATHERAsset(resolve(bundle.root, schema['pretrained_assets']))
    visual = resolve(bundle.root, schema['visual_extraction_manifest'])
    metadata = verify_visual_metadata(bundle, asset.encoder, read_json(visual))
    records = feature_population(bundle)
    value = {'schema_version': 1, 'kind': 'FEATHER24K_native_pretrained_patient_bound_features',
        'schema': schema, 'identity': identity(bundle), 'encoder': asset.encoder,
        'pretrained_assets': str(asset.path), 'pretrained_contract': asset.contract,
        'visual_extraction_manifest': str(visual), 'visual_metadata': metadata,
        'files': {str(asset.path): digest_file(asset.path), str(visual): digest_file(visual)},
        'records': records, 'extraction_provenance_independently_verified': False,
        'full_pretrained_model_validation': False, 'formal_admission': False}
    value['contract'] = digest_object(value)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json(output, value)
    return output


class FEATHERInputs:
    def __init__(self, bundle):
        config = bundle.encoder.get('feather')
        if not isinstance(config, dict) or set(config) != {'manifest'}:
            raise ValueError('Bind encoder.feather.manifest to the complete native input contract')
        self.bundle = bundle
        self.path = resolve(bundle.root, config['manifest'])
        value = read_json(self.path)
        self.contract = value.pop('contract', None)
        if (self.contract != digest_object(value) or value.get('schema_version') != 1
                or value.get('kind') != 'FEATHER24K_native_pretrained_patient_bound_features'
                or value.get('identity') != identity(bundle)):
            raise ValueError('FEATHER patient, label, input feature space or binding changed')
        self.value = value
        self.asset = OriginalFEATHERAsset(value['pretrained_assets'])
        self.encoder = self.asset.encoder
        if (value['encoder'] != self.encoder or self.asset.contract != value['pretrained_contract']
                or verify_visual_metadata(bundle, self.encoder, value['visual_metadata']) != value['visual_metadata']):
            raise ValueError('FEATHER pretrained checkpoint/native feature pairing changed')
        self.records = value['records']
        self.files = [self.path, *[Path(p) for p in value['files']]]
        self.verify(deep=True)

    def verify(self, deep=False):
        if (read_json(self.path) != {**self.value, 'contract': self.contract}
                or self.value['identity'] != identity(self.bundle)):
            raise ValueError('FEATHER native input contract changed')
        self.asset.verify(deep=deep)
        for path, sha in self.value['files'].items():
            if digest_file(path) != sha:
                raise ValueError('FEATHER visual provenance or pretrained asset manifest changed')
        if set(self.records) != set(self.bundle.rows.slide_id):
            raise ValueError('FEATHER actual slide population changed')
        for row in self.bundle.rows.to_dict('records'):
            old = self.records[row['slide_id']]
            path = resolve(self.bundle.root, row['path'])
            stat = path.stat()
            if (str(path) != old['path'] or row['subset'] != old['subset']
                    or row['global_patient_id'] != old['global_patient_id']
                    or int(row['target']) != old['target'] or stat.st_size != old['bytes']
                    or stat.st_mtime_ns != old['mtime_ns']
                    or (deep and digest_file(path) != old['sha256'])):
                raise ValueError('FEATHER actual bag bytes, labels or patient membership changed')

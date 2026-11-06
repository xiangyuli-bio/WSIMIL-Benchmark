"""Immutable native text prototypes, full bags and patient-disjoint support pool."""
import hashlib
from pathlib import Path

import numpy as np

from .data import resolve
from .io import digest_file, digest_object, read_json, write_json
from .miladapter_source import COMMIT
from .upstream import verified_source

KIND = 'miladapter_original_prototypes_and_support_pool'
ENCODER_DIMENSIONS = {'CONCH': 512, 'KEEP': 768}
PROJECT_CLASSES = {'BRCA': ['IDC', 'ILC'], 'NSCLC': ['LUAD', 'LUSC'],
                   'RCC': ['CHRCC', 'CCRCC', 'PRCC']}


def import_author_prototypes(output, project='BRCA', encoder='CONCH'):
    if project not in PROJECT_CLASSES or encoder not in {'CONCH', 'KEEP'}:
        raise ValueError('Author MIL-Adapter provides BRCA/NSCLC/RCC prototypes for CONCH/KEEP; no TITAN prototypes')
    root, sha = verified_source('miladapter')
    if sha != COMMIT:
        raise ValueError('MIL-Adapter author prototype commit changed')
    relative = f'local_data/prompts/{encoder}/{project}.npy'
    import subprocess
    original = subprocess.check_output(['git', 'show', sha+':'+relative], cwd=root)
    source_sha = hashlib.sha256(original).hexdigest()
    if digest_file(root/relative) != source_sha:
        raise ValueError('Author public prototype bytes differ from their fixed Git blob')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'prototypes.npy').write_bytes(original)
    data = np.load(output/'prototypes.npy', allow_pickle=False)
    if data.dtype != np.float32 or data.shape != (ENCODER_DIMENSIONS[encoder], len(PROJECT_CLASSES[project])):
        raise ValueError('Author prototype dimension, column count or dtype changed')
    meta = {'schema_version': 1, 'kind': 'miladapter_author_public_text_prototypes',
        'source_commit': COMMIT, 'source_relative_path': relative, 'sha256': source_sha,
        'project': project, 'encoder': encoder, 'classes': PROJECT_CLASSES[project],
        'path': str(output/'prototypes.npy'),
        'provenance': 'Exact author-distributed prototype matrix; generation code, exact original prompts and encoder weight revision not disclosed here. No regenerated text ensemble claim.',
        'prompt_patient_ids': [], 'prompt_provenance_status': 'author_import_generation_details_undisclosed'}
    write_json(output/'text.json', meta)
    return output/'text.json'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'miladapter'}})


def text_assets(path):
    path = Path(path).expanduser().resolve(); meta = read_json(path)
    if (meta.get('schema_version') != 1 or meta.get('kind') != 'miladapter_author_public_text_prototypes'
            or meta.get('source_commit') != COMMIT or meta.get('project') not in PROJECT_CLASSES
            or meta.get('encoder') not in {'CONCH', 'KEEP'}
            or meta.get('classes') != PROJECT_CLASSES[meta['project']]
            or meta.get('prompt_provenance_status') != 'author_import_generation_details_undisclosed'
            or meta.get('prompt_patient_ids') != []):
        raise ValueError('MIL-Adapter needs genuine author prototype provenance, not fabricated prompt regeneration')
    source, sha = verified_source('miladapter')
    relative = f"local_data/prompts/{meta['encoder']}/{meta['project']}.npy"
    import subprocess
    expected = hashlib.sha256(subprocess.check_output(['git', 'show', sha+':'+relative], cwd=source)).hexdigest()
    value_path = Path(meta['path'])
    if meta.get('source_relative_path') != relative or meta.get('sha256') != expected or digest_file(value_path) != expected:
        raise ValueError('MIL-Adapter actual source text prototype bytes changed')
    value = np.load(value_path, allow_pickle=False)
    if (value.dtype != np.float32 or value.shape != (ENCODER_DIMENSIONS[meta['encoder']], len(meta['classes']))
            or not np.isfinite(value).all() or not np.all(np.linalg.norm(value, axis=0) > 0)):
        raise ValueError('MIL-Adapter source prototype matrix is invalid')
    return meta, value, [path, value_path]


def validate_metadata(bundle, meta, text):
    if (bundle.task_type != 'classification' or bundle.dim != ENCODER_DIMENSIONS[text['encoder']]
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or meta.get('schema_version') != 1 or meta.get('kind') != KIND
            or meta.get('classes') != bundle.classes or text['classes'] != bundle.classes
            or meta.get('synthetic') != bool(bundle.config.get('synthetic', False))
            or meta.get('support_policy') != 'source_classwise_sampling_from_train_pool_only_patient_disjoint_extension'):
        raise ValueError('MIL-Adapter needs native512CONCH/768KEEP slide classification, ordered source classes and explicit TRAIN-only support policy')
    if meta.get('alignment_policy') == 'synthetic_native_dimension_fixture':
        if not bundle.config.get('synthetic', False):
            raise ValueError('Synthetic MIL-Adapter bags cannot represent actual extracted pathology features')
    elif meta.get('alignment_policy') == 'author_native_visual_text_space':
        if (bundle.encoder.get('model') != text['encoder']
                or not isinstance(bundle.encoder.get('weights_sha256'), str)
                or len(bundle.encoder['weights_sha256']) != 64):
            raise ValueError('MIL-Adapter requires the declared actual CONCH/KEEP visual encoder weight identity')
        if meta.get('alignment_evidence') != 'author_released_prototypes_weight_revision_undisclosed':
            raise ValueError('Disclose unknown author prototype weight revision; do not claim exact recovered image/text pretraining identity')
    else:
        raise ValueError('Unrelated FM cannot substitute for MIL-Adapter native visual/text space')
    for key in ['feature_preprocessing', 'feature_provenance', 'task_semantics']:
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError('MIL-Adapter requires actual preprocessing, feature source and task definition')
    # The public script samples slides. Restrict its support pool to a single
    # slide per globally identified patient; do not silently call slide shots
    # patient shots or draw another slide from the same support patient.
    train = bundle.rows[bundle.rows.subset == 'train']
    if train.global_patient_id.duplicated().any():
        raise ValueError('MIL-Adapter source slide-shot route needs one eligible TRAIN slide per patient; prepare an explicit patient-level slide choice first')


def describe(bundle, row):
    value, _ = bundle.load(row)
    return {'path': str(resolve(bundle.root, row['path']).resolve()),
        'sha256': digest_file(resolve(bundle.root, row['path'])), 'count': len(value), 'dimension': bundle.dim,
        **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}


def prepare(bundle, metadata_path, output):
    output = resolve(bundle.root, output)
    if output.exists():
        raise FileExistsError('MIL-Adapter assets exist; prepare a new immutable version')
    metadata_path = resolve(bundle.root, metadata_path).resolve(); meta = read_json(metadata_path)
    text_path = resolve(bundle.root, meta['text_assets']).resolve()
    text, _, files = text_assets(text_path)
    validate_metadata(bundle, meta, text)
    manifest = {'schema_version': 1, 'kind': KIND, 'encoder_identity': encoder_identity(bundle),
        'metadata_path': str(metadata_path), 'metadata': meta, 'metadata_sha256': digest_file(metadata_path),
        'text_path': str(text_path), 'text_sha256': digest_file(text_path),
        'text_files': {str(p): digest_file(p) for p in files},
        'records': {row['slide_id']: describe(bundle, row) for row in bundle.rows.to_dict('records')},
        'preparer_sha256': digest_file(Path(__file__))}
    manifest['contract'] = digest_object(manifest); write_json(output, manifest)
    return output


class MILAdapterAssets:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.manifest_sha = digest_file(self.path); self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('contract') != digest_object({k: v for k, v in self.meta.items() if k != 'contract'})):
            raise ValueError('MIL-Adapter immutable preparation/source/encoder contract changed')
        path = Path(self.meta['metadata_path'])
        if digest_file(path) != self.meta['metadata_sha256'] or read_json(path) != self.meta['metadata']:
            raise ValueError('MIL-Adapter original input metadata changed')
        self.text_meta, self.prototypes, files = text_assets(self.meta['text_path'])
        if (digest_file(self.meta['text_path']) != self.meta['text_sha256']
                or self.meta['text_files'] != {str(p): digest_file(p) for p in files}):
            raise ValueError('MIL-Adapter actual source prototype identity changed')
        validate_metadata(bundle, self.meta['metadata'], self.text_meta)
        if set(self.meta['records']) != set(bundle.rows.slide_id):
            raise ValueError('MIL-Adapter feature coverage changed')
        for row in bundle.rows.to_dict('records'):
            r = self.meta['records'][row['slide_id']]
            if (r['path'] != str(resolve(bundle.root, row['path']).resolve())
                    or any(r[k] != row[k] for k in ['global_patient_id', 'subset', 'target'])):
                raise ValueError('MIL-Adapter feature path, patient, target or split changed')
        self.contract = self.meta['contract']; self.files = [self.path, path, *files]; self._verified = {}
        self._fixed_files = {str(p): digest_file(p) for p in self.files}

    def load(self, row):
        for fixed, expected_sha in self._fixed_files.items():
            if digest_file(fixed) != expected_sha:
                raise ValueError('MIL-Adapter immutable metadata or actual text prototype file changed during use')
        path = resolve(self.bundle.root, row['path']); s = path.stat()
        signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            if describe(self.bundle, row) != self.meta['records'][row['slide_id']]:
                raise ValueError('MIL-Adapter original complete feature bag bytes or row identity changed')
            self._verified[str(path)] = signature
        if digest_file(self.path) != self.manifest_sha:
            raise ValueError('MIL-Adapter prepared manifest changed')
        return self.bundle.load(row)[0]

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        return {'contract': self.contract, 'feature': self.meta['records'][row['slide_id']],
                'actual_text_prototypes': self.text_meta['sha256']}

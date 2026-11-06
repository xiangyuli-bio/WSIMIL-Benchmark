"""Bind native CONCH visual/text space, immutable queries and TRAIN-only review."""
from pathlib import Path

import torch

from .conmil_text import load_text
from .data import resolve
from .focus_network import WEIGHTS_SHA
from .io import digest_file, digest_object, read_json, write_json

KIND = 'conmil_original_semantic_attention_assets'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'conmil'}})


def validate_metadata(bundle, metadata, text):
    if (bundle.task_type != 'classification' or bundle.dim != 512
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or metadata.get('schema_version') != 1 or metadata.get('kind') != KIND
            or metadata.get('classes') != bundle.classes
            or metadata.get('synthetic') != bool(bundle.config.get('synthetic', False))):
        raise ValueError('ConMIL needs native512D slide classification, ordered classes and explicit synthetic identity')
    policy = metadata.get('alignment_policy')
    if text.metadata['synthetic'] and not bool(bundle.config.get('synthetic', False)):
        raise ValueError('ConMIL software prompt fixtures cannot be presented as actual clinical task assets')
    if policy == 'native_conch':
        if (bundle.encoder.get('model') != 'conch_ViT-B-16'
                or bundle.encoder.get('weights_sha256') != WEIGHTS_SHA
                or bundle.encoder.get('normalization') != 'L2'
                or text.metadata['weights_sha256'] != WEIGHTS_SHA):
            raise ValueError('ConMIL visual and positive/negative text features must share the actual original CONCH space')
    elif policy == 'synthetic_native_dimension_fixture':
        if not bool(bundle.config.get('synthetic', False)):
            raise ValueError('Synthetic ConMIL features cannot be presented as actual extracted CONCH embeddings')
    else:
        raise ValueError('ConMIL requires explicit shared native CONCH space; an unrelated FM is not a text-aligned substitute')
    if any(not isinstance(metadata.get(k), str) or not metadata[k].strip()
           for k in ['feature_preprocessing', 'feature_provenance', 'task_semantics']):
        raise ValueError('ConMIL requires actual visual preprocessing, feature source and task semantics')
    roles = metadata.get('class_roles')
    expected_roles = ['negative', 'positive'] if len(bundle.classes) == 2 else ['task_class']*len(bundle.classes)
    if roles != [{'class': name, 'role': role} for name, role in zip(bundle.classes, expected_roles)]:
        raise ValueError('ConMIL original positive target must be class1, with explicit C-way target roles for repairs')
    review = text.metadata['prompts']['review']
    train = {row['global_patient_id'] for row in bundle.records('train')}
    if not set(review['global_patient_ids']) <= train:
        raise ValueError('ConMIL prompt selection/refinement contains non-TRAIN patient identities')
    return metadata


def describe(bundle, row):
    features, _ = bundle.load(row)
    if not torch.allclose(features.norm(dim=1), torch.ones(len(features)), rtol=1e-3, atol=1e-3):
        raise ValueError('ConMIL source input must retain actual per-patch L2 normalization; no silent normalization repair')
    return {'path': str(resolve(bundle.root, row['path']).resolve()),
        'sha256': digest_file(resolve(bundle.root, row['path'])), 'count': len(features), 'dimension': 512,
        **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}


def prepare(bundle, metadata_path, output):
    output = resolve(bundle.root, output)
    if output.exists():
        raise FileExistsError('ConMIL manifest exists; prepare a new immutable input version')
    metadata_path = resolve(bundle.root, metadata_path).resolve()
    metadata = read_json(metadata_path)
    text_path = resolve(bundle.root, metadata['text_assets']).resolve()
    text = load_text(text_path)
    validate_metadata(bundle, metadata, text)
    result = {'schema_version': 1, 'kind': KIND, 'encoder_identity': encoder_identity(bundle),
        'metadata': metadata, 'metadata_path': str(metadata_path), 'metadata_sha256': digest_file(metadata_path),
        'text_path': str(text_path), 'text_sha256': digest_file(text_path),
        'records': {row['slide_id']: describe(bundle, row) for row in bundle.rows.to_dict('records')},
        'preparer_sha256': digest_file(Path(__file__))}
    result['contract'] = digest_object(result)
    write_json(output, result)
    return output


class ConMILAssets:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.manifest_sha = digest_file(self.path)
        self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('contract') != digest_object({k: v for k, v in self.meta.items() if k != 'contract'})):
            raise ValueError('ConMIL immutable visual/text/preparation contract changed')
        metadata_path = Path(self.meta['metadata_path'])
        if (digest_file(metadata_path) != self.meta['metadata_sha256']
                or read_json(metadata_path) != self.meta['metadata']
                or digest_file(self.meta['text_path']) != self.meta['text_sha256']):
            raise ValueError('ConMIL actual prompt metadata or text asset identity changed')
        self.text = load_text(self.meta['text_path'])
        validate_metadata(bundle, self.meta['metadata'], self.text)
        if set(self.meta['records']) != set(bundle.rows.slide_id):
            raise ValueError('ConMIL registered feature coverage changed')
        for row in bundle.rows.to_dict('records'):
            record = self.meta['records'][row['slide_id']]
            if (record['path'] != str(resolve(bundle.root, row['path']).resolve())
                    or any(record[key] != row[key] for key in ['global_patient_id', 'subset', 'target'])):
                raise ValueError('ConMIL patient labels, feature paths or split changed')
        self.contract = self.meta['contract']
        self.files = [self.path, metadata_path, *self.text.files]
        self._verified = {}

    def load(self, row):
        path = resolve(self.bundle.root, row['path']); stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(str(path)) != signature:
            if describe(self.bundle, row) != self.meta['records'][row['slide_id']]:
                raise ValueError('ConMIL original normalized patch features, row identity or patient changed')
            self._verified[str(path)] = signature
        if digest_file(self.path) != self.manifest_sha:
            raise ValueError('ConMIL manifest bytes changed')
        features, _ = self.bundle.load(row)
        return features

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        return {'contract': self.contract, 'ordered_features': self.meta['records'][row['slide_id']],
                'actual_native_text': self.text.contract, 'dimension': 512}

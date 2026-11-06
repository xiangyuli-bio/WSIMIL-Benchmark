"""Immutable ordered Libra-MIL features, language embeddings and labels."""
from pathlib import Path

import torch

from .data import resolve
from .focus_network import WEIGHTS_SHA
from .io import digest_file, digest_object, read_json, write_json
from .libramil_source import COMMIT, author_text_encoder, encode_original_prompts

KIND = 'libramil_native_conch_classification'


def identity(bundle):
    return digest_object({'classes': bundle.classes, 'fm': bundle.fm,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'libramil'},
        'synthetic': bundle.config.get('synthetic', False)})


def spec(bundle, metadata):
    if (bundle.task_type != 'classification' or bundle.dim != 512
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'):
        raise ValueError('Libra-MIL requires native512D slide classification and patient-safe splits')
    if (metadata.get('schema_version') != 1 or metadata.get('kind') != KIND
            or metadata.get('class_order') != list(bundle.classes)
            or type(metadata.get('synthetic')) is not bool
            or metadata['synthetic'] != bool(bundle.config.get('synthetic', False))):
        raise ValueError('Libra-MIL class order, task or synthetic provenance changed')
    for key in ['feature_preprocessing', 'feature_provenance', 'text_provenance', 'label_semantics']:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError('Libra-MIL requires explicit '+key)
    if metadata.get('alignment_policy') == 'native_conch':
        if (bundle.encoder.get('model') != 'conch_ViT-B-16'
                or bundle.encoder.get('weights_sha256') != WEIGHTS_SHA
                or bundle.encoder.get('feature_preprocessing') != metadata['feature_preprocessing']):
            raise ValueError('Libra-MIL requires aligned native CONCH features, not an arbitrary FM projection')
    elif metadata.get('alignment_policy') != 'synthetic_native_dimension_fixture' or not metadata['synthetic']:
        raise ValueError('Libra-MIL requires actual native alignment or an explicitly synthetic control')
    weights = resolve(bundle.root, metadata['checkpoint']).resolve()
    if digest_file(weights) != WEIGHTS_SHA:
        raise ValueError('Libra-MIL requires the actual registered full CONCH pretrained weights')
    if metadata.get('text_mode') == 'author_RCC':
        if len(bundle.classes) != 3:
            raise ValueError('Libra-MIL released RCC descriptions have exactly three ordered classes')
        if not metadata['synthetic'] and list(bundle.classes) != ['KICH', 'KIRC', 'KIRP']:
            raise ValueError('Libra-MIL released RCC prompt order is KICH/KIRC/KIRP; map clinical labels explicitly')
    elif metadata.get('text_mode') == 'provided_ordered_descriptions':
        a, b = metadata.get('structural_descriptions'), metadata.get('class_descriptions')
        if (not isinstance(a, list) or not a or not isinstance(b, list) or len(b) != len(bundle.classes)
                or any(not isinstance(x, str) or not x.strip() for x in a+b)):
            raise ValueError('Libra-MIL needs every complete ordered structural/class description')
    else:
        raise ValueError('Libra-MIL text source must be explicitly specified')
    return weights


def record(bundle, row):
    feature, _ = bundle.load(row)
    if feature.ndim != 2 or feature.shape[1] != 512 or not len(feature) or not torch.isfinite(feature).all():
        raise ValueError('Libra-MIL needs full finite ordered [N,512] bags')
    path = resolve(bundle.root, row['path']).resolve()
    return {**{k: row[k] for k in ['slide_id', 'global_patient_id', 'patient_id', 'subset', 'target']},
            'path': str(path), 'sha256': digest_file(path), 'patches': len(feature)}


def prepare(bundle, metadata_path, output):
    output = resolve(bundle.root, output)
    if output.exists() or (output.parent/'text_embeddings.pt').exists():
        raise FileExistsError('Libra-MIL input version exists; preserve it and prepare a new directory')
    metadata_path = resolve(bundle.root, metadata_path).resolve()
    metadata = read_json(metadata_path); weights = spec(bundle, metadata)
    if metadata['text_mode'] == 'author_RCC':
        structural, bags, prompts = encode_original_prompts(weights)
    else:
        encoder = author_text_encoder(weights)
        structural = encoder(metadata['structural_descriptions']).detach().cpu()
        bags = encoder(metadata['class_descriptions']).detach().cpu()
        prompts = []
    output.parent.mkdir(parents=True, exist_ok=True)
    text_path = output.parent/'text_embeddings.pt'
    torch.save({'structural': structural, 'bags': bags}, text_path)
    result = {'schema_version': 1, 'kind': KIND, 'source_commit': COMMIT,
        'encoder_identity': identity(bundle), 'metadata': metadata,
        'metadata_path': str(metadata_path), 'metadata_sha256': digest_file(metadata_path),
        'text_path': str(text_path.resolve()), 'text_sha256': digest_file(text_path),
        'text_shapes': [list(structural.shape), list(bags.shape)],
        'prompt_files': {str(p): digest_file(p) for p in prompts},
        'records': {r['slide_id']: record(bundle, r) for r in bundle.rows.to_dict('records')},
        'preparer_sha256': digest_file(Path(__file__))}
    result['contract'] = digest_object(result)
    write_json(output, result)
    return output


class LibraInputs:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.meta = read_json(self.path)
        if (self.meta.get('kind') != KIND or self.meta.get('source_commit') != COMMIT
                or self.meta.get('encoder_identity') != identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('contract') != digest_object({k: v for k, v in self.meta.items() if k != 'contract'})):
            raise ValueError('Libra-MIL source, original input or text preparation contract changed')
        self.weights = spec(bundle, self.meta['metadata'])
        self.files = [self.path, Path(self.meta['metadata_path']), Path(self.meta['text_path'])]
        expected = [digest_file(self.path), self.meta['metadata_sha256'], self.meta['text_sha256']]
        for p, sha in self.meta['prompt_files'].items():
            self.files.append(Path(p)); expected.append(sha)
        self.hashes = dict(zip(map(str, self.files), expected))
        self.contract = self.meta['contract']
        self._stats = {}; self._feature_stats = {}
        self.check_unchanged()
        if read_json(self.meta['metadata_path']) != self.meta['metadata']:
            raise ValueError('Libra-MIL actual metadata bytes changed')
        self.text = torch.load(self.meta['text_path'], map_location='cpu', weights_only=True)
        if (set(self.text) != {'structural', 'bags'}
                or [list(self.text[k].shape) for k in ['structural', 'bags']] != self.meta['text_shapes']
                or self.text['structural'].shape[1] != 512
                or self.text['bags'].shape != (len(bundle.classes), 512)
                or any(not torch.isfinite(x).all() for x in self.text.values())):
            raise ValueError('Libra-MIL actual full text tensor shapes/values changed')
        if set(self.meta['records']) != set(bundle.rows.slide_id):
            raise ValueError('Libra-MIL complete slide identity changed')

    def check_unchanged(self):
        for name, sha in self.hashes.items():
            path = Path(name); stat = path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self._stats.get(name) != signature:
                if digest_file(path) != sha:
                    raise ValueError('Libra-MIL immutable manifest/metadata/text/description bytes changed')
                self._stats[name] = signature

    def load(self, row):
        self.check_unchanged()
        expected = self.meta['records'][row['slide_id']]
        path = resolve(self.bundle.root, row['path']).resolve(); stat = path.stat()
        if any(row[k] != expected[k] for k in ['slide_id', 'global_patient_id', 'patient_id', 'subset', 'target']) or str(path) != expected['path']:
            raise ValueError('Libra-MIL slide/patient/label/path binding changed')
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._feature_stats.get(str(path)) != signature:
            if record(self.bundle, row) != expected:
                raise ValueError('Libra-MIL original complete feature bytes/order changed')
            self._feature_stats[str(path)] = signature
        return self.bundle.load(row)[0]

    def inventory(self, row, deep=False):
        if deep:
            self.load(row)
        return {'contract': self.contract, 'full_features': self.meta['records'][row['slide_id']]}

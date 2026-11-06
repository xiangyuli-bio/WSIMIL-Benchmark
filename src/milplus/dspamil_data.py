"""Patient-bound inputs for the complete MICCAI2025 DSPA reconstruction.

The paper does not release its expert texts or explain text-to-UNI alignment.
This contract binds producer-supplied evidence; it never certifies that those
assets are the original paper assets. Held-out slides expose no teacher input.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .feather_assets import native_lock
from .io import digest_file, digest_object, read_json, write_json


KIND = 'DSPA_MICCAI2025_patient_bound_explicit_reference_inputs'


def resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _file(path):
    stat = path.stat()
    sha = digest_file(path)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) != (
            after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
        raise ValueError('DSPA input file changed while hashing its actual contents')
    return {'sha256': sha, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def identity(bundle):
    return {'task_id': bundle.config['task_id'], 'task_type': bundle.task_type,
        'classes': bundle.classes, 'fm': bundle.fm, 'dimension': bundle.dim,
        'split': bundle.split, 'rows': bundle.rows.to_dict('records'),
        'training_unit': bundle.config.get('training_unit', 'slide'),
        'evaluation_unit': bundle.config.get('evaluation_unit', 'slide')}


def visual_identity(value):
    """Require native UNI weights, not merely a1024-column tensor."""
    native = native_lock()['native_encoders']['uni']
    sha = next(iter(native['files'].values()))['sha256']
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or value.get('encoder_id') != 'uni' or value.get('feature_dim') != 1024
            or value.get('model_id') != native['model_id']
            or value.get('revision') != native['revision']
            or value.get('weights_sha256') != sha
            or not _text(value.get('extraction_provenance'))):
        raise ValueError('DSPA requires native UNI visual identity and extraction provenance; shape alone is insufficient')
    return value


def _concepts(root, specification, visual):
    """Read all twelve pattern and six grade embeddings and their evidence."""
    keys = {'pattern_embeddings', 'grade_embeddings', 'pattern_texts', 'grade_texts',
            'text_encoder', 'alignment', 'provenance'}
    if (not isinstance(specification, dict) or set(specification) != keys
            or not _text(specification['provenance'])):
        raise ValueError('Declare all original-concept gaps and producer-supplied concept provenance')
    patterns, grades = specification['pattern_texts'], specification['grade_texts']
    if (not isinstance(patterns, list) or len(patterns) != 3
            or any(not isinstance(row, dict) or set(row) != {'pattern', 'texts'}
                or type(row['pattern']) is not int or row['pattern'] != i + 3
                or not isinstance(row['texts'], list) or len(row['texts']) != 4
                or any(not _text(text) for text in row['texts'])
                or len(set(row['texts'])) != 4 for i, row in enumerate(patterns))
            or not isinstance(grades, list) or len(grades) != 6
            or any(not isinstance(row, dict) or set(row) != {'grade', 'text'}
                or type(row['grade']) is not int or row['grade'] != i
                or not _text(row['text']) for i, row in enumerate(grades))
            or len({row['text'] for row in grades}) != 6):
        raise ValueError('DSPA requires ordered four actual texts per pattern3/4/5 and six distinct grade texts0..5')
    encoder = specification['text_encoder']
    if (not isinstance(encoder, dict)
            or set(encoder) != {'model_id', 'revision', 'checkpoint_files', 'provenance'}
            or any(not _text(encoder[key]) for key in ['model_id', 'revision', 'provenance'])
            or not isinstance(encoder['checkpoint_files'], list) or not encoder['checkpoint_files']
            or any(not _text(path) for path in encoder['checkpoint_files'])):
        raise ValueError('Bind the actual text encoder version and locally available checkpoint files')
    alignment = specification['alignment']
    if (not isinstance(alignment, dict)
            or set(alignment) != {'target_encoder_id', 'target_weights_sha256', 'procedure',
                                  'evidence_file', 'projection_checkpoint'}
            or alignment['target_encoder_id'] != 'uni'
            or alignment['target_weights_sha256'] != visual['weights_sha256']
            or not _text(alignment['procedure']) or not _text(alignment['evidence_file'])
            or (alignment['projection_checkpoint'] is not None
                and not _text(alignment['projection_checkpoint']))):
        raise ValueError('DSPA text-to-UNI alignment needs an explicit procedure and evidence; dimensions do not prove alignment')
    paths = [resolve(root, specification[key]) for key in ['pattern_embeddings', 'grade_embeddings']]
    if any(path.suffix != '.npy' for path in paths) or len(set(paths)) != 2:
        raise ValueError('Supply separate non-pickle NPY pattern and grade concept arrays')
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    for array, shape in zip(arrays, [(3, 4, 1024), (6, 1024)]):
        if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError('DSPA concepts must retain all finite float32 native3x4x1024 and6x1024 embeddings')
    paths += [resolve(root, path) for path in encoder['checkpoint_files']]
    paths.append(resolve(root, alignment['evidence_file']))
    if alignment['projection_checkpoint'] is not None:
        paths.append(resolve(root, alignment['projection_checkpoint']))
    if len(paths) != len(set(paths)):
        raise ValueError('DSPA concept, text-weight and alignment evidence paths must identify distinct files')
    return {'pattern_embeddings': str(paths[0]), 'grade_embeddings': str(paths[1]),
            'files': {str(path): _file(path) for path in paths},
            'pattern_texts': patterns, 'grade_texts': grades,
            'specification': specification, 'producer_alignment_assertion_only': True,
            'original_paper_concepts_independently_verified': False}


def _population(bundle, grade_map):
    if (bundle.task_type != 'ordinal' or bundle.dim != 1024
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') not in {'slide', 'patient'}
            or not isinstance(grade_map, dict) or set(grade_map) != {str(i) for i in range(6)}
            or bundle.classes != [grade_map[str(i)] for i in range(6)]):
        raise ValueError('DSPA needs six explicitly ordered Grade Groups0..5, native1024D bags and slide training')
    records, patients, paths, hashes = {}, {}, set(), {}
    for row in bundle.rows.to_dict('records'):
        subset, patient, slide = row['subset'], row['global_patient_id'], row['slide_id']
        if (subset not in {'train', 'val', 'test'} or not _text(patient) or not _text(slide)
                or patients.setdefault(patient, subset) != subset):
            raise ValueError('DSPA requires global patient identities disjoint across TRAIN/VAL/TEST')
        if slide in records or type(row['target']) is not int or not 0 <= row['target'] <= 5:
            raise ValueError('DSPA needs unique slide IDs with complete six-grade targets')
        path = resolve(bundle.root, row['path'])
        if path in paths:
            raise ValueError('DSPA slides cannot alias one physical feature file')
        paths.add(path)
        features, unused = bundle.load(row)
        if features.ndim != 2 or features.shape[1] != 1024 or len(features) < 1:
            raise ValueError('DSPA needs complete nonempty native1024D patch bags')
        info = _file(path)
        if hashes.setdefault(info['sha256'], subset) != subset:
            raise ValueError('DSPA byte-identical bags cross patient partitions')
        records[slide] = {'path': str(path), **info, 'instances': len(features),
            'dimension': 1024, 'subset': subset, 'global_patient_id': patient,
            'grade_group': row['target'], 'row': row}
    if set(patients.values()) != {'train', 'val', 'test'}:
        raise ValueError('DSPA requires nonempty TRAIN, validation and locked test populations')
    return records


def _patterns(path, records):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = {'slide_id', 'global_patient_id', 'grade_group', 'pattern_3', 'pattern_4',
               'pattern_5', 'annotation_provenance'}
    train = {key for key, row in records.items() if row['subset'] == 'train'}
    if (set(table) != columns or table.slide_id.duplicated().any()
            or set(table.slide_id) != train or table.eq('').any().any()):
        raise ValueError('DSPA measured pattern table must cover exactly TRAIN slides, never held-out slides')
    output = {}
    for row in table.to_dict('records'):
        expected = records[row['slide_id']]
        if (row['global_patient_id'] != expected['global_patient_id']
                or row['grade_group'] != str(expected['grade_group'])
                or any(row[f'pattern_{k}'] not in {'0', '1'} for k in [3, 4, 5])
                or not _text(row['annotation_provenance'])):
            raise ValueError('DSPA needs measured binary pattern targets matched to the true TRAIN patient and grade')
        output[row['slide_id']] = {**row,
            'targets': [int(row[f'pattern_{k}']) for k in [3, 4, 5]]}
    return output


def prepare(bundle, schema, output):
    keys = {'schema_version', 'grade_map', 'visual_extraction_manifest',
            'concepts', 'training_pattern_table', 'provenance'}
    if (not isinstance(schema, dict) or set(schema) != keys or schema['schema_version'] != 1
            or not _text(schema['provenance'])):
        raise ValueError('Declare the complete DSPA visual/concept/pattern/patient input schema')
    if 'dspamil' in bundle.encoder:
        raise ValueError('Prepare DSPA before adding encoder.dspamil.manifest')
    visual_path = resolve(bundle.root, schema['visual_extraction_manifest'])
    visual = visual_identity(read_json(visual_path))
    concepts = _concepts(bundle.root, schema['concepts'], visual)
    records = _population(bundle, schema['grade_map'])
    pattern_path = resolve(bundle.root, schema['training_pattern_table'])
    patterns = _patterns(pattern_path, records)
    value = {'schema_version': 1, 'kind': KIND, 'identity': identity(bundle),
        'schema': schema, 'visual_metadata': visual, 'concepts': concepts,
        'records': records, 'training_patterns': patterns,
        'files': {str(visual_path): _file(visual_path), str(pattern_path): _file(pattern_path),
                  **concepts['files']},
        'original_paper_assets_verified': False, 'full_method_validation': False,
        'formal_admission': False}
    value['contract'] = digest_object(value)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json(output, value)
    return output


class DSPAInputs:
    def __init__(self, bundle):
        config = bundle.encoder.get('dspamil')
        if not isinstance(config, dict) or set(config) != {'manifest'}:
            raise ValueError('Bind encoder.dspamil.manifest to the complete DSPA input contract')
        self.bundle = bundle
        self.path = resolve(bundle.root, config['manifest'])
        value = read_json(self.path)
        self.contract = value.pop('contract', None)
        if (self.contract != digest_object(value) or value.get('schema_version') != 1
                or value.get('kind') != KIND or value.get('identity') != identity(bundle)
                or value.get('original_paper_assets_verified') is not False
                or value.get('full_method_validation') is not False
                or value.get('formal_admission') is not False):
            raise ValueError('DSPA reference input identity, provenance or patient contract changed')
        if _population(bundle, value['schema']['grade_map']) != value['records']:
            raise ValueError('DSPA actual patient/feature population changed')
        schema = value['schema']
        visual_path = resolve(bundle.root, schema['visual_extraction_manifest'])
        visual = visual_identity(read_json(visual_path))
        concepts = _concepts(bundle.root, schema['concepts'], visual)
        pattern_path = resolve(bundle.root, schema['training_pattern_table'])
        if (visual != value['visual_metadata'] or concepts != value['concepts']
                or _patterns(pattern_path, value['records']) != value['training_patterns']
                or {str(visual_path): _file(visual_path), str(pattern_path): _file(pattern_path),
                    **concepts['files']} != value['files']):
            raise ValueError('DSPA actual concept alignment, pattern supervision or evidence files changed')
        self.value, self.records = value, value['records']
        self.files = [self.path, *[Path(path) for path in value['files']]]
        self.verify(deep=True)
        concepts = value['concepts']
        self._pattern_concepts = torch.from_numpy(np.load(concepts['pattern_embeddings'], allow_pickle=False).copy())
        self._grade_concepts = torch.from_numpy(np.load(concepts['grade_embeddings'], allow_pickle=False).copy())

    @property
    def pattern_concepts(self):
        return self._pattern_concepts.clone()

    def verify(self, deep=False):
        if (read_json(self.path) != {**self.value, 'contract': self.contract}
                or self.value['identity'] != identity(self.bundle)):
            raise ValueError('DSPA task labels, global patients or input binding changed')
        visual_identity(self.value['visual_metadata'])
        for path, expected in self.value['files'].items():
            stat = Path(path).stat()
            if (stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']
                    or (deep and digest_file(path) != expected['sha256'])):
                raise ValueError('DSPA actual concept, pattern-label or provenance evidence changed')
        for row in self.bundle.rows.to_dict('records'):
            self._record(row, deep=deep)

    def _record(self, row, *, deep=False):
        expected = self.records.get(row['slide_id'])
        if expected is None or expected['row'] != row:
            raise ValueError('DSPA actual slide row, patient membership or grade changed')
        path = resolve(self.bundle.root, row['path'])
        stat = path.stat()
        if (str(path) != expected['path'] or stat.st_size != expected['bytes']
                or stat.st_mtime_ns != expected['mtime_ns']
                or (deep and digest_file(path) != expected['sha256'])):
            raise ValueError('DSPA complete feature bag changed')
        return expected

    def load_features(self, row):
        actual = self._record(row, deep=True)
        features, unused = self.bundle.load(row)
        self._record(row, deep=True)
        if features.shape != (actual['instances'], 1024) or not torch.isfinite(features).all():
            raise ValueError('DSPA actual feature values, instance count or native dimension changed during loading')
        return features

    def training_supervision(self, row):
        """Only actual TRAIN rows can select a ground-truth grade embedding."""
        actual = self._record(row, deep=True)
        if actual['subset'] != 'train':
            raise ValueError('DSPA held-out slides must not consume true-grade concepts or auxiliary pattern targets')
        targets = self.value['training_patterns'][row['slide_id']]['targets']
        return (torch.tensor(actual['grade_group'], dtype=torch.long),
                torch.tensor(targets, dtype=torch.long),
                self._grade_concepts[actual['grade_group']].clone())

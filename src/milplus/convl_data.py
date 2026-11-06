"""Immutable full native CONCH patient bags and explicit four-bin survival data."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .convl_source import COMMIT, original_concepts
from .data import resolve
from .focus_network import WEIGHTS_SHA
from .io import digest_file, digest_object, read_json, write_json

KIND = 'convl_native_conch_patient_survival'


def _identity(bundle):
    return digest_object({'fm': bundle.fm, 'encoder': {k: v for k, v in bundle.encoder.items() if k != 'convl'},
        'time_unit': bundle.config.get('time_unit'), 'synthetic': bundle.config.get('synthetic', False)})


def _spec(bundle, metadata):
    if (bundle.task_type != 'survival' or bundle.dim != 512
            or bundle.config.get('training_unit') != 'patient'
            or bundle.config.get('evaluation_unit') != 'patient'
            or bundle.config.get('time_unit') not in {'months', 'days'}):
        raise ValueError('ConVL requires native512D patient survival with explicit event/time units')
    if (metadata.get('schema_version') != 1 or metadata.get('kind') != KIND
            or metadata.get('synthetic') != bool(bundle.config.get('synthetic', False))
            or type(metadata.get('synthetic')) is not bool):
        raise ValueError('ConVL requires an explicit immutable native survival metadata specification')
    for key in ['feature_preprocessing', 'feature_provenance', 'patient_order_provenance', 'task_semantics']:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError('ConVL requires explicit '+key)
    policy = metadata.get('alignment_policy')
    if policy == 'native_conch':
        if (bundle.encoder.get('model') != 'conch_ViT-B-16'
                or bundle.encoder.get('weights_sha256') != WEIGHTS_SHA
                or bundle.encoder.get('feature_preprocessing') != metadata['feature_preprocessing']):
            raise ValueError('ConVL requires actual aligned native512 CONCH features, not an arbitrary FM projection')
    elif policy == 'synthetic_native_dimension_fixture':
        if not metadata['synthetic']:
            raise ValueError('Synthetic ConVL features cannot be presented as a clinical task')
    else:
        raise ValueError('ConVL shared visual/text embedding space must be explicit')
    concepts, cohort = original_concepts(metadata.get('concept_cohort'))
    weights = resolve(bundle.root, metadata['checkpoint']).resolve()
    if digest_file(weights) != WEIGHTS_SHA:
        raise ValueError('ConVL requires actual original pretrained CONCH checkpoint bytes')
    patients = bundle.rows.drop_duplicates('global_patient_id')
    train = patients.loc[patients.subset == 'train']
    binning = metadata.get('binning', {})
    if not isinstance(binning.get('provenance'), str) or not binning['provenance'].strip():
        raise ValueError('ConVL four-bin survival preparation requires explicit label provenance')
    if binning.get('policy') == 'provided_author_four_bin_labels':
        table = pd.read_csv(resolve(bundle.root, binning['table']), dtype={'global_patient_id': str})
        required = {'global_patient_id', 'bin', 'time', 'event'}
        if (set(table) != required or table.isna().any().any()
                or table.global_patient_id.duplicated().any()
                or set(table.global_patient_id) != set(patients.global_patient_id)):
            raise ValueError('ConVL original discrete label table must cover each true patient exactly once')
        if (not pd.to_numeric(table.bin).between(0, 3).all()
                or not (pd.to_numeric(table.bin) % 1 == 0).all()):
            raise ValueError('ConVL original provided labels must be integer four-bin IDs0..3')
        labels = {}
        indexed = table.set_index('global_patient_id')
        for row in patients.to_dict('records'):
            original = indexed.loc[row['global_patient_id']]
            if float(original.time) != float(row['time']) or float(original.event) != int(row['event']):
                raise ValueError('ConVL supplied original bins disagree with actual patient time/event')
            labels[row['global_patient_id']] = {'bin': int(original.bin), 'time': float(row['time']),
                'event': int(row['event']), 'subset': row['subset']}
        return weights, concepts, cohort, None, train.global_patient_id.tolist(), labels
    if binning.get('policy') != 'train_observed_event_quartiles':
        raise ValueError('ConVL needs original provided labels or explicitly named TRAIN-event quartiles')
    # Author code receives disc_label from an external split CSV, but does not
    # provide its generation protocol. This named data-preparation extension
    # uses TRAIN observed events only and open terminal bounds, never TEST times.
    observed = train.loc[train.event == 1, 'time'].to_numpy(dtype=float)
    if len(observed) < 4:
        raise ValueError('ConVL four-bin TRAIN event discretization needs at least four observed events')
    cuts = np.quantile(observed, [.25, .5, .75])
    if not np.isfinite(cuts).all() or len(set(cuts)) != 3 or min(cuts) <= 0:
        raise ValueError('ConVL requires four distinct TRAIN-derived survival intervals')
    labels = {r['global_patient_id']: {'bin': int(np.searchsorted(cuts, r['time'], side='right')),
        'time': float(r['time']), 'event': int(r['event']), 'subset': r['subset']}
        for r in patients.to_dict('records')}
    return weights, concepts, cohort, cuts.tolist(), train.global_patient_id.tolist(), labels


def _record(bundle, row):
    features, _ = bundle.load(row)
    if (features.ndim != 2 or not len(features) or features.shape[1] != 512
            or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError('ConVL needs every original finite [N,512] patch feature')
    path = resolve(bundle.root, row['path']).resolve()
    return {'path': str(path), 'sha256': digest_file(path), 'patches': len(features),
        **{key: row[key] for key in ['global_patient_id', 'subset', 'time', 'event']}}


def prepare(bundle, metadata_path, output):
    output = resolve(bundle.root, output)
    if output.exists():
        raise FileExistsError('ConVL input version exists; preserve it and prepare a new manifest')
    metadata_path = resolve(bundle.root, metadata_path).resolve()
    metadata = read_json(metadata_path)
    weights, concepts, cohort, cuts, train, labels = _spec(bundle, metadata)
    extra_files = []
    if metadata['binning']['policy'] == 'provided_author_four_bin_labels':
        extra_files.append(resolve(bundle.root, metadata['binning']['table']).resolve())
    result = {'schema_version': 1, 'kind': KIND, 'source_commit': COMMIT,
        'encoder_identity': _identity(bundle), 'metadata': metadata, 'metadata_path': str(metadata_path),
        'metadata_sha256': digest_file(metadata_path), 'checkpoint': str(weights), 'weights_sha256': WEIGHTS_SHA,
        'concepts_path': str(concepts), 'concepts_sha256': digest_file(concepts), 'concept_cohort': cohort,
        'finite_time_cuts': cuts, 'terminal_bound': '+infinity' if cuts is not None else 'author_predefined_not_inferred',
        'lower_bound': 0. if cuts is not None else 'author_predefined_not_inferred',
        'binning_files': {str(p): digest_file(p) for p in extra_files},
        'training_patients': train, 'patients': labels,
        'records': {row['slide_id']: _record(bundle, row) for row in bundle.rows.to_dict('records')},
        'preparer_sha256': digest_file(Path(__file__))}
    result['contract'] = digest_object(result)
    write_json(output, result)
    return output


class ConVLInputs:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest)
        self.meta = read_json(self.path); self.manifest_sha = digest_file(self.path)
        if (self.meta.get('kind') != KIND or self.meta.get('source_commit') != COMMIT
                or self.meta.get('encoder_identity') != _identity(bundle)
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('contract') != digest_object({k: v for k, v in self.meta.items() if k != 'contract'})):
            raise ValueError('ConVL source, patient or original feature input contract changed')
        self.weights, concepts, cohort, cuts, train, labels = _spec(bundle, self.meta['metadata'])
        metadata_path = Path(self.meta['metadata_path'])
        if (digest_file(metadata_path) != self.meta['metadata_sha256']
                or read_json(metadata_path) != self.meta['metadata']
                or str(concepts) != self.meta['concepts_path'] or digest_file(concepts) != self.meta['concepts_sha256']
                or cohort != self.meta['concept_cohort'] or cuts != self.meta['finite_time_cuts']
                or train != self.meta['training_patients'] or labels != self.meta['patients']
                or set(self.meta['records']) != set(bundle.rows.slide_id)):
            raise ValueError('ConVL bound concepts, labels, TRAIN bins or metadata changed')
        self.contract = self.meta['contract']
        self.files = [self.path, metadata_path, concepts]
        for path, expected in self.meta['binning_files'].items():
            if digest_file(path) != expected:
                raise ValueError('ConVL original author discrete label bytes changed')
            self.files.append(Path(path))
        self.file_hashes = {str(p): digest_file(p) for p in self.files}
        stat = self.weights.stat(); self.weight_stat = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        self._verified = {}

    def check_unchanged(self):
        if any(digest_file(path) != expected for path, expected in self.file_hashes.items()):
            raise ValueError('ConVL immutable manifest, concepts or metadata bytes changed')
        stat = self.weights.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != self.weight_stat:
            raise ValueError('ConVL actual pretrained language weights changed')

    def inventory(self, row, deep=False):
        self.check_unchanged()
        if deep and _record(self.bundle, row) != self.meta['records'][row['slide_id']]:
            raise ValueError('ConVL original feature inventory changed')
        return {'contract': self.contract, 'original_full_features': self.meta['records'][row['slide_id']],
                'concept_cohort': self.meta['concept_cohort'], 'native_dimension': 512,
                'patient_label': self.meta['patients'][row['global_patient_id']]}

    def patient(self, rows, device='cpu'):
        self.check_unchanged()
        if not rows or len({row['global_patient_id'] for row in rows}) != 1:
            raise ValueError('ConVL requires exactly one complete registered patient')
        gid = rows[0]['global_patient_id']
        expected = self.bundle.rows.loc[self.bundle.rows.global_patient_id == gid, 'slide_id'].tolist()
        if [r['slide_id'] for r in rows] != expected:
            raise ValueError('ConVL requires all patient slides in declared stable order')
        features = []
        for row in rows:
            path = resolve(self.bundle.root, row['path']).resolve(); stat = path.stat()
            bound = self.meta['records'][row['slide_id']]
            if (str(path) != bound['path']
                    or any(row[key] != bound[key] for key in ['global_patient_id', 'subset', 'time', 'event'])):
                raise ValueError('ConVL patient identity, event/time or feature path changed')
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self._verified.get(str(path)) != signature:
                if _record(self.bundle, row) != self.meta['records'][row['slide_id']]:
                    raise ValueError('ConVL original full patch features or patient labels changed')
                self._verified[str(path)] = signature
            feature, _ = self.bundle.load(row); features.append(feature)
        label = self.meta['patients'][gid]
        return torch.cat(features).to(device), torch.tensor([label['bin']], device=device), torch.tensor([1-label['event']], device=device)

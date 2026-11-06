"""Versioned tissue prototypes in the SAME FM space as the patient bags."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_array, resolve
from .io import digest_file, digest_object, write_json


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'encoder': {k: v for k, v in bundle.encoder.items()
                         if k not in {'index', 'pamoe'}}})


class TissuePrototypes:
    def __init__(self, bundle):
        config = bundle.encoder['pamoe']
        path = resolve(bundle.root, config['manifest'])
        self.metadata = json.loads(path.read_text())
        m = self.metadata
        if m.get('schema_version') != 1 or m.get('kind') not in {'tissue_mean', 'cluster', 'external'}:
            raise ValueError('PAMoE requires a versioned tissue_mean/cluster/external prototype manifest')
        if m.get('encoder_sha256') != encoder_identity(bundle):
            raise ValueError('PAMoE prototype encoder/preprocessing identity mismatch')
        if not isinstance(m.get('provenance'), str) or not m['provenance'].strip():
            raise ValueError('PAMoE prototype provenance is required')
        if m.get('synthetic', False) and not bundle.config.get('synthetic', False):
            raise ValueError('Synthetic PAMoE prototypes cannot be used in a real task')
        fit = m.get('fit_scope')
        if fit not in {'training_patients', 'external_disjoint', 'external_unverified'}:
            raise ValueError('Declare PAMoE prototype fit_scope explicitly')
        patients = m.get('fit_patient_ids')
        if not isinstance(patients, list) or any(not isinstance(x, str) or not x for x in patients) or len(set(patients)) != len(patients):
            raise ValueError('PAMoE fit_patient_ids must be a unique string list')
        train_ids = {r['global_patient_id'] for r in bundle.records('train')}
        all_ids = set(bundle.rows.global_patient_id)
        if fit == 'training_patients' and (not patients or not set(patients) <= train_ids):
            raise ValueError('PAMoE prototype patients leak across the training split')
        if fit == 'external_disjoint' and (not patients or set(patients) & all_ids):
            raise ValueError('PAMoE external-disjoint prototype patients overlap or are not documented')
        tensor_path = resolve(path.parent, m['tensor'])
        if digest_file(tensor_path) != m.get('sha256'):
            raise ValueError('PAMoE prototype tensor hash mismatch')
        values = load_array(tensor_path)
        names = m.get('names')
        if (not isinstance(names, list) or not names or
                any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names) or
                values.shape != (len(names), bundle.dim) or not np.isfinite(values).all() or
                (np.linalg.norm(values, axis=1) == 0).any()):
            raise ValueError('PAMoE prototypes need unique names and finite nonzero [experts,FM_dim] vectors')
        self.tensor = torch.as_tensor(values, dtype=torch.float32).clone()
        if not torch.isfinite(self.tensor).all():
            raise ValueError('PAMoE prototype values overflow float32')
        self.files = [path, tensor_path]
        self.training_inputs = m.get('training_inputs', [])
        if fit == 'training_patients' and m['kind'] == 'tissue_mean':
            training = {r['slide_id']: r for r in bundle.records('train')
                        if r['global_patient_id'] in patients}
            if (not isinstance(self.training_inputs, list) or not self.training_inputs or
                    len(self.training_inputs) != len(training) or
                    {row.get('slide_id') for row in self.training_inputs} != set(training) or
                    any(row.get('patient_id') != training[row['slide_id']]['global_patient_id'] for row in self.training_inputs)):
                raise ValueError('PAMoE tissue-mean asset requires its complete training feature ledger')

    def check_inventory(self, records):
        inventory = {row['slide_id']: row for row in records}
        if self.metadata['fit_scope'] == 'training_patients':
            for row in self.training_inputs:
                if inventory[row['slide_id']]['sha256'] != row.get('feature_sha256'):
                    raise ValueError('PAMoE prototype training features changed; rebuild the tissue means')


def build_tissue_prototypes(bundle, assignments, output, names, provenance):
    """Source per-tissue mean, restricted to train slides with checked patch order.

    assignments is a CSV: slide_id,path,sha256,coords_sha256. Arrays contain
    integer patch categories in [0,K). The upstream classifier is external and
    must be identified by provenance; this function never invents tissue labels.
    """
    import pandas as pd
    assignments = Path(assignments).resolve()
    table = pd.read_csv(assignments, dtype=str, keep_default_na=False)
    if (not {'slide_id', 'path', 'sha256', 'coords_sha256'} <= set(table) or
            table.slide_id.duplicated().any()):
        raise ValueError('PAMoE assignment index needs unique slide IDs, arrays and content/coordinate hashes')
    rows = sorted(bundle.records('train'), key=lambda r: r['slide_id'])
    if set(table.slide_id) != {r['slide_id'] for r in rows}:
        raise ValueError('PAMoE assignment index must contain exactly the training slides')
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names) or not provenance:
        raise ValueError('Explicit tissue names and classifier provenance are required')
    table = table.set_index('slide_id')
    sums = np.zeros((len(names), bundle.dim), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    ledger = []
    for row in rows:
        x, coords = bundle.load(row, require_coords=True)
        item = table.loc[row['slide_id']]
        labels_path = resolve(assignments.parent, item.path)
        if digest_file(labels_path) != item.sha256:
            raise ValueError('PAMoE tissue assignment hash mismatch')
        # Coordinate tensor is normalized through Bundle.load, exactly like feature reads.
        coord_hash = digest_object(coords.tolist())
        if coord_hash != item.coords_sha256:
            raise ValueError('PAMoE assignment/feature coordinate order mismatch')
        labels = load_array(labels_path, 'labels')
        if labels.shape != (len(x),) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= len(names)).any():
            raise ValueError('PAMoE assignments must contain one valid integer tissue category per patch')
        for k in range(len(names)):
            sums[k] += x.numpy()[labels == k].sum(axis=0, dtype=np.float64)
            counts[k] += int((labels == k).sum())
        ledger.append({'slide_id': row['slide_id'], 'patient_id': row['global_patient_id'],
                       'feature_sha256': digest_file(resolve(bundle.root, row['path'])),
                       'assignments_sha256': item.sha256, 'coords_sha256': coord_hash})
    if (counts == 0).any():
        raise ValueError('PAMoE cannot construct a tissue prototype with no training patches')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    array = (sums/counts[:, None]).astype(np.float32)
    if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError('PAMoE produced a nonfinite or zero-norm tissue prototype')
    np.save(output/'prototypes.npy', array)
    manifest = {'schema_version': 1, 'kind': 'tissue_mean', 'tensor': 'prototypes.npy',
                'sha256': digest_file(output/'prototypes.npy'), 'names': names,
                'encoder_sha256': encoder_identity(bundle), 'fit_scope': 'training_patients',
                'fit_patient_ids': sorted({r['global_patient_id'] for r in rows}),
                'provenance': provenance, 'synthetic': bool(bundle.config.get('synthetic', False)),
                'counts': counts.tolist(), 'training_inputs': ledger,
                'difference_from_source': 'stream all train slides in float64; reject missing/misaligned tissue arrays'}
    write_json(output/'manifest.json', manifest)
    return output/'manifest.json'


def add_synthetic_prototypes(data):
    """Explicit synthetic asset for software tests; never a fallback on real data."""
    import pandas as pd
    import yaml

    from .data import Bundle
    bundle = Bundle(data)
    if not bundle.config.get('synthetic'):
        raise ValueError('Synthetic PAMoE helper requires a synthetic task')
    root = Path(data)/'synthetic_tissues'
    root.mkdir(exist_ok=False)
    records = []
    for row in bundle.records('train'):
        x, coords = bundle.load(row, require_coords=True)
        path = root/(row['slide_id']+'.npy')
        np.save(path, np.arange(len(x), dtype=np.int64) % 4)
        records.append({'slide_id': row['slide_id'], 'path': path.name, 'sha256': digest_file(path),
                        'coords_sha256': digest_object(coords.tolist())})
    index = root/'index.csv'
    pd.DataFrame(records).to_csv(index, index=False)
    manifest = build_tissue_prototypes(bundle, index, root/'asset', [f'synthetic_tissue_{i}' for i in range(4)],
                                      'synthetic cyclic patch assignments; not biological tissue classification')
    bundle.encoder['pamoe'] = {'manifest': str(manifest.relative_to(bundle.root))}
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(bundle.config))
    return data

"""Versioned original precomputed MGGM-MIL graphs; no guessed graph builder."""
from pathlib import Path
import json

import numpy as np
import torch

from .io import digest_file, digest_object
from .data import resolve
from .spatial_attention import spatial_metadata

SCALES = (4, 8, 12)


class GraphInputs:
    def __init__(self, bundle):
        if (bundle.task_type != 'classification' or len(bundle.classes) != 2 or bundle.dim != 512
                or bundle.config.get('training_unit', 'slide') != 'slide'
                or bundle.config.get('evaluation_unit', 'slide') != 'slide'):
            raise ValueError('Original MGGM-MIL requires binary single-slide512D bags and slide evaluation')
        self.bundle = bundle
        self.spatial = spatial_metadata(bundle)
        declaration = bundle.encoder.get('mggmmil')
        if not isinstance(declaration, dict) or set(declaration) != {'manifest'}:
            raise ValueError('MGGM-MIL needs a provenance-bound original graph manifest')
        self.manifest = resolve(bundle.root, declaration['manifest'])
        raw = json.loads(self.manifest.read_text())
        if (raw.get('schema_version') != 1 or raw.get('kind') != 'mggmmil_original_precomputed_graphs'
                or raw.get('node_order') != 'stored_feature_rows'
                or raw.get('edge_layout') != 'E_by_2_source_target'
                or raw.get('scales') != list(SCALES)
                or raw.get('imaging_modality') not in {'UAF', 'H&E'}
                or raw.get('feature_encoder') not in {'ResNet18_ImageNet', 'PLIP'}
                or not isinstance(raw.get('graph_provenance'), str) or not raw['graph_provenance'].strip()):
            raise ValueError('Declare original modality/encoder, K4/8/12 graph origin and exact feature-row edge layout')
        rows = bundle.rows.to_dict('records')
        names = [r['slide_id'] for r in rows]
        if not isinstance(raw.get('slides'), dict) or set(raw['slides']) != set(names):
            raise ValueError('MGGM-MIL graphs must bind every exact slide once, with no missing or foreign IDs')
        self.raw = raw
        self.row_binding = [{k: row.get(k) for k in ['slide_id', 'global_patient_id', 'subset', 'target', 'path', 'coords_path']}
                            for row in rows]
        self.files = {str(self.manifest): digest_file(self.manifest)}
        self.graphs = {}
        for row in rows:
            x, coords = bundle.load(row, require_coords=True)
            if (len(x) < 13 or coords.shape != (len(x), 2) or not torch.isfinite(coords).all()
                    or len(torch.unique(coords, dim=0)) != len(x)):
                raise ValueError('Native K12 MGGM-MIL needs at least13 distinct physically aligned nodes')
            record = raw['slides'][row['slide_id']]
            if (not isinstance(record, dict) or set(record) != {'features_sha256', 'coordinates_sha256', 'nodes', 'edges'}
                    or type(record['nodes']) is not int or record['nodes'] != len(x)
                    or set(record['edges']) != {'4', '8', '12'}):
                raise ValueError('MGGM-MIL node count or complete three-scale graph declaration changed')
            feature_path = resolve(bundle.root, row['path'])
            coordinate_path = resolve(bundle.root, row.get('coords_path') or row['path'])
            for path, sha in [(feature_path, record['features_sha256']),
                              (coordinate_path, record['coordinates_sha256'])]:
                if digest_file(path) != sha:
                    raise ValueError('MGGM-MIL graph/feature/coordinate byte binding changed')
                self.files[str(path)] = sha
            graph = []
            for scale in SCALES:
                entry = record['edges'][str(scale)]
                if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
                    raise ValueError('Each native MGGM-MIL graph needs its actual file and SHA256')
                path = resolve(bundle.root, entry['path'])
                if digest_file(path) != entry['sha256']:
                    raise ValueError('Original MGGM-MIL edge bytes changed')
                if path.suffix == '.npy':
                    array = np.load(path, allow_pickle=False)
                    edges = torch.from_numpy(array.copy())
                elif path.suffix in {'.pt', '.pth'}:
                    edges = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    raise ValueError('Use actual int64 original tensor or NumPy edge lists')
                if (not torch.is_tensor(edges) or edges.dtype != torch.int64 or edges.ndim != 2
                        or edges.shape[1] != 2 or len(edges) < 1 or edges.min() < 0 or edges.max() >= len(x)
                        or len(torch.unique(edges, dim=0)) != len(edges)):
                    raise ValueError('MGGM-MIL edges need unique in-range int64 source/target node pairs')
                # Preserve the original list order, directedness, and any
                # documented self edges; the public repository omits its graph
                # builder. Never invent KNN/self-edge semantics for that code.
                graph.append(edges.contiguous())
                self.files[str(path)] = entry['sha256']
            self.graphs[row['slide_id']] = graph
        self.contract = digest_object({'bundle': bundle.fingerprint, 'rows': self.row_binding,
                                       'spatial': self.spatial, 'manifest': raw, 'files': self.files})
        self.stats = {name: (Path(name).stat().st_size, Path(name).stat().st_mtime_ns) for name in self.files}

    def verify(self, deep=False):
        rows = [{k: row.get(k) for k in ['slide_id', 'global_patient_id', 'subset', 'target', 'path', 'coords_path']}
                for row in self.bundle.rows.to_dict('records')]
        if rows != self.row_binding or spatial_metadata(self.bundle) != self.spatial:
            raise ValueError('MGGM-MIL bound patient labels/splits/node order changed')
        for name, sha in self.files.items():
            path = Path(name)
            if (not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != self.stats[name]
                    or deep and digest_file(path) != sha):
                raise ValueError('MGGM-MIL feature, coordinate or graph input bytes changed')

    def load(self, row):
        self.verify()
        x, _ = self.bundle.load(row, require_coords=True)
        return x, [e.clone() for e in self.graphs[row['slide_id']]]

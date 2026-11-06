"""Input-bound TTIS partitions and frozen semantic scores, without target leakage."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.cluster import KMeans as SKKMeans

from .data import load_array, resolve
from .io import digest_file, digest_object, write_json
from .smmile_data import array_identity, positive_int
from .upstream import load_definitions, resource


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'encoder': {k: v for k, v in bundle.encoder.items() if k != 'ttis'}})


def spatial_inputs(bundle, row):
    x, _ = bundle.load(row, require_coords=True)
    xy = np.asarray(load_array(resolve(bundle.root, row.get('coords_path') or row['path']), 'coords'))
    if xy.shape != (len(x), 2) or xy.dtype.kind not in 'iuf' or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 2**53-1).any() or len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('TTIS requires finite, nonnegative, unique real physical coordinates in original feature order')
    return x, xy


def validate_clusters(clusters, count):
    if not isinstance(clusters, dict) or not clusters:
        raise ValueError('TTIS clusters must be a nonempty indexed partition')
    flattened = []
    for key, indices in clusters.items():
        if not isinstance(key, str) or not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices):
            raise ValueError('TTIS cluster lists require integer original patch indices')
        flattened.extend(indices)
    if sorted(flattened) != list(range(count)):
        raise ValueError('TTIS clusters duplicate, omit or exceed original feature rows')


def semantic_metadata(path, bundle):
    path = Path(path).resolve(); meta = json.loads(path.read_text())
    if meta.get('schema_version') != 1 or meta.get('classes') != bundle.classes:
        raise ValueError('TTIS semantic metadata must match exact ordered task classes')
    if bool(meta.get('synthetic', False)) != bool(bundle.config.get('synthetic', False)):
        raise ValueError('TTIS semantic/task synthetic markers differ')
    for key in ['model', 'weights_sha256', 'embedding_space', 'visual_embedding_space', 'provenance']:
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError('TTIS semantic metadata requires '+key)
    if len(meta['weights_sha256']) != 64 or any(c not in '0123456789abcdef' for c in meta['weights_sha256']):
        raise ValueError('TTIS semantic weights require SHA256 provenance')
    dimension = positive_int(meta.get('dimension'), 'semantic dimension')
    text = resolve(path.parent, meta['text_path']).resolve()
    embeddings = np.asarray(load_array(text))
    prompts, classes = meta.get('prompts'), meta.get('prompt_class_indices')
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension or embeddings.dtype.kind not in 'f' or not len(embeddings) or not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError('TTIS text vectors require finite nonzero [prompts,semantic_dimension]')
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1., rtol=.005, atol=.005):
        raise ValueError('TTIS MUSK text vectors must preserve native normalized outputs')
    if not isinstance(prompts, list) or len(prompts) != len(embeddings) or any(not isinstance(t, str) or not t.strip() for t in prompts):
        raise ValueError('TTIS text vector rows must retain their exact prompts')
    if not isinstance(classes, list) or len(classes) != len(prompts) or any(type(v) is not int or not 0 <= v < len(bundle.classes) for v in classes) or set(classes) != set(range(len(bundle.classes))):
        raise ValueError('TTIS needs prompt-class mapping covering every task class')
    scale = meta.get('logit_scale')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('TTIS original MUSK multiplier must be finite positive')
    if meta.get('text_sha256') and digest_file(text) != meta['text_sha256']:
        raise ValueError('TTIS semantic text hash mismatch')
    return {**meta, 'text_path': str(text), 'text_sha256': digest_file(text)}


def aligned_guidance(path, coordinates, dimension):
    guidance = np.asarray(load_array(path))
    coords = np.asarray(load_array(path, 'coords'))
    if guidance.shape != (len(coordinates), dimension) or guidance.dtype.kind not in 'f' or not np.isfinite(guidance).all() or coords.dtype.kind not in 'iuf' or not np.array_equal(coords, coordinates):
        raise ValueError('TTIS secondary visual features must match exact primary patch rows/coordinates')
    if (np.linalg.norm(guidance, axis=1) == 0).any():
        raise ValueError('TTIS semantic visual features must be nonzero')
    return torch.tensor(guidance, dtype=torch.float32)


class TTISAssets:
    def __init__(self, bundle):
        if bundle.task_type != 'classification' or bundle.raw_images is not None or bundle.raw_regions is not None:
            raise ValueError('TTIS assets require frozen-feature classification bags')
        self.bundle = bundle
        self.path = resolve(bundle.root, bundle.encoder['ttis']['manifest'])
        self.meta = json.loads(self.path.read_text())
        m = self.meta
        if m.get('schema_version') != 1 or m.get('encoder_sha256') != encoder_identity(bundle) or m.get('classes') != bundle.classes:
            raise ValueError('TTIS manifest encoder or task identity differs')
        if m.get('coordinate_frame') not in {'level0_pixels', 'micrometers'}:
            raise ValueError('TTIS requires declared physical coordinate units')
        if m.get('source_commit') != resource('sources.lock.json')['ttis']['commit']:
            raise ValueError('TTIS asset source revision differs')
        self.rows = m['slides']
        if set(self.rows) != {r['slide_id'] for r in bundle.records('test')}:
            raise ValueError('TTIS assets must cover exactly TEST slides, without target values')
        self.policy = m.get('semantic_policy')
        if self.policy not in {'source_primary', 'aligned_secondary'}:
            raise ValueError('TTIS semantic space policy must be explicit')
        # Validate the same metadata without creating or mutating any model state.
        semantic_path = resolve(self.path.parent, m['semantic_metadata'])
        self.semantic = semantic_metadata(semantic_path, bundle)
        if digest_file(semantic_path) != m['semantic_metadata_sha256']:
            raise ValueError('TTIS semantic registration changed')
        self.files = [self.path, semantic_path, Path(self.semantic['text_path'])]
        self.dimension = self.semantic['dimension']
        if self.policy == 'source_primary' and self.dimension != bundle.dim:
            raise ValueError('Source TTIS dots primary features directly with text; dimensions must match')
        if self.policy == 'aligned_secondary' and self.semantic['embedding_space'] != self.semantic['visual_embedding_space']:
            raise ValueError('Aligned TTIS requires matching declared visual/text embedding spaces')
        self.text = torch.tensor(load_array(self.semantic['text_path']), dtype=torch.float32)

    def load(self, row):
        if digest_file(self.semantic['text_path']) != self.semantic['text_sha256']:
            raise ValueError('TTIS frozen semantic vectors changed')
        asset = self.rows[row['slide_id']]
        x, xy = spatial_inputs(self.bundle, row)
        if digest_file(resolve(self.bundle.root, row['path'])) != asset['feature_sha256'] or array_identity(xy) != asset['coordinates']:
            raise ValueError('TTIS prepared feature/coordinate row identity changed')
        validate_clusters(asset['clusters'], len(x))
        if self.policy == 'aligned_secondary':
            path = resolve(self.path.parent, asset['guidance_path'])
            if digest_file(path) != asset['guidance_sha256']:
                raise ValueError('TTIS secondary semantic visual features changed')
            guidance = aligned_guidance(path, xy, self.dimension)
        else:
            guidance = x
        return asset['clusters'], guidance

    def inventory(self, row, deep):
        if row['subset'] != 'test':
            return {'used': False, 'scope': 'TRAIN and VAL always use full bags'}
        asset = self.rows[row['slide_id']]
        result = {'used': True, 'prepared': digest_object(asset), 'semantic_policy': self.policy}
        if deep:
            clusters, _ = self.load(row)
            result.update(clusters=len(clusters), original_instances=sum(map(len, clusters.values())))
        return result


def hierarchical_clusters(x, coords, clusters, output, slide, backend, seed=2025, n_init=10):
    positive_int(clusters, 'n_clusters'); positive_int(n_init, 'n_init')
    if math.isqrt(clusters)**2 != clusters or len(x) < math.isqrt(clusters):
        raise ValueError('TTIS requires square cluster budget and sufficient real spatial instances')
    if backend == 'cuml':
        if not torch.cuda.is_available():
            raise ValueError('Native TTIS cuML clustering requires CUDA; no implicit CPU substitution')
        try:
            import cupy as cp
            from cuml.cluster import KMeans
        except ImportError as error:
            raise ImportError('Native TTIS clustering requires compatible cuML/CuPy CUDA dependencies') from error
        x, coords = x.cuda(), torch.as_tensor(coords, device='cuda')
    elif backend == 'sklearn_cpu':
        class KMeans:
            def __init__(self, **kwargs):
                kwargs.pop('output_type'); kwargs['init'] = 'k-means++'
                self.estimator = SKKMeans(**kwargs)
            def fit(self, data):
                self.estimator.fit(data); self.labels_ = self.estimator.labels_
        cp = SimpleNamespace(asarray=lambda t: t.detach().cpu().numpy())
        coords = torch.as_tensor(coords)
    else:
        raise ValueError('Explicit TTIS backend must be cuml or sklearn_cpu')
    scope = {'os': os, 'json': json, 'torch': torch, 'Tensor': torch.Tensor, 'math': math, 'np': np,
             'KMeans': KMeans, 'cp': cp, 'is_perfect_square': lambda n: math.isqrt(n)**2 == n}
    fn = load_definitions('ttis', 'src/ttis/clustering.py', ['h_kmeans'], scope).h_kmeans
    result = fn(slide, x[None], torch.as_tensor(coords)[None], clusters, str(output), n_init=n_init, seed=seed)
    validate_clusters(result, len(x)); return result


def prepare_ttis(bundle, output, semantic_path, backend, clusters=256, seed=2025,
                 n_init=10, guidance=None, coordinate_frame='level0_pixels'):
    if bundle.task_type != 'classification' or bundle.config.get('training_unit', 'slide') != 'slide':
        raise ValueError('TTIS supports individual classification slide bags')
    if type(seed) is not int or seed < 0 or coordinate_frame not in {'level0_pixels', 'micrometers'}:
        raise ValueError('TTIS preparation seed/coordinate units are invalid')
    semantic = semantic_metadata(semantic_path, bundle)
    policy = 'aligned_secondary' if guidance is not None else 'source_primary'
    if policy == 'source_primary' and bundle.dim != semantic['dimension']:
        raise ValueError('Source TTIS requires equal primary/text width; supply a real aligned secondary stream otherwise')
    if guidance is not None:
        if semantic['embedding_space'] != semantic['visual_embedding_space']:
            raise ValueError('Aligned TTIS requires matching declared visual/text embedding spaces')
        test_ids = {row['slide_id'] for row in bundle.records('test')}
        if not isinstance(guidance, dict) or set(guidance) != test_ids:
            raise ValueError('TTIS secondary guidance must cover exactly TEST slides')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    registration = output/'semantic.json'; write_json(registration, semantic)
    rows = {}
    for row in bundle.records('test'):
        x, xy = spatial_inputs(bundle, row)
        if coordinate_frame == 'level0_pixels' and not np.equal(xy, np.floor(xy)).all():
            raise ValueError('TTIS level0 pixel coordinates must be integer-valued')
        partition = hierarchical_clusters(x, xy, clusters, output/'source_clusters.json', row['slide_id'], backend, seed, n_init)
        entry = {'feature_sha256': digest_file(resolve(bundle.root, row['path'])), 'coordinates': array_identity(xy), 'clusters': partition}
        if guidance is not None:
            path = Path(guidance[row['slide_id']]).resolve()
            if path == resolve(bundle.root, row['path']).resolve():
                raise ValueError('A secondary visual stream must be independently registered')
            aligned_guidance(path, xy, semantic['dimension'])
            entry.update(guidance_path=str(path), guidance_sha256=digest_file(path))
        rows[row['slide_id']] = entry
    manifest = output/'manifest.json'
    write_json(manifest, {'schema_version': 1, 'source_commit': resource('sources.lock.json')['ttis']['commit'],
        'encoder_sha256': encoder_identity(bundle), 'classes': bundle.classes, 'coordinate_frame': coordinate_frame,
        'semantic_policy': policy, 'semantic_metadata': str(registration), 'semantic_metadata_sha256': digest_file(registration),
        'clustering': {'backend': backend, 'n_clusters': clusters, 'seed': seed, 'n_init': n_init,
                       'initialization': 'k-means||' if backend == 'cuml' else 'explicit CPU k-means++ variant',
                       'sklearn': importlib.metadata.version('scikit-learn') if backend == 'sklearn_cpu' else None},
        'slides': rows})
    return manifest

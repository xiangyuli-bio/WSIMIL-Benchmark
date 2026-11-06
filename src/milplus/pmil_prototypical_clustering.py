"""Two actual affinity-propagation stages, preserving literal source distances."""
import warnings

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances


def similarity(features, lamb=.25):
    x = np.asarray(features)
    if (x.ndim != 2 or x.shape[1] != 512 or len(x) < 2
            or x.dtype != np.float32 or not np.isfinite(x).all()):
        raise ValueError('PMIL AP requires at least two actual finite float32/512D patches')
    dists = pairwise_distances(x, metric='euclidean')
    maximum = np.max(dists)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Original PMIL AP distance normalization is undefined on identical features')
    dists = dists/maximum
    return np.exp(-dists*lamb)


def ap(features):
    matrix = similarity(features)
    # Preserve the source's positive exp(-normalized Euclidean*.25), explicit
    # preference0, damping.5 and seed24. This is neither negative squared
    # Euclidean nor KMeans. Do not silently sample/cap the candidate patches.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = AffinityPropagation(preference=0, damping=.5,
            affinity='precomputed', random_state=24).fit(matrix)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Original PMIL affinity propagation did not converge; preserve failure, do not substitute clusters')
    centers = model.cluster_centers_indices_
    if centers is None or not len(centers) or not np.isfinite(features[centers]).all():
        raise RuntimeError('Original PMIL affinity propagation produced no usable prototypes')
    return model, np.asarray(centers, dtype=np.int64)


def discover(records, load):
    if not records or any(row.get('subset') != 'train' for row in records):
        raise ValueError('PMIL prototypes may only use the registered TRAIN partition')
    if len({row['slide_id'] for row in records}) != len(records):
        raise ValueError('Duplicate PMIL training slides')
    local_features, origins, local = [], [], []
    for row in records:
        x = load(row)
        model, indices = ap(x)
        local_features.append(x[indices].astype(np.float32))
        origins.extend({'slide_id': row['slide_id'], 'patch_index': int(i)} for i in indices)
        local.append({'slide_id': row['slide_id'], 'centers': indices.tolist(),
            'labels': model.labels_.tolist(), 'iterations': int(model.n_iter_), 'patches': len(x)})
    x = np.concatenate(local_features, axis=0)
    global_model, indices = ap(x)
    return {'features': x[indices].astype(np.float32),
        'origins': [origins[int(i)] for i in indices], 'local': local,
        'global_centers': indices.tolist(), 'global_labels': global_model.labels_.tolist(),
        'global_iterations': int(global_model.n_iter_), 'source': 'TRAIN-only literal two-level AP',
        'all_patch_candidates': sum(row['patches'] for row in local)}

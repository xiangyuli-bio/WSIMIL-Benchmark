"""Full TRAIN-only morphological clustering and true patient bags for AMIML."""
import numpy as np
import torch
from sklearn.cluster import KMeans

from .engine import units
from .io import digest_object


def patient_bags(bundle, subset):
    groups = units(bundle, subset)
    if bundle.config.get('training_unit') != 'patient':
        raise ValueError('AMIML paper packs all slides of each patient into one actual bag')
    return groups


def feature_bag(bundle, rows):
    return torch.cat([bundle.load(row)[0] for row in rows])


def check_cluster_capacity(bundle, R):
    """Reject impossible patient populations before fitting expensive KMeans.

    Four disjoint real clusters each require max(11,R) patches. This necessary
    count bound never claims that a sufficient clustering actually exists.
    Only bag lengths, not held-out features/labels, enter this feasibility test.
    Count each actual patient across all its registered slides.
    """
    if type(R) is not int or R < 1:
        raise ValueError('AMIML R must be a positive integer')
    required = 4 * max(11, R)
    for subset in ['train', 'val', 'test']:
        for rows in patient_bags(bundle, subset):
            count = sum(len(bundle.load(row)[0]) for row in rows)
            if count < required:
                raise ValueError('AMIML requires >10 and >=R real patches in EVERY morphology cluster per patient; '
                    f'{rows[0]["global_patient_id"]} ({subset}) has {count}, needs at least {required}; '
                    'no repeating/padding patches or dropping held-out patients')


def train_kmeans(bundle, seed, max_bytes):
    groups = patient_bags(bundle, 'train')
    # Never use held-out features or labels in clustering. A finite explicit
    # budget fails before allocating the concatenated fitting population.
    values, total = [], 0
    for rows in groups:
        x = feature_bag(bundle, rows)
        total += x.numel()*x.element_size()
        if 8*total > max_bytes:
            raise ValueError('AMIML complete TRAIN KMeans exceeds max_cluster_bytes; no silent subsampling')
        values.append(x.numpy())
    all_features = np.concatenate(values)
    fitted = KMeans(n_clusters=4, n_init=10, max_iter=300, random_state=seed,
                    algorithm='lloyd').fit(all_features)
    centers = torch.from_numpy(fitted.cluster_centers_.copy()).float()
    if not torch.isfinite(centers).all() or len(torch.unique(centers, dim=0)) != 4:
        raise ValueError('AMIML TRAIN morphology does not contain four distinct clusters')
    return centers, {'patients': len(groups), 'patches': len(all_features),
        'population_sha256': digest_object([{'patient': r[0]['global_patient_id'],
                'slides': [s['slide_id'] for s in r]} for r in groups]),
        'inertia': float(fitted.inertia_), 'iterations': int(fitted.n_iter_),
        'implementation': 'sklearn Lloyd n_init10/max_iter300; these missing source choices are explicit'}


def assign(x, centers):
    if x.ndim != 2 or centers.shape != (4, x.shape[1]) or not torch.isfinite(x).all():
        raise ValueError('AMIML requires finite aligned full-dimensional KMeans features')
    # Float64 distances avoid overflow/cancellation for ordinary signed FM
    # vectors. Stable lowest-index argmin handles exactly equidistant patches.
    a, c = x.detach().cpu().double(), centers.detach().cpu().double()
    d = (a.square().sum(1)[:, None]+c.square().sum(1)[None]-2*a@c.T).clamp_min(0)
    if not torch.isfinite(d).all():
        raise ValueError('AMIML nonfinite full-dimensional cluster distances')
    return d.argmin(1)


def cluster_bags(bundle, groups, centers, R, cluster):
    bags, ids = [], []
    for rows in groups:
        x = feature_bag(bundle, rows)
        labels = assign(x, centers)
        counts = labels.bincount(minlength=4)
        if (counts < max(11, R)).any():
            raise ValueError('AMIML requires >10 and >=R real patches in EVERY morphology cluster per patient; '
                'missing original short-bag loader cannot justify repeating/padding patches: '+rows[0]['global_patient_id'])
        idx = (labels == cluster).nonzero().flatten()
        bags.append(x[idx]); ids.append(idx)
    return bags, ids

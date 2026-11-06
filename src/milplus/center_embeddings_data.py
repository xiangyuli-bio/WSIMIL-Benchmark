"""Original measured256RGB tile bags, with immutable WSI/coordinate provenance."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from .center_embeddings_assets import ASSETS, verify
from .data import resolve
from .io import digest_file, digest_object, read_json, write_json

KIND = 'center_embeddings_original_rgb_bags'
TOP_LEFT_POLICY = 'author_norm_coord_inverse_verified'


def encoder_identity(bundle):
    return digest_object({'fm': bundle.fm, 'dimension': bundle.dim,
        'encoder': {k: v for k, v in bundle.encoder.items() if k != 'center_embeddings'}})


def validate_schema(bundle, schema):
    if (bundle.task_type != 'classification' or len(bundle.classes) != 2 or bundle.dim != 3
            or bundle.config.get('training_unit', 'slide') != 'slide'
            or bundle.config.get('evaluation_unit', 'slide') != 'slide'
            or schema.get('schema_version') != 1 or schema.get('kind') != KIND
            or schema.get('layout') != 'NHWC' or schema.get('dtype') != 'uint8' or schema.get('color') != 'RGB'
            or schema.get('patch_pixels') != 256 or schema.get('normalization') != 'torchvision070_resize256_RHF05_Tensor_norm05_01'
            or schema.get('coordinate_frame') not in {'level0_centers', 'level0_top_left_pixels'}
            or schema.get('positive_class') != bundle.classes[1]
            or any(not isinstance(schema.get(k), str) or not schema[k].strip() for k in
                ['pixel_key', 'patch_ids_key', 'coords_key', 'image_provenance', 'coordinate_provenance', 'asset_root'])
            or any(k in bundle.encoder for k in ['c2c', 'sparseconvmil', 'raw_images', 'raw_regions',
                'spatial', 'vision_language', 'multiscale', 'dual_scale', 'omics'])):
        raise ValueError('Center Embeddings requires original measured256RGB, binary slide labels and exact source preprocessing')
    if schema.get('synthetic_coordinates', False) and not bundle.config.get('synthetic', False):
        raise ValueError('Synthetic tile centers cannot be presented as measured WSI coordinates')
    if schema.get('coordinate_frame') == 'level0_top_left_pixels' and schema.get('source_coordinate_policy') != TOP_LEFT_POLICY:
        raise ValueError('Top-left cached RGB requires explicit original norm_coord/read_region equivalence')
    return schema


def top_left_reader_equivalence(coords, provenance, synthetic=False):
    """Prove original integer reader anchors reproduce these exact top-lefts.

    An author's reader anchor is not a claim about an OpenSlide physical
    footprint center. The source uses2**level, not measured level_downsamples.
    Coordinates never enter the cached-pixel classifier/center loss.
    """
    import torch
    from torch.utils.data import Dataset
    from .center_embeddings_source import COMMIT
    from .upstream import load_definitions
    try:
        source = json.loads(provenance)
    except (ValueError, TypeError) as error:
        raise ValueError('Top-left cache requires structured actual read_region provenance') from error
    allowed = {'actual_wsi_read_region_original_clam_coordinates'}
    if synthetic:
        allowed.add('synthetic_read_region_equivalence_fixture')
    level = source.get('patch_level')
    if (source.get('kind') not in allowed or type(level) is not int or not 0 <= level <= 30
            or source.get('patch_pixels') != 256
            or source.get('coordinates_int64_le_sha256') != array_sha256(coords)
            or not isinstance(source.get('source_wsi'), str) or not source['source_wsi'].strip()):
        raise ValueError('Top-left cache read level, pixels, source or original coordinate identity differs')
    if (coords > 2**52).any():
        raise ValueError('Original source integer reader arithmetic requires exactly representable coordinates')
    # This is an exact inverse of the source API's nominal integer offset,
    # never a rounded/replaced physical footprint center or fabricated grid.
    offset = 128 * (2**level)
    anchors = coords.astype(np.int64) + offset
    original = load_definitions('center_embeddings', 'research_mil/loaders/datasets.py', ['MILdataset'],
        {'Dataset': Dataset, 'torch': torch, 'np': np}).MILdataset.norm_coord
    context = SimpleNamespace(level=level, size=256)
    actual = np.asarray([original(context, pair) for pair in anchors], dtype=np.int64)
    if not np.array_equal(actual, coords):
        raise ValueError('Original norm_coord does not recover the exact registered top-left read_region')
    return {'policy': TOP_LEFT_POLICY, 'author_commit': COMMIT, 'read_level': level, 'read_size': 256,
        'mult': 1, 'resize': 'none', 'all_rows_verified': len(coords),
        'reader_anchor_coords_sha256': array_sha256(anchors),
        'top_left_coords_sha256': array_sha256(coords),
        'physical_center_coordinates_invented_or_rounded': False,
        'original_read_region_identity': 'Every original norm_coord(anchor) equals the cached read_region top-left at identical source level and size'}


def array_sha256(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<i8').tobytes()).hexdigest()


def describe(bundle, row, schema):
    path = resolve(bundle.root, row['path'])
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        raise ValueError('Center Embeddings needs actual raw pixel archives, not frozen FM features')
    with h5py.File(path, 'r') as archive:
        for key in ['pixel_key', 'patch_ids_key', 'coords_key']:
            name = schema[key]
            if name not in archive or isinstance(archive.get(name, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError('Center Embeddings images, IDs and centers must be embedded datasets')
            a = archive[name]
            if not isinstance(a, h5py.Dataset) or a.is_virtual or a.external:
                raise ValueError('Unregistered external pixel/coordinate storage is not permitted')
        pixels = archive[schema['pixel_key']]
        if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (256, 256, 3) or len(pixels) < 1:
            raise ValueError('Center Embeddings must retain the original256RGB patches and complete candidate population')
        # The source loader silently omits such TRAIN/VAL slides. Refuse the
        # incompatible cohort rather than changing the registered test cohort.
        if row['subset'] in {'train', 'val'} and len(pixels) < 32:
            raise ValueError('Original Center Embeddings TRAIN/VAL requires at least32 candidates; filter and register explicitly')
        coords = archive[schema['coords_key']][:]
        if (coords.shape != (len(pixels), 2) or not np.issubdtype(coords.dtype, np.integer)
                or (coords < 0).any() or len(np.unique(coords, axis=0)) != len(coords)):
            raise ValueError('Center Embeddings level0 positions must be unique integers in their declared frame and image-aligned')
        ids = archive[schema['patch_ids_key']].asstr()[:].tolist()
        if len(ids) != len(pixels) or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
            raise ValueError('Center Embeddings original patch identities must be unique and ordered')
        provenance = archive.attrs.get('extraction_provenance')
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError('Center Embeddings requires actual WSI extraction or explicit synthetic provenance')
        equivalence = None
        if schema['coordinate_frame'] == 'level0_top_left_pixels':
            equivalence = top_left_reader_equivalence(coords, provenance, bool(bundle.config.get('synthetic', False)))
        shape = list(pixels.shape)
    result = {'shape': shape, 'patch_ids_sha256': digest_object(ids), 'coords_sha256': digest_object(coords.tolist()),
              'extraction_provenance': provenance}
    if equivalence is not None:
        result['source_reader_equivalence'] = equivalence
    return result


def prepare(bundle, schema, output):
    schema = validate_schema(bundle, schema); output = resolve(bundle.root, output)
    if output.exists():
        raise FileExistsError('Center Embeddings assets exist; register a new immutable version')
    asset_root = verify(resolve(bundle.root, schema['asset_root']))
    records = {}
    for row in bundle.rows.to_dict('records'):
        records[row['slide_id']] = {**describe(bundle, row, schema),
            'path': str(resolve(bundle.root, row['path']).resolve()),
            'sha256': digest_file(resolve(bundle.root, row['path'])),
            **{key: row[key] for key in ['global_patient_id', 'subset', 'target']}}
    meta = {'schema_version': 1, 'kind': KIND, 'schema': schema, 'encoder_identity': encoder_identity(bundle),
        'asset_root': str(asset_root), 'assets': {name: sha for name, (_, sha) in ASSETS.items()}, 'slides': records,
        'synthetic': bool(bundle.config.get('synthetic', False)), 'preparer_sha256': digest_file(Path(__file__))}
    meta['contract'] = digest_object(meta); write_json(output, meta)
    return output


class CenterEmbeddingAssets:
    def __init__(self, bundle, manifest):
        self.bundle, self.path = bundle, resolve(bundle.root, manifest); self.meta = read_json(self.path)
        if (self.meta.get('schema_version') != 1 or self.meta.get('kind') != KIND
                or self.meta.get('encoder_identity') != encoder_identity(bundle)
                or self.meta.get('synthetic') != bool(bundle.config.get('synthetic', False))
                or self.meta.get('preparer_sha256') != digest_file(Path(__file__))
                or self.meta.get('assets') != {name: sha for name, (_, sha) in ASSETS.items()}
                or self.meta.get('contract') != digest_object({k: v for k, v in self.meta.items() if k != 'contract'})
                or set(self.meta.get('slides', {})) != set(bundle.rows.slide_id)):
            raise ValueError('Center Embeddings original input, source, preparer or historical asset identity changed')
        self.schema = validate_schema(bundle, self.meta['schema']); self.assets = verify(self.meta['asset_root'])
        for row in bundle.rows.to_dict('records'):
            r = self.meta['slides'][row['slide_id']]
            if r['path'] != str(resolve(bundle.root, row['path']).resolve()) or any(r[k] != row[k] for k in ['global_patient_id', 'subset', 'target']):
                raise ValueError('Center Embeddings original raw images, patient IDs, labels or splits changed')
        self.contract = self.meta['contract']; self.manifest_sha = digest_file(self.path)
        self.files = [self.path, *[self.assets/name for name in ASSETS]]; self._verified = {}

    def check(self, row):
        p = resolve(self.bundle.root, row['path']); s = p.stat()
        signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if self._verified.get(str(p)) != signature:
            expected = self.meta['slides'][row['slide_id']]
            actual = describe(self.bundle, row, self.schema)
            if digest_file(p) != expected['sha256'] or any(actual[k] != expected[k] for k in actual):
                raise ValueError('Center Embeddings original raw pixel, tile order or coordinate bytes changed')
            self._verified[str(p)] = signature
        if digest_file(self.path) != self.manifest_sha:
            raise ValueError('Center Embeddings prepared manifest bytes changed')

    @contextmanager
    def open(self, row):
        self.check(row)
        with h5py.File(resolve(self.bundle.root, row['path']), 'r') as archive:
            yield archive[self.schema['pixel_key']]

    def count(self, row):
        return self.meta['slides'][row['slide_id']]['shape'][0]

    def inventory(self, row, deep=False):
        if deep:
            self.check(row)
        return {k: self.meta['slides'][row['slide_id']][k] for k in
                ['shape', 'sha256', 'patch_ids_sha256', 'coords_sha256', 'extraction_provenance']}

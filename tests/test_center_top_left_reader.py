"""Exact original reader/API equivalence; no fabricated physical-center claim."""
import copy
import json

import h5py
import numpy as np
import pytest
import yaml

from milplus.center_embeddings_data import (TOP_LEFT_POLICY, array_sha256, describe,
    prepare, top_left_reader_equivalence, validate_schema)
from milplus.center_embeddings_demo import create_center_embeddings_demo
from milplus.center_embeddings_loaders import CandidateTiles
from milplus.center_embeddings_pack import norm_coord
from milplus.data import Bundle
from milplus.io import read_json


def fixture_provenance(coords, level=1):
    return json.dumps({'kind': 'synthetic_read_region_equivalence_fixture',
        'source_wsi': 'explicit_synthetic_software_fixture_not_a_real_wsi', 'patch_level': level,
        'patch_pixels': 256, 'coordinates_int64_le_sha256': array_sha256(coords),
        'actual_openslide_downsample': 4.000067967171461})


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_exact_inverse_does_not_need_physical_centers_or_regular_lattice(level):
    coords = np.array([[3, 11], [1030, 17], [4097, 5055]], dtype=np.int64)
    before = coords.copy()
    result = top_left_reader_equivalence(coords, fixture_provenance(coords, level), synthetic=True)
    anchors = coords + 128*(2**level)
    assert [norm_coord(anchor, level, 1) for anchor in anchors] == list(map(tuple, coords))
    assert result['all_rows_verified'] == len(coords)
    assert result['reader_anchor_coords_sha256'] == array_sha256(anchors)
    assert result['physical_center_coordinates_invented_or_rounded'] is False
    np.testing.assert_array_equal(coords, before)


def test_fixture_cannot_claim_real_source_or_changed_coordinates():
    coords = np.array([[13, 25]], dtype=np.int64)
    with pytest.raises(ValueError, match='source or original coordinate identity'):
        top_left_reader_equivalence(coords, fixture_provenance(coords), synthetic=False)
    with pytest.raises(ValueError, match='source or original coordinate identity'):
        top_left_reader_equivalence(coords+1, fixture_provenance(coords), synthetic=True)


def test_top_left_requires_explicit_source_policy(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data'); bundle = Bundle(root)
    schema = copy.deepcopy(bundle.center_embeddings.schema)
    schema['coordinate_frame'] = 'level0_top_left_pixels'
    with pytest.raises(ValueError, match='explicit original norm_coord'):
        validate_schema(bundle, schema)


def test_new_representation_preserves_all_raw_rows_and_original_pixels(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data'); old = Bundle(root)
    schema = copy.deepcopy(old.center_embeddings.schema)
    before = {}
    for row in old.rows.to_dict('records'):
        with h5py.File(row['path'], 'r+') as archive:
            before[row['slide_id']] = archive['pixels'][0]
            archive.attrs['extraction_provenance'] = fixture_provenance(archive['coords'][:])
    task = yaml.safe_load((root/'task.yaml').read_text())
    del task['encoders']['raw_resnet34']['center_embeddings']
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    schema.update(coordinate_frame='level0_top_left_pixels', source_coordinate_policy=TOP_LEFT_POLICY,
        coordinate_provenance='Explicit synthetic top-left reader equivalence fixture; not clinical coordinates')
    prepare(Bundle(root), schema, root/'center_embeddings/top_left_manifest.json')
    task['encoders']['raw_resnet34']['center_embeddings'] = {'manifest': 'center_embeddings/top_left_manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    bundle = Bundle(root)
    assert bundle.check(deep=True)['slides'] == 6
    for row in bundle.rows.to_dict('records'):
        proof = describe(bundle, row, schema)['source_reader_equivalence']
        assert proof['all_rows_verified'] == 33
        with bundle.center_embeddings.open(row) as pixels:
            np.testing.assert_array_equal(pixels[0], before[row['slide_id']])
    candidates = CandidateTiles(bundle.center_embeddings, bundle.records('train'))
    assert len(candidates) == 66 and candidates.counts == [33, 33]
    candidates.close()


@pytest.mark.parametrize('field,value', [('patch_level', -1), ('patch_level', True), ('patch_pixels', 224),
    ('kind', 'unspecified_invented_extraction'), ('coordinates_int64_le_sha256', '0'*64)])
def test_invalid_source_reader_evidence_rejected(field, value):
    coords = np.array([[11, 21]], dtype=np.int64)
    value_dict = json.loads(fixture_provenance(coords)); value_dict[field] = value
    with pytest.raises(ValueError):
        top_left_reader_equivalence(coords, json.dumps(value_dict), synthetic=True)

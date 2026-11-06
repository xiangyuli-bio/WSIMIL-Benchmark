"""Real image/geometry contracts, source packing, and contamination refusal."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image
import yaml

from milplus.data import Bundle
from milplus.io import read_json
from milplus.registry import check_compatibility, require_runnable
from milplus.sparseconvmil_data import prepare
from milplus.sparseconvmil_demo import create_sparseconvmil_demo
from milplus.sparseconvmil_pack import pack


def test_actual_spatial_rgb_inventory_is_separate_from_fm_features(tmp_path):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10)
    bundle = Bundle(data); inventory = bundle.check(deep=True)
    assert bundle.sparseconvmil is not None and len(inventory['features']) == 10
    check_compatibility(require_runnable('sparseconvmil_raw_source_c32'), bundle)
    with pytest.raises(ValueError, match='not frozen'):
        bundle.load(bundle.records('train')[0])
    with pytest.raises(ValueError, match='frozen FM'):
        check_compatibility(require_runnable('abmil'), bundle)


@pytest.mark.parametrize('field,value', [('normalization', 'imagenet'), ('coordinate_frame', 'invented_sequence'),
    ('patch_pixels', 16), ('color', 'BGR'), ('dtype', 'float32')])
def test_schema_refuses_silent_preprocessing_or_fake_space(tmp_path, field, value):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10)
    config = yaml.safe_load((data/'task.yaml').read_text())
    config['encoders']['raw_resnet18'].pop('sparseconvmil')
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    schema = read_json(data/'sparseconvmil_schema.json'); schema[field] = value
    with pytest.raises(ValueError, match='original uint8 RGB'):
        prepare(Bundle(data), schema, data/'new_manifest.json')


@pytest.mark.parametrize('mutation', ['pixels', 'coords', 'ids'])
def test_image_geometry_and_identity_changes_are_detected(tmp_path, mutation):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10)
    bundle = Bundle(data); row = bundle.records('train')[0]
    # Populate the signature cache before mutating actual bytes.
    bundle.sparseconvmil.inventory(row, deep=True)
    with h5py.File(row['path'], 'r+') as file:
        if mutation == 'pixels':
            file['pixels'][0, 0, 0, 0] = (int(file['pixels'][0, 0, 0, 0])+1) % 255
        elif mutation == 'coords':
            file['coords'][0] = [100, 100]
        else:
            file['patch_ids'][0] = 'different_actual_patch.png'
    with pytest.raises(ValueError, match='content changed'):
        bundle.sparseconvmil.inventory(row, deep=True)


def test_patient_labels_and_split_are_bound(tmp_path):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10)
    slides = pd.read_csv(data/'slides.csv'); slides.loc[0, 'label'] = 'class_1'
    slides.to_csv(data/'slides.csv', index=False)
    with pytest.raises(ValueError, match='patient labels or partition changed'):
        Bundle(data)


def test_real_author_folder_pack_is_pixel_exact_and_records_coordinates(tmp_path):
    folder = tmp_path/'actual'; folder.mkdir()
    values = {}
    for x, y in [(24, 16), (0, 8), (8, 0)]:
        rgb = np.random.default_rng(x+y).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        path = folder/f'{x}_{y}.png'; Image.fromarray(rgb).save(path); values[str(path)] = rgb
    table = tmp_path/'folders.csv'; pd.DataFrame([{'slide_id': 'real/id-with-slash', 'path': str(folder)}]).to_csv(table, index=False)
    index = pack(table, tmp_path/'packed'); rows = pd.read_csv(index)
    assert '/' not in __import__('pathlib').Path(rows.path[0]).name
    with h5py.File(rows.path[0], 'r') as file:
        ids = file['patch_ids'].asstr()[:]
        for i, path in enumerate(ids):
            assert np.array_equal(file['pixels'][i], values[path])
            assert file['coords'][i].tolist() == list(map(int, __import__('pathlib').Path(path).stem.split('_')))
    receipt = json.loads((index.parent/'packing.json').read_text())
    assert len(receipt['records'][0]['images']) == 3
    assert receipt['records'][0]['shape'] == [3, 64, 64, 3]
    with pytest.raises(FileExistsError):
        pack(table, index.parent)


@pytest.mark.parametrize('bad_name', ['missing_coordinates.png', '-1_0.png', '1_2_3.png'])
def test_original_folder_names_must_have_real_two_axis_coordinates(tmp_path, bad_name):
    folder = tmp_path/'actual'; folder.mkdir(); Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(folder/bad_name)
    table = tmp_path/'folders.csv'; pd.DataFrame([{'slide_id': 's', 'path': str(folder)}]).to_csv(table, index=False)
    with pytest.raises(ValueError, match='coordinates'):
        pack(table, tmp_path/'packed')


def test_actual_original_sizes_are_not_resized_to_fit(tmp_path):
    folder = tmp_path/'actual'; folder.mkdir()
    for x, size in enumerate([64, 96]):
        Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(folder/f'{x}_0.png')
    table = tmp_path/'folders.csv'; pd.DataFrame([{'slide_id': 's', 'path': str(folder)}]).to_csv(table, index=False)
    with pytest.raises(ValueError, match='inconsistent original sizes'):
        pack(table, tmp_path/'packed')

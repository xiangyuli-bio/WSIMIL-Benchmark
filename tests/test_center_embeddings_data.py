"""Measured raw-image identity, source selection inputs and exact RGB transforms."""
import copy

import h5py
import numpy as np
import pytest
import torch
import yaml

from milplus.center_embeddings_assets import default_root, transforms, verify
from milplus.center_embeddings_data import describe, prepare, validate_schema
from milplus.center_embeddings_demo import create_center_embeddings_demo
from milplus.center_embeddings_loaders import CandidateTiles, SelectedBags
from milplus.center_embeddings_pack import norm_coord
from milplus.center_embeddings_source import definitions
from milplus.data import Bundle
from milplus.io import read_json


def test_raw256_bags_deep_check_and_generic_fm_rejection(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data'); bundle = Bundle(root)
    report = bundle.check(deep=True)
    assert report['patients'] == report['slides'] == 6
    row = bundle.records('train')[0]
    with pytest.raises(ValueError, match='not frozen FM features'):
        bundle.load(row)
    assert bundle.center_embeddings.count(row) == 33
    with pytest.raises(FileExistsError):
        prepare(bundle, read_json(root/'center_schema.json'), root/'center_embeddings/manifest.json')


@pytest.mark.parametrize('key,value', [('patch_pixels', 128), ('normalization', 'imagenet'),
    ('coordinate_frame', 'fabricated_sequence_grid'), ('positive_class', 'class_0'), ('dtype', 'float16')])
def test_incompatible_raw_input_cannot_be_silently_adapted(tmp_path, key, value):
    root = create_center_embeddings_demo(tmp_path/'data'); bundle = Bundle(root)
    schema = copy.deepcopy(bundle.center_embeddings.schema); schema[key] = value
    with pytest.raises(ValueError):
        validate_schema(bundle, schema)


def test_original_pixel_and_coordinate_mutation_is_rejected_at_actual_use(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data'); bundle = Bundle(root)
    row = bundle.records('train')[0]; bundle.center_embeddings.check(row)
    with h5py.File(row['path'], 'r+') as archive:
        archive['coords'][0, 0] += 1
    with pytest.raises(ValueError, match='bytes changed'):
        bundle.center_embeddings.check(row)


def test_candidate_scan_and_top32_keep_source_score_order_and_all_real_images(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data'); bundle = Bundle(root)
    candidates = CandidateTiles(bundle.center_embeddings, bundle.records('train'))
    assert len(candidates) == 66 and candidates.counts == [33, 33]
    selected = definitions()[3].group_argtopk(candidates.groups, np.arange(66, dtype=float), 32)
    bags = SelectedBags(candidates, selected)
    assert bags.selected == {0: list(range(1, 33)), 1: list(range(1, 33))}
    images, label, patch_labels = bags[0]
    assert images.shape == (32, 3, 256, 256) and patch_labels.eq(label).all()
    candidates.close()


def test_source_train_flip_rng_and_old_normalization_are_retained(tmp_path):
    from PIL import Image
    image = np.arange(256*256*3, dtype=np.uint32).reshape(256, 256, 3).astype(np.uint8)
    original = Image.fromarray(image)
    transform = transforms(default_root(), True)
    torch.manual_seed(81); state = torch.random.get_rng_state()
    actual = transform(original); after = torch.random.get_rng_state()
    torch.random.set_rng_state(state); flip = bool(torch.rand(1) < .5)
    expected_image = image[:, ::-1].copy() if flip else image
    expected = torch.from_numpy(expected_image.transpose(2, 0, 1).copy()).float().div(255)
    expected.sub_(torch.tensor([.5]*3)[:, None, None]).div_(torch.tensor([.1]*3)[:, None, None])
    assert torch.equal(actual, expected) and torch.equal(after, torch.random.get_rng_state())


@pytest.mark.parametrize('level,mult', [(0, 1), (1, 2), (2, 1)])
def test_exact_source_level_agnostic_center_reader_geometry(level, mult):
    x, y = 10001, 12345
    size = int(np.round(256*mult)); offset = int(size*(2**level))/2
    assert norm_coord((x, y), level, mult) == (int(x-offset), int(y-offset))


def test_under32_trainval_refused_but_short_test_uses_source_feature_padding(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data', test_patches=1); bundle = Bundle(root)
    assert all(bundle.center_embeddings.count(r) == 1 for r in bundle.records('test'))
    schema = bundle.center_embeddings.schema
    row = bundle.records('test')[0].copy(); row['subset'] = 'train'
    with pytest.raises(ValueError, match='at least32'):
        describe(bundle, row, schema)


def test_no_unbound_multiclass_or_unmeasured_real_coordinate_claim(tmp_path):
    root = create_center_embeddings_demo(tmp_path/'data'); bundle = Bundle(root)
    schema = copy.deepcopy(bundle.center_embeddings.schema)
    bundle.classes = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        validate_schema(bundle, schema)
    assert verify(default_root()).is_dir()
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['synthetic'] = False; (root/'task.yaml').write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError):
        Bundle(root)

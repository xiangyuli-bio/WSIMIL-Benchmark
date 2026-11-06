"""Native source JPEG containers; synthetic software fixtures, not BRCA scores."""
import copy
import os
from pathlib import Path

import cv2
import h5py
import kornia as K
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.engine import restore_rng, rng_state, seed_everything
from milplus.io import digest_file, digest_object, read_json, write_json
from milplus.promptmil import PromptMIL
from milplus.promptmil_data import prepare
from milplus.promptmil_demo import create_demo
from milplus.promptmil_jpeg import KIND, STORAGE, describe, pixels, runtime_identity
from milplus.promptmil_source import COMMIT
from milplus.upstream import load_definitions

WEIGHTS = Path('/data/xiangyuli/mil_baseline_puls_work/assets/promptmil_author_weights_20261026_r3/tcga-brca-vitt.pth')


def jpeg_bag(root, arrays, ids):
    root.mkdir()
    schema = {'storage': STORAGE, 'source_commit': COMMIT, 'merge_batch': 200,
        'jpeg_decoder': 'source_cv2_fallback', 'jpeg_protocol_sha256': digest_object({'software_fixture': True}),
        **runtime_identity()}
    chunks = []
    for offset in range(0, len(arrays), 200):
        block = arrays[offset:offset+200]
        path = root / ('%06d.jpg' % (offset // 200))
        assert cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(block), cv2.COLOR_RGB2BGR))
        chunks.append({'path': path.name, 'sha256': digest_file(path), 'offset': offset, 'count': len(block)})
    index = root / 'bag.json'
    write_json(index, {'schema_version': 1, 'kind': KIND, 'protocol_sha256': schema['jpeg_protocol_sha256'],
        'patch_ids': ids, 'chunks': chunks})
    return index, schema


@pytest.fixture
def bag(tmp_path):
    image = np.random.default_rng(79).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    index, schema = jpeg_bag(tmp_path / 'bag', [image] * 201, ['synthetic_%d' % i for i in range(201)])
    return index, schema


def test_full200_and_tail_keep_original_reader_tensor_and_row_order(bag, tmp_path):
    index, schema = bag
    actual = pixels(index, schema)
    assert actual.shape == (201, 224, 224, 3) and actual.dtype == np.uint8
    csv = tmp_path / 'source.csv'
    pd.DataFrame([{'wsi_id': 'source', 'label': 1, 'len_img': 2, 'is_test': 0}]).to_csv(csv, index=False)
    # Native filenames are a dataset convention, not a change to JPEG bytes.
    import shutil
    for i in range(2):
        shutil.copy2(index.parent / ('%06d.jpg' % i), tmp_path / ('source_%d.jpg' % i))
    original = load_definitions('promptmil', 'dataset/merge_patch_wsi_dataset.py',
        ['read_rgb_img', 'MergePatchWsiDataset'], {'cv2': cv2, 'use_jpeg4py': False,
        'jpeg': None, 'np': np, 'pd': pd, 'os': os, 'K': K, 'torch': torch,
        'Dataset': torch.utils.data.Dataset}).MergePatchWsiDataset
    expected, label = original(str(tmp_path), str(csv), 'train')[0]
    got = torch.from_numpy(actual).permute(0, 3, 1, 2) / 255.
    assert torch.equal(got, expected) and label == 1
    assert describe(index, schema)['ordered_patch_ids_sha256'] == digest_object(['synthetic_%d' % i for i in range(201)])


@pytest.mark.parametrize('mutation', ['duplicate_ids', 'duplicate_chunk', 'count', 'offset', 'jpeg_bytes'])
def test_missing_changed_reused_or_misordered_pixels_are_refused(bag, mutation):
    index, schema = bag; value = read_json(index)
    if mutation == 'duplicate_ids': value['patch_ids'][1] = value['patch_ids'][0]
    elif mutation == 'duplicate_chunk': value['chunks'][1]['path'] = value['chunks'][0]['path']
    elif mutation == 'count': value['chunks'][1]['count'] = 2
    elif mutation == 'offset': value['chunks'][1]['offset'] = 199
    else:
        path = index.parent / value['chunks'][0]['path']
        with path.open('ab') as stream: stream.write(b'changed')
    write_json(index, value)
    with pytest.raises(ValueError): describe(index, schema)


@pytest.mark.parametrize('key,value', [('jpeg_decoder', 'Pillow'), ('merge_batch', 128), ('opencv_version', 'unknown')])
def test_no_silent_decoder_chunk_or_library_substitution(bag, key, value):
    index, schema = bag; schema[key] = value
    with pytest.raises(ValueError, match='actual pinned'): describe(index, schema)


@pytest.fixture(scope='module')
def equivalent_views(tmp_path_factory):
    base = tmp_path_factory.mktemp('promptmil_jpeg_equivalent')
    hdf5 = create_demo(base / 'hdf5', WEIGHTS, patches=3)
    task = yaml.safe_load((hdf5 / 'task.yaml').read_text()); encoder = task['encoders']['native_raw_RGB']
    encoder.pop('promptmil')
    (hdf5 / 'task.yaml').write_text(yaml.safe_dump(task))
    old_schema = read_json(hdf5 / 'promptmil.json')['schema']
    jpeg = base / 'jpeg'; jpeg.mkdir()
    for name in ['slides.csv', 'split.csv']: __import__('shutil').copy2(hdf5 / name, jpeg / name)
    features = pd.read_csv(hdf5 / 'features.csv'); mapped = []
    for row in features.to_dict('records'):
        with h5py.File(hdf5 / row['path'], 'r+') as file:
            path, extra = jpeg_bag(jpeg / row['slide_id'], list(file['rgb'][:]), file['patch_ids'].asstr()[:].tolist())
            # Same decoded source JPEG pixels in the old HDF5 route, so this
            # comparison isolates container behavior, not lossy-vs-lossless data.
            file['rgb'][:] = pixels(path, extra)
        mapped.append({'slide_id': row['slide_id'], 'path': str(path)})
    pd.DataFrame(mapped).to_csv(jpeg / 'features.csv', index=False)
    (jpeg / 'task.yaml').write_text(yaml.safe_dump(task))
    prepare(Bundle(hdf5), old_schema, hdf5 / 'decoded_jpeg_manifest.json')
    task['encoders']['native_raw_RGB']['promptmil'] = {'manifest': 'decoded_jpeg_manifest.json'}
    (hdf5 / 'task.yaml').write_text(yaml.safe_dump(task))
    jpeg_task = copy.deepcopy(task)
    jpeg_task['encoders']['native_raw_RGB'].pop('promptmil')
    (jpeg / 'task.yaml').write_text(yaml.safe_dump(jpeg_task))
    prepare(Bundle(jpeg), {**old_schema, **extra}, jpeg / 'jpeg_manifest.json')
    jpeg_task['encoders']['native_raw_RGB']['promptmil'] = {'manifest': 'jpeg_manifest.json'}
    (jpeg / 'task.yaml').write_text(yaml.safe_dump(jpeg_task))
    return hdf5, jpeg


def exact(a, b):
    if torch.is_tensor(a): assert torch.equal(a, b)
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else: assert a == b


def test_same_real_teacher_full_network_two_epochs_weights_adam_pending_gradients_rng(equivalent_views):
    torch.set_num_threads(2)
    models = []
    for root in equivalent_views:
        bundle = Bundle(root); seed_everything(2026)
        model = PromptMIL(bundle, 'promptmil_brca_dino_dsmil_source')
        model.configure_training({'epochs': 2, 'max_instances': None, 'grad_clip': None})
        optimizer = model.make_optimizer(model.recipe['lr'], model.recipe['weight_decay'])
        model.train(); models.append((bundle, model, optimizer))
    for epoch in range(2):
        for _, model, _ in models: model.start_epoch(epoch)
        for i, rows in enumerate(zip(*(bundle.records('train') for bundle, _, _ in models))):
            initial = rng_state(); after = []; terms = []
            for (bundle, model, optimizer), row in zip(models, rows):
                restore_rng(initial)
                terms.append(model.train_bag(bundle.promptmil.pixels(row), torch.tensor([row['target']]), optimizer, i)[0])
                after.append(rng_state())
            exact(after[0], after[1]); exact(terms[0], terms[1])
            for attr in ['backbone', 'classifier']: exact(getattr(models[0][1], attr).state_dict(), getattr(models[1][1], attr).state_dict())
            exact(models[0][2].state_dict(), models[1][2].state_dict())
            for a, b in zip(models[0][1].parameters(), models[1][1].parameters()):
                assert (a.grad is None) == (b.grad is None)
                if a.grad is not None: exact(a.grad, b.grad)
        for bundle, model, _ in models: model.finish_epoch(bundle, 'cpu', {})
        assert all(model.pending_bags == 1 and model.optimizer_updates == epoch + 1 for _, model, _ in models)

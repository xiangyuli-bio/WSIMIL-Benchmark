"""Full ResNet18/LPA reference checks; not full LNPL method admission."""
import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.engine import seed_everything
from milplus.io import digest_file, read_json
from milplus.lnplmil_raw import KIND, RawInputs, prepare
from milplus.lnplmil_weak import assets, build, fit, optimizer, original_definitions, rgb_tensor, score_raw
from milplus.hybrid_supervision_checkpoint import load
from test_pmil_prototypical_components import equal


def official_assets():
    return assets(Path(os.environ.get('MILPLUS_LNPL_ASSETS', '/root/MIL/development/lnplmil_v103/assets_r1')))


def raw_fixture(tmp_path, heldout_annotation=False, patch_count=200, varied=False):
    root = tmp_path / 'data'
    root.mkdir()
    rows = []
    for subset in ['train', 'val', 'test']:
        for target in [0, 1]:
            name = subset + '_' + str(target)
            path = root / (name + '.h5')
            with h5py.File(path, 'w') as file:
                # Actual RGB arrays, compressed repeated synthetic tissue patterns.
                pixels = file.create_dataset('rgb', (patch_count, 224, 224, 3), dtype='uint8',
                    fillvalue=64 + target * 64, chunks=(1, 224, 224, 3), compression='gzip')
                for i in range(2):
                    image = np.full((224, 224, 3), 64 + target * 64 + i * 7, dtype=np.uint8)
                    image[32:96, 11:119, target] = 200
                    pixels[i] = image
                if varied:
                    for i in range(2, patch_count):
                        image = np.full((224, 224, 3), 32 + i % 180, dtype=np.uint8)
                        image[32:96, 11:119, i % 3] = 180
                        pixels[i] = image
                file.create_dataset('ids', data=np.asarray([name + '_p' + str(i) for i in range(patch_count)],
                    dtype=h5py.string_dtype('utf-8')))
                file.create_dataset('coords', data=np.asarray([[(i % 100) * 1200, (i // 100) * 1200]
                    for i in range(patch_count)], dtype=np.int64))
                labelled = subset == 'train' or heldout_annotation and subset == 'test'
                file.create_dataset('labelled', data=np.asarray([0, 1] if labelled else [], dtype=np.int64))
                file.create_dataset('patch_labels', data=np.asarray([target, target] if labelled else [], dtype=np.int64))
            rows.append({'slide_id': name, 'patient_id': name, 'global_patient_id': 'synthetic:' + name,
                         'target': target, 'subset': subset, 'path': str(path)})
    schema = {'kind': KIND, 'schema_version': 1, 'layout': 'NHWC', 'color': 'RGB', 'patch_pixels': 224,
        'coordinate_frame': 'level0_top_left_pixels', 'magnification': 40, 'mpp': .25,
        'pixels_key': 'rgb', 'patch_ids_key': 'ids', 'coords_key': 'coords',
        'labelled_indices_key': 'labelled', 'patch_labels_key': 'patch_labels',
        'image_provenance': 'explicit synthetic RGB software test, no actual WSI',
        'annotation_provenance': 'explicit synthetic observed TRAIN patch labels',
        'patch_provenance': 'explicit synthetic unique coordinates and ordered patch IDs',
        'initial_assets': str(official_assets()), 'weak_classes': ['negative', 'positive'],
        'positive_weak_class_ids': [1]}
    return root, rows, schema


def test_full_actual_ResNet18_source_logits_all_gradients_and_SGD_momentum_exact():
    torch.set_num_threads(2)
    seed_everything(103)
    root = official_assets()
    actual = build(root, 'weak')
    source = original_definitions(root)
    literal = source.ResNet(source.BasicBlock, [2, 2, 2, 2])
    literal.fc = nn.Linear(512, 2)
    literal.load_state_dict(actual.state_dict(), strict=True)
    assert sum(p.numel() for p in actual.parameters()) == 11177538
    assert all(len(getattr(actual, 'layer' + str(i))) == 2 for i in range(1, 5))
    image = np.random.default_rng(103).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    x = torch.stack([rgb_tensor(image), rgb_tensor(image[:, ::-1].copy())])
    expected = torch.from_numpy(np.stack([image, image[:, ::-1].copy()])).permute(0, 3, 1, 2).float() / 255
    expected = ((expected - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]).contiguous()
    equal(x, expected)
    assert x.stride() == expected.stride() and x.is_contiguous()
    left, right = optimizer(actual), torch.optim.SGD(literal.parameters(), lr=.02, momentum=.9, nesterov=True)
    for _ in range(2):
        left.zero_grad(); right.zero_grad()
        a, b = actual(x), literal(expected)
        equal(a, b)
        F.cross_entropy(a, torch.tensor([0, 1])).backward()
        F.cross_entropy(b, torch.tensor([0, 1])).backward()
        for p, q in zip(actual.parameters(), literal.parameters()):
            equal(p.grad, q.grad)
        left.step(); right.step()
        equal(actual.state_dict(), literal.state_dict())
        equal(left.state_dict(), right.state_dict())


def test_separate_frozen_ImageNet512D_branch_stays_original_when_weak_is_trained():
    torch.set_num_threads(2); seed_everything(103)
    root = official_assets()
    frozen = build(root, 'frozen')
    weak = build(root, 'weak', classes=9)
    assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())
    assert weak.fc.out_features == 9 and frozen.fc.__class__ is nn.Identity
    before = copy.deepcopy(frozen.state_dict())
    x = torch.stack([rgb_tensor(np.full((224, 224, 3), i, dtype=np.uint8)) for i in [64, 128]])
    opt = optimizer(weak)
    F.cross_entropy(weak(x), torch.tensor([0, 8])).backward(); opt.step()
    with torch.no_grad():
        a = frozen(x)
        source = original_definitions(root)
        literal = source.ResNet(source.BasicBlock, [2, 2, 2, 2])
        literal.load_state_dict(torch.load(root / 'resnet18-5c106cde.pth', weights_only=True), strict=True)
        literal.fc = nn.Identity(); literal.eval()
        equal(a, literal(x))
    assert a.shape == (2, 512)
    equal(frozen.state_dict(), before)


@pytest.mark.parametrize('cut', [1, 2, 3])
def test_actual_LPA_weak_recovery_at_midbatch_and_epoch_boundaries_exact_all_state(tmp_path, cut):
    torch.set_num_threads(2)
    root, rows, schema = raw_fixture(tmp_path)
    manifest = prepare(root, rows, schema, 'manifest.json')
    raw = RawInputs(manifest)
    assert len(raw.labelled) == 4 and {e[2] for e in raw.labelled} == {0, 1}
    baseline = fit(raw, tmp_path / 'baseline', epochs=2, batch_size=2)
    resumed = fit(raw, tmp_path / 'resumed', epochs=2, batch_size=2, stop_after_updates=cut)
    assert read_json(resumed / 'status.json')['state'] == 'interrupted'
    fit(raw, resumed, epochs=2, batch_size=2, resume=True)
    a, b = load(baseline / 'last.pt'), load(resumed / 'last.pt')
    equal(a, b)
    equal(load(baseline / 'selected.pt'), load(resumed / 'selected.pt'))
    assert a['updates'] == 4 and len(a['history']) == 2 and a['offset'] == 0
    assert set(a['gradients']) == set(dict(build(raw.asset_root, 'weak').named_parameters()))
    assert read_json(resumed / 'status.json')['native50epoch256batch'] is False


def test_raw_heldout_LPA_and_patient_cross_partition_rejected(tmp_path):
    root, rows, schema = raw_fixture(tmp_path, heldout_annotation=True)
    with pytest.raises(ValueError, match='Only TRAIN'):
        prepare(root, rows, schema, 'rejected.json')
    for row in rows:
        if row['subset'] == 'test':
            with h5py.File(row['path'], 'r+') as file:
                del file['labelled']; del file['patch_labels']
                file.create_dataset('labelled', data=np.asarray([], dtype=np.int64))
                file.create_dataset('patch_labels', data=np.asarray([], dtype=np.int64))
    rows[-1]['global_patient_id'] = rows[0]['global_patient_id']
    with pytest.raises(ValueError, match='cannot cross'):
        prepare(root, rows, schema, 'rejected.json')
    assert not (root / 'rejected.json').exists()


def test_observed_LPA_mutation_detected_before_any_training_update(tmp_path):
    root, rows, schema = raw_fixture(tmp_path)
    raw = RawInputs(prepare(root, rows, schema, 'manifest.json'))
    row = rows[0]
    with h5py.File(row['path'], 'r+') as file:
        file['patch_labels'][0] = 1
    with pytest.raises(ValueError, match='changed'):
        raw.labelled_batch([raw.labelled[0]])


def test_real_raw_scoring_matches_selected_weak_and_separate_frozen_original(tmp_path):
    torch.set_num_threads(2)
    root, rows, schema = raw_fixture(tmp_path)
    raw = RawInputs(prepare(root, rows, schema, 'manifest.json'))
    run = fit(raw, tmp_path / 'weak', epochs=1, batch_size=256)
    # Execute the same real export operator, on all six complete200patch bags.
    manifest = score_raw(raw, run, tmp_path / 'scored', batch_size=64)
    meta = read_json(manifest)
    assert len(meta['slides']) == 6 and meta['raw_manifest_sha256'] == raw.contract
    weak = build(raw.asset_root, 'weak');weak.load_state_dict(load(run / 'selected.pt')['model']);weak.eval()
    frozen = build(raw.asset_root, 'frozen')
    row = raw.rows[-1]
    with raw.open(row) as file, h5py.File(meta['slides'][row['slide_id']]['path'], 'r') as scored:
        image = file['rgb'][0]
        x = rgb_tensor(image)[None]
        with torch.no_grad():
            probability = weak(x).softmax(1)[0, 1]
            features = frozen(x)[0]
        # Batch shape can change convolution rounding; use the same64batch oracle.
        xb = torch.stack([rgb_tensor(a) for a in file['rgb'][:64]])
        with torch.no_grad():
            probability = weak(xb).softmax(1)[0, 1]
            features = frozen(xb)[0]
        value = torch.from_numpy(scored['scored_features'][0])
        equal(value[0], probability); equal(value[3:], features)
        equal(torch.from_numpy(scored['coords'][:]), torch.from_numpy(file['coords'][:]))
        assert scored['patch_ids'].asstr()[:].tolist() == file['ids'].asstr()[:].tolist()
        assert scored.attrs['weak_checkpoint_sha256'] == meta['weak_checkpoint_sha256']
    assert digest_file(manifest) and meta['not_original_unpublished_recipe']

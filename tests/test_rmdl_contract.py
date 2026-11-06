"""Spatial/supervision failures that would silently change the full RMDL method."""
from pathlib import Path
import json
import os
import shutil

import h5py
import numpy as np
import pytest

from milplus.rmdl_pixels import (Slide, annotation, materialized, training_candidates,
    balanced_batch, augmented_crop, grid_shape, select_instances, preprocess)
from milplus.rmdl_source import normalize_features


def raw(tmp_path):
    path = tmp_path/'slide.h5'
    y,x = np.indices((918,918))
    pixels = np.stack([x % 256,y % 256,(x+y) % 256],axis=-1).astype(np.uint8)
    mask = np.where(x < 400,0,np.where(x < 650,1,2)).astype(np.uint8)
    with h5py.File(path,'w') as file:
        file['rgb'] = pixels;file['annotation'] = mask
        file.attrs['working_mpp'] = .5034
    return {'kind':'working_rgb_hdf5','path':str(path),'annotation':str(path),
            'subset':'train','working_size':[918,918]},pixels


def test_actual_spatial_read_and_only_edge_context_padding(tmp_path):
    record,pixels = raw(tmp_path)
    with Slide(record) as slide:
        assert np.array_equal(slide.read(17,29,299),pixels[29:328,17:316])
        edge = slide.read(900,900,299)
    assert np.array_equal(edge[:18,:18],pixels[900:,900:])
    assert np.all(edge[18:] == 255)
    assert np.all(edge[:18,18:] == 255)


@pytest.mark.parametrize('subset',['val','test'])
def test_heldout_pixel_labels_cannot_be_used(tmp_path,subset):
    record,unused = raw(tmp_path);record['subset'] = subset
    with pytest.raises(ValueError,match='TRAIN'):
        annotation(record,np.asarray([459]),np.asarray([459]))


def test_sampler_uses_actual_pixel_classes_and_is_replayable(tmp_path):
    record,unused = raw(tmp_path)
    first = training_candidates({'actual':record},73,normals_per_slide=17)
    second = training_candidates({'actual':record},73,normals_per_slide=17)
    assert first == second
    assert {r['target'] for r in first} == {0,1,2}
    for row in first:
        assert row['target'] == int(annotation(record,np.asarray([row['center'][0]]),np.asarray([row['center'][1]]))[0])
    counts = np.zeros(3,dtype=int)
    for step in range(3):
        indices = balanced_batch(first,73,step)
        assert len(indices) == 80
        current = np.bincount([first[int(i)]['target'] for i in indices],minlength=3)
        assert sorted(current.tolist()) == [26,27,27]
        counts += current
    assert counts.tolist() == [80,80,80]


def test_missing_pixel_class_is_not_created_from_slide_label(tmp_path):
    record,unused = raw(tmp_path)
    with h5py.File(record['annotation'],'a') as file:
        file['annotation'][:] = 0
    with pytest.raises(ValueError,match='slide labels are not a substitute'):
        training_candidates({'actual':record},73,normals_per_slide=17)


def test_native_augmentation_preserves_shape_and_full470_requirement(tmp_path):
    record,unused = raw(tmp_path)
    with Slide(record) as slide:
        a = augmented_crop(slide,[459,459],np.random.RandomState(73))
        b = augmented_crop(slide,[459,459],np.random.RandomState(73))
        assert a.shape == (299,299,3) and a.dtype == np.uint8
        assert np.array_equal(a,b)
        with pytest.raises(ValueError,match='complete470'):
            augmented_crop(slide,[0,0],np.random.RandomState(73))


def test_grid_retains_original_full51_block_geometry():
    assert grid_shape(1899,1899) == (51,51)
    assert grid_shape(3531,3531) == (102,102)
    with pytest.raises(ValueError):
        grid_shape(298,299)


def brute_selection(p):
    # Independent scalar geometric oracle: no vectorized suppression/reused
    # production selection code. Ties follow row-major source coordinates.
    result = []
    for target in range(3):
        candidates = sorted([(float(p[y,x,target]),y,x) for y in range(p.shape[0]) for x in range(p.shape[1])],
                            key=lambda item:(-item[0],item[1],item[2]))
        kept = []
        for score,y,x in candidates:
            overlap = False
            for unused,ky,kx in kept:
                width = max(0,min(x*32+299,kx*32+299)-max(x*32,kx*32))
                height = max(0,min(y*32+299,ky*32+299)-max(y*32,ky*32))
                if width*height > .68*299*299:
                    overlap = True;break
            if not overlap:
                kept.append((score,y,x))
            if len(kept) == 100:
                break
        result.extend((target,y*p.shape[1]+x) for score,y,x in kept)
    return result


@pytest.mark.parametrize('ties',[False,True])
def test_nms_matches_independent_geometry_including_ties(ties):
    p = np.random.RandomState(73).uniform(size=(51,51,3)).astype(np.float32)
    if ties:
        p[:] = 1
    p /= p.sum(-1,keepdims=True)
    actual = select_instances(p)
    assert [(r['selection_class'],r['grid_index']) for r in actual] == brute_selection(p)
    assert len(actual) == 300
    assert all(len({tuple(r['xy']) for r in actual if r['selection_class'] == c}) == 100 for c in range(3))


def test_insufficient_real_regions_fails_without_duplicate_padding():
    with pytest.raises(ValueError,match='fewer than100'):
        select_instances(np.full((5,5,3),1/3,dtype=np.float32))


@pytest.mark.parametrize('kind',['soft','external','virtual','raw_external'])
def test_unhashed_hdf5_pixel_sources_rejected(tmp_path,kind):
    backing = tmp_path/'actual.h5'
    with h5py.File(backing,'w') as file:
        file['actual'] = np.zeros((3,3,3),dtype=np.uint8)
    path = tmp_path/'wrapper.h5'
    with h5py.File(path,'w') as file:
        if kind == 'soft':
            file['actual'] = np.zeros((3,3,3),dtype=np.uint8);file['rgb'] = h5py.SoftLink('/actual')
        elif kind == 'external':
            file['rgb'] = h5py.ExternalLink(str(backing),'/actual')
        elif kind == 'virtual':
            layout = h5py.VirtualLayout(shape=(3,3,3),dtype=np.uint8)
            layout[:] = h5py.VirtualSource(str(backing),'actual',shape=(3,3,3))
            file.create_virtual_dataset('rgb',layout)
        else:
            file.create_dataset('rgb',shape=(3,3,3),dtype=np.uint8,external=[(str(tmp_path/'pixels.bin'),0,h5py.h5f.UNLIMITED)])
    with h5py.File(path,'r') as file:
        with pytest.raises(ValueError,match='materialized|external'):
            materialized(file,'rgb')


def test_exact_source_minmax_and_input_preprocessing():
    x = np.arange(300*1024,dtype=np.float32).reshape(300,1024)
    assert np.array_equal(normalize_features(x),2*(x-x.min())/(x.max()-x.min()+np.float32(1e-8))-1)
    assert np.array_equal(normalize_features(np.zeros_like(x)),-np.ones_like(x))
    rgb = np.asarray([[[[0,127,255]]]],dtype=np.uint8)
    assert np.array_equal(preprocess(rgb),rgb.astype(np.float32)/np.float32(127.5)-np.float32(1))
    with pytest.raises(ValueError):
        normalize_features(x[:299])


@pytest.fixture
def real_bound_demo(tmp_path):
    root = os.environ.get('MILPLUS_RMDL_FULL_DEMO')
    if not root:
        pytest.skip('Set MILPLUS_RMDL_FULL_DEMO to the actually generated full1899 validation slides and real ImageNet assets')
    root = Path(root)
    for name in ['task.yaml','slides.csv','split.csv','index.csv','raw_records.jsonl']:
        shutil.copy2(root/name,tmp_path/name)
    return tmp_path


def test_complete_bundle_binds_actual_supervision_and_rejects_frozen_loading(real_bound_demo):
    from milplus.data import Bundle
    from milplus.registry import check_compatibility
    bundle = Bundle(real_bound_demo)
    value = bundle.rmdl.export()
    assert len(value['records']) == 9
    assert sum(r['annotation'] is not None for r in value['records'].values()) == 3
    with pytest.raises(ValueError,match='complete localization'):
        bundle.load(bundle.records('train')[0])
    with pytest.raises(ValueError,match='frozen FM'):
        check_compatibility({'id':'meanmil'},bundle)


def test_complete_binding_refuses_heldout_annotation_path(real_bound_demo):
    from milplus.data import Bundle
    path = real_bound_demo/'raw_records.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[3]['annotation'] = rows[0]['annotation']
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Validation/test'):
        Bundle(real_bound_demo)


def test_missing_train_mask_fails_before_any_training(real_bound_demo):
    from milplus.data import Bundle
    path = real_bound_demo/'raw_records.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()];rows[0]['annotation'] = None
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Every TRAIN'):
        Bundle(real_bound_demo)


def test_manifest_edits_are_detected_after_initial_binding(real_bound_demo):
    from milplus.data import Bundle
    bundle = Bundle(real_bound_demo)
    path = real_bound_demo/'raw_records.jsonl';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='bytes changed'):
        bundle.rmdl.check(deep=True)


def test_working_scale_cannot_be_silently_substituted(real_bound_demo):
    from milplus.data import Bundle
    path = real_bound_demo/'raw_records.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()];rows[0]['working_mpp'] = 1.0068
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='original-scale'):
        Bundle(real_bound_demo)


def test_registered_full_profiles_keep_pooling_and_native_contracts_distinct():
    from milplus.registry import inspect_method
    default=inspect_method('RMDL');source=inspect_method('rmdl_source_sum_full_reference')
    assert default['id']=='rmdl_paper_mean_full_reference'
    assert default['native_recipe']['pooling']=='mean'
    assert source['native_recipe']['pooling']=='sum'
    for spec in [default,source]:
        assert spec['native_recipe']['localizer_batch']==80
        assert spec['native_recipe']['MIL_batch']==256
        assert spec['native_recipe']['MIL_iterations']==7000
        assert spec['native_recipe']['actual_selected_instances']==300
        assert spec['native_recipe']['FConv_dimension']==1024
        assert spec['formal_admission']is False

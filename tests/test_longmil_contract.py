import copy
from pathlib import Path

import h5py
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.longmil import LongMIL, input_spec
from milplus.registry import check_compatibility, inspect_method, resolve_for_bundle


def data(tmp_path, survival=False):
    root = tmp_path / 'data'
    create_demo(root, patients=18, dimension=32, survival=survival)
    p = root / 'task.yaml'
    config = yaml.safe_load(p.read_text())
    config['encoders']['toy']['longmil'] = {'coordinate_frame': 'level0_xy', 'patch_step_level0': 256}
    p.write_text(yaml.safe_dump(config))
    return root


def test_public_method_alias_resolves_both_classification_and_survival(tmp_path):
    root = data(tmp_path)
    bundle = Bundle(root)
    assert inspect_method('LongMIL')['id'] == 'longmil_v2_paper_classification_reference'
    assert resolve_for_bundle('LongMIL', bundle)['id'] == 'longmil_v2_paper_classification_reference'
    check_compatibility(resolve_for_bundle('LongMIL', bundle), bundle)
    p = root / 'task.yaml'
    config = yaml.safe_load(p.read_text())
    config.update(task_type='survival', time_unit='months', training_unit='patient')
    p.write_text(yaml.safe_dump(config))
    bundle = Bundle(root)
    assert resolve_for_bundle('LongMIL', bundle)['id'] == 'longmil_v2_paper_survival_reference'
    check_compatibility(resolve_for_bundle('LongMIL', bundle), bundle)


def test_physical_protocol_and_native_architecture_cannot_be_silently_reduced(tmp_path):
    bundle = Bundle(data(tmp_path))
    model = LongMIL(bundle, 'longmil_v2_paper_classification_reference')
    protocol = model.protocol()
    assert protocol['input_projection'] == 384 and protocol['heads'] == 1
    assert protocol['ffn_hidden'] == 1536 and protocol['local_blocks'] == 2
    assert protocol['reference_max_epochs'] == 200 and protocol['reference_patience'] == 20
    assert protocol['paper_reported_weight_decay'] == .01
    assert protocol['author_v2_code_available'] is False
    assert protocol['paper_reference_reconstruction'] is True
    with pytest.raises(TypeError):
        LongMIL(bundle, 'longmil_v2_paper_classification_reference', hidden=16)
    with pytest.raises(ValueError, match='complete actual bags'):
        model.configure_training({'max_instances': 10, 'grad_clip': None})
    with pytest.raises(ValueError, match='no gradient clipping'):
        model.configure_training({'max_instances': None, 'grad_clip': 5})
    row = bundle.records('train')[0]
    features, coords = bundle.load(row, require_coords=True)
    with pytest.raises(ValueError, match='requires CUDA'):
        model(features, coords=coords)


def test_missing_real_coordinates_and_wrong_physical_scale_fail(tmp_path):
    root = data(tmp_path)
    bundle = Bundle(root)
    first = bundle.records('train')[0]
    with h5py.File(root / first['path'], 'a') as f:
        f['coords'][1, 0] = 257
    with pytest.raises(ValueError, match='declared physical patch grid'):
        input_spec(Bundle(root), 'longmil_v2_paper_classification_reference')
    with h5py.File(root / first['path'], 'a') as f:
        del f['coords']
    with pytest.raises(ValueError, match='real coordinates required'):
        input_spec(Bundle(root), 'longmil_v2_paper_classification_reference')


def test_v1_never_accepts_small_fake_or_unbound_full_table(tmp_path):
    root = data(tmp_path)
    with pytest.raises(ValueError, match='complete original8.1GB'):
        input_spec(Bundle(root), 'longmil_v1_source_classification_reference')
    fake = root / 'fake.pt'
    torch.save(torch.zeros(2, 2, dtype=torch.int8), fake)
    p = root / 'task.yaml'
    config = yaml.safe_load(p.read_text())
    config['encoders']['toy']['longmil'].update(v1_alibi_path='fake.pt', v1_alibi_sha256='0' * 64)
    p.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='fully verified original'):
        input_spec(Bundle(root), 'longmil_v1_source_classification_reference')


def test_multi_slide_survival_keeps_separate_physical_frames(tmp_path):
    bundle = Bundle(data(tmp_path, survival=True))
    model = LongMIL(bundle, 'longmil_v2_paper_survival_reference')
    assert model.native.head.out_features == 1
    assert model.protocol()['multislide_patient'] == 'mean_independent_slide_logits_or_risks_never_merge_coordinate_frames'
    check_compatibility(resolve_for_bundle('LongMIL', bundle), bundle)


@pytest.mark.parametrize('survival', [False, True])
def test_public_demo_command_records_actual_two_dimensional_bags(tmp_path, survival):
    import os
    import subprocess
    import sys
    root = tmp_path/'public_demo'
    command = [sys.executable, '-m', 'milplus', 'demo-longmil', '--output', str(root),
               '--patches', '25', '--dimension', '1024']
    if survival:
        command.append('--survival')
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    bundle = Bundle(root)
    assert bundle.config['synthetic'] and bundle.dim == 1024
    assert bundle.task_type == ('survival' if survival else 'classification')
    for row in bundle.records('train'):
        value, coords = bundle.load(row, require_coords=True)
        assert value.shape == (25, 1024) and coords.shape == (25, 2)
        assert len(torch.unique(coords[:, 0])) > 1 and len(torch.unique(coords[:, 1])) > 1
    check_compatibility(resolve_for_bundle('LongMIL', bundle), bundle)

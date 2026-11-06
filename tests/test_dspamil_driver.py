"""Configuration fidelity and actual live/terminal process ownership."""
from dataclasses import asdict
import subprocess
import sys
from types import SimpleNamespace

import pytest

from milplus.dspamil_configuration import resolve, validate
from milplus.dspamil_driver import claim, process_identity, environment, runtime
from milplus.dspamil_equations import EquationChoices
from milplus.dspamil_reference import HeadReference
from milplus.dspamil_training import TrainingChoices
from milplus.io import read_json, write_json


def supplied():
    return {'threads': 2, 'model': {
        'equations': asdict(EquationChoices('instances_reference', 'student_to_teacher_eq8', True, 'l2_norm')),
        'heads': asdict(HeadReference(1024, 1024, 1024, 'gelu', .1, 'Explicit complete-width reconstruction')),
        'training': asdict(TrainingChoices(50, 50, 'half_up_expected_grade_reference', 'Full fifty-epoch reference'))}}


@pytest.mark.parametrize('method', ['dspamil_2025_released_mask_reference', 'dspamil_2025_paper_mask_reference'])
def test_each_complete_mask_reference_keeps_full_head_width_and50epoch_budget(method):
    config, stop = resolve(method, SimpleNamespace(config={}), supplied())
    assert config['recipe']['epochs'] == 50 and stop is None
    assert config['recipe']['lr'] == 2e-4 and config['recipe']['weight_decay'] == 1e-5
    assert all(config['heads'][key] == 1024 for key in ['ptg_hidden', 'teacher_hidden', 'student_hidden'])
    assert config['recipe']['reference_choices']['evaluation_unit'] == 'slide'
    assert validate(config) == config


def test_patient_selection_and_final_evaluation_share_the_same_explicit_fusion_unit():
    value = supplied()
    value['model']['training']['evaluation_unit'] = 'patient'
    config, stop = resolve('dspamil_2025_released_mask_reference',
        SimpleNamespace(config={'evaluation_unit': 'patient'}), value)
    assert config['recipe']['reference_choices']['evaluation_unit'] == 'patient'


@pytest.mark.parametrize('updates', [[7, 7], [0], [True]])
def test_audit_snapshots_do_not_accept_ambiguous_or_nonpositive_update_counts(updates):
    value = supplied()
    value['model']['audit_checkpoint_updates'] = updates
    with pytest.raises(ValueError, match='audit checkpoints'):
        resolve('dspamil_2025_released_mask_reference', SimpleNamespace(config={}), value)


@pytest.mark.parametrize('change', ['reduce_head', 'reduce_epochs', 'unknown_override', 'missing_equation', 'patient_unit'])
def test_complete_recipe_and_patient_selection_unit_cannot_drift(change):
    value = supplied()
    if change == 'reduce_head':
        value['model']['heads']['ptg_hidden'] = 512
    elif change == 'unknown_override':
        value['epochs'] = 2
    elif change == 'missing_equation':
        value['model']['equations'].pop('detach_teacher_distillation')
    elif change == 'patient_unit':
        value['model']['training']['evaluation_unit'] = 'patient'
    else:
        config, unused = resolve('dspamil_2025_released_mask_reference', SimpleNamespace(config={}), value)
        config['recipe']['epochs'] = 2
        with pytest.raises(ValueError, match='full50epoch'):
            validate(config)
        return
    with pytest.raises(ValueError):
        resolve('dspamil_2025_released_mask_reference', SimpleNamespace(config={}), value)


def test_live_child_blocks_recovery_even_after_original_parent_is_terminal(tmp_path):
    terminal = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.3)'])
    original = process_identity(terminal.pid)
    terminal.wait()
    assert original is not None and process_identity(terminal.pid) is None
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        actual = process_identity(child.pid)
        assert actual is not None
        write_json(tmp_path / 'RUNNING.lock', original)
        write_json(tmp_path / 'native_process.json', {'identity': actual})
        with pytest.raises(ValueError, match='still running'):
            claim(tmp_path)
        assert read_json(tmp_path / 'RUNNING.lock') == original
    finally:
        child.terminate()
        child.wait()


def test_actual_terminal_process_allows_audited_recovery_without_losing_old_lock(tmp_path):
    process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.3)'])
    previous = process_identity(process.pid)
    process.wait()
    assert previous is not None
    write_json(tmp_path / 'RUNNING.lock', previous)
    write_json(tmp_path / 'native_process.json', {'identity': previous})
    path, owner = claim(tmp_path)
    assert process_identity(owner['pid']) == owner and read_json(path) == owner
    old = list(tmp_path.glob('verified_terminal_previous_lock_*.json'))
    assert len(old) == 1 and read_json(old[0]) == previous


def test_real_runtime_preflight_keeps_fresh_output_absent_and_overwrite_protection_strict(tmp_path):
    config, stop = resolve('dspamil_2025_released_mask_reference', SimpleNamespace(config={}), supplied())
    output = tmp_path / 'new_complete_run'
    executable, env = environment(config, output, probe=True)
    identity = runtime(executable, env)
    assert identity['torch'].split('+')[0] == '2.5.1'
    assert not output.exists()
    assert str(output) not in env['TORCH_EXTENSIONS_DIR']
    output.mkdir(exist_ok=False)
    with pytest.raises(FileExistsError):
        output.mkdir(exist_ok=False)

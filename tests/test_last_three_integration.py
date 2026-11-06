"""Code-entry checks only: never substitute random weights for restricted assets."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from milplus import registry
from milplus.cli import main
from milplus.demo import create_demo
from milplus.io import read_json, write_json

FAMILIES = ['dyko', 'dspamil', 'feather']


@pytest.mark.parametrize('family,count', [('dyko', 8), ('dspamil', 2), ('feather', 9)])
def test_registry_specs_and_real_driver_conditions_are_identical(family, count):
    spec = importlib.import_module('milplus.' + family + '_spec')
    driver = importlib.import_module('milplus.' + family + '_driver')
    registered = [p for p in registry.profiles() if p['family'] == family]
    assert registered == spec.PROFILES and len(registered) == count
    assert driver.PROFILES == spec.EXECUTION_CONDITIONS
    assert next(r for r in registry.methods() if r['id'] == family) == spec.METHOD
    root = Path(__file__).resolve().parents[1]
    assert read_json(root / 'method_specs' / (family + '.json')) == {
        **spec.METHOD, 'execution_profiles': spec.PROFILES}
    for profile in registered:
        actual = registry.require_runnable(profile['id'])
        assert actual['id'] == profile['id'] and actual['formal_admission'] is False
        assert actual['gpu_validation'] == actual['real_wsi_validation'] == actual['paper_reproduction'] == 'not_run'


def test_coverage_counts_code_availability_not_research_validation():
    coverage = registry.coverage()
    assert coverage['planned_records'] == coverage['implemented_families'] == 164
    assert coverage['execution_profiles'] == 415 and coverage['pending_ids'] == []
    assert coverage['paper_reproductions'] == 0
    assert coverage['unsupported_variants'] == {'dspamil': ['DSPA-U-MIL']}
    dspamil = next(r for r in registry.methods() if r['id'] == 'dspamil')
    assert dspamil['DSPA_U_implemented'] is False
    assert dspamil['unsupported_variants'] == ['DSPA-U-MIL']


@pytest.mark.parametrize('name', ['DSPA-U', 'DSPA-U-MIL', 'dspamil_u'])
def test_DSPA_U_never_silently_executes_the_distinct_2025_reference(name):
    with pytest.raises(ValueError, match='DSPA-U-MIL is not implemented'):
        registry.require_runnable(name)


@pytest.mark.parametrize('cohort', ['CAMELYON16', 'NSCLC', 'RCC', 'UBC'])
def test_DyKo_family_uses_prepared_cohort_and_explicit_CPU_FAISS(cohort):
    task = SimpleNamespace(dyko=SimpleNamespace(cohort=cohort))
    actual = registry.resolve_for_bundle('DyKo', task)
    assert actual['cohort'] == cohort and actual['faiss_backend'] == 'cpu_faiss_reference'
    explicit = 'dyko_' + cohort.lower() + '_cuda_faiss_requested_reference'
    assert registry.resolve_for_bundle(explicit, task)['faiss_backend'] == 'cuda_faiss'


@pytest.mark.parametrize('encoder', ['conch_v15', 'uni', 'uni_v2'])
@pytest.mark.parametrize('name', ['FEATHER', 'FEATHER / MIL-Lab models'])
def test_FEATHER_family_uses_prepared_encoder_without_changing_explicit_protocol(encoder, name):
    task = SimpleNamespace(feather=SimpleNamespace(encoder=encoder))
    actual = registry.resolve_for_bundle(name, task)
    assert actual['encoder'] == encoder and actual['mode'] == 'finetune_val20'
    explicit = 'feather_' + encoder + '_frozen_knn20_reference'
    assert registry.resolve_for_bundle(explicit, task)['mode'] == 'frozen_knn20'


@pytest.mark.parametrize('family,prepare_command', [
    ('dyko', 'prepare-dyko'), ('dspamil', 'prepare-dspamil'), ('feather', 'prepare-feather'),
])
def test_real_unprepared_bundle_fails_before_creating_any_training_output(tmp_path, family, prepare_command):
    from milplus.engine import train
    data = create_demo(tmp_path / 'ordinary-demo', patients=20)
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match=prepare_command):
        train(data, family, output)
    assert not output.exists()


@pytest.mark.parametrize('family', FAMILIES)
def test_prepared_inputs_are_deep_checked_and_cannot_enter_generic_baselines(family):
    profile = registry.require_runnable(family)
    inputs = SimpleNamespace(verify=Mock(), cohort='NSCLC', encoder='uni')
    task = SimpleNamespace(**{key: inputs if key == family else None for key in FAMILIES})
    registry.check_compatibility(profile, task)
    inputs.verify.assert_called_once_with(deep=True)
    with pytest.raises(ValueError, match='Bound'):
        registry.check_compatibility(registry.require_runnable('abmil'), task)


@pytest.mark.parametrize('family,attribute,value', [('dyko', 'cohort', 'RCC'), ('feather', 'encoder', 'uni_v2')])
def test_explicit_profile_cannot_silently_switch_to_different_native_inputs(family, attribute, value):
    task = SimpleNamespace(dyko=None, dspamil=None, feather=None)
    setattr(task, family, SimpleNamespace(**{attribute: value}, verify=Mock()))
    with pytest.raises(ValueError, match='profile.*differs'):
        registry.check_compatibility(registry.require_runnable(family), task)


@pytest.mark.parametrize('family', FAMILIES)
def test_common_training_entry_reaches_existing_full_driver(monkeypatch, family):
    from milplus import engine
    driver = importlib.import_module('milplus.' + family + '_driver')
    task = SimpleNamespace(dyko=None, dspamil=None, feather=None)
    monkeypatch.setattr(engine, 'Bundle', lambda *a: task)
    compatibility = Mock()
    monkeypatch.setattr(registry, 'check_compatibility', compatibility)
    native = Mock(return_value='actual-driver-route')
    monkeypatch.setattr(driver, 'train', native)
    result = engine.train('input', family, 'output', device='cpu', config={}, resume=True)
    assert result == 'actual-driver-route'
    assert native.call_args.args == ('input', registry.require_runnable(family)['id'], 'output')
    assert native.call_args.kwargs['resume'] is True
    compatibility.assert_called_once()


@pytest.mark.parametrize('family,backend', [
    ('dyko', 'dyko_full_original_TITAN'), ('feather', 'feather24k_full_pretrained_reference'),
    ('dspamil', 'dspamil2025_full_PAT_reference'),
])
def test_common_evaluation_dispatch_uses_full_driver_not_generic_model(tmp_path, monkeypatch, family, backend):
    from milplus.engine import evaluate
    driver = importlib.import_module('milplus.' + family + '_driver')
    write_json(tmp_path / 'request.json', {'backend': backend})
    native = Mock(return_value={'driver': family})
    monkeypatch.setattr(driver, 'evaluate', native)
    assert evaluate(tmp_path, 'val', 'cpu') == {'driver': family}
    native.assert_called_once_with(tmp_path, 'val', 'cpu')


@pytest.mark.parametrize('family', FAMILIES)
def test_runtime_failure_includes_actual_dependency_error_and_next_action(monkeypatch, family):
    driver = importlib.import_module('milplus.' + family + '_driver')
    monkeypatch.setattr(driver.subprocess, 'run', Mock(return_value=SimpleNamespace(
        returncode=1, stderr='actual missing native dependency', stdout='')))
    with pytest.raises(RuntimeError, match='actual missing native dependency') as error:
        driver.runtime('/explicit/python', {})
    assert f'.[{family}]' in str(error.value)


@pytest.mark.parametrize('family', FAMILIES)
def test_installed_CLI_inspects_public_code_profile_without_loading_weights(capsys, family):
    assert main(['inspect', '--method', family]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['family'] == family and value['runnable'] is True
    assert value['formal_admission'] is False


def test_DSPA_sample_config_preserves_full_reference_graph_and_explicit_choices():
    from milplus.config_io import load_training_config
    from milplus.dspamil_configuration import resolve
    config = Path(__file__).resolve().parents[1] / 'configs/dspamil_2025_full_reference.yaml'
    for method in importlib.import_module('milplus.dspamil_driver').PROFILES:
        actual, stop = resolve(method, SimpleNamespace(config={}), load_training_config(config))
        assert actual['recipe']['epochs'] == 50 and stop is None
        assert actual['heads']['ptg_hidden'] == actual['heads']['student_hidden'] == 1024

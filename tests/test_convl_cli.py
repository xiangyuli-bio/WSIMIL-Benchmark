"""Unified source method names, immutable data preparation and actual training."""
import gc

import pytest
import yaml

from milplus.cli import dispatch, parser
from milplus.data import Bundle
from milplus.engine import evaluate
from milplus.io import digest_file, read_json
from milplus.registry import inspect_method, profiles
from test_convl_training import full_native_bundle


def test_precision_routes_are_explicit_without_silent_cpu_substitution():
    assert inspect_method('ConVL')['id'] == 'convl_yaml_fp16_source_repaired'
    assert inspect_method('convl_yaml_fp32_reference')['id'] == 'convl_yaml_fp32_reference'
    assert {p['id'] for p in profiles() if p['family'] == 'convl'} == {
        'convl_yaml_fp16_source_repaired', 'convl_yaml_fp32_reference'}


def test_actual_patient_data_deep_check_and_preparation_entry(tmp_path):
    bundle = full_native_bundle(tmp_path/'data')
    result = dispatch(parser().parse_args(['check', '--data', str(bundle.root), '--method', 'ConVL', '--deep']))
    assert result['patients'] == 7 and result['slides'] > 7
    config = yaml.safe_load((bundle.root/'task.yaml').read_text()); config['encoders']['toy'].pop('convl')
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(config))
    output = bundle.root/'new_manifest.json'
    prepared = dispatch(parser().parse_args(['prepare-convl', '--data', str(bundle.root),
        '--metadata', str(bundle.root/'metadata.json'), '--output', str(output)]))
    assert prepared['manifest'] == str(output)
    assert read_json(output)['metadata']['binning']['policy'] == 'provided_author_four_bin_labels'
    with pytest.raises(FileExistsError, match='preserve'):
        dispatch(parser().parse_args(['prepare-convl', '--data', str(bundle.root),
            '--metadata', str(bundle.root/'metadata.json'), '--output', str(output)]))


@pytest.mark.parametrize('mode', ['--miladapter', '--vlsa', '--survival', '--center-embeddings'])
def test_convl_synthetic_mode_cannot_mix_inputs(tmp_path, mode):
    with pytest.raises(ValueError, match='cannot be mixed'):
        dispatch(parser().parse_args(['demo', '--convl', '--convl-weights', 'does_not_exist', mode, '--output', str(tmp_path/'bad')]))
    assert not (tmp_path/'bad').exists()


def test_source_requires_complete_assets_before_generic_survival_fallback(tmp_path):
    bundle = full_native_bundle(tmp_path/'data')
    task = yaml.safe_load((bundle.root/'task.yaml').read_text()); task['encoders']['toy'].pop('convl')
    (bundle.root/'task.yaml').write_text(yaml.safe_dump(task))
    assert Bundle(bundle.root).convl is None
    with pytest.raises(ValueError, match='original concept/bin assets'):
        dispatch(parser().parse_args(['check', '--data', str(bundle.root), '--method', 'ConVL']))


def test_dataset_path_and_method_run_complete_native_train_select_and_reload_twice(tmp_path):
    bundle = full_native_bundle(tmp_path/'data')
    config = tmp_path/'software.yaml'; config.write_text(yaml.safe_dump({'epochs': 1, 'threads': 1}))
    run = tmp_path/'run'
    dispatch(parser().parse_args(['run', '--data', str(bundle.root), '--method', 'convl_yaml_fp32_reference',
        '--config', str(config), '--output', str(run)]))
    assert read_json(run/'status.json')['state'] == 'completed'
    request = read_json(run/'request.json'); cfg = request['config']
    assert cfg['lr'] == .0002 and cfg['weight_decay'] == .00001 and cfg['patience'] == 8
    assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric'] == 'convl_val_nll'
    assert not (run/'test_predictions.csv').exists()
    protocol = read_json(run/'survival_protocol.json')
    assert protocol['completed_epochs'] == 1 and protocol['optimizer_updates'] == 2
    assert (run/'last.pt').stat().st_size > 10**9 and (run/'best.pt').stat().st_size > 10**9
    gc.collect()
    first = evaluate(run); digest = digest_file(run/'test_predictions.csv')
    gc.collect()
    assert evaluate(run) == first and digest_file(run/'test_predictions.csv') == digest

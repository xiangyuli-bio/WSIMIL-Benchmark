"""Unified entry and genuine original raw-image preparation routes."""
import json

import pytest
import yaml

from milplus.cli import dispatch, parser
from milplus.data import Bundle
from milplus.registry import inspect_method, profiles


def test_both_original_inference_routes_and_aliases_are_registered():
    assert inspect_method('MIL with Center Embeddings')['id'] == 'center_embeddings_bag_classifier_repaired'
    assert inspect_method('ChikonMIL')['id'] == 'center_embeddings_bag_classifier_repaired'
    assert {p['id'] for p in profiles() if p['family'] == 'center_embeddings'} == {
        'center_embeddings_bag_classifier_repaired', 'center_embeddings_soft_assignment_repaired'}


def test_actual_cli_demo_deep_check_and_immutable_prepare(tmp_path):
    root = tmp_path/'data'
    dispatch(parser().parse_args(['demo', '--center-embeddings', '--patients', '6', '--output', str(root)]))
    checked = dispatch(parser().parse_args(['check', '--data', str(root), '--method', 'Center Embeddings', '--deep']))
    assert checked['patients'] == checked['slides'] == 6
    config = yaml.safe_load((root/'task.yaml').read_text()); config['encoders']['raw_resnet34'].pop('center_embeddings')
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    result = dispatch(parser().parse_args(['prepare-center-embeddings', '--data', str(root),
        '--schema', str(root/'center_schema.json'), '--output', str(root/'new_manifest.json')]))
    assert result['manifest'] == str(root/'new_manifest.json')


@pytest.mark.parametrize('flag', ['--miladapter', '--c2c', '--survival'])
def test_mixed_modes_cannot_silently_use_a_different_method(tmp_path, flag):
    with pytest.raises(ValueError, match='cannot be mixed'):
        dispatch(parser().parse_args(['demo', '--center-embeddings', flag, '--output', str(tmp_path/'bad')]))
    assert not (tmp_path/'bad').exists()


def test_dataset_path_and_method_reach_actual_end_to_end_training_cli(tmp_path):
    root = tmp_path/'data'
    dispatch(parser().parse_args(['demo', '--center-embeddings', '--patients', '6', '--output', str(root)]))
    cfg = tmp_path/'software.yaml'
    cfg.write_text(yaml.safe_dump({'epochs': 1, 'threads': 1, 'model': {'train_batch_size': 1}}))
    run = tmp_path/'run'
    dispatch(parser().parse_args(['run', '--data', str(root), '--method', 'Center Embeddings',
        '--config', str(cfg), '--output', str(run)]))
    assert json.loads((run/'status.json').read_text())['state'] == 'completed'
    assert Bundle(root).center_embeddings is not None
    assert (run/'best.pt').is_file() and (run/'method_protocol.json').is_file()

"""Unified dataset-path/method routing and actual text-preparation CLI."""
import copy

import pytest
import yaml

from milplus.cli import parser, dispatch
from milplus.conmil_demo import create_conmil_demo, default_text
from milplus.conmil_text import load_text
from milplus.data import Bundle
from milplus.engine import train, evaluate
from milplus.io import read_json, write_json
from milplus.registry import inspect_method, resolve_for_bundle


def test_alias_and_explicit_class_route_are_traceable(tmp_path):
    assert inspect_method('ConMIL')['id'] == 'conmil_posneg_source'
    data = create_conmil_demo(tmp_path/'data', patients=15, classes=3)
    assert resolve_for_bundle('ConMIL', Bundle(data))['id'] == 'conmil_posneg_multiclass_repaired'


def test_demo_deep_check_and_manifest_binding_are_real_cli_entries(tmp_path):
    args = parser().parse_args(['demo', '--conmil', '--patients', '10', '--output', str(tmp_path/'data')])
    dispatch(args)
    args = parser().parse_args(['check', '--data', str(tmp_path/'data'), '--method', 'ConMIL', '--deep'])
    result = dispatch(args); assert result['slides'] == result['patients'] == 10
    path = tmp_path/'data/task.yaml'; task = yaml.safe_load(path.read_text())
    task['encoders']['synthetic_conch_dimension'].pop('conmil'); path.write_text(yaml.safe_dump(task))
    args = parser().parse_args(['prepare-conmil', '--data', str(tmp_path/'data'),
        '--metadata', str(tmp_path/'data/conmil_metadata.json'), '--output', str(tmp_path/'data/cli_manifest.json')])
    assert dispatch(args)['manifest'] == str(tmp_path/'data/cli_manifest.json')


@pytest.mark.parametrize('flag', ['--c2c', '--sparseconvmil', '--survival'])
def test_conmil_demo_cannot_hide_other_input_modes(tmp_path, flag):
    args = parser().parse_args(['demo', '--conmil', flag, '--output', str(tmp_path/'invalid')])
    with pytest.raises(ValueError, match='cannot be mixed'):
        dispatch(args)
    assert not (tmp_path/'invalid').exists()


def test_actual_text_preparation_cli_supports_original_positive_only_route(tmp_path):
    import torch
    torch.set_num_threads(2)
    spec = copy.deepcopy(read_json(default_text())['prompts']); spec['negative'] = []
    write_json(tmp_path/'prompts.json', spec)
    args = parser().parse_args(['prepare-conmil-text', '--prompts', str(tmp_path/'prompts.json'),
        '--weights', '/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth',
        '--output', str(tmp_path/'actual_text'), '--device', 'cpu'])
    text = dispatch(args)['text_assets']; assets = load_text(text)
    assert assets.positive.shape == (4, 512) and assets.negative.shape == (0, 512)
    data = create_conmil_demo(tmp_path/'data', patients=10, text_assets=text)
    run = train(data, 'conmil_pos_source', tmp_path/'run', config={'epochs': 1, 'threads': 2})
    assert evaluate(run) == evaluate(run)
    with pytest.raises(ValueError, match='negative queries'):
        train(data, 'conmil_posneg_source', tmp_path/'bad', config={'epochs': 1})

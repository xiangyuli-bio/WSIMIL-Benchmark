"""Two-argument entry, full26 registry and genuine prototype preparation."""
import pytest
import yaml

from milplus.cli import dispatch, parser
from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.miladapter_demo import create_miladapter_demo
from milplus.miladapter_source import PROFILES
from milplus.registry import inspect_method, profiles


def test_all_actual26_configurations_have_source_and_no_unsupported_cross_product():
    actual = {p['id'] for p in profiles() if p['family'] == 'miladapter'}
    assert actual == set(PROFILES) and len(actual) == 26
    assert inspect_method('MIL-Adapter')['id'] == 'miladapter_taskres_abmil'
    assert not any('taskres_wikg' in p or 'lr_' in p or 'titan' in p for p in actual)


def test_demo_deep_check_and_prepare_are_actual_cli_routes(tmp_path):
    root = tmp_path/'data'
    dispatch(parser().parse_args(['demo', '--miladapter', '--patients', '20', '--output', str(root)]))
    checked = dispatch(parser().parse_args(['check', '--data', str(root), '--method', 'MIL-Adapter', '--deep']))
    assert checked['patients'] == checked['slides'] == 20
    cfg = yaml.safe_load((root/'task.yaml').read_text())
    del cfg['encoders']['synthetic_native512']['miladapter']
    (root/'task.yaml').write_text(yaml.safe_dump(cfg))
    prepared = dispatch(parser().parse_args(['prepare-miladapter', '--data', str(root),
        '--metadata', str(root/'miladapter_metadata.json'), '--output', str(root/'cli_manifest.json')]))
    assert prepared['manifest'] == str(root/'cli_manifest.json')


@pytest.mark.parametrize('flag', ['--conmil', '--c2c', '--survival'])
def test_miladapter_demo_refuses_mixed_input_modes(tmp_path, flag):
    args = parser().parse_args(['demo', '--miladapter', flag, '--output', str(tmp_path/'bad')])
    with pytest.raises(ValueError, match='cannot be mixed'):
        dispatch(args)
    assert not (tmp_path/'bad').exists()


def test_rcc_keep_multiclass_and_source_import_command_are_not_fake_text(tmp_path):
    args = parser().parse_args(['import-miladapter-prototypes', '--project', 'RCC', '--encoder', 'KEEP',
        '--output', str(tmp_path/'text')])
    text = dispatch(args)['text_assets']
    assert text == str(tmp_path/'text/text.json')
    root = create_miladapter_demo(tmp_path/'data', patients=30, project='RCC', encoder='KEEP')
    assert Bundle(root).classes == ['CHRCC', 'CCRCC', 'PRCC']
    run = train(root, 'MIL-Adapter', tmp_path/'run', seed=81,
        config={'epochs': 1, 'threads': 1, 'model': {'k_shots': 2}})
    assert evaluate(run) == evaluate(run)

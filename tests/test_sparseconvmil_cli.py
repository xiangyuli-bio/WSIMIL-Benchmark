"""User entry points and the explicit source/data-mode contract."""
import pytest

from milplus.cli import dispatch, parser
from milplus.data import Bundle
from milplus.registry import inspect_method
from milplus.sparseconvmil_demo import create_sparseconvmil_demo


def test_alias_is_the_explicit_complete_source_recipe():
    spec = inspect_method('SparseConvMIL')
    assert spec['id'] == 'sparseconvmil_raw_source_c32'
    assert spec['external_source'] == 'sparseconvmil'
    assert 'compiled_sparseconvnet' in spec['required_inputs']
    assert inspect_method('sparseconvmil_raw_source_c128')['family'] == 'sparseconvmil'


def test_demo_and_deep_check_use_real_rgb_and_spatial_interface(tmp_path):
    args = parser().parse_args(['demo', '--sparseconvmil', '--patients', '10', '--output', str(tmp_path/'data')])
    dispatch(args)
    assert Bundle(tmp_path/'data').sparseconvmil is not None
    report = dispatch(parser().parse_args(['check', '--data', str(tmp_path/'data'), '--method', 'SparseConvMIL', '--deep']))
    assert report['slides'] == 10 and report['patients'] == 10 and report['deep'] is True


@pytest.mark.parametrize('flag', ['--c2c', '--raw-images', '--survival'])
def test_spatial_rgb_cannot_be_mixed_with_other_demo_modes(tmp_path, flag):
    args = parser().parse_args(['demo', '--sparseconvmil', flag, '--output', str(tmp_path/'invalid')])
    with pytest.raises(ValueError, match='cannot be mixed'):
        dispatch(args)
    assert not (tmp_path/'invalid').exists()


def test_manifest_preparation_is_a_real_cli_entry(tmp_path):
    data = create_sparseconvmil_demo(tmp_path/'data', patients=10)
    import yaml
    config = yaml.safe_load((data/'task.yaml').read_text())
    config['encoders']['raw_resnet18'].pop('sparseconvmil')
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    args = parser().parse_args(['prepare-sparseconvmil', '--data', str(data),
        '--schema', str(data/'sparseconvmil_schema.json'), '--output', str(data/'cli_manifest.json')])
    assert dispatch(args)['manifest'] == str(data/'cli_manifest.json')

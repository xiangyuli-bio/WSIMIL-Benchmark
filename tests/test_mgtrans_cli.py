import json
import subprocess
import sys

from milplus.data import Bundle
from milplus.mgtrans_demo import create_mgtrans_demo
from milplus.registry import inspect_method


def cli(*arguments):
    return subprocess.run([sys.executable, '-m', 'milplus', *map(str, arguments)],
                          capture_output=True, text=True, timeout=180)


def test_public_method_alias_resolves_complete_source_profile():
    result = cli('inspect', '--method', 'MG-Trans')
    assert result.returncode == 0, result.stdout+result.stderr
    record = json.loads(result.stdout)
    assert record['id'] == 'mgtrans_source_repaired' and record['runnable'] is True
    assert 'mgtrans_dual_scale_alignment' in record['required_inputs']
    assert inspect_method('MG-Trans')['external_source'] == 'mgtrans'


def test_cli_creates_explicit_synthetic_dual_scale_bundle(tmp_path):
    result = cli('demo', '--mgtrans', '--patients', '36', '--classes', '3', '--output', tmp_path/'data')
    assert result.returncode == 0, result.stdout+result.stderr
    bundle = Bundle(tmp_path/'data')
    assert bundle.config['synthetic'] is True and bundle.mgtrans is not None
    assert len(bundle.records('train')) == 36


def test_cli_prepares_a_new_immutable_physical_input_manifest(tmp_path):
    data = create_mgtrans_demo(tmp_path/'data', patients=18, classes=2)
    original = json.loads((data/'mgtrans/manifest.json').read_text())
    schema = data/'schema.json'; schema.write_text(json.dumps(original['schema']))
    result = cli('prepare-mgtrans', '--data', data, '--index', data/'mgtrans_index.csv',
                 '--schema', schema, '--output', data/'new_manifest.json')
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((data/'new_manifest.json').read_text()) == original


def test_cli_does_not_silently_combine_incompatible_input_modes(tmp_path):
    result = cli('demo', '--mgtrans', '--zoommil', '--output', tmp_path/'invalid')
    assert result.returncode != 0 and not (tmp_path/'invalid').exists()

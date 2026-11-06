import json
import subprocess
import sys

from milplus.data import Bundle
from milplus.sammil_demo import create_sammil_demo


def cli(*arguments):
    return subprocess.run([sys.executable, '-m', 'milplus', *map(str, arguments)],
                          capture_output=True, text=True, timeout=180)


def test_public_alias_identifies_explicit_objective_repairs():
    result = cli('inspect', '--method', 'SAM-MIL')
    assert result.returncode == 0, result.stdout+result.stderr
    spec = json.loads(result.stdout)
    assert spec['id'] == 'sammil_attn_objective_repaired'
    assert spec['external_source'] == 'sammil' and spec['runnable'] is True


def test_public_demo_keeps_synthetic_segmentation_provenance(tmp_path):
    result = cli('demo', '--sammil', '--patients', 27, '--classes', 3, '--output', tmp_path/'data')
    assert result.returncode == 0, result.stdout+result.stderr
    bundle = Bundle(tmp_path/'data')
    assert bundle.dim == 1024 and bundle.config['synthetic'] is True
    assert bundle.sammil.schema['sam_model'] == 'vit_h'
    assert 'NOT a SAM segmentation' in bundle.sammil.schema['segmentation_provenance']


def test_public_prepare_round_trips_complete_measured_input_identity(tmp_path):
    root = create_sammil_demo(tmp_path/'data', patients=18, classes=2)
    result = cli('prepare-sammil', '--data', root, '--index', root/'sammil_index.csv',
        '--schema', root/'sammil_schema.json', '--output', root/'new_manifest.json')
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((root/'new_manifest.json').read_text()) == json.loads((root/'sammil/manifest.json').read_text())


def test_incompatible_demo_modes_refused(tmp_path):
    result = cli('demo', '--sammil', '--mgtrans', '--output', tmp_path/'invalid')
    assert result.returncode != 0 and not (tmp_path/'invalid').exists()
